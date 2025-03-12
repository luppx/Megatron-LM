# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# 移植huggingface transformers库中qwen2的yarn代码
from __future__ import annotations

import logging
import math
from functools import lru_cache

import torch
from torch import Tensor

from megatron.core import parallel_state
from megatron.core.models.common.embeddings.rope_utils import get_pos_emb_on_this_cp_rank
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

logger = logging.getLogger(__name__)


class YarnRotaryEmbeddingHF(RotaryEmbedding):
    """Yarn Rotary Embedding for language model. Based on the yarn code of qwen2 in huggingface transformers.

    Args:
        kv_channels (int): Projection weights dimension in multi-head attention. Obtained from
            transformer config. This is set to hidden_size // num_attention_heads if not provided.
        rotary_percent (float): Percent of rotary dimension to use for rotary position embeddings.
        rotary_interleaved (bool, optional): If True, interleaved rotary position embeddings.
            Defaults to False.
        seq_len_interpolation_factor (float, optional): scale of linearly interpolating RoPE for
            longer sequences. The value must be a float larger than 1.0. Defaults to None
        rotary_base (float, optional): Base period for rotary position embeddings. Defaults to
            10000.
        use_cpu_initialization (bool, optional): If False, initialize the inv_freq directly on
            the GPU. Defaults to False
        scaling_factor (float, optional): Scaling factor for Yarn RoPE. Defaults to 1.0.
        original_max_position_embeddings (int, optional): Original maximum position embeddings
            length. Defaults to 4096.
        beta_fast (float, optional): Fast beta value for Yarn RoPE. Defaults to 32.
        beta_slow (float, optional): Slow beta value for Yarn RoPE. Defaults to 1.
        mscale (float, optional): Mscale value for Yarn RoPE. Defaults to 1.
        mscale_all_dim (float, optional): Mscale all dim value for Yarn RoPE. Defaults to 0.
    """

    def __init__(
        self,
        kv_channels: int,
        rope_theta: int = 10000,
        partial_rotary_factor: float = 1.0,
        max_position_embeddings: int = 4096,
        factor: float = 1.0,
        attention_factor: float = None,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,

        rotary_percent: float = 1.0,
        rotary_interleaved: bool = False,
        seq_len_interpolation_factor: float = None,
        use_cpu_initialization: bool = False,
    ):
        self.base = rope_theta
        self.dim = int(kv_channels * partial_rotary_factor)
        self.max_position_embeddings = max_position_embeddings
        self.factor = factor
        # Sets the attention factor as suggested in the paper
        if attention_factor is None:
            self.attention_factor = 0.1 * math.log(factor) + 1.0

        # Optional config options
        # beta_fast/beta_slow: as suggested in the paper, default to 32/1 (correspondingly)
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        device = 'cpu' if use_cpu_initialization else torch.cuda.current_device()
        # Note on variable naming: "interpolation" comes from the original technique, where we interpolate the position IDs
        # to expand the possible context length. In other words, interpolation = apply scaling factor.
        pos_freqs = self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim)
        self.inv_freq_extrapolation = 1.0 / pos_freqs
        self.inv_freq_interpolation = 1.0 / (self.factor * pos_freqs)

        self.low, self.high = _find_correction_range(self.beta_fast, self.beta_slow, self.dim, self.base, self.max_position_embeddings)

        # Get n-dimensional rotational scaling corrected for extrapolation
        self.inv_freq_extrapolation_factor = 1 - _linear_ramp_factor(self.low, self.high, self.dim // 2).float().to(device)
        self.inv_freq = (
            self.inv_freq_interpolation * (1 - self.inv_freq_extrapolation_factor)
            + self.inv_freq_extrapolation * self.inv_freq_extrapolation_factor
        )
        self.original_inv_freq = self.inv_freq
        self.attention_scaling = self.attention_factor

        super().__init__(
            kv_channels=kv_channels,
            rotary_percent=rotary_percent,
            rotary_interleaved=rotary_interleaved,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            rotary_base=rope_theta,
            use_cpu_initialization=use_cpu_initialization,
        )

    @lru_cache(maxsize=32)
    def forward(self, x, position_ids) -> Tensor:
        """Forward pass of Yarn Rotary Embedding.

        Args:
            max_seq_len (int): Maximum size of sequence
            offset (int, optional): RoPE offset. Defaults to 0.

        Returns:
            Tensor: Embeddings after applying Yarn RoPE.
        """
        if self.inv_freq_extrapolation.device.type == 'cpu':
            # move `inv_freq_extrapolation` to GPU once at the first micro-batch forward pass
            self.inv_freq_extrapolation = self.inv_freq_extrapolation.to(device=torch.cuda.current_device())

        if self.inv_freq_interpolation.device.type == 'cpu':
            # move `inv_freq_interpolation` to GPU once at the first micro-batch forward pass
            self.inv_freq_interpolation = self.inv_freq_interpolation.to(device=torch.cuda.current_device())

        self.inv_freq = self.inv_freq.to(device=torch.cuda.current_device())

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        # with torch.autocast(device_type=device_type, enabled=False):
        #     freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        #     emb = torch.cat((freqs, freqs), dim=-1)
        #     cos = emb.cos()
        #     sin = emb.sin()

        # # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        # cos = cos * self.attention_scaling
        # sin = sin * self.attention_scaling

        # return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb


# Inverse dim formula to find dim based on number of rotations
def _find_correction_dim(
    num_rotations: float, dim: int, rotary_base: float = 10000, max_position_embeddings: int = 2048
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(rotary_base)
    )


# Find dim range bounds based on rotations
def _find_correction_range(
    low_rot: float,
    high_rot: float,
    dim: int,
    rotary_base: float = 10000,
    max_position_embeddings: int = 2048,
) -> tuple[int, int]:
    low = math.floor(_find_correction_dim(low_rot, dim, rotary_base, max_position_embeddings))
    high = math.ceil(_find_correction_dim(high_rot, dim, rotary_base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def _linear_ramp_factor(min: float, max: float, dim: int) -> Tensor:
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def _get_mscale(scale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0 #多了mscale,mscale默认值为1,取默认值时和yarn作者代码一样
