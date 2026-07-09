# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""FP4 format math layer — NVFP4 → BF16 dequant.

Provides the shared "calculator" for the NVFP4 conversion pipeline.
The BF16 → HiF4 fake-quant step (e6m2) is handled by the existing
``amct_pytorch.common.utils.hifx4_quant.quant_dequant_float``, not here.

This module covers:
  1. Raw FP4 codec: nibble unpack/pack, E2M1 decode.
  2. Scale helpers: E4M3 decode.
  3. FP4 → BF16 dequant: ``nvfp4_to_bf16``.
"""

from __future__ import annotations

import torch
from torch import Tensor

# ══════════════════════════════════════════════════════════════════════
# 1. Raw FP4 codec — byte ↔ nibble, E2M1 ↔ value
# ══════════════════════════════════════════════════════════════════════

_E2M1_LUT_VALUES = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def e2m1_lut(
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return the 16-entry E2M1 lookup table (nibble → float value)."""
    return torch.tensor(_E2M1_LUT_VALUES, device=device, dtype=dtype)


def _as_uint8(x: Tensor, name: str) -> Tensor:
    """Cast *x* to uint8 bytes; int8 is accepted and reinterpreted byte-wise."""
    if x.dtype == torch.uint8:
        return x.contiguous()
    if x.dtype == torch.int8:
        return x.to(torch.uint8).contiguous()
    raise TypeError(
        f"{name} must be torch.uint8 or torch.int8 packed bytes, got {x.dtype}"
    )


def unpack_nibbles(packed_u8: Tensor, low_first: bool = True) -> Tensor:
    """Unpack uint8 bytes into 4-bit nibble codes along the last dimension.

    Args:
        packed_u8: ``[..., K/2]`` uint8 byte tensor.
        low_first: If True, low nibble first.

    Returns:
        ``[..., K]`` uint8 tensor with values in [0, 15].
    """
    packed_u8 = _as_uint8(packed_u8, "packed_u8")
    lo = packed_u8 & 0x0F
    hi = (packed_u8 >> 4) & 0x0F
    if low_first:
        out = torch.stack((lo, hi), dim=-1)
    else:
        out = torch.stack((hi, lo), dim=-1)
    return out.reshape(
        *packed_u8.shape[:-1], packed_u8.shape[-1] * 2
    ).contiguous()


def pack_nibbles(nibbles_u8: Tensor, low_first: bool = True) -> Tensor:
    """Pack 4-bit nibble codes along the last dimension back into uint8 bytes."""
    nibbles_u8 = _as_uint8(nibbles_u8, "nibbles_u8")
    if nibbles_u8.shape[-1] % 2 != 0:
        raise ValueError(
            f"last dim must be even to pack nibbles, got {nibbles_u8.shape[-1]}"
        )
    a = nibbles_u8[..., 0::2] & 0x0F
    b = nibbles_u8[..., 1::2] & 0x0F
    if low_first:
        return (a | (b << 4)).contiguous()
    return ((a << 4) | b).contiguous()


def decode_fp4_e2m1(codes: Tensor) -> Tensor:
    """Decode E2M1 4-bit codes (0..15) to float32 values using a LUT."""
    return e2m1_lut(codes.device, torch.float32)[codes.long()]


# ══════════════════════════════════════════════════════════════════════
# 2. Scale helpers
# ══════════════════════════════════════════════════════════════════════


def decode_sf_to_fp32(sf: Tensor) -> Tensor:
    """Decode NVFP4 block-scale (E4M3 or uint8 view of E4M3) to float32."""
    if sf.dtype == torch.float8_e4m3fn:
        return sf.to(torch.float32)
    if sf.dtype == torch.uint8:
        return sf.view(torch.float8_e4m3fn).to(torch.float32)
    return sf.to(torch.float32)


# ══════════════════════════════════════════════════════════════════════
# 3. FP4 → BF16 dequant
# ══════════════════════════════════════════════════════════════════════


def nvfp4_to_bf16(
    weight_fp4_packed: Tensor,
    weight_scale: Tensor,
    weight_scale_2: Tensor,
    out_features: int,
    in_features: int,
    sf_vec_size: int = 16,
) -> Tensor:
    """Decode NVFP4 (TRT-LLM) packed weight to BF16.

    NVFP4 format: E2M1 packed weight × block-scale (E4M3, one per *sf_vec_size*
    elements) × global per-channel scale.
    """
    device = weight_fp4_packed.device

    w_e2m1 = decode_fp4_e2m1(
        unpack_nibbles(weight_fp4_packed, low_first=True)
    ).to(torch.float32)

    def _pad_up(x: int, m: int) -> int:
        return ((x + m - 1) // m) * m

    nrows = _pad_up(out_features, 128)
    ncols = _pad_up(in_features // sf_vec_size, 4)

    sf = weight_scale
    if sf.dim() == 1:
        sf = sf.view(nrows, ncols)
    elif sf.dim() != 2:
        raise ValueError(f"Unexpected weight_scale dim={sf.dim()}, expected 1 or 2")

    sf = sf[:out_features, : in_features // sf_vec_size].contiguous()
    sf = decode_sf_to_fp32(sf).to(device=device)

    sf_full = sf.repeat_interleave(sf_vec_size, dim=1)
    global_scale = weight_scale_2.to(torch.float32).to(device)

    w_fp32 = w_e2m1[:out_features, :in_features] * sf_full * global_scale
    return w_fp32.to(torch.bfloat16).contiguous()

