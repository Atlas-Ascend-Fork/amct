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
"""Shared FP4 codec — nibble ↔ byte, E2M1 code ↔ value.

The reusable "calculator" shared by the FP4 decoders (NVFP4, and downstream MXFP4): unpack
uint8 bytes into 4-bit nibbles and decode E2M1 codes to values. **Format-specific dequant**
(e.g. ``nvfp4_to_bf16``, the E4M3/E8M0 block-scale math) lives in each ``unpack_*.py``, not
here, so this module stays stable as formats are added.
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


def decode_fp4_e2m1(codes: Tensor) -> Tensor:
    """Decode E2M1 4-bit codes (0..15) to float32 values using a LUT."""
    return e2m1_lut(codes.device, torch.float32)[codes.long()]

