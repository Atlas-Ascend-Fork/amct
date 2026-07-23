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
"""FP4 format math layer — FP4 → BF16 dequant, plus optional HiF4_E6M0 encode.

Provides the shared "calculator" for NVFP4 and MXFP4 conversion pipelines.
The BF16 → HiF4 fake-quant step (e6m2) is handled by the existing
``amct_pytorch.common.utils.hifloat4_quant.quant_dequant_float``, not here.

This module covers:
  1. Raw FP4 codec: nibble unpack/pack, E2M1 decode.
  2. Scale helpers: E4M3/E8M0 decode.
  3. FP4 → BF16 dequant: ``mxfp4_to_bf16``, ``nvfp4_to_bf16``.
  4. (Optional) HiF4_E6M0 simplified variant: local pure-torch encode/decode.
"""

from __future__ import annotations

from typing import Dict

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


def exp2_int(exp: Tensor) -> Tensor:
    """Return ``2**exp`` for an integer exponent tensor."""
    return torch.ldexp(
        torch.ones_like(exp, dtype=torch.float32), exp.to(torch.int32)
    )


def e8m0_to_unbiased(e8m0_code: Tensor, e8_bias: int = 127) -> Tensor:
    """Convert E8M0 scale codes to signed unbiased integer exponents."""
    e8m0_code = _as_uint8(e8m0_code, "e8m0_code")
    if bool((e8m0_code == 255).any()):
        raise ValueError("E8M0 code 255 is reserved/NaN; found in scale tensor")
    return e8m0_code.to(torch.int16) - int(e8_bias)


def decode_sf_to_fp32(sf: Tensor) -> Tensor:
    """Decode NVFP4 block-scale (E4M3 or uint8 view of E4M3) to float32."""
    if sf.dtype == torch.float8_e4m3fn:
        return sf.to(torch.float32)
    if sf.dtype == torch.uint8:
        return sf.view(torch.float8_e4m3fn).to(torch.float32)
    return sf.to(torch.float32)


def _round_to_int(x: Tensor, mode: str) -> Tensor:
    """Round float tensor to integer by *mode* (``nearest_even`` / ``half_up``)."""
    if mode == "nearest_even":
        return torch.round(x)
    if mode == "half_up":
        return torch.floor(x + 0.5)
    raise ValueError(f"unsupported round mode: {mode}")


# ══════════════════════════════════════════════════════════════════════
# 3. FP4 → BF16 dequant
# ══════════════════════════════════════════════════════════════════════


def mxfp4_to_bf16(
    packed_fp4: Tensor,
    e8m0_scale: Tensor,
    *,
    low_first: bool = True,
    e8_bias: int = 127,
    block_size: int = 32,
) -> Tensor:
    """Decode MXFP4 packed tensor to BF16.

    One E8M0 scale code is shared by every ``block_size`` (=32) MXFP4 elements.
    Real value = E2M1(nibble) × 2^(E8M0_code − e8_bias).
    """
    if block_size != 32:
        raise ValueError("MXFP4 block_size is expected to be 32")
    packed_fp4 = _as_uint8(packed_fp4, "packed_fp4")
    e8m0_scale = _as_uint8(e8m0_scale, "e8m0_scale")

    nibbles = unpack_nibbles(packed_fp4, low_first=low_first)
    num_values = nibbles.shape[-1]
    if num_values % block_size != 0:
        raise ValueError(
            f"decoded value dim must be multiple of {block_size}, got {num_values}"
        )
    if e8m0_scale.shape[-1] != num_values // block_size:
        raise ValueError(
            f"scale last dim mismatch: scale={e8m0_scale.shape[-1]}, "
            f"expected={num_values // block_size}"
        )
    if packed_fp4.shape[:-1] != e8m0_scale.shape[:-1]:
        raise ValueError(
            f"prefix shape mismatch: packed={packed_fp4.shape[:-1]}, "
            f"scale={e8m0_scale.shape[:-1]}"
        )

    vals = decode_fp4_e2m1(nibbles)
    vals = vals.reshape(*vals.shape[:-1], e8m0_scale.shape[-1], block_size)

    e8_exp = e8m0_to_unbiased(e8m0_scale, e8_bias=e8_bias)
    scale = exp2_int(e8_exp).unsqueeze(-1)
    x = vals * scale
    return x.reshape(*nibbles.shape[:-1], num_values).to(torch.bfloat16).contiguous()


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


# ══════════════════════════════════════════════════════════════════════
# 4. HiFloat4 E6M0 codec (simplified variant — **NOT official HiF4**)
# ══════════════════════════════════════════════════════════════════════


def compute_e6_exp_from_e8_pair(
    e8m0_scale: Tensor,
    *,
    e8_bias: int = 127,
    e6_rule: str = "unbiased",
) -> Tensor:
    """Convert pairs of E8M0 scales into one level-1 E6 exponent per 64 values.

    Args:
        e8m0_scale: ``[..., G*2]`` uint8 E8M0 codes.
        e8_bias: E8M0 exponent bias.
        e6_rule: ``"unbiased"`` — ``e6_exp = max(e8_code − bias) − 2``;
                 ``"raw"``      — ``e6_exp = max(e8_code) − 2``.

    Returns:
        ``[..., G]`` int16 exponent tensor.
    """
    e8m0_scale = _as_uint8(e8m0_scale, "e8m0_scale")
    if e8m0_scale.shape[-1] % 2 != 0:
        raise ValueError(
            f"scale last dim must be even, got {e8m0_scale.shape[-1]}"
        )
    pair = e8m0_scale.reshape(
        *e8m0_scale.shape[:-1], e8m0_scale.shape[-1] // 2, 2
    )
    if e6_rule == "unbiased":
        pair_exp = e8m0_to_unbiased(pair, e8_bias=e8_bias)
        return pair_exp.max(dim=-1).values.to(torch.int16) - 2
    if e6_rule == "raw":
        return pair.to(torch.int16).max(dim=-1).values - 2
    raise ValueError("e6_rule must be 'unbiased' or 'raw'")


def bf16_to_hif4_e6m0(
    x_bf16: Tensor,
    e8m0_scale: Tensor,
    *,
    e8_bias: int = 127,
    e6_rule: str = "unbiased",
    e6_bias: int = 0,
    round_mode: str = "nearest_even",
    pack_low_first: bool = True,
) -> Dict[str, Tensor]:
    """Encode BF16 tensor into HiFloat4_E6M0 components (**NOT official HiF4**).

    64 values per group; Level-1 E6 exponent from two E8M0 scales;
    Level-2 8×1-bit + Level-3 16×1-bit + 64×S1P2.

    Returns dict: e6_exp, e6_code, e2, e3, s1p2, s1p2_packed, qmag, y_scaled.
    """
    if x_bf16.shape[-1] % 64 != 0:
        raise ValueError(f"x last dim must be multiple of 64, got {x_bf16.shape[-1]}")
    if e8m0_scale.shape[-1] * 32 != x_bf16.shape[-1]:
        raise ValueError(
            f"scale last dim mismatch: {e8m0_scale.shape[-1] * 32} vs {x_bf16.shape[-1]}"
        )
    if x_bf16.shape[:-1] != e8m0_scale.shape[:-1]:
        raise ValueError("prefix shape mismatch")

    x = x_bf16.to(torch.float32).reshape(
        *x_bf16.shape[:-1], x_bf16.shape[-1] // 64, 64
    )
    e6_exp = compute_e6_exp_from_e8_pair(
        e8m0_scale, e8_bias=e8_bias, e6_rule=e6_rule
    ).to(x.device)
    e6_code = torch.clamp(e6_exp.to(torch.int16) + int(e6_bias), 0, 63).to(torch.uint8)

    base = exp2_int(e6_exp).unsqueeze(-1).to(x.device)
    abs_x = x.abs()

    m4 = abs_x.reshape(*abs_x.shape[:-1], 16, 4).amax(dim=-1)
    m8 = m4.reshape(*m4.shape[:-1], 8, 2).amax(dim=-1)

    e2 = (m8 > (3.5 * base)).to(torch.uint8)
    e2_for_e3 = e2.repeat_interleave(2, dim=-1)
    e3_threshold = 1.75 * base * torch.pow(2.0, e2_for_e3.float())
    e3 = (m4 > e3_threshold).to(torch.uint8)

    e2_elem = e2.repeat_interleave(8, dim=-1)
    e3_elem = e3.repeat_interleave(4, dim=-1)
    total_scale = base * torch.pow(2.0, e2_elem.float()) * torch.pow(2.0, e3_elem.float())

    y = x / total_scale
    qmag = _round_to_int(y.abs() * 4.0, round_mode).clamp(0, 7).to(torch.uint8)
    sign = (y < 0).to(torch.uint8)
    sign = torch.where(qmag == 0, torch.zeros_like(sign), sign)
    s1p2 = ((sign << 3) | qmag).to(torch.uint8)
    s1p2_packed = pack_nibbles(s1p2, low_first=pack_low_first)

    return {
        "e6_exp": e6_exp.to(torch.int16).contiguous(),
        "e6_code": e6_code.contiguous(),
        "e2": e2.contiguous(),
        "e3": e3.contiguous(),
        "s1p2": s1p2.contiguous(),
        "s1p2_packed": s1p2_packed.contiguous(),
        "qmag": qmag.contiguous(),
        "y_scaled": y.contiguous(),
    }


def hif4_e6m0_to_fp32(
    e6_exp: Tensor,
    e2: Tensor,
    e3: Tensor,
    s1p2: Tensor,
) -> Tensor:
    """Reconstruct FP32 values from HiF4_E6M0 components (validation only)."""
    if s1p2.shape[-1] != 64:
        raise ValueError(f"s1p2 last dim should be 64, got {s1p2.shape[-1]}")
    sign = torch.where(((s1p2 >> 3) & 1) != 0, -1.0, 1.0).to(torch.float32)
    mag = (s1p2 & 0x07).to(torch.float32) / 4.0
    base = exp2_int(e6_exp.to(torch.int16)).unsqueeze(-1).to(mag.device)
    e2_elem = e2.repeat_interleave(8, dim=-1).to(torch.float32)
    e3_elem = e3.repeat_interleave(4, dim=-1).to(torch.float32)
    x = sign * mag * base * torch.pow(2.0, e2_elem) * torch.pow(2.0, e3_elem)
    return x.reshape(*s1p2.shape[:-2], s1p2.shape[-2] * 64).contiguous()

