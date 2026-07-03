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
"""FP4 format math layer — pure tensor-to-tensor conversions, no file I/O.

Provides the shared "calculator" for both NVFP4 and MXFP4 conversion pipelines:

  1. Raw FP4 codec: nibble unpack/pack, E2M1 decode.
  2. Scale helpers: E4M3/E8M0 decode, rounding.
  3. FP4 → BF16 decoders: ``mxfp4_to_bf16``, ``nvfp4_to_bf16``.
  4. BF16 → HiFloat4 encoder / decoder (two modes):
     - ``e6m2`` (default, official): E6M2 + 8×E1 + 16×E1 + S1P2, block=64.
       Bit-exact with the HiFloat4 spec. 36 bytes per 64-element block.
     - ``e6m0`` (optional, simplified variant): local implementation;
       **not equivalent to official HiFloat4 (E6M2)**.

All functions are stateless pure-PyTorch, run on CPU, and have no NPU dependency.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

# ══════════════════════════════════════════════════════════════════════
# HiFloat4 E6M2 format constants (official spec)
# ══════════════════════════════════════════════════════════════════════

_HIF4_BLOCK = 64
_HIF4_MAN_BITS = 3             # S1P2: 1 sign + 2-bit mantissa
_HIF4_S1P2_STEP = 0.25         # = 1 / 2^(man_bits-1)
_HIF4_S1P2_MAX = 1.75          # = 2 - step
_HIF4_MAG_MAX = 7              # = 2^man_bits - 1
_HIF4_BYTES_PER_BLOCK = 36     # 1(E6M2) + 1(lv2) + 2(lv3) + 32(data)
_HIF4_E6M2_EXP_BIAS = 48
_HIF4_SF_MIN = 2.0 ** -48
_HIF4_SF_MAX = 49152.0         # 2^15 * 1.5


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
# 3. FP4 → BF16 decoders
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
# 4. HiFloat4 E6M2 codec (official — built-in, no external dependency)
# ══════════════════════════════════════════════════════════════════════


def _round_e6m2(sf: Tensor) -> Tensor:
    """Round a positive scale into E6M2 (6-bit exp, 2-bit mantissa).

    Bit-exact with :func:`quant_hifx` / the AscendC kernel.
    """
    sf = sf.to(torch.bfloat16).to(torch.float32).clip(min=_HIF4_SF_MIN, max=_HIF4_SF_MAX)
    e = torch.floor(torch.log2(sf))
    sf = torch.round(sf / 2 ** e * 2 ** 7) / 2 ** 7 * 2 ** e  # → bf16 (7-bit mantissa)
    e = torch.floor(torch.log2(sf))
    sf = torch.round(sf * torch.exp2(2 - e)) * torch.exp2(e - 2)  # → E6M2 (2-bit mantissa)
    return sf


@torch.no_grad()
def _hif4_block_encode(absx: Tensor, sign: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Core HiF4 E6M2 encode on the last dim (size 64).

    Args:
        absx: ``[..., nblk, 64]`` float32 absolute values.
        sign: ``[..., nblk, 64]`` float32 sign (±1.0).

    Returns:
        ``(codes, e1_8, e1_16, sf)`` — integer codes for packing.
    """
    lead = absx.shape[:-1]
    v = absx.reshape(*lead, 8, 2, 4)
    max_lv3 = v.amax(dim=-1)                     # [...,8,2]
    max_lv2 = max_lv3.amax(dim=-1)               # [...,8]
    max_lv1 = max_lv2.amax(dim=-1)               # [...]

    sf = _round_e6m2(max_lv1 / 7.0)
    rec = (1.0 / sf).to(torch.bfloat16).to(torch.float32)

    e1_8 = torch.floor((max_lv2 * rec[..., None]).clip(0, 4) / 4)     # [...,8]  ∈ {0,1}
    lv2_scale = torch.exp2(e1_8)
    e1_16 = torch.floor(
        (max_lv3 * rec[..., None, None] / lv2_scale[..., None]).clip(0, 2) / 2
    )                                                                   # [...,8,2] ∈ {0,1}

    # S1P2 magnitude index
    lv2_e = lv2_scale[..., None, None].expand(*lead, 8, 2, 4)
    lv3_e = torch.exp2(e1_16)[..., None].expand(*lead, 8, 2, 4)
    rec_e = rec[..., None, None, None]

    mant = absx.reshape(*lead, 8, 2, 4) / lv2_e / lv3_e * rec_e
    mag_idx = torch.floor(mant * 2 + 0.5).clip(0, _HIF4_MAG_MAX).to(torch.int64)

    return (
        mag_idx.reshape(*lead, 64),
        e1_8.to(torch.int64),
        e1_16.to(torch.int64),
        sf,
    )


@torch.no_grad()
def encode_to_hifloat4(x: Tensor, axis: int = -1) -> Tuple[Tensor, int]:
    """Encode a FP16/BF16/FP32 tensor to the packed HiFloat4 E6M2 byte stream.

    Output format (36 bytes per 64-element block):
      byte 0      : E6M2 scale_factor
      byte 1      : lv2 bits (8 × 1-bit)
      bytes 2..3  : lv3 bits (16 × 1-bit, little-endian)
      bytes 4..35 : S1P2 codes (64 × 4-bit, two per byte, low nibble first)

    Returns:
        ``(packed_uint8, orig_len)`` — packed tensor and the unpadded length
        along *axis*.
    """
    # — pad to block multiple —
    x = x.movedim(axis, -1)
    orig_len = x.shape[-1]
    pad = (_HIF4_BLOCK - orig_len % _HIF4_BLOCK) % _HIF4_BLOCK
    if pad:
        x = F.pad(x, (0, pad))

    lead = x.shape[:-1]
    nblk = x.shape[-1] // _HIF4_BLOCK
    xb = x.to(torch.float32).reshape(*lead, nblk, _HIF4_BLOCK)
    codes, e1_8, e1_16, sf = _hif4_block_encode(xb.abs(), torch.sign(xb))
    sign_bit = (torch.sign(xb) < 0).to(torch.int64)

    # — pack S1P2: (sign<<3)|mag, two nibbles per byte —
    s1p2 = (sign_bit << 3) | codes                          # [...,nblk,64] 0..15
    s1p2 = s1p2.reshape(*lead, nblk, 32, 2)
    data_bytes = (s1p2[..., 0] | (s1p2[..., 1] << 4)).to(torch.uint8)   # [...,nblk,32]

    # — pack E6M2 scale_factor byte —
    e = torch.floor(torch.log2(sf))
    exp_field = (e + _HIF4_E6M2_EXP_BIAS).clip(0, 63).to(torch.int64)
    mant2 = torch.round((sf / torch.exp2(e) - 1.0) * 4).clip(0, 3).to(torch.int64)
    sf_byte = ((exp_field << 2) | mant2).to(torch.uint8)[..., None]     # [...,nblk,1]

    # — pack lv2 (8 bits → 1 byte) and lv3 (16 bits → 2 bytes LE) —
    bit_k8 = torch.arange(8, device=x.device)
    lv2_byte = (e1_8 << bit_k8).sum(dim=-1).to(torch.uint8)[..., None]  # [...,nblk,1]
    e1_16_flat = e1_16.reshape(*lead, nblk, 16)
    bit_k16 = torch.arange(16, device=x.device)
    lv3_val = (e1_16_flat << bit_k16).sum(dim=-1).to(torch.int64)       # [...,nblk]
    lv3_bytes = torch.stack(
        [(lv3_val & 0xFF), ((lv3_val >> 8) & 0xFF)], dim=-1
    ).to(torch.uint8)                                                     # [...,nblk,2]

    packed = torch.cat([sf_byte, lv2_byte, lv3_bytes, data_bytes], dim=-1)
    packed = packed.reshape(*lead, nblk * _HIF4_BYTES_PER_BLOCK)
    return packed.movedim(-1, axis).contiguous(), orig_len


@torch.no_grad()
def decode_from_hifloat4(
    packed: Tensor,
    orig_len: Optional[int] = None,
    axis: int = -1,
    dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Decode a packed HiFloat4 E6M2 byte stream back to FP."""
    p = packed.movedim(axis, -1)
    lead = p.shape[:-1]
    nblk = p.shape[-1] // _HIF4_BYTES_PER_BLOCK
    p = p.reshape(*lead, nblk, _HIF4_BYTES_PER_BLOCK).to(torch.int64)

    sf_byte = p[..., 0]
    exp_field = sf_byte >> 2
    mant2 = sf_byte & 3
    sf = torch.exp2((exp_field - _HIF4_E6M2_EXP_BIAS).to(torch.float32)) * (
        1.0 + mant2.to(torch.float32) / 4.0
    )

    lv2_byte = p[..., 1]
    e1_8 = ((lv2_byte[..., None] >> torch.arange(8, device=packed.device)) & 1)
    lv3_val = p[..., 2] | (p[..., 3] << 8)
    e1_16 = (
        (lv3_val[..., None] >> torch.arange(16, device=packed.device)) & 1
    ).reshape(*lead, nblk, 8, 2)

    data = p[..., 4:]
    lo = data & 0x0F
    hi = (data >> 4) & 0x0F
    s1p2 = torch.stack([lo, hi], dim=-1).reshape(*lead, nblk, 64)
    sign = torch.where((s1p2 >> 3) & 1 == 1, -1.0, 1.0)
    mag = (s1p2 & 0x07).to(torch.float32) * _HIF4_S1P2_STEP

    mag = mag.reshape(*lead, nblk, 8, 2, 4)
    sign = sign.reshape(*lead, nblk, 8, 2, 4)
    lv2_e = torch.exp2(e1_8.to(torch.float32))[..., None, None].expand(*lead, nblk, 8, 2, 4)
    lv3_e = torch.exp2(e1_16.to(torch.float32))[..., None].expand(*lead, nblk, 8, 2, 4)
    sf_e = sf[..., None, None, None]

    val = (sign * mag * lv2_e * lv3_e * sf_e).reshape(*lead, nblk * _HIF4_BLOCK)
    if orig_len is not None:
        val = val[..., :orig_len]
    return val.movedim(-1, axis).to(dtype)


# ══════════════════════════════════════════════════════════════════════
# 5. HiFloat4 E6M0 codec (simplified variant — **NOT official HiF4**)
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


# ══════════════════════════════════════════════════════════════════════
# 6. Unified BF16 → HiF4 entry point (shared by NVFP4 and MXFP4 chains)
# ══════════════════════════════════════════════════════════════════════


def bf16_to_hif4(
    x_bf16: Tensor,
    hif4_format: str = "e6m2",
    *,
    e8m0_scale: Optional[Tensor] = None,
    **kwargs,
) -> Tuple[Tensor, Optional[int]]:
    """Encode BF16 tensor to packed HiFloat4 bytes (shared by both pipelines).

    Args:
        x_bf16: Input BF16 tensor.
        hif4_format: ``"e6m2"`` (official, default) or ``"e6m0"``
            (simplified variant — **not official HiF4**).
        e8m0_scale: E8M0 scale tensor (required for e6m0; unused for e6m2).
        **kwargs: Passed through to e6m0 encoder.

    Returns:
        ``(packed, orig_len)`` for e6m2;
        ``(result_dict, None)`` for e6m0 (caller extracts ``s1p2_packed``).
    """
    if hif4_format == "e6m2":
        return encode_to_hifloat4(x_bf16, axis=-1)

    if hif4_format == "e6m0":
        if e8m0_scale is None:
            raise ValueError("e8m0_scale is required for hif4_format='e6m0'")
        result = bf16_to_hif4_e6m0(x_bf16, e8m0_scale, **kwargs)
        return result, None  # type: ignore[return-value]

    raise ValueError(
        f"Unknown hif4_format: {hif4_format!r}. "
        f"Use 'e6m2' (official) or 'e6m0' (simplified)."
    )


def hif4_to_bf16(
    packed: Tensor,
    hif4_format: str = "e6m2",
    *,
    orig_len: Optional[int] = None,
    e6_exp: Optional[Tensor] = None,
    e2: Optional[Tensor] = None,
    e3: Optional[Tensor] = None,
    s1p2: Optional[Tensor] = None,
) -> Tensor:
    """Decode packed HiFloat4 back to BF16 (for validation / metrics)."""
    if hif4_format == "e6m2":
        return decode_from_hifloat4(
            packed, orig_len=orig_len, axis=-1, dtype=torch.bfloat16
        )

    if hif4_format == "e6m0":
        if any(t is None for t in (e6_exp, e2, e3, s1p2)):
            raise ValueError(
                "e6_exp, e2, e3, s1p2 are all required for hif4_format='e6m0'"
            )
        recon = hif4_e6m0_to_fp32(e6_exp, e2, e3, s1p2)  # type: ignore[arg-type]
        return recon.to(torch.bfloat16).contiguous()

    raise ValueError(f"Unknown hif4_format: {hif4_format!r}")
