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
"""Bit-exact pure-PyTorch reference for the HiFloat4 (hifx4) packed codec.

This is the GOLDEN REFERENCE and the format spec for the AscendC operator in
``../../op_kernel/hifloat4_cast_kernel.cpp``. It is pure torch (runs on CPU/any device,
no compilation) so it can be unit-tested, and it is constructed so that

    decode_from_hifloat4(encode_to_hifloat4(x)) == hifloat4_quant_dequant(x)
                                                ≈ quant_hifx(x)        (the fake-quant sim)

i.e. the encode→decode round trip reproduces the existing HiFloat4 simulation exactly.

PACKED FORMAT — per 64-element block, 36 bytes (= 4.5 bit/value), as in arXiv:2602.11287:
    byte  0      : E6M2 scale_factor   (Level-1, 8 bit: [7:2]=exp field bias 48, [1:0]=2-bit mantissa)
    byte  1      : lv2 bits            (Level-2, 8 × 1-bit, bit k = group k of 8 elements, k=i//8)
    bytes 2..3   : lv3 bits            (Level-3, 16 × 1-bit, little-endian, bit k = group k of 4 elements, k=i//4)
    bytes 4..35  : 32 × S1P2 codes     (64 × 4-bit, two per byte: low nibble = elem 2j, high nibble = elem 2j+1)
                   S1P2 code (4 bit) = (sign << 3) | mag_index,  mag_index = round(|mant|·4) ∈ [0, 7]  (value = mag_index/4 ∈ {0,0.25,…,1.75})

Decoded value of element i (paper eq. 2):
    V_i = sign · (mag_index_i / 4) · 2^lv2[i//8] · 2^lv3[i//4] · E6M2_scale_factor
"""
import torch

BLOCK = 64
MAN_BITS = 3                              # hifx4: S1P2 = 1 sign + (implicit 1).2-frac
S1P2_STEP = 1.0 / (1 << (MAN_BITS - 1))   # 0.25
S1P2_MAX = 2.0 - S1P2_STEP                # 1.75
MAG_MAX = (1 << MAN_BITS) - 1             # 7  (max magnitude index)
BYTES_PER_BLOCK = 36                      # 1 (E6M2) + 1 (lv2) + 2 (lv3) + 32 (data)
E6M2_EXP_BIAS = 48
SF_MIN = 2.0 ** -48
SF_MAX = 49152.0                          # 2^15 * 1.5


def _round_e6m2(sf):
    """Round a positive scale into E6M2, matching quant_hifx's two-step rounding exactly."""
    sf = sf.to(torch.bfloat16).to(torch.float32).clip(min=SF_MIN, max=SF_MAX)
    e = torch.floor(torch.log2(sf))                       # → bf16 (7-bit mantissa) round
    sf = torch.round(sf / 2 ** e * 2 ** 7) / 2 ** 7 * 2 ** e
    e = torch.floor(torch.log2(sf))                       # → E6M2 (2-bit mantissa) round
    sf = torch.round(sf * torch.exp2(2 - e)) * torch.exp2(e - 2)
    return sf


@torch.no_grad()
def _block_quant(absx, sign):
    """Core HiFloat4 quantization on the last dim (size 64). Returns the integer codes
    and reconstructed FP value. `absx`/`sign` shape: [..., nblk, 64] (float32)."""
    lead = absx.shape[:-1]
    v = absx.reshape(*lead, 8, 2, 4)                      # 64 = 8 groups × 2 × 4
    max_lv3 = v.amax(dim=-1)                              # [...,8,2]  (per 4-elem group, 16 groups)
    max_lv2 = max_lv3.amax(dim=-1)                        # [...,8]    (per 8-elem group, 8 groups)
    max_lv1 = max_lv2.amax(dim=-1)                        # [...]      (per 64 block)

    sf = _round_e6m2(max_lv1 / 7.0)                       # E6M2 block scale; /7 = 1.75·2·2 (max element value)
    rec = (1.0 / sf).to(torch.bfloat16).to(torch.float32)

    # Level-2: 1 bit per 8-elem group; Level-3: 1 bit per 4-elem group.
    e1_8 = torch.floor((max_lv2 * rec[..., None]).clip(0, 4) / 4)                       # [...,8] ∈ {0,1}
    lv2_scale = torch.exp2(e1_8)
    e1_16 = torch.floor((max_lv3 * rec[..., None, None] / lv2_scale[..., None]).clip(0, 2) / 2)  # [...,8,2] ∈ {0,1}
    lv3_scale = torch.exp2(e1_16)

    # Per-element scales (broadcast groups back to the 4 / 8 elements they cover).
    lv2_e = lv2_scale[..., None, None].expand(*lead, 8, 2, 4)
    lv3_e = lv3_scale[..., None].expand(*lead, 8, 2, 4)
    sf_e = sf[..., None, None, None]
    rec_e = rec[..., None, None, None]

    mant = absx.reshape(*lead, 8, 2, 4) / lv2_e / lv3_e * rec_e
    mag_idx = torch.floor(mant * (1 << (MAN_BITS - 1)) + 0.5).clip(0, MAG_MAX).to(torch.int64)  # 0..7
    mag = mag_idx.to(torch.float32) * S1P2_STEP

    recon = (sign.reshape(*lead, 8, 2, 4) * mag * lv2_e * lv3_e * sf_e).reshape(*lead, 64)
    codes = mag_idx.reshape(*lead, 64)                    # magnitude index 0..7 per element
    return codes, e1_8.to(torch.int64), e1_16.to(torch.int64), sf, recon


def _pad_to_block(x, axis):
    x = x.movedim(axis, -1)
    c = x.shape[-1]
    pad = (BLOCK - c % BLOCK) % BLOCK
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    return x, c, pad


@torch.no_grad()
def hifloat4_quant_dequant(x, axis=-1):
    """FP → HiFloat4 → FP fake-quant (no byte packing). Same shape/dtype as `x`.

    This equals `decode_from_hifloat4(encode_to_hifloat4(x))` by construction and matches
    the existing `quant_hifx` simulation. Use this as the NPU-free reference for the cast op.
    """
    orig_dtype = x.dtype
    xf, c, pad = _pad_to_block(x.to(torch.float32), axis)
    lead = xf.shape[:-1]
    xb = xf.reshape(*lead, -1, BLOCK)
    _, _, _, _, recon = _block_quant(xb.abs(), torch.sign(xb))
    out = recon.reshape(*lead, -1)
    if pad:
        out = out[..., :c]
    return out.movedim(-1, axis).to(orig_dtype)


@torch.no_grad()
def encode_to_hifloat4(x, axis=-1):
    """Encode a FP16/BF16/FP32 tensor to the packed HiFloat4 byte stream.

    Returns a uint8 tensor where the quantized axis (moved to last) has length
    ``nblk * 36``. Use the returned ``orig_len`` with ``decode_from_hifloat4`` to strip
    block padding. Shape: [..., nblk*36], uint8.
    """
    xf, c, pad = _pad_to_block(x.to(torch.float32), axis)
    lead = xf.shape[:-1]
    nblk = xf.shape[-1] // BLOCK
    xb = xf.reshape(*lead, nblk, BLOCK)
    codes, e1_8, e1_16, sf, _ = _block_quant(xb.abs(), torch.sign(xb))
    sign_bit = (torch.sign(xb) < 0).to(torch.int64)       # [...,nblk,64]

    # --- pack S1P2 codes: (sign<<3)|mag, two nibbles per byte ---
    s1p2 = (sign_bit << 3) | codes                        # [...,nblk,64] 0..15
    s1p2 = s1p2.reshape(*lead, nblk, 32, 2)
    data_bytes = (s1p2[..., 0] | (s1p2[..., 1] << 4)).to(torch.uint8)   # [...,nblk,32]

    # --- pack E6M2 scale_factor byte ---
    e = torch.floor(torch.log2(sf))
    exp_field = (e + E6M2_EXP_BIAS).clip(0, 63).to(torch.int64)
    mant2 = torch.round((sf / torch.exp2(e) - 1.0) * 4).clip(0, 3).to(torch.int64)
    sf_byte = ((exp_field << 2) | mant2).to(torch.uint8)[..., None]     # [...,nblk,1]

    # --- pack lv2 (8 bits → 1 byte) and lv3 (16 bits → 2 bytes, little-endian) ---
    bit_k8 = torch.arange(8, device=x.device)
    lv2_byte = (e1_8 << bit_k8).sum(dim=-1).to(torch.uint8)[..., None]  # [...,nblk,1]
    e1_16_flat = e1_16.reshape(*lead, nblk, 16)
    bit_k16 = torch.arange(16, device=x.device)
    lv3_val = (e1_16_flat << bit_k16).sum(dim=-1).to(torch.int64)       # [...,nblk]
    lv3_bytes = torch.stack([(lv3_val & 0xFF), ((lv3_val >> 8) & 0xFF)], dim=-1).to(torch.uint8)  # [...,nblk,2]

    packed = torch.cat([sf_byte, lv2_byte, lv3_bytes, data_bytes], dim=-1)  # [...,nblk,36]
    packed = packed.reshape(*lead, nblk * BYTES_PER_BLOCK)
    return packed.movedim(-1, axis).contiguous(), c


@torch.no_grad()
def decode_from_hifloat4(packed, orig_len=None, axis=-1, dtype=torch.bfloat16):
    """Decode a packed HiFloat4 byte stream back to FP. Inverse of `encode_to_hifloat4`."""
    p = packed.movedim(axis, -1)
    lead = p.shape[:-1]
    nblk = p.shape[-1] // BYTES_PER_BLOCK
    p = p.reshape(*lead, nblk, BYTES_PER_BLOCK).to(torch.int64)

    sf_byte = p[..., 0]
    exp_field = sf_byte >> 2
    mant2 = sf_byte & 3
    sf = torch.exp2((exp_field - E6M2_EXP_BIAS).to(torch.float32)) * (1.0 + mant2.to(torch.float32) / 4.0)  # [...,nblk]

    lv2_byte = p[..., 1]
    e1_8 = ((lv2_byte[..., None] >> torch.arange(8, device=packed.device)) & 1)        # [...,nblk,8]
    lv3_val = p[..., 2] | (p[..., 3] << 8)
    e1_16 = ((lv3_val[..., None] >> torch.arange(16, device=packed.device)) & 1).reshape(*lead, nblk, 8, 2)

    data = p[..., 4:]                                       # [...,nblk,32]
    lo = data & 0x0F
    hi = (data >> 4) & 0x0F
    s1p2 = torch.stack([lo, hi], dim=-1).reshape(*lead, nblk, 64)
    sign = torch.where((s1p2 >> 3) & 1 == 1, -1.0, 1.0)
    mag = (s1p2 & 0x07).to(torch.float32) * S1P2_STEP

    mag = mag.reshape(*lead, nblk, 8, 2, 4)
    sign = sign.reshape(*lead, nblk, 8, 2, 4)
    lv2_e = torch.exp2(e1_8.to(torch.float32))[..., None, None].expand(*lead, nblk, 8, 2, 4)
    lv3_e = torch.exp2(e1_16.to(torch.float32))[..., None].expand(*lead, nblk, 8, 2, 4)
    sf_e = sf[..., None, None, None]

    val = (sign * mag * lv2_e * lv3_e * sf_e).reshape(*lead, nblk * BLOCK)
    if orig_len is not None:
        val = val[..., :orig_len]
    return val.movedim(-1, axis).to(dtype)
