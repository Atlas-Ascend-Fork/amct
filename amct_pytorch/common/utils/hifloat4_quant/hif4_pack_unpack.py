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
"""HiF4 (N=4) real bit dump -- pack model weights to the deployable HiFloat4 layout.

Take an [M, N] tensor, cast it to fp32, run the *same* shared HiF4 quantization
(``_hif4_encode_decode.hif4_encode``, also used by ``quant_dequant_float``) but stop before dequantizing, and emit the raw
bit representation of the scale and the in-group codes. This is what turns a weight tensor
into the uint8 buffers stored in a "real" HiF4 safetensors file (vs. the FP->HiF4->FP
fake-quant simulation used for PPL/accuracy evaluation).

Grouping is along the LAST dim (N): every 64 consecutive elements of a row form one
group, laid out as (8, 2, 4) exactly like ``quant_dequant_float``'s ``unflatten(-1, (-1, 8, 2, 4))``.
So ``N`` must be a multiple of 64.

Outputs
-------
scale : uint8 [M, N // 64, 4]      one 4-byte record per 64-group
    byte 0 : E6M2                  level-1 scale factor, E6M2 = 8 bits = (E6_biased << 2) | M2
    byte 1 : E1_8  (8 x 1 bit)     "level-1" per-8-subgroup exponent bit; subgroup 0 at MSB
    byte 2 : E1_16 high 8 bits     "level-2" per-4-subgroup exponent bit, blocks 0..7  (block 0 at MSB)
    byte 3 : E1_16 low  8 bits     blocks 8..15                                        (block 8 at MSB)
value : uint8 [M, N // 2]          two adjacent 4-bit codes packed into one byte
    each 4-bit nibble = (sign << 3) | E1M2,  E1M2 = round(|in_grp| * 4) in 0..7 (S1P2)
    the first (even-index) element goes in the HIGH nibble, the next in the LOW nibble

Note E1_8 / E1_16 are named "level-1 / level-2 scale" per the requested byte layout;
they correspond to the level-2 (8x) and level-3 (16x) exponents inside the shared ``_hif4_encode_decode`` encode.
This format is HiF4-specific (Ng = 2, so the in-group code fits in 3 bits + 1 sign bit).

NPU pitfall: ``torch.exp2`` / ``2**e`` / ``torch.pow(2, e)`` are exp-approximation based on
Ascend and are NOT bit-exact even for integer exponents. Since a power-of-two scale
multiplies a whole 64-group, that offsets every element and makes the dump disagree with
the reference (~90% of elements). So EVERY power of two here goes through ``torch.ldexp``
(writes the exponent field only -> exact on NPU, identical to ``2**e`` on CPU/GPU), and the
bit packing uses plain integer arithmetic (a*2**k + b) instead of shift/or.
"""

import numpy as np
import torch

from ._hif4_encode_decode import hif4_encode, hif4_decode, _pow2, E6_OFFSET, E6M2_MBITS

N_FMT = 4            # HiF4
NG = N_FMT - 2       # S1PNg, Ng = 2  -> in-group code is 3 bits, value = code / 2**Ng


@torch.no_grad()
def hif4_cast(x, output="packed"):
    """Cast an [M, N] tensor to HiFloat4 -- the single bf16 -> HiF4 entry point.

    The bf16 -> HiF4 quantization (block scaling + S1P2 rounding) is computed once, then the
    caller picks what comes out via ``output`` (typically driven by user config):

    - ``output="packed"``   -> return the real HiF4 bytes ``(scale, value)`` as torch.uint8
      tensors ``scale [M, N//64, 4]`` and ``value [M, N//2]`` on ``x.device`` (for storing a
      real HiF4 safetensors). See the module docstring for the exact bit layout.
    - ``output="dequant"``  -> return the HiF4 -> float round trip: a tensor of shape [M, N] on
      ``x.device`` with the input dtype (fake-quant simulation, for PPL / accuracy eval).

    Both branches share the same quantization, so ``dequant`` is exactly what a stored
    ``packed`` weight decodes to -- the eval path and the deployed weight never disagree.
    """
    if output not in ("packed", "dequant"):
        raise ValueError(f"output must be 'packed' or 'dequant', got {output!r}")

    dtype_ori = x.dtype
    x = x.detach().to(torch.float32)
    assert x.dim() == 2, "input must be [M, N]"
    M, N = x.shape
    assert N % 64 == 0, "N must be a multiple of 64"
    gn = N // 64

    xg = x.reshape(M, gn, 8, 2, 4)
    # Shared HiF4 encode (same _hif4_encode_decode as quant_dequant_float, bit-for-bit identical).
    enc = hif4_encode(xg, ng=NG)

    # ---- HiF4 -> float: dequantize the codes we just produced (same bits a stored weight has) ----
    if output == "dequant":
        dequant = hif4_decode(enc.sign_bit, enc.e6m2, enc.de, enc.code, ng=NG)
        return dequant.reshape(M, N).to(dtype_ori)

    # ---- pack value: nibble = sign*8 + code, pair adjacent along N (even -> high nibble) ----
    nibble = (enc.sign_bit * 8 + enc.code).reshape(M, N // 2, 2)
    value = (nibble[..., 0] * 16 + nibble[..., 1]).to(torch.uint8)      # [M, N//2]

    # ---- pack scale (integer arithmetic, no bitwise ops) ----
    byte_e6m2 = ((enc.e_e6 + E6_OFFSET).to(torch.int32) * (2 ** E6M2_MBITS)
                 + (enc.q.to(torch.int32) - 4))                        # [M,gn]
    # NaN/Inf block poison: hif4_encode flags non-finite groups by making the E6M2 scale NaN;
    # store them as the reserved 0xFF E6M2 code (2^15 * 1.75) so the block decodes back to NaN.
    byte_e6m2 = torch.where(torch.isnan(enc.e6m2),
                            torch.full_like(byte_e6m2, 0xFF), byte_e6m2)

    pw8 = _pow2(torch.arange(7, -1, -1, device=x.device, dtype=torch.float32)).to(torch.int32)  # [128..1]
    byte_l1 = (enc.e1_8.to(torch.int32) * pw8).sum(dim=-1)              # [M,gn]
    e1_16f = enc.e1_16.reshape(M, gn, 16).to(torch.int32)              # k = a*2+b
    byte_l2_hi = (e1_16f[..., 0:8] * pw8).sum(dim=-1)
    byte_l2_lo = (e1_16f[..., 8:16] * pw8).sum(dim=-1)

    scale = torch.stack([byte_e6m2.to(torch.int32), byte_l1, byte_l2_hi, byte_l2_lo], dim=-1).to(torch.uint8)
    return scale, value


def hif4_bit_reconstruct(scale, value):
    """Decode (scale, value) back to the dequantized fp32 tensor [M, N] (for verification).

    Device-agnostic: accepts the uint8 outputs of :func:`hif4_hif4_pack_unpack` as torch tensors
    (moved to CPU) or numpy arrays, and rebuilds the same values ``quant_dequant_float`` (via ``_hif4_encode_decode``) would produce.
    """
    if isinstance(scale, torch.Tensor):
        scale = scale.cpu().numpy()
    if isinstance(value, torch.Tensor):
        value = value.cpu().numpy()
    scale = np.asarray(scale, dtype=np.int64)
    value = np.asarray(value, dtype=np.int64)
    M, gn, _ = scale.shape
    N = gn * 64

    b_e6m2 = scale[..., 0]
    b_l1 = scale[..., 1]
    b_l2_hi = scale[..., 2]
    b_l2_lo = scale[..., 3]

    e = (b_e6m2 >> E6M2_MBITS) - E6_OFFSET
    mant = b_e6m2 & ((1 << E6M2_MBITS) - 1)
    e6m2 = (2.0 ** e * (1.0 + mant / (2 ** E6M2_MBITS)))                  # [M,gn]

    a_idx = np.arange(8)
    e1_8 = ((b_l1[..., None] >> (7 - a_idx)) & 1).astype(np.float64)      # [M,gn,8]
    hi = (b_l2_hi[..., None] >> (7 - a_idx)) & 1
    lo = (b_l2_lo[..., None] >> (7 - a_idx)) & 1
    e1_16 = np.concatenate([hi, lo], axis=-1).reshape(M, gn, 8, 2).astype(np.float64)

    nib = np.stack([value >> 4, value & 0xF], axis=-1).reshape(M, N)      # [M,N]
    nib = nib.reshape(M, gn, 8, 2, 4)
    sign = np.where((nib >> 3) & 1 == 1, -1.0, 1.0)
    code = (nib & 0x7).astype(np.float64)
    in_grp_val = code / (2 ** NG)

    de = e1_16 + e1_8[..., None]                                          # [M,gn,8,2]
    dequant = sign * e6m2[..., None, None, None] * 2.0 ** de[..., None] * in_grp_val
    # byte0 (E6M2) == 0xFF is the reserved NaN code -> the whole 64-block is NaN.
    nan_block = (b_e6m2 == 0xFF)                                          # [M,gn]
    dequant = np.where(nan_block[..., None, None, None], np.float32(np.nan), dequant)
    return dequant.reshape(M, N).astype(np.float32)
