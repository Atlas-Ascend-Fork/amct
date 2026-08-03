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
"""Shared HiF4 (N=4) quantization core -- one copy of the 3-level-scale + S1P2 math.

Both the fake-quant path (``quant_dequant.quant_dequant_float``) and the real byte packer
(``hif4_pack_unpack.hif4_cast``) call the same ``hif4_encode`` / ``hif4_decode`` here, so the
evaluation path and the deployed packed weight can never drift apart -- change the math
once and both stay in sync.

NPU pitfall: every power of two goes through ``torch.ldexp`` (writes the exponent field
only -> bit-exact on Ascend, unlike ``torch.exp2`` / ``2**e``), and each shared scale gets
an explicit bf16 round to mirror the hardware.
"""
from typing import NamedTuple

import torch
from torch import Tensor

E6_OFFSET = 48          # E6M2 exponent bias: unbiased in [-48, 15] -> biased in [0, 63]
E6M2_MBITS = 2          # E6M2 mantissa bits


class Hif4Encoded(NamedTuple):
    """Per-64-group HiF4 encode outputs. ``B`` = all batch dims before the (8, 2, 4) block."""
    sign_bit: Tensor    # int32 [*B, 8, 2, 4]  1 where x < 0 (zeros -> 0)
    e6m2: Tensor        # fp32  [*B]           level-1 E6M2 scale value per 64-group
    de: Tensor          # fp32  [*B, 8, 2]     lv2 + lv3 exponent in {0, 1, 2}
    code: Tensor        # int32 [*B, 8, 2, 4]  S1P2 magnitude index in [0, 2**(ng+1) - 1]
    e1_8: Tensor        # bool  [*B, 8]        level-2 (8x) exponent bit (for byte packing)
    e1_16: Tensor       # bool  [*B, 8, 2]     level-3 (16x) exponent bit (for byte packing)
    e_e6: Tensor        # fp32  [*B]           E6M2 unbiased exponent (for byte packing)
    q: Tensor           # fp32  [*B]           E6M2 mantissa quotient in {4..7} (for packing)
    special_vals: Tensor  # fp32 [*B, 8, 2, 4]  original non-finite values (NaN/Inf), 0 for finite


def _pow2(e: Tensor) -> Tensor:
    """Exact 2**e via ldexp -- writes the exponent field only (bit-exact on NPU)."""
    return torch.ldexp(torch.ones_like(e, dtype=torch.float32), e.to(torch.int32))


def _to_bf16(x: Tensor) -> Tensor:
    """Round to bf16 and back to fp32 (round-half-to-even), matching the hardware scale path."""
    return x.bfloat16().float()


@torch.no_grad()
def hif4_encode(xg: Tensor, ng: int = 2) -> Hif4Encoded:
    """Encode a tensor whose trailing dims are ``(..., 8, 2, 4)`` -- one 64-group per (8, 2, 4).

    ``xg`` must already be float32. Everything before the (8, 2, 4) is treated as batch, so
    this serves both the fake-quant reshape ``unflatten(qdim, (-1, 8, 2, 4))`` and the
    packer's ``reshape(M, N // 64, 8, 2, 4)``. ``ng`` is the S1P2 in-group mantissa bit count
    (HiF4: 2, so the in-group code is 3 bits and value = code / 2**ng).
    """
    # Non-finite handling, matching the reference (hifx.py To_HiFX):
    # zero non-finite values before the max reduction so they don't pollute the shared scale,
    # then restore the original NaN/Inf element-by-element after decode.
    finite_mask = torch.isfinite(xg)                          # [*B, 8, 2, 4]
    special_vals = torch.where(finite_mask, torch.zeros_like(xg), xg)
    xf = torch.where(finite_mask, xg, torch.zeros_like(xg))
    tmp = xf.abs()
    sign_bit = (xf < 0).to(torch.int32)

    # Max values along the block hierarchy
    v16 = tmp.amax(dim=-1)                                    # [*B, 8, 2]
    v8 = v16.amax(dim=-1)                                     # [*B, 8]
    vmax = v8.amax(dim=-1)                                    # [*B]

    # E6M2 (level-1 scale): e = floor(log2(sf)); q = round(sf * 2^(2-e)) in {4..7}
    const_rec = _to_bf16(torch.tensor(1.0 / 7.0, dtype=torch.float32, device=xg.device))
    sf = _to_bf16(vmax * const_rec)
    sf_c = sf.clamp(2.0 ** -E6_OFFSET, 2.0 ** 15 * 1.5)
    e_e6 = torch.floor(torch.log2(sf_c))
    q = torch.round(sf_c * _pow2(2.0 - e_e6))                # {4, 5, 6, 7}
    e6m2 = q * _pow2(e_e6 - 2.0)                              # E6M2 value
    rec = _to_bf16(1.0 / e6m2)                               # reciprocal, bf16-rounded

    # Level-2 (8x) and level-3 (16x) exponent bits
    e1_8 = (v8 * rec.unsqueeze(-1) >= 4.0)                    # [*B, 8] bool
    e1_8x2 = e1_8.unsqueeze(-1).expand(*e1_8.shape, 2)       # [*B, 8, 2]
    e1_16 = (v16 * rec[..., None, None] * _pow2(-e1_8x2.to(torch.float32)) >= 2.0)

    # In-group S1P2 code
    de = e1_16.to(torch.float32) + e1_8x2.to(torch.float32)  # {0, 1, 2}
    de64 = de.unsqueeze(-1).expand(*de.shape, 4)             # [*B, 8, 2, 4]
    in_grp = tmp * rec[..., None, None, None] * _pow2(-de64)
    in_grp = _to_bf16(in_grp) * (2 ** ng)                    # extra bf16 rounding
    code = torch.floor(in_grp + 0.5).clamp(0, (1 << (ng + 1)) - 1).to(torch.int32)

    return Hif4Encoded(sign_bit, e6m2, de, code, e1_8, e1_16, e_e6, q, special_vals)


@torch.no_grad()
def hif4_decode(sign_bit: Tensor, e6m2: Tensor, de: Tensor, code: Tensor, ng: int = 2,
                special_vals: Tensor | None = None) -> Tensor:
    """Decode encoder outputs back to fp32, shaped like ``code`` (``[*B, 8, 2, 4]``).

    If ``special_vals`` is provided, it contains the original non-finite values (NaN/Inf) where
    the input was non-finite and 0 elsewhere. Adding it to the decoded tensor restores the exact
    original NaN/Inf per element — matching the private hifx.py reference (NaN→NaN, +Inf→+Inf,
    -Inf→-Inf). The IEEE 754 addition ``finite + NaN = NaN``, ``0 + Inf = Inf`` makes this exact.
    """
    de64 = de.unsqueeze(-1).expand(*de.shape, 4)
    sign = 1.0 - 2.0 * sign_bit.to(torch.float32)            # +1 / -1 (zero -> +1)
    out = sign * e6m2[..., None, None, None] * _pow2(de64) * (code.to(torch.float32) / (2 ** ng))

    if special_vals is not None:
        # Restore original non-finite values element-by-element (0 + NaN → NaN, 0 + Inf → Inf).
        out = out + special_vals
    return out
