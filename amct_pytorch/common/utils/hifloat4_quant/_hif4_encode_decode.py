# -*- coding: UTF-8 -*-
from __future__ import annotations
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
"""Shared HiF4 (N=4) quantization core — one copy of the 3-level-scale + S1P2 math.

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
    block_nan: Tensor   # bool  [*B]           True where the 64-group has any non-finite (NaN/Inf)


def _pow2(e: Tensor) -> Tensor:
    """Exact 2**e via ldexp -- writes the exponent field only (bit-exact on NPU)."""
    return torch.ldexp(torch.ones_like(e, dtype=torch.float32), e.to(torch.int32))


def _to_bf16(x: Tensor) -> Tensor:
    """Round to bf16 and back to fp32 (round-half-to-even), matching the hardware scale path."""
    return x.bfloat16().float()


@torch.no_grad()
def hif4_encode(xg: Tensor, ng: int = 2, qdim: int = -1) -> Hif4Encoded:
    """Encode a tensor whose block hierarchy dims ``(8, 2, 4)`` are located at
    ``(qdim-2, qdim-1, qdim)`` — ``qdim`` is the dim index of the innermost ``4``
    block-sub-dim (negative, relative to ``xg.ndim``).

    Uses ``amax(dim=…, keepdim=True)`` matching the private ``hifx.py`` reference
    (``torch.max(..., dim=qdim, keepdim=True)``).  With ``keepdim`` every intermediate
    tensor shares the same ndim as ``xg``, so broadcasting of ``rec`` is automatic --
    no manual ``unsqueeze`` / ``expand``.

    When the caller guarantees ``(8, 2, 4)`` are the trailing three dims (as
    ``hif4_pack_unpack.hif4_cast`` does), the default ``qdim=-1`` is correct.
    ``quant_dequant_float`` passes the adjusted negative ``qdim`` from ``Q.q_dim``
    (after ``unflatten`` the innermost block-sub-dim "4" sits at that index).
    """
    # Non-finite handling -- HiF4 block-poison: zero non-finite values before the max
    # reduction; if any element in a 64-group is non-finite the whole group becomes NaN.
    finite_mask = torch.isfinite(xg)
    block_nan = (~finite_mask).flatten(start_dim=qdim - 2).any(dim=-1)
    xf = torch.where(finite_mask, xg, torch.zeros_like(xg))
    tmp = xf.abs()
    sign_bit = (xf < 0).to(torch.int32)

    # Max hierarchy with keepdim=True (matching private hifx.py).
    # All tensors below share xg.ndim; 1s in reduced dims make broadcast with rec trivial.
    v16 = tmp.amax(dim=qdim, keepdim=True)
    v8 = v16.amax(dim=qdim - 1, keepdim=True)
    vmax = v8.amax(dim=qdim - 2, keepdim=True)

    # E6M2 (level-1 scale)
    const_rec = _to_bf16(torch.tensor(1.0 / 7.0, dtype=torch.float32, device=xg.device))
    sf = _to_bf16(vmax * const_rec)
    sf_c = sf.clamp(2.0 ** -E6_OFFSET, 2.0 ** 15 * 1.5)
    e_e6 = torch.floor(torch.log2(sf_c))
    q = torch.round(sf_c * _pow2(2.0 - e_e6))
    e6m2_val = q * _pow2(e_e6 - 2.0)
    rec = _to_bf16(1.0 / e6m2_val)

    # Level-2 (8x) and level-3 (16x) exponent bits — all same ndim, rec broadcasts naturally.
    e1_8 = (v8 * rec >= 4.0)                                  # [*B, 8, 1, 1, …]
    # Expand the "2" keepdim (at qdim-1) from 1 → 2.
    pl = e1_8.ndim + qdim - 1                                  # split before "2" keepdim
    e1_8x2 = e1_8.expand(*e1_8.shape[:pl], 2, *e1_8.shape[pl + 1:])
    e1_16 = (v16 * rec * _pow2(-e1_8x2.to(torch.float32)) >= 2.0)

    # In-group S1P2 code
    de = e1_16.to(torch.float32) + e1_8x2.to(torch.float32)
    in_grp = tmp * rec * _pow2(-de.to(tmp.dtype))
    in_grp = _to_bf16(in_grp) * (2 ** ng)
    code = torch.floor(in_grp + 0.5).clamp(0, (1 << (ng + 1)) - 1).to(torch.int32)

    # Collapse keepdim-1s back to the shapes expected by hif4_decode / packer.
    # ─ sign_bit, code: original xg shape, no squeeze needed.
    # ─ e6m2_val, e_e6, q: 3 keepdim 1s at qdim-2, qdim-1, qdim → squeeze all.
    # ─ e1_8: 2 keepdim 1s at qdim-1, qdim → squeeze from the right.
    # ─ e1_16, de: 1 keepdim 1 at qdim → squeeze that one.
    e1_8_sq = e1_8.squeeze(qdim).squeeze(qdim)       # squeeze backwards: after first, the remaining 1 shifts to qdim
    return Hif4Encoded(
        sign_bit,
        e6m2_val.squeeze(),
        de.squeeze(qdim),
        code,
        e1_8_sq,
        e1_16.squeeze(qdim),
        e_e6.squeeze(),
        q.squeeze(),
        block_nan,
    )


@torch.no_grad()
def hif4_decode(sign_bit: Tensor, e6m2: Tensor, de: Tensor, code: Tensor, ng: int = 2,
                block_nan: Tensor | None = None, qdim: int = -1) -> Tensor:
    """Decode encoder outputs back to fp32, shaped like ``code`` (``[*B, 8, 2, 4]``).

    ``qdim`` is the innermost block-sub-dim index (negative) of the *original* encoded
    tensor.  It controls where the three broadcast dimensions are inserted so that
    ``e6m2`` and ``block_nan`` line up with the block hierarchy regardless of whether
    ``(8, 2, 4)`` are trailing.  Default ``-1`` = trailing (the packer path).

    If ``block_nan`` is provided, every 64-group flagged there is set **entirely to
    NaN** — HiF4 **block-poison**: a group that had any non-finite input (NaN or +-Inf)
    reconstructs as an all-NaN group.
    """
    # Expand de from [*B, 8, 2] to [*B, 8, 2, 4].
    de64 = de.unsqueeze(qdim)
    target = list(de64.shape)
    target[de64.ndim + qdim] = 4
    de64 = de64.expand(target)

    sign = 1.0 - 2.0 * sign_bit.to(torch.float32)

    # Broadcast e6m2 [*B] to match sign's ndim: insert 1s at positions
    # qdim-2 (8), qdim-1 (2), qdim (4) of the target tensor.
    e6m2_exp = e6m2
    nd_target = sign.ndim
    for pos in sorted([nd_target + qdim - 2, nd_target + qdim - 1, nd_target + qdim]):
        e6m2_exp = e6m2_exp.unsqueeze(pos)

    out = sign * e6m2_exp * _pow2(de64) * (code.to(torch.float32) / (2 ** ng))

    if block_nan is not None:
        bn_exp = block_nan
        # Insert 1s at the block-keepdim positions, then pad any trailing dims.
        for pos in sorted([nd_target + qdim - 2, nd_target + qdim - 1, nd_target + qdim]):
            bn_exp = bn_exp.unsqueeze(pos)
        while bn_exp.ndim < nd_target:
            bn_exp = bn_exp.unsqueeze(-1)
        out = torch.where(bn_exp, torch.full_like(out, float("nan")), out)
    return out
