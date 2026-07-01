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
"""Self-contained bit-exact HiFloat4 (hifx4) fake-quant reference (pure torch, CPU).

This is the correctness oracle for the AscendC kernel (P1.6): the kernel output must match
this to ~0 error (< 1e-6), including zero-input blocks. It re-implements the same math as
amct_pytorch/common/utils/hifx4_quant/quant_dequant.py::quant_hifx, but standalone (no
amct_pytorch dependency) so amct_ops can ship and be validated on its own. The amct test
(tests/amct_ops/test_hifloat4_cast.py) cross-checks this against amct's quant_hifx.

Format: per element S1P2 (man_bits=3); per 64 elements a 3-level scale
(L1 scale_factor E6M2 over 64, L2 over 8, L3 over 4).
"""
import torch
import torch.nn.functional as F
from torch import Tensor

_MAN_BITS = 3
_BLOCK = 64


@torch.no_grad()
def _quant_hifx_blocked(x: Tensor, qdim: int) -> Tensor:
    """Core reference on a tensor already padded so x.shape[qdim] % 64 == 0."""
    x = x.unflatten(qdim, (-1, 8, 2, 4))
    x_unsigned = torch.abs(x)
    sign = torch.sign(x)

    max_lv3 = torch.max(x_unsigned, dim=qdim, keepdim=True)[0]
    max_lv2 = torch.max(max_lv3, dim=qdim - 1, keepdim=True)[0]
    max_lv1 = torch.max(max_lv2, dim=qdim - 2, keepdim=True)[0]

    div7 = (torch.ones_like(max_lv1) / 7.0).to(torch.bfloat16).to(x.dtype)
    scale_factor = (max_lv1 * div7).to(torch.bfloat16).to(x.dtype).clip(min=2 ** (-48), max=49152)
    # round to 7-bit mantissa (bf16 rint), then to E6M2 (2-bit mantissa)
    e_sf = torch.floor(torch.log2(scale_factor))
    mant_sf = scale_factor / 2 ** e_sf * 2 ** 7
    scale_factor = torch.round(mant_sf) / 2 ** 7 * 2 ** e_sf
    e_sf = torch.floor(torch.log2(scale_factor))
    scale_factor = torch.round(scale_factor * torch.exp2(2 - e_sf)) * torch.exp2(e_sf - 2)

    rec_sf = (1.0 / scale_factor).to(torch.bfloat16).to(x.dtype)
    scale_lv2 = torch.exp2(((max_lv2 * rec_sf).clip(0, 4) / 4).floor())
    scale_lv3 = torch.exp2(((max_lv3 * rec_sf / scale_lv2).clip(0, 2) / 2).floor())

    mant = x_unsigned / scale_lv2 / scale_lv3 * rec_sf
    mant = torch.floor(mant * 2 ** (_MAN_BITS - 1) + 0.5) / 2 ** (_MAN_BITS - 1)
    mant[mant >= 2] = 2 - 2 ** (-_MAN_BITS + 1)

    out = sign * mant * scale_lv2 * scale_lv3 * scale_factor
    return out.flatten(qdim - 3, qdim)


@torch.no_grad()
def hifloat4_reference(x: Tensor, qdim: int = -1, force_fp32: bool = True) -> Tensor:
    """Reference FP -> HiF4 -> FP round trip. Pads ``qdim`` to a multiple of 64, runs the
    blocked reference, slices back, and restores dtype. ``force_fp32`` upcasts for accuracy.
    """
    dtype_ori = x.dtype
    if force_fp32:
        x = x.to(torch.float32)

    nd = x.dim()
    qd = qdim if qdim >= 0 else qdim + nd
    channels = x.shape[qd]
    pad = (_BLOCK - channels % _BLOCK) % _BLOCK
    neg_qdim = qd - nd
    if pad > 0:
        pads = [0] * (-neg_qdim * 2 - 1) + [pad]
        x = F.pad(x, pads, value=0.0)

    out = _quant_hifx_blocked(x, neg_qdim)

    if pad > 0:
        slices = [slice(0, out.shape[i]) for i in range(nd + neg_qdim)] + [slice(0, channels)]
        out = out[tuple(slices)]

    return out.to(dtype_ori)
