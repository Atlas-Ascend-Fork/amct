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
"""Pure-PyTorch hifx4 quant-dequant simulation.

Only the ``force_py`` path is kept: ``quant_hifx`` (the bit-accurate torch reference) plus a
trimmed ``quant_dequant_float`` driver that pads the quant dimension to the block size, runs
the reference, and slices/casts back. The compiled CUDA/CCE kernels and the autograd
``Function`` wrappers are intentionally dropped -- amct uses this purely for inference-time
fake-quant (``@torch.no_grad``), so no speed kernel and no backward is needed.
"""
import torch
import torch.nn.functional as F
from torch import Tensor

from .qtype import QType


@torch.no_grad()
def quant_hifx(x: Tensor, Q: QType, qdim: int) -> Tensor:
    """HiFx fake-quant reference (3-level hierarchical block scaling), pure torch.

    The quant dimension is reshaped into (-1, 8, 2, 4) blocks of 64 elements, so the caller
    must have padded ``x.shape[qdim]`` to a multiple of 64 (done by quant_dequant_float).
    """
    # reshape x
    x = x.unflatten(qdim, (-1, 8, 2, 4))
    x_unsigned = torch.abs(x)
    sign = torch.sign(x)

    assert Q.exp_bits == 0
    # compute initial shared exp
    max_lv3 = torch.max(x_unsigned, dim=qdim, keepdim=True)[0]
    max_lv2 = torch.max(max_lv3, dim=qdim - 1, keepdim=True)[0]
    max_lv1 = torch.max(max_lv2, dim=qdim - 2, keepdim=True)[0]

    div7 = torch.ones_like(max_lv1) / 7.0
    div7 = div7.to(torch.bfloat16).to(x.dtype)
    scale_factor = max_lv1 * div7
    scale_factor = (scale_factor).to(torch.bfloat16).to(x.dtype).clip(min=2 ** (-48), max=49152)
    # change to tobf16(rint)
    e_sf = torch.floor(torch.log2(scale_factor))
    mant_sf = scale_factor / 2 ** e_sf * 2 ** 7
    scale_factor = torch.round(mant_sf) / 2 ** 7 * 2 ** e_sf

    # scale_factor to e6m2
    e_sf = torch.floor(torch.log2(scale_factor))
    scale_factor = torch.round(scale_factor * torch.exp2(2 - e_sf)) * torch.exp2(e_sf - 2)

    rec_sf = (1.0 / scale_factor).to(torch.bfloat16).to(x.dtype)
    scale_lv2 = (max_lv2 * rec_sf)
    scale_lv2 = torch.exp2((scale_lv2.clip(0, 4) / 4).floor())
    scale_lv3 = torch.exp2(((max_lv3 * rec_sf / scale_lv2).clip(0, 2) / 2).floor())

    mant = x_unsigned / scale_lv2 / scale_lv3 * rec_sf
    mant = torch.floor(mant * 2 ** (Q.man_bits - 1) + 0.5) / 2 ** (Q.man_bits - 1)
    mant[mant >= 2] = 2 - 2 ** (-Q.man_bits + 1)

    out = sign * mant * scale_lv2 * scale_lv3 * scale_factor

    out = out.flatten(qdim - 3, qdim)
    return out


@torch.no_grad()
def quant_dequant_float(x: Tensor, Q: QType, force_py: bool = False, force_fp32: bool = False, **kwargs) -> Tensor:
    """Fake-quantize ``x`` to the format described by ``Q`` and dequantize back to float.

    ``force_py`` is accepted for API compatibility but ignored: only the pure-python path is
    shipped. ``force_fp32`` upcasts to float32 before the round
    trip (recommended for accuracy), restoring the original dtype on the way out.
    """
    # convert to fp32 if forced
    dtype_ori = x.dtype
    if force_fp32:
        x = x.to(torch.float32)

    # pad to fit block size
    C = x.shape[Q.q_dim]
    blk_size_total = Q.blk_size * Q.blk_outer_size
    padC = (blk_size_total - C % blk_size_total) % blk_size_total
    qdim = Q.q_dim
    if qdim >= 0:
        qdim = qdim - len(x.shape)
    if padC > 0:
        pads = [0] * (-qdim * 2 - 1) + [padC]
        x = F.pad(x, pads, value=0.0)

    out = _get_torch_func(Q, qdim)(x)

    # slice and resize back. Use a tuple (basic indexing) -- a list of slices is treated as
    # advanced indexing by torch/numpy and would not mean "slice every dim".
    if padC > 0:
        slices = [slice(0, out.shape[i]) for i in range(len(x.shape) + qdim)] + [slice(0, C), ]
        out = out[tuple(slices)]

    if out.dtype != dtype_ori:
        out = out.to(dtype_ori)
    return out


def _get_torch_func(Q: QType, qdim: int):
    """Resolve the pure-python quant function for ``Q``. Only hifx* is supported here."""
    import re
    from functools import partial
    if re.match(r'^hifx[0-9]*$', Q.desc):
        return partial(quant_hifx, Q=Q, qdim=qdim)
    raise ValueError(
        f"hifx4_quant only implements the 'hifx*' fake-quant path, got '{Q.desc}'. "
        "Add the corresponding QFuncs reference if another format is needed."
    )
