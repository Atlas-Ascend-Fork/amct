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

``quant_dequant_float`` is the only entry point: it pads the quant dimension to the 64-block
size, runs the shared HiF4 encode/decode from ``_hif4_encode_decode`` (bit-for-bit identical to the
real byte packer ``hif4_pack_unpack.hif4_cast``), then slices/casts back. The compiled CUDA/CCE
kernels and the autograd ``Function`` wrappers are intentionally dropped -- amct uses this
purely for inference-time fake-quant (``@torch.no_grad``), so no speed kernel and no backward
is needed.
"""
import torch
import torch.nn.functional as F
from torch import Tensor

from .qtype import QType
from ._hif4_encode_decode import hif4_encode, hif4_decode


@torch.no_grad()
def quant_dequant_float(x: Tensor, Q: QType, force_py: bool = False, force_fp32: bool = False, **kwargs) -> Tensor:
    """Fake-quantize ``x`` to the HiF4 format described by ``Q`` and dequantize back to float.

    Pads ``x.shape[Q.q_dim]`` to a multiple of 64, reshapes into (-1, 8, 2, 4) blocks, runs the
    shared HiF4 encode/decode (``_hif4_encode_decode``, bit-for-bit identical to the real byte packer
    ``hif4_pack_unpack.hif4_cast``), then slices the pad off and restores the original shape/dtype.
    Only ``hifx*`` formats are supported.

    ``force_py`` is accepted for API compatibility but ignored (only the pure-python path is
    shipped). ``force_fp32`` upcasts to float32 before the round trip (recommended for accuracy,
    matching the NPU kernel's internal fp32), restoring the original dtype on the way out.
    """
    import re
    if not re.match(r'^hifx[0-9]*$', Q.desc):
        raise ValueError(
            f"hifloat4_quant only implements the 'hifx*' fake-quant path, got '{Q.desc}'. "
            "Add the corresponding QFuncs reference if another format is needed."
        )
    assert Q.exp_bits == 0

    # convert to fp32 if forced
    dtype_ori = x.dtype
    if force_fp32:
        x = x.to(torch.float32)

    # pad quant dim to a multiple of the block size
    C = x.shape[Q.q_dim]
    blk_size_total = Q.blk_size * Q.blk_outer_size
    padC = (blk_size_total - C % blk_size_total) % blk_size_total
    qdim = Q.q_dim
    if qdim >= 0:
        qdim = qdim - len(x.shape)
    if padC > 0:
        pads = [0] * (-qdim * 2 - 1) + [padC]
        x = F.pad(x, pads, value=0.0)

    # HiF4 fake-quant: reshape quant dim into (-1, 8, 2, 4) blocks, then shared encode -> decode
    # in fp32 (same math as hif4_pack_unpack.hif4_cast, see _hif4_encode_decode).
    ng = Q.man_bits - 1                       # HiF4: man_bits=3 -> ng=2
    xb = x.unflatten(qdim, (-1, 8, 2, 4)).to(torch.float32)
    enc = hif4_encode(xb, ng=ng)
    out = hif4_decode(enc.sign_bit, enc.e6m2, enc.de, enc.code, ng=ng,
                      special_vals=enc.special_vals).flatten(qdim - 3, qdim)

    # slice the pad off. Use a tuple (basic indexing) -- a list of slices is treated as
    # advanced indexing by torch/numpy and would not mean "slice every dim".
    if padC > 0:
        slices = [slice(0, out.shape[i]) for i in range(len(x.shape) + qdim)] + [slice(0, C), ]
        out = out[tuple(slices)]

    if out.dtype != dtype_ori:
        out = out.to(dtype_ori)
    return out
