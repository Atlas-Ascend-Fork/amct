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

    All arithmetic is performed in fp32 with torch.ldexp for exact power-of-2 (NPU-safe),
    threshold comparison for level bits, and an extra bf16 rounding before the in-group
    code rounding.
    """
    assert Q.exp_bits == 0

    dtype_ori = x.dtype

    # Reshape quant dim into (-1, 8, 2, 4) blocks
    x = x.unflatten(qdim, (-1, 8, 2, 4))

    # Cast to fp32 -- all computation in fp32
    x = x.to(torch.float32)

    # Handle non-finite values
    xf = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
    tmp = xf.abs()
    sign_bit = (xf < 0).to(torch.int32)

    # Max values along block hierarchy
    v16 = tmp.amax(dim=-1)       # [..., gn, 8, 2]
    v8 = v16.amax(dim=-1)        # [..., gn, 8]
    vmax = v8.amax(dim=-1)       # [..., gn]

    NG = Q.man_bits - 1           # HiF4: man_bits=3 → NG=2
    E6_OFFSET = 48

    def _pow2(e):
        """Exact 2^e via ldexp -- NPU-safe, unlike torch.exp2 / 2**e."""
        return torch.ldexp(torch.ones_like(e, dtype=torch.float32), e.to(torch.int32))

    # E6M2 (level-1 scale): e = floor(log2(sf)); q = round(sf * 2^(2-e)) in {4..7}
    const_rec = (torch.tensor(1.0 / 7.0, dtype=torch.float32, device=x.device)).bfloat16().float()
    sf = (vmax * const_rec).bfloat16().float()

    sf_c = sf.clamp(2.0 ** -E6_OFFSET, 2.0 ** 15 * 1.5)
    e_e6 = torch.floor(torch.log2(sf_c))
    q_val = torch.round(sf_c * _pow2(2.0 - e_e6))          # {4,5,6,7}
    e6m2 = q_val * _pow2(e_e6 - 2.0)                        # E6M2 value
    rec = (1.0 / e6m2).bfloat16().float()                   # reciprocal, bf16-rounded

    # Level-2 (8x) exponent bits
    e1_8 = (v8 * rec.unsqueeze(-1) >= 4.0)                  # [..., gn, 8] bool
    e1_8x2 = e1_8.unsqueeze(-1).expand(*e1_8.shape, 2)      # [..., gn, 8, 2]

    # Level-3 (16x) exponent bits
    e1_16 = (v16 * rec[..., None, None] * _pow2(-e1_8x2.to(torch.float32)) >= 2.0)

    # In-group S1P2 code
    de = e1_16.to(torch.float32) + e1_8x2.to(torch.float32)  # {0,1,2}
    de64 = de.unsqueeze(-1).expand(*de.shape, 4)              # [..., gn, 8, 2, 4]

    in_grp = tmp * rec[..., None, None, None] * _pow2(-de64)
    in_grp = in_grp.bfloat16().float() * (2 ** NG)            # extra bf16 rounding
    code = torch.floor(in_grp + 0.5).clamp(0, (1 << (NG + 1)) - 1).to(torch.int32)

    # HiF4 → float: dequantize
    sign = 1.0 - 2.0 * sign_bit.to(torch.float32)             # +1 / -1 (zero → +1)
    dequant = (sign * e6m2[..., None, None, None] * _pow2(de64)
               * (code.to(torch.float32) / (2 ** NG)))

    # Restore original shape and dtype
    out = dequant.flatten(qdim - 3, qdim)
    if out.dtype != dtype_ori:
        out = out.to(dtype_ori)
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
