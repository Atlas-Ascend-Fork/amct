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
"""Python wrapper for the HiFloat4 cast NPU operator.

The device kernel (``torch.ops.amct.hifloat4_cast``) is deliberately simple: it fake-quantizes
a contiguous FP16/BF16 buffer whose length is a multiple of 64, treating every consecutive
64 elements as one shared-scale block. This wrapper does the layout work so any tensor / any
quant dim is supported: move ``qdim`` to the last axis, pad it to a multiple of 64, flatten,
call the kernel, then undo the pad/permute.
"""
import torch
import torch.nn.functional as F

_BLOCK = 64


def hifloat4_fake_quant(x: torch.Tensor, qdim: int = -1) -> torch.Tensor:
    """FP -> HiFloat4 -> FP fake-quant round trip on the NPU.

    Args:
        x: Input tensor on NPU, dtype float16 or bfloat16.
        qdim: Reduction/quant dim along which 64-element blocks are formed (default: last).

    Returns:
        Tensor of the same shape and dtype, fake-quantized to HiFloat4.

    Example:
        >>> from amct_ops.hifloat4_cast import hifloat4_fake_quant
        >>> y = hifloat4_fake_quant(x)            # qdim = -1
        >>> y = hifloat4_fake_quant(w, qdim=1)    # e.g. Linear weight [out, in] along in
    """
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"hifloat4_fake_quant expects float16 or bfloat16, got {x.dtype}")

    nd = x.dim()
    qd = qdim if qdim >= 0 else qdim + nd

    # Move the quant dim to the last axis and make it contiguous so a block of 64 along
    # qdim is 64 contiguous elements in memory.
    perm = [i for i in range(nd) if i != qd] + [qd]
    xp = x.permute(perm).contiguous()

    channels = xp.shape[-1]
    pad = (_BLOCK - channels % _BLOCK) % _BLOCK
    if pad:
        xp = F.pad(xp, (0, pad), value=0.0)  # zero-pad -> padded lanes contribute 0 to block max
    padded_shape = xp.shape

    flat = xp.reshape(-1)
    out = torch.ops.amct.hifloat4_cast(flat)
    out = out.reshape(padded_shape)

    if pad:
        out = out[..., :channels]

    inv = [0] * nd
    for new_pos, old_dim in enumerate(perm):
        inv[old_dim] = new_pos
    return out.permute(inv).contiguous()
