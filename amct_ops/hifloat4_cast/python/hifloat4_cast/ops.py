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
"""Thin python wrapper for the HiFloat4 cast NPU operator."""

import torch


def hifloat4_fake_quant(x: torch.Tensor, qdim: int = -1) -> torch.Tensor:
    """FP -> HiFloat4 -> FP fake-quant round trip on the NPU.

    HiFloat4 (hifx4) is a 4-bit block-scaling float: per element S1P2, plus a 3-level scale
    shared by every 64 elements along ``qdim``. This is a simulation (fake-quant): the output
    is the same shape and dtype as the input, not real packed 4-bit.

    Args:
        x: Input tensor on NPU, dtype float16 or bfloat16 (device/dtype are enforced by the op).
        qdim: Reduction/quant dim along which 64-element blocks are formed (default: last).
              Its length must be a multiple of 64, otherwise the op raises RuntimeError.
              The op moves it to the last axis internally.

    Returns:
        Tensor of the same shape and dtype, fake-quantized to HiFloat4.

    Example:
        >>> from amct_ops.hifloat4_cast import hifloat4_fake_quant
        >>> y = hifloat4_fake_quant(x)             # blocks along the last dim
        >>> y = hifloat4_fake_quant(w, qdim=1)     # e.g. Linear weight [out, in] along in
        >>> y = torch.ops.amct.hifloat4_fake_quant(x, -1)   # equivalent
    """
    return torch.ops.amct.hifloat4_fake_quant(x, qdim)
