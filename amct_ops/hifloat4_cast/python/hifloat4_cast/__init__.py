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
"""
HiFloat4 (hifx4) FP->HiF4->FP fake-quant operator for Ascend NPU.

Registers torch.ops.amct.hifloat4_fake_quant and exposes it as hifloat4_fake_quant(x, qdim=-1).

Usage:
    from amct_ops.hifloat4_cast import hifloat4_fake_quant
    y = hifloat4_fake_quant(x)                       # FP16/BF16 -> HiF4 -> FP16/BF16
    y = torch.ops.amct.hifloat4_fake_quant(x, -1)    # equivalent
"""

__all__ = [
    'hifloat4_fake_quant',
]

import os
import torch_npu  # noqa: F401 — registers PrivateUse1 backend
import torch

from .ops import hifloat4_fake_quant  # noqa: F401

# The C++ extension must be loaded before the first call so torch.ops.amct resolves.
_lib_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "libhifloat4_cast_ops.so"
)
torch.ops.load_library(_lib_path)
