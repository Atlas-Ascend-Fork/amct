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
HiFloat8 encode/decode operators for Ascend NPU

Registers torch.ops.amct.encode_to_hifloat8 and torch.ops.amct.decode_from_hifloat8
and re-exports them as module-level names for convenience.

Usage:
    # After loading the library, either form works:
    import torch.ops.amct
    torch.ops.amct.encode_to_hifloat8(x)          # FP16/BF16 → HiF8
    torch.ops.amct.decode_from_hifloat8(y)         # HiF8 → BF16 (default)
    torch.ops.amct.decode_from_hifloat8(y, torch.float16)

    from amct_ops.hifloat8_cast import encode_to_hifloat8, decode_from_hifloat8
    encode_to_hifloat8(x)
    decode_from_hifloat8(y, torch.float16)
"""
__all__ = [
    'encode_to_hifloat8',
    'decode_from_hifloat8',
]

import os
import torch_npu  # noqa: F401 — registers PrivateUse1 backend
import torch

_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libhifloat8_cast_ops.so")
torch.ops.load_library(_lib_path)

from .ops import encode_to_hifloat8, decode_from_hifloat8  # noqa: E402, F401
