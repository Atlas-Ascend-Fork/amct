#!/usr/bin/env python3
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
amct_ops — AMCT custom operators for Ascend NPU
===============================================

Available submodules
--------------------
amct_ops.hifloat8_cast
    FP16 / BF16  ↔  HiFloat8 (8-bit float) conversion.

    Functions
    ~~~~~~~~~
    encode_to_hifloat8(x)                    FP16/BF16 → uint8 (HiFloat8)
    decode_from_hifloat8(x, dtype=bfloat16)  uint8 → FP16/BF16

    Also accessible as:
        torch.ops.amct.encode_to_hifloat8
        torch.ops.amct.decode_from_hifloat8

Quick start
-----------
>>> import torch
>>> from amct_ops.hifloat8_cast import encode_to_hifloat8, decode_from_hifloat8
>>>
>>> x = torch.randn(1024, dtype=torch.float16, device="npu")
>>> y = encode_to_hifloat8(x)           # → uint8, same shape
>>> z = decode_from_hifloat8(y)         # → bfloat16, same shape
"""

__all__ = ['hifloat8_cast']

from . import hifloat8_cast
