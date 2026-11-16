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

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""fp16fixed operator classes (F16xF16 / F16xS8 fixed point simulation)."""

from .conv2d import Fp16FixedConv2d
from .conv2d_s8 import Fp16FixedConv2dS8
from .convtranspose2d import Fp16FixedConvTranspose2d
from .convtranspose2d_s8 import Fp16FixedConvTranspose2dS8
from .linear import Fp16FixedLinear
from .linear_s8 import Fp16FixedLinearS8
from .shift_value_search import ShiftValueSearchMixin

__all__ = [
    'Fp16FixedConv2d',
    'Fp16FixedConv2dS8',
    'Fp16FixedConvTranspose2d',
    'Fp16FixedConvTranspose2dS8',
    'Fp16FixedLinear',
    'Fp16FixedLinearS8',
    'ShiftValueSearchMixin',
]
