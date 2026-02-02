# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

__all__ = [
    'BaseQuantizeModule',
    'GPTQuant',
    'LinearAWQuant',
    'SmoothQuant',
    'MinMaxQuant',
    'OfmrQuant'
]

from amct_pytorch.quantize_op.base_quant_module import BaseQuantizeModule
from amct_pytorch.quantize_op.gptq_module import GPTQuant
from amct_pytorch.quantize_op.linear_awq_module import LinearAWQuant
from amct_pytorch.quantize_op.smooth_quant_module import SmoothQuant
from amct_pytorch.quantize_op.minmax_module import MinMaxQuant
from amct_pytorch.quantize_op.ofmr_quant_module import OfmrQuant
