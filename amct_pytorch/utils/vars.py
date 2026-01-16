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
import torch

SUPPORT_WEIGHT_QUANT_DTYPE = ['int8', 'int4']
SUPPORT_INPUT_QUANT_DTYPE = ['int8', 'int4']
SUPPORT_QUANT_STRATEGY_WEIGHT = ['tensor', 'channel', 'group']
SUPPORT_QUANT_STRATEGY_INPUT = ['tensor', 'token']

INT8_MAX = torch.iinfo(torch.int8).max
INT8_MIN = torch.iinfo(torch.int8).min
INT4_MAX = 7
INT4_MIN = -8
INT8 = 'int8'
INT4 = 'int4'

SUPPORTED_QUANT_DTYPE_COMB = [
    'int8 int8',
    'NOT_QUANTIZE int8',
    'NOT_QUANTIZE int4',
]

ALGORITHM_SUPPORTED_QUANT_TYPE_COMB = {
    'int8 int8': ['minmax', 'smoothquant'],
    'NOT_QUANTIZE int8': ['minmax', 'awq', 'gptq'],
    'NOT_QUANTIZE int4': ['minmax', 'awq', 'gptq'],
}

ALLOWED_WEIGHT_DTYPES = {
    'NOT_QUANTIZE int4': [torch.float32, torch.bfloat16, torch.float16],
    'NOT_QUANTIZE int8': [torch.bfloat16, torch.float16],
    'int8 int8': [torch.bfloat16, torch.float16],
}