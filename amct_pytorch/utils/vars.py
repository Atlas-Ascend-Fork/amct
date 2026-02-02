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

SUPPORT_WEIGHT_QUANT_DTYPE = ['int8', 'int4', 'hifloat8', 'float8_e4m3fn', 'mxfp4_e2m1', 'float4_e2m1', 'mxfp8_e4m3fn']
SUPPORT_INPUT_QUANT_DTYPE = ['int8', 'int4', 'hifloat8', 'float8_e4m3fn', 'mxfp8_e4m3fn']
SUPPORT_QUANT_STRATEGY_WEIGHT = ['tensor', 'channel', 'group']
SUPPORT_QUANT_STRATEGY_INPUT = ['tensor', 'token']

INT8_MAX = torch.iinfo(torch.int8).max
INT8_MIN = torch.iinfo(torch.int8).min
INT4_MAX = 7
INT4_MIN = -8
INT8 = 'int8'
INT4 = 'int4'
HIFLOAT8 = 'hifloat8'
FLOAT8_E4M3FN = 'float8_e4m3fn'
FLOAT4_E2M1 = 'float4_e2m1'
MXFP8_E4M3FN = 'mxfp8_e4m3fn'
MXFP4_E2M1 = 'mxfp4_e2m1'
FLOAT16 = 'float16'

WTS_ASYMMETRIC_DTYPE = [INT8, INT4]
GROUP_SIZE_SUPPORTED_DTYPE = [INT4, INT8, FLOAT4_E2M1, MXFP4_E2M1, MXFP8_E4M3FN]

GROUP_SIZE_SUPPORTED_MAP = {
    MXFP8_E4M3FN: [32],
    FLOAT4_E2M1: [32, 64, 128, 256],
    MXFP4_E2M1: [32]
}

ALGORITHM_SUPPORTED_QUANT_TYPE_COMB = {
    'int8 int8': ['minmax', 'smoothquant'],
    'NOT_QUANTIZE int8': ['minmax', 'awq', 'gptq'],
    'NOT_QUANTIZE int4': ['minmax', 'awq', 'gptq'],
    'hifloat8 hifloat8': ['ofmr'],
    'float8_e4m3fn float8_e4m3fn': ['ofmr'],
    'mxfp8_e4m3fn mxfp8_e4m3fn': ['mxquant'],
    'float8_e4m3fn float4_e2m1': ['minmax', 'smoothquant'],
    'NOT_QUANTIZE hifloat8': ['ofmr', 'gptq'],
    'NOT_QUANTIZE float8_e4m3fn': ['ofmr', 'gptq'],
    'NOT_QUANTIZE mxfp4_e2m1': ['awq', 'gptq', 'mxquant'],
    'NOT_QUANTIZE float4_e2m1': ['minmax', 'awq', 'gptq'],
}

ALLOWED_WEIGHT_DTYPES = {
    'NOT_QUANTIZE int4': [torch.float32, torch.bfloat16, torch.float16],
    'NOT_QUANTIZE int8': [torch.bfloat16, torch.float16],
    'int8 int8': [torch.bfloat16, torch.float16],
    'hifloat8 hifloat8': [torch.float32, torch.float16, torch.bfloat16],
    'float8_e4m3fn float8_e4m3fn': [torch.float32, torch.float16, torch.bfloat16],
    'mxfp8_e4m3fn mxfp8_e4m3fn': [torch.bfloat16],
    'float8_e4m3fn float4_e2m1': [torch.bfloat16],
    'NOT_QUANTIZE hifloat8': [torch.bfloat16, torch.float16],
    'NOT_QUANTIZE float8_e4m3fn': [torch.bfloat16, torch.float16],
    'NOT_QUANTIZE mxfp4_e2m1': [torch.bfloat16, torch.float16],
    'NOT_QUANTIZE float4_e2m1': [torch.bfloat16, torch.float16],
}

WTS_PER_TENSOR_SUPPORT_COMBINATION = ['hifloat8 hifloat8', 'float8_e4m3fn float8_e4m3fn', 
                                'int8 int8', 'NOT_QUANTIZE int8', 'NOT_QUANTIZE int4',
                                'NOT_QUANTIZE hifloat8', 'NOT_QUANTIZE float8_e4m3fn']
WTS_PER_CHANNEL_SUPPORT_COMBINATION = ['hifloat8 hifloat8', 'float8_e4m3fn float8_e4m3fn', 
                                'int8 int8', 'NOT_QUANTIZE int8', 'NOT_QUANTIZE int4',
                                'NOT_QUANTIZE hifloat8', 'NOT_QUANTIZE float8_e4m3fn']
WTS_PER_GROUP_SUPPORT_COMBINATION = ['mxfp8_e4m3fn mxfp8_e4m3fn', 'float8_e4m3fn float4_e2m1',
                                'NOT_QUANTIZE int4', 'NOT_QUANTIZE int8',
                                'NOT_QUANTIZE mxfp4_e2m1', 'NOT_QUANTIZE float4_e2m1']

ACT_PER_TENSOR_SUPPORT_COMBINATION = ['hifloat8 hifloat8', 'float8_e4m3fn float8_e4m3fn', 
                                'int8 int8', 'float8_e4m3fn float4_e2m1']
ACT_PER_TOKEN_SUPPORT_COMBINATION = ["int8 int8"]

# Quantization bit width combinations supported by different quantization granularities
WTS_GRANULARITY_SUPPORT_MAP = {
    'tensor': WTS_PER_TENSOR_SUPPORT_COMBINATION,
    'channel': WTS_PER_CHANNEL_SUPPORT_COMBINATION,
    'group': WTS_PER_GROUP_SUPPORT_COMBINATION
}

ACT_GRANULARITY_SUPPORT_MAP = {
    'tensor': ACT_PER_TENSOR_SUPPORT_COMBINATION,
    'token': ACT_PER_TOKEN_SUPPORT_COMBINATION
}

CONVERT_DTYPE_MAP = {
    INT4: [torch.float32, torch.bfloat16, torch.float16],
    INT8: [torch.float32, torch.bfloat16, torch.float16],
    HIFLOAT8: [torch.float32, torch.bfloat16, torch.float16],
    FLOAT8_E4M3FN: [torch.float32, torch.bfloat16, torch.float16],
    MXFP4_E2M1: [torch.bfloat16, torch.float16],
    MXFP8_E4M3FN: [torch.bfloat16, torch.float16],
    FLOAT4_E2M1: [torch.bfloat16, torch.float16],
}
