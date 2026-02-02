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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import platform
import sys

import torch

from ...amct_pytorch.capacity import CAPACITY
from ...amct_pytorch.common.utils.vars_util import INT4, INT8, INT16
from ...amct_pytorch.utils.log import LOGGER

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

QUANTIZABLE_TYPES = CAPACITY.get_value('QUANTIZABLE_TYPES')
QUANTIZABLE_ONNX_TYPES = CAPACITY.get_value('QUANTIZABLE_ONNX_TYPES')
INT16_QUANTIZABLE_TYPES = CAPACITY.get_value('INT16_QUANTIZABLE_TYPES')
INT16_QUANTIZABLE_ONNX_TYPES = CAPACITY.get_value('INT16_QUANTIZABLE_ONNX_TYPES')
CHANNEL_WISE_TYPES = CAPACITY.get_value('CHANNEL_WISE_TYPES')
CHANNEL_WISE_ONNX_TYPES = CAPACITY.get_value('CHANNEL_WISE_ONNX_TYPES')

FUSE_TYPES = CAPACITY.get_value('FUSE_TYPES')
FUSE_ONNX_TYPES = CAPACITY.get_value('FUSE_ONNX_TYPES')

AMCT_OPERATIONS = CAPACITY.get_value('AMCT_OPERATIONS')
AMCT_RETRAIN_OPERATIONS = CAPACITY.get_value('AMCT_RETRAIN_OPERATIONS')
AMCT_DISTILL_OPERATIONS = CAPACITY.get_value('AMCT_DISTILL_OPERATIONS')

RETRAIN_TYPES = CAPACITY.get_value('RETRAIN_TYPES')
RETRAIN_ONNX_TYPES = CAPACITY.get_value('RETRAIN_ONNX_TYPES')

DISTILL_TYPES = CAPACITY.get_value('DISTILL_TYPES')

MULT_OUTPUT_TYPES = CAPACITY.get_value('MULT_OUTPUT_TYPES')

PRUNABLE_TYPES = CAPACITY.get_value('PRUNABLE_TYPES')
PRUNABLE_ONNX_TYPES = CAPACITY.get_value('PRUNABLE_ONNX_TYPES')
PASSIVE_PRUNABLE_TYPES = CAPACITY.get_value('PASSIVE_PRUNABLE_TYPES')
PASSIVE_PRUNABLE_ONNX_TYPES = CAPACITY.get_value('PASSIVE_PRUNABLE_ONNX_TYPES')
SELECTIVE_PRUNABLE_TYPES = CAPACITY.get_value('SELECTIVE_PRUNABLE_TYPES')
CHANNEL_UNRELATED_ONNX_TYPES_P1 = CAPACITY.get_value('CHANNEL_UNRELATED_ONNX_TYPES_P1')
CHANNEL_UNRELATED_ONNX_TYPES_P2 = CAPACITY.get_value('CHANNEL_UNRELATED_ONNX_TYPES_P2')
ELTWISE_ONNX_TYPES = CAPACITY.get_value('ELTWISE_ONNX_TYPES')
CHANNEL_UNRELATED_ONNX_TYPES = CHANNEL_UNRELATED_ONNX_TYPES_P1 + CHANNEL_UNRELATED_ONNX_TYPES_P2

KV_CACHE_QUANTIZE_TYPES = CAPACITY.get_value('KV_CACHE_QUANTIZE_TYPES')
ADA_ROUND_TYPES = CAPACITY.get_value('ADA_ROUND_TYPES')

MATMUL_DEQUANT_AFTER_ADD = CAPACITY.get_value('MATMUL_DEQUANT_AFTER_ADD')

CLIBRATION_BIT = 8
QUANT_BIAS_BITS = 32
ZERO = 0.0
ONE = 1.0
BASE = 2
EPSILON = 1E-6
FLT_EPSILON = 1.192092896e-7
MIN_FP16 = 2**-14
MAX_FP16 = 65504.0

# act quant terms
ACTIVATION_QUANT_PARAMS = 'activation_quant_params'
ACT_ALGO = 'act_algo'
IFMR = 'ifmr'
HFMG = 'hfmg'

## act quant common params
NUM_BITS = 'num_bits'
BATCH_NUM = 'batch_num'
ASYMMETRIC = 'asymmetric'
ACTIVATION_OFFSET = 'activation_offset'
DST_TYPE = 'dst_type'
QUANT_GRANULARITY = 'quant_granularity'
PER_TENSOR = 'per_tensor'
PER_CHANNEL = 'per_channel'
WITH_OFFSET = 'with_offset'
CHANNEL_WISE = 'channel_wise'

## IFMR quant algo params
MAX_PERCENTILE = 'max_percentile'
MIN_PERCENTILE = 'min_percentile'
SEARCH_STEP = 'search_step'
SEARCH_RANGE = 'search_range'
SEARCH_RANGE_START = 'search_range_start'
SEARCH_RANGE_END = 'search_range_end'

CLIP_MAX = 'clip_max'
CLIP_MIN = 'clip_min'
FIXED_MIN = 'fixed_min'
H_FIXED_MIN = 'h_fixed_min'

## HFMG quant algo params
HFMG_POW = 3
MIN_BIN_RATIO = 4
STEP_DIVISOR = 100000
NUM_OF_BINS = 'num_of_bins'

TENSOR_BALANCE_FACTOR = 'tensor_balance_factor'
NUM_BITS_MAP = {
    INT4: 4,
    INT8: 8,
    INT16: 16,
}

QUANTIZE_LINEAR = 'QuantizeLinear'
DEQUANTIZE_LINEAR = 'DequantizeLinear'
TRANSPOSE = 'Transpose'

QUANT_LAYER_SUFFIX = ('.quant', '.dequant', '.anti_quant')

HIFLOAT8 = 'HIFLOAT8'
FLOAT8_E4M3FN = 'FLOAT8_E4M3FN'
FLOAT4_E2M1 = 'FLOAT4_E2M1'
FLOAT4_E1M2 = 'FLOAT4_E1M2'
FLOAT16 = 'FLOAT16'
INT8 = 'INT8'
INT4 = 'INT4'
MXFP4_E2M1 = 'MXFP4_E2M1'
MXFP8_E4M3FN = 'MXFP8_E4M3FN'

# AWQ algorithm parameters
GROUP_SIZE = 32
GRIDS_NUM = "grids_num"
DEFAULT_GRIDS_NUM = 20

LUT_DEFAULT_GROUP_SIZE = 256
LUT_ALGO = 'lut_algo'

QUANT_RESULT_FILE = 'quant_result.pt'

DEFAULT_SMOOTH_STRENGTH = 0.5
WTS_ASYMMETRIC_DTYPE = [INT8, INT4]

WTS_PER_TENSOR_SUPPORT_COMBINATION = ['HIFLOAT8 HIFLOAT8', 'FLOAT8_E4M3FN FLOAT8_E4M3FN', 
                                  'INT8 INT8', 'INT8 INT4', 'NOT_QUANTIZE INT8', 'NOT_QUANTIZE INT4',
                                  'NOT_QUANTIZE HIFLOAT8', 'NOT_QUANTIZE FLOAT8_E4M3FN']
WTS_PER_CHANNEL_SUPPORT_COMBINATION = ['HIFLOAT8 HIFLOAT8', 'FLOAT8_E4M3FN FLOAT8_E4M3FN', 
                                  'INT8 INT8', 'INT8 INT4', 'NOT_QUANTIZE INT8', 'NOT_QUANTIZE INT4',
                                  'NOT_QUANTIZE HIFLOAT8', 'NOT_QUANTIZE FLOAT8_E4M3FN']
WTS_PER_GROUP_SUPPORT_COMBINATION = ['NOT_QUANTIZE MXFP4_E2M1', 'MXFP8_E4M3FN MXFP4_E2M1', 
                                 'MXFP8_E4M3FN MXFP8_E4M3FN', 'NOT_QUANTIZE INT4',
                                 'NOT_QUANTIZE FLOAT4_E1M2', 'NOT_QUANTIZE FLOAT4_E2M1',
                                 'FLOAT8_E4M3FN FLOAT4_E2M1', 'NOT_QUANTIZE INT8']

ACT_PER_TENSOR_SUPPORT_COMBINATION = ['HIFLOAT8 HIFLOAT8', 'FLOAT8_E4M3FN FLOAT8_E4M3FN', 
                                  'INT8 INT8', 'INT8 INT4', 'FLOAT8_E4M3FN FLOAT4_E2M1']
ACT_PER_TOKEN_SUPPORT_COMBINATION = ["INT8 INT8"]

# Quantization bit width combinations supported by different quantization granularities
WTS_GRANULARITY_SUPPORT_MAP = {
    'PER_TENSOR': WTS_PER_TENSOR_SUPPORT_COMBINATION,
    'PER_CHANNEL': WTS_PER_CHANNEL_SUPPORT_COMBINATION,
    'PER_GROUP': WTS_PER_GROUP_SUPPORT_COMBINATION
}

ACT_GRANULARITY_SUPPORT_MAP = {
    'PER_TENSOR': ACT_PER_TENSOR_SUPPORT_COMBINATION,
    'PER_TOKEN': ACT_PER_TOKEN_SUPPORT_COMBINATION
}

SMOOTH_QUANT_SUPPORT_COMBINATION = ['INT8 INT8', 'FLOAT8_E4M3FN FLOAT4_E2M1']

# Model weight data types supported by quantization bit width combinations
ALLOWED_WEIGHT_DTYPES = {
    'HIFLOAT8 HIFLOAT8': [torch.float32, torch.float16, torch.bfloat16],
    'FLOAT8_E4M3FN FLOAT8_E4M3FN': [torch.float32, torch.float16, torch.bfloat16],
    'NOT_QUANTIZE MXFP4_E2M1': [torch.bfloat16, torch.float16],
    'NOT_QUANTIZE FLOAT8_E4M3FN': [torch.bfloat16, torch.float16],
    'NOT_QUANTIZE HIFLOAT8': [torch.bfloat16, torch.float16],
    'MXFP8_E4M3FN MXFP4_E2M1': [torch.bfloat16],
    'MXFP8_E4M3FN MXFP8_E4M3FN': [torch.bfloat16],
    'NOT_QUANTIZE INT4': [torch.float32, torch.bfloat16, torch.float16],
    'NOT_QUANTIZE INT8': [torch.bfloat16, torch.float16],
    'NOT_QUANTIZE FLOAT4_E1M2': [torch.bfloat16, torch.float16], 
    'NOT_QUANTIZE FLOAT4_E2M1': [torch.bfloat16, torch.float16],
    'FLOAT8_E4M3FN FLOAT4_E2M1': [torch.bfloat16],
}

ALLOWED_ROUND_MODE_MAP = {
    HIFLOAT8: ['ROUND', 'HYBRID'],
    FLOAT8_E4M3FN: ['RINT'],
    INT8: ['RINT'],
    INT4: ['RINT'],
    MXFP4_E2M1: ['RINT'],
    MXFP8_E4M3FN: ['RINT'],
    FLOAT4_E1M2: ['RINT'],
    FLOAT4_E2M1: ['RINT']
}

INT8_MAX = torch.iinfo(torch.int8).max
INT8_MIN = torch.iinfo(torch.int8).min

INT4_MAX = 7
INT4_MIN = -8

INT32_MAX = torch.iinfo(torch.int32).max
INT32_MIN = torch.iinfo(torch.int32).min


LUT_DEFAULT_GROUP_SIZE = 256


def find_torch_version():
    """
    Function: find torch's valid version.
    """
    version = torch.__version__
    for support_version in SUPPORT_TORCH_VERSIONS:
        if version.startswith(support_version):
            return support_version
    LOGGER.logw("amct_pytorch cannot support torch %s" % (version))
    return version


def find_platform():
    """
    Function: find hardware platform for load op.
    """
    support_platforms = ['x86_64', 'aarch64']
    hardware_platform = platform.platform()
    for arch in support_platforms:
        if arch in hardware_platform:
            return arch
    raise RuntimeError("Unsupport platform {}".format(hardware_platform))


def find_op_py():
    """
    Function: obtain python version for loading op.
    """
    python_version = sys.version_info
    op_py = '{}{}'.format(python_version.major, python_version.minor)
    if op_py in ['37']:
        op_py = '{}m'.format(op_py)
    return op_py

SUPPORT_TORCH_VERSIONS = ['1.5.0', '1.8.0', '1.10.0', '2.1.0', '2.7.1']
TORCH_VERSION = find_torch_version()
PLATFORM = find_platform()
OP_PY = find_op_py()
