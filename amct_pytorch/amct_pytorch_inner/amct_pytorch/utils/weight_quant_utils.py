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
import torch
import numpy as np

from ...amct_pytorch.utils.data_utils import convert_precision, float_to_mxfp4e2m1, convert_dtype, \
    scale_input_by_shared_exponents
from ...amct_pytorch.custom_op.utils import convert_to_per_group_shape
from ...amct_pytorch.utils.vars import INT8_MAX, INT8_MIN, MXFP4_E2M1, MXFP8_E4M3FN, INT8, FLOAT8_E4M3FN, HIFLOAT8, \
    FLOAT4_E2M1, FLOAT4_E1M2, INT8, INT4


def quant_dequant_weight(tensor, quant_params=None, scale=None, offset=None):
    """
    do tensor quantize
    Params:
        tensor: quantized tensor from original operator
        scale: scale in quant factors
    Return:
        tensor: quantized tensor
    """
    wts_dtype = quant_params.get('wts_type', INT8)
    round_mode = quant_params.get('round_mode', None)
    group_size = quant_params.get('group_size', None)
    ori_dtype = tensor.dtype

    if isinstance(scale, np.ndarray):
        scale = torch.Tensor(scale).reshape(-1, 1).to(tensor.device)
    if wts_dtype in [HIFLOAT8, FLOAT8_E4M3FN]:
        scale = scale.reshape(-1, 1)
        tensor = convert_precision(tensor / scale, wts_dtype, round_mode)
        tensor = (tensor * scale).to(ori_dtype)
    elif wts_dtype in (MXFP4_E2M1, MXFP8_E4M3FN):
        tensor = scale_input_by_shared_exponents(tensor, -1 * scale.to(tensor.dtype))
        tensor = convert_precision(tensor, wts_dtype, 'RINT')
        tensor = scale_input_by_shared_exponents(tensor, scale.to(ori_dtype))
    elif wts_dtype in (FLOAT4_E2M1, FLOAT4_E1M2):
        ori_shape = tensor.shape
        if group_size:
            tensor = convert_to_per_group_shape(tensor, group_size)
        scale = scale.to(tensor.device)
        tensor = convert_precision(tensor / scale, wts_dtype, 'RINT')
        tensor = (tensor * scale).reshape(ori_shape[0], -1)[:, :ori_shape[1]].to(ori_dtype)
    elif wts_dtype in (INT4, INT8):
        ori_shape = tensor.shape
        if group_size:
            tensor = convert_to_per_group_shape(tensor, group_size)
        tensor = tensor / scale
        quant_bits = int(wts_dtype.replace('INT', ''))
        if offset is not None:
            tensor = tensor + offset
        tensor = torch.clamp(torch.round(
            tensor), -pow(2, quant_bits - 1), pow(2, quant_bits - 1) - 1)
        if offset is not None:
            tensor = tensor - offset
        tensor = (tensor) * scale
        tensor = tensor.reshape(ori_shape[0], -1)[:, :ori_shape[1]].to(ori_dtype)
    return tensor


def quant_weight(tensor, wts_type, scale, offset=None, group_size=None):
    if group_size:
        tensor = convert_to_per_group_shape(tensor, group_size)
    tensor = tensor / scale
    if offset is not None:
        tensor = tensor + offset
    return convert_dtype(tensor, wts_type)

