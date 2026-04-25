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
from ....amct_pytorch.custom_op.utils import calculate_scale_offset
from ....amct_pytorch.custom_op.utils import apply_fake_quantize_and_anti_quantize, apply_true_quantize


def arq_cali_pytorch(data,
                     num_bits,
                     channel_wise,
                     with_offset):
    """
    ARQ calibration to find scale and offset.

    Args:
        data (torch.Tensor): The input tensor to be quantized.
        num_bits (int): The number of bits used for quantization.
        channel_wise (bool): Whether to quantize data on first channel.
        with_offset (bool): Whether to include an offset in the quantization process.

    Returns:
        tuple: A tuple containing:
            - scale (torch.Tensor): The scaling factor used for quantization.
            - offset (torch.Tensor): The offset used for quantization.
            - quantized_input (torch.Tensor): The quantized output tensor.
    """
    ori_shape = data.shape
    # out channel should be transposed to axis 0
    data = data.reshape(ori_shape[0], -1)

    if channel_wise:
        data_max = torch.max(data, dim=1).values
        data_min = torch.min(data, dim=1).values
    else:
        data_max = torch.max(data)
        data_min = torch.min(data)
    data_type = 'INT' + str(num_bits)
    scale, offset = calculate_scale_offset(data_max, data_min, with_offset, data_type)

    data = apply_fake_quantize_and_anti_quantize(data, scale, offset, data_type)
    data = data.reshape(ori_shape)
    return scale.reshape(-1), offset.reshape(-1), data


def arq_real_pytorch(data, 
                     scale, 
                     offset, 
                     num_bits):
    """
    ARQ real to quant data.

    Args:
        data (torch.Tensor): The input tensor to be quantized.
        scale (torch.Tensor): scale in quant factors
        offset (torch.Tensor): offset in quant factors
        data_type (str): destination data type

    Returns:
        quantized_input (torch.Tensor): The quantized output tensor.
    """
    ori_shape = data.shape
    # out channel should be transposed to axis 0
    data = data.reshape(scale.shape[0], -1)
    data_type = 'INT' + str(num_bits)
    data = apply_true_quantize(data, scale, offset, data_type)
    data = data.reshape(ori_shape)
    return data