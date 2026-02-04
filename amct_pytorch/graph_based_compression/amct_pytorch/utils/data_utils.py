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

from ...amct_pytorch.custom_op.cast.amct_pytorch_op_cast import float_to_hifp8, hifp8_to_float, float_to_fp8e4m3fn, \
    fp8e4m3fn_to_float, float_to_fp4e2m1, fp4e2m1_to_float, float_to_fp4e1m2, fp4e1m2_to_float
from ...amct_pytorch.utils.vars import HIFLOAT8, FLOAT8_E4M3FN, MXFP8_E4M3FN, MXFP4_E2M1, FLOAT4_E2M1, FLOAT4_E1M2
from ...amct_pytorch.utils.vars import INT8_MAX, INT8_MIN, INT4_MAX, INT4_MIN


DATA_MAP = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2
}

REVERSE_DATA_MAP = {v: k for k, v in DATA_MAP.items()}

ROUND_MODE_MAP = {
    'ROUND': 0,
    'HYBRID': 1,
    'RINT': 2
}

FP4E2M1_MAX_EXP = 2
FP8E4M3_MAX_EXP = 8


def check_data_type(tensor_dtype, data_types):
    """
    Function: check tensor dtype in data_types
    Args:
        tensor_dtype: torch.dtype
        data_types: list or tuple. torch.dtypes
    """
    if tensor_dtype not in data_types:
        raise RuntimeError('Not support tensor dtype {}, support dtypes {}.'.format(tensor_dtype, data_types))


def check_linear_input_dim(input):
    """
    Function: Check if the input dimension for linear operation is between 2 and 6
    Args:
        input: Input tensor to check dimension
    """
    input_dim = len(input.shape)
    if input_dim < 2 or input_dim > 6:
        raise RuntimeError("Linear quant only support dim from 2 to 6")


@torch.no_grad()
def convert_dtype(ori_tensor, quant_dtype):
    """
    Function: tensor to dst data type. Used only by NPU.
    Args:
        ori_tensor: torch.tensor
        quant_dtype: quant type
    Returns:
        torch.tensor
    """
    device = ori_tensor.device
    if quant_dtype == 'HIFLOAT8':
        # convert require npu cast
        converted_tensor = to_hifloat8(ori_tensor).to(device=device)
    elif quant_dtype == 'FLOAT8_E4M3FN':
        converted_tensor = to_float8(ori_tensor).to(device=device)
    elif quant_dtype == 'INT8':
        converted_tensor = ori_tensor.round().clamp(INT8_MIN, INT8_MAX).to(torch.int8)
    elif quant_dtype == 'INT4':
        converted_tensor = ori_tensor.round().clamp(INT4_MIN, INT4_MAX).to(torch.int32)
    else:
        raise ValueError('Not supported quant_dtype {}'.format(quant_dtype))
    return converted_tensor


def process_tensor(input_tensor):
    """
    Function: npu_quant_matmul's x2 input require two fp4 in one 8-bit,
        now float_to_mxfp4e2m1's output one fp4 in one 8-bit, so need to splice
        and fill zeros in rear half.
    Args:
        input_tensor: torch.tensor
    Returns:
        output_tensor: torch.tensor
    """
    low_4bits = input_tensor & 0x0F
    # flatten to process odd and even element
    flatten_tensor = low_4bits.view(-1)
    even_elements = flatten_tensor[::2]
    odd_elements = flatten_tensor[1::2]
    # combine two fp4 to a new 8-bit
    combined = (odd_elements << 4) | even_elements

    target_shape = list(input_tensor.shape)
    target_shape[1] //= 2
    return combined.reshape(target_shape)


@torch.no_grad()
def convert_mx_dtype(ori_tensor, quant_dtype):
    """
    Function: tensor to dst mx data type. Used only by NPU.
    Args:
        ori_tensor: torch.tensor
        quant_dtype: quant type
    Returns:
        converted_tensor: torch.tensor
        shared_exponent: torch.tensor
    """
    if quant_dtype == MXFP4_E2M1:
        converted_tensor, shared_exponent = float_to_mxfp4e2m1(ori_tensor)
        converted_tensor = process_tensor(converted_tensor)
        shared_exponent = shared_exponent + 6 # npu op require x2_scale mul 2^6
    elif quant_dtype == MXFP8_E4M3FN:
        converted_tensor, shared_exponent = float_to_mxfp8e4m3fn(ori_tensor)
    else:
        raise ValueError('Not supported quant_dtype {}'.format(quant_dtype))
    shared_exponent = torch.clamp(shared_exponent + 127, max=255) # E8M0 bias is 127
    shared_exponent = torch.where(torch.isnan(shared_exponent), 255, shared_exponent)
    return converted_tensor, shared_exponent.to(torch.uint8)


@torch.no_grad()
def convert_precision(ori_tensor, quant_dtype, round_mode):
    """
    Function: convert precision to quant_dtype and back.
    Args:
        ori_tensor: torch.tensor
        quant_dtype: quant type
        round_mode: quant round mode
    Returns:
        torch.tensor
    """
    original_dtype_index = DATA_MAP.get(ori_tensor.dtype)
    if original_dtype_index is None:
        raise RuntimeError(
            "dtype {} not support now, only support float32/float16/bfloat16.".format(ori_tensor.dtype))

    if quant_dtype == HIFLOAT8:
        round_mode_index = ROUND_MODE_MAP.get(round_mode)
        hifp8_data = float_to_hifp8(ori_tensor, round_mode_index)
        converted_data = hifp8_to_float(hifp8_data, original_dtype_index)
    elif quant_dtype in (FLOAT8_E4M3FN, MXFP8_E4M3FN):
        fp8_data = float_to_fp8e4m3fn(ori_tensor)
        converted_data = fp8e4m3fn_to_float(fp8_data, original_dtype_index)
    elif quant_dtype in (FLOAT4_E2M1, MXFP4_E2M1):
        float_data = float_to_fp4e2m1(ori_tensor)
        converted_data = fp4e2m1_to_float(float_data, original_dtype_index)
    elif quant_dtype == FLOAT4_E1M2:
        float_data = float_to_fp4e1m2(ori_tensor)
        converted_data = fp4e1m2_to_float(float_data, original_dtype_index)
    else:
        quant_bits = int(quant_dtype.replace('INT', ''))
        converted_data = torch.clamp(torch.round(
            ori_tensor), -pow(2, quant_bits - 1), pow(2, quant_bits - 1) - 1)

    return converted_data


def to_float8(ori_tensor):
    fp8_data = float_to_fp8e4m3fn(ori_tensor)
    return fp8_data


def to_hifloat8(ori_tensor, round_mode='ROUND'):
    round_mode_index = ROUND_MODE_MAP.get(round_mode)
    hifp8_data = float_to_hifp8(ori_tensor, round_mode_index)
    return hifp8_data


def cal_shared_exponent(input_tensor, mx_dtype, block_size=32):
    """
    Function: cal shared exponent for MXFP4 or MXFP8
    Args:
        input_tensor: weight tensor
        mx_dtype: MXFP8_E4M3FN or MXFP4_E2M1
        block_size: block size of calculate shared exponent
    Returns:
        shared_exponent: torch.tensor
    """
    ori_shape = input_tensor.shape
    reshape_input_tensor = input_tensor.reshape(-1, ori_shape[-1])
    first_dim, _ = reshape_input_tensor.shape
    # fill 0 if the number of data is not divisible by 32
    reshape_input_tensor = pad_zero_by_group(reshape_input_tensor, block_size)
    # reshape to [first_dim, block_size]
    reshaped_tensor = reshape_input_tensor.view(first_dim, -1, block_size)
    max_values = torch.max(torch.abs(reshaped_tensor), dim=-1).values
    zero_mask = (max_values == 0)
    invalid_mask = ~torch.isfinite(max_values)

    non_zero_max_vals = torch.where(zero_mask, torch.ones_like(max_values), max_values)

    exponents = torch.floor(torch.log2(non_zero_max_vals))
    if mx_dtype == MXFP4_E2M1:
        mantissas = non_zero_max_vals / torch.pow(2, exponents)
        # exp add 1 if mantissas larger than 1.75;
        shared_exponents = torch.where(mantissas > 1.75, exponents + 1, exponents) - FP4E2M1_MAX_EXP
    elif mx_dtype == MXFP8_E4M3FN:
        shared_exponents = exponents - FP8E4M3_MAX_EXP
    shared_exponents[zero_mask] = 0
    shared_exponents[invalid_mask] = torch.nan
    shared_exponents = shared_exponents.reshape(ori_shape[:-1] + ((ori_shape[-1] + block_size - 1) // block_size,))
    return shared_exponents


def scale_input_by_shared_exponents(input_tensor, shared_exponents, block_size=32):
    """
    Function: scale input by shared exponents
    Args:
        input_tensor: torch.tensor
        shared_exponents: torch.tensor
        block_size: block size of calculate shared exponent
    Return:
        torch.tensor, shape is same with input_tensor
    """
    n = input_tensor.shape[-1]
    expanded_tensor = torch.repeat_interleave(torch.pow(2, shared_exponents), repeats=block_size, dim=-1)[..., :n]
    result = input_tensor * expanded_tensor
    return result


def float_to_mxfp4e2m1(input_tensor):
    """
    Function: convert FP32/FP16/BF16 to mxfp4
    Args:
        input_tensor: torch.tensor
    Return:
        fp4e2m1_data: torch.uint8
        shared_exponent: mx shared_exponent
    """
    shared_exponent = cal_shared_exponent(input_tensor, MXFP4_E2M1)
    fp4e2m1_data = float_to_fp4e2m1(scale_input_by_shared_exponents(input_tensor, \
                    -1 * shared_exponent.to(input_tensor.dtype)))
    return fp4e2m1_data, shared_exponent


def mxfp4_convert_to_float(input_tensor, shared_exponent, data_index):
    """
    Function: convert mxfp4 to BF16/FP16/FP32
    Args:
        input_tensor: fp4_e2m1 tensor, dtype is torch.uint8
        shared_exponent: mxfp4 shared_exponent
        data_index: torch.dtype index in REVERSE_DATA_MAP
    Return:
        float_data: torch.tensor, dtype is data_type
    """
    data_type = REVERSE_DATA_MAP.get(data_index)
    float_data = fp4e2m1_to_float(input_tensor, data_index)
    float_data = scale_input_by_shared_exponents(float_data, shared_exponent.to(data_type))
    return float_data


def float_to_mxfp8e4m3fn(input_tensor):
    """
    Function: convert FP32/FP16/BF16 to mxfp8
    Args:
        input_tensor: torch.tensor
    Return:
        fp8e4m3fn_data: torch.uint8
        shared_exponent: mx shared_exponent
    """
    shared_exponent = cal_shared_exponent(input_tensor, MXFP8_E4M3FN)
    fp8e4m3fn_data = float_to_fp8e4m3fn(scale_input_by_shared_exponents(input_tensor, 
        -1 * shared_exponent.to(input_tensor.dtype)))
    return fp8e4m3fn_data, shared_exponent


def mxfp8_convert_to_float(input_tensor, shared_exponent, data_index):
    """
    Function: convert mxfp8 to BF16/FP16/FP32
    Args:
        input_tensor: fp8_e4m3fn tensor, dtype is torch.uint8
        shared_exponent: mxfp8 shared_exponent
        data_index: torch.dtype index in REVERSE_DATA_MAP
    Return:
        float_data: torch.tensor, dtype is data_type
    """
    data_type = REVERSE_DATA_MAP.get(data_index)
    float_data = fp8e4m3fn_to_float(input_tensor, data_index)
    float_data = scale_input_by_shared_exponents(float_data, shared_exponent.to(data_type))
    return float_data


def pad_zero_by_group(tensor, group_size):
    """
    Pads the input tensor so that the size of its last dimension is divisible by group_size.
 
    Args:
        tensor (Tensor): Input tensor to be padded, 2 dim
        group_size (int): Group size that the last dimension should be divisible by
 
    Returns:
        Tensor: Padded tensor with last dimension size aligned to group_size
    """
    pad = (group_size - (tensor.shape[-1] % group_size)) % group_size
    tensor_padded = torch.nn.functional.pad(tensor, (0, pad), 'constant', 0)
    return tensor_padded