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


DATA_MAP = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2
}


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

    quant_bits = int(quant_dtype.replace('INT', ''))
    converted_data = torch.clamp(torch.round(
        ori_tensor), -pow(2, quant_bits - 1), pow(2, quant_bits - 1) - 1)

    return converted_data



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