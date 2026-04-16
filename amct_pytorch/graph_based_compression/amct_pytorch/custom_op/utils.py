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
import math
import torch

from ...amct_pytorch.utils.vars import FLT_EPSILON
from ...amct_pytorch.utils.weight_quant_api import adjust_deconv_weight_shape
from ...amct_pytorch.utils.data_utils import convert_precision, pad_zero_by_group
from ...amct_pytorch.parser.module_based_record_parser import get_layer_quant_params


def check_quant_data(data, data_type):
    """
    Function: check quant data for ops
    Inputs:
        data: torch.Tensor, data to ops
        data_type: a string, type of data, including: "weight", "input"
    """
    if data.dtype not in [torch.float32, torch.float16]:
        raise TypeError(
            "Only {} with dtype 'torch.float32' and 'torch.float16' is supported to be " \
            "quantized, but got {}.".format(data_type, data.dtype))
    if not torch.all(torch.isfinite(data)):
        raise RuntimeError(
            "The {} to be quantized has invalid value inf or nan!"
            .format(data_type))


def check_scale_offset(scale, offset):
    """
    Function: check scale and offset for ops
    Inputs:
        scale: torch.Tensor
        offset: torch.Tensor
    """
    valid = True
    if not torch.all(torch.isfinite(scale)):
        valid = False
    if not torch.all(torch.isfinite(1 / scale)):
        valid = False
    if not valid:
        raise RuntimeError(
            "Param scale has invalid value inf, nan or zeros!")

    if not torch.all(torch.isfinite(offset)):
        raise RuntimeError(
            "Param offset has invalid value inf or nan!")


def check_tensor_balance_factor(tensor_balance_factor):
    """
    Function: check tensor_balance_factor for ops
    Inputs:
        tensor_balance_factor: torch.Tensor
    """
    valid = True
    if not torch.all(torch.gt(tensor_balance_factor, FLT_EPSILON)):
        valid = False
    if not torch.all(torch.lt(tensor_balance_factor, 1 / FLT_EPSILON)):
        valid = False

    if not valid:
        raise RuntimeError(
            "The tensor balance factor is less than FLT_EPSILON or larger than 1/FLT_EPSILON!")


def copy_tensor(target, source):
    """
    Function: copy tensor data
    Inputs:
        target: torch.Tensor, taget tensor
        source: torch.Tensor, source tensor
    """
    if isinstance(target, torch.Tensor):
        target.data.copy_(source.data)


def tensor(value, dtype=torch.float, requires_grad=False, device=None):
    """
    Function: check scale and offset for ops
    Inputs:
        value: torch.Tensor
        dtype: tensor type
        requires_grad: tensor grad
        device: tensor device
    Outputs:
        output: torch.Tensor
    """
    if isinstance(value, torch.Tensor):
        if device is not None:
            return value.to(device).clone().detach().requires_grad_(requires_grad)
        return value.clone().detach().requires_grad_(requires_grad)
    else:
        if device is not None:
            return torch.tensor(value, dtype=dtype,
                                requires_grad=requires_grad, device=device)
        return torch.tensor(value, dtype=dtype,
                            requires_grad=requires_grad)


def process_scale(scale, offset, with_offset=True, numbit=8):
    """
    Function: check the validity of quant factor(scale)'s range
    Inputs:
        scale: torch.Tensor, quant factor
        offset: torch.Tensor, quant factor
        with_offset: bool, with offset or not
        numbit: int, quant bits
    Outputs:
        scale: torch.Tensor, quant factor
        offset: torch.Tensor, quant factor
    """
    base_bit = 2.0
    if with_offset:
        offset = torch.where(
            scale < FLT_EPSILON,
            tensor(-pow(base_bit, numbit - 1), device=offset.device), offset)
        scale = torch.where(
            scale < FLT_EPSILON,
            tensor(1.0, device=scale.device), scale)
    else:
        scale = torch.where(
            scale < FLT_EPSILON,
            tensor(1.0, device=scale.device), scale)
    check_scale_offset(scale, offset)
    return scale, offset


def process_tensor_shape(input_tensor, module_type, module):
    """
    Function: adjust tensor shape according to module_type
    Inputs:
    data: torch.Tensor, data to ops
    module_type: a string, type of the nn.module
    """
    # ConvTranspose wts is in cin,cout,h,w need to be adjust.
    if module_type in ['ConvTranspose1d', 'ConvTranspose2d']:
        group = module.groups
        processed_tensor = adjust_deconv_weight_shape(group, input_tensor)
    else:
        processed_tensor = input_tensor
    return processed_tensor


def get_distribute_config():
    process_group = torch.distributed.group.WORLD
    try:
        world_size = torch.distributed.get_world_size(process_group)
    except (AttributeError, AssertionError, RuntimeError, ValueError):
        process_group = None
        world_size = 1
    need_sync = world_size > 1
    config = dict()
    config['need_sync'] = need_sync
    config['process_group'] = process_group
    config['world_size'] = world_size
    return config


def check_group_param(inputs, channel_wise_flag, group, axis):
    """
    Function: group wise conflicted with channel wise and axis smaller than input dim
    Inputs:
    inputs: torch.Tensor, data to ops
    channel_wise_flag: bool, whether do channel wise
    group: int, group num fot wts quant
    axis: int, out channel axis for weight
    """
    if channel_wise_flag:
        if group != 1 or axis != 0:
            raise RuntimeError(
                "channel wise quant do not support group and axis setting")
        return False

    if group > 1:
        if inputs.dim() <= axis:
            raise RuntimeError(
                "group wise quant do not support axis setting larger than weight dim")
        return True
    return False


def convert_to_per_group_shape(input_tensor, group_size):
    """
    Converts the input 2D tensor into a shape grouped by the specified group size. 
    If the total number of elements isn't divisible by the group size, zero-padding is applied.
 
    Parameters:
        input_tensor (Tensor): Input 2D tensor to be processed
        group_size (int): Target size for each group
 
    Returns:
        Tensor: Reshaped tensor with grouped dimensions
    """
    input_tensor_pad = pad_zero_by_group(input_tensor, group_size)

    return input_tensor_pad.reshape(input_tensor_pad.shape[0], input_tensor_pad.shape[1] // group_size, group_size)


def get_int_quant_scope(data_type, asymmetric, abs_max_is_negative=False):
    """
    Get the quantization step for integer data types.

    Args:
        data_type: Type of integer data (e.g., INT8, INT4)
        asymmetric: True if using asymmetric quantization, False for symmetric
        abs_max_is_negative: The absolute value of the maximum value of the original data is negative

    Returns:
        quantization step
    """
    quant_bits = int(data_type.replace('INT', ''))
    if asymmetric:
        return pow(2, quant_bits) - 1

    # Symmetric quantization, asymmetric quantization range (-128, 127), use 127 for positive values and
    # 128 for negative values when the absolute value of the original data is the maximum. 
    return abs_max_is_negative + pow(2, quant_bits - 1) - 1


def calculate_scale_offset(data_max, data_min, asymmetric, data_type):
    """
    Function: calculate scale and offset using min-max algorithms
    Parameters: data_max: max value in calibration data
                data_min: min value in calibration data
                asymmetric: do asymmetric quant
                data_type: Type of integer data (e.g., INT8, INT4)
    Return: scale: scale in quant factors
            offset: offset in quant factors
    """
    data_max = data_max.to(torch.float32)
    data_min = data_min.to(torch.float32)
    # asymmetric quant
    if asymmetric:
        # clamp lower bound of data_max to 0
        data_max = torch.clamp(data_max, 0, None)
        # clamp upper bound of data_max to 0
        data_min = torch.clamp(data_min, None, 0)
        quant_scope = get_int_quant_scope(data_type, asymmetric)

        scale = (data_max - data_min) / quant_scope
        offset = torch.where(data_max != data_min,
            -math.ceil(quant_scope / 2) - data_min / scale,
            torch.tensor(-math.ceil(quant_scope / 2), dtype=data_max.dtype, device=data_max.device))
    # symmetric quant
    else:
        abs_max_is_negative = data_max.abs() < data_min.abs()
        quant_scope = get_int_quant_scope(data_type, asymmetric, abs_max_is_negative)
        boundary = torch.where(abs(data_max) > abs(data_min), abs(data_max), abs(data_min))
        scale = boundary / quant_scope
        offset = torch.zeros_like(scale)
    int_quant_bits = int(data_type.replace('INT', '')) if 'INT' in data_type else None
    scale, offset = process_scale(scale, offset, asymmetric, int_quant_bits)
    scale = scale.to(torch.float32)
    offset = offset.round().to(torch.int32)
    return scale, offset


def apply_fake_quantize(data, scale, offset, data_type):
    """
    Do fake quantize
 
    Parameters:
    data: The input tensor
    scale: scale in quant factors
    offset: offset in quant factors
    data_type: destination data type

    Returns:
    A tensor of fake quantization
    """
    # do scale offset broadcast
    if scale.dim() >= 1 and len(scale) > 1:
        ori_shape = data.shape
        shape = [1, ] * len(data.shape)
        shape[0] = ori_shape[0]
        
        scale = scale.reshape(shape)
        offset = offset.reshape(shape)
    # do fake quantize
    data = data / scale + offset
    data = convert_precision(data, data_type, None)
    return data, scale, offset


def apply_fake_quantize_and_anti_quantize(data, scale, offset, data_type):
    """
    Do fake quantize and anti quantize to introduce quant error

    Parameters:
    data: The input tensor
    scale: scale in quant factors
    offset: offset in quant factors
    data_type: destination data type

    Returns:
    A tensor for which the quantize error is introduced
    """
    # do fake quantize
    data, scale, offset = apply_fake_quantize(data, scale, offset, data_type)
    # do anti quantize
    data = (data - offset) * scale
    return data


def apply_true_quantize(data, scale, offset, data_type):
    """
    Do true quantize

    Parameters:
    data: The input tensor
    scale: scale in quant factors
    offset: offset in quant factors
    data_type: destination data type

    Returns:
    A tensor of true quantization
    """
    # do fake quantize
    data, scale, offset = apply_fake_quantize(data, scale, offset, data_type)
    # do true quantize
    data = data.to(torch.int8)
    return data


@torch.no_grad()
def calculate_scale_by_group_size(input_tensor, wts_type, group_size, is_padded=False, asymmetric=False):
    """
    Calculates and returns the scale factor for each group based on the input tensor, weight type, and group size.

    Parameters:
    input_tensor: The input tensor
    wts_type: The weight type (e.g., quantization type)
    group_size: The size of each group for per-group processing
    is_padded: Indicates whether the input_tensor has been padded.

    Returns:
    A input_tensor of shape (group_num, 1) containing the scale factor for each group.
    """
    if not is_padded:
        input_tensor = convert_to_per_group_shape(input_tensor, group_size)
    weight_max = input_tensor.max(dim=-1, keepdim=True).values  # [group_num, 1]
    weight_min = input_tensor.min(dim=-1, keepdim=True).values  # [group_num, 1]
    scale, offset = calculate_scale_offset(weight_max, weight_min, asymmetric, wts_type)
    return scale.reshape(weight_max.shape), offset.reshape(weight_max.shape)
