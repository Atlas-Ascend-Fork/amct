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
import os
import torch

from ...amct_pytorch.utils.vars import FLT_EPSILON
from ...amct_pytorch.utils.vars import BASE
from ...amct_pytorch.utils.weight_quant_api import adjust_deconv_weight_shape
from ...amct_pytorch.utils.vars import FLOAT4_E2M1, FLOAT4_E1M2, FLOAT8_E4M3FN, HIFLOAT8
from ...amct_pytorch.utils.weight_quant_api import apply_lut_quantize_weight
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


def convert_to_dst_shape(input_tensor, dst_shape):
    """
    Converts a flattened tensor back to its destination 2D shape by truncating excess elements.
 
    Parameters:
        input_tensor (Tensor): Input tensor to be reshaped (can be padded or extended)
        dst_shape (tuple): destination 2D shape (rows, columns) to restore
 
    Returns:
        Tensor: Reshaped tensor matching the destination dimensions
    """
    if input_tensor.shape == dst_shape:
        return input_tensor

    out_tensor = input_tensor.reshape(input_tensor.shape[0], \
        input_tensor.shape[1] * input_tensor.shape[2])[:dst_shape[0], :dst_shape[1]]
    return out_tensor


def get_float_quant_scope(data_type):
    """
    Get the quantization step for float data types.

    Args:
        data_type: Type of integer data (e.g., FLOAT4E2M1, FLOAT4E1M2)

    Returns:
        quantization step
    """
    quant_max_scope = {
        FLOAT4_E2M1: 6.0,
        FLOAT4_E1M2: 1.75,
        FLOAT8_E4M3FN: 448.0, # 256 * 1.75
        HIFLOAT8: 32768.0, # 2^15
    }

    return quant_max_scope[data_type]


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


def get_weight_min_max_by_granularity(weight_data, quant_config):
    """
    Calculate the minimum and maximum values of weights based on the module type and weight data.
    The calculation method depends on the granularity setting in the quantization configuration.

    Parameters:
    weight_data (Tensor): Weight data
    quant_config (dict): Quantization configuration

    Returns:
    tuple: A tuple containing the minimum and maximum values of the weights
    """
    if quant_config.get('weight_granularity') == 'PER_CHANNEL':
        weight_max = weight_data.max(dim=1, keepdim=True).values
        weight_min = weight_data.min(dim=1, keepdim=True).values
    elif quant_config.get('weight_granularity') == 'PER_TENSOR':
        weight_max = weight_data.max().reshape(1, 1)
        weight_min = weight_data.min().reshape(1, 1)
    else:
        # weight: [n, k] -> [cout, cin//group_size, group_size]
        group_size = quant_config.get("group_size", 128)
        weight_all = convert_to_per_group_shape(weight_data, group_size)
        weight_max = weight_all.max(dim=-1, keepdim=True).values  # [cout, cin//group_size, 1]
        weight_min = weight_all.min(dim=-1, keepdim=True).values  # [cout, cin//group_size, 1]
 
    return weight_min, weight_max


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
        if data_type in (FLOAT8_E4M3FN, FLOAT4_E2M1, FLOAT4_E1M2, HIFLOAT8):
            quant_scope = get_float_quant_scope(data_type)
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


def get_optimized_weight(records, layer_name):
    layer_quant_param = get_layer_quant_params(records, layer_name)
    optimized_weight = layer_quant_param.get('optimized_weight')
    return optimized_weight


def get_quant_factor(records, layer_name):
    layer_quant_param = get_layer_quant_params(records, layer_name)
    quant_factors = layer_quant_param.get('quant_factors')
    return quant_factors


def get_algo_params(records, layer_name, algo):
    layer_quant_param = get_layer_quant_params(records, layer_name)
    algo_params = layer_quant_param.get(algo)
    return algo_params


def save_algo_params(algo, algo_param, algo_param_dict):
    if algo == 'smooth_quantize':
        algo_param_dict['smooth_factor'] = algo_param.get('smooth_factor')
    if algo == 'awq_quantize':
        algo_param_dict['awq_scale'] = algo_param.get('awq_scale')


def cal_deq_scale(scale_w, scale_d, module_type):
    if module_type == 'Linear':
        deq_scale = scale_w.reshape(-1) * scale_d
        scale_w = scale_w.reshape(-1, 1)
    if module_type == 'Conv2d':
        scale_w = scale_w.reshape(-1, 1, 1, 1)
        deq_scale = (scale_w * scale_d).reshape(1, -1, 1, 1)
    if module_type == 'ConvTranspose2d':
        scale_w = scale_w.reshape(1, -1, 1, 1)
        deq_scale = (scale_w * scale_d).reshape(1, -1, 1, 1)
    return scale_w, deq_scale


def apply_smooth_weight(smooth_params, ori_weight):
    if smooth_params.get('smooth_factor') is None:
        raise RuntimeError("smooth_factor is None!")
    smooth_factor = smooth_params.get('smooth_factor').to(device=ori_weight.device, dtype=ori_weight.dtype)
    ori_weight_shape = ori_weight.shape
    if list(smooth_factor.shape) != [1, ori_weight_shape[1]]:
        raise RuntimeError("smooth_factor shape should be {} current shape is {}"
                        .format([1, ori_weight_shape[1]], list(smooth_factor.shape)))
    weight = ori_weight * smooth_factor.to(device=ori_weight.device)
    return weight


def apply_awq_quantize_weight(weight_tensor, awq_param, group_size):
    if awq_param.get('scale') is None:
        raise RuntimeError("AWQ params scale is None!")
    scale = awq_param.get('scale').to(weight_tensor.device)
    cin = weight_tensor.shape[1]
    cout = weight_tensor.shape[0]
    if list(scale.shape) != [1, cin]:
        raise RuntimeError("AWQ params scale.shape should be [1, {}] current shape is {}".format(
            cin, list(scale.shape)))

    weight_tensor = weight_tensor / scale
    if awq_param.get('clip_max') is None:
        return weight_tensor

    clip_max = awq_param.get('clip_max')
    clip_max = clip_max.to(weight_tensor.device)
    if group_size is not None:
        if list(clip_max.shape) != [cout, (cin + group_size - 1) // group_size, 1]:
            raise RuntimeError("AWQ params clip_max.shape should be [{}, {}, 1] current shape is {}" .format(
                cout, (cin + group_size - 1) // group_size, list(clip_max.shape)))

        # clip_max reshape to [cout, num_groups]
        clip_max = clip_max.squeeze(-1)
        # repeat to [cout, num_groups * group_size], clip to [cout, cin]
        clip_max = clip_max.repeat_interleave(group_size, -1)[:, :cin]

    weight_tensor = torch.clamp(weight_tensor, -1 * clip_max, clip_max)
    return weight_tensor


def apply_quantize_by_algo(weight, group_size, algo, algo_param):
    if algo == 'smooth_quantize':
        quantized_weight = apply_smooth_weight(algo_param, weight)
    elif algo == 'awq_quantize':
        quantized_weight = apply_awq_quantize_weight(weight, algo_param, group_size)
    elif algo == 'lut_quantize':
        quantized_weight = apply_lut_quantize_weight(weight, algo_param, group_size)
    return quantized_weight


def apply_progressive_quant(weight, quant_factors, round_mode, group_size):
    scale_w1 = quant_factors.get('scale_w1').to(device=weight.device)
    scale_w2 = quant_factors.get('scale_w2').to(device=weight.device)
    scale_d = quant_factors.get('scale_d').to(device=weight.device)
    if list(scale_w1.shape) != [weight.shape[0], 1]:
        raise RuntimeError("scale_w1.shape should be [{}, 1] current shape is {}"
                           .format(weight.shape[0], list(scale_w1.shape)))
    group = int(weight.shape[0] * weight.shape[1] / group_size)
    if list(scale_w2.shape) != [group, 1]:
        raise RuntimeError("scale_w2.shape should be [{}, 1] current shape is {}".format(group, list(scale_w2.shape)))
    
    quantized_weight_fp8 = \
        convert_precision(weight / scale_w1, "FLOAT8_E4M3FN", round_mode)
    quantized_weight_fp4 = \
        convert_precision((quantized_weight_fp8.reshape(-1, group_size) / scale_w2), "FLOAT4_E2M1", round_mode)
    quantized_weight = \
        (convert_precision(quantized_weight_fp4, "FLOAT8_E4M3FN", round_mode) * scale_w2).reshape(weight.shape)
    deq_scale = scale_w1 * scale_d
    return quantized_weight, deq_scale.transpose(0, 1)


def calculate_progressive_weights_scale_factor(weight, group_size=32):
    """
    Function: calculate two level weights's quant factor and do fakequant
    Parameters: 
    quant_config: configuration of quantization
    group_size: scale w2 per group size
    """
    # weight per-channel to fp8_e4m3fn, per-group to fp4_e2m1
    scale_w1, _ = calculate_scale_offset(weight.max(dim=-1).values, weight.min(dim=-1).values,
                                            False, FLOAT8_E4M3FN)
    # scale_w1 [n] -> [n,1]
    scale_w1 = scale_w1.unsqueeze(1)
    weight_fp8e4m3fn = convert_precision(weight / scale_w1, FLOAT8_E4M3FN, 'RINT')
    weight_fp8e4m3fn = weight_fp8e4m3fn.reshape(-1, group_size)

    scale_w2, _ = calculate_scale_offset(weight_fp8e4m3fn.max(dim=-1).values, 
                                            weight_fp8e4m3fn.min(dim=-1).values, False, FLOAT4_E2M1)
    # scale_w1 [g] -> [g,1]
    scale_w2 = scale_w2.unsqueeze(1)

    return scale_w1, scale_w2


def check_module_device(model, quantize_layers):
    """
    Function: check not support device to do quantize, such as meta device
    Parameters: 
    model: pytorch model
    quantize_layers: list, layer name to do quantize
    """
    for name, mod in model.named_modules():
        if name in quantize_layers and mod.weight.device == torch.device('meta'):
            raise RuntimeError(
                "quantifiable module not support meta device, please check {} layer's device in model".format(name))


def check_scale_offset_shape(weight, scale_w, offset_w=None, group_size=None):
    if group_size is None:
        if scale_w.shape[0] == 1:
            return
        if scale_w.shape[0] == weight.shape[0]:
            return
        else:
            raise RuntimeError("scale.shape should be equal to 1 or cout. current is {}, "
                               "pls check quant factors from file".format(scale_w.shape[0]))
    if list(scale_w.shape) != [weight.shape[0], math.ceil(weight.shape[1] / group_size), 1]:
        raise RuntimeError(
            "scale.shape should be [{}] current shape is {}, please check quant factors from file.".format(
            (weight.shape[0], math.ceil(weight.shape[1] / group_size), 1), list(scale_w.shape)))

    if offset_w is None:
        return
    if scale_w.shape != offset_w.shape:
        raise RuntimeError("offset_w.shape should be equal to scale_w.shape, pls check quant factors from file")
