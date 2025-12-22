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
import math
import torch

from amct_pytorch.utils.vars import INT8_MAX, INT8_MIN, INT4_MIN, INT4_MAX, INT8, INT4


def pad_zero_by_group(tensor, group_size):
    """
    Function: Pads the input tensor so that the size of its last dimension is divisible by group_size.
 
    Parameters:
        tensor (Tensor): Input tensor to be padded, 2 dim
        group_size (int): Group size that the last dimension should be divisible by
 
    Returns:
        Tensor: Padded tensor with last dimension size aligned to group_size
    """
    pad = (group_size - (tensor.shape[-1] % group_size)) % group_size
    tensor_padded = torch.nn.functional.pad(tensor, (0, pad), 'constant', 0)
    return tensor_padded


def convert_to_per_group_shape(input_tensor, group_size):
    """
    Function: Converts the input 2D tensor into a shape grouped by the specified group size. 
    If the total number of elements isn't divisible by the group size, zero-padding is applied.
 
    Parameters:
        input_tensor (Tensor): Input 2D tensor to be processed
        group_size (int): Target size for each group
 
    Returns:
        Tensor: Reshaped tensor with grouped dimensions
    """
    input_tensor_pad = pad_zero_by_group(input_tensor, group_size)

    return input_tensor_pad.reshape(input_tensor_pad.shape[0], input_tensor_pad.shape[1] // group_size, group_size)


@torch.no_grad()
def convert_dtype(ori_tensor, quant_dtype):
    """
    Function: tensor to dst data type. Used only by NPU.
    Parameters:
        ori_tensor: torch.tensor
        quant_dtype: quant type
    Returns:
        torch.tensor
    """
    device = ori_tensor.device
    if quant_dtype == INT8:
        converted_tensor = ori_tensor.round().clamp(INT8_MIN, INT8_MAX).to(torch.int8)
    elif quant_dtype == INT4:
        converted_tensor = ori_tensor.round().clamp(INT4_MIN, INT4_MAX).to(torch.int32)
    else:
        raise ValueError('Not supported quant_dtype {}'.format(quant_dtype))
    return converted_tensor


def quant_weight(tensor, wts_type, scale, offset=None, group_size=None):
    """
    Function: weight quant.
    Params:
        weight: the original or scaled weight to do quantization
        wts_type: weight quantized data type
        scale: torch.tensor, scale factor for weight
        offset: torch.tensor or None, offset factor for weight
        group_size: group_size of quantization
    Returns:
        torch.tensor
    """
    if group_size:
        tensor = convert_to_per_group_shape(tensor, group_size)
    tensor = tensor / scale
    if offset is not None:
        tensor = tensor + offset
    return convert_dtype(tensor, wts_type)


def apply_smooth_weight(smooth_factor, ori_weight):
    """
    Function: apply smooth factor to scale weight
    Params:
        smooth_factor: smooth factor of smooth_quant
        ori_weight: weight tensor from original operator
    Return:
        tensor: scale weight tensor
    """
    smooth_factor = smooth_factor.to(device=ori_weight.device, dtype=ori_weight.dtype)
    ori_weight_shape = ori_weight.shape
    if list(smooth_factor.shape) != [1, ori_weight_shape[1]]:
        raise RuntimeError("smooth_factor shape should be {} current shape is {}"
                        .format([1, ori_weight_shape[1]], list(smooth_factor.shape)))
    weight = ori_weight * smooth_factor.to(device=ori_weight.device)
    return weight


def check_scale_offset_shape(weight, scale_w, offset_w=None, group_size=None):
    """
    Function: check whether the quant factor satisfy the requirements.
    Params:
        weight: weight tensor from original operator
        scale_w: torch.tensor, scale factor for weight
        offset_w: torch.tensor or None, offset factor for weight
        group_size: group_size of quantization
    """
    if group_size is None:
        if scale_w.shape[0] == 1:
            return
        if scale_w.shape[0] == weight.shape[0]:
            return
        else:
            raise RuntimeError("scale.shape should be equal to 1 or cout. current is {}, "
                               "pls check quant factors from quantize model".format(scale_w.shape[0]))
    if list(scale_w.shape) != [weight.shape[0], math.ceil(weight.shape[1] / group_size), 1]:
        raise RuntimeError(
            "scale.shape should be [{}] current shape is {}, please check quant factors from quantize model.".format(
            (weight.shape[0], math.ceil(weight.shape[1] / group_size), 1), list(scale_w.shape)))

    if offset_w is None:
        return
    if scale_w.shape != offset_w.shape:
        raise RuntimeError("offset_w.shape should be equal to scale_w.shape,"
            " pls check quant factors from quantize model")
    

def apply_awq_quantize_weight(weight_tensor, awq_scale, group_size):
    """
    Function: apply awq factor to scale & clamp weight
    Params:
        weight_tensor: quantized tensor from original operator
        awq_scale: scale factor of awq
        group_size: group_size of awq
    Return:
        tensor: scale weight tensor
    """
    cin = weight_tensor.shape[1]
    if list(awq_scale.shape) != [1, cin]:
        raise RuntimeError("AWQ params scale.shape should be [1, {}] current shape is {}".format(
            cin, list(awq_scale.shape)))

    weight_tensor = weight_tensor / awq_scale
    return weight_tensor


def quant_dequant_weight(tensor, quant_params=None, scale=None, offset=None):
    """
    Function: do tensor quantize
    Params:
        tensor: quantized tensor from original operator
        scale: scale in quant factors
    Return:
        tensor: quantized tensor
    """
    wts_dtype = quant_params.get('weights_cfg').get('quant_type')
    group_size = quant_params.get('weights_cfg').get('group_size')
    ori_dtype = tensor.dtype

    ori_shape = tensor.shape
    if group_size:
        tensor = convert_to_per_group_shape(tensor, group_size)
    tensor = tensor / scale
    quant_bits = int(wts_dtype.replace('int', ''))
    if offset is not None:
        tensor = tensor + offset
    tensor = torch.clamp(torch.round(
        tensor), -pow(2, quant_bits - 1), pow(2, quant_bits - 1) - 1)
    if offset is not None:
        tensor = tensor - offset
    tensor = (tensor) * scale
    tensor = tensor.reshape(ori_shape[0], -1)[:, :ori_shape[1]].to(ori_dtype)
    return tensor
