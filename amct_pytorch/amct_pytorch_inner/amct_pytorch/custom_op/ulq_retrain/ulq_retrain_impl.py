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

from ....amct_pytorch.utils.vars import FLT_EPSILON
from ....amct_pytorch.custom_op.utils import calculate_scale_offset
from ....amct_pytorch.custom_op.utils import apply_fake_quantize_and_anti_quantize


def ulq_retrain_forward_pytorch(data,
                clip_max,
                clip_min,
                clip_max_pre,
                clip_min_pre,
                num_bits,
                fixed_min,
                asymmetric):
    """
    Perform the forward pass for ULQ.

    Args:
        data (torch.Tensor): The input tensor to be quantized.
        clip_max (torch.Tensor): The maximum value to which the input tensor is clipped before quantization.
        clip_min (torch.Tensor): The minimum value to which the input tensor is clipped before quantization.
        clip_max_pre (torch.Tensor): Predefined maximum clipping value.
        clip_min_pre (torch.Tensor): Predefined minimum clipping value.
        num_bits (int): The number of bits used for quantization.
        fixed_min (bool): If True, the minimum clipping value is fixed to zero.
        asymmetric (bool): Whether to use asymmetric quantization.
    
    Returns:
        tuple: A tuple containing:
            - output (torch.Tensor): The quantized output tensor.
            - scale (torch.Tensor): The scaling factor used for quantization.
            - offset (torch.Tensor): The offset used for quantization.
            - clip_max (torch.Tensor): The maximum clipping value.
            - clip_min (torch.Tensor): The minimum clipping value.
    """
    if fixed_min:
        clip_min = torch.zeros_like(clip_min)

    if (clip_max - clip_min).all() <= FLT_EPSILON:
        clip_min = clip_min_pre
        clip_max = clip_max_pre
    elif (clip_max < 0).all():
        clip_max = clip_max_pre
    elif (clip_min > 0).all():
        clip_min = clip_min_pre
    
    data_type = 'INT' + str(num_bits)
    scale, offset = calculate_scale_offset(clip_max, clip_min, asymmetric, data_type)
    data = apply_fake_quantize_and_anti_quantize(data, scale, offset, data_type)

    return data, scale, offset, clip_max, clip_min


def ulq_retrain_backward_pytorch(data,
                         grad_outputs,
                         clip_max,
                         clip_min,
                         num_bits,
                         asymmetric):
    """
    Perform the backward pass for ULQ.

    Args:
        data (torch.Tensor): The input tensor to be quantizd.
        grad_outputs (torch.Tensor): The gradient of the loss with respect to the output.
        clip_max (torch.Tensor): The maximum value to which the input tensor is clipped before quantization.
        clip_min (torch.Tensor): The minimum value to which the input tensor is clipped before quantization.
        num_bits (int): The number of bits used for quantization.
        asymmetric (bool): Whether to use asymmetric quantization.
    
    Returns:
        grad_inputs (torch.Tensor) The gradient of the loss with respect to the input of the quantization function.
        grad_acts_clip_max (torch.Tensor) The gradient of the loss with respect to the maximum clipping value.
        grad_acts_clip_min (torch.Tensor) The gradient of the loss with respect to the minimum clipping value.
    """
    data_type = 'INT' + str(num_bits)
    scale, offset = calculate_scale_offset(clip_max, clip_min, asymmetric, data_type)
    upper_mask = torch.where(data > scale * ((pow(2, num_bits - 1) - 1) - offset), 1, 0)
    lower_mask = torch.where(data < scale * (-pow(2, num_bits - 1) - offset), 1, 0)

    not_round_tensor = torch.where(upper_mask > 0, clip_max, data)
    not_round_tensor = torch.where(lower_mask > 0, clip_min, not_round_tensor)
    not_round_tensor = not_round_tensor / scale

    round_tensor = torch.round(not_round_tensor)
    quant_error = (round_tensor - not_round_tensor) / (pow(2, num_bits) - 1)
    
    grad_acts_clip_max = torch.sum((quant_error + upper_mask) * grad_outputs)
    grad_acts_clip_min = torch.sum((lower_mask - quant_error) * grad_outputs)

    grad_inputs = torch.where(
        (upper_mask + lower_mask < 1), grad_outputs, torch.tensor(0., dtype=torch.float32, device=data.device))

    return grad_inputs, grad_acts_clip_max, grad_acts_clip_min