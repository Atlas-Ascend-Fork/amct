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

from ....amct_pytorch.custom_op.utils import calculate_scale_offset, calculate_scale_by_group_size, \
    convert_to_per_group_shape, apply_fake_quantize_and_anti_quantize


def fake_quant_functor(data, dims, data_type, scale, offset, channel_wise, group):
    """
    Function: Apply fake quantization and de-quantization to the data
    Inputs:
        data: torch.Tensor - Input tensor to be quantized
        dims: tuple - Original dimensions of the input tensor
        data_type: str - Data type for quantization (e.g., 'INT8')
        scale: torch.Tensor - Scale factor for quantization
        offset: torch.Tensor - Offset for quantization
        channel_wise: bool - Whether to apply channel-wise quantization
        group: int - Number of groups for group-wise quantization
    Outputs:
        output: torch.Tensor - Quantized and de-quantized tensor
    """
    if not channel_wise:
        data_num = torch.numel(data)
        group_size = data_num // group
        data = convert_to_per_group_shape(data, group_size)

    data = apply_fake_quantize_and_anti_quantize(data, scale, offset, data_type)

    return data.reshape(dims)


def ulq_scale_retrain_forward_pytorch(input_tensor, scale, offset, num_bits, channel_wise, \
                                      arq_init, s_rec_flag, group):
    """
    Function: Perform forward pass for ULQ scale retraining
    Inputs:
        input_tensor: torch.Tensor - Input tensor
        scale: torch.Tensor - Scale factor for quantization
        offset: torch.Tensor - Offset for quantization
        num_bits: int - Number of bits for quantization
        channel_wise: bool - Whether to apply channel-wise quantization
        arq_init: bool - Whether to initialize scale and offset using ARQ
        s_rec_flag: bool - Whether to use reciprocal of scale
        group: int - Number of groups for group-wise quantization
    Outputs:
        output_tensor: torch.Tensor - Quantized and de-quantized tensor
        scale: torch.Tensor - Updated scale factor
        offset: torch.Tensor - Updated offset
    """
    if input_tensor.size(0) % group != 0:
        raise ValueError("Input tensor size must be divisible by the number of groups")
    output_tensor = input_tensor.clone().detach()
    sizes = input_tensor.size()

    group_size = torch.numel(input_tensor) // len(scale)
    input_tensor = input_tensor.reshape(-1, group_size)

    data_type = 'INT' + str(num_bits)
    if arq_init:
        if channel_wise:
            data_max = torch.max(input_tensor, dim=1).values
            data_min = torch.min(input_tensor, dim=1).values
            scale, offset = calculate_scale_offset(data_max, data_min, False, data_type)
        else:
            scale, _ = calculate_scale_by_group_size(input_tensor, data_type, group_size)
            scale = scale.squeeze(1).squeeze(1)
            offset = torch.zeros_like(scale)
        scale = scale.reciprocal() if s_rec_flag else scale

    scale = scale.reciprocal() if s_rec_flag else scale
    output_tensor = fake_quant_functor(input_tensor, sizes, data_type, scale, offset, channel_wise, group)
    
    if not arq_init:
        scale = scale.unsqueeze(1)
        offset = offset.unsqueeze(1)
    
    scale = scale.reciprocal() if s_rec_flag else scale
    scale.requires_grad = True
    output_tensor = output_tensor.reshape(sizes)
    return output_tensor, scale, offset
    
    
    
def ulq_scale_retrain_backward_pytorch(input_tensor, grad_outputs, scale, num_bits, srec_flag, group, axis):
    """
    Function: Perform backward pass for ULQ scale retraining
    Inputs:
        input_tensor: torch.Tensor - Input tensor
        grad_outputs: torch.Tensor - Gradient of the loss with respect to the output tensor
        scale: torch.Tensor - Scale factor for quantization
        num_bits: int - Number of bits for quantization
        srec_flag: bool - Whether to use reciprocal of scale
        group: int - Number of groups for group-wise quantization
        axis: int - Axis for channel-wise quantization
    Outputs:
        grad_outputs: torch.Tensor - Gradient of the loss with respect to the input tensor
        grad_scales: torch.Tensor - Gradient of the loss with respect to the scale
    """
    if input_tensor.size(0) % group != 0:
        raise ValueError("Input tensor size must be divisible by the number of groups")
    
    sizes = list(input_tensor.size())
    
    if len(sizes) == 4:
        input_tensor = input_tensor.view(sizes[0], -1)
        grad_outputs = grad_outputs.view(sizes[0], -1)
    
    if scale.numel() > 1:
        input_tensor = input_tensor.view(sizes[axis], -1)
        grad_outputs = grad_outputs.view(sizes[axis], -1)
    
    if group > 1:
        scale_process = scale.repeat_interleave(input_tensor.size(axis) // group)
    else:
        scale_process = scale
    
    if srec_flag:
        scale_process = 1 / scale_process
    
    grad_scales = calc_scale_gradient(input_tensor, scale_process, grad_outputs, num_bits, srec_flag)

    if srec_flag:
        scale_process = 1 / scale_process
    
    if group > 1:
        grad_scales = grad_scales.view(group, -1).mean(dim=1)
    
    if len(sizes) == 4:
        grad_outputs = grad_outputs.view(sizes[0], sizes[1], sizes[2], sizes[3])
    
    return grad_outputs, grad_scales


def calc_scale_gradient(input_tensor, scale_process, grad_outputs, num_bits, srec_flag):
    """
    Function: Calculate the gradient of the loss with respect to the scale
    Inputs:
        input_tensor: torch.Tensor - Input tensor
        scale_process: torch.Tensor - Processed scale factor
        grad_outputs: torch.Tensor - Gradient of the loss with respect to the output tensor
        num_bits: int - Number of bits for quantization
        srec_flag: bool - Whether to use reciprocal of scale
    Outputs:
        grad_scales: torch.Tensor - Gradient of the loss with respect to the scale
    """
    half_stage = 2 ** (num_bits - 1)
    
    upper_mask = (input_tensor > scale_process.unsqueeze(1) * (half_stage - 1)).float()
    lower_mask = (input_tensor < -scale_process.unsqueeze(1) * half_stage).float()
    inner_mask = 1 - (upper_mask + lower_mask)
    
    not_round_tensor = input_tensor / scale_process.unsqueeze(1)
    round_tensor = not_round_tensor.round()
    grad_scales = (round_tensor - not_round_tensor) * inner_mask + \
        (-half_stage * lower_mask) + ((half_stage - 1) * upper_mask)
    
    if srec_flag:
        grad_scales = -grad_scales * (scale_process ** 2).unsqueeze(1)
    
    grad_scales = grad_scales * (torch.sqrt(torch.tensor(input_tensor.size(1) * (half_stage - 1))).reciprocal())
    
    if scale_process.size(0) > 1:
        grad_scales = (grad_scales * grad_outputs).sum(1)
    else:
        grad_scales = (grad_scales * grad_outputs).sum()
        grad_scales = grad_scales.unsqueeze(0)
    
    return grad_scales