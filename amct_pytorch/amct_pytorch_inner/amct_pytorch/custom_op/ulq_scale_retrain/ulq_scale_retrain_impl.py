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
    
    
    
def ulq_scale_retrain_backward_pytorch(input_tensor, gradOutputs, scale, numBits, sRecFlag, group, axis):
    """
    Function: Perform backward pass for ULQ scale retraining
    Inputs:
        input_tensor: torch.Tensor - Input tensor
        gradOutputs: torch.Tensor - Gradient of the loss with respect to the output tensor
        scale: torch.Tensor - Scale factor for quantization
        numBits: int - Number of bits for quantization
        sRecFlag: bool - Whether to use reciprocal of scale
        group: int - Number of groups for group-wise quantization
        axis: int - Axis for channel-wise quantization
    Outputs:
        gradOutputs: torch.Tensor - Gradient of the loss with respect to the input tensor
        gradScales: torch.Tensor - Gradient of the loss with respect to the scale
    """
    if input_tensor.size(0) % group != 0:
        raise ValueError("Input tensor size must be divisible by the number of groups")
    
    sizes = list(input_tensor.size())
    
    if len(sizes) == 4:
        input_tensor = input_tensor.view(sizes[0], -1)
        gradOutputs = gradOutputs.view(sizes[0], -1)
    
    if scale.numel() > 1:
        input_tensor = input_tensor.view(sizes[axis], -1)
        gradOutputs = gradOutputs.view(sizes[axis], -1)
    
    if group > 1:
        scaleProcess = scale.repeat_interleave(input_tensor.size(axis) // group)
    else:
        scaleProcess = scale
    
    if sRecFlag:
        scaleProcess = 1 / scaleProcess
    
    gradScales = calc_scale_gradient(input_tensor, scaleProcess, gradOutputs, numBits, sRecFlag)

    if sRecFlag:
        scaleProcess = 1 / scaleProcess
    
    if group > 1:
        gradScales = gradScales.view(group, -1).mean(dim=1)
    
    if len(sizes) == 4:
        gradOutputs = gradOutputs.view(sizes[0], sizes[1], sizes[2], sizes[3])
    
    return gradOutputs, gradScales


def calc_scale_gradient(input_tensor, scaleProcess, gradOutputs, numBits, sRecFlag):
    """
    Function: Calculate the gradient of the loss with respect to the scale
    Inputs:
        input_tensor: torch.Tensor - Input tensor
        scaleProcess: torch.Tensor - Processed scale factor
        gradOutputs: torch.Tensor - Gradient of the loss with respect to the output tensor
        numBits: int - Number of bits for quantization
        sRecFlag: bool - Whether to use reciprocal of scale
    Outputs:
        gradScales: torch.Tensor - Gradient of the loss with respect to the scale
    """
    halfStage = 2 ** (numBits - 1)
    
    upperMask = (input_tensor > scaleProcess.unsqueeze(1) * (halfStage - 1)).float()
    lowerMask = (input_tensor < -scaleProcess.unsqueeze(1) * halfStage).float()
    innerMask = 1 - (upperMask + lowerMask)
    
    notRoundTensor = input_tensor / scaleProcess.unsqueeze(1)
    roundTensor = notRoundTensor.round()
    gradScales = (roundTensor - notRoundTensor) * innerMask + \
        (-halfStage * lowerMask) + ((halfStage - 1) * upperMask)
    
    if sRecFlag:
        gradScales = -gradScales * (scaleProcess ** 2).unsqueeze(1)
    
    gradScales = gradScales * (torch.sqrt(torch.tensor(input_tensor.size(1) * (halfStage - 1))).reciprocal())
    
    if scaleProcess.size(0) > 1:
        gradScales = (gradScales * gradOutputs).sum(1)
    else:
        gradScales = (gradScales * gradOutputs).sum()
        gradScales = gradScales.unsqueeze(0)
    
    return gradScales