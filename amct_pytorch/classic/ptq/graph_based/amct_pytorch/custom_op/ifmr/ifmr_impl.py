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

from ....amct_pytorch.custom_op.utils import process_scale
from ....amct_pytorch.utils.vars import BASE
from ....amct_pytorch.utils.vars import FLT_EPSILON


def find_optimal_scale_offset(data, ifmr_param, max_value, min_value):
    """
    Find the optimal scale and offset for quantization based on the given data and parameters.
    Args:
        data (torch.Tensor): The input data tensor for which to find the optimal scale and offset.
        ifmr_param (dict): A dictionary containing parameters related to the quantization process.
        max_value (torch.Tensor): The maximum value obtained by percentile
        min_value (torch.Tensor): The minimum value obtained by percentile
    
    Returns:
        tuple: A tuple containing two elements:
            - scale (torch.Tensor): The optimal scaling factor.
            - offset (torch.Tensor): The optimal offset value.
    """
    device = data.device
    start_ratio = ifmr_param['start_ratio']
    end_ratio = ifmr_param['end_ratio']
    step = ifmr_param['step']
    num_bits = ifmr_param['num_bits']
    with_offset = ifmr_param['with_offset']
    
    if not with_offset:
        bound_negative = abs(max_value) < abs(min_value)
        max_value = max(abs(max_value), abs(min_value))

    max_candidates = torch.arange(start_ratio, end_ratio, step,
                                  dtype=torch.float32, device=device)
    max_candidates = max_candidates * max_value
    losses = torch.zeros_like(max_candidates, device=device)

    if with_offset:
        max_limit = pow(BASE, num_bits) - 1
        scale_candidates = (max_candidates - min_value) / max_limit
    else:
        if bound_negative:
            max_limit = pow(BASE, num_bits - 1)
        else:
            max_limit = pow(BASE, num_bits - 1) - 1
        scale_candidates = max_candidates / max_limit
    
    scale_candidates, _ = process_scale(
        scale_candidates, torch.zeros_like(scale_candidates, device=device))

    for idx, max_candidate in enumerate(max_candidates):
        if not with_offset:
            min_value = -max_candidate
        clipped = torch.clamp(data, min_value, max_candidate)
        quantized = torch.round(clipped / scale_candidates[idx]) * scale_candidates[idx]
        losses[idx] = torch.nn.functional.mse_loss(quantized, data)
    
    scale = scale_candidates[torch.argmin(losses)]
    if with_offset:
        offset = -torch.round(min_value / scale) - (pow(BASE, num_bits - 1))
    else:
        offset = torch.tensor(0, dtype=torch.float32, device=device)

    return scale, offset


def find_max_and_min_value_with_percentile(data, max_percentile, min_percentile):
    """
    Find max and min searching init data based on given percentile
    Args:
        data (torch.Tensor): The input data for activation calibration
        max_percentile (float): The percentile (between 0 and 1) used to init the maximum value.
        min_percentile (float): The percentile (between 0 and 1) used to init the minimum value.
    
    Returns:
        tuple: A tuple containing the maximum value and the minimum value.
    """
    length = data.numel()
    if (1.0 - max_percentile) < FLT_EPSILON:
        max_index = 1
    else:
        max_index = length - int(max_percentile * length)

    if (1.0 - min_percentile) < FLT_EPSILON:
        min_index = 1 
    else:
        min_index = length - int(min_percentile * length)

    max_value = torch.topk(data, max_index, largest=True).values[-1]
    min_value = torch.topk(data, min_index, largest=False).values[-1]

    max_value = max(torch.tensor(0., device=data.device), max_value)
    min_value = min(torch.tensor(0., device=data.device), min_value)

    return max_value, min_value


def ifmr_forward_pytorch(data,
                device,
                num_bits=8,
                with_offset=False,
                max_percentile=0.999999,
                min_percentile=0.999999,
                search_start=0.7,
                search_end=1.3,
                search_step=0.01):
    """
    Function: forward function for ifmr to find optimal clip_max, clip_min
        then calculate scale and offset

    Args:
        data (torch.Tensor): Input tensor to be quantized.
        device (torch.device): device to do calculate, -1 while using cpu.
        num_bits (int, optional): Number of bits for quantization.
        with_offset (bool, optional): Whether to use asymmetric quant.
        max_percentile (float, optional): Percentile to clip data while finding max candidates
        min_percentile (float, optional): Percentile to clip data while finding min candidates
        search_start (float, optional): Start value for searching optimal clip_max and clip_min
        search_end (float, optional): End value for searching optimal clip_max and clip_min
        search_step (float, optional): Step size for searching optimal clip_max and clip_min    
    Returns:
        scale (torch.Tensor): Quant factor to do scaling
        offset (torch.Tensor): Quant factor to do offseting
        clip_max (torch.Tensor): Max limit to clip input data
        clip_min (torch.Tensor): Min limit to clip input data
    """
    data = data.view(-1)
    data = data.to(device)

    ifmr_param = {
        'num_bits': num_bits,
        'with_offset': with_offset,
        'start_ratio': search_start,
        'end_ratio': search_end,
        'step': search_step
    }

    max_value, min_value = find_max_and_min_value_with_percentile(data,
                                                                  max_percentile,
                                                                  min_percentile)
    scale, offset = find_optimal_scale_offset(
        data, ifmr_param, max_value, min_value)

    clip_max = torch.tensor(1., device=data.device).mul_(BASE ** (num_bits - 1) - 1).sub_(offset).mul_(scale)
    clip_min = torch.tensor(1., device=data.device).mul_(BASE ** (num_bits - 1)).add_(offset).mul_(-scale)

    return scale, offset, clip_max, clip_min


def ifmr_backward_pytorch(grad):
    """
    Compute the gradient for the IFMR backward pass.

    Args:
        grad (torch.Tensor): The gradient tensor from the subsequent layer.
    """
    return grad
