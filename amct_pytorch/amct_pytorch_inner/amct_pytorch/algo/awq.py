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

from amct_pytorch.utils.data_utils import convert_precision, pad_zero_by_group, \
    cal_shared_exponent, scale_input_by_shared_exponents
from amct_pytorch.utils.vars import GRIDS_NUM, FLOAT4_E2M1, FLOAT4_E1M2, INT8, INT4, MXFP4_E2M1
from amct_pytorch.custom_op.utils import convert_to_per_group_shape, convert_to_dst_shape
from amct_pytorch.custom_op.utils import get_weight_min_max_by_granularity, calculate_scale_offset

TOKENS_NUM = 512
MAX_SHRINK = 0.5


@torch.no_grad()
def process_weights_for_layers(layers, scale_awq, quant_config):
    wts_type = quant_config.get("wts_type")
    for layer in layers:
        layer.weight.data = layer.weight.data.mul(scale_awq)
 
        if wts_type in (MXFP4_E2M1,):
            # for MXFP4
            shared_exponent = cal_shared_exponent(layer.weight.data, quant_config.get("wts_type"))
            layer.weight.data = scale_input_by_shared_exponents(layer.weight.data, \
                -1 * shared_exponent.to(layer.weight.data.dtype))
        else:
            # In addition to calculating the scale of awq, float4 also needs to calculate the quantization factor
            scale, offset = calculate_scale_offset_by_granularity(layer.weight.data, quant_config)
            layer.weight.data = do_quant(layer.weight.data, scale, offset, quant_config.get("weight_granularity"), 
                                            quant_config.get("group_size"))
        
        layer.weight.data = convert_precision(layer.weight.data, wts_type, 'RINT')
        if wts_type in (MXFP4_E2M1,):
            layer.weight.data = scale_input_by_shared_exponents(layer.weight.data, \
                shared_exponent.to(layer.weight.data.dtype))
        else:
            layer.weight.data = do_dequant(layer.weight.data, scale, offset, quant_config.get("weight_granularity"), 
                                            quant_config.get("group_size"))
 
        layer.weight.data = layer.weight.data / scale_awq


@torch.no_grad()
def search_scale(inputs, layers, block, quant_config, kwargs=None):
    """
    Function: use AWQ search scale
    Args:
        inputs: torch.tensor
        layers: list of torch.nn.module
        block: evaluate unit
        quant_config: quant parameters
    Returns:
        scale: torch.tensor
    """
    algo_params = quant_config.get("weight_quant_algo_params")
    wts_type = quant_config.get("wts_type")

    if not torch.isfinite(inputs).all():
        raise RuntimeError("Run AWQ error! Invalid value(nan or inf) in activation!")
    for layer in layers:
        if not torch.isfinite(layer.weight.data).all():
            raise RuntimeError("Run AWQ error! Invalid value(nan or inf) in weight!")

    kwargs = dict() if kwargs is None else kwargs
    with torch.no_grad():
        ori_out = block(inputs, **kwargs)
        if isinstance(ori_out, tuple):
            ori_out = ori_out[0]

    min_loss = torch.tensor(float('inf'))
    best_scale = None
    inputs_mean = inputs.abs().contiguous().view(-1, inputs.shape[-1]).mean(0)  # (cin)
    grids_num = algo_params.get('awq_quantize').get(GRIDS_NUM)
    ori_state = {name: tensor.detach().cpu() for name, tensor in block.state_dict().items()}
    for grid in range(grids_num):
        ratio = grid / grids_num
        scale_awq = inputs_mean.pow(ratio).clamp(min=1e-4).to(torch.float64).view(1, -1) # (1, cin)
        scale_awq = scale_awq / (scale_awq.max() * scale_awq.min()).sqrt()
        scale_awq = torch.clamp(scale_awq.to(inputs.dtype), 1e-4, torch.finfo(inputs.dtype).max)
        scale_awq = scale_awq.to(next(block.parameters()).device)
        process_weights_for_layers(layers, scale_awq, quant_config)

        quant_out = block(inputs, **kwargs)
        if isinstance(quant_out, tuple):
            quant_out = quant_out[0]

        loss = (ori_out - quant_out).float().pow(4).mean()
        if not torch.isnan(loss) and loss < min_loss:
            min_loss = loss
            best_scale = scale_awq

        block.load_state_dict(ori_state)

    if best_scale is None or torch.isinf(min_loss):
        raise RuntimeError("Run AWQ error! Got invalid loss.")

    return best_scale


@torch.no_grad()
def apply_scale(scale_list, named_linears, input_feat):
    """
    Function: apply scale to weight and activation
    Args:
        scale_list: dict. {name: scale}
        named_linears: dict. {name: module}
        input_feat: dict. {name: input_data}
    """
    for name, scale in scale_list.items():
        scale = scale.to(named_linears[name].weight.device)
        named_linears[name].weight.data.mul_(scale)
        input_feat[name].div_(scale)


@torch.no_grad()
def apply_clip_by_weight_granularity(clip_max_list, named_linears, weight_granularity, group_size):
    """
    Function: apply clip to weight If the linear layer computes the truncation value 
    Args:
        clip_max_list: dict. {name: clip}
        named_linears: dict. {name: module}
        group_size: Size of each group for quantization calculations
    Returns:
        dict. {name: cliped_weight}
    """
    cliped_weight = {}
    for name in named_linears:
        weight = named_linears[name].weight.data
        if name in clip_max_list:
            # do clip
            clip_max = clip_max_list[name]
            ori_shape = weight.shape
            if weight_granularity == 'PER_GROUP':
                weight_all = pad_zero_by_group(weight, group_size)
                weight_all = weight_all.reshape(weight.shape[0], -1, group_size)
                weight_all = torch.clamp(weight_all, -1 * clip_max, clip_max)
                weight_all = weight_all.reshape(weight_all.shape[0], -1)
                weight = weight_all[:, :weight.shape[1]]
                weight = weight.reshape(ori_shape)
            else:
                weight = torch.clamp(weight, -1 * clip_max, clip_max)

        cliped_weight[name] = weight
    
    return cliped_weight


@torch.no_grad()
def process_weight(clip_weight, quant_config, ori_cin, ori_dtype):
    weight_granularity = quant_config.get('weight_granularity')
    group_size = quant_config.get('group_size')
    if weight_granularity == 'PER_GROUP':
        clip_weight = clip_weight.reshape(clip_weight.shape[0], -1)
        clip_weight = clip_weight[:clip_weight.shape[0], :ori_cin]
    # In addition to calculating the scale of awq, float4 also needs to calculate the quantization factor
    if quant_config.get("wts_type") in (FLOAT4_E2M1, FLOAT4_E1M2, INT8, INT4):
        scale, offset = calculate_scale_offset_by_granularity(clip_weight, quant_config)
        clip_weight = do_quant(clip_weight, scale, offset, weight_granularity, group_size)
    else:
        # for MXFP4
        shared_exponent = cal_shared_exponent(clip_weight, quant_config.get("wts_type"))
        clip_weight = scale_input_by_shared_exponents(clip_weight, \
            -1 * shared_exponent.to(clip_weight.dtype))

    quant_weight = convert_precision(clip_weight, quant_config.get("wts_type"), 'RINT')

    if quant_config.get("wts_type") in (FLOAT4_E2M1, FLOAT4_E1M2, INT8, INT4):
        quant_weight = do_dequant(quant_weight, scale, offset, weight_granularity, group_size)
    else:
        quant_weight = scale_input_by_shared_exponents(quant_weight, shared_exponent.to(ori_dtype))

    if weight_granularity == 'PER_GROUP': # [batch_size, 1, group_num, group_size]
        quant_weight = pad_zero_by_group(quant_weight, group_size)
        quant_weight = quant_weight.reshape(quant_weight.shape[0], 1, -1, group_size)
    return quant_weight


@torch.no_grad()
def reshape_input_weight(input_data, weight, weight_granularity, group_size):
    if weight_granularity == 'PER_GROUP':
        # weight: [co, ci] -> [co, 1, group_num, group_size] 
        weight = pad_zero_by_group(weight, group_size)
        weight_all = weight.reshape(weight.shape[0], 1, -1, group_size)
        # input_data: [tokens_num, ci] -> [1, tokens_num, group_num, group_size]
        input_data = pad_zero_by_group(input_data, group_size)
        input_data = input_data.view(-1, input_data.shape[-1])
        input_data = input_data.reshape(1, input_data.shape[0], -1, group_size)
        if input_data.shape[1] > TOKENS_NUM:
            input_data = input_data[:, 0 :: input_data.shape[1] // TOKENS_NUM]
    else:
        weight_all = weight.reshape(weight.shape[0], 1, -1)
        input_data = input_data.reshape(1, -1, input_data.shape[-1])
    return input_data, weight_all


@torch.no_grad()
def search_clip(named_linears, input_feat, quant_config):
    """
    Function: search clip_max for weight
    Args:
        named_linears: dict. {name: module}
        input_feat: dict. {name: input_data}
        quant_config: quant parameters
    Returns:
        dict. {name: clip_max}
    """
    grids_num = quant_config.get("weight_quant_algo_params").get('awq_quantize').get(GRIDS_NUM)
    group_size = quant_config.get("group_size")
    weight_granularity = quant_config.get('weight_granularity')
    clip_max_list = dict()
    for name, mod in named_linears.items():
        if any([_ in name for _ in ['q_', 'k_']]): # q/k not suitable for clip
            continue
        weight = mod.weight
        ori_cout = weight.shape[0]
        ori_cin = weight.shape[1]
        input_data = input_feat[name]
        input_data, weight_all = reshape_input_weight(input_data, weight, weight_granularity, group_size)

        batch_size = 256 if weight_all.shape[0] % 256 == 0 else 64 # prevent overflow memory  
        if weight_all.shape[0] % batch_size != 0:
            pad = (len(weight_all.shape) * 2) * [0]
            pad[-1] = batch_size - weight_all.shape[0] % batch_size
            weight_all = torch.nn.functional.pad(weight_all, pad, 'constant', 0)

        best_clip_max_all = list()
        for batch_num in range(weight_all.shape[0] // batch_size):
            weight = weight_all[batch_num * batch_size : (batch_num + 1) * batch_size]
            ori_out = input_data * weight
            if weight_granularity != 'PER_TENSOR':
                ori_out = ori_out.sum(dim=-1)
                ori_max_val = weight.abs().max(dim=-1, keepdim=True).values # [co, 1, group_num, 1] / [co, 1, cin, 1]
            else: # PER_TENSOR
                ori_out = ori_out.sum()
                ori_max_val = weight.abs().max().reshape(-1)

            best_clip_max = ori_max_val.clone()
            min_loss = torch.ones_like(ori_max_val) * float('inf')
            for i in range(int(MAX_SHRINK * grids_num)):
                clip_max = ori_max_val * (1 - i / grids_num)
                clip_weight = torch.clamp(weight, -clip_max, clip_max) # [batch_size, 1, group_num, group_size]
                quant_weight = process_weight(clip_weight, quant_config, ori_cin, weight.dtype)
                quant_out = (input_data * quant_weight)
                if weight_granularity != 'PER_TENSOR':
                    quant_out = quant_out.sum(dim=-1)
                    loss = (ori_out - quant_out).pow(2).mean(dim=1).view(min_loss.shape)
                else:
                    quant_out = quant_out.sum()
                    loss = (ori_out - quant_out).pow(2).view(min_loss.shape)
                best_loss_idx = loss < min_loss
                min_loss[best_loss_idx] = loss[best_loss_idx]
                best_clip_max[best_loss_idx] = clip_max[best_loss_idx]
            best_clip_max_all.append(best_clip_max)

        best_clip_max = torch.cat(best_clip_max_all, dim=0)
        if weight_granularity != 'PER_TENSOR':
            best_clip_max = best_clip_max.squeeze(1)
            clip_max_list[name] = best_clip_max[:ori_cout]
        else:
            clip_max_list[name] = best_clip_max[0]
    return clip_max_list


@torch.no_grad()
def get_quant_scale(adjusted_weight, quant_config):
    """
    Apply clipping to layer weights and compute scaling factors based on quantization configuration.

    Args:
        adjusted_weight (dict): The weight after awq_scale and clip adjustment
        quant_config (dict): Quantization configuration parameters (e.g., group_size, wts_type)

    Returns:
        dict: Dictionary mapping layer names to their computed scaling factors
    """
    scale_group_list = dict()
    offset_group_list = dict()
    group_size = quant_config.get("group_size")
    for name in adjusted_weight:
        weight = adjusted_weight[name]
        scale_group, offset_group = calculate_scale_offset_by_granularity(weight, quant_config)
        scale_group_list[name] = scale_group
        offset_group_list[name] = offset_group

    return scale_group_list, offset_group_list


def calculate_scale_offset_by_granularity(weight, quant_config):
    weight_min, weight_max = get_weight_min_max_by_granularity(weight, quant_config)
    scale_w, offset_w = calculate_scale_offset(weight_max, weight_min, quant_config.get('wts_asymmetric'), 
                                               quant_config.get("wts_type"))

    return scale_w, offset_w


def do_quant(weight, scale_w, offset_w, weight_granularity, group_size=None):
    """
    Scales weights to a quantization type scope based on specified group size.

    Parameters:
    weight: Original weights to be scaled
    quant_type: Quantization type for scaling reference
    group_size: Size of each group for quantization calculations

    Returns:
    Returns tuple of (scaled weights, corresponding scaling factors)
    """
    ori_type = weight.dtype
    ori_shape = weight.shape
    if weight_granularity == 'PER_GROUP':
        weight = convert_to_per_group_shape(weight, group_size)
    scaled_weights = weight / scale_w 
    if offset_w is not None:
        scaled_weights += offset_w
    weight = convert_to_dst_shape(scaled_weights, ori_shape)
    return weight.to(ori_type)


def do_dequant(weight, scale, offset, weight_granularity, group_size=None):
    """
    Scale the weight tensor by group factors and restore its original shape.

    Parameters:
    weight: Multi-dimensional weight tensor to be scaled.
    scale_group: quant factor, shape is (group_num, 1).
    group_size: Integer representing the size of each quantization group.

    Returns:
    A tensor with the same shape as the original input 'weight' after group-wise scaling.
    """
    ori_shape = weight.shape
    if weight_granularity == 'PER_GROUP':
        weight = convert_to_per_group_shape(weight, group_size)
    if offset is not None:
        weight = weight - offset
    dequant_weights = weight * scale
    return convert_to_dst_shape(dequant_weights, ori_shape).to(weight.dtype)
