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


def cal_shape(ori_tensor_shape, prune_axis, group_num, group_size, padding_size):
    """ calculate tensor shapes for generate selective mask """
    group_shape = []
    padding_shape = []
    ungroup_shape = []
    # ori shape is [Cout, Cin, H, W]
    # group shape is [Cout, group_num, 4, H, W]
    # padding shape is [Cout, padding_size, H, W]
    # ungroup shape is [Cout, -1, H, W]
    for i in range(0, prune_axis):
        group_shape.append(ori_tensor_shape[i])
        padding_shape.append(ori_tensor_shape[i])
        ungroup_shape.append(ori_tensor_shape[i])
    group_shape.append(group_num)
    group_shape.append(group_size)
    padding_shape.append(padding_size)
    ungroup_shape.append(-1)
    for i in range(prune_axis + 1, len(ori_tensor_shape)):
        group_shape.append(ori_tensor_shape[i])
        padding_shape.append(ori_tensor_shape[i])
        ungroup_shape.append(ori_tensor_shape[i])
    
    return group_shape, padding_shape, ungroup_shape


def selective_mask_gen(tensor, prune_axis, group_size, pruned_size):
    """
    Function: selective mask gen funtion.
    Args:
    tensor: data used for selective prune in torch.tensor
    prune_axis: prune axis for selective
    group_size: base num of selective
    pruned_size: prune num of selective
    """
    if prune_axis >= tensor.dim():
        raise RuntimeError(
                "selective input tensor dim is less than prune axis.")
    ori_tensor_shape = tensor.shape
    channel_num = ori_tensor_shape[prune_axis]
    # if prune channel size less or equal to prune size, do not prune
    if channel_num <= pruned_size:
        return torch.ones_like(tensor)

    # channel_num should be multiple of 4, otherwise padding
    group_num = (channel_num + group_size - 1) // group_size
    padding_size = group_num * group_size - channel_num
    group_shape, padding_shape, ungroup_shape = cal_shape(ori_tensor_shape, prune_axis,
                                                        group_num, group_size, padding_size)

    padding_tensor = torch.zeros(padding_shape).to(tensor.device).to(tensor.dtype)
    tensor_padding = torch.cat((tensor.abs(), padding_tensor), prune_axis)
    tensor_padding_group = tensor_padding.reshape(group_shape)

    # pick smallest top2, find its indices
    sort_values, sort_indices = tensor_padding_group.topk(pruned_size, prune_axis + 1, largest=False)
    # gen mask
    mask = torch.ones_like(tensor_padding_group).scatter_(dim=prune_axis + 1, index=sort_indices, value=0)
    result = torch.split(mask.view(ungroup_shape), tensor.size(prune_axis), dim=prune_axis)
    return result[0]
