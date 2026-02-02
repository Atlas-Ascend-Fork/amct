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
from ....amct_pytorch.utils.log import LOGGER


ASCNED_OPTIMIZED_VALUE = 16


def check_params(tensor_list, prune_axis_list, prune_ratio, prune_group, ascend_optimized):
    """ check inputs params is valid for bcp """
    tensor_num = len(tensor_list)
    flag = True
    if tensor_num != len(prune_axis_list):
        LOGGER.loge("bcp input tensor num is not equal to prune axis num.")
        flag = False
    
    if prune_ratio >= 1 or prune_ratio <= 0:
        raise RuntimeError(
                "prune_ratio should be larger than 0 and samller than 1, pls check config.")

    for i in range(0, tensor_num):
        if tensor_list[i].dim() <= prune_axis_list[i]:
            LOGGER.loge("bcp input tensor[{}] dim is less than prune axis.".format(i))
            flag = False
    
        cout_length = tensor_list[i].shape[prune_axis_list[i]]
        if cout_length < prune_group:
            LOGGER.loge("bcp input tensor[{}] cout is less than prune group.".format(i))
            flag = False
        if not ascend_optimized and cout_length % prune_group != 0:
            LOGGER.loge("bcp input tensor[{}] cout is not multiple of prune group.".format(i))
            flag = False

    if not flag:
        raise RuntimeError("Inner Error of bcp op in prune process.")


def cal_prune_num(num, prune_ratio, ascend_optimized, prune_group=1):
    """ calculate the prune channel num in total """
    remain_num = num - round(num * prune_ratio)
    # make sure remain is multiple of prune group
    if ascend_optimized:
        remain_num = math.ceil(remain_num / ASCNED_OPTIMIZED_VALUE) * ASCNED_OPTIMIZED_VALUE
        if remain_num == 0:
            remain_num = ASCNED_OPTIMIZED_VALUE
    # for multi groups, make sure each group prune same channel num
    if prune_group > 1:
        remain_num = (remain_num // prune_group) * prune_group
    if remain_num == 0:
        remain_num = prune_group
    return num - remain_num


def cal_tensor_norm(tensor, prune_axis):
    """ calculate norm value for each tensor, norm.shape = [num_channel] = [prune_group * group_len] """
    group_size = tensor.numel() // tensor.shape[0]
    norm_axis = []
    for axis in range(0, tensor.dim()):
        if axis != prune_axis:
            norm_axis.append(axis)
    l2_norm = torch.norm(tensor, p=2, dim=norm_axis) / group_size
    l2_norm = (l2_norm - l2_norm.min()) / (l2_norm.max() - l2_norm.min())

    return l2_norm


def gen_mask_by_group(num_channel, prune_group, group_len, prune_num_group, norm_sum):
    """
    Function: sort norm value for each group, if need prune set 0, otherwise set 1
    Args:
    num_channel: channel in total
    prune_group: group nums
    group_len: the channel num in each group
    prune_num_group: the prune channel num in each group
    norm_sum: l2 norm value of tensor by channel
    """
    groups_mask = torch.zeros(num_channel)
    for i in range(prune_group):
        norm_group = norm_sum[i * group_len: (i + 1) * group_len]
        indexed_data = sorted(enumerate(norm_group), key=lambda x: x[1])
        # original_indices is a list of idx sorted by the norm value
        original_indices = [idx for idx, val in indexed_data]
        for j in range(group_len):
            value = 0 if j < prune_num_group else 1
            groups_mask[group_len * i + original_indices[j]] = value

    return groups_mask


def bcp(tensor_list, prune_axis_list, prune_ratio, prune_group, ascend_optimized):
    """
    Function: bcp funtion.
    Args:
    tensor_list: data used for channel prune in list of torch.tensor.
    prune_axis_list: prune axis for channel prune in list of dim value
    prune_ratio: ratio of prune channels to the total number of channels
    prune_group: num of prune groups
    ascend_optimized: bool, is optimation for ascend is needed
    """
    check_params(tensor_list, prune_axis_list, prune_ratio, prune_group, ascend_optimized)
    # prune axis channel num in total
    num_channel = tensor_list[0].shape[prune_axis_list[0]]
    # prune axis channel num in each group
    group_len = num_channel // prune_group
    # prune channel in total 
    prune_num = cal_prune_num(tensor_list[0].shape[prune_axis_list[0]], prune_ratio, ascend_optimized, prune_group)
    # prune channel in each group
    prune_num_group = prune_num // prune_group

    norm_sum = []
    for i, tensor in enumerate(tensor_list):
        norm_sum += cal_tensor_norm(tensor, prune_axis_list[i])

    groups_mask = gen_mask_by_group(num_channel, prune_group, group_len, prune_num_group, norm_sum)
    return groups_mask.to(tensor_list[0].device)

