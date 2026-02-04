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
import numpy as np
from ...amct_pytorch.common.utils.onnx_node_util import AttributeProtoHelper


def get_deconv_group(node_deconv):
    """Get ConvTranspose's group param value"""
    attrs_helper = AttributeProtoHelper(node_deconv.proto)
    if not attrs_helper.has_attr('group'):
        raise RuntimeError("can not get the 'group' attr of {}".format(node_deconv.name))
    group = attrs_helper.get_attr_value('group')
    return group


def adjust_deconv_weight_shape(group, weights_array):
    """Adjust ConvTranspose weight shape to fit group param """
    weight_shape = weights_array.shape
    trans_common = [1, 0, 2, 3, 4]
    trans_shape = tuple(trans_common[:len(weight_shape)])
    if isinstance(weights_array, torch.Tensor):
        device = weights_array.device
        weights_array = weights_array.cpu().detach()
    if group == 1:
        weights_array = np.transpose(weights_array, trans_shape)
        if isinstance(weights_array, torch.Tensor):
            weights_array = weights_array.to(device)
        return weights_array

    new_shape = tuple([group, -1] + list(weight_shape)[1:])
    weights_array = np.reshape(weights_array, new_shape)
    trans_axes = (0, 2, 1, 3, 4)[:len(weight_shape) + 1]
    weights_array = np.transpose(weights_array, trans_axes)

    weight_shape = weights_array.shape
    new_shape = tuple([-1] + list(weight_shape)[2:])
    weights_array = np.reshape(weights_array, new_shape)

    if isinstance(weights_array, torch.Tensor):
        weights_array = weights_array.to(device)

    return weights_array


def adjust_axis_for_group_wise(axis, input_tensor):
    if axis == 0:
        return input_tensor
    """Adjust weight shape to fit group as first axis """
    dim_num = input_tensor.dim()
    return torch.transpose(input_tensor, 0, axis)


def adjust_conv_weight_shape(group, weight):
    """Adjust Conv weight shape to fit group param """
    weight_shape = weight.shape
    new_shape = tuple([group, -1] + list(weight_shape)[1:])
    weight = weight.reshape(new_shape)
    if len(weight.shape) == 5:
        weight = np.transpose(weight, (0, 2, 1, 3, 4))
    elif len(weight.shape) == 4:
        weight = np.transpose(weight, (0, 2, 1, 3))
    else:
        weight = np.transpose(weight, (0, 2, 1, 3, 4, 5))
    weight_shape = weight.shape
    new_shape = tuple([-1] + list(weight_shape)[2:])
    weight = weight.reshape(new_shape)
    return weight


def apply_lut_quantize_weight(weight, lut, group_size=256):
    """
    Function: quantize weight tensor with lut to introduce error 
    Parameter: weight: a torch.tensor, original weight (cout, cin)
               lut: a torch.tensor mapping index and quantized value (ceil(cin/group_size)*cout, 16)
               group_size: a integer by which weight is grouped on axis 1
    Return: weightq: a torch.tensor Quantized weight (cout, cin)
    """
    if lut is None:
        raise RuntimeError("lut table is None!")
    err = torch.full_like(weight, torch.inf)
    weightq = torch.zeros_like(weight)
    lut_ = lut.reshape(weight.shape[0], -1).to(weight.device)

    # weight get nearest center from all the 16 lut
    for idx in range(lut.shape[1]):
        cur_lut = lut_[:, idx:: lut.shape[1]].repeat_interleave(group_size, dim=1)
        cur_lut = cur_lut[:, : weight.shape[1]]
        cur_err = torch.abs(weight - cur_lut)
        mask = (cur_err < err)
        err[mask] = cur_err[mask]
        weightq[mask] = cur_lut[mask]

    return weightq
