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
__all__ = ['weight_cali_tensor', 'weight_quant_np']

import torch

from ....amct_pytorch.custom_op import arq_cali_pytorch
from ....amct_pytorch.custom_op import arq_real_pytorch
from ....amct_pytorch.custom_op.utils import check_quant_data
from ....amct_pytorch.custom_op.utils import check_scale_offset


def weight_cali_tensor(weight_tensor, wts_param):
    """
    Function: Do calibration on weight_tensor
    Inputs:
        weight_tensor: torch.tensor, weight to be calibrated
        wts_param: a dict, parameters for calibration
        module_type: a string, type of layer weight_tensor attached to.
    Returns:
        scale_list: a 1-dim list, scale of weight_tensor
        offset_list: a 1-dim list, offset of weight_tensor
        calied_weight: torch.tensor, calibrated weight
    """
    # data's shape should be [channle_out, -1]
    if wts_param.get('wts_algo') == 'arq_quantize':
        return weight_arq(weight_tensor, wts_param)

    raise RuntimeError("Unsupport wts_algo %s " % (wts_param.get('wts_algo')))


def weight_arq(weight_tensor, wts_param):
    """
    Function: Do calibration on weight_tensor with 'ARQ' algorithm
    Inputs:
        weight_tensor: torch.tensor, weight to be calibrated
        wts_param: a dict, parameters for calibration
        module_type: a string, type of layer weight_tensor attached to.
    Returns:
        scale_list: a 1-dim list, scale of weight_tensor
        offset_list: a 1-dim list, offset of weight_tensor
        calied_weight: torch.tensor, calibrated weight
    """
    check_quant_data(weight_tensor, 'weight')
    convert_flag = False
    if weight_tensor.dtype is torch.float16:
        convert_flag = True
        weight_tensor = weight_tensor.to(dtype=torch.float)

    scale, offset, calied_weight = arq_cali_pytorch(
        weight_tensor, wts_param.get('num_bits'),
        wts_param.get('channel_wise'), wts_param.get('with_offset'))

    scale_list = scale.cpu().numpy().tolist()
    offset_list = offset.cpu().numpy().tolist()
    if convert_flag:
        calied_weight = calied_weight.to(dtype=torch.float16)
    return scale_list, offset_list, calied_weight


def weight_quant_np(weight_np, scale_list, offset_list, num_bit):
    """
    Function: Quant weight_tensor from float32 to int8
    Inputs:
        weight_np: np array, weight to be quant
        scale_list: a list, scale of weight_np
        offset_list: a list, offset of weight_np
        num_bit: a int number, bit the weight will be quant to
        module_type: a string, type of layer weight_tensor attached to.
    Returns:
        int_weight_np: np array, the weight of int8
    """
    device = 'cpu'
    weight_tensor = torch.from_numpy(weight_np).to(device)
    scale = torch.tensor(scale_list).reshape([-1]).to(device)
    offset = torch.tensor(offset_list).reshape([-1]).to(device)

    check_quant_data(weight_tensor, 'weight')
    check_scale_offset(scale, offset)

    int_weight_tensor = arq_real_pytorch(
        weight_tensor, scale, offset.to(torch.int32), num_bit)

    int_weight_np = int_weight_tensor.numpy()

    return int_weight_np
