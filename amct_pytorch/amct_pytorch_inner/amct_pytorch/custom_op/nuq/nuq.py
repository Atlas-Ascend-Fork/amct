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

from amct_pytorch.custom_op import nuq_cali
from amct_pytorch.custom_op.utils import check_quant_data
from amct_pytorch.common.utils import vars_util

__all__ = ['weight_nuq_cali_tensor']


def weight_nuq_cali_tensor(weight_tensor, wts_param):
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
    if int(wts_param.get('num_bits')) == vars_util.INT4_BIT:
        raise RuntimeError('quantization algorithm "NUQuantize" cannot supprot 4 bit.')
    # data's shape should be [channle_out, -1]
    if wts_param.get('wts_algo') == 'nuq_quantize':
        return weight_nuq(weight_tensor, wts_param)

    raise RuntimeError("Unsupport wts_algo %s " % (wts_param.get('wts_algo')))


def weight_nuq(weight_tensor, wts_param):
    """
    Function: Do calibration on weight_tensor with 'NUQ' algorithm
    Inputs:
        weight_tensor: torch.tensor, weight to be calibrated
        wts_param: a dict, parameters for calibration
        module_type: a string, type of layer weight_tensor attached to.
    Returns:
        scale_list: a 1-dim list, scale of weight_tensor
        offset_list: a 1-dim list, offset of weight_tensor
        calied_weight: torch.tensor, calibrated weight
    """
    # only 2-D tensor is valid for nuq_cpp.cali()
    convert_flag = False
    if weight_tensor.dtype is torch.float16:
        convert_flag = True
        weight_tensor = weight_tensor.to(dtype=torch.float)
    need_reshape = weight_tensor.dim() > 2
    if need_reshape:
        weight_size = weight_tensor.size()
        weight_tensor = weight_tensor.reshape([weight_size[0], -1])

    check_quant_data(weight_tensor, 'weight')
    if weight_tensor.numel() == 0:
        raise RuntimeError("Weight tensor length should not be 0!")

    is_ok, scale, offset, calied_weight = nuq_cali(
        weight_tensor, int(wts_param.get('num_bits')),
        wts_param.get('with_offset'), int(wts_param.get('num_steps')),
        int(wts_param.get('num_of_iteration')), need_reshape)
    if is_ok != 0:
        raise RuntimeError("Weight calibration with NUQ failed!")

    scale_list = scale.cpu().numpy().tolist()
    offset_list = offset.cpu().numpy().tolist()
    # restore shape
    if need_reshape:
        weight_tensor = weight_tensor.reshape(weight_size)
        calied_weight = calied_weight.reshape(weight_size)
    if convert_flag:
        calied_weight = calied_weight.to(dtype=torch.float16)
    return scale_list, offset_list, calied_weight.to(weight_tensor.device)
