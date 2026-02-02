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
from torch.autograd import Function

from ....amct_pytorch.utils.log import LOGGER
from ....amct_pytorch.custom_op.utils import check_quant_data
from ....amct_pytorch.custom_op.utils import check_tensor_balance_factor


class DMQBalancerFunction(Function):
    """
    Function: Run calibration process for quantization of the given layer.
    APIs: forward
    """
    @staticmethod
    def forward(ctx, inputs, weights, migration_strength):
        """
        Function: dmq_balancer foward funtion.
        """
        # check data's type and range
        check_quant_data(inputs, 'activation')
        check_quant_data(weights, 'weight')
        if inputs.dtype is torch.float16:
            inputs = inputs.to(dtype=torch.float)
        if weights.dtype is torch.float16:
            weights = weights.to(dtype=torch.float)

        tensor_balance_factor = dmq_balancer_forward(inputs, weights, migration_strength)

        check_tensor_balance_factor(tensor_balance_factor.cpu())

        return tensor_balance_factor


def dmq_balancer_forward(inputs, weights, migration_strength):
    """
    Function: smooth the input tensor by smooth_strength
    Parameters:
        inputs: activation of the given layer.
        weights: weights of the given layer.
        migration_strength: migration_strength of dmq balancer.
    Return: output_tensor: tensor_balance_factor
    """
    if migration_strength < 0.2 or migration_strength > 0.8:
        raise ValueError(f'migration_strength:{migration_strength} not support, should be in [0.2, 0.8]!')

    if inputs.abs().max() < torch.finfo(inputs.dtype).eps or \
        weights.abs().max() < torch.finfo(weights.dtype).eps:
        return torch.ones_like(inputs.shape[0], dtype=inputs.dtype, device=inputs.device)

    act_max = inputs.abs().amax(dim=1, keepdim=True)
    wts_max = weights.abs().amax(dim=1, keepdim=True)
    act_zero_mask = act_max < torch.finfo(act_max.dtype).eps
    wts_zero_mask = wts_max < torch.finfo(wts_max.dtype).eps
    
    tensor_balance_factor = torch.where(wts_zero_mask, torch.tensor(0, dtype=act_max.dtype, device=act_max.device),
                                        (act_max ** migration_strength) / (wts_max ** (1 - migration_strength)))

    dmq_zero_mask = tensor_balance_factor < torch.finfo(act_max.dtype).eps
    act_outlier = torch.where(dmq_zero_mask, torch.tensor(0, dtype=act_max.dtype, device=act_max.device), 
                              (act_max / tensor_balance_factor)).max()
    wts_outiler = torch.where(dmq_zero_mask, torch.tensor(0, dtype=act_max.dtype, device=act_max.device),
                              (wts_max * tensor_balance_factor)).max()

    tensor_balance_factor[act_zero_mask & wts_zero_mask] = torch.tensor(1.0, dtype=tensor_balance_factor.dtype,
                                                                        device=tensor_balance_factor.device)
    tensor_balance_factor[act_zero_mask & (~wts_zero_mask)] = wts_outiler / wts_max[act_zero_mask & (~wts_zero_mask)]
    tensor_balance_factor[(~act_zero_mask) & wts_zero_mask] = act_max[(~act_zero_mask) & wts_zero_mask] / act_outlier

    return tensor_balance_factor.reshape(-1)

