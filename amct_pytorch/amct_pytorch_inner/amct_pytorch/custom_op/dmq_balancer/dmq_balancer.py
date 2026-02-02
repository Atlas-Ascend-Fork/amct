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
from torch import nn
import torch

from ....amct_pytorch.custom_op.dmq_balancer.dmq_balancer_func import DMQBalancerFunction
from ....amct_pytorch.utils.module_info import ModuleInfo
from ....amct_pytorch.utils.log import LOGGER


class DMQBalancer(nn.Module):
    """
    Function: Run calibration process for dmq_balancer of the given layer.
    APIs: forward
    """
    act_channel_dims = {
        'Conv2d': 1,
        'Conv3d': 1,
        'ConvTranspose2d': 1,
        'ConvTranspose1d': 1,
        'Linear': -1,
        'Conv1d': 1,
    }

    def __init__(self, module, record_module, migration_strength, layers_name):
        super().__init__()
        self.replaced_module = module
        self.record_module = record_module
        self.migration_strength = migration_strength
        self.layers_name = layers_name
        self.dmq_algo_name = 'dmq_balancer'

    def forward(self, inputs):
        """
        Function: DMQBalancer foward.
        """
        weights = self.replaced_module.weight
        sub_out = self.replaced_module(inputs)

        # transpose activation and weight to c-first format
        module_type = type(self.replaced_module).__name__
        act_channel_dim = self.act_channel_dims.get(module_type)
        act_channel_num = inputs.shape[act_channel_dim]
        inputs = inputs.transpose(0, act_channel_dim)

        weights = self._transpose_wts_for_dmq_balancer(module_type, weights)
        wts_channel_num = weights.shape[0]

        input_data = inputs.reshape([inputs.shape[0], -1])
        input_weight = weights.reshape([weights.shape[0], -1])

        if act_channel_num != wts_channel_num:
            raise ValueError(
                "the activation channel_num[{}] of {} must equal to weight channel_num[{}]"
                .format(act_channel_num, self.layers_name, wts_channel_num))

        tensor_balance_factor = DMQBalancerFunction.apply(
            input_data, input_weight, self.migration_strength)

        # save tensor_balance_factor to record_module
        self.record_module(self.layers_name, self.dmq_algo_name,
                           {'tensor_balance_factor': tensor_balance_factor.cpu().tolist()})

        LOGGER.logi("Do layer {} dmq_balancer calibration succeeded!"
                    .format(self.layers_name), 'DMQBalancer')

        return sub_out

    def _transpose_wts_for_dmq_balancer(self, module_type, weights):
        """ transpose activation and weight to c-first format """
        _, wts_cin_axis = ModuleInfo.get_wts_cout_cin(self.replaced_module)
        if module_type in ('Conv1d', 'Conv2d', 'Conv3d') and self.replaced_module.groups > 1:
            group = self.replaced_module.groups
            weight_shape = weights.shape
            new_shape = tuple([group, -1] + list(weight_shape)[1:])
            weights = weights.reshape(new_shape).transpose(2, 1)
            weight_shape = weights.shape
            new_shape = tuple([-1] + list(weight_shape)[2:])
            weights = weights.reshape(new_shape)
        else:
            weights = weights.transpose(0, wts_cin_axis)

        return weights




