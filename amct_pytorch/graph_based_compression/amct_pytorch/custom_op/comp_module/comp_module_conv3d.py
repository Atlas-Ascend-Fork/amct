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
import torch.nn.functional as F

from ....amct_pytorch.custom_op.comp_module.comp_module_base import CompModuleBase


class CompModuleConv3d(CompModuleBase):
    """
    Function: Quantization module class after conv3d encapsulation.
    APIs: __init__, forward
    """
    def __init__(self, *args, **kwargs):
        super(CompModuleConv3d, self).__init__(*args, **kwargs)
        if not self.wts_config.get('channel_wise'):
            self.num_scales = 1
        else:
            self.num_scales = self.replaced_module.weight.size(0)
        self._init_output()

    def forward(self, inputs):
        """
        Defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        compressed_inputs, compressed_weights = \
            super(CompModuleConv3d, self).forward(inputs)

        # amct_pytorch only support Conv3d with padding_mode 'zeros'
        padding_mode = self.replaced_module.padding_mode
        if 'quant' in self.comp_algs and padding_mode != 'zeros':
            raise RuntimeError('Do not support Conv3d with padding mode {}!'.format(padding_mode))

        # Forward
        with torch.enable_grad():
            output = F.conv3d(
                compressed_inputs, compressed_weights,
                self.replaced_module.bias, self.replaced_module.stride,
                self.replaced_module.padding, self.replaced_module.dilation,
                self.replaced_module.groups)

            return output
