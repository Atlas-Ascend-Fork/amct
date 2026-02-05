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
import numpy as np
from torch import nn # pylint: disable=E0401
import torch # pylint: disable=E0401

S16_BASE = 16
S32_BASE = 32


class FakeDeQuant(nn.Module): # pylint: disable=R0903, R0902
    """
    Function: Customized torch.nn.Module of the fake dequant operator.
    APIs: forward
    """
    def __init__(self, # pylint: disable=R0913
                 scale_d,
                 scale_w,
                 deq_shape,
                 num_bits=8,
                 layer_name=None):
        super().__init__()
        self.layer_name = layer_name
        self.deq_shape = deq_shape
        self.num_bits = num_bits
        self.deq_scale = torch.tensor(scale_w * scale_d, requires_grad=False)

    def forward(self, inputs): # pylint: disable=W0221
        """
        Function: fake dequant forward function.
        Inputs:
            inputs: intput data in torch.tensor.
        """
        device = inputs.device
        deq_scale = self.deq_scale.reshape(self.deq_shape)
        deq_scale = deq_scale.to(device)
        dequantized_data = torch.mul(inputs, deq_scale)

        return dequantized_data
