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

from .fake_quant import FakeQuant
from .fake_dequant import FakeDeQuant


class FakeQuantizedConv(nn.Module):
    """
    Function: Customized torch.nn.Module of the fake quantized conv2d operator.
    APIs: forward
    """
    def __init__(self,
                 sub_module,
                 quant_params,
                 layer_name,
                 num_bits=8):
        super().__init__()
        self.sub_module = sub_module
        self.layer_name = layer_name
        self.quant_params = quant_params
        self.fake_quant = FakeQuant(
            scale_d=quant_params['data_scale'],
            offset_d=quant_params['data_offset'],
            num_bits=num_bits,
            layer_name=layer_name)
        dequant_shape = [1] * len(self.sub_module.weight.shape)
        dequant_shape[1] = -1

        self.fake_dequant = FakeDeQuant(
            scale_d=quant_params['data_scale'],
            scale_w=quant_params['weight_scale'],
            deq_shape=dequant_shape,
            layer_name=layer_name)

    def forward(self, inputs):
        """
        Function: fake quantized conv forward function.
        Inputs:
            inputs: intput data in torch.tensor.
        """
        int8_out = self.fake_quant(inputs)
        int32_out = self.sub_module(int8_out)
        dequant_out = self.fake_dequant(int32_out).to(inputs.dtype)
        return dequant_out
