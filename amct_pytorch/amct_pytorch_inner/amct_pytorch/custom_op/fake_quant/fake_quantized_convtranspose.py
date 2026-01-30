#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

fake quantized convtranspose module and Function

"""

from torch import nn

from .fake_quant import FakeQuant
from .fake_dequant import FakeDeQuant


class FakeQuantizedConvTranspose(nn.Module):
    """
    Function: Customized torch.nn.Module of the fake quantized ConvTranspose operator.
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
        dequant_shape = [1] * len(self.sub_module.weight.shape)
        dequant_shape[1] = -1

        self.fake_dequant = FakeDeQuant(
            scale_d=quant_params['data_scale'],
            scale_w=quant_params['weight_scale'],
            deq_shape=dequant_shape,
            layer_name=layer_name)
        self.fake_quant = FakeQuant(
            scale_d=quant_params['data_scale'],
            offset_d=quant_params['data_offset'],
            num_bits=num_bits,
            layer_name=layer_name)


    def forward(self, inputs): # pylint: disable=W0221
        """
        Function: fake quantized ConvTranspose forward function.
        Inputs:
            inputs: intput data in torch.tensor.
        """
        int8_out = self.fake_quant(inputs)
        int32_out = self.sub_module(int8_out)
        dequant_out = self.fake_dequant(int32_out).to(inputs.dtype)
        return dequant_out
