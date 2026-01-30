#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

fake dequant module and Function

"""
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
