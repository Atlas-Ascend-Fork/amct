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

fake quant module and Function

"""

from torch import nn # pylint: disable=E0401
import torch # pylint: disable=E0401

class FakeQuant(nn.Module): # pylint: disable=R0903, R0902
    """
    Function: Customized torch.nn.Module of the fake quant operator.
    APIs: forward
    """
    def __init__(self, # pylint: disable=R0913
                 scale_d,
                 offset_d,
                 num_bits=8,
                 with_offset=True,
                 layer_name=None):
        super().__init__()
        self.layer_name = layer_name
        self.with_offset = with_offset
        self.scale_d = torch.tensor(1 / scale_d, requires_grad=False)
        self.offset_d = torch.tensor(offset_d, requires_grad=False)
        self.clamp_min = torch.tensor(
            -2**(num_bits - 1), requires_grad=False)
        self.clamp_max = torch.tensor(
            2**(num_bits - 1) - 1, requires_grad=False)

    def forward(self, inputs): # pylint: disable=W0221
        """
        Function: fake quant forward function.
        Inputs:
            inputs: intput data in torch.tensor.
        """
        device = inputs.device
        temp_data = torch.add(torch.round(
            torch.mul(inputs, self.scale_d.to(device))), self.offset_d.to(device))
        clamped_data = torch.clamp(temp_data, self.clamp_min.to(device), self.clamp_max.to(device))
        quantized_data = torch.sub(clamped_data, self.offset_d.to(device))
        return quantized_data


    def extra_repr(self):
        """ extra_repr for torch print
        """
        extra_s = (
            'scale_d={}, offset_d={}, clamp_min={}, clamp_max={}').format(
                self.scale_d, self.offset_d, self.clamp_min, self.clamp_max)
        return extra_s
