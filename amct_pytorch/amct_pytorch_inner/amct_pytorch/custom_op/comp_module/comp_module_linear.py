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

Quantization module after linear encapsulation

"""
import torch # pylint: disable=E0401
import torch.nn.functional as F # pylint: disable=E0401

from amct_pytorch.custom_op.comp_module.comp_module_base \
    import CompModuleBase


class CompModuleLinear(CompModuleBase): # pylint: disable=R0903
    """
    Function: Quantization module class after linear encapsulation.
    APIs: __init__, forward
    """
    def __init__(self, *args, **kwargs):
        super(CompModuleLinear, self).__init__(*args, **kwargs)
        self.num_scales = 1
        self._init_output()

    def forward(self, inputs):
        """
        Defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        compressed_inputs, compressed_weights = \
            super(CompModuleLinear, self).forward(inputs)

        # Forward
        with torch.enable_grad():
            output = F.linear(compressed_inputs, compressed_weights,
                              self.replaced_module.bias)

        return output
