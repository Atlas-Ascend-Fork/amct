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
import amct_pytorch.amct_pytorch_inner.amct_pytorch.utils as amct_utils
from ....amct_pytorch.common.utils.util import version_higher_than


VERSION = amct_utils.vars.find_torch_version()
SYMBOLIC_OP_NAME = "QuantIdentity"
if version_higher_than(VERSION, '1.5.0'):
    SYMBOLIC_OP_NAME = "custom_op_domain::QuantIdentity"


class QuantIdentity(Function):
    """Function to export onnx op with quantizable op name"""
    @staticmethod
    def forward(ctx, in_data, op_name, module_type):
        """QuantIdentity forward method"""
        return in_data.clone()

    @staticmethod
    def symbolic(g, *inputs):
        """QuantIdentity symbolic method"""
        return g.op(SYMBOLIC_OP_NAME, inputs[0], op_name_s=inputs[1], module_type_s=inputs[2])


class MarkedQuantizableModule(torch.nn.Module):
    """Custom Module to mark quantizable op"""
    def __init__(self, sub_module, layer_name):
        """MarkedQuantizableModule init method"""
        super().__init__()
        self.sub_module = sub_module
        self.layer_name = layer_name
        self.module_type = self.sub_module.__class__.__name__

    def forward(self, in_data, extra_input=None):
        """MarkedQuantizableModule forward method"""
        quant_identity = QuantIdentity.apply
        in_data = quant_identity(in_data, self.layer_name, self.module_type)
        if extra_input is None:
            out_data = self.sub_module(in_data)
        else:
            out_data = self.sub_module(in_data, extra_input)
        return out_data
