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

from ....amct_pytorch.utils.data_utils import convert_precision, cal_shared_exponent, scale_input_by_shared_exponents
from ....amct_pytorch.utils.vars import MXFP4_E2M1
from ....amct_pytorch.utils.weight_quant_utils import quant_dequant_weight


class FakeMXQuantLinear(nn.Module):
    """
    Function: class for fake mx_data quant operator inherited from nn.module.
    APIs: forward
    """
    def __init__(self, ori_module, quant_params):
        super().__init__()
        self.ori_dtype = ori_module.weight.dtype
        self.quant_params = quant_params
        shared_exponent = cal_shared_exponent(ori_module.weight.data, quant_params.get('wts_type'))
        weight = quant_dequant_weight(ori_module.weight.data, quant_params, scale=shared_exponent)
        self.register_buffer('quantized_weight', weight)
        self.act_type = quant_params.get('act_type')
        self.bias = ori_module.bias
 
    def forward(self, inputs):
        """
        Function: fake quantize process
        Inputs:
            inputs: intput data in torch.tensor.
        """
        if len(inputs.shape) < 2 or len(inputs.shape) > 6:
            raise RuntimeError("Only support activation dims in [2,6] for linear.")
        inputs = inputs.to(self.quantized_weight.device)
        shared_exponent = cal_shared_exponent(inputs, self.act_type)
        inputs = scale_input_by_shared_exponents(inputs, -1 * shared_exponent.to(inputs.dtype))
        inputs = convert_precision(inputs, self.act_type, 'RINT')
        inputs = scale_input_by_shared_exponents(inputs, shared_exponent.to(inputs.dtype))
        if self.bias is not None:
            bias = self.bias.to(torch.float32)
        else:
            bias = None
        output = nn.functional.linear(inputs.to(torch.float32), self.quantized_weight.to(torch.float32), bias)
        return output.to(self.ori_dtype)