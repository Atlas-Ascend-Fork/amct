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
import torch.nn as nn
from torch.nn import functional as F

from .....amct_pytorch.nn.module.quantization.qat_base import QATBase

SUPPORTED_DATA_DIMS = 3


class Conv1dQAT(nn.Conv1d, QATBase):
    _float_module = nn.Conv1d
    _required_params = ("in_channels", "out_channels", "kernel_size", "stride",
                         "padding", "dilation", "groups", "bias", "padding_mode")
 
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 device=None,
                 dtype=None,
                 config=None):
        nn.Conv1d.__init__(self, in_channels, out_channels, kernel_size,
                           stride=stride, padding=padding, dilation=dilation,
                           groups=groups, bias=bias, padding_mode=padding_mode)
        if dtype not in [torch.float32, None]:
            raise ValueError(f'Only support dtype torch.float32, but your input dtype is {dtype}')
        self.to(device, dtype)
        QATBase.__init__(self, 'Conv1d', device=device, config=config)

    def check_quantifiable(self):
        if self.retrain_enable and self.padding_mode != 'zeros':
            raise ValueError(f'Do not support Conv1d with padding mode {self.padding_mode}')

    def forward(self, inputs):
        if inputs.dim() != SUPPORTED_DATA_DIMS:
            raise RuntimeError(f"Only {SUPPORTED_DATA_DIMS}-dimensional input data is supported.")
        quantized_acts, quantized_wts = self.forward_qat(inputs)
        output = F.conv1d(
            quantized_acts,
            quantized_wts,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )
        return output