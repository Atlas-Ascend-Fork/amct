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
from .....amct_pytorch.common.utils.util import version_higher_than

SUPPORTED_DATA_DIMS = 4


class ConvTranspose2dQAT(nn.ConvTranspose2d, QATBase):
    _float_module = nn.ConvTranspose2d
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
        nn.ConvTranspose2d.__init__(self, in_channels, out_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation,
                                    groups=groups, bias=bias, padding_mode=padding_mode)
        QATBase.__init__(self, 'ConvTranspose2d',
                         device=device, config=config)

    def check_quantifiable(self):
        if self.retrain_enable and self.padding_mode != 'zeros':
            raise ValueError(f'Do not support ConvTranspose2d with padding mode {self.padding_mode}')

    def forward(self, inputs, output_size=None):
        if inputs.dim() != SUPPORTED_DATA_DIMS:
            raise RuntimeError(f"Only {SUPPORTED_DATA_DIMS}-dimensional input data is supported.")
        # num_spatial_dims is module's kernel_size
        num_spatial_dims = 2
        quantized_acts, quantized_wts = self.forward_qat(inputs)
        if version_higher_than(torch.__version__, '1.12.0'):
            output_padding = self._output_padding(inputs, output_size,
                                                  self.stride, self.padding, self.kernel_size,
                                                  num_spatial_dims, self.dilation)
        elif version_higher_than(torch.__version__, '1.7.0'):
            output_padding = self._output_padding(inputs, output_size,
                                                  self.stride, self.padding, self.kernel_size, self.dilation)
        else:
            output_padding = self._output_padding(inputs, output_size,
                                                  self.stride, self.padding, self.kernel_size)
        output = F.conv_transpose2d(
            quantized_acts,
            quantized_wts,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation
        )
        return output
