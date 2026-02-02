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
import torch.nn.functional as F

from .....amct_pytorch.utils.log import LOGGER
from .....amct_pytorch.nn.module.quantization.qat_base import QATBase

SUPPORTED_DATA_DIMS = 5


class Conv3dQAT(nn.Conv3d, QATBase):
    """
    Function: Quantization module class after conv3d encapsulation.
    APIs: __init__, check_quantifiable, forward, from_float
    """
    _float_module = nn.Conv3d
    _required_params = ("in_channels", "out_channels", "kernel_size", "stride",
                         "padding", "dilation", "groups", "bias", "padding_mode")

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        config=None
    ) -> None:
        """Init Conv3dQat amct op module"""

        nn.Conv3d.__init__(self, in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias, padding_mode=padding_mode)
        self.to(device, dtype)
        if config is None:
            config = dict()

        QATBase.__init__(self, 'Conv3d', device=device, config=config)

    @classmethod
    def from_float(cls, mod, config=None):
        """
        Create a qat module from a float module
        Args: `mod` a float module, 'config' amct op quant config
        """
        if not isinstance(mod, cls._float_module):
            raise RuntimeError(f'{cls.__name__}.from_float can only works for '
                               f'{cls._float_module.__name__}')

        qat_conv3d = cls(
            mod.in_channels,
            mod.out_channels,
            mod.kernel_size,
            stride=mod.stride,
            padding=mod.padding,
            dilation=mod.dilation,
            groups=mod.groups,
            bias=mod.bias is not None,
            padding_mode=mod.padding_mode,
            config=config
        )

        setattr(qat_conv3d, 'weight', mod.weight)
        setattr(qat_conv3d, 'bias', mod.bias)
        qat_conv3d.to(mod.weight.device)
        LOGGER.logi(
            f'Convert {cls._float_module.__name__} to QAT op successfully.')
        return qat_conv3d

    def check_quantifiable(self):
        """check qat config for Conv3dQat"""
        if self.padding_mode != 'zeros':
            raise RuntimeError(f'Do not support Conv3d with padding_mode {self.padding_mode}')

        if len(self.dilation) != 3 or self.dilation[0] != 1:
            raise RuntimeError(f'Only support Conv3d with dilation[0] 1, current is {self.dilation[0]}')
        return True

    def forward(self, inputs):
        """
        Defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        if inputs.dim() != SUPPORTED_DATA_DIMS:
            raise RuntimeError(f"Only {SUPPORTED_DATA_DIMS}-dimensional input data is supported.")
        quantized_acts, quantized_wts = self.forward_qat(inputs)

        with torch.enable_grad():
            output = F.conv3d(
                quantized_acts,
                quantized_wts,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups)

        return output
