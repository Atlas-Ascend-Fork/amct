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
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .....amct_pytorch.utils.log import LOGGER
from .....amct_pytorch.nn.module.quantization.qat_base import QATBase
from .....amct_pytorch.utils.vars import CHANNEL_WISE

RETRAIN_WEIGHT_CONFIG = 'retrain_weight_config'


class LinearQAT(nn.Linear, QATBase):
    """
    Function: Quantization module class after linear encapsulation.
    APIs: __init__, check_quantifiable, forward, from_float
    """
    _float_module = nn.Linear
    _required_params = ("in_features", "out_features", "bias")

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
        config=None
    ) -> None:
        """Init LinearQat amct op module"""

        nn.Linear.__init__(self, in_features, out_features, bias=bias)
        self.to(device, dtype)
        self.out_channels = out_features
        if config is None:
            config = dict()
        if not config.get(RETRAIN_WEIGHT_CONFIG):
            config = copy.deepcopy(config)
            config[RETRAIN_WEIGHT_CONFIG] = {CHANNEL_WISE: False}
        if config.get(RETRAIN_WEIGHT_CONFIG).get(CHANNEL_WISE) is None:
            config = copy.deepcopy(config)
            config[RETRAIN_WEIGHT_CONFIG][CHANNEL_WISE] = False

        QATBase.__init__(self, 'Linear', device=device, config=config)

    @classmethod
    def from_float(cls, mod, config=None):
        """
        Create a qat module from a float module
        Args: `mod` a float module, 'config' amct op quant config
        """
        if not isinstance(mod, cls._float_module):
            raise RuntimeError(f'{cls.__name__}.from_float can only works for '
                               f'{cls._float_module.__name__}')

        qat_linear = cls(
            mod.in_features,
            mod.out_features,
            bias=mod.bias is not None,
            config=config
        )

        setattr(qat_linear, 'weight', mod.weight)
        setattr(qat_linear, 'bias', mod.bias)
        qat_linear.to(mod.weight.device)
        LOGGER.logi(
            f'Convert {cls._float_module.__name__} to QAT op successfully.')
        return qat_linear

    def check_quantifiable(self):
        """check qat config for LinearQat"""
        if self.retrain_weight_config.get(CHANNEL_WISE, True):
            raise RuntimeError('Do not support Linear with channel_wise.')
        return True

    def forward(self, input):
        """
        Defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        quantized_acts, quantized_wts = self.forward_qat(input)

        with torch.enable_grad():
            output = F.linear(quantized_acts, quantized_wts, self.bias)

        return output
