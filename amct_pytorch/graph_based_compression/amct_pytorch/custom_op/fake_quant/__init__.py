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
__all__ = [
    'FakeDeQuant',
    'FakeQuant',
    'FakeQuantizedConv',
    'FakeQuantizedConvTranspose',
    'FakeQuantizedLinear',
    'FakeQuantizedAvgPool2d',
    'FakeQuantConv2d',
    'FakeQuantLinear',
    'FakeMXQuantLinear',
    'FakeWeightQuantizedConv2d',
    'FakeWeightQuantizedLinear',
    'LutFakeWeightQuantizedLinear'
    ]

from ....amct_pytorch.custom_op.fake_quant.fake_dequant import FakeDeQuant
from ....amct_pytorch.custom_op.fake_quant.fake_quant import FakeQuant
from ....amct_pytorch.custom_op.fake_quant.fake_quantized_conv import FakeQuantizedConv
from ....amct_pytorch.custom_op.fake_quant.fake_quantized_convtranspose import FakeQuantizedConvTranspose
from ....amct_pytorch.custom_op.fake_quant.fake_quantized_linear import FakeQuantizedLinear
from ....amct_pytorch.custom_op.fake_quant.fake_quantized_avgpool2d import FakeQuantizedAvgPool2d
from ....amct_pytorch.custom_op.fake_quant.fake_quant_module import \
    FakeQuantConv2d, FakeQuantLinear
from ....amct_pytorch.custom_op.fake_quant.weight_fake_quant_module import \
    FakeWeightQuantizedConv2d, FakeWeightQuantizedLinear, LutFakeWeightQuantizedLinear
from ....amct_pytorch.custom_op.fake_quant.fake_mx_quant_module import FakeMXQuantLinear


FAKE_CONV_TRANSPOSE = "FakeQuantizedConvTranspose"
FAKE_LINEAR = "FakeQuantizedLinear"
FAKE_CONV = "FakeQuantizedConv"

FAKE_MODULES = [FAKE_CONV_TRANSPOSE,
                FAKE_LINEAR,
                FAKE_CONV]
