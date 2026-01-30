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

init file

"""
from amct_pytorch.custom_op.fake_quant.fake_dequant import FakeDeQuant
from amct_pytorch.custom_op.fake_quant.fake_quant import FakeQuant
from amct_pytorch.custom_op.fake_quant.fake_quantized_conv import FakeQuantizedConv
from amct_pytorch.custom_op.fake_quant.fake_quantized_convtranspose import FakeQuantizedConvTranspose
from amct_pytorch.custom_op.fake_quant.fake_quantized_linear import FakeQuantizedLinear
from amct_pytorch.custom_op.fake_quant.fake_quantized_avgpool2d import FakeQuantizedAvgPool2d
from amct_pytorch.custom_op.fake_quant.fake_quant_module import \
    FakeQuantConv2d, FakeQuantLinear
from amct_pytorch.custom_op.fake_quant.weight_fake_quant_module import \
    FakeWeightQuantizedConv2d, FakeWeightQuantizedLinear, LutFakeWeightQuantizedLinear
from amct_pytorch.custom_op.fake_quant.fake_mx_quant_module import FakeMXQuantLinear


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

FAKE_CONV_TRANSPOSE = "FakeQuantizedConvTranspose"
FAKE_LINEAR = "FakeQuantizedLinear"
FAKE_CONV = "FakeQuantizedConv"

FAKE_MODULES = [FAKE_CONV_TRANSPOSE,
                FAKE_LINEAR,
                FAKE_CONV]
