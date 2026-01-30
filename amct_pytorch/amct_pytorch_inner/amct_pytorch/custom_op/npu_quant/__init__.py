#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.

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
from amct_pytorch.custom_op.npu_quant.npu_quantization_linear import NpuQuantizationLinear
from amct_pytorch.custom_op.npu_quant.npu_quantization_conv2d import NpuQuantizationConv2d
from amct_pytorch.custom_op.npu_quant.npu_mx_quantization_linear import NpuMXQuantizationLinear
from amct_pytorch.custom_op.npu_quant.weight_npu_quant_module import NpuWeightQuantizedLinear

__all__ = [
    'NpuQuantizationLinear',
    'NpuQuantizationConv2d',
    'NpuMXQuantizationLinear',
    'NpuWeightQuantizedLinear',
]
