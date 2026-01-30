#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0
"""
from amct_pytorch.nn.module.quantization.conv1d import Conv1dQAT
from amct_pytorch.nn.module.quantization.conv2d import Conv2dQAT
from amct_pytorch.nn.module.quantization.conv_transpose_1d import ConvTranspose1dQAT
from amct_pytorch.nn.module.quantization.conv_transpose_2d import ConvTranspose2dQAT
from amct_pytorch.nn.module.quantization.conv3d import Conv3dQAT
from amct_pytorch.nn.module.quantization.linear import LinearQAT
from amct_pytorch.nn.module.quantization.quant_calibration_op import QuantCalibrationOp
from amct_pytorch.nn.module.quantization.lstm import LSTMQAT
from amct_pytorch.nn.module.quantization.gru import GRUQAT
from amct_pytorch.nn.module.quantization.matmul import MatMulQAT

__all__ = [Conv1dQAT, Conv2dQAT, ConvTranspose1dQAT, ConvTranspose2dQAT,
           Conv3dQAT, LinearQAT, QuantCalibrationOp, LSTMQAT, GRUQAT, MatMulQAT]
