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

from .....amct_pytorch.nn.module.quantization.conv1d import Conv1dQAT
from .....amct_pytorch.nn.module.quantization.conv2d import Conv2dQAT
from .....amct_pytorch.nn.module.quantization.conv_transpose_1d import ConvTranspose1dQAT
from .....amct_pytorch.nn.module.quantization.conv_transpose_2d import ConvTranspose2dQAT
from .....amct_pytorch.nn.module.quantization.conv3d import Conv3dQAT
from .....amct_pytorch.nn.module.quantization.linear import LinearQAT
from .....amct_pytorch.nn.module.quantization.quant_calibration_op import QuantCalibrationOp
from .....amct_pytorch.nn.module.quantization.lstm import LSTMQAT
from .....amct_pytorch.nn.module.quantization.gru import GRUQAT
from .....amct_pytorch.nn.module.quantization.matmul import MatMulQAT

__all__ = [Conv1dQAT, Conv2dQAT, ConvTranspose1dQAT, ConvTranspose2dQAT,
           Conv3dQAT, LinearQAT, QuantCalibrationOp, LSTMQAT, GRUQAT, MatMulQAT]