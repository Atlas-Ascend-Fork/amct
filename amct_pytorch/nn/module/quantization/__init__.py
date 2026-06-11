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
"""Public re-export of the QAT quantization modules.

The implementation lives under
``amct_pytorch.classic.graph_based.amct_pytorch.nn.module.quantization``.
This package exposes the documented import paths such as
``from amct_pytorch.nn.module.quantization.conv2d import Conv2dQAT``.
"""

__all__ = [
    'Conv2dQAT',
    'Conv3dQAT',
    'ConvTranspose2dQAT',
    'LinearQAT',
    'QuantCalibrationOp',
]

from .conv2d import Conv2dQAT
from .conv3d import Conv3dQAT
from .conv_transpose_2d import ConvTranspose2dQAT
from .linear import LinearQAT
from .quant_calibration_op import QuantCalibrationOp
