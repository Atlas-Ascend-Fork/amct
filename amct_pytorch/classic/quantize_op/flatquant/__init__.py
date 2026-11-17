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
"""FlatQuant algorithm package: generic parts + per-model subpackages.

The package root holds the generic parts (config parser, fake-quant
linear, transforms, quantizers, calibrator); qwen3/ and clip/ hold the
per-model reconstruction modules."""

from .flat_config import FlatConfig, flat_config_parser
from .flat_linear import FlatQuantizedLinear
from .trans_utils import (
    SVDSingleTransMatrix,
    SVDDecomposeTransMatrix,
)
from .quantizer import ActivationQuantizer, WeightQuantizer
from .cali_flatquant import calibrate_flatquant, collect_fp_reference, train_block
from .flat_utils import save_flat_matrices, load_flat_matrices

__all__ = [
    'FlatConfig',
    'flat_config_parser',
    'FlatQuantizedLinear',
    'SVDSingleTransMatrix',
    'SVDDecomposeTransMatrix',
    'ActivationQuantizer',
    'WeightQuantizer',
    'calibrate_flatquant',
    'collect_fp_reference',
    'train_block',
    'save_flat_matrices',
    'load_flat_matrices',
]
