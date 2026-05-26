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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = [
    'create_quant_config', 'quantize_preprocess', 'quantize_model', 'save_model',
    'create_quant_retrain_config', 'create_quant_retrain_model',
    'restore_quant_retrain_model', 'save_quant_retrain_model',
    'accuracy_based_auto_calibration', 'tensor_decompose',
    'create_prune_retrain_model', 'restore_prune_retrain_model', 'save_prune_retrain_model',
    'create_compressed_retrain_model', 'restore_compressed_retrain_model',
    'save_compressed_retrain_model',
    'ModelEvaluator', 'auto_channel_prune_search',
    'create_distill_config', 'create_distill_model', 'distill', 'save_distill_model',
    'create_quant_cali_config', 'create_quant_cali_model',
    ]

import os

from ..amct_pytorch.quantize_tool import create_quant_config
from ..amct_pytorch.quantize_tool import quantize_preprocess
from ..amct_pytorch.quantize_tool import quantize_model
from ..amct_pytorch.quantize_tool import save_model
from ..amct_pytorch.quantize_tool import create_quant_retrain_config
from ..amct_pytorch.quantize_tool import create_quant_retrain_model
from ..amct_pytorch.quantize_tool import restore_quant_retrain_model
from ..amct_pytorch.quantize_tool import save_quant_retrain_model
from ..amct_pytorch.accuracy_based_auto_calibration import accuracy_based_auto_calibration
from ..amct_pytorch import tensor_decompose
from ..amct_pytorch.prune_interface import create_prune_retrain_model
from ..amct_pytorch.prune_interface import restore_prune_retrain_model
from ..amct_pytorch.prune_interface import save_prune_retrain_model
from ..amct_pytorch.prune_interface import create_compressed_retrain_model
from ..amct_pytorch.prune_interface import restore_compressed_retrain_model
from ..amct_pytorch.prune_interface import save_compressed_retrain_model
from ..amct_pytorch.utils.evaluator import ModelEvaluator
from ..amct_pytorch.auto_channel_prune_search import auto_channel_prune_search
from ..amct_pytorch.distillation_interface import create_distill_config
from ..amct_pytorch.distillation_interface import create_distill_model
from ..amct_pytorch.distillation_interface import distill
from ..amct_pytorch.distillation_interface import save_distill_model
from ..amct_pytorch.quant_calibration_interface import create_quant_cali_config
from ..amct_pytorch.quant_calibration_interface import create_quant_cali_model
