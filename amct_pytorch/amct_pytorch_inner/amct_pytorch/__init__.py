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
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from amct_pytorch.quantize_tool import create_quant_config
from amct_pytorch.quantize_tool import quantize_preprocess
from amct_pytorch.quantize_tool import quantize_model
from amct_pytorch.quantize_tool import save_model
from amct_pytorch.quantize_tool import create_quant_retrain_config
from amct_pytorch.quantize_tool import create_quant_retrain_model
from amct_pytorch.quantize_tool import restore_quant_retrain_model
from amct_pytorch.quantize_tool import save_quant_retrain_model
from amct_pytorch.accuracy_based_auto_calibration import accuracy_based_auto_calibration
from amct_pytorch import tensor_decompose
from amct_pytorch.prune_interface import create_prune_retrain_model
from amct_pytorch.prune_interface import restore_prune_retrain_model
from amct_pytorch.prune_interface import save_prune_retrain_model
from amct_pytorch.prune_interface import create_compressed_retrain_model
from amct_pytorch.prune_interface import restore_compressed_retrain_model
from amct_pytorch.prune_interface import save_compressed_retrain_model
from amct_pytorch.utils.evaluator import ModelEvaluator
from amct_pytorch.auto_channel_prune_search import auto_channel_prune_search
from amct_pytorch.distillation_interface import create_distill_config
from amct_pytorch.distillation_interface import create_distill_model
from amct_pytorch.distillation_interface import distill
from amct_pytorch.distillation_interface import save_distill_model
from amct_pytorch.quant_calibration_interface import create_quant_cali_config
from amct_pytorch.quant_calibration_interface import create_quant_cali_model
from amct_pytorch.lm_compression.quantize import quantize, convert, algorithm_register
from amct_pytorch.lm_compression.config import INT4_AWQ_WEIGHT_QUANT_CFG, INT4_GPTQ_WEIGHT_QUANT_CFG
from amct_pytorch.lm_compression.config import INT8_SMOOTHQUANT_CFG, INT8_MINMAX_WEIGHT_QUANT_CFG

__all__ = [
    'create_quant_config', 'quantize_preprocess', 'quantize_model', 'save_model',
    'create_quant_retrain_config', 'create_quant_retrain_model',
    'restore_quant_retrain_model', 'save_quant_retrain_model',
    'accuracy_based_auto_calibration', 'tensor_decompose', '__version__',
    'create_prune_retrain_model', 'restore_prune_retrain_model', 'save_prune_retrain_model',
    'create_compressed_retrain_model', 'restore_compressed_retrain_model',
    'save_compressed_retrain_model',
    'ModelEvaluator', 'auto_channel_prune_search',
    'create_distill_config', 'create_distill_model', 'distill', 'save_distill_model',
    'create_quant_cali_config', 'create_quant_cali_model',
    'quantize', 'convert', 'algorithm_register', 
    'INT4_AWQ_WEIGHT_QUANT_CFG', 'INT4_GPTQ_WEIGHT_QUANT_CFG', 'INT8_SMOOTHQUANT_CFG', 'INT8_MINMAX_WEIGHT_QUANT_CFG'
    ]

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

with open(os.path.join(CUR_DIR, '.version')) as fid:
    __version__ = fid.readlines()[0].strip()
