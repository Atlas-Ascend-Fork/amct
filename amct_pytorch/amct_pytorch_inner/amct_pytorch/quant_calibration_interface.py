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
import os

import torch
from torch import nn

import amct_pytorch.amct_pytorch_inner.amct_pytorch.optimizer as opt
from ..amct_pytorch.utils.log import LOGGER
from ..amct_pytorch.common.utils.check_params import check_params
from ..amct_pytorch.common.utils import files as files_util
from ..amct_pytorch.utils.model_util import ModuleHelper
from ..amct_pytorch.utils.singleton_record import SingletonScaleOffsetRecord
from ..amct_pytorch.configuration.quant_calibration_config \
    import inner_create_quant_calibration_config
from ..amct_pytorch.configuration.quant_calibration_config import parse_cali_quant_config
from ..amct_pytorch.custom_op.recorder.recorder import Recorder


@check_params(config_file=str,
              model=nn.Module,
              quant_layers=(type(None), dict),
              config_defination=(type(None), str))
def create_quant_cali_config(config_file,
                             model,
                             quant_layers=None,
                             config_defination=None):
    """
    Function: Create quantize configuration json file for amct_pytorch tool
    Parameter: config_file: file path of quantize configuration json file
               model: user mode instance of Torch.nn.Module
               quant_layers: layers need to be quantized which is distinguished on quant method
               config_defination: simply config file from user to set
    Return: None
    """
    files_util.is_valid_name(config_file, 'config_file')
    config_file = files_util.create_empty_file(config_file, check_exist=True)

    ModuleHelper(model).check_amct_op()
    inner_create_quant_calibration_config(config_file, model, quant_layers, config_defination)
    LOGGER.logi('Create quant config file {} success.'.format(config_file))


@check_params(config_file=str,
              record_file=str,
              model=nn.Module)
def create_quant_cali_model(config_file, record_file, model):
    """
    Function: Modify user's model for calibration in inference process.
    Parameter: config_file: quantize configuration json file
               record_file: temporary file to store scale and offset
               model: user pytorch model's model file
    Return: model: modified pytorch model for calibration inference.
    """
    files_util.is_valid_name(config_file, 'config_file')
    files_util.is_valid_name(record_file, 'record_file')
    config_file = os.path.realpath(config_file)
    record_file = files_util.create_empty_file(record_file, check_exist=True)

    ModuleHelper(model).check_amct_op()
    cali_quant_config = parse_cali_quant_config(config_file, model)
 
    recorder = Recorder(record_file)
    optimizer = opt.ModelOptimizer()
    optimizer.add_pass(opt.InsertKVCacheQuantPass(recorder, cali_quant_config))
    optimizer.do_optimizer(model, graph=None)
    return model