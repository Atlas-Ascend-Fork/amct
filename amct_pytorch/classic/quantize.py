# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
__all__ = ['quantize', 'convert', 'algorithm_register']

from torch import nn

from amct_pytorch.algorithms import AlgorithmRegistry
from amct_pytorch.classic.optimizer import (InsertQuantizeModulePass,
                                                ModelOptimizer,
                                                ReplaceNpuQuantModulePass)
from amct_pytorch.quantize_op.base_quant_module import BaseQuantizeModule
from amct_pytorch.common.config import parse_config, set_default_config
from amct_pytorch.common.utils.check_params import check_params


@check_params(model=nn.Module,
              config=(dict, type(None)))
def quantize(model, config=None):
    """
    Function: Modify user's model for quantization
    Parameter: model: user mode instance of Torch.nn.Module
               config: simply config dict from user to set
    Return: None
    """
    if config is None:
        config = set_default_config()

    layer_config = parse_config(model, config, AlgorithmRegistry)

    optimizer = ModelOptimizer()
    optimizer.add_pass(InsertQuantizeModulePass(layer_config))
    optimizer.do_optimizer(model)


@check_params(model=nn.Module)
def convert(model):
    """
    Function: Convert quantized calibration model to quantized deployment model
    Parameter: model: quantized calibration model instance of Torch.nn.Module
    Return: None
    """
    optimizer = ModelOptimizer()
    optimizer.add_pass(ReplaceNpuQuantModulePass())
    optimizer.do_optimizer(model)


@check_params(name=str,
              src_op=str,
              quant_op=BaseQuantizeModule,
              deploy_op=(nn.Module, type(None)))
def algorithm_register(name, src_op, quant_op, deploy_op=None):
    """
    Function: quantization algorithm registration
    Parameter: name: str, algorithm's name
               src_op: str, Source operators corresponding to the algorithm
               quant_op: nn.module or type, Quantized operators corresponding to the algorithm
               deploy_op: nn.module or type, Deployment operators corresponding to the algorithm
    Return: None
    """
    AlgorithmRegistry.register(name, src_op, quant_op, deploy_op=deploy_op)
