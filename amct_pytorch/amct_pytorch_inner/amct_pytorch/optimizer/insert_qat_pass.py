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
from ...amct_pytorch.configuration.distill_config import get_enable_quant_layers
from ...amct_pytorch.configuration.distill_config import get_quant_layer_config
from ...amct_pytorch.optimizer.base_module_fusion_pass import BaseModuleFusionPass
from ...amct_pytorch.nn.module.quantization.conv2d import Conv2dQAT
from ...amct_pytorch.nn.module.quantization.linear import LinearQAT
from ...amct_pytorch.utils.model_util import ModuleHelper
from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.utils.vars import DISTILL_TYPES


REPLACE_DICT = {
    'Conv2d': Conv2dQAT,
    'Linear': LinearQAT,
}


class InsertQatPass(BaseModuleFusionPass):
    """
    Function: Insert QAT mudule about compressed quantization.
    APIs: match_pattern, do_pass
    """
    def __init__(self, distill_config):
        """
        Function: init object
        Parameter: 
            distill_config: dict
        Return: None
        """
        BaseModuleFusionPass.__init__(self)
        self.conf = distill_config

    def match_pattern(self, module, name, graph=None):
        """
        Function: Match the module to be compressed in graph
        Parameters:
            module: module to be matched
            name: module's name
            graph: graph structure, not necessary
        Return: True: matched
                False: mismatch
        """
        if type(module).__name__ not in DISTILL_TYPES:
            return False
        if name not in get_enable_quant_layers(self.conf):
            return False

        return True

    def do_pass(self, model, object_module, object_name, graph=None):
        """
        Function: Do actual Insert QAT module
        Parameters: model: torch.nn.Module, the model to be modified.
                    object_module: module to process
                    object_name: name of object_module
                    graph: graph structure, not necessary
        Return: None
        """
        # Step2: construct a new module
        # activation quant config
        act_config = get_quant_layer_config(
            object_name, self.conf).get('distill_data_config')
        act_config['batch_num'] = self.conf.get('batch_num')
        # weights quant config
        wts_config = get_quant_layer_config(
            object_name, self.conf).get('distill_weight_config')

        # quant config
        config = {
            'distill': True,
            'retrain_data_config': act_config,
            'retrain_weight_config': wts_config,
        }

        # Step1: find module and its parent
        model_helper = ModuleHelper(model)
        parent_module = model_helper.get_parent_module(object_name)
        if REPLACE_DICT.get(type(object_module).__name__):
            qat_module = REPLACE_DICT.get(type(object_module).__name__).from_float(
                object_module, config)

            setattr(parent_module, object_name.split('.')[-1], qat_module)
            LOGGER.logd(
                "Insert QAT module to '{}' success!".format(object_name), 'InsertQatPass')
