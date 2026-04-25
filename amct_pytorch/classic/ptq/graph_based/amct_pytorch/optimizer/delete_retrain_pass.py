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
from ...amct_pytorch.optimizer.base_module_fusion_pass \
    import BaseModuleFusionPass

from ...amct_pytorch.utils.model_util import ModuleHelper
from ...amct_pytorch.utils.log import LOGGER


class DeleteRetrainPass(BaseModuleFusionPass):
    """
    Function: Delete some module about retrain quantization
    APIs: match_pattern, do_pass
    """
    def __init__(self):
        """
        Function: init object
        Parameter: None
        Return: None
        """
        BaseModuleFusionPass.__init__(self)

    @staticmethod
    def match_pattern(module, name, graph=None):
        """
        Function:Match the module quantized in model.
        Parameters:
            module: module to be matched
            name: module's name
            graph: graph structure, not necessary
        Return: True: matched
                False: mismatch
        """
        if module.type not in ['RetrainQuant', 'RNNRetrainQuant']:
            return False

        return True

    @staticmethod
    def do_pass(model, object_module, object_name, graph=None):
        """
        Function: Delete some module about retrain quantization.
        Parameters: model: torch.nn.Module, the model to be modified.
                    object_module: module to process
                    object_name: name of object_module
                    graph: graph structure, not necessary
        Return: None
        """
        # Step1: find module and its parent
        model_helper = ModuleHelper(model)
        parent_node = model_helper.get_parent_module(object_name)

        # Step2: replace new model
        setattr(parent_node, object_name.split('.')[-1],
                object_module.quant_module.replaced_module)
        bn_module_name = object_module.bn_module_name if hasattr(object_module, 'bn_module_name') else None
        if bn_module_name:
            bn_parent_module = model_helper.get_parent_module(bn_module_name)
            setattr(bn_parent_module, bn_module_name.split('.')[-1],
                    object_module.bn_module)

        LOGGER.logd(
            "Delete ActivationQAT and WeightQAT module "
            "of '{}' success!".format(object_name), 'DeleteRetrainPass')
