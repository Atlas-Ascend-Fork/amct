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
from ...amct_pytorch.utils.log import LOGGER


class SetRecorderPass(BaseModuleFusionPass):
    """
    Function: Set record_module in RetrainQuant module.
    APIs: match_pattern, do_pass
    """
    def __init__(self, torch_recorder=None):
        """
        Function: init object
        Parameter: None
        Return: None
        """
        BaseModuleFusionPass.__init__(self)
        self.torch_recorder = torch_recorder

    @staticmethod
    def match_pattern(module, name, graph=None):
        """
        Function:Match the RetrainQuant module in model.
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

    def do_pass(self, model, object_module, object_name, graph=None):
        """
        Function: Set record_module in RetrainQuant module.
        Parameters: model: torch.nn.Module, the model to be modified.
                    object_module: module to process
                    object_name: name of object_module
                    graph: graph structure, not necessary
        Return: None
        """
        object_module.record_module = self.torch_recorder
        LOGGER.logd(
            "Set recorder module of '{}' success!".format(object_name),
            'SetRecorderPass')
