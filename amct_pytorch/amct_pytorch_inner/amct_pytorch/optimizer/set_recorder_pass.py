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

Set record_module in RetrainQuant module.

"""
from amct_pytorch.optimizer.base_module_fusion_pass \
    import BaseModuleFusionPass
from amct_pytorch.utils.log import LOGGER


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
