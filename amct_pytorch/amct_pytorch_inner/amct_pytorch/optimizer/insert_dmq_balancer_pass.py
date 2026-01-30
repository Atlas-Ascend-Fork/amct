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

Do caliration for tensor_balance_factor for weight and activation

"""
from amct_pytorch.configuration.configuration import Configuration
from amct_pytorch.optimizer.base_module_fusion_pass import BaseModuleFusionPass
from amct_pytorch.custom_op.dmq_balancer.dmq_balancer import DMQBalancer

from amct_pytorch.utils.log import LOGGER
from amct_pytorch.utils.vars import QUANTIZABLE_TYPES
from amct_pytorch.utils.model_util import ModuleHelper


class InsertDMQBalancerPass(BaseModuleFusionPass):
    """
    Function: Insert DMQBalancer for quantizable module
    APIs: match_pattern, do_pass
    """
    def __init__(self, torch_recorder):
        """
        Function: init object
        Parameter:
        Return: None
        """
        BaseModuleFusionPass.__init__(self)
        self.conf = Configuration()
        self.record_module = torch_recorder
        self.dmq_layers_name = []

    def match_pattern(self, module, name, graph=None):
        """
        Function:Match the module to be quantized in model
        Parameters:
            module: module to be matched
            name: module's name
            graph: graph structure, not necessary
        Return: True: matched
                False: mismatch
        """
        if type(module).__name__ not in QUANTIZABLE_TYPES:
            return False
        if type(module).__name__ == 'AvgPool2d':
            return False
        if name not in self.conf.get_quant_config():
            return False
        if not self.conf.get_layer_config(name) or \
            not self.conf.get_layer_config(name).get('dmq_balancer_param'):
            return False
        return True

    def do_pass(self, model, object_module, object_name, graph=None):
        """
        Function: Do actual Insert DMQBalancer module
        Parameters: model: torch.nn.Module, the model to be modified.
                    object_module: module to process
                    object_name: name of object_module
                    graph: graph structure, not necessary
        Return: None
        """
        self.dmq_layers_name.append(object_name)
        # Step1: find module's parent
        model_helper = ModuleHelper(model)
        parent_module = model_helper.get_parent_module(object_name)

        # Step2: construct a new module
        migration_strength = self.conf.get_layer_config(
            object_name).get('dmq_balancer_param')
        dmq_balancer_args = {
            'record_module': self.record_module,
            'migration_strength': migration_strength,
            'layers_name': [object_name]
        }
        dmq_balancer_module = DMQBalancer(object_module, **dmq_balancer_args)

        # Step3: replace new model
        setattr(parent_module, object_name.split('.')[-1], dmq_balancer_module)

        LOGGER.logd(
            "Insert DMQBalancer module to '{}' success!".format(
                object_name), 'InsertDMQBalancerPass')

    def tear_down(self):
        """
        Function: Write full layers' names to the record
        Parameter: None
        Return: None
        """
        self.record_module.record_quant_layer(self.dmq_layers_name)





