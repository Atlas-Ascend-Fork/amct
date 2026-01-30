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

insert fakequant conv module

"""

from amct_pytorch.optimizer.base_module_fusion_pass import BaseModuleFusionPass

from amct_pytorch.utils.model_util import ModuleHelper
from amct_pytorch.utils.log import LOGGER
from amct_pytorch.custom_op.fake_quant import FakeQuantizedConv


class InsertFakequantConvPass(BaseModuleFusionPass):
    """
    Function: insert fakequant conv module
    APIs: match_pattern, do_pass
    """
    def __init__(self, records, num_bits):
        """
        Function: init object
        Parameter:
            records: dict including quant factors such as scale_w
            num_bits: int number indicating the bit to be quanted such 8
        Return: None
        """
        BaseModuleFusionPass.__init__(self)
        self.records = records
        self.num_bits = num_bits

    def match_pattern(self, module, name, graph=None):
        """
        Function: Match pattern of node to be quantized in graph
        Parameters: node: node in graph to be matched
        Return: True: matched
                False: mismatch
        """
        if type(module).__name__ in ["Conv1d", "Conv2d", "Conv3d"] and name in self.records:
            return True
        return False


    def do_pass(self, model, object_module, object_name, graph=None):
        """
        Function: insert fake quant Conv.
        Parameters:
            graph: graph structure
            object_module: module to process
            model: torch.nn.Module, the model to be modified. if it's
                None, the graph will be modified.
        Return: None
        """
        # Step1: find module's parent
        model_helper = ModuleHelper(model)
        parent_module = model_helper.get_parent_module(object_name)

        # Step2: fake quant
        quant_params = self.records[object_name]
        channel_wise = len(quant_params['weight_scale'].flatten()) > 1
        quant_params['channel_wise'] = channel_wise
        fakequant_conv_module = FakeQuantizedConv(
            object_module, quant_params, object_name, self.num_bits)

        # Step3: replace new model
        setattr(
            parent_module, object_name.split('.')[-1], fakequant_conv_module)

        LOGGER.logd(
            "Insert FakeQuantizedConv module to '{}' success!".format(
                object_name), 'InsertFakequantConvPass')
