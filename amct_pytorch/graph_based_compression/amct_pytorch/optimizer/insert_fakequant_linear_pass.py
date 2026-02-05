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

from ...amct_pytorch.optimizer.base_module_fusion_pass import BaseModuleFusionPass

from ...amct_pytorch.utils.model_util import ModuleHelper
from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.custom_op.fake_quant import FakeQuantizedLinear


class InsertFakequantLinearPass(BaseModuleFusionPass):
    """
    Function: insert fakequant linear module
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
        if type(module).__name__ == "Linear" and name in self.records:
            return True
        return False

    def do_pass(self, model, object_module, object_name, graph=None):
        """
        Function: insert fake quant linear.
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
        quant_params = self.records[object_name]
        quant_params['channel_wise'] = False

        # Step2: fake quant
        fakequant_linear_module = FakeQuantizedLinear(
            object_module, quant_params, object_name, self.num_bits)

        # Step3: replace new model
        setattr(parent_module, object_name.split('.')[-1],
                fakequant_linear_module)

        LOGGER.logd(
            "Insert FakeQuantizedLinear module to '{}' success!".format(
                object_name), 'InsertFakequantLinearPass')
