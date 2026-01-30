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

Fakequant bias from int32 to float32.

"""
import numpy as np
from torch import Tensor # pylint: disable=E0401
from torch.nn.parameter import Parameter # pylint: disable=E0401

from amct_pytorch.optimizer.base_module_fusion_pass import BaseModuleFusionPass
from amct_pytorch.custom_op.fake_quant import FAKE_MODULES
from amct_pytorch.utils.log import LOGGER
from amct_pytorch.optimizer.insert_bias_quant_pass import InsertBiasQuantPass


class BiasFakequantModulePass(BaseModuleFusionPass):
    """
    Function: Fakequant bias from int32 to float32
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
        if name not in self.records:
            return False

        if type(module).__name__ in FAKE_MODULES and \
                module.sub_module.bias is not None:
            return True

        return False

    def do_pass(self, model, object_module, object_name, graph=None):
        """
        Function: Do actual quantization and node's bias is changed to float32.
        Parameters:
            graph: graph structure
            object_node: node to process
            model: torch.nn.Module, the model to be modified. if it's
                None, the gaph will be modified.
        Return: None
        """
        bias_np = object_module.sub_module.bias.cpu().detach().numpy()
        bias_shape = bias_np.shape
        int32_bias = InsertBiasQuantPass.quant_bias(
            bias_np.reshape([-1]),
            self.records.get(object_name).get('weight_scale'),
            self.records.get(object_name).get('data_scale'),
            object_name)
        float32_bias = int32_bias.reshape(bias_shape).astype(np.float32)
        object_module.sub_module.bias = Parameter(
            Tensor(float32_bias).to(device=object_module.sub_module.bias.device))
        LOGGER.logd("FakeQuant bias from float32 to int32 for module '{}' " \
            "success!".format(object_name), 'BiasFakequantModulePass')
