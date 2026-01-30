#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Replace "npu_quantize + quantconv2d/quantbatchmatmulV3" node in graph.

"""
from amct_pytorch.optimizer.base_module_fusion_pass import BaseModuleFusionPass
from amct_pytorch.utils.log import LOGGER
from amct_pytorch.utils.model_util import ModuleHelper
from amct_pytorch.custom_op.npu_quant import NpuQuantizationConv2d
from amct_pytorch.custom_op.npu_quant import NpuQuantizationLinear
from amct_pytorch.custom_op.npu_quant import NpuMXQuantizationLinear
from amct_pytorch.custom_op.npu_quant import NpuWeightQuantizedLinear
from amct_pytorch.utils.vars import MXFP8_E4M3FN


class ReplaceNpuQuantPass(BaseModuleFusionPass):
    """
    Function: Replace "npu_quantize + quantconv2d/quantbatchmatmulV3" node in graph.
    APIs: match_pattern, do_pass
    """
    def __init__(self, records):
        """
        Function: init object
        Parameter: None
        Return: None
        """
        super().__init__()
        self.records = records

    def match_pattern(self, module, name, graph=None):
        """
        Function:Match layers to replace to npu quantization op
        Parameters:
            module: module to be matched
            name: module's name
            graph: graph structure, not necessary
        Return: True: matched
                False: mismatch
        """
        if name not in self.records:
            return False

        return True

    def do_pass(self, model, object_module, object_name, graph=None):
        """
        Function: Replace npu quantization op
        Parameters: model: torch.nn.Module, the model to be modified.
                    object_module: module to process
                    object_name: name of object_module
                    graph: graph structure, not necessary
        Return: None
        """
        record = self.records.get(object_name)
        if record.get('weight_compress_only'):
            if type(object_module).__name__ == 'Linear':
                npu_module = NpuWeightQuantizedLinear(object_module, record, object_name)
        else:
            if type(object_module).__name__ == 'Linear':
                if record.get('act_type') == MXFP8_E4M3FN:
                    npu_module = NpuMXQuantizationLinear(object_module, record)
                else:
                    npu_module = NpuQuantizationLinear(object_module, record, object_name)
            else:
                npu_module = NpuQuantizationConv2d(object_module, record, object_name)
        ModuleHelper.replace_module_by_name(model, object_name, npu_module)
        LOGGER.logd("Replace npu module to '{}' success!".format(object_name), 'ReplaceNpuQuantPass')