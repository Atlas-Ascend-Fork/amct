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
from amct_pytorch.amct_pytorch_inner.amct_pytorch.custom_op.utils import get_algo_params
from amct_pytorch.amct_pytorch_inner.amct_pytorch.custom_op.fake_quant import FakeQuantConv2d
from amct_pytorch.amct_pytorch_inner.amct_pytorch.custom_op.fake_quant import FakeQuantLinear
from amct_pytorch.amct_pytorch_inner.amct_pytorch.custom_op.fake_quant import FakeMXQuantLinear
from amct_pytorch.amct_pytorch_inner.amct_pytorch.custom_op.fake_quant import FakeWeightQuantizedLinear
from amct_pytorch.amct_pytorch_inner.amct_pytorch.custom_op.fake_quant import FakeWeightQuantizedConv2d
from amct_pytorch.amct_pytorch_inner.amct_pytorch.custom_op.fake_quant import LutFakeWeightQuantizedLinear
from amct_pytorch.amct_pytorch_inner.amct_pytorch.optimizer.base_module_fusion_pass import BaseModuleFusionPass
from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.log import LOGGER
from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.model_util import ModuleHelper
from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.vars import MXFP8_E4M3FN


class ReplaceFakeQuantModulePass(BaseModuleFusionPass):
    """
    Function: replace quantized module to fakequant module
    APIs: match_pattern, do_pass
    """
    def __init__(self, records):
        """
        Function: init object of insert kv_cache quant pass.
        Parameter:
            record: a dict, record quant factors into file
        Return: None
        """
        super().__init__()
        self.records = records

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
        if name not in self.records:
            return False
        return True

    def do_pass(self, model, object_module, object_name, graph=None):
        """
        Function: replace quantized module to fakequant module
        Parameters: model: torch.nn.Module, the model to be modified.
                    object_module: module to process
                    object_name: name of object_module
                    graph: graph structure, not necessary
        Return: None
        """
        # Step1: generate quantized module
        record = self.records.get(object_name)
        if record.get('weight_compress_only'):
            # weight compress only
            if type(object_module).__name__ == 'Conv2d':
                quant_module = FakeWeightQuantizedConv2d(object_module, record, object_name)
            else:
                if record.get('quant_result_path') and get_algo_params(record, object_name, 'lut_quantize') is not None:
                    quant_module = LutFakeWeightQuantizedLinear(object_module, record, object_name)
                else:
                    quant_module = FakeWeightQuantizedLinear(object_module, record, object_name)
        else:
            # compress both wts and acts
            if type(object_module).__name__ == 'Conv2d':
                quant_module = FakeQuantConv2d(object_module, record, object_name)
            else:
                if record.get('act_type') == MXFP8_E4M3FN:
                    quant_module = FakeMXQuantLinear(object_module, record)
                else:
                    quant_module = FakeQuantLinear(object_module, record, object_name)
        # Step2: replace with new module
        ModuleHelper.replace_module_by_name(model, object_name, quant_module)
        LOGGER.logd("Replace quantized module to '{}' success!".format(
            object_name), 'ReplaceFakeQuantModulePass')
