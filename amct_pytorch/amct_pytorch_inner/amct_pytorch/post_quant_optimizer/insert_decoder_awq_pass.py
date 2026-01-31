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
from amct_pytorch.amct_pytorch_inner.amct_pytorch.custom_op.awq import DecoderAWQuant
from amct_pytorch.amct_pytorch_inner.amct_pytorch.optimizer.base_module_fusion_pass import BaseModuleFusionPass
from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.log import LOGGER
from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.model_util import ModuleHelper
from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.vars import MXFP4_E2M1, FLOAT4_E2M1, FLOAT4_E1M2

decoder_linear_names = (
    'self_attn.q_proj',
    'self_attn.k_proj',
    'self_attn.v_proj',
    'self_attn.o_proj',
    'mlp.gate_proj',
    'mlp.up_proj',
    'mlp.down_proj'
)


class InsertDecoderAWQuantPass(BaseModuleFusionPass):
    """
    Function: Insert DecoderAWQuant in model for quantizable LlamaDecoderLayer
    APIs: match_pattern, do_pass
    """
    def __init__(self, quant_config, recorder):
        """
        Function: init object of insert decoder awq pass
        Parameter:
            recorder: nn.module, record quant factors into file
            quant_config: dict, config
        Return: None
        """
        super().__init__()
        self.config = quant_config
        self.post_quant_record_module = recorder
        self.awq_layer_names = list()

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
        if module.__class__.__name__ == 'LlamaDecoderLayer':
            return True

        return False

    def do_pass(self, model, object_module, object_name, graph=None):
        """
        Function: Do actual Insert DecoderAWQuant module
        Parameters: model: torch.nn.Module, the model to be modified.
                    object_module: module to process
                    object_name: name of object_module
                    graph: graph structure, not necessary
        Return: None
        """
        # check linear in LlamaDecoderLayer do MXFP4_E2M1 AWQ
        linears_name = [object_name + '.' + linear_name for linear_name in decoder_linear_names]
        for name in linears_name:
            linear_config = self.config.get(name)
            # whether quantizable
            if linear_config is None:
                LOGGER.logd("{} is non-quantizable since {} not do quant.".format(object_name, name))
                return
            # whether MXFP4_E2M1
            if linear_config.get('wts_type') not in (MXFP4_E2M1, FLOAT4_E2M1, FLOAT4_E1M2):
                LOGGER.logd("{} is non-quantizable since {} wts_type is {}.".format(
                    object_name, name, linear_config.get('wts_type')))
                return
            # whether AWQ
            if 'awq_quantize' not in linear_config.get('weight_quant_algo_params', {}):
                LOGGER.logd("{} is non-quantizable since {} not do AWQ.".format(object_name, name))
                return

        # Step1: construct a new module
        new_module = DecoderAWQuant(object_module, self.post_quant_record_module, [object_name,], linear_config)

        # Step2: replace with new module
        ModuleHelper.replace_module_by_name(model, object_name, new_module)
        self.awq_layer_names.extend(linears_name)

        LOGGER.logd("Insert DecoderAWQuant module to '{}' success!".format(object_name), 'InsertDecoderAWQuantPass')

    def tear_down(self):
        """
        Function: Write all layers' names to the record
        """
        self.post_quant_record_module.record_quant_result_layer(self.awq_layer_names)