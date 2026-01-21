# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

import re

from torch import nn
from amct_pytorch.optimizer.base_module_fusion_pass import BaseModuleFusionPass
from amct_pytorch.utils.log import LOGGER
from amct_pytorch.utils.model_util import ModuleHelper
from amct_pytorch.deploy_op import NpuQuantizationLinear
from amct_pytorch.deploy_op import NpuWeightQuantizedLinear
from amct_pytorch.experimental.flatquant.npu_flat_quant_module.flat_quant_module\
     import NpuFlatQuantAttention, NpuFlatQuantMLP
from amct_pytorch.quantize_op import MinMaxQuant
from amct_pytorch.experimental.flatquant.flat_quant_module.default_model_utils import FlatQuantAttention, FlatQuantMLP
from amct_pytorch.experimental.flatquant.flat_quant_module.flat_utils import reparameterize_ln
from amct_pytorch.algorithm import AlgorithmRegistry


def get_layer_from_submodule(model: nn.Module, object_name: str):
    object_name_tks = object_name.split(".")
    layer_module = model
    for i in range(len(object_name_tks) - 1):
        layer_module = getattr(layer_module, object_name_tks[i])
    return layer_module


class ReplaceNpuQuantModulePass(BaseModuleFusionPass):
    """
    Function: Replace npu quant module in graph.
    APIs: match_pattern, do_pass
    """
    def __init__(self):
        """
        Function: init object
        Parameter: None
        Return: None
        """
        super().__init__()

    def match_pattern(self, module, name):
        """
        Function:Match layers to replace to npu quantization op
        Parameters:
            module: module to be matched
            name: module's name
        Return: True: matched
                False: mismatch
        """
        if isinstance(module, tuple(AlgorithmRegistry.quant_op)):
            return True

        return False

    def do_pass(self, model, object_module, object_name):
        """
        Function: Replace npu quantization op
        Parameters: model: torch.nn.Module, the model to be modified.
                    object_module: module to process
                    object_name: name of object_module
        Return: None
        """
        if not AlgorithmRegistry.quant_to_deploy[type(object_module)]:
            raise RuntimeError(f"The deploy_op for {type(object_module).__name__} is None! "
                               "pls invoke algorithm_register to register deploy_op!")

        if isinstance(object_module, MinMaxQuant):
            if object_module.scale_d:
                npu_module = NpuQuantizationLinear(object_module)
            else:
                npu_module = NpuWeightQuantizedLinear(object_module)
        elif isinstance(object_module, FlatQuantAttention):
            object_module.reparameterize()
            if object_module.ln_trans is not None and object_module.ln_trans.add_diag:
                matched = re.match(r".*\.layers\..*\.self_attn", object_name)
                if not matched:
                    raise ValueError(f"object_name {object_name} not matched with required structure for self_attn")
                layer_module = get_layer_from_submodule(model, object_name)
                # reparameterize_ln needs to access layernorm and trans from higher level
                # so it cannot be done within the module (same for mlp)
                reparameterize_ln(layer_module.input_layernorm, object_module.ln_trans)
            npu_module = NpuFlatQuantAttention(object_module)
        elif isinstance(object_module, FlatQuantMLP):
            object_module.reparameterize()
            if object_module.up_gate_trans is not None and object_module.up_gate_trans.add_diag:
                matched = re.match(r".*\.layers\..*\.mlp", object_name)
                if not matched:
                    raise ValueError(f"object_name {object_name} not matched with required structure for mlp")
                layer_module = get_layer_from_submodule(model, object_name)
                reparameterize_ln(layer_module.post_attention_layernorm, object_module.up_gate_trans)
            npu_module = NpuFlatQuantMLP(object_module)
        else:
            npu_module = AlgorithmRegistry.quant_to_deploy[type(object_module)](object_module)
        
        ModuleHelper.replace_module_by_name(model, object_name, npu_module)
        LOGGER.logd("Replace npu module to '{}' success!".format(object_name), 'ReplaceNpuQuantPass')