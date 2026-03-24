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

from amct_pytorch.optimizer.base_module_fusion_pass import BaseModuleFusionPass
from amct_pytorch.utils.log import LOGGER
from amct_pytorch.utils.model_util import ModuleHelper
from amct_pytorch.deploy_op import NpuQuantizationLinear
from amct_pytorch.deploy_op import NpuWeightQuantizedLinear
from amct_pytorch.deploy_op import NpuQuantizationConv2d
from amct_pytorch.quantize_op import MinMaxQuant
from amct_pytorch.quantize_op import OfmrQuant
from amct_pytorch.algorithm import AlgorithmRegistry


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
        if type(module) in AlgorithmRegistry.quant_to_deploy.keys():
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
        if AlgorithmRegistry.quant_to_deploy[type(object_module)][0] is None:
            raise RuntimeError(f"The deploy_op for {type(object_module).__name__} is None! "
                               "pls invoke algorithm_register to register deploy_op!")

        if isinstance(object_module, AlgorithmRegistry.quant_to_deploy[type(object_module)][0]):
            LOGGER.logd(f'{type(object_module)} do not need to invoke convert')
            return

        if hasattr(object_module, 'ori_module_type') and object_module.ori_module_type == 'Conv2d':
            npu_module = NpuQuantizationConv2d(object_module)
        elif type(object_module) in [MinMaxQuant, OfmrQuant]:
            if object_module.scale_d:
                npu_module = NpuQuantizationLinear(object_module)
            else:
                npu_module = NpuWeightQuantizedLinear(object_module)
        elif type(object_module).__name__ == 'FlatQuantAttention' or type(object_module).__name__ == 'FlatQuantMLP':
            # We needs to access layernorm and trans from higher level, so it cannot be done within the module
            # TODO: eventually we need to decouple the experimental part and avoid importing it in the main logic
            from amct_pytorch.experimental.flatquant.reparam_utils import get_replacement_module
            npu_module = get_replacement_module(model, type(object_module).__name__, object_name, object_module)
        else:
            npu_module = AlgorithmRegistry.quant_to_deploy[type(object_module)][0](object_module)
        
        ModuleHelper.replace_module_by_name(model, object_name, npu_module)
        LOGGER.logd("Replace npu module to '{}' success!".format(object_name), 'ReplaceNpuQuantPass')