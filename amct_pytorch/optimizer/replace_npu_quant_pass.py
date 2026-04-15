# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
#
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
    
    @staticmethod
    def match_pattern(module, name):
        """
        Function:Match layers to replace to npu quantization op
        Parameters:
            module: module to be matched
            name: module's name
        Return: True: matched
                False: mismatch
        """
        module_type = type(module)
        if module_type in AlgorithmRegistry.quant_to_deploy.keys():
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
        module_type = type(object_module)
        deploy_ops = AlgorithmRegistry.quant_to_deploy.get(module_type)
        
        if deploy_ops is None:
            raise RuntimeError(f"The deploy_op for {module_type.__name__} is None! "
                                "pls invoke algorithm_register to register deploy_op!")
        
        if isinstance(object_module, deploy_ops[0]):
            LOGGER.logd(f'{module_type.__name__} do not need to invoke convert')
            return

        if type(object_module).__name__ == 'FlatQuantAttention' or type(object_module).__name__ == 'FlatQuantMLP':
            # We needs to access layernorm and trans from higher level, so it cannot be done within the module
            # TODO: eventually we need to decouple the experimental part and avoid importing it in the main logic
            from amct_pytorch.experimental.flatquant.reparam_utils import get_replacement_module
            npu_module = get_replacement_module(model, type(object_module).__name__, object_name, object_module)
        else:
            npu_module = self._get_deploy_module(object_module, module_type)
        
        ModuleHelper.replace_module_by_name(model, object_name, npu_module)
        LOGGER.logd("Replace npu module to '{}' success!".format(object_name), 'ReplaceNpuQuantPass')
    
    def _get_deploy_module(self, object_module, module_type):
        """
        Function: Get deploy module based on object type
        Parameters: object_module: module to process
                    module_type: type of object_module
        Return: deploy module
        """
        deploy_ops = AlgorithmRegistry.quant_to_deploy.get(module_type)
        if isinstance(deploy_ops, list):
            for deploy_op in deploy_ops:
                if self._should_use_deploy_op(object_module, deploy_op):
                    return deploy_op(object_module)
            return deploy_ops[0](object_module)
        else:
            return deploy_ops(object_module)
    
    def _should_use_deploy_op(self, object_module, deploy_op):
        """
        Function: Check if should use specific deploy op based on object module attributes
        Parameters: object_module: module to process
                    deploy_op: deploy op function
        Return: bool
        """
        if hasattr(object_module, 'ori_module_type') and object_module.ori_module_type == 'Conv2d':
            return deploy_op in [NpuQuantizationConv2d]
        elif hasattr(object_module, 'dynamic') and object_module.dynamic is True:
            return deploy_op == NpuQuantizationLinear
        elif hasattr(object_module, 'scale_d') and object_module.scale_d is not None:
            return deploy_op == NpuQuantizationLinear
        elif not hasattr(object_module, 'scale_d') or object_module.scale_d is None:
            return deploy_op == NpuWeightQuantizedLinear
        return False
