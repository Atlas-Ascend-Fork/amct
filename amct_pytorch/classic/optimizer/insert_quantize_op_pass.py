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

from amct_pytorch.classic.optimizer.base_module_fusion_pass import BaseModuleFusionPass
from amct_pytorch.common.utils.log import LOGGER
from amct_pytorch.common.config.utils import get_alg_name_from_config
from amct_pytorch.common.utils.model_util import ModuleHelper
from amct_pytorch.algorithms import AlgorithmRegistry
from amct_pytorch.classic.quantize_op.quant_result_mixin import QuantResultSupportMixin


class InsertQuantizeModulePass(BaseModuleFusionPass):
    """
    Function: insert quantize operator in model for quantifiable layer
    APIs: match_pattern, do_pass
    """

    def __init__(self, quant_config, quant_result_writer=None):
        """
        Function: init object of insert quantize op quant pass
        Parameter:
            quant_config: dict, config
            quant_result_writer: optional QuantResultWriter injected into every
                inserted quantize module so it can dump its quant params at its
                own finalization point. None means quant results are not
                persisted.
        Return: None
        """
        super().__init__()
        self.config = quant_config
        self.quant_layers = list(quant_config.keys())
        self.quantize_ops = dict()
        self.quant_result_writer = quant_result_writer

    def match_pattern(self, module, name):
        """
        Function:Match the module to be quantized in model
        Parameters:
            module: module to be matched
            name: module's name
        Return: True: matched
                False: mismatch
        """
        if name not in self.quant_layers:
            return False
        alg = self.config[name].get('algorithm')
        alg_names, _ = get_alg_name_from_config(alg)
        alg_name = alg_names[0]  # each module corresponds to only one algo
        if AlgorithmRegistry.algo.get(alg_name):
            ori_ops = AlgorithmRegistry.algo.get(alg_name).keys()
            for ori_op in ori_ops:
                if type(module).__name__ == ori_op:
                    self.quantize_ops[name] = AlgorithmRegistry.algo.get(alg_name).get(
                        ori_op
                    )
                    return True

        return False

    def do_pass(self, model, object_module, object_name):
        """
        Function: Do actual Insert calibration operator
        Parameters: model: torch.nn.Module, the model to be modified.
                    object_module: module to process
                    object_name: name of object_module
        Return: None
        """
        layer_config = self.config.get(object_name)
        new_module = self.quantize_ops[object_name](
            object_module, object_name, layer_config
        )

        helper = ModuleHelper(model)
        helper.replace_module_by_name(model, object_name, new_module)
        if self.quant_result_writer is not None:
            if not isinstance(new_module, QuantResultSupportMixin):
                raise TypeError(
                    "quantize module '{}' of layer '{}' does not implement "
                    'QuantResultSupportMixin; refusing to attach a quant result '
                    'writer.'.format(type(new_module).__name__, object_name)
                )
            new_module.attach_quant_result_writer(self.quant_result_writer)
        LOGGER.logd(
            "Insert quantize op module to '{}' success!".format(object_name),
            'InsertQuantizeModulePass',
        )
