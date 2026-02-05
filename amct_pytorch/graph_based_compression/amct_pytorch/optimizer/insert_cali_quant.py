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
import numpy as np

from ...amct_pytorch.configuration.configuration import Configuration
from ...amct_pytorch.optimizer.base_module_fusion_pass import BaseModuleFusionPass
from ...amct_pytorch.custom_op.cali_quant import CaliQuant
from ...amct_pytorch.custom_op.cali_quant import CaliQuantHfmg

from ...amct_pytorch.utils.model_util import ModuleHelper
from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.utils.vars import QUANTIZABLE_TYPES
from ...amct_pytorch.utils.vars import TENSOR_BALANCE_FACTOR
from ...amct_pytorch.utils.vars import NUM_BITS
from ...amct_pytorch.utils.vars import BATCH_NUM

WITH_OFFSET = 'with_offset'
FAKEQUANT_PRECISION_MODE = 'fakequant_precision_mode'


class InsertCaliQuantPass(BaseModuleFusionPass):
    """
    Function: Insert CaliQuant for quantizable module
    APIs: match_pattern, do_pass
    """
    def __init__(self, torch_recorder, records=None, dump_config=None, mode='cali_dump'):
        """
        Function: init object of insert act cali quant pass.
        Parameter:
            records: a dict containing record
            DumpConfig, contains dump_dir, batch_num.
        Return: str, 'cali', 'dump', 'cali_dump'
        """
        BaseModuleFusionPass.__init__(self)
        self.conf = Configuration()
        self.record_module = torch_recorder
        self.records = {} if records is None else records
        self.dump_config = dump_config
        self.mode = mode

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
        if type(module).__name__ not in QUANTIZABLE_TYPES:
            return False
        if name not in self.conf.get_quant_config():
            return False

        return True

    def do_pass(self, model, object_module, object_name, graph=None):
        """
        Function: Do actual Insert CaliQuant module
        Parameters: model: torch.nn.Module, the model to be modified.
                    object_module: module to process
                    object_name: name of object_module
                    graph: graph structure, not necessary
        Return: None
        """
        # Step1: find module's parent
        model_helper = ModuleHelper(model)
        parent_module = model_helper.get_parent_module(object_name)
        # Step2: construct a new module
        tensor_balance_factor = self.broad_cast_tensor_balance_factor(
            object_name, object_module, graph)
        act_config = self.conf.get_layer_config(
            object_name)['activation_quant_params']
        act_algo = act_config.get('act_algo', 'ifmr')
        num_bits = act_config.get(NUM_BITS, 8)
        if act_algo == 'hfmg':
            hfmg_args = {
                'record_module': self.record_module,
                'layers_name': [object_name],
                NUM_BITS: num_bits,
                BATCH_NUM: act_config[BATCH_NUM],
                WITH_OFFSET: act_config[WITH_OFFSET],
                'nbins': act_config['num_of_bins'],
                'dump_config': self.dump_config,
                'mode': self.mode,
                TENSOR_BALANCE_FACTOR: tensor_balance_factor,
                FAKEQUANT_PRECISION_MODE: act_config.get(FAKEQUANT_PRECISION_MODE)
            }
            act_cali_module = CaliQuantHfmg(object_module, **hfmg_args)
        else:
            ifmr_args = {
                'record_module': self.record_module,
                'layers_name': [object_name],
                NUM_BITS: num_bits,
                BATCH_NUM: act_config[BATCH_NUM],
                WITH_OFFSET: act_config[WITH_OFFSET],
                'max_percentile': act_config['max_percentile'],
                'min_percentile': act_config['min_percentile'],
                'search_start': act_config['search_range_start'],
                'search_end': act_config['search_range_end'],
                'search_step': act_config['search_step'],
                'dump_config': self.dump_config,
                'mode': self.mode,
                TENSOR_BALANCE_FACTOR: tensor_balance_factor,
                FAKEQUANT_PRECISION_MODE: act_config.get(FAKEQUANT_PRECISION_MODE)
            }
            act_cali_module = CaliQuant(object_module, **ifmr_args)

        # Step3: replace new model
        setattr(parent_module, object_name.split('.')[-1], act_cali_module)

        LOGGER.logd(
            "Insert CaliQuant module of {} to '{}' success!".format(
                act_algo, object_name), 'InsertCaliQuantPass')

    def broad_cast_tensor_balance_factor(self, node_name, module, graph):
        '''
        Function: broad cast dmq factor to align with module's cin axis
        Parameters: node_name: name of object_module
                    module: module to process
                    graph: graph structure, not necessary
        Return: tensor_balance_factor: dmq factor of which shape is aligned with input's cin
        '''
        dmq_param = self.conf.get_layer_config(node_name).get('dmq_balancer_param')
        # broadcast tensor_balance_factor to activation shape
        tensor_balance_factor = None
        if dmq_param and type(module).__name__ != 'AvgPool2d':
            if graph is None:
                raise RuntimeError(' There is no graph in Caliquantpass!')
            object_node = graph.get_node_by_name(node_name)
            tensor_balance_factor = self.records.get(object_node.name).get(TENSOR_BALANCE_FACTOR)
            tensor_balance_factor = np.array(tensor_balance_factor, np.float32)
            if type(module).__name__ in ('Conv2d', 'ConvTranspose2d'):
                tensor_balance_factor = tensor_balance_factor.reshape([1, -1, 1, 1])
            elif type(module).__name__ in 'Conv3d':
                tensor_balance_factor = tensor_balance_factor.reshape([1, -1, 1, 1, 1])
            elif type(module).__name__ in ('Conv1d', 'ConvTranspose1d'):
                tensor_balance_factor = tensor_balance_factor.reshape([1, -1, 1])
        return tensor_balance_factor