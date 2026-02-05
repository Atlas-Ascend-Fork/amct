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

import amct_pytorch.graph_based_compression.amct_pytorch.configuration.quant_calibration_config as config_func
from ...amct_pytorch.capacity import CAPACITY
from ...amct_pytorch.custom_op.hfmg.hfmg import HFMG
from ...amct_pytorch.custom_op.ifmr.ifmr import IFMR
from ...amct_pytorch.custom_op.kv_cache_quant import KVCacheQuant
from ...amct_pytorch.optimizer.base_module_fusion_pass import BaseModuleFusionPass
from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.utils.model_util import ModuleHelper
from ...amct_pytorch.utils.vars import BATCH_NUM, WITH_OFFSET, QUANT_GRANULARITY, NUM_BITS,\
    NUM_OF_BINS, MAX_PERCENTILE, MIN_PERCENTILE, SEARCH_RANGE_START, SEARCH_RANGE_END, SEARCH_STEP


class InsertKVCacheQuantPass(BaseModuleFusionPass):
    """
    Function: Insert KVCacheQuant in model for quantizable kv-cache layer
    APIs: match_pattern, do_pass
    """
    def __init__(self, recorder, kv_cache_config):
        """
        Function: init object of insert kv_cache quant pass.
        Parameter:
            recorder: nn.module, record quant factors into file
            kv_cache_config: dict
        Return: None
        """
        super().__init__()
        self.config = kv_cache_config
        self.record_module = recorder
        self.record_module.enable_kv_cache_quant = True
        self.kv_cache_quant_layers = config_func.get_kv_cache_quant_layers(self.config)
        self.kv_quantized_layer_names = []

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
        if type(module).__name__ not in CAPACITY.get_value('KV_CACHE_QUANTIZE_TYPES'):
            return False
        if name not in self.kv_cache_quant_layers:
            return False

        return True

    def do_pass(self, model, object_module, object_name, graph=None):
        """
        Function: Do actual Insert KVQuant module
        Parameters: model: torch.nn.Module, the model to be modified.
                    object_module: module to process
                    object_name: name of object_module
                    graph: graph structure, not necessary
        Return: None
        """
        # Step1: construct a new module
        layer_config = config_func.get_quant_layer_config(object_name, self.config).get('kv_data_quant_config')
        act_algo = layer_config.get('act_algo', 'ifmr')
        if act_algo == 'hfmg':
            cali_algo_params = {
                'layers_name': [object_name],
                NUM_BITS: 8,
                BATCH_NUM: layer_config[BATCH_NUM],
                WITH_OFFSET: layer_config[WITH_OFFSET],
                'nbins': layer_config[NUM_OF_BINS],
                QUANT_GRANULARITY: layer_config[QUANT_GRANULARITY]
            }
            cali_module = HFMG(**cali_algo_params)
        else:
            cali_algo_params = {
                'layers_name': [object_name],
                NUM_BITS: 8,
                BATCH_NUM: layer_config[BATCH_NUM],
                WITH_OFFSET: layer_config[WITH_OFFSET],
                MAX_PERCENTILE: layer_config[MAX_PERCENTILE],
                MIN_PERCENTILE: layer_config[MIN_PERCENTILE],
                'search_start': layer_config[SEARCH_RANGE_START],
                'search_end': layer_config[SEARCH_RANGE_END],
                SEARCH_STEP: layer_config[SEARCH_STEP],
                QUANT_GRANULARITY: layer_config[QUANT_GRANULARITY]
            }
            cali_module = IFMR(**cali_algo_params)

        kv_cache_quant_module = KVCacheQuant(
            object_module, cali_module, self.record_module, [object_name], cali_algo_params)

        # Step2: find object_module's parent
        model_helper = ModuleHelper(model)
        parent_module = model_helper.get_parent_module(object_name)

        # Step3: replace with new module
        setattr(parent_module, object_name.split('.')[-1], kv_cache_quant_module)

        self.kv_quantized_layer_names.append(object_name)
        LOGGER.logd("Insert KVCacheQuant module to '{}' success!".format(object_name), 'InsertKVCacheQuantPass')

    def tear_down(self):
        """
        Function: Write all layers' names to the record
        Parameter: None
        Return: None
        """
        self.record_module.record_quant_layer(self.kv_quantized_layer_names)