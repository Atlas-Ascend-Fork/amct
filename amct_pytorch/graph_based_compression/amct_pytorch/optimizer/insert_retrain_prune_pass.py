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
import torch.nn as nn
from ...amct_pytorch.configuration.retrain_config import RetrainConfig as \
    Configuration
from ...amct_pytorch.optimizer.base_fusion_pass import BaseFusionPass
from ...amct_pytorch.utils.model_util import ModuleHelper
from ...amct_pytorch.utils.vars import RETRAIN_ONNX_TYPES
from ...amct_pytorch.utils.vars import CHANNEL_WISE
from ...amct_pytorch.utils.vars import FIXED_MIN
from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.utils.module_info import ModuleInfo
from ...amct_pytorch.custom_op.comp_module.comp_module_base import COMP_ALG_PRUNE
from .insert_retrain_pass import REPLACE_DICT, get_node_onnx_type

FIXED_MIN = 'fixed_min'


class InsertRetrainPrunePass(BaseFusionPass):
    """
    Function: Insert some mudule about retrain quantization.
    APIs: match_pattern, do_pass
    """
    def __init__(self, device='cpu'):
        """
        Function: init object
        Parameter: None
        Return: None
        """
        BaseFusionPass.__init__(self)
        self.conf = Configuration()
        self.device = device

    def match_pattern(self, node):
        """
        Function: Match the node to be retrain quantized in graph
        Parameters: node: node in graph to be matched
        Return: True: matched
                False: mismatch
        """
        if node.type not in RETRAIN_ONNX_TYPES:
            return False
        if not self.conf.selective_prune_enable(node.name):
            return False
        return True

    def do_pass(self, graph, object_node, model=None):
        """
        Function: Insert some mudule about retrain quantization.
        Parameters: graph: graph structure
                    object_node: node to process
                    model: torch.nn.Module, the model to be modified. if it's
                        None, the gaph will be modified.
        Return: None
        """
        # Step1: find module and its parent
        object_node_name = object_node.name
        model_helper = ModuleHelper(model)
        object_module = model_helper.get_module(object_node_name)
        parent_module = model_helper.get_parent_module(object_node_name)

        # Step2: construct a new module
        # activation prune config
        if not self.conf.get_retrain_config(object_node_name):
            # only prune act_conig empty
            act_config = {}
            wts_config = {}
            wts_config[CHANNEL_WISE] = False
        else:
            act_config = self.conf.get_retrain_config(
                object_node_name)['retrain_data_config']
            if act_config.get('clip_min') is None or \
                act_config.get('clip_max') is None:
                act_config['ifmr_init'] = True
            pre_node = \
                object_node.get_input_anchor(0).get_peer_output_anchor().node
            if act_config.get(FIXED_MIN) is None:
                if pre_node.type == 'Relu':
                    act_config[FIXED_MIN] = True
                else:
                    act_config[FIXED_MIN] = False
            act_config['num_bits'] = 4 if act_config.get('dst_type') == 'INT4' else 8

            wts_config = self.conf.get_retrain_config(
                object_node_name)['retrain_weight_config']
            if wts_config.get(CHANNEL_WISE) is None:
                wts_config[CHANNEL_WISE] = False

        # common prun config
        common_config = {
            'device': self.device,
            'layers_name': [object_node_name],
            'batch_num': self.conf.retrain_config.get('batch_num'),
        }

        # weights prune config
        _, cin_axis = ModuleInfo.get_wts_cout_cin(object_module)
        layer_prune_config = self.conf.get_layer_prune_config(object_node_name)

        wts_config['n_out_of_m_type'] = layer_prune_config['n_out_of_m_type']
        wts_config['prune_axis'] = cin_axis
        wts_config['update_freq'] = layer_prune_config['update_freq']
        wts_config['prune_algo'] = layer_prune_config['algo']
        wts_config['layer_name'] = object_node_name

        if hasattr(object_module, 'comp_algs') and object_module.comp_algs:
            LOGGER.logd(
                "Current module is already replaced module, not to replace again "
                "'{}'.".format(object_node.name), 'InsertRetrainPrunePass')
            return


        object_node_type = get_node_onnx_type(object_node)
        if REPLACE_DICT.get(object_node_type):
            act_wts_qat_module = REPLACE_DICT.get(object_node_type)(
                object_module,
                act_config,
                wts_config,
                common_config)

            # Step3: replace new model
            act_wts_qat_module.comp_algs.append(COMP_ALG_PRUNE)
            setattr(parent_module, object_node_name.split('.')[-1],
                    act_wts_qat_module)

            LOGGER.logd(
                "Insert ActivationQAT and WeightQAT module to "
                "'{}' success!".format(object_node.name), 'InsertRetrainPrunePass')
