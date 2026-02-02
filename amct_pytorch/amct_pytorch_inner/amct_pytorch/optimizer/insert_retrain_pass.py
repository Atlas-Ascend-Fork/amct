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
from ...amct_pytorch.common.config.field import DEFAULT_NUM_BITS
from ...amct_pytorch.configuration.retrain_config import RetrainConfig as \
    Configuration
from ...amct_pytorch.optimizer.base_fusion_pass import BaseFusionPass
from ...amct_pytorch.custom_op.comp_module.comp_module_conv1d import CompModuleConv1d
from ...amct_pytorch.custom_op.comp_module.comp_module_conv2d import CompModuleConv2d
from ...amct_pytorch.custom_op.comp_module.comp_module_conv3d import CompModuleConv3d
from ...amct_pytorch.custom_op.comp_module.comp_module_linear import CompModuleLinear
from ...amct_pytorch.custom_op.comp_module.comp_module_conv_transpose1d import CompModuleConvTranspose1d
from ...amct_pytorch.custom_op.comp_module.comp_module_conv_transpose2d import CompModuleConvTranspose2d
from ...amct_pytorch.custom_op.comp_module.comp_module_conv_transpose3d import CompModuleConvTranspose3d
from ...amct_pytorch.custom_op.comp_module.comp_module_rnn import CompModuleRNN
from ...amct_pytorch.utils.model_util import ModuleHelper
from ...amct_pytorch.utils.vars import RETRAIN_ONNX_TYPES
from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.utils.module_info import ModuleInfo
from ...amct_pytorch.custom_op.comp_module.comp_module_base import COMP_ALG_QUANT
from ...amct_pytorch.common.utils.onnx_node_util import AttributeProtoHelper
from ...amct_pytorch.utils.vars import FIXED_MIN, H_FIXED_MIN
from ...amct_pytorch.utils.vars import NUM_BITS, NUM_BITS_MAP
from ...amct_pytorch.utils.quant_node import QuantOpInfo
from ...amct_pytorch.common.utils.vars_util import INITIAL_H_INDEX

REPLACE_DICT = {
    'Conv1d': CompModuleConv1d,
    'Conv2d': CompModuleConv2d,
    'Conv3d': CompModuleConv3d,
    'Gemm': CompModuleLinear,
    'MatMul': CompModuleLinear,
    'ConvTranspose1d': CompModuleConvTranspose1d,
    'ConvTranspose2d': CompModuleConvTranspose2d,
    'ConvTranspose3d': CompModuleConvTranspose3d,
    'LSTM': CompModuleRNN,
    'GRU': CompModuleRNN
}


class InsertRetrainPass(BaseFusionPass):
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
        if not self.conf.retrain_enable(node.name):
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
        # activation quant config
        act_config = self._get_act_config(object_node)
        # weights quant config
        wts_config = self.conf.get_retrain_config(object_node_name)['retrain_weight_config']
        if wts_config.get('channel_wise') is None:
            wts_config['channel_wise'] = False
        wts_config[NUM_BITS] = 4 if wts_config.get('dst_type') == 'INT4' else 8
        wts_config['s_rec_flag'] = False

        # common quant config
        common_config = {
            'layers_name': [object_node_name],
            'node_type': object_node.type,
            'device': self.device,
            'batch_num': self.conf.retrain_config.get('batch_num'),
            'fakequant_precision_mode': self.conf.retrain_config.get('fakequant_precision_mode'),
        }

        if hasattr(object_module, 'comp_algs') and object_module.comp_algs:
            wts_config['n_out_of_m_type'] = object_module.wts_config['n_out_of_m_type']
            wts_config['prune_axis'] = object_module.wts_config['prune_axis']
            wts_config['update_freq'] = object_module.wts_config['update_freq']
            wts_config['prune_algo'] = object_module.wts_config['prune_algo']
            wts_config['layer_name'] = object_module.wts_config['layer_name']
            object_module.comp_algs.append(COMP_ALG_QUANT)
            object_module.act_config = act_config
            object_module.wts_config = wts_config
            object_module.common_config = common_config
            setattr(parent_module, object_node_name.split('.')[-1], object_module)
            LOGGER.logd("Current module is already replaced module. open quant retrain"
                "'{}'.".format(object_node.name), 'InsertRetrainPass')
            return

        object_node_type = get_node_onnx_type(object_node)
        if REPLACE_DICT.get(object_node_type):
            act_wts_qat_module = REPLACE_DICT.get(object_node_type)(
                object_module,
                act_config,
                wts_config,
                common_config)

            act_wts_qat_module.comp_algs.append(COMP_ALG_QUANT)
            setattr(parent_module, object_node_name.split('.')[-1], act_wts_qat_module)
            LOGGER.logd("Insert ActivationQAT and WeightQAT module to "
                "'{}' success!".format(object_node.name), 'InsertRetrainPass')

    def _get_act_config(self, object_node):
        """
        Function: Get the config of activation.
        """
        act_config = self.conf.get_retrain_config(
            object_node.name)['retrain_data_config']
        act_config['ifmr_init'] = False
        if act_config.get('clip_min') is None or \
            act_config.get('clip_max') is None:
            act_config['ifmr_init'] = True
        pre_node = object_node.get_input_anchor(0).get_peer_output_anchor().node
        h_index = QuantOpInfo.get_quant_index(object_node).get(INITIAL_H_INDEX)
        h_pre_node = object_node.get_input_anchor(h_index).get_peer_output_anchor().node \
            if h_index is not None else None
        act_config[H_FIXED_MIN] = act_config.get(FIXED_MIN)
        if act_config.get(FIXED_MIN) is None:
            if pre_node.type == 'Relu':
                act_config[FIXED_MIN] = True
            else:
                act_config[FIXED_MIN] = False
            if h_pre_node is not None and h_pre_node.type == 'Relu':
                act_config[H_FIXED_MIN] = True
            else:
                act_config[H_FIXED_MIN] = False
        act_config[NUM_BITS] = NUM_BITS_MAP.get(act_config.get('dst_type'), DEFAULT_NUM_BITS)

        return act_config


def get_node_onnx_type(node):
    node_type = node.type
    if node_type not in ['Conv', 'ConvTranspose']:
        return node_type

    kernel_shape = AttributeProtoHelper(
        node.proto).get_attr_value('kernel_shape')
    if len(kernel_shape) not in [1, 2, 3]:
        raise RuntimeError("not support node type '{}'".format(node_type))

    node_type += str(len(kernel_shape)) + 'd'
    return node_type