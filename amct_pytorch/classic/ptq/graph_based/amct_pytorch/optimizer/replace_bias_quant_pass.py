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

from ...amct_pytorch.common.utils.vars_util import RNN_LAYER_TYPE
from ...amct_pytorch.optimizer.base_fusion_pass import BaseFusionPass
from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.utils.onnx_initializer_util import TensorProtoHelper
from ...amct_pytorch.utils.quant_node import QuantOpInfo


class ReplaceBiasQuantPass(BaseFusionPass):
    """
    Function: Fakequant weight from int8 to int9
    APIs: match_pattern, do_pass
    """
    def __init__(self, records):
        """
        Function: init object
        Parameter:
            records: dict including quant factors such as scale_w
        Return: None
        """
        BaseFusionPass.__init__(self)
        self.records = records

    def match_pattern(self, node):
        """
        Function: Match pattern of node to be quantized in graph
        Parameters: node: node in graph to be matched
        Return: True: matched
                False: mismatch
        """
        if node.name not in self.records:
            return False

        if node.type == 'AveragePool':
            return False

        bias_node = QuantOpInfo.get_bias_node(node)
        if bias_node is None:
            return False

        return True

    def do_pass(self, graph, object_node, model=None):
        """
        Function: Do actual quantization and node's weight is changed to int9.
        Parameters:
            graph: graph structure
            object_node: node to process
            model: torch.nn.Module, the model to be modified. if it's
                None, the gaph will be modified.
        Return: None
        """
        # Determine the quantization type based on the value of dst_type and obtain the corresponding num_bits
        if self.records.get(object_node.name).get('dst_type') == 'UNSET':
            num_bits = QuantOpInfo.get_dst_num_bits(self.records, object_node.name, 'act')
        else:
            num_bits = QuantOpInfo.get_dst_num_bits(self.records, object_node.name)

        if num_bits == 4:
            bias_index = QuantOpInfo.get_quant_index(object_node).get('bias_index')
            cast_input_anchor = object_node.get_input_anchor(bias_index)
            cast_node = cast_input_anchor.get_peer_output_anchor().node
            bias_input_anchor = cast_node.get_input_anchor(0)
            bias_node = bias_input_anchor.get_peer_output_anchor().node

            # remove input links
            graph.remove_edge(bias_node, 0, cast_node, 0)
            # remove output links
            output_anchor = cast_node.get_output_anchor(0)

            peer_input_anchors = []
            for input_anchor in output_anchor.get_peer_input_anchor():
                peer_input_anchors.append(input_anchor)

            for input_anchor in peer_input_anchors:
                peer_input_node = input_anchor.node
                input_index = input_anchor.index
                graph.remove_edge(cast_node, 0, peer_input_node, input_index)
                graph.add_edge(bias_node, 0, peer_input_node, input_index)

            graph.remove_node(cast_node)

        bias_param = QuantOpInfo.get_bias_node(object_node)
        bias_helper = TensorProtoHelper(bias_param.proto, bias_param.model_path)
        int32_bias = bias_helper.get_data()
        if object_node.type in RNN_LAYER_TYPE:
            float_bias = self.fakequant_rnn_bias(int32_bias, object_node.name)
        else:
            float_bias = int32_bias.astype(np.float32)

        bias_helper.clear_data()
        bias_helper.set_data(float_bias.reshape(-1), 'FLOAT')

        LOGGER.logd(
            "Replace bias quant layer '{}' to fake weight quant layer success!".
            format(object_node.name), 'ReplaceBiasQuantPass')

    def fakequant_rnn_bias(self, int32_bias, layer_name):
        """
        Function: Do dequant to rnn op's bias to introduce quant error
        Parameters:
            layer_name: name of node to process
        Return: None
        """
        # dequant bias corresponding to w and to r separately
        bias_length = int32_bias.shape[1]
        bias_length = len(int32_bias.flatten())
        weight_bias = int32_bias.flatten()[:bias_length // 2]
        recurrence_weight_bias = int32_bias.flatten()[bias_length // 2:]

        scale_w = self.records.get(layer_name).get('weight_scale')
        if weight_bias.size != scale_w.size:
            scale_w = np.repeat(scale_w, weight_bias.size // scale_w.size)

        scale_d = self.records.get(layer_name).get('data_scale')
        deq_scale = np.multiply(scale_w, scale_d).reshape([-1])
        weight_bias = weight_bias.astype(np.float32) * deq_scale

        scale_r = self.records.get(layer_name).get('recurrence_weight_scale')
        if recurrence_weight_bias.size != scale_r.size:
            scale_r = np.repeat(scale_r, recurrence_weight_bias.size // scale_r.size)
        scale_h = self.records.get(layer_name).get('h_scale')
        deq_scale_h = np.multiply(scale_r, scale_h).reshape([-1])
        recurrence_weight_bias = recurrence_weight_bias.astype(np.float32) * deq_scale_h

        bias_float32 = np.concatenate([weight_bias, recurrence_weight_bias], 0)
        LOGGER.logd("Fakequant bias from int32 to float32 for layer '{}'" \
            .format(layer_name), 'ReplaceBiasQuantPass')
        return bias_float32