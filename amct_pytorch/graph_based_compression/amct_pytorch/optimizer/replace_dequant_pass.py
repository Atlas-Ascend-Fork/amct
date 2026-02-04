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

from ...amct_pytorch.optimizer.base_fusion_pass import BaseFusionPass
from ...amct_pytorch.module.dequant_module import add_fake_dequant

from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.utils.quant_node import QuantOpInfo
from ...amct_pytorch.utils.onnx_initializer_util import TensorProtoHelper
from ...amct_pytorch.common.utils.onnx_node_util import AttributeProtoHelper


class ReplaceDequantPass(BaseFusionPass):
    """
    Function: Replace AscendDequant to fakeqaunt 'Dequant' with onnx's ops
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

    @staticmethod
    def _delete_ascend_dequant_node(graph, object_node):
        """delete dequant node and it's param node"""
        param_in_anchor = object_node.get_input_anchor(1)
        peer_output_anchor = param_in_anchor.get_peer_output_anchor()
        param_node = peer_output_anchor.node
        graph.remove_edge(param_node, peer_output_anchor.index, object_node, 1)
        if not peer_output_anchor.get_peer_input_anchor():
            graph.remove_node(param_node)
        graph.remove_node(object_node)

    def match_pattern(self, node):
        """
        Function: Match the AscendQuant node
        Parameters: node: node in graph to be matched
        Return: True: matched
                False: mismatch
        """
        if node.type != 'AscendDequant':
            return False

        return True

    def do_pass(self, graph, object_node, model=None):
        """
        Function: Do actually replacing AscendDequant to fakeqaunt 'Dequant'
            with onnx's ops
        Parameters: graph: graph structure
                    object_node: node to process
                    model: torch.nn.Module, the model to be modified. if it's
                        None, the gaph will be modified.
        Return: None
        """
        input_anchor = object_node.get_input_anchor(0)
        # Step1: add a new_node
        quantized_node = input_anchor.get_peer_output_anchor().node
        enter_node, out_node = _get_dequant_param(graph, quantized_node,
                                                object_node)
        # Step2: Relink nodes in th graph
        # remove input links
        peer_output_anchor = input_anchor.get_peer_output_anchor()
        peer_node = peer_output_anchor.node
        peer_output_anchor_index = peer_output_anchor.index
        graph.remove_edge(peer_node, peer_output_anchor_index, object_node, 0)
        graph.add_edge(peer_node, peer_output_anchor_index, enter_node, 0)
        # Step2: relink fake dequant output to original dequant outputs
        output_anchor = object_node.get_output_anchor(0)
        peer_input_anchors = output_anchor.get_peer_input_anchor().copy()
        for input_anchor in peer_input_anchors:
            peer_input_node = input_anchor.node
            input_index = input_anchor.index
            graph.remove_edge(object_node, 0, peer_input_node, input_index)
            graph.add_edge(out_node, 0, peer_input_node, input_index)
            # If peer_input_node is graph.output, rename object_node name
            # and set dequant output name to graph.output.name
            if peer_input_node.type == 'graph_anchor':
                out_node.get_output_anchor(0).set_name(peer_input_node.name)
        # delete deploy dequant nodes
        self._delete_ascend_dequant_node(graph, object_node)

        LOGGER.logd(
            "Replace dequant layer '{}' to fake dequant layer '{}' success!".
            format(object_node.name, \
            '.'.join(object_node.name.split('.')[0:-1]) + '.fakedequant'),
            'ReplaceDequantPass')


def _get_dequant_param(graph, quantized_node, node):
    ''' get essential params for dequant'''
    quantized_layer_name = '.'.join(node.name.split('.')[0:-1])

    weight_anchor = node.get_input_anchor(1)
    dequant_param = weight_anchor.get_peer_output_anchor().node
    fused_quant_param = TensorProtoHelper(dequant_param.proto, dequant_param.model_path).get_data()
    _, deq_scale_value = _split_dequant_param(fused_quant_param)

    dequant_shape = QuantOpInfo.get_dequant_shape(quantized_node)
    deq_scale_value = np.reshape(deq_scale_value, dequant_shape)
    return add_fake_dequant(graph, quantized_layer_name, deq_scale_value)


def _split_dequant_param(fused_quant_param):
    '''split dequant_param to offset_w, n, deq_scale '''
    mask = int('0x0000ff0000000000', 16)
    offset_weight = np.array((np.bitwise_and(fused_quant_param, mask)) >> 40,
                             np.int8)
    mask = int('0x00000000ffffffff', 16)
    deq_scale_value = np.array(np.bitwise_and(fused_quant_param, mask),
                               np.uint32)
    deq_scale_value = np.frombuffer(deq_scale_value, np.float32)

    return offset_weight, deq_scale_value
