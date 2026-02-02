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
from onnx import onnx_pb # pylint: disable=import-error
from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.optimizer.base_fusion_pass import BaseFusionPass
from ...amct_pytorch.optimizer.quant_fusion_pass import QuantFusionPass
from ...amct_pytorch.optimizer.insert_quant_pass import construct_quant_node
from ...amct_pytorch.common.utils.onnx_node_util import AttributeProtoHelper
from ...amct_pytorch.utils.vars import MULT_OUTPUT_TYPES


class MultQuantOptimizerPass(BaseFusionPass):
    """
    Function: Fusion quant_layers that from same input and have
              same scale and offset
    APIs: match_pattern, do_pass
    """
    def __init__(self, records):
        """
        Function: Init MultQuantOptimizerPass object
        Parameters: None
        Return: None
        """
        BaseFusionPass.__init__(self)
        self._records = records
        self._supported_output_node_types = MULT_OUTPUT_TYPES
        self._optimizer_type = ('MaxPool')
        self.dst_type = None

    def match_pattern(self, node):
        """
        Function: Find node that have multiple output quant layer
        Parameters: node: node in graph
        Return: True: node that need to do quant layer fusion operation
        """
        if node.type not in self._supported_output_node_types:
            return False
        for output_anchor in node.output_anchors:
            quant_layer_count = 0
            non_quant_layer_num = 0
            for peer_input_anchor in output_anchor.get_peer_input_anchor():
                peer_node_type = peer_input_anchor.node.type
                if peer_node_type == 'AscendQuant':
                    quant_layer_count += 1
                elif peer_node_type not in self._optimizer_type:
                    return False
                else:
                    non_quant_layer_num += 1
            if quant_layer_count and non_quant_layer_num:
                LOGGER.logd('Find node {} can do optimizer'.format(
                    node.name), 'MultQuantOptimizerPass')
                return True

        return False

    def do_pass(self, graph, object_node, model=None):
        """
        Function: Do quant layer fusion operation.
        Parameters: graph: graph that contains object node
                    object_node: node to process
        Return: None
        """
        for output_anchor in object_node.output_anchors:
            # record one output_anchor's all consumer quant node
            matched, copied_quant_node, non_quant_peer_anchors = \
                self._find_output_need_optimizer(object_node, output_anchor)
            if not matched:
                continue

            copied_quant_node_name = copied_quant_node.get_attr('object_node')
            for peer_input_anchor in non_quant_peer_anchors:
                peer_node = peer_input_anchor.node
                quant_node, anti_node = self.generate_quant_antiquant_node(
                    graph, peer_node, copied_quant_node_name)
                # Insert Quant and AntiQuant between object node and peer node
                graph.remove_edge(object_node, output_anchor.index,
                                  peer_node, peer_input_anchor.index)
                graph.add_edge(object_node, output_anchor.index, quant_node, 0)
                graph.add_edge(quant_node, 0, anti_node, 0)
                graph.add_edge(anti_node, 0,
                               peer_node, peer_input_anchor.index)

        LOGGER.logd('Do concat optimizer for "%s" success!'\
                    % (object_node.name), 'MultQuantOptimizerPass')

    def generate_quant_antiquant_node(self, graph, peer_node,
                                      copied_quant_node_name):
        """Generate AscendQuant and AscendAntiQuant node in pair
        """
        scale = self._records.get(copied_quant_node_name).get('data_scale')
        offset = self._records.get(copied_quant_node_name).get('data_offset')

        quant_proto = construct_quant_node(
            inputs=[peer_node.ori_name + '.quant.input0'],
            outputs=[peer_node.ori_name + '.quant.output0'],
            attrs={
                'scale': 1.0 / scale,
                'offset': offset,
                'dst_type': self.dst_type
            },
            layer_name=peer_node.ori_name)
        anti_quant_proto = construct_anti_quant_node(
            inputs=[peer_node.ori_name + '.quant.input0'],
            outputs=[peer_node.ori_name + '.quant.output0'],
            attrs={
                'scale': scale,
                'offset': offset,
                'dst_type': self.dst_type
            },
            layer_name=peer_node.ori_name)

        quant_node = graph.add_node(quant_proto)
        quant_node.set_attr('object_node', copied_quant_node_name)
        anti_quant_node = graph.add_node(anti_quant_proto)
        anti_quant_node.set_attr('object_node', copied_quant_node_name)

        return quant_node, anti_quant_node

    def _find_output_need_optimizer(self, node, output_anchor):
        """find object node's peer output that can be optimizer"""
        quant_node_list = []
        non_quant_peer_anchors = []
        for peer_input_anchor in output_anchor.get_peer_input_anchor():
            peer_node = peer_input_anchor.node
            if peer_node.type == "AscendQuant":
                attr_helper = AttributeProtoHelper(peer_node.proto)
                self.dst_type = attr_helper.get_attr_value('dst_type').decode()
                quant_node_list.append(peer_node)
                LOGGER.logd('Find node {} has quant output {}'.format(
                    node.name, peer_node.name), 'MultQuantOptimizerPass')
            elif peer_node.type not in self._optimizer_type:
                LOGGER.logd('Find node {} has unsupprted output {}'.format(
                    node.name, peer_node.name), 'MultQuantOptimizerPass')
                return False, None, None
            else:
                LOGGER.logd('Find node {} has supprted output {}'.format(
                    node.name, peer_node.name), 'MultQuantOptimizerPass')
                non_quant_peer_anchors.append(peer_input_anchor)
        # record quant layer with same scale and offset
        fusion_quant_nodes = QuantFusionPass.find_same_quant_node(
            self._records, quant_node_list)
        if len(fusion_quant_nodes) != 1:
            LOGGER.logd('Node {} with different output activation quantize ' \
                'parameters.'.format(node.name))
            return False, None, None

        nodes = fusion_quant_nodes.get(list(fusion_quant_nodes.keys())[0])
        return True, nodes[0], non_quant_peer_anchors


def construct_anti_quant_node(inputs, # pylint: disable=no-member
                              outputs,
                              attrs,
                              layer_name):
    """
    Function: construct anti-quant node in onnx
    Inputs:
        input: a list of inputs' name
        output: a list of outputs' name
        scale: numpy.array,
        offset: numpy.array,
        dst_type: a string
        quantize_layer: a string, layer to be quantized
    """
    node_proto = onnx_pb.NodeProto()

    node_proto.name = layer_name + '.anti_quant'
    node_proto.op_type = 'AscendAntiQuant'
    node_proto.input.extend(inputs) # pylint: disable=E1101
    node_proto.output.extend(outputs) # pylint: disable=E1101

    attr_helper = AttributeProtoHelper(node_proto)
    attr_helper.set_attr_value('scale', 'FLOAT', attrs['scale'])
    attr_helper.set_attr_value('offset', 'FLOAT', attrs['offset'])
    attr_helper.set_attr_value('dst_type', 'STRING', bytes(attrs['dst_type'], 'utf-8'))

    return node_proto
