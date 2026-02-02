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

from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.optimizer.base_fusion_pass import BaseFusionPass


class QuantFusionPass(BaseFusionPass):
    """
    Function: Fusion quant_layers that from same input and have
              same scale and offset
    APIs: match_pattern, do_pass
    """
    def __init__(self, records):
        """
        Function: Init QuantFusionPass object
        Parameters: None
        Return: None
        """
        BaseFusionPass.__init__(self)
        self.records = records

    @staticmethod
    def match_pattern(node):
        """
        Function: Find node that have multiple output quant layer
        Parameters: node: node in graph
        Return: True: node that need to do quant layer fusion operation
        """
        for output_anchor in node.output_anchors:
            quant_layer_count = 0
            for peer_input_anchor in output_anchor.get_peer_input_anchor():
                if peer_input_anchor.node.type == 'AscendQuant':
                    quant_layer_count += 1
            if quant_layer_count > 1:
                return True

        return False

    @staticmethod
    def find_same_quant_node(records, quant_node_list):
        """find quant node with same scale and offset"""
        fusion_quant_nodes = {}
        for node in quant_node_list:
            object_name = node.get_attr('object_node')
            scale_d = records.get(object_name).get('data_scale')
            offset_d = records.get(object_name).get('data_offset')
            act_type = records.get(object_name).get('act_type', 'INT8')
            encode_id = QuantFusionPass.encode_scale_offset(scale_d, offset_d) + act_type
            if encode_id not in fusion_quant_nodes:
                fusion_quant_nodes[encode_id] = [node]
            else:
                fusion_quant_nodes.get(encode_id).append(node)
        return fusion_quant_nodes

    @staticmethod
    def encode_scale_offset(scale, offset):
        """encode scale and offset to unique key"""
        int32_offset = np.int32(offset)
        uint32_offset = np.frombuffer(int32_offset, np.uint32)

        float_scale = np.float32(scale)
        uint32_scale = np.frombuffer(float_scale, np.uint32)

        encode_result = np.uint64(0)
        encode_result = np.uint64(uint32_scale) << np.uint32(32) | np.uint64(
            uint32_offset)

        return str(encode_result)

    def run(self, graph, model):
        """
        Function: The main control of QuantFusionPass.
        Parameters: graph: amct graph that contains object node
                    model: torch model
        Return: None
        """
        # Step1: do match pattern to get matched node
        matched_nodes = []
        for node in graph.nodes + graph._in_out_nodes:
            if self.match_pattern(node):
                matched_nodes.append(node)

        # Step2: do each matched node fusion operation
        for node in matched_nodes:
            self.do_pass(graph, node)
        # Step3: do topological sort
        graph.topologic_sort()

    def do_pass(self, graph, object_node, model=None):
        """
        Function: Do quant layer fusion operation.
        Parameters: graph: graph that contains object node
                    object_node: node to process
        Return: None
        """
        for output_anchor in object_node.output_anchors:
            # record one output_anchor's all consumer quant node
            quant_node_list = []
            for peer_input_anchor in output_anchor.get_peer_input_anchor():
                if peer_input_anchor.node.type == "AscendQuant":
                    quant_node_list.append(peer_input_anchor.node)
            # record quant layer with same scale and offset
            fusion_quant_nodes = self.find_same_quant_node(
                self.records, quant_node_list)
            # do quant fusion
            for _, nodes in fusion_quant_nodes.items():
                if len(nodes) < 2:
                    continue
                # retain first quant node, remove the rest
                for node in nodes[1:]:
                    # remove input edge at first
                    graph.remove_edge(object_node, output_anchor.index,
                                      node, 0)
                    # relink from node -> consumer to first_node -> consumer
                    node_output = node.get_output_anchor(0)
                    peers = node_output.get_peer_input_anchor().copy()
                    for peer_input_anchor in peers:
                        dst_node = peer_input_anchor.node
                        dst_anchor_index = peer_input_anchor.index
                        graph.remove_edge(node, 0, dst_node, dst_anchor_index)
                        graph.add_edge(nodes[0], 0, dst_node, dst_anchor_index)
                    graph.remove_node(node)
                    LOGGER.logd("Remove node {} from graph".format(
                        node.name), 'QuantFusionPass')

        LOGGER.logd(f'Do fusion same quant layer from "{object_node.name}" success!', 'QuantFusionPass')

