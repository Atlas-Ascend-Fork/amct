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
from ...amct_pytorch.optimizer.base_fusion_pass import BaseFusionPass
from ...amct_pytorch.module.quant_module import add_fakequant

from ...amct_pytorch.common.utils.onnx_node_util import AttributeProtoHelper
from ...amct_pytorch.utils.log import LOGGER


class ReplaceQuantPass(BaseFusionPass):
    """
    Function: Replace AscendQuant to fakeqaunt 'Quant' with onnx's ops
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
    def _generate_fake_quant_nodes(graph, object_node):
        """Generate fake Quant nodes"""
        attr_helper = AttributeProtoHelper(object_node.proto)
        num_bits = int(str(attr_helper.get_attr_value('dst_type'), encoding="utf-8")[3:])
        enter_node, out_node = add_fakequant(
            graph,
            object_node.name,
            attr_helper.get_attr_value('scale'),
            attr_helper.get_attr_value('offset'),
            num_bits)
        return enter_node, out_node

    def match_pattern(self, node):
        """
        Function: Match the AscendQuant node
        Parameters: node: node in graph to be matched
        Return: True: matched
                False: mismatch
        """
        if node.type != 'AscendQuant':
            return False

        return True

    def do_pass(self, graph, object_node, model=None):
        """
        Function: Do actually replacing AscendQuant to fakeqaunt 'Quant'
            with onnx's ops
        Parameters: graph: graph structure
                    object_node: node to process
                    model: torch.nn.Module, the model to be modified. if it's
                        None, the gaph will be modified.
        Return: None
        """
        # Step1: add a new_node
        enter_node, out_node = \
            self._generate_fake_quant_nodes(graph, object_node)

        # Step2: Relink nodes in th graph
        # remove input links
        input_anchor = object_node.get_input_anchor(0)
        peer_output_anchor = input_anchor.get_peer_output_anchor()
        peer_node = peer_output_anchor.node
        peer_output_anchor_index = peer_output_anchor.index
        graph.remove_edge(peer_node, peer_output_anchor_index, object_node, 0)
        graph.add_edge(peer_node, peer_output_anchor_index, enter_node, 0)
        # remove output links
        output_anchor = object_node.get_output_anchor(0)

        peer_input_anchors = list()
        for input_anchor in output_anchor.get_peer_input_anchor():
            peer_input_anchors.append(input_anchor)

        for input_anchor in peer_input_anchors:
            peer_input_node = input_anchor.node
            input_index = input_anchor.index
            graph.remove_edge(object_node, 0, peer_input_node, input_index)
            graph.add_edge(out_node, 0, peer_input_node, input_index)

        graph.remove_node(object_node)

        LOGGER.logd(
            "Replace quant layer '{}' to fake quant layer '{}' success!".
            format(object_node.name,
                   object_node.name + '.fakequant'), 'ReplaceQuantPass')
