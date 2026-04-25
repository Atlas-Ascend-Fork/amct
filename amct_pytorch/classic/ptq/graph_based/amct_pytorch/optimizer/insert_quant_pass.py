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
from ...amct_pytorch.optimizer.base_fusion_pass import BaseFusionPass
from ...amct_pytorch.common.utils.onnx_node_util import AttributeProtoHelper
from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.utils.quant_node import QuantOpInfo
from ...amct_pytorch.common.utils.vars_util import RNN_LAYER_TYPE


class InsertQuantPass(BaseFusionPass):
    """
    Function: Insert AscendQuant
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
        if node.type in RNN_LAYER_TYPE:
            return False
        return True

    def do_pass(self, graph, object_node, model=None):
        """
        Function: Do actually inserting AscendQuant.
        Parameters:
            graph: graph structure
            object_node: node to process
            model: torch.nn.Module, the model to be modified. if it's
                None, the gaph will be modified.
        Return: None
        """
        object_name = object_node.name
        # Determine the quantization type based on the value of dst_type and obtain the corresponding num_bits
        if self.records.get(object_name).get('dst_type') == 'UNSET':
            num_bits = QuantOpInfo.get_dst_num_bits(self.records, object_name, 'act')
        else:
            num_bits = QuantOpInfo.get_dst_num_bits(self.records, object_name)
        # Step1: add a new_node
        node_proto = construct_quant_node(
            inputs=['.'.join([object_node.name, 'quant', 'input0'])],
            outputs=['.'.join([object_node.name, 'quant', 'output0'])],
            attrs={
                'scale':
                1.0 / self.records.get(object_node.name).get('data_scale'),
                'offset':
                self.records.get(object_node.name).get('data_offset'),
                'dst_type': 'INT{:d}'.format(num_bits)
            },
            layer_name=object_node.name)
        quant_node = graph.add_node(node_proto)
        quant_node.set_attr('object_node', object_node.name)

        # Step2: Relink nodes in th graph
        # remove output links
        input_anchor = object_node.get_input_anchor(0)
        peer_output_anchor = input_anchor.get_peer_output_anchor()
        peer_node = peer_output_anchor.node
        if object_node.type == 'AveragePool' and peer_node.type == 'Pad':
            output_anchor = peer_node.get_output_anchor(0)
            if len(output_anchor.get_peer_input_anchor()) == 1:
                # If AveragePool with pool, add quant before it
                if peer_node.name == '%s_pad' % (object_node.name):
                    object_node = peer_node
                    input_anchor = object_node.get_input_anchor(0)
                    peer_output_anchor = input_anchor.get_peer_output_anchor()
                    peer_node = peer_output_anchor.node

        peer_output_anchor_index = peer_output_anchor.index
        graph.remove_edge(peer_node, peer_output_anchor_index,
                          object_node, 0)
        # add links
        graph.add_edge(peer_node, peer_output_anchor_index, quant_node, 0)
        graph.add_edge(quant_node, 0, object_node, 0)

        LOGGER.logd("Insert quant layer '{}' before '{}' " \
            "success!".format(quant_node.name, object_name), \
                              'InsertQuantPass')


def construct_quant_node(inputs, # pylint: disable=no-member
                         outputs,
                         attrs,
                         layer_name):
    """
    Function: construct quant node in onnx
    Inputs:
        input: a list of inputs' name
        output: a list of outputs' name
        attrs: a dict of attrs including
            scale: numpy.array
            offset: numpy.array
            dst_type: a string
        quantize_layer: a string, layer to be quantized
    """
    node_proto = onnx_pb.NodeProto()

    node_proto.name = '.'.join([layer_name, 'quant'])
    node_proto.op_type = 'AscendQuant'
    node_proto.input.extend(inputs) # pylint: disable=E1101
    node_proto.output.extend(outputs) # pylint: disable=E1101

    attr_helper = AttributeProtoHelper(node_proto)
    attr_helper.set_attr_value('scale', 'FLOAT', attrs['scale'])
    attr_helper.set_attr_value('offset', 'FLOAT', attrs['offset'])
    attr_helper.set_attr_value('dst_type', 'STRING', bytes(attrs['dst_type'], 'utf-8'))

    return node_proto
