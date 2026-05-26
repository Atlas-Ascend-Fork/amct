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

from onnx import onnx_pb # pylint: disable=import-error
from ...amct_pytorch.optimizer.base_fusion_pass import BaseFusionPass
from ...amct_pytorch.custom_op.arq.arq import weight_quant_np
from ...amct_pytorch.common.utils.onnx_node_util import AttributeProtoHelper
from ...amct_pytorch.utils.onnx_initializer_util import TensorProtoHelper
from ...amct_pytorch.utils.quant_node import QuantOpInfo
from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.utils.weight_quant_api import get_deconv_group
from ...amct_pytorch.utils.weight_quant_api import adjust_deconv_weight_shape
from ...amct_pytorch.common.utils.vars_util import RNN_LAYER_TYPE

WEIGHT_QUANT = 'weight_quant'


class InsertWeightQuantPass(BaseFusionPass):
    """
    Function: Insert AscendWeightQuant
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
        if node.name not in self.records or node.type in ['AveragePool']:
            return False
        return True

    def do_pass(self, graph, object_node, model=None):
        """
        Function: Do actually inserting AscendWeightQuant.
        Parameters:
            graph: graph structure
            object_node: node to process
            model: torch.nn.Module, the model to be modified. if it's
                None, the gaph will be modified.
        Return: None
        """
        # Determine the quantization type based on the value of dst_type and obtain the corresponding num_bits
        if self.records.get(object_node.name).get('dst_type') == 'UNSET':
            num_bits = QuantOpInfo.get_dst_num_bits(self.records, object_node.name, 'wts')
        else:
            num_bits = QuantOpInfo.get_dst_num_bits(self.records, object_node.name)
        weight_param = QuantOpInfo.get_weight_node(object_node)

        weight_helper = TensorProtoHelper(weight_param.proto, weight_param.model_path)
        weight = weight_helper.get_data().astype(np.float32)

        if object_node.type == 'ConvTranspose':
            group = get_deconv_group(object_node)
            weight = adjust_deconv_weight_shape(group, weight)
        scale_w = self.records.get(object_node.name).get('weight_scale')
        offset_w = self.records.get(object_node.name).get('weight_offset')
        int8_weight = weight_quant_np(
            weight,
            scale_w,
            offset_w,
            num_bits)
        if object_node.type == 'ConvTranspose':
            group = get_deconv_group(object_node)
            int8_weight = adjust_deconv_weight_shape(group, int8_weight)
        int8_weight = int8_weight.reshape([-1])

        weight_helper.clear_data()
        weight_helper.set_data(int8_weight, 'INT8')

        if object_node.type in RNN_LAYER_TYPE:
            self.quant_recurrence_weight(object_node)
        if num_bits == 4:
            # Step1: add a new_node
            node_proto = construct_weight_quant_node(
                inputs=['.'.join([object_node.name, WEIGHT_QUANT, 'input0']),
                        '.'.join([object_node.name, WEIGHT_QUANT, 'input1'])],
                outputs=['.'.join([object_node.name, WEIGHT_QUANT, 'output0'])],
                attrs={
                    'dst_type': 'INT{:d}'.format(num_bits)
                },
                layer_name=object_node.name)
            weight_quant_node = graph.add_node(node_proto)
            weight_quant_node.set_attr('object_node', object_node.name)
            weight_offset_node = graph.add_node(construct_weight_offset(
                layer_name=object_node.name,
                weight_offset=offset_w))

            # Step2: Relink nodes in th graph
            # remove output links
            input_anchor = object_node.get_input_anchor(1)
            peer_output_anchor = input_anchor.get_peer_output_anchor()
            peer_node = peer_output_anchor.node

            peer_output_anchor_index = peer_output_anchor.index
            graph.remove_edge(peer_node, peer_output_anchor_index,
                            object_node, 1)
            # add links
            graph.add_edge(peer_node, peer_output_anchor_index, weight_quant_node, 0)
            graph.add_edge(weight_offset_node, 0, weight_quant_node, 1)
            graph.add_edge(weight_quant_node, 0, object_node, 1)

        LOGGER.logd("Quant weight from int32 to int8 for layer '{}' " \
            "success!".format(object_node.name), 'WeightQuantPass')

    def quant_recurrence_weight(self, object_node):
        """
        Function: quant recurrence weight of rnn op
        Inputs:
            object_node: node to process
        Returns: None
        """
        recurrence_weight_node = QuantOpInfo.get_recurrence_weight_node(object_node)
        recurrence_weight_tensor = QuantOpInfo.get_node_tensor(recurrence_weight_node)
        recurrence_weight_helper = TensorProtoHelper(recurrence_weight_tensor,
                                                     model_path=recurrence_weight_node.model_path)
        recurrence_weight = recurrence_weight_helper.get_data()

        scale_r = self.records.get(object_node.name).get('recurrence_weight_scale')
        offset_r = self.records.get(object_node.name).get('recurrence_weight_offset')

        int8_recurrence_weight = weight_quant_np(recurrence_weight,
                                                 scale_r,
                                                 offset_r,
                                                 8)
        int8_recurrence_weight = int8_recurrence_weight.reshape([-1])
        recurrence_weight_helper.clear_data()
        recurrence_weight_helper.set_data(int8_recurrence_weight, 'INT8')


def construct_weight_quant_node(inputs, # pylint: disable=no-member
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

    node_proto.name = '.'.join([layer_name, WEIGHT_QUANT])
    node_proto.op_type = 'AscendWeightQuant'
    node_proto.input.extend(inputs) # pylint: disable=E1101
    node_proto.output.extend(outputs) # pylint: disable=E1101

    attr_helper = AttributeProtoHelper(node_proto)
    attr_helper.set_attr_value('dst_type', 'STRING', bytes(attrs['dst_type'], 'utf-8'))

    return node_proto


def construct_weight_offset(layer_name, weight_offset):
    '''construct weight_offset op '''
    node_proto = onnx_pb.NodeProto()
    node_proto.name = '.'.join([layer_name, 'weight_offset'])
    node_proto.op_type = 'Constant'
    node_proto.doc_string = 'weight offset'
    node_proto.output.extend( # pylint: disable=E1101
        ['.'.join([layer_name, 'weight_offset', 'output0'])])

    AttributeProtoHelper(node_proto).set_attr_value('value', 'TENSOR', None)
    attr = node_proto.attribute[0] # pylint: disable=E1101
    TensorProtoHelper(attr.t).set_data(
        weight_offset, 'INT8', list(weight_offset.shape))

    return node_proto
