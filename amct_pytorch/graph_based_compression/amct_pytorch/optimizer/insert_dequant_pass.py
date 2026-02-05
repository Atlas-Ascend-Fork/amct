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

from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.utils.onnx_initializer_util import TensorProtoHelper
from ...amct_pytorch.common.utils.onnx_node_util import AttributeProtoHelper
from ...amct_pytorch.common.utils.vars_util import RNN_LAYER_TYPE
from ...amct_pytorch.common.utils.util import cast_to_s19
from ...amct_pytorch.utils.quant_node import QuantOpInfo
from ...amct_pytorch.utils.vars import MATMUL_DEQUANT_AFTER_ADD


class InsertDequantPass(BaseFusionPass):
    """
    Function: Insert AscendDequant
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
        Function: Do actually inserting AscendDequant.
        Parameters:
            graph: graph structure
            object_node: node to process
            model: torch.nn.Module, the model to be modified. if it's
                None, the gaph will be modified.
        Return: None
        """
        # Step1: add dequant_node and dequant_param in graph
        dequant_node, dequant_param = self.insert_layer(
            graph, object_node)
        insert_node = object_node
        # if 'MatMul' has 'Add' bias, insert dequant after 'Add'
        if MATMUL_DEQUANT_AFTER_ADD and object_node.type == 'MatMul' and \
            QuantOpInfo.get_bias_node(object_node) is not None:
            insert_node = object_node.get_consumers(0)[0][0]
        # Step2: Relink nodes in th graph
        output_anchor = insert_node.get_output_anchor(0)
        peer_input_anchors = list()
        for input_anchor in output_anchor.get_peer_input_anchor():
            peer_input_anchors.append(input_anchor)

        for input_anchor in peer_input_anchors:
            peer_input_node = input_anchor.node
            input_index = input_anchor.index
            graph.remove_edge(insert_node, 0, peer_input_node, input_index)
            graph.add_edge(dequant_node, 0, peer_input_node, input_index)
            # If peer_input_node is graph.output, rename insert_node name
            # and set dequant output name to graph.output.name
            if peer_input_node.type == 'graph_anchor':
                insert_node.get_output_anchor(0).set_name('{}_output0'.format(
                    insert_node.name))
                dequant_node.get_output_anchor(0).set_name(
                    peer_input_node.name)

        graph.add_edge(insert_node, 0, dequant_node, 0)
        graph.add_edge(dequant_param, 0, dequant_node, 1)

        LOGGER.logd(
            f'Add dequant "{dequant_node.name}" to "{object_node.name}" success.', 'InsertDequantPass')

    def insert_layer(self, graph, object_node):
        """
        Function: Insert a node for dequant_param in the graph.
        Parameters:
            graph: graph structure
            layer_name: a string, name of layer for dequant_param
        Returns:
            dequant_node: a dequant node
            dequant_param: a node containing dequant_param
        """
        # concat dequant param to uint64
        deq_scale = self.records.get(object_node.name).get('weight_scale') * \
            self.records.get(object_node.name).get('data_scale')
        if self.records.get(object_node.name).get('fakequant_precision_mode') == 'FORCE_FP16_QUANT':
            deq_scale = np.vectorize(cast_to_s19)(deq_scale).astype(np.float32)
        quant_param = fuse_dequant_param(
            deq_scale,
            self.records.get(object_node.name).get('weight_offset'))
        # add quant_param as initializer in graph
        if object_node.has_attr('op_data_type'):
            op_data_type = object_node.get_attr('op_data_type')
        else:
            op_data_type = 'float32'
        dtype = AttributeProtoHelper.ge_dtype_map.get(op_data_type, 0)
        initializer_proto = construct_dequant_param(quant_param,
                                                    object_node.name)
        dequant_param = graph.add_node(initializer_proto)
        # add dequant ad node in graph
        node_proto = construct_dequant_node(
            inputs=[
                object_node.name + '.dequant.input0', \
                object_node.name + '.dequant.param'
            ],
            outputs=[object_node.name + '.dequant.output0'],
            layer_name=object_node.name,
            dtype=dtype)
        dequant_node = graph.add_node(node_proto)

        return dequant_node, dequant_param


def fuse_dequant_param(float32_deq_scale, int8_offset_w):
    """Fused dequant scale, offset_w and shift_bits to uint64 data
    """
    uint32_deq_scale = np.frombuffer(float32_deq_scale, np.uint32)
    uint8_offset_w = np.frombuffer(int8_offset_w, np.uint8)

    # fuse parameter
    # |-----------------|47:40|--------------------------|31:0|
    #                  offset_w [8]                   deq_scale [32]
    scale_length = float32_deq_scale.size
    quant_param = np.zeros(scale_length, dtype=np.uint64)
    for index in range(scale_length):
        quant_param[index] = uint8_offset_w[index]
        quant_param[index] = (quant_param[index] << np.uint32(32))\
                                + uint32_deq_scale[index]
    return quant_param


def construct_dequant_param(dequant_param, # pylint: disable=no-member
                            layer_name):
    """ Construct a initializer for dequant_param
    """
    initializer_proto = onnx_pb.TensorProto()
    initializer_proto.name = layer_name + '.dequant.param'
    tensor_helper = TensorProtoHelper(initializer_proto)
    tensor_helper.set_data(dequant_param, 'UINT64', [len(dequant_param)])

    return initializer_proto


def construct_dequant_node(inputs, # pylint: disable=no-member
                           outputs,
                           layer_name,
                           dtype):
    """
    Function: construct quant node in onnx
    Inputs:
        input: a list of inputs' name
        output: a list of outputs' name
        quantize_layer: a string, layer to be quantized
    Return: node_proto: AscendDequant's onnx_pb.NodeProto
    """
    node_proto = onnx_pb.NodeProto()

    node_proto.name = layer_name + '.dequant'
    node_proto.op_type = 'AscendDequant'

    node_proto.input.extend(inputs) # pylint: disable=E1101
    node_proto.output.extend(outputs) # pylint: disable=E1101
    attr_helper = AttributeProtoHelper(node_proto)
    attr_helper.set_attr_value('dtype', 'INT', dtype)
    return node_proto
