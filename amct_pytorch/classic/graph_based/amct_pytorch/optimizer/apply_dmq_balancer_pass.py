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
from onnx import onnx_pb

from ...amct_pytorch.optimizer.base_fusion_pass import BaseFusionPass
from ...amct_pytorch.common.utils.onnx_node_util import AttributeProtoHelper
from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.utils.onnx_initializer_util import TensorProtoHelper
from ...amct_pytorch.utils.quant_node import QuantOpInfo
from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.utils.weight_quant_api import get_deconv_group
from ...amct_pytorch.utils.weight_quant_api import adjust_conv_weight_shape

RAW_DATA_TYPE_SET = (np.float16, )
TENSOR_BALANCE_FACTOR = 'tensor_balance_factor'
MUL = 'mul'

# enum value of tensor DataType in onnx.proto
PROTO_TYPE_FLOAT = 1
PROTO_TYPE_FLOAT16 = 10


class ApplyDMQBalancerPass(BaseFusionPass):
    """
    Function: Apply DMQBalancer
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
    def _construct_factor_node(layer_name, tensor_balance_factor, factor_data_type, reshape_size):
        ''' construct tensor_balance_factor op'''
        node_proto = onnx_pb.NodeProto()
        node_proto.name = '.'.join([layer_name, TENSOR_BALANCE_FACTOR])
        node_proto.op_type = 'Constant'
        node_proto.doc_string = TENSOR_BALANCE_FACTOR
        node_proto.output.extend(
            ['.'.join([layer_name, TENSOR_BALANCE_FACTOR, 'output0'])])

        AttributeProtoHelper(node_proto).set_attr_value('value', 'TENSOR', None)
        attr = node_proto.attribute[0]
        if factor_data_type == 'FLOAT16':
            tensor_balance_factor = tensor_balance_factor.astype(np.float16)
        TensorProtoHelper(attr.t).set_data(tensor_balance_factor, factor_data_type, reshape_size)

        return node_proto

    @staticmethod
    def _construct_mul_node(layer_name):
        ''' construct mul op'''
        node_proto = onnx_pb.NodeProto()
        node_proto.name = '.'.join([layer_name, MUL])
        node_proto.op_type = 'Mul'
        node_proto.doc_string = 'mul of the dmq_balancer'
        node_proto.input.extend([
            '.'.join([layer_name, MUL, 'input0']),
            '.'.join([layer_name, TENSOR_BALANCE_FACTOR, 'input1'])
        ])
        node_proto.output.extend(
            ['.'.join([layer_name, MUL, 'output0'])])

        return node_proto

    def match_pattern(self, node):
        """
        Function: Match pattern of node to be quantized in graph
        Parameters: node: node in graph to be matched
        Return: True: matched
                False: mismatch
        """
        if node.name not in self.records or node.type == 'AveragePool':
            return False
        if not self.records.get(node.name).get(TENSOR_BALANCE_FACTOR):
            return False
        return True

    def do_pass(self, graph, object_node, model=None):
        """
        Function: Do actually inserting DMQBalancer.
        Parameters:
            graph: graph structure
            object_node: node to process
            model: torch.nn.Module, the model to be modified. if it's
                None, the gaph will be modified.
        Return: None
        """
        tensor_balance_factor = self.records.get(object_node.name).get(TENSOR_BALANCE_FACTOR)
        tensor_balance_factor = np.array(tensor_balance_factor, np.float32)

        weight_param = QuantOpInfo.get_weight_node(object_node)
        weight_helper = TensorProtoHelper(weight_param.proto, weight_param.model_path)
        weight = weight_helper.get_data()
        weight_shape = weight.shape

        if weight_helper.tensor.data_type == PROTO_TYPE_FLOAT:
            factor_data_type = 'FLOAT'
        elif weight_helper.tensor.data_type == PROTO_TYPE_FLOAT16:
            factor_data_type = 'FLOAT16'

        # Apply tensor_balance_factor to activation
        object_name = object_node.name
        # step0: broadcast tensor_balance_factor to activation shape
        if len(weight_shape) == 4:
            reshape_size = [1, len(tensor_balance_factor), 1, 1]
        elif len(weight_shape) == 5:
            reshape_size = [1, len(tensor_balance_factor), 1, 1, 1]
        elif len(weight_shape) == 3:
            reshape_size = [1, len(tensor_balance_factor), 1]
        else:
            reshape_size = [1, len(tensor_balance_factor)]
        # Step1: add a new_node
        tensor_balance_factor = 1.0 / tensor_balance_factor
        tensor_balance_factor_node = graph.add_node(ApplyDMQBalancerPass._construct_factor_node(
            object_name, tensor_balance_factor, factor_data_type, reshape_size))
        mul_node = graph.add_node(ApplyDMQBalancerPass._construct_mul_node(object_name))

        # Step2: Relink nodes in th graph
        # remove output links
        input_anchor = object_node.get_input_anchor(0)
        peer_output_anchor = input_anchor.get_peer_output_anchor()
        peer_node = peer_output_anchor.node
        peer_output_anchor_index = peer_output_anchor.index
        graph.remove_edge(peer_node, peer_output_anchor_index,
                          object_node, 0)
        # add links
        graph.add_edge(peer_node, peer_output_anchor_index, mul_node, 0)
        graph.add_edge(tensor_balance_factor_node, 0, mul_node, 1)
        graph.add_edge(mul_node, 0, object_node, 0)

        LOGGER.logd("Insert dmq_balancer layer '{}' before '{}' " \
            "success!".format(mul_node.name, object_name), \
                              'ApplyDMQBalancerPass')