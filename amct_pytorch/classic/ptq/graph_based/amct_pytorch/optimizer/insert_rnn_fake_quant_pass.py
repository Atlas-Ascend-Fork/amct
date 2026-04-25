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

from ...amct_pytorch.common.utils.onnx_node_util import AttributeProtoHelper
from ...amct_pytorch.common.utils.vars_util import RNN_LAYER_TYPE
from ...amct_pytorch.configuration.check import check_lstm_limit, check_gru_limit
from ...amct_pytorch.optimizer.base_fusion_pass import BaseFusionPass
from ...amct_pytorch.optimizer.insert_quant_pass import construct_quant_node
from ...amct_pytorch.optimizer.mult_output_with_quant_optimizer import construct_anti_quant_node
from ...amct_pytorch.utils.log import LOGGER
from ...amct_pytorch.utils.quant_node import QuantOpInfo


class InsertRNNFakeQuantPass(BaseFusionPass):
    """
    Function: insert quant and anti-quant op in graph
    APIs: match_pattern, do_pass
    """
    def __init__(self, records):
        """
        Function: init object
        Parameter:
            records: dict including quant factors
        Return: None
        """
        super().__init__()
        self.records = records

    def match_pattern(self, node):
        """
        Function: Match pattern of node which type is rnn and name in record
        Parameters: 
            node: node in graph to be matched
        Return: True: matched
                False: mismatch
        """
        if node.type not in RNN_LAYER_TYPE:
            return False
        if node.name not in self.records:
            return False
        return True

    def do_pass(self, graph, object_node, model=None):
        """
        Function: Do actually insert quant and anti-quant op in graph
        Parameters:
            graph: graph structure
            object_node: node to process
        Return: None
        """
        if object_node.type == 'LSTM' and not check_lstm_limit(object_node):
            raise RuntimeError("Layer {} don't support quantization, "
                "but in record file.".format(object_node.name))
        if object_node.type == 'GRU' and not check_gru_limit(object_node):
            raise RuntimeError("Layer {} don't support quantization, "
                "but in record file.".format(object_node.name))

        quant_indexs = QuantOpInfo.get_quant_index(object_node)
        act_index0 = quant_indexs.get('act_index')
        act_index1 = quant_indexs.get('initial_h_index')

        # insert quant and anti-quant node before rnn node in act_index0 port
        quant_node0, antiquant_node0 = self.generate_fake_quant_node(graph, object_node, act_index0)
        graph.insert_node_before(quant_node0, 0, 0, object_node, act_index0)
        graph.insert_node_before(antiquant_node0, 0, 0, object_node, act_index0)

        # insert quant and anti-quant node before rnn node in act_index1 port
        quant_node1, antiquant_node1 = self.generate_fake_quant_node(graph, object_node, act_index1)
        graph.insert_node_before(quant_node1, 0, 0, object_node, act_index1)
        graph.insert_node_before(antiquant_node1, 0, 0, object_node, act_index1)

        LOGGER.logd("Insert quant and antiquant layer before '{}' success.".format(
            object_node.name), 'InsertRNNFakeQuantPass')

    def generate_fake_quant_node(self, graph, object_node, act_index):
        """
        Function: generate quant node and anti-quant node
        Parameters:
            graph: graph structure
            object_node: node to process
            act_index: index of activation port
        Return: quant node, anti-quant node
        """
        if act_index == 0:
            scale_d = self.records.get(object_node.name).get('data_scale')
            offset_d = self.records.get(object_node.name).get('data_offset')
        elif act_index == QuantOpInfo.get_quant_index(object_node).get('initial_h_index'):
            scale_d = self.records.get(object_node.name).get('h_scale')
            offset_d = self.records.get(object_node.name).get('h_offset')
        quant_bit = QuantOpInfo.get_dst_num_bits(self.records, object_node.name, 'act')

        # generate quant node
        quant_attrs = {
            'scale': 1.0 / scale_d,
            'offset': offset_d,
            'dst_type': 'INT{:d}'.format(quant_bit)
        }
        quant_proto = construct_quant_node(
            ['.'.join([object_node.name, 'quant', 'input' + str(act_index)])],
            ['.'.join([object_node.name, 'quant', 'output' + str(act_index)])],
            quant_attrs,
            object_node.name)
        quant_node = graph.add_node(quant_proto)
        quant_node.set_attr('object_node', object_node.name)

        # generate anti-quant node
        antiquant_attrs = {
            'scale': scale_d,
            'offset': offset_d,
            'dst_type': 'INT{:d}'.format(quant_bit),
            'op_dtype': AttributeProtoHelper.ge_dtype_map.get('FLOAT32', 0)
        }
        antiquant_proto = construct_anti_quant_node(
            ['.'.join([object_node.name, 'antiquant', 'input' + str(act_index)])],
            ['.'.join([object_node.name, 'antiquant', 'output' + str(act_index)])],
            antiquant_attrs,
            object_node.name)
        antiquant_node = graph.add_node(antiquant_proto)
        antiquant_node.set_attr('object_node', object_node.name)

        return quant_node, antiquant_node
