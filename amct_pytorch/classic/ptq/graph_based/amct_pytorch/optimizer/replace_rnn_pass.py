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
from ...amct_pytorch.common.rnn.gru import BasicGRUInplaceFillWindowCache
from ...amct_pytorch.common.rnn.lstm import BasicLSTMInplaceFillWindowCache
from ...amct_pytorch.common.utils.vars_util import RNN_LAYER_TYPE
from ...amct_pytorch.common.utils.vars_util import RNN_INPUT_ORDER_MAP
from ...amct_pytorch.common.utils.vars_util import RNN_DEQ_SCALE_INDEX
from ...amct_pytorch.configuration.check import check_lstm_limit, check_gru_limit
from ...amct_pytorch.optimizer.base_fusion_pass import BaseFusionPass
from ...amct_pytorch.optimizer.insert_dequant_pass import construct_dequant_param
from ...amct_pytorch.optimizer.insert_dequant_pass import fuse_dequant_param
from ...amct_pytorch.utils.log import LOGGER


class ReplaceRNNPass(BaseFusionPass):
    """
    Function: replace rnn op
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
        Function: Do actually replace rnn op
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
        input_order_map = RNN_INPUT_ORDER_MAP.get(object_node.type)
        # make up input_anchors
        if len(object_node.input_anchors) < len(input_order_map):
            for _ in range(len(input_order_map) - len(object_node.input_anchors)):
                object_node.add_input_anchor('')
        if len(object_node.input_anchors) > len(input_order_map):
            for index in range(len(input_order_map), len(object_node.input_anchors)):
                del object_node.input_anchors[index]

        if object_node.type == 'LSTM':
            new_node_proto = BasicLSTMInplaceFillWindowCache.construct_node_proto(
                object_node.dump_proto(), self.records)
        else:
            new_node_proto = BasicGRUInplaceFillWindowCache.construct_node_proto(
                object_node.dump_proto(), self.records)

        new_node = graph.add_node(new_node_proto)
        new_node.set_attr('object_node', object_node.name)
        # new_node name is modified when add into graph, so need be re-modified
        new_node.set_name(object_node.name)

        # relink input to new node
        for index in range(len(object_node.input_anchors)):
            producer_anchor = object_node.get_input_anchor(index).get_peer_output_anchor()
            # skip optional input
            if not producer_anchor:
                new_node.input_anchors[input_order_map.get(index)].set_name('')
                continue
            producer = producer_anchor.node
            graph.remove_edge(producer, producer_anchor.index, object_node, index)
            graph.add_edge(producer, producer_anchor.index, new_node, input_order_map.get(index))

        # set empty input clean_cache
        new_node.add_input_anchor('')
        # link deq_scale to new node
        deq_scale_node = self.construct_deq_scale_node(graph, object_node)
        new_node.add_input_anchor(deq_scale_node.name)
        graph.add_edge(deq_scale_node, 0, new_node, RNN_DEQ_SCALE_INDEX.get(object_node.type))

        # relink output to new node
        for index in range(len(object_node.output_anchors)):
            consumer_anchors = object_node.get_output_anchor(index).get_peer_input_anchor().copy()
            for consumer_anchor in consumer_anchors:
                consumer = consumer_anchor.node
                graph.remove_edge(object_node, index, consumer, consumer_anchor.index)
                graph.add_edge(new_node, index, consumer, consumer_anchor.index)

        graph.remove_node(object_node)
        LOGGER.logd("Replace RNN node '{}' success.".format(new_node.proto.name), 'ReplaceRNNPass')

    def construct_deq_scale_node(self, graph, object_node):
        """
        Function: construct deq scale node
        Parameters:
            graph: graph structure
            object_node: rnn node
        Return: deq scale node
        """
        scale_d = self.records.get(object_node.name).get('data_scale')
        scale_w_array = self.records.get(object_node.name).get('weight_scale')
        offset_w_array = self.records.get(object_node.name).get('weight_offset')
        deq_scale_x = scale_d * scale_w_array

        scale_h = self.records.get(object_node.name).get('h_scale')
        scale_r_array = self.records.get(object_node.name).get('recurrence_weight_scale')
        offset_r_array = self.records.get(object_node.name).get('recurrence_weight_offset')
        deq_scale_h = scale_h * scale_r_array

        deq_scale = np.concatenate([deq_scale_x, deq_scale_h])
        offset = np.concatenate([offset_w_array, offset_r_array])
        quant_param = fuse_dequant_param(deq_scale, offset)
        deq_scale_proto = construct_dequant_param(quant_param, object_node.name)
        deq_scale_node = graph.add_node(deq_scale_proto)

        return deq_scale_node
