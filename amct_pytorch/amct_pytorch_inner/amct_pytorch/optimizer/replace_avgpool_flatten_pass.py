#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Replace "GlobalAveragePool + Flatten" node in graph.

"""
from onnx import onnx_pb

from amct_pytorch.optimizer.base_fusion_pass import BaseFusionPass
from amct_pytorch.utils.log import LOGGER
from amct_pytorch.common.utils.onnx_node_util import AttributeProtoHelper


class ReplaceAvgpoolFlattenPass(BaseFusionPass):
    """
    Function: Replace "GlobalAveragePool + Flatten" node in graph.
    APIs: match_pattern, do_pass
    """
    def __init__(self):
        """
        Function: init object
        Parameter: None
        Return: None
        """
        BaseFusionPass.__init__(self)

    @staticmethod
    def match_pattern(node):
        """
        Function: Match the GlobalAveragePool node to be replaced in graph
        Parameters: node: node in graph to be matched
        Return: True: matched / False: mismatch
        """
        if node.type != 'GlobalAveragePool':
            return False

        consumers, _ = node.get_consumers(0)
        if len(consumers) != 1 or consumers[0].type != 'Flatten':
            LOGGER.logd("node {} match_pattern fail for it must have only one consumer Flatten.".format(node.name),
                        'ReplaceAvgpoolFlattenPass')
            return False

        consumer = consumers[0]
        attr_helper = AttributeProtoHelper(consumer.proto)
        axis = 1
        if attr_helper.has_attr('axis'):
            axis = attr_helper.get_attr_value('axis')
        if axis != 1:
            LOGGER.logd("node {} match_pattern fail for its consumer Flatten has axis {} not equal to 1."
                        .format(node.name, axis), 'ReplaceAvgpoolFlattenPass')
            return False

        return True

    @staticmethod
    def do_pass(graph, object_node):
        """
        Function: Do actually replacement.
        Parameters:
        graph: graph structure
        object_node: node to process
        Return: None
        """
        LOGGER.logd("Doing: delete node {} in graph.".format(object_node.name),
                    'ReplaceAvgpoolFlattenPass')
        consumers, _ = object_node.get_consumers(0)
        consumer = consumers[0]

        proto_p1ton1 = ReplaceAvgpoolFlattenPass.construct_d4tod2('.'.join([object_node.name, consumer.name]))
        node_p1ton1 = graph.add_node(proto_p1ton1)
        graph.insert_node_before(node_p1ton1, 0, 0, object_node, 0)

        graph.delete_node(object_node, 0, 0)
        graph.remove_node(object_node)
        graph.delete_node(consumer, 0, 0)
        graph.remove_node(consumer)

        LOGGER.logd("Finished: delete node {} in graph.".format(object_node.name),
                    'ReplaceAvgpoolFlattenPass')

    @staticmethod
    def construct_d4tod2(layer_name):
        """
        Function: construct D4toD2 op, which means the data is from [n, c, *] to [n, c]
        param:layer_name: name for node
        return:node_proto, onnx_pb.NodeProto
        """
        node_proto = onnx_pb.NodeProto()
        node_proto.name = layer_name
        node_proto.op_type = 'D4toD2'
        node_proto.doc_string = 'input[1] is equal to input[-1]'
        node_proto.input.extend(['.'.join([layer_name, 'input0'])])
        node_proto.output.extend(['.'.join([layer_name, 'output0'])])

        return node_proto
