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

Delete 'Resize' node in graph id 'Resize' will change Channel(N, C, **)

"""

from amct_pytorch.optimizer.base_fusion_pass import BaseFusionPass
from amct_pytorch.utils.log import LOGGER
from amct_pytorch.utils.quant_node import QuantOpInfo


class DeleteResizePass(BaseFusionPass):
    """
    Function: Delete 'Resize' node in graph.
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
        Function: Match the node to be quantized in graph
        Parameters: node: node in graph to be matched
        Return: True: matched / False: mismatch
        """
        if node.type != 'Resize':
            return False

        producer, _ = node.get_producer(2)
        if producer.type not in ['initializer', 'Constant']:
            return False

        scales = QuantOpInfo.get_node_value(producer)
        if len(scales) != 4 or scales[1] != 1:
            return False

        return True

    @staticmethod
    def do_pass(graph, object_node, model=None):
        """
        Function: Do actual quantization and node's weight is changed to int8.
        Parameters:
        graph: graph structure
        object_node: node to process
        Return: None
        """
        LOGGER.logd("Doing: delete node {} in graph.".format(object_node.name),
                     'DeleteResizePass')

        roi_node, _ = object_node.get_producer(1)
        scales_node, _ = object_node.get_producer(2)
        graph.remove_edge(roi_node, 0, object_node, 1)
        graph.remove_node(roi_node)
        graph.remove_edge(scales_node, 0, object_node, 2)
        graph.remove_node(scales_node)
        graph.delete_node(object_node, 0, 0)
        graph.remove_node(object_node)

        LOGGER.logd("Finished: delete node {} in graph.".format(object_node.name),
                     'DeleteResizePass')
