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

Delete circular padding pattern in graph.

"""

from amct_pytorch.optimizer.base_fusion_pass import BaseFusionPass
from amct_pytorch.utils.log import LOGGER

PADDING_TARGET_NODE = 'padding_target_node'


class DeleteCircularPaddingPass(BaseFusionPass):
    """
    Function: Delete circular padding pattern in graph.
    APIs: match_pattern, do_pass
    """
    def __init__(self):
        """
        Function: init object
        Parameter: None
        Return: None
        """
        BaseFusionPass.__init__(self)
        self.circular_pad_op_dict = {}

    def match_pattern(self, node):
        """
        Function: Match the node to be quantized in graph
        Parameters: node: node in graph to be matched
        Return: True: matched / False: mismatch
        """
        if node.type != 'Conv':
            if not node.has_attr(PADDING_TARGET_NODE) or not node.get_attr(PADDING_TARGET_NODE):
                return False
            # save the corresponding pad ops for conv node in dict
            conv_name = node.get_attr(PADDING_TARGET_NODE)
            if self.circular_pad_op_dict.get(conv_name, None) is None:
                self.circular_pad_op_dict[conv_name] = []
            self.circular_pad_op_dict.get(conv_name).append(node.name)
            return False

        if not node.has_attr('padding_circle') or not node.get_attr('padding_circle'):
            return False

        return True

    def do_pass(self, graph, object_node):
        """
        Function: Do actual delete action.
        Parameters:
        graph: graph structure
        object_node: node to process
        Return: None
        """
        LOGGER.logd("Doing: delete conv {} circular padding pattern in graph.".format(object_node.name),
                    'DeleteCircularPaddingPass')
        # find the pattern head's producer
        conv_node = object_node

        # find producer of the whole padding block
        padding_producer = None
        for pad_node in self.circular_pad_op_dict.get(conv_node.name):
            pad_node = graph.get_node_by_name(pad_node)
            producer, out_idx = pad_node.get_producer(0)
            # the producer of the whole padding block will not be saved in list
            if producer not in self.circular_pad_op_dict.get(conv_node.name):
                padding_producer = producer
                break

        # delete link of the padding producer
        consumers, in_idxes = padding_producer.get_consumers(out_idx)
        for consumer, in_idx in zip(consumers, in_idxes):
            if consumer.has_attr(PADDING_TARGET_NODE):
                if consumer.get_attr(PADDING_TARGET_NODE) == conv_node.name:
                    graph.remove_edge(padding_producer, out_idx, consumer, in_idx)

        # relink the padding_producer node to conv_node
        conv_producer, in_idx = conv_node.get_producer(0)
        graph.remove_edge(conv_producer, in_idx, conv_node, 0)
        graph.add_edge(padding_producer, out_idx, conv_node, 0)

        LOGGER.logd("Finished: delete conv {} circular padding pattern in graph.".format(object_node.name),
                    'DeleteCircularPaddingPass')
