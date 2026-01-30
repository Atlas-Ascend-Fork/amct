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

Delete pass-throught node in graph. Pass-throught node has only one input and
one output.

"""


from ...utils.log import LOGGER
from .base_pass import BasePass


class DeletePassThroughNodePass(BasePass):
    def __init__(self, delete_types):
        """
        Function: init object
        Parameter: None
        Return: None
        """
        BasePass.__init__(self)
        self._delete_types = delete_types

    @staticmethod
    def do_pass(graph, object_node):
        """
        Function: Do actual quantization and node's weight is changed to int8.
        Parameters:
        graph: graph structure
        object_node: node to process
        Return: None
        """
        LOGGER.logd("Doing: delete node {} in graph.".format(object_node.name),
                     'DeletePassThroughNodePass')

        graph.delete_node(object_node, 0, 0)
        if object_node.is_isolated:
            graph.remove_node(object_node)
        LOGGER.logd("Finished: delete node {} in graph.".format(object_node.name),
                     'DeletePassThroughNodePass')

    def match_pattern(self, node):
        """
        Function: Match the node to be quantized in graph
        Parameters: node: node in graph to be matched
        Return:
        True: matched
        False: mismatch
        """
        if node.type not in self._delete_types:
            return False

        return True