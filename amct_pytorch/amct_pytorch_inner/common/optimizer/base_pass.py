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

Base class for fusion passes

"""


class BasePass():
    """
    Function: Base class of graph optimizer pass
    APIs: set_up, tear_down, match_pattern, do_fusion, run
    """
    def __init__(self):
        """
        Function: init object
        Parameter: None
        Return: None
        """
        pass

    @staticmethod
    def match_pattern(node):
        """
        Function: Match pattern of specific structure in graph
        Parameter: None
        Return: None
        """
        return False

    def set_up(self):
        """
        Function: Do some set up for pass
        Parameter: None
        Return: None
        """
        pass

    def tear_down(self):
        """
        Function: Tear down some setting for pass
        Parameter: None
        Return: None
        """
        pass

    def do_pass(self, graph, object_node):
        """
        Function: Do actual fusion operation
        Parameters: graph: graph structure
        object_node: matched node
        Return: None
        """
        pass

    def run(self, graph):
        """
        Function:do optimization for graph
        Parameters:graph
        Return:None
        """
        self.set_up()
        # Step1: match pattern and record first matched node
        matched_nodes = []
        for node in graph.nodes:
            if self.match_pattern(node):
                matched_nodes.append(node)

        # Step2: do each matched node fusion operation
        for node in matched_nodes:
            self.do_pass(graph, node)
        # Step3: do topological sort
        graph.topologic_sort()
        self.tear_down()
