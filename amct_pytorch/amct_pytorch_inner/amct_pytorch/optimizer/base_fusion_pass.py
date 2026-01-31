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
import inspect


class BaseFusionPass():# pylint: disable=unused-argument, no-self-use
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
        self.records = {}

    def set_up(self):
        """
        Function: Do some set up for pass
        Parameter: None
        Return: None
        """
        pass # pylint: disable=unnecessary-pass

    def tear_down(self):
        """
        Function: Tear down some setting for pass
        Parameter: None
        Return: None
        """
        pass # pylint: disable=unnecessary-pass

    def match_pattern(self, node):
        """
        Function: Match pattern of specific structure in graph
        Parameter: None
        Return: None
        """
        return False

    def do_pass(self, graph, object_node, model):
        """
        Function: Do actual fusion operation
        Parameters: graph: graph structure
                    object_node: matched node
        Return: None
        """
        pass # pylint: disable=unnecessary-pass

    def run(self, graph, model):
        """
        Function:
        Parameters:
        Return:
        """
        self.set_up()
        # Step1: match pattern and record first matched node
        matched_nodes = []
        for node in graph.nodes:
            if self.match_pattern(node):
                matched_nodes.append(node)

        # Step2: do each matched node fusion operation
        for node in matched_nodes:
            params = inspect.signature(self.do_pass).parameters
            if 'model' in params:
                self.do_pass(graph, node, model)
            else:
                self.do_pass(graph, node)
        # Step3: do topological sort
        graph.topologic_sort()
        self.tear_down()
