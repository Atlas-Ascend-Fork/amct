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

Optimizer is manager and executor of graph passes.

"""
import inspect
from amct_pytorch.utils.log import LOGGER


class GraphOptimizer():
    """
    Function: Optimizer is manager and executor of graph passes.
    APIs: add_pass, clear_pass, do_optimizer
    """
    def __init__(self):
        """
        Function: init
        Parameters: None
        Return: None
        """
        self.__passes = []

    def add_pass(self, graph_pass):
        """
        Function: add graph_pass
        Parameters: graph_pass
        Return: None
        """
        self.__passes.append(graph_pass)

    def clear_pass(self):
        """
        Function: clear all pass
        Parameters: None
        Return: None
        """
        self.__passes.clear()

    def do_optimizer(self, graph, model=None):
        """
        Function: do optimization for passes sequentially
        Parameters: graph: Graph
                    model: torch.nn.module
        Return: None
        """
        for graph_pass in self.__passes:
            LOGGER.logi('Do {}'.format(type(graph_pass)), 'Optimizer')
            params = inspect.signature(graph_pass.run).parameters
            if 'model' in params:
                graph_pass.run(model=model, graph=graph)
            else:
                graph_pass.run(graph)
                graph.topologic_sort()
