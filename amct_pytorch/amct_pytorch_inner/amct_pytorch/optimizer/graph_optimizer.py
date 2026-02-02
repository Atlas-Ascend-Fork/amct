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
from ...amct_pytorch.utils.log import LOGGER


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
