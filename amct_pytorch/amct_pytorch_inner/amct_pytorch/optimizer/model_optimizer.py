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

from amct_pytorch.utils.log import LOGGER


class ModelOptimizer():
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

    def add_pass(self, model_pass):
        """
        Function: add model_pass
        Parameters: model_pass
        Return: None
        """
        self.__passes.append(model_pass)

    def clear_pass(self):
        """
        Function: clear all pass
        Parameters: None
        Return: None
        """
        self.__passes.clear()

    def do_optimizer(self, model, graph):
        """
        Function: do optimization for passes sequentially
        Parameters: model: torch.nn.module
        Return: None
        """
        for model_pass in self.__passes:
            LOGGER.logi('Do {}'.format(type(model_pass)), 'Optimizer')
            model_pass.run(model, graph)
