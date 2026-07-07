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

from ...amct_pytorch.utils.log import LOGGER


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
