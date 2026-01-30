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

automatic channel prune sensitivity base class

"""


class SensitivityBase:
    """ base class of sensitivity"""
    def __init__(self, loss_func=None):
        self.loss_func = loss_func

    def setup_initialization(self, graph_tuple, input_data, test_iteration, output_nodes=None):
        """
        Function: setup initialization
        Param: graph_tuple (graph, graph_info)
        Return: None
        """
        pass

    def get_sensitivity(self, search_records):
        """
        Function: get sensitivity
        Param: search_records
        Return: None, revise record
        """
        pass