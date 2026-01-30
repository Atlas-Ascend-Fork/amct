#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or moduleify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Base class for fusion passes

"""

from amct_pytorch.utils.model_util import ModuleHelper


class BaseModuleFusionPass():# pylint: disable=unused-argument, no-self-use
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
        pass # pylint: disable=unnecessary-pass

    @staticmethod
    def match_pattern(self, module, name, graph=None):
        """
        Function: Match pattern of specific structure in module
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
        pass # pylint: disable=unnecessary-pass

    def tear_down(self):
        """
        Function: Tear down some setting for pass
        Parameter: None
        Return: None
        """
        pass # pylint: disable=unnecessary-pass

    def do_pass(self, model, object_module, object_name, graph=None):
        """
        Function: Do actual fusion operation
        Parameters: moduleel: graph structure
                    object_node: matched module
                    object_name: name of object_node
                    graph: graph structure
        Return: None
        """
        pass # pylint: disable=unnecessary-pass

    def run(self, model, graph):
        """
        Function:
        Parameters:
        Return:
        """
        self.set_up()

        # Step1: match pattern and record first matched module
        matched_modules = {}
        model_helper = ModuleHelper(model)
        for name, module in model_helper.named_module_dict.items():
            if self.match_pattern(module, name, graph):
                matched_modules[name] = module

        # Step2: do each matched module fusion operation
        for name, module in matched_modules.items():
            self.do_pass(model, module, name, graph)

        self.tear_down()
