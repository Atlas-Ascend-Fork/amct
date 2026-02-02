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
import sys
import os
import shutil
import unittest
import json
import numpy as np
import torch

from .utils import models

import amct_pytorch.amct_pytorch_inner.amct_pytorch
from amct_pytorch.amct_pytorch_inner.amct_pytorch.parser.parser import Parser
from amct_pytorch.amct_pytorch_inner.amct_pytorch.configuration.configuration import Configuration
from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.model_util import ModuleHelper
from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.model_util import get_node_output_info
from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.model_util import get_node_output_info

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestModelHelper(unittest.TestCase):
    """
    The UT for QuantizeTool
    """
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = os.path.join(CUR_DIR, 'test_model_helper')
        cls.model_001 = models.Net001().to(torch.device("cpu"))
        cls.model_helper = ModuleHelper(cls.model_001)
        if not os.path.exists(cls.temp_dir):
            os.makedirs(cls.temp_dir)
        cls.config_file = os.path.join(cls.temp_dir, 'config.json')
        amct_pytorch.amct_pytorch_inner.amct_pytorch.create_quant_config(cls.config_file, cls.model_001, torch.randn((1, 2, 28, 28)))

        cls.args = torch.randn(16, 2, 28, 28)
        cls.onnx_file = os.path.join(cls.temp_dir, 'net_001.onnx')
        Parser.export_onnx(cls.model_001, cls.args, cls.onnx_file)
        cls.graph = Parser.parse_net_to_graph(cls.onnx_file)
        cls.graph.add_model(cls.model_001)
        record_file = os.path.join(cls.temp_dir, 'test_get_node_output_info_tensor_input.txt')
        Configuration().init(cls.config_file, record_file, cls.graph)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_module(self):
        self.model_helper.get_module('layer1.0')
        self.assertRaises(RuntimeError,  self.model_helper.get_module, 'error_name')

    def test_get_parent_module(self):
        self.model_helper.get_parent_module('layer')
        self.assertRaises(RuntimeError, self.model_helper.get_parent_module, 'lay.1')
        self.assertRaises(RuntimeError, self.model_helper.get_parent_module, '')

    def test_get_node_output_info_tensor_input(self):
        input_data = torch.randn(16, 2, 28, 28)
        ret = get_node_output_info(self.model_001, input_data)
        for _, val in ret.items():
            for item in val:
                self.assertTrue('attr_name' in item)
                self.assertTrue('attr_type' in item)
                self.assertTrue('attr_val' in item)

    def test_get_node_output_info_tuple_input(self):
        input_data = (torch.randn(16, 2, 28, 28),)
        ret = get_node_output_info(self.model_001, input_data)
        for _, val in ret.items():
            for item in val:
                self.assertTrue('attr_name' in item)
                self.assertTrue('attr_type' in item)
                self.assertTrue('attr_val' in item)

    def test_get_node_output_info_tuple_input_with_dict(self):
        input_data = ({'x': torch.randn(16, 2, 28, 28)},)
        ret = get_node_output_info(self.model_001, input_data)
        for _, val in ret.items():
            for item in val:
                self.assertTrue('attr_name' in item)
                self.assertTrue('attr_type' in item)
                self.assertTrue('attr_val' in item)

    def test_get_node_output_info_tensor_input(self):
        input_data = torch.randn(16, 2, 28, 28)
        ret = get_node_output_info(self.model_001, input_data)
        for _, val in ret.items():
            for item in val:
                self.assertTrue('attr_name' in item)
                self.assertTrue('attr_type' in item)
                self.assertTrue('attr_val' in item)
 
    def test_get_node_output_info_tuple_input(self):
        input_data = (torch.randn(16, 2, 28, 28),)
        ret = get_node_output_info(self.model_001, input_data)
        for _, val in ret.items():
            for item in val:
                self.assertTrue('attr_name' in item)
                self.assertTrue('attr_type' in item)
                self.assertTrue('attr_val' in item)
 
    def test_get_node_output_info_tuple_input_with_dict(self):
        input_data = ({'x': torch.randn(16, 2, 28, 28)},)
        ret = get_node_output_info(self.model_001, input_data)
        for _, val in ret.items():
            for item in val:
                self.assertTrue('attr_name' in item)
                self.assertTrue('attr_type' in item)
                self.assertTrue('attr_val' in item)