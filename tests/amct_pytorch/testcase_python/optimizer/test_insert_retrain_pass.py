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
import unittest
import json
import numpy as np
import torch

from .utils import models
import amct_pytorch.graph_based_compression.amct_pytorch
from amct_pytorch.graph_based_compression.amct_pytorch.parser.parser import Parser
from amct_pytorch.graph_based_compression.amct_pytorch.custom_op.recorder.recorder import Recorder
from amct_pytorch.graph_based_compression.amct_pytorch.optimizer.graph_optimizer import GraphOptimizer
from amct_pytorch.graph_based_compression.amct_pytorch.configuration.retrain_config import RetrainConfig

from amct_pytorch.graph_based_compression.amct_pytorch.optimizer.insert_retrain_pass import InsertRetrainPass
from amct_pytorch.graph_based_compression.amct_pytorch.custom_op.comp_module.comp_module_conv1d \
    import CompModuleConv1d
from amct_pytorch.graph_based_compression.amct_pytorch.custom_op.comp_module.comp_module_conv2d \
    import CompModuleConv2d
from amct_pytorch.graph_based_compression.amct_pytorch.custom_op.comp_module.comp_module_conv3d \
    import CompModuleConv3d
from amct_pytorch.graph_based_compression.amct_pytorch.custom_op.comp_module.comp_module_linear \
    import CompModuleLinear
from amct_pytorch.graph_based_compression.amct_pytorch.custom_op.comp_module.comp_module_base \
    import CompModuleBase
from amct_pytorch.graph_based_compression.amct_pytorch.optimizer.share_act_comp_pass import \
    ShareActCompPass

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestInsertRetrainPass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_insert_retrain_pass')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.model_002 = models.Net002().to(torch.device("cpu"))
        cls.device = next(cls.model_002.parameters()).device
        cls.args_shape = [(1, 2, 28, 28)]
        cls.args = list()
        for input_shape in cls.args_shape:
            cls.args.append(torch.randn(input_shape))
        cls.args = tuple(cls.args)

        cls.onnx_file = os.path.join(cls.temp_folder, 'net_002.onnx')
        Parser.export_onnx(cls.model_002, cls.args, cls.onnx_file)
        cls.graph = Parser.parse_net_to_graph(cls.onnx_file)
        cls.graph.add_model(cls.model_002)

        cls.config_file = os.path.join(CUR_DIR, 'utils/net_002.json')
        cls.record_file = os.path.join(cls.temp_folder, 'net_002.txt')
        RetrainConfig.init_retrain(cls.config_file, cls.record_file, cls.graph)

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_insert_retrain_pass(self):
        optimizer = GraphOptimizer()
        optimizer.add_pass(InsertRetrainPass(self.device))
        optimizer.add_pass(ShareActCompPass())
        optimizer.do_optimizer(self.graph, self.model_002)

        named_module_dict = {name: mod for name, mod in self.model_002.named_modules()}
        # print('named_module_dict', named_module_dict)

        self.assertEqual(isinstance(named_module_dict['branch1'], CompModuleConv2d), True)
        self.assertEqual(isinstance(named_module_dict['branch2'], torch.nn.Conv2d), True)
        self.assertEqual(isinstance(named_module_dict['branch3'], CompModuleConv2d), True)
        self.assertEqual(isinstance(named_module_dict['branch3'].acts_comp_reuse, CompModuleConv2d), True)
        self.assertEqual(isinstance(named_module_dict['branch4'], CompModuleConv2d), True)
        self.assertEqual(isinstance(named_module_dict['conv'], CompModuleConv2d), True)
        self.assertEqual(isinstance(named_module_dict['linear'], CompModuleLinear), True)


class TestInsertRetrainConv3dPass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_insert_retrain_conv3d_pass')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.model_3d = models.Net3d().to(torch.device("cpu"))
        cls.device = next(cls.model_3d.parameters()).device
        cls.args_shape = [(1, 2, 16, 28, 28)]
        cls.args = list()
        for input_shape in cls.args_shape:
            cls.args.append(torch.randn(input_shape))
        cls.args = tuple(cls.args)

        cls.onnx_file = os.path.join(cls.temp_folder, 'net_3d.onnx')
        Parser.export_onnx(cls.model_3d, cls.args, cls.onnx_file)
        cls.graph = Parser.parse_net_to_graph(cls.onnx_file)
        cls.graph.add_model(cls.model_3d)

        cls.config_file = os.path.join(CUR_DIR, 'utils/net_3d.json')
        cls.record_file = os.path.join(cls.temp_folder, 'net_3d.txt')
        RetrainConfig.init_retrain(cls.config_file, cls.record_file, cls.graph)

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_insert_retrain_pass(self):
        optimizer = GraphOptimizer()
        optimizer.add_pass(InsertRetrainPass(self.device))
        optimizer.add_pass(ShareActCompPass())
        optimizer.do_optimizer(self.graph, self.model_3d)

        named_module_dict = {name: mod for name, mod in self.model_3d.named_modules()}
        print('named_module_dict', named_module_dict)
        for _, module in self.model_3d.named_modules():
            if isinstance(module, CompModuleBase):
                module.acts_clip_min_pre.data = torch.tensor(1.0)
                module.acts_clip_max_pre.data = torch.tensor(1.0)
        new_output = self.model_3d.forward(self.args[0])
        self.assertEqual(isinstance(named_module_dict['layer1.0'], CompModuleConv3d), True)

class TestInsertRetrainConv1dPass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_insert_retrain_conv1d_pass')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.model_1d = models.Net1d().to(torch.device("cpu"))
        cls.device = next(cls.model_1d.parameters()).device
        cls.args_shape = [(1, 2, 14)]
        cls.args = list()
        for input_shape in cls.args_shape:
            cls.args.append(torch.randn(input_shape))
        cls.args = tuple(cls.args)

        cls.onnx_file = os.path.join(cls.temp_folder, 'net_1d.onnx')
        Parser.export_onnx(cls.model_1d, cls.args, cls.onnx_file)
        cls.graph = Parser.parse_net_to_graph(cls.onnx_file)
        cls.graph.add_model(cls.model_1d)

        cls.config_file = os.path.join(CUR_DIR, 'utils/net_1d.json')
        cls.record_file = os.path.join(cls.temp_folder, 'net_1d.txt')
        RetrainConfig.init_retrain(cls.config_file, cls.record_file, cls.graph)

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_insert_retrain_pass(self):
        optimizer = GraphOptimizer()
        optimizer.add_pass(InsertRetrainPass(self.device))
        optimizer.add_pass(ShareActCompPass())
        optimizer.do_optimizer(self.graph, self.model_1d)

        named_module_dict = {name: mod for name, mod in self.model_1d.named_modules()}
        print('named_module_dict', named_module_dict)
        self.assertEqual(isinstance(named_module_dict['layer1.0'], CompModuleConv1d), True)

    def test_check_conv1d_retrain_padding_mode(self):
        mod = CompModuleConv1d(module=torch.nn.Conv1d(1,1,1, padding_mode='reflect'),
                               act_config={'ifmr_init':True, 'algo': 'ulq_quantize', 'num_bits': 8, 'fixed_min': False},
                               wts_config={'algo': 'arq_retrain', 'num_bits': 8, 'channel_wise': True},
                               common_config={'device': 'cpu', })
        mod.comp_algs.append('quant')
        self.assertRaises(RuntimeError, mod, torch.randn(1,1,1))

if __name__ == '__main__':
    unittest.main()
