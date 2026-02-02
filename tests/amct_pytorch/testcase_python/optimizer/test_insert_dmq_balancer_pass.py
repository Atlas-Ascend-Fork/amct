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

from unittest import mock
from unittest.mock import patch, mock_open

from .utils import models
from amct_pytorch.amct_pytorch_inner.amct_pytorch.parser.parser import Parser
from amct_pytorch.amct_pytorch_inner.amct_pytorch.custom_op.recorder.recorder import Recorder
from amct_pytorch.amct_pytorch_inner.amct_pytorch.optimizer.model_optimizer import ModelOptimizer
from amct_pytorch.amct_pytorch_inner.amct_pytorch.configuration.configuration import Configuration

from amct_pytorch.amct_pytorch_inner.amct_pytorch.optimizer.insert_dmq_balancer_pass import InsertDMQBalancerPass
from amct_pytorch.amct_pytorch_inner.amct_pytorch.custom_op.dmq_balancer.dmq_balancer import DMQBalancer

from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.vars import QUANTIZABLE_TYPES

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestInsertCaliQuantPass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_insert_dmq_balancer_pass')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.model_001 = models.Net001().to(torch.device("cpu"))
        cls.args_shape = [(1, 2, 28, 28)]
        cls.args = list()
        for input_shape in cls.args_shape:
            cls.args.append(torch.randn(input_shape))
        cls.args = tuple(cls.args)

        cls.onnx_file = os.path.join(cls.temp_folder, 'net_001.onnx')
        Parser.export_onnx(cls.model_001, cls.args, cls.onnx_file)
        cls.graph = Parser.parse_net_to_graph(cls.onnx_file)
        cls.graph.add_model(cls.model_001)

        cls.config_file = os.path.join(CUR_DIR, 'utils/net_dmq.json')
        cls.record_file = os.path.join(cls.temp_folder, 'net_001.txt')
        Configuration().init(cls.config_file, cls.record_file, cls.graph)

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_fuse(self):
        ''' test: conv(+ bias), Gemm, Matmul '''
        torch_recorder = Recorder(self.record_file)
        optimizer = ModelOptimizer()
        optimizer.add_pass(InsertDMQBalancerPass(torch_recorder))
        optimizer.do_optimizer(self.model_001, self.graph)

        named_module_dict = {name: mod for name, mod in self.model_001.named_modules()}
        # print('named_module_dict', named_module_dict)

        self.assertEqual(True, isinstance(named_module_dict['layer1.0'], DMQBalancer))
        self.assertEqual(True, isinstance(named_module_dict['layer2.0'], DMQBalancer))
        self.assertEqual(True, isinstance(named_module_dict['layer3.0'], DMQBalancer))
        self.assertEqual(True, isinstance(named_module_dict['layer4.0'], DMQBalancer))
        self.assertEqual(True, isinstance(named_module_dict['layer5.0'], DMQBalancer))
        self.assertEqual(True, isinstance(named_module_dict['layer6.0'], DMQBalancer))

        self.assertEqual(True, isinstance(named_module_dict['fc.0'], torch.nn.Linear))
        self.assertEqual(True, isinstance(named_module_dict['fc.2'], DMQBalancer))
        self.assertEqual(True, isinstance(named_module_dict['fc.5'], torch.nn.Linear))

