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
from io import BytesIO
import json
import numpy as np
import torch
import copy

from unittest import mock
from unittest.mock import patch, mock_open

from .utils import models
from amct_pytorch.graph_based_compression.amct_pytorch.parser.parser import Parser
from amct_pytorch.graph_based_compression.amct_pytorch.custom_op.recorder.recorder import Recorder
from amct_pytorch.graph_based_compression.amct_pytorch.optimizer.model_optimizer import ModelOptimizer
from amct_pytorch.graph_based_compression.amct_pytorch.configuration.configuration import Configuration

from amct_pytorch.graph_based_compression.amct_pytorch.optimizer.insert_cali_quant import InsertCaliQuantPass
from amct_pytorch.graph_based_compression.amct_pytorch.custom_op.ifmr.ifmr import IFMR
from amct_pytorch.graph_based_compression.amct_pytorch.custom_op.hfmg.hfmg import HFMG

from amct_pytorch.graph_based_compression.amct_pytorch.utils.vars import QUANTIZABLE_TYPES

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestInsertCaliQuantPass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        QUANTIZABLE_TYPES.extend(['ConvTranspose2d','AvgPool2d'])
        cls.temp_folder = os.path.join(CUR_DIR, 'test_insert_ifmr_pass')
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

        cls.config_file = os.path.join(CUR_DIR, 'utils/net_001.json')
        cls.record_file = os.path.join(cls.temp_folder, 'net_001.txt')
        Configuration().init(cls.config_file, cls.record_file, cls.graph)

    @classmethod
    def tearDownClass(cls):
        QUANTIZABLE_TYPES.remove('ConvTranspose2d')
        QUANTIZABLE_TYPES.remove('AvgPool2d')
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_fuse(self):
        ''' test: conv(+ bias), Gemm, Matmul '''
        torch_recorder = Recorder(self.record_file)
        optimizer = ModelOptimizer()
        optimizer.add_pass(InsertCaliQuantPass(torch_recorder))
        model = copy.deepcopy(self.model_001)
        optimizer.do_optimizer(model, self.graph)

        named_module_dict = {name: mod for name, mod in model.named_modules()}
        # print('named_module_dict', named_module_dict)

        self.assertEqual(True, isinstance(named_module_dict['layer1.0'].cali_quant_module, IFMR))
        self.assertEqual(True, isinstance(named_module_dict['layer2.0'].cali_quant_module, IFMR))
        self.assertEqual(True, isinstance(named_module_dict['layer3.0'].cali_quant_module, IFMR))
        self.assertEqual(True, isinstance(named_module_dict['layer4.0'].cali_quant_module, IFMR))
        self.assertEqual(True, isinstance(named_module_dict['layer5.0'].cali_quant_module, IFMR))
        self.assertEqual(True, isinstance(named_module_dict['layer6.0'].cali_quant_module, IFMR))

        self.assertEqual(True, isinstance(named_module_dict['fc.0'], torch.nn.Linear))
        self.assertEqual(True, isinstance(named_module_dict['fc.2'].cali_quant_module, IFMR))
        self.assertEqual(True, isinstance(named_module_dict['fc.5'], torch.nn.Linear))

    @patch.object(Configuration, 'get_layer_config')
    def test_insert_hfmg(self, mock_get_layer_config):
        ''' test: conv(+ bias), Gemm, Matmul '''
        mock_get_layer_config.return_value = {
            "quant_enable":True,
            "activation_quant_params":{
                "act_algo": "hfmg",
                "num_of_bins": 4096,
                "with_offset":True,
                "batch_num":2,
                "num_bits":8
            },
            "weight_quant_params":{
                "channel_wise":True,
                "num_bits":8,
                "with_offset":False
            }
        }
        model = copy.deepcopy(self.model_001)
        torch_recorder = Recorder(self.record_file)
        optimizer = ModelOptimizer()
        optimizer.add_pass(InsertCaliQuantPass(torch_recorder))
        optimizer.do_optimizer(model, self.graph)

        named_module_dict = {name: mod for name, mod in model.named_modules()}
        # print('named_module_dict', named_module_dict)
        self.assertEqual(True, isinstance(named_module_dict['layer1.0'].cali_quant_module, HFMG))
        self.assertEqual(True, isinstance(named_module_dict['layer2.0'].cali_quant_module, HFMG))
        self.assertEqual(True, isinstance(named_module_dict['layer3.0'].cali_quant_module, HFMG))
        self.assertEqual(True, isinstance(named_module_dict['layer4.0'].cali_quant_module, HFMG))
        self.assertEqual(True, isinstance(named_module_dict['layer5.0'].cali_quant_module, HFMG))
        self.assertEqual(True, isinstance(named_module_dict['layer6.0'].cali_quant_module, HFMG))

        self.assertEqual(True, isinstance(named_module_dict['fc.0'], torch.nn.Linear))
        self.assertEqual(True, isinstance(named_module_dict['fc.2'].cali_quant_module, HFMG))
        self.assertEqual(True, isinstance(named_module_dict['fc.5'], torch.nn.Linear))

    @patch.object(Configuration, 'get_layer_config')
    def test_broad_cast_tensor_balance_factor(self, mock_get_layer_config):
        mock_get_layer_config.return_value = {
            "quant_enable":True,
            "activation_quant_params":{
                "act_algo": "hfmg",
                "num_of_bins": 4096,
                "with_offset":True,
                "batch_num":2,
                "num_bits":8
            },
            "weight_quant_params":{
                "channel_wise":True,
                "num_bits":8,
                "with_offset":False
            },
            'dmq_balancer_param': {
                0.5
            }
        }
        class Conv1dModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1d = torch.nn.Conv1d(1,1,1)
            def forward(self, x):
                return self.conv1d(x)
        conv1d_module = Conv1dModule()
        tmp_onnx = BytesIO()
        Parser.export_onnx(conv1d_module, torch.randn(1,1,1), tmp_onnx)
        graph = Parser.parse_net_to_graph(tmp_onnx)
        for name, mod in conv1d_module.named_modules():
            if name == 'conv1d':
                conv1d_mod = mod
                break
        records = {'conv1d': {'tensor_balance_factor': [3, 3, 3]}}

        cali_quant_pass = InsertCaliQuantPass(None,records=records)
        cali_quant_pass.broad_cast_tensor_balance_factor('conv1d', conv1d_mod, graph)