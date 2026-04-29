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
import copy

from collections import OrderedDict

from .utils import models
from amct_pytorch.graph_based_compression.amct_pytorch.parser.parser import Parser
from amct_pytorch.graph_based_compression.amct_pytorch.configuration.distill_config import parse_distill_config
from amct_pytorch.graph_based_compression.amct_pytorch.optimizer.model_optimizer import ModelOptimizer
from amct_pytorch.graph_based_compression.amct_pytorch.configuration.distill_config import get_enable_quant_layers

from amct_pytorch.graph_based_compression.amct_pytorch.optimizer.insert_qat_pass import InsertQatPass
from amct_pytorch.graph_based_compression.amct_pytorch.nn.module.quantization.conv2d import Conv2dQAT
from amct_pytorch.graph_based_compression.amct_pytorch.nn.module.quantization.linear import LinearQAT

from amct_pytorch.graph_based_compression.amct_pytorch.utils.vars import DISTILL_TYPES

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestInsertQatPass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_insert_qat_pass')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.model_001 = models.Conv2dLinear().to(torch.device("cpu"))
        args_shape = [(4, 3, 16, 16)]
        cls.args = list()
        for input_shape in args_shape:
            cls.args.append(torch.randn(input_shape))
        cls.args = tuple(cls.args)

        cls.onnx_file = os.path.join(cls.temp_folder, 'net_001.onnx')
        Parser.export_onnx(cls.model_001, cls.args, cls.onnx_file)
        cls.graph = Parser.parse_net_to_graph(cls.onnx_file)
        cls.graph.add_model(cls.model_001)

        cls.config_file = os.path.join(CUR_DIR, 'utils/Conv2dLinear_cfg.json')
        cls.distill_config = parse_distill_config(cls.config_file, cls.model_001)

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_insert_qat_match_pattern_success(self):
        for name, mod in self.model_001.named_modules():
            if name == 'layer1':
                conv2d = mod
                break
        self.assertTrue(InsertQatPass(self.distill_config).match_pattern(conv2d, 'layer1'))

    def test_insert_qat_match_pattern_not_distill_type(self):
        mod = torch.nn.Conv3d(1, 1, 1)
        self.assertFalse(InsertQatPass(self.distill_config).match_pattern(mod, 'layer1'))

    def test_insert_qat_match_pattern_not_enable_quant(self):
        mod = torch.nn.Conv2d(1, 1, 1, padding_mode='zeros')
        distill_config = OrderedDict([('layer2', OrderedDict([
            ('quant_enable', True),
            ('distill_data_config', OrderedDict([('algo', 'ulq_quantize'), ('dst_type', 'INT8')])),
            ('distill_weight_config', OrderedDict([('algo', 'arq_distill'), ('channel_wise', True), ('dst_type', 'INT8')]))]))])
        self.assertFalse(InsertQatPass(distill_config).match_pattern(mod, 'layer1'))

    def test_insert_qat_do_pass_success(self):
        optimizer = ModelOptimizer()
        optimizer.add_pass(InsertQatPass(self.distill_config))
        model = copy.deepcopy(self.model_001)
        optimizer.do_optimizer(model, self.graph)

        self.assertTrue(isinstance(model.layer1, Conv2dQAT))
        self.assertTrue(isinstance(model.layer3, LinearQAT))

