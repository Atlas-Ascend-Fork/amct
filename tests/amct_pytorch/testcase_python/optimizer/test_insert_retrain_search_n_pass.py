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
import json
import os
import sys
import unittest

import numpy as np
import torch
import torch.nn as nn

from amct_pytorch.classic.graph_based.amct_pytorch.configuration.retrain_config import (
    RetrainConfig,
)
from amct_pytorch.classic.graph_based.amct_pytorch.custom_op.comp_module.comp_module_conv2d import (
    CompModuleConv2d,
)
from amct_pytorch.classic.graph_based.amct_pytorch.custom_op.comp_module.comp_module_linear import (
    CompModuleLinear,
)
from amct_pytorch.classic.graph_based.amct_pytorch.custom_op.recorder.recorder import (
    Recorder,
)
from amct_pytorch.classic.graph_based.amct_pytorch.custom_op.retrain_quant import (
    RetrainQuant,
)
from amct_pytorch.classic.graph_based.amct_pytorch.optimizer.graph_optimizer import (
    GraphOptimizer,
)
from amct_pytorch.classic.graph_based.amct_pytorch.optimizer.insert_retrain_pass import (
    InsertRetrainPass,
)
from amct_pytorch.classic.graph_based.amct_pytorch.optimizer.insert_retrain_quant_pass import (
    InsertRetrainQuantPass,
)
from amct_pytorch.classic.graph_based.amct_pytorch.optimizer.share_act_comp_pass import (
    ShareActCompPass,
)
from amct_pytorch.classic.graph_based.amct_pytorch.parser.parser import Parser
from tests.amct_pytorch.testcase_python.optimizer.utils import models

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]


class TestInsertRetrainSearchNPass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_insert_retrain_searchn_pass')
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

    def test_insert_retrain_searchn_pass(self):
        torch_recorder = Recorder(self.record_file)
        optimizer = GraphOptimizer()
        optimizer.add_pass(InsertRetrainPass(self.device))
        optimizer.add_pass(ShareActCompPass())
        optimizer.add_pass(InsertRetrainQuantPass(torch_recorder, self.device))
        optimizer.do_optimizer(self.graph, self.model_002)

        named_module_dict = {name: mod for name, mod in self.model_002.named_modules()}

        self.assertIsInstance(named_module_dict['branch1'], RetrainQuant)
        self.assertIsInstance(named_module_dict['branch2'], nn.Conv2d)
        self.assertIsInstance(named_module_dict['branch3'], RetrainQuant)
        self.assertIsInstance(named_module_dict['branch3'].quant_module.acts_comp_reuse, CompModuleConv2d)
        self.assertIsInstance(named_module_dict['branch4'], RetrainQuant)
        self.assertIsInstance(named_module_dict['conv'], RetrainQuant)
        self.assertIsInstance(named_module_dict['bn'], nn.Identity)
        self.assertIsInstance(named_module_dict['linear'], RetrainQuant)


if __name__ == '__main__':
    unittest.main()
