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
import logging
import os
import sys
import unittest
from copy import deepcopy
from io import BytesIO
from unittest.mock import patch

import numpy as np
import torch
from google.protobuf import text_format
from onnx import onnx_pb

import amct_pytorch.classic.graph_based.amct_pytorch as amct
from amct_pytorch.classic.graph_based.amct_pytorch.common.utils import (
    files as files_util,
)
from amct_pytorch.classic.graph_based.amct_pytorch.configuration.configuration import (
    Configuration,
)
from amct_pytorch.classic.graph_based.amct_pytorch.graph.graph import Graph
from amct_pytorch.classic.graph_based.amct_pytorch.optimizer.graph_optimizer import (
    GraphOptimizer,
)
from amct_pytorch.classic.graph_based.amct_pytorch.optimizer.insert_bias_quant_pass import (
    InsertBiasQuantPass,
)
from amct_pytorch.classic.graph_based.amct_pytorch.optimizer.insert_quant_pass import (
    construct_quant_node,
)
from amct_pytorch.classic.graph_based.amct_pytorch.optimizer.replace_bias_quant_pass import (
    ReplaceBiasQuantPass,
)
from amct_pytorch.classic.graph_based.amct_pytorch.parser.parse_record_file import (
    RecordFileParser,
)
from amct_pytorch.classic.graph_based.amct_pytorch.parser.parser import Parser
from amct_pytorch.classic.graph_based.amct_pytorch.proto import (
    scale_offset_record_pb2,
)
from amct_pytorch.classic.graph_based.amct_pytorch.utils.onnx_initializer_util import (
    TensorProtoHelper,
)

from .util import models, record_file

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

logger = logging.getLogger(__name__)


class TestReplaceBiasQuantPass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_bias_fakequant_pass')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.model = models.Net001().to(torch.device("cpu"))
        cls.args_shape = [(2, 2, 28, 28)]
        cls.args = list()
        for input_shape in cls.args_shape:
            cls.args.append(torch.randn(input_shape))
        cls.args = tuple(cls.args)

        cls.onnx_file = os.path.join(cls.temp_folder, 'net_bias_fakequant.onnx')
        Parser.export_onnx(cls.model, cls.args, cls.onnx_file)
        cls.graph = Parser.parse_net_to_graph(cls.onnx_file)

        cls.config_file = os.path.join(cls.temp_folder, 'net_bias_fakequant.json')
        skip_layers = []
        batch_num = 2
        amct.create_quant_config(cls.config_file,
                                 cls.model,
                                 cls.args,
                                 skip_layers,
                                 batch_num)
        cls.records = record_file.generate_records(
            layers_length={
                "layer1.0": 16,
                "layer2.0": 16,
                "layer3.0": 16,
                "layer4.0": 16,
                "layer5.0": 32,
                "layer6.0": 8,
                "fc.0": 1,
                "fc.2": 1,
                "fc.5": 1,
                "avg_pool": 1,
            })

    @classmethod
    def tearDownClass(cls):
        os.system('rm -r ' + cls.temp_folder)
        logger.info("[UNITTEST END bias_fakequant_pass.py]")

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_bias_fake_quant_pass_succ(self):
        optimizer = GraphOptimizer()
        optimizer.add_pass(InsertBiasQuantPass(self.records))
        optimizer.add_pass(ReplaceBiasQuantPass(self.records))
        optimizer.do_optimizer(self.graph, self.model)
        bias_dtype = TensorProtoHelper(self.graph.get_node_by_name('fc.5.sub_module.bias').proto).get_data().dtype
        self.assertEqual(bias_dtype, 'float32')

    def test_bias_fake_quant_pass_fail(self):
        self.records = record_file.generate_records(
            layers_length={
                "layer1.0": 16,
                "fc.5": 1
            })

        optimizer = GraphOptimizer()
        optimizer.add_pass(InsertBiasQuantPass(self.records))
        optimizer.add_pass(ReplaceBiasQuantPass(self.records))
        optimizer.do_optimizer(self.graph, self.model)
        bias_dtype = TensorProtoHelper(self.graph.get_node_by_name('fc.5.sub_module.bias').proto).get_data().dtype
        self.assertEqual(bias_dtype, 'float32')

    def test_rnn_bias_fake_quant_success(self):
        class RNNModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(10, 20, 1)

            def forward(self, input_data, hx):
                x = self.lstm(input_data, hx)
                return x
        model = RNNModule()
        tmp_onnx = BytesIO()
        Parser.export_onnx(model, (torch.randn(1, 1, 10), (torch.randn(1, 1, 20), torch.randn(1, 1, 20))), tmp_onnx)
        graph = Parser.parse_net_to_graph(tmp_onnx)
        node_name = 'lstm'
        node = graph.get_node_by_name(node_name)

        records = {
            node_name: {
                'data_scale': np.array(1.0, dtype=np.float32),
                'h_scale': np.array(1.0, dtype=np.float32),
                'weight_scale': np.array([1.0] * 4, dtype=np.float32),
                'weight_offset': np.array([0] * 4, dtype=np.int8),
                'recurrence_weight_scale': np.array([1.0] * 4, dtype=np.float32),
                'recurrence_weight_offset': np.array([0] * 4, dtype=np.int8),
            }
        }

        quant_bias = np.random.random([1, 160]).astype(np.int32)
        passer = ReplaceBiasQuantPass(records)
        float_bias = passer.fakequant_rnn_bias(quant_bias, node_name)
        self.assertEqual(float_bias.dtype, np.float32)