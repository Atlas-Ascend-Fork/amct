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
from io import BytesIO
import unittest
from unittest import mock
from unittest.mock import patch, mock_open
import json
import numpy as np
import torch

from .utils import models
from amct_pytorch.graph_based_compression.amct_pytorch.parser.parser import Parser
from amct_pytorch.graph_based_compression.amct_pytorch.optimizer.model_optimizer import ModelOptimizer
from amct_pytorch.graph_based_compression.amct_pytorch.configuration.configuration import Configuration
from amct_pytorch.graph_based_compression.amct_pytorch.proto import scale_offset_record_pb2
from amct_pytorch.graph_based_compression.amct_pytorch.utils.onnx_initializer_util import TensorProtoHelper
from google.protobuf import text_format

from amct_pytorch.graph_based_compression.amct_pytorch.optimizer.weight_calibration import WeightsCalibrationPass

from amct_pytorch.graph_based_compression.amct_pytorch.utils.vars import QUANTIZABLE_TYPES

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]


class TestWeightsCalibrationPass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        QUANTIZABLE_TYPES.extend(['ConvTranspose2d','AvgPool2d', 'ConvTranspose1d'])
        cls.temp_folder = os.path.join(CUR_DIR, 'test_weight_calibration_pass')
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
        with open(cls.record_file, 'w') as record_file:
            record_file.write('')
        Configuration().init(cls.config_file, cls.record_file, cls.graph)

    @classmethod
    def tearDownClass(cls):
        QUANTIZABLE_TYPES.remove('ConvTranspose2d')
        QUANTIZABLE_TYPES.remove('AvgPool2d')
        QUANTIZABLE_TYPES.remove('ConvTranspose1d')
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_weight_cali(self):
        ''' test: conv(+ bias), Gemm, Matmul'''
        optimizer = ModelOptimizer()
        optimizer.add_pass(WeightsCalibrationPass())
        optimizer.do_optimizer(self.model_001, self.graph)

        records = scale_offset_record_pb2.ScaleOffsetRecord()
        with open(self.record_file, 'r') as record_read_file:
            pbtxt_string = record_read_file.read()
            text_format.Merge(pbtxt_string, records)

        self.assertEqual(8, len(records.record))

    def test_weight_cali_trans(self):
        ''' test: conv(+ bias), Gemm, Matmul'''
        matmul_node = self.graph.get_node_by_name('fc.2')
        matmul_node.set_attr('with_weights_trans', True)

        optimizer = ModelOptimizer()
        optimizer.add_pass(WeightsCalibrationPass())
        optimizer.do_optimizer(self.model_001, self.graph)

        records = scale_offset_record_pb2.ScaleOffsetRecord()
        with open(self.record_file, 'r') as record_read_file:
            pbtxt_string = record_read_file.read()
            text_format.Merge(pbtxt_string, records)

        self.assertEqual(8, len(records.record))

    @patch.object(Configuration, 'get_layer_config')
    def test_conv_transpose2d_weights_calibration(self, mock_get_layer_config):
        class ConvTransposeNet(torch.nn.Module):
            def __init__(self):
                super(ConvTransposeNet, self).__init__()
                self.conv_transpose = torch.nn.ConvTranspose2d(3, 3, 8, stride=32, bias=False)

            def forward(self, x):
                return self.conv_transpose(x)

        model = ConvTransposeNet()
        mock_get_layer_config.return_value = {'weight_quant_params': {'wts_algo': 'arq_quantize',
                                                                      'num_bits': 8,
                                                                      'channel_wise': False,
                                                                      'with_offset': False}}

        tmp_onnx = BytesIO()
        Parser.export_onnx(model, torch.randn((1, 3, 224, 224)), tmp_onnx)
        graph = Parser.parse_net_to_graph(tmp_onnx)

        named_module_dict = {}
        deconv_module = None
        deconv_name = 'conv_transpose'
        for name, mod in model.named_modules():
            named_module_dict[name] = mod
            if name == deconv_name:
                deconv_module = mod
        self.assertIsNone(WeightsCalibrationPass().do_pass(model, deconv_module, deconv_name, graph))

    @patch.object(Configuration, 'get_layer_config')
    def test_conv_transpose_1d_weights_calibration(self, mock_get_layer_config):
        class ConvTransposeNet(torch.nn.Module):
            def __init__(self):
                super(ConvTransposeNet, self).__init__()
                self.conv_transpose = torch.nn.ConvTranspose1d(3, 3, 8, stride=32, bias=False)

            def forward(self, x):
                return self.conv_transpose(x)

        model = ConvTransposeNet()
        mock_get_layer_config.return_value = {'weight_quant_params': {'wts_algo': 'arq_quantize',
                                                                      'num_bits': 8,
                                                                      'channel_wise': False,
                                                                      'with_offset': False}}

        tmp_onnx = BytesIO()
        Parser.export_onnx(model, torch.randn((1, 3, 224)), tmp_onnx)
        graph = Parser.parse_net_to_graph(tmp_onnx)

        named_module_dict = {}
        deconv_module = None
        deconv_name = 'conv_transpose'
        for name, mod in model.named_modules():
            named_module_dict[name] = mod
            if name == deconv_name:
                deconv_module = mod
        self.assertIsNone(WeightsCalibrationPass().do_pass(model, deconv_module, deconv_name, graph))


    def test_broad_cast_tensor_balance_factor(self):
        class Conv1dModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1d = torch.nn.Conv1d(3,3,3)
            def forward(self, x):
                return self.conv1d(x)
        conv1d_module = Conv1dModule()
        tmp_onnx = BytesIO()
        Parser.export_onnx(conv1d_module, torch.randn(3,3,3), tmp_onnx)
        graph = Parser.parse_net_to_graph(tmp_onnx)
        node = graph.get_node_by_name('conv1d')
        for name, mod in conv1d_module.named_modules():
            if name == 'conv1d':
                conv1d_mod = mod
                break

        modified_weight = WeightsCalibrationPass.apply_balance_scale_to_weight(conv1d_mod, 'conv1d', [3, 3, 3], torch.ones(3,3,3))
        self.assertTrue((modified_weight == torch.ones(3, 3, 3) * 3).all())