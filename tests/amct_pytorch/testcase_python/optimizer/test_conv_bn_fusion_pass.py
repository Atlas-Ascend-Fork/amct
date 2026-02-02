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
from unittest.mock import patch
import json
import numpy as np
import torch
import copy

from .utils import models
from amct_pytorch.amct_pytorch_inner.amct_pytorch.common.utils.util import version_higher_than
from amct_pytorch.amct_pytorch_inner.amct_pytorch.parser.parser import Parser
from amct_pytorch.amct_pytorch_inner.amct_pytorch.optimizer.model_optimizer import ModelOptimizer
from amct_pytorch.amct_pytorch_inner.amct_pytorch.configuration.configuration import Configuration
from amct_pytorch.amct_pytorch_inner.amct_pytorch.common.utils.record_file_operator import \
    ScaleOffsetRecordHelper
from amct_pytorch.amct_pytorch_inner.amct_pytorch.proto import scale_offset_record_pb2

from amct_pytorch.amct_pytorch_inner.amct_pytorch.optimizer.conv_bn_fusion_pass import ConvBnFusionPass

from amct_pytorch.amct_pytorch_inner.amct_pytorch.utils.vars import QUANTIZABLE_TYPES


CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestConvbnFusionPass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        QUANTIZABLE_TYPES.extend(['ConvTranspose2d','AvgPool2d'])
        cls.temp_folder = os.path.join(CUR_DIR, 'test_conv_bn_fusion_pass')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.model_001 = models.Net001().to(torch.device("cpu"))
        # cls.model_001.eval()
        cls.args_shape = [(1, 2, 28, 28)]
        cls.args = list()
        for input_shape in cls.args_shape:
            cls.args.append(torch.randn(input_shape))
        cls.args = tuple(cls.args)

        cls.onnx_file = os.path.join(cls.temp_folder, 'net_001.onnx')
        Parser.export_onnx(cls.model_001, cls.args, cls.onnx_file)
        cls.graph = Parser.parse_net_to_graph(cls.onnx_file)
        cls.graph.add_model(cls.model_001)

        config_file = os.path.join(CUR_DIR, 'utils/net_001.json')
        record_file = os.path.join(cls.temp_folder, 'utils/net_001.txt')
        Configuration().init(config_file, record_file, cls.graph)

        skip_fusion_layers = Configuration().get_skip_fusion_layers()
        print('skip_fusion_layers', skip_fusion_layers)

    @classmethod
    def tearDownClass(cls):
        QUANTIZABLE_TYPES.remove('ConvTranspose2d')
        QUANTIZABLE_TYPES.remove('AvgPool2d')
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_fuse_eval(self):
        ''' test GraphQuerier.get_support_quant_layer2type '''
        model_001 = models.Net001().to(torch.device("cpu"))
        graph = Parser.parse_net_to_graph(self.onnx_file)

        model_001.eval()
        optimizer = ModelOptimizer()
        optimizer.add_pass(ConvBnFusionPass(Configuration))
        optimizer.do_optimizer(model_001, graph)

        named_module_dict = {name: mod for name, mod in model_001.named_modules()}

        # BatchNormalization is replaced by Indntity
        self.assertEqual(True, isinstance(named_module_dict['layer1.1'], torch.nn.Identity))
        if version_higher_than(torch.__version__, '2.1.0'):
            # affine cannot be false version later than 2.1.0
            self.assertEqual(True, isinstance(named_module_dict['layer2.1'], torch.nn.Identity))
        else:
            self.assertEqual(True, isinstance(named_module_dict['layer2.1'], torch.nn.BatchNorm2d))
        self.assertEqual(True, isinstance(named_module_dict['layer3.1'], torch.nn.Identity))
        self.assertEqual(True, isinstance(named_module_dict['layer4.1'], torch.nn.Identity))
        self.assertEqual(True, isinstance(named_module_dict['layer5.1'], torch.nn.Identity))
        self.assertEqual(True, isinstance(named_module_dict['layer6.1'], torch.nn.Identity))

    def test_fuse_train(self):
        ''' test GraphQuerier.get_support_quant_layer2type '''
        model_001 = models.Net001().to(torch.device("cpu"))
        graph = Parser.parse_net_to_graph(self.onnx_file)

        model_001.train()
        optimizer = ModelOptimizer()
        optimizer.add_pass(ConvBnFusionPass(Configuration))
        optimizer.do_optimizer(model_001, graph)

        named_module_dict = {name: mod for name, mod in model_001.named_modules()}
        # print('*'*20)
        # print(named_module_dict)

        # BatchNormalization is not replaced by Indntity
        self.assertEqual(True, isinstance(named_module_dict['layer1.1'], torch.nn.BatchNorm2d))
        self.assertEqual(True, isinstance(named_module_dict['layer2.1'], torch.nn.BatchNorm2d))
        self.assertEqual(True, isinstance(named_module_dict['layer3.1'], torch.nn.BatchNorm2d))
        self.assertEqual(True, isinstance(named_module_dict['layer4.1'], torch.nn.BatchNorm2d))
        self.assertEqual(True, isinstance(named_module_dict['layer5.1'], torch.nn.BatchNorm2d))
        self.assertEqual(True, isinstance(named_module_dict['layer6.1'], torch.nn.BatchNorm2d))


    @patch.object(Configuration, 'get_skip_fusion_layers')
    def test_conv_bn_fusion_pass_with_unsupport_padding_typpe(self, mock_get_skip_fusion_layers):
        mock_get_skip_fusion_layers.return_value = []
        class SingleConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=[3, 3], stride=1,
                                             padding=[1, 3], dilation=1, groups=1,
                                             bias=False,
                                             padding_mode='circular')
                self.bn1 = torch.nn.BatchNorm2d(3)
                self.conv2 = torch.nn.Conv2d(3, 3, kernel_size=[3, 3], stride=1,
                                             padding=[1, 3], dilation=1, groups=1,
                                             bias=False,
                                             padding_mode='replicate')
                self.bn2 = torch.nn.BatchNorm2d(3)
                self.conv3 = torch.nn.Conv2d(3, 3, kernel_size=[3, 3], stride=1,
                                             padding=[1, 3], dilation=1, groups=1,
                                             bias=False,
                                             padding_mode='reflect')
                self.bn3 = torch.nn.BatchNorm2d(3)
                self.conv4 = torch.nn.Conv2d(3, 3, kernel_size=[3, 3], stride=1,
                                             padding=[1, 3], dilation=1, groups=1,
                                             bias=False,
                                             padding_mode='zeros')
                self.bn4 = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.conv3(x)
                x = self.bn3(x)
                x = self.conv4(x)
                x = self.bn4(x)
                return x
        model = SingleConv()
        model.eval()
        # models.create_onnx(model, [(1, 3, 19, 19)], os.path.join(self.temp_folder, 'conv_padding.onnx'))

        # tmp_onnx = BytesIO()
        tmp_onnx = os.path.join(self.temp_folder, 'conv_padding.onnx')
        Parser.export_onnx(model, torch.randn(1, 3, 19, 19), tmp_onnx)
        graph = Parser.parse_net_to_graph(tmp_onnx)

        bn3 = graph.get_node_by_name('bn3')
        conv3 = bn3.get_input_anchor(0).get_peer_output_anchor().node
        conv3.set_name('conv3')
        before_nodes = len(graph.nodes)
        optimizer = ModelOptimizer()
        optimizer.add_pass(ConvBnFusionPass(Configuration))
        optimizer.do_optimizer(model, graph)
        after_nodes = len(graph.nodes)
        self.assertEqual(before_nodes - after_nodes, 4)


    @patch.object(Configuration, 'get_skip_fusion_layers')
    def test_conv_bn_fusion_pass_dialation_2(self, mock_get_skip_fusion_layers):
        mock_get_skip_fusion_layers.return_value = []
        class SingleConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=[3, 3], stride=1,
                                             padding=[1, 3], dilation=2, groups=1)
                self.bn1 = torch.nn.BatchNorm2d(3)


            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                return x
        model = SingleConv()
        model.eval()

        tmp_onnx = BytesIO()
        Parser.export_onnx(model, torch.randn(1, 3, 19, 19), tmp_onnx)
        graph = Parser.parse_net_to_graph(tmp_onnx)
        before_nodes = len(graph.nodes)

        optimizer = ModelOptimizer()
        optimizer.add_pass(ConvBnFusionPass(Configuration))
        optimizer.do_optimizer(model, graph)
        after_nodes = len(graph.nodes)
        self.assertEqual(before_nodes - after_nodes, 5)

    def test_update_record_for_fusion_invalid_record(self):
        class SingleConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=[3, 3], stride=1,
                                             padding=[1, 3], dilation=2, groups=1)
                self.bn1 = torch.nn.BatchNorm2d(3)


            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                return x
        model = SingleConv()
        model.eval()

        tmp_onnx = BytesIO()
        Parser.export_onnx(model, torch.randn(1, 3, 19, 19), tmp_onnx)
        graph = Parser.parse_net_to_graph(tmp_onnx)

        record_helper = ScaleOffsetRecordHelper(scale_offset_record_pb2.ScaleOffsetRecord)
        scale = [1]
        offset = [0, 0]
        record_helper.record_weights_scale_offset('conv1', scale, offset)
        record_helper.record_activation_scale_offset('conv1', 1, 0)


        optimizer = ModelOptimizer()
        optimizer.add_pass(ConvBnFusionPass(None, record_helper))
        self.assertRaises(RuntimeError, optimizer.do_optimizer, model, graph)

    def test_update_record_for_fusion(self):
        class SingleConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=[3, 3], stride=1,
                                             padding=[1, 3], dilation=2, groups=1)
                self.bn1 = torch.nn.BatchNorm2d(3)


            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                return x
        model = SingleConv()
        model.eval()

        tmp_onnx = BytesIO()
        Parser.export_onnx(model, torch.randn(1, 3, 19, 19), tmp_onnx)
        graph = Parser.parse_net_to_graph(tmp_onnx)

        record_helper = ScaleOffsetRecordHelper(scale_offset_record_pb2.ScaleOffsetRecord)
        scale = [1]
        offset = [0]
        record_helper.record_weights_scale_offset('conv1', scale, offset)
        record_helper.record_activation_scale_offset('conv1', 1, 0)


        optimizer = ModelOptimizer()
        optimizer.add_pass(ConvBnFusionPass(None, record_helper))
        optimizer.do_optimizer(model, graph)
        scale_w, offset_w = record_helper.read_weights_scale_offset('conv1')
        self.assertEqual(3, len(offset_w))
        self.assertEqual(3, len(scale_w))
        self.assertNotEqual(1, scale_w[0])

    # def test_conv_bn_fusion_pass_overflow_fp16(self):
    #     model = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3),
    #                                 torch.nn.BatchNorm2d(3, 3)
    #                                 )
    #     for name, module in model.named_modules(model):
    #         if name == '0':
    #             module.weight = torch.tensor([40000.]*3*3*3*3).reshape(3,3,3,3)
    #         if name == '1':
    #             module.weight = torch.nn.Parameter(torch.tensor([4.]*3*3*3).reshape(3,3,3))
    #     model.to(dtype=torch.float16, device=torch.device('cuda:0'))
    #     model.eval()
    #     tmp_onnx = BytesIO()
    #     Parser.export_onnx(model, torch.randn(1, 3, 19, 19), tmp_onnx)
    #     graph = Parser.parse_net_to_graph(tmp_onnx)

    #     optimizer = ModelOptimizer()
    #     optimizer.add_pass(ConvBnFusionPass(Configuration))
    #     optimizer.do_optimizer(model, graph)
