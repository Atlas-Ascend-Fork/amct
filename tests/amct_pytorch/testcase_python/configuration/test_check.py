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
import os
from io import BytesIO
import sys
sys.path.append("~/amct/llt/asl/aoetools/amct/amct_pytorch.graph_based_compression.amct_pytorch/ut/testcase_python/configuration")

import unittest
import torch
import torch.nn.functional as F
from onnx import onnx_pb

from .utils import models
from amct_pytorch.graph_based_compression.amct_pytorch.parser.parser import Parser
from amct_pytorch.graph_based_compression.amct_pytorch.graph.graph import Graph
from amct_pytorch.graph_based_compression.amct_pytorch.utils.model_util import ModuleHelper
from amct_pytorch.graph_based_compression.amct_pytorch.custom_op.ifmr.ifmr import IFMR
from amct_pytorch.graph_based_compression.amct_pytorch.custom_op.recorder.recorder import Recorder

from amct_pytorch.graph_based_compression.amct_pytorch.configuration.check import GraphQuerier
from amct_pytorch.graph_based_compression.amct_pytorch.configuration.check import GraphChecker

from amct_pytorch.graph_based_compression.amct_pytorch.utils.vars import QUANTIZABLE_TYPES
from amct_pytorch.graph_based_compression.amct_pytorch.utils.vars import QUANTIZABLE_ONNX_TYPES

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]

class TestCheckModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_check')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)
        cls.model_001 = models.Net001().to(torch.device("cpu"))
        cls.args_shape = [(1, 2, 28, 28)]
        cls.args = list()
        for input_shape in cls.args_shape:
            cls.args.append(torch.randn(input_shape))
        cls.args = tuple(cls.args)

        cls.onnx_model = BytesIO()
        Parser.export_onnx(cls.model_001, cls.args, cls.onnx_model)
        cls.graph = Parser.parse_net_to_graph(cls.onnx_model)


    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_support_quant_layer2type(self):
        ''' test GraphQuerier.get_support_quant_layer2type '''
        self.graph.add_model(self.model_001)
        layer_types = GraphQuerier.get_support_quant_layer2type(self.graph)
        self.assertEqual(layer_types['layer1.0'], 'Conv2d')
        self.assertEqual(layer_types['fc.0'], 'Linear')
        self.assertEqual(layer_types['fc.2'], 'Linear')
        self.assertEqual(layer_types['fc.5'], 'Linear')

    def test_dilation_not1(self):
        model_002 = models.Net002().to(torch.device("cpu"))

        tmp_onnx = BytesIO()
        Parser.export_onnx(model_002, self.args, tmp_onnx)
        graph = Parser.parse_net_to_graph(tmp_onnx)
        graph.add_model(model_002)
        layers = GraphQuerier.get_support_quant_layers(graph)
        self.assertEqual(layers, ['layer1.0', 'layer2.0'])

    def test_conv3d(self):
        model_conv3d = models.Net3d().to(torch.device("cpu"))
        args = list()
        for input_shape in [(1, 2, 4, 14, 14)]:
            args.append(torch.randn(input_shape))
        args = tuple(args)

        tmp_onnx = BytesIO()
        Parser.export_onnx(model_conv3d, args, tmp_onnx)
        graph = Parser.parse_net_to_graph(tmp_onnx)
        graph.add_model(model_conv3d)
        layers = GraphQuerier.get_support_quant_layers(graph)
        self.assertEqual(layers, ['layer1.0'])
        layer_types = GraphQuerier.get_support_quant_layer2type(graph)
        self.assertEqual(layer_types['layer1.0'], 'Conv3d')

    def test_get_support_int16_quantizable_layers(self):
        support_layers = GraphQuerier.get_support_int16_quantizable_layers(self.graph)
        self.assertEqual(support_layers, \
            ['layer1.0', 'layer2.0', 'layer3.0', 'layer4.0', 'layer5.0', 'layer6.0', 'fc.0', 'fc.5'])

    def test_convtranspose2d_int16(self):
        model = models.SingleConv().to(torch.device("cpu"))
        args = list()
        for input_shape in model.args_shape:
            args.append(torch.randn(input_shape))
        args = tuple(args)

        onnx_file = os.path.join(self.temp_folder, 'SingleConv.onnx')
        Parser.export_onnx(model, args, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        self.assertEqual(GraphQuerier.get_support_int16_quantizable_layers(graph), ['layer1','layer2'])

    def test_get_act_symmetric_limit_types(self):
        ret = GraphQuerier.get_act_symmetric_limit_types()
        self.assertEqual(ret, ['Conv3d'])

    def test_get_act_symmetric_limit_layers(self):
        model_conv3d = models.Net3d().to(torch.device("cpu"))
        args = list()
        for input_shape in [(1, 2, 4, 14, 14)]:
            args.append(torch.randn(input_shape))
        args = tuple(args)

        tmp_onnx = BytesIO()
        Parser.export_onnx(model_conv3d, args, tmp_onnx)
        graph = Parser.parse_net_to_graph(tmp_onnx)
        graph.add_model(model_conv3d)

        layers = GraphQuerier.get_act_symmetric_limit_layers(graph)
        self.assertEqual(layers, ['layer1.0'])
        layer_types = GraphQuerier.get_support_quant_layer2type(graph)
        self.assertEqual(layer_types['layer1.0'], 'Conv3d')

    def test_submodel(self):
        model_001_sub = models.Net001Sub().to(torch.device("cpu"))
        args_shape = [(1, 2, 28, 28)]
        args = list()
        for input_shape in args_shape:
            args.append(torch.randn(input_shape))
        args = tuple(args)

        tmp_onnx = BytesIO()
        Parser.export_onnx(model_001_sub, args, tmp_onnx)
        graph = Parser.parse_net_to_graph(tmp_onnx)
        graph.add_model(model_001_sub)
        layers = GraphQuerier.get_support_quant_layers(graph)
        self.assertEqual(layers, ['layer1.0', 'layer2.0', 'layer3.0'])

    def test_amct_ops_in_model(self):
        temp_folder = self.temp_folder
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                record_file = os.path.join(temp_folder, 'conv_model.txt')
                record_module = Recorder(record_file)
                self.layer1 = IFMR(torch.nn.Conv2d(2, 4, kernel_size=2),
                    record_module, ['conv'])
            def forward(self, x):
                x = self.layer1(x)

        model = Net()
        graph = Parser.parse_net_to_graph(self.onnx_model)
        graph.add_model(model)

        self.assertRaises(RuntimeError, GraphChecker.check_quant_behaviours, graph)


    def test_check_reused_node_not_support(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
                self.conv1_1 = torch.nn.Conv2d(32, 32, 1, 1)
                self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
                self.global_avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

            # x represents our data
            def forward(self, x):
                # Pass data through conv1
                x = self.conv1(x)
                # Use the rectified-linear activation function over x
                x = F.relu(x)

                # repeated conv
                x = self.conv1_1(x)
                x = F.relu(x)

                x = self.conv1_1(x)
                x = F.relu(x)

                x = self.conv2(x)
                x = F.relu(x)
                x = self.global_avg_pool(x)
                return x

        model = Net()
        model.eval()
        dummy_input = torch.randn(1, 1, 28, 28)

        tmp_onnx = BytesIO()
        Parser.export_onnx(model, dummy_input, tmp_onnx)
        graph = Parser.parse_net_to_graph(tmp_onnx)

        model_helper = ModuleHelper(model)
        mod_conv1 = model_helper.get_module('conv1')
        mod_conv1_1 = model_helper.get_module('conv1_1')
        mod_conv2 = model_helper.get_module('conv2')

        for node in graph.nodes:
            if node.type == 'GlobalAveragePool':
                global_avg_pool_node = node

        self.assertTrue(GraphChecker.check_quantize_type('conv1', mod_conv1, graph))
        self.assertFalse(GraphChecker.check_quantize_type('conv1_1', mod_conv1_1, graph))
        self.assertTrue(GraphChecker.check_quantize_type('conv2', mod_conv2, graph))
        self.assertFalse(GraphChecker.check_special_limit(global_avg_pool_node))

    def test_check_matmul_not_const_not_support(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.layer1 = torch.nn.Linear(16, 1024, bias=True)
                self.layer2 = torch.nn.Linear(16, 1024, bias=True)

            # x represents our data
            def forward(self, x):
                a = self.layer1(x)
                b = self.layer2(x)
                x = torch.matmul(a, b)
                return x

        model = Net()
        model.eval()

        dummy_input = torch.randn(1024, 16)
        tmp_onnx = BytesIO()
        Parser.export_onnx(model, dummy_input, tmp_onnx)
        graph = Parser.parse_net_to_graph(tmp_onnx)

        for node in graph.nodes:
            if node.type == 'MatMul':
                matmul_node = node

        self.assertFalse(GraphChecker.check_special_limit(matmul_node))

    def test_invalid_padding(self):
        model = models.SingleConv().to(torch.device("cpu"))
        args = list()
        for input_shape in model.args_shape:
            args.append(torch.randn(input_shape))
        args = tuple(args)

        onnx_file = os.path.join(self.temp_folder, 'SingleConv.onnx')
        Parser.export_onnx(model, args, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        graph.model = model

        layers = GraphQuerier.get_support_quant_layers(graph)
        self.assertEqual(layers, ['layer2'])


    def test_op_matching(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.args_shape = [(1, 16, 17, 17)]
                self.layer1 = torch.nn.Conv2d(16, 16, kernel_size=[3, 3], stride=1,
                    padding_mode='zeros', padding=0, dilation=(1, 1),
                    groups=1, bias=False)
                self.layer2 = torch.nn.Conv2d(16, 16, kernel_size=[3, 3], stride=1,
                    padding_mode='zeros', padding=0, dilation=(1, 1),
                    groups=1, bias=False)

            def forward(self, x):
                x = self.layer1(x)
                output = self.layer2(x)
                return output

        onnx_file = BytesIO()
        test_model = Net().to(torch.device("cpu"))
        args = list()
        for input_shape in test_model.args_shape:
            args.append(torch.randn(input_shape))
        args = tuple(args)
        test_model.eval()

        Parser.export_onnx(test_model, args, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        graph.model = test_model

        fused_op_list = ['layer1', 'layer2', 'layer1..quant',
            'layer1.dequant', 'layer1.anti_quant']
        GraphQuerier.check_op_matching(graph, fused_op_list)

        # with self.assertRaises(RuntimeError) as cm:
        fused_op_list = ['layer3', 'layer4']
        GraphQuerier.check_op_matching(graph, fused_op_list)


    def test_conv3d_valid_dilation(self):
        # torch model
        torch_model = models.Net3d001().to(torch.device("cpu"))
        onnx_file = os.path.join(self.temp_folder, 'net3d.onnx')
        models.create_onnx(torch_model, torch_model.args_shape, onnx_file)
        # graph
        graph = Parser.parse_net_to_graph(onnx_file)
        model_helper = ModuleHelper(torch_model)
        conv_node = None
        for node in graph.nodes:
            if node.type == 'Conv':
                conv_node = node
                break

        mod_conv = model_helper.get_module('layer1.0')
        ret = GraphChecker.check_quantize_type(conv_node, mod_conv)
        self.assertEqual(ret, True)

    def test_conv3d_valid_dilation_(self):
        # torch model
        torch_model = models.Net3d002().to(torch.device("cpu"))
        onnx_file = os.path.join(self.temp_folder, 'net3d.onnx')
        models.create_onnx(torch_model, torch_model.args_shape, onnx_file)
        # graph
        graph = Parser.parse_net_to_graph(onnx_file)
        model_helper = ModuleHelper(torch_model)
        conv_node = None
        for node in graph.nodes:
            if node.type == 'Conv':
                conv_node = node
                break
        mod_conv = model_helper.get_module('layer1.0')
        ret = GraphChecker.check_quantize_type(conv_node, mod_conv)
        self.assertEqual(ret, True)

    def test_conv3d_invalid_dilation(self):
        # torch model
        torch_model = models.Net3d003().to(torch.device("cpu"))
        onnx_file = os.path.join(self.temp_folder, 'net3d.onnx')
        models.create_onnx(torch_model, torch_model.args_shape, onnx_file)
        # graph
        graph = Parser.parse_net_to_graph(onnx_file)
        model_helper = ModuleHelper(torch_model)
        conv_node = None
        for node in graph.nodes:
            if node.type == 'Conv':
                conv_node = node
                break

        mod_conv = model_helper.get_module('layer1.0')
        ret = GraphChecker.check_quantize_type(conv_node, mod_conv)
        self.assertEqual(ret, False)

    def test_conv3d_invalid_dilation_(self):
        # torch model
        torch_model = models.Net3d004().to(torch.device("cpu"))
        onnx_file = os.path.join(self.temp_folder, 'net3d.onnx')
        models.create_onnx(torch_model, torch_model.args_shape, onnx_file)
        # graph
        graph = Parser.parse_net_to_graph(onnx_file)
        model_helper = ModuleHelper(torch_model)
        conv_node = None
        for node in graph.nodes:
            if node.type == 'Conv':
                conv_node = node
                break

        mod_conv = model_helper.get_module('layer1.0')
        ret = GraphChecker.check_quantize_type(conv_node, mod_conv)
        self.assertEqual(ret, False)

    def test_get_support_dmq_balancer_types(self):
        ret = GraphQuerier.get_support_dmq_balancer_types()
        ans = set(['Conv2d', 'Conv3d', 'Linear', 'Conv1d', 'ConvTranspose2d',
                  'Conv', 'Gemm', 'MatMul', 'ConvTranspose', 'ConvTranspose1d'])
        self.assertEqual(set(ret), ans)

    def test_get_support_dmq_balancer_layers(self):
        self.graph.add_model(self.model_001)
        layer_names = GraphQuerier.get_support_dmq_balancer_layers(self.graph)
        ans = ['layer1.0', 'layer2.0', 'layer3.0', 'layer4.0', 'layer5.0', 'layer6.0', 'fc.0', 'fc.2', 'fc.5']
        self.assertEqual(layer_names, ans)


class TestCheckGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_check_graph')
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


    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_support_quant_layer2type(self):
        ''' test GraphQuerier.get_support_quant_layer2type '''
        layer_types = GraphQuerier.get_support_quant_layer2type(self.graph)
        self.assertEqual(layer_types['layer1.0'], 'Conv')
        self.assertEqual(layer_types['fc.0'], 'Gemm')
        self.assertEqual(layer_types['fc.2'], 'MatMul')

    def test_get_support_qat_layer2type(self):
        ''' test GraphQuerier.get_support_qat_layer2type'''
        layer_types = GraphQuerier.get_support_qat_layer2type(self.graph)
        self.assertEqual(layer_types['layer1.0'], 'Conv')
        self.assertEqual(layer_types['fc.0'], 'Gemm')
        self.assertEqual(layer_types['fc.2'], 'MatMul')

    def test_dilation_not1(self):
        onnx_file = os.path.join(self.temp_folder, 'net_002.onnx')
        model_002 = models.Net002().to(torch.device("cpu"))
        Parser.export_onnx(model_002, self.args, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)

        layers = GraphQuerier.get_support_quant_layers(graph)
        self.assertEqual(layers, ['layer1.0', 'layer2.0'])


    def test_deconv_group_not1(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.deconv = torch.nn.ConvTranspose2d(
                    in_channels=4, out_channels=12, kernel_size=[3, 3],
                    stride=1, padding=0, output_padding=0,
                    groups=2, bias=True, dilation=1,
                    padding_mode='zeros')

            def forward(self, x):
                return self.deconv(x)

        onnx_file = BytesIO()
        test_model =TestModel()
        test_model.eval()
        Parser.export_onnx(test_model, torch.randn(1, 4, 12, 12), onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)

        layers = GraphQuerier.get_support_quant_layers(graph)
        self.assertEqual(layers, ['deconv'])


    def test_deconv_group_not1_with_module(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.deconv = torch.nn.ConvTranspose2d(
                    in_channels=4, out_channels=12, kernel_size=[3, 3],
                    stride=1, padding=0, output_padding=0,
                    groups=2, bias=True, dilation=1,
                    padding_mode='zeros')

            def forward(self, x):
                return self.deconv(x)

        onnx_file = BytesIO()
        test_model =TestModel()
        test_model.eval()
        Parser.export_onnx(test_model, torch.randn(1, 4, 12, 12), onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        graph.add_model(test_model)
        layers = GraphQuerier.get_support_quant_layers(graph)
        self.assertEqual(layers, ['deconv'])

    def test_deconv_group_with_default(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.deconv = torch.nn.ConvTranspose2d(
                    in_channels=4, out_channels=12, kernel_size=[3, 3],
                    stride=1, padding=0, output_padding=0,
                    groups=1, bias=True, dilation=1,
                    padding_mode='zeros')

            def forward(self, x):
                return self.deconv(x)

        onnx_file = BytesIO()
        test_model =TestModel()
        test_model.eval()
        Parser.export_onnx(test_model, torch.randn(1, 4, 12, 12), onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        deconv_node = graph.get_node_by_name('deconv')
        for attribute in deconv_node.proto.attribute:
            if attribute.name == 'group':
                attribute.name = 'groups'
                break
        layers = GraphQuerier.get_support_quant_layers(graph)
        self.assertEqual(layers, ['deconv'])


    def test_conv3d(self):
        onnx_file = os.path.join(self.temp_folder, 'Net3d.onnx')
        model_conv3d = models.Net3d().to(torch.device("cpu"))
        args = list()
        for input_shape in [(1, 2, 4, 14, 14)]:
            args.append(torch.randn(input_shape))
        args = tuple(args)

        Parser.export_onnx(model_conv3d, args, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)

        layers = GraphQuerier.get_support_quant_layers(graph)
        self.assertEqual(layers, ['layer1.0'])
        layer_types = GraphQuerier.get_support_quant_layer2type(graph)
        self.assertEqual(layer_types['layer1.0'], 'Conv')

    def test_get_act_symmetric_limit_layers(self):
        onnx_file = os.path.join(self.temp_folder, 'Net3d.onnx')
        model_conv3d = models.Net3d().to(torch.device("cpu"))
        args = list()
        for input_shape in [(1, 2, 4, 14, 14)]:
            args.append(torch.randn(input_shape))
        args = tuple(args)

        Parser.export_onnx(model_conv3d, args, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)

        layers = GraphQuerier.get_act_symmetric_limit_layers(graph)
        self.assertEqual(layers, ['layer1.0'])
        layer_types = GraphQuerier.get_support_quant_layer2type(graph)
        self.assertEqual(layer_types['layer1.0'], 'Conv')


    def test_amct_ops(self):
        model_proto = onnx_pb.ModelProto()
        model_proto.producer_name = 'model'
        graph_proto = onnx_pb.GraphProto()
        ascend_quant = graph_proto.node.add()
        ascend_quant.name = 'layer1.quant'
        ascend_quant.op_type = 'AscendQuant'
        model_proto.graph.CopyFrom(graph_proto)

        graph = Graph(model_proto)
        self.assertRaises(RuntimeError, GraphChecker.check_quant_behaviours, graph)

    def test_check_reused_node_not_support(self):
        onnx_file = os.path.join(self.temp_folder, 'reused_model.onnx')
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
                self.conv1_1 = torch.nn.Conv2d(32, 32, 1, 1)
                self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)

            # x represents our data
            def forward(self, x):
                # Pass data through conv1
                x = self.conv1(x)
                # Use the rectified-linear activation function over x
                x = F.relu(x)

                # repeated conv
                x = self.conv1_1(x)
                x = F.relu(x)

                x = self.conv1_1(x)
                x = F.relu(x)

                x = self.conv2(x)
                x = F.relu(x)
                return x
        model = Net()
        model.eval()
        dummy_input = torch.randn(1, 1, 28, 28)
        Parser.export_onnx(model, dummy_input, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)

        node_conv1 = graph.get_node_by_name('conv1')
        node_conv1_1 = graph.get_node_by_name('conv1_1')
        node_conv2 = graph.get_node_by_name('conv2')
        self.assertTrue(GraphChecker.check_graph_quantize_type(node_conv1))
        self.assertFalse(GraphChecker.check_graph_quantize_type(node_conv1_1))
        self.assertTrue(GraphChecker.check_graph_quantize_type(node_conv2))

    def test_check_input_dim_reduction_not_support(self):
        onnx_file = os.path.join(self.temp_folder, 'dim_reduction_model.onnx')
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = torch.nn.Conv2d(16, 32, 3, 1)
                self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)

            # x represents our data
            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                x = F.relu(x)
                return x
        model = Net()
        model.eval()
        dummy_input = torch.randn(1, 16, 28, 28)
        Parser.export_onnx(model, dummy_input, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)

        node_conv1 = graph.get_node_by_name('conv1')
        node_conv2 = graph.get_node_by_name('conv2')
        node_conv2.set_attr('input_dimension_reduction', True)
        self.assertTrue(GraphChecker.check_graph_selective_prune_type(node_conv1))
        self.assertFalse(GraphChecker.check_graph_selective_prune_type(node_conv2))
        self.assertFalse(GraphChecker.check_special_limit(node_conv2))
        self.assertFalse(GraphChecker.check_prune_limit(node_conv2))

        layers = GraphQuerier.get_support_selective_prune_layer2type(graph)
        self.assertEqual(list(layers.keys()), ['conv1'])
        graph.add_model(model)
        layers = GraphQuerier.get_support_selective_prune_layer2type(graph)
        self.assertEqual(list(layers.keys()), ['conv1'])
        node_conv1.set_attr('is_reuse', True)
        self.assertFalse(GraphChecker.check_prune_limit(node_conv1))
        self.assertFalse(GraphChecker.check_graph_selective_prune_type(node_conv1))

    def test_get_name_type_dict(self):
        layer_types = GraphQuerier.get_name_type_dict(self.graph)
        self.assertEqual(layer_types['layer1.0'], 'Conv')
        self.assertEqual(layer_types['fc.0'], 'Gemm')
        self.assertEqual(layer_types['fc.2'], 'MatMul')

    def test_invalid_padding(self):
        model = models.SingleConv().to(torch.device("cpu"))
        args = list()
        for input_shape in model.args_shape:
            args.append(torch.randn(input_shape))
        args = tuple(args)

        onnx_file = os.path.join(self.temp_folder, 'SingleConv.onnx')
        Parser.export_onnx(model, args, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        graph.add_model(model)
        layers = GraphQuerier.get_support_quant_layers(graph)
        self.assertEqual(layers, ['layer2'])

    def test_matmul_2dim(self):
        model = models.MatmulDim().to(torch.device("cpu"))
        args = list()
        for input_shape in model.args_shape:
            args.append(torch.randn(input_shape))
        args = tuple(args)

        onnx_file = os.path.join(self.temp_folder, 'matmul_2dim.onnx')
        Parser.export_onnx(model, args, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        graph.add_model(model)
        layers = GraphQuerier.get_support_quant_layers(graph)
        self.assertEqual(layers, ['layer1', 'layer2'])

        graph.model = None
        layers = GraphQuerier.get_support_quant_layers(graph)
        if '1.10' in torch.__version__:
            self.assertEqual(layers, ['layer1', 'layer2', 'MatMul_4'])
        else:
            self.assertEqual(layers, ['layer1', 'layer2'])

    def test_check_prune_reused_node(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
                self.conv1_1 = torch.nn.Conv2d(32, 32, 1, 1)
                self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)

            # x represents our data
            def forward(self, x):
                # Pass data through conv1
                x = self.conv1(x)
                # Use the rectified-linear activation function over x
                x = F.relu(x)

                # repeated conv
                x = self.conv1_1(x)
                x = F.relu(x)

                x = self.conv1_1(x)
                x = F.relu(x)

                x = self.conv2(x)
                x = F.relu(x)
                return x

        model = Net()
        dummy_input = torch.randn(1, 1, 28, 28)

        tmp_onnx = BytesIO()
        Parser.export_onnx(model, dummy_input, tmp_onnx)
        graph = Parser.parse_net_to_graph(tmp_onnx)
        graph.model = model

        conv1_prune = GraphChecker.check_graph_prune_type(graph.get_node_by_name('conv1'))
        self.assertEqual(conv1_prune, False)

        layers = GraphQuerier.get_support_prune_layer2type(graph)
        self.assertEqual(list(layers.keys()), ['conv2'])

    def test_export_onnx_args_type(self):
        model = torch.nn.Module()
        onnx_file = os.path.join(self.temp_folder, 'test_exception.onnx')
        args = list()
        try:
            Parser.export_onnx(model, args, onnx_file)
        except Exception as e:
            self.assertEqual("input data type must be tuple or torch.Tensor" in str(e), True)

    def test_get_support_dmq_balancer_layers(self):
        layer_names = GraphQuerier.get_support_dmq_balancer_layers(self.graph)
        ans = ['layer1.0', 'layer2.0', 'layer3.0', 'layer4.0', 'layer5.0', 'layer6.0', 'fc.0', 'fc.2', 'fc.5']
        self.assertEqual(layer_names, ans)

    def test_check_distill_type_conv2d(self):
        mod_name = 'conv1'
        mod = torch.nn.Conv2d(3, 3, 3)
        self.assertTrue(GraphChecker.check_distill_type(mod_name, mod))

    def test_check_distill_type_relu(self):
        mod_name = 'relu'
        mod = torch.nn.ReLU()
        self.assertFalse(GraphChecker.check_distill_type(mod_name, mod))

    def test_get_support_winograd_quant_layers(self):
        self.graph.model = None
        layer_names = GraphQuerier.get_support_winograd_quant_layers(self.graph)
        self.assertEqual(len(layer_names), 6)
        self.graph.add_model(self.model_001)
        layer_names = GraphQuerier.get_support_winograd_quant_layers(self.graph)
        self.assertEqual(len(layer_names), 6)

    def test_get_support_winograd_layer_types(self):
        layer_types = GraphQuerier.get_support_winograd_layer_types()
        self.assertIn('Conv2d', layer_types)

    def test_check_padding_mode_conv1d(self):
        mod = torch.nn.Conv1d(3,3,3,padding_mode='zeros')
        mod_type = 'Conv1d'
        mod_name = 'conv1'
        self.assertTrue(GraphChecker.check_padding_mode(mod_type, mod_name, mod))
        mod = torch.nn.Conv1d(3,3,3,padding_mode='reflect')
        self.assertFalse(GraphChecker.check_padding_mode(mod_type, mod_name, mod))

    def test_check_graph_int16_quantize_type_conv1d(self):
        model = torch.nn.Sequential(torch.nn.Conv1d(3,3,3,padding_mode='zeros'))
        model_onnx = BytesIO()
        Parser.export_onnx(model, torch.randn(3, 3, 3), model_onnx)
        graph = Parser.parse_net_to_graph(model_onnx)
        node = graph.get_node_by_name('0')
        self.assertTrue(GraphChecker.check_graph_int16_quantize_type(node))
        self.assertTrue(GraphChecker.check_graph_shared_type(node))

    def test_check_rnn_limit_not_rnn(self):
        mod = torch.nn.Conv2d(3,3,3,padding_mode='zeros')
        mod_type = 'Conv2d'
        mod_name = 'conv'
        self.assertTrue(GraphChecker.check_rnn_limit(mod_type, mod_name, mod))

    def test_check_rnn_limit_invalid_num_layers(self):
        mod = torch.nn.LSTM(10, 20, 2)
        mod_type = 'LSTM'
        mod_name = 'lstm'
        self.assertFalse(GraphChecker.check_rnn_limit(mod_type, mod_name, mod))

    def test_check_rnn_limit_invalid_bidirectional(self):
        mod = torch.nn.LSTM(1, 20, 2, bidirectional=True)
        mod_type = 'LSTM'
        mod_name = 'lstm'
        self.assertFalse(GraphChecker.check_rnn_limit(mod_type, mod_name, mod))

    def test_check_rnn_limit_invalid_dropout(self):
        mod = torch.nn.LSTM(1, 20, 2, dropout=1)
        mod_type = 'LSTM'
        mod_name = 'lstm'
        self.assertFalse(GraphChecker.check_rnn_limit(mod_type, mod_name, mod))
