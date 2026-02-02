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
import unittest
import torch
from onnx import onnx_pb

from amct_pytorch.amct_pytorch_inner.amct_pytorch.ada_round.ada_round_groups import get_ada_round_groups
from amct_pytorch.amct_pytorch_inner.amct_pytorch.ada_round.ada_round_groups import _is_weight_ada_round
from amct_pytorch.amct_pytorch_inner.amct_pytorch.ada_round.ada_round_groups import _match_gelu_subgraph
from amct_pytorch.amct_pytorch_inner.amct_pytorch.ada_round.ada_round_groups import _match_gelu_tanh_subgraph
from amct_pytorch.amct_pytorch_inner.amct_pytorch.ada_round.ada_round_groups import _match_rrelu_subgraph
from amct_pytorch.amct_pytorch_inner.amct_pytorch.ada_round.ada_round_groups import _get_other_activation
from amct_pytorch.amct_pytorch_inner.amct_pytorch.graph.graph import Graph
from amct_pytorch.amct_pytorch_inner.amct_pytorch.parser.parser import Parser
from amct_pytorch.amct_pytorch_inner.amct_pytorch.common.utils.util import version_higher_than

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]


class TestAdaRoundGroups(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_ada_round_groups')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.quant_config = {
            'linear':{
                'weight_quant_params': {
                    'wts_algo': 'ada_quantize'
                }
            }
        }

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_ada_round_groups_relu(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)
            def forward(self, x):
                x = self.linear(x)
                x = torch.nn.functional.relu(x)
                return x

        model = TestModel()
        input_data = torch.randn(5, 5)
        onnx_file = os.path.join(self.temp_folder, 'test_model.onnx')
        Parser.export_onnx(model, input_data, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        ada_round_groups = get_ada_round_groups(graph, self.quant_config)
        self.assertEqual(ada_round_groups[0][0], 'linear')
        self.assertTrue(isinstance(ada_round_groups[0][1], torch.nn.ReLU))

    def test_get_ada_round_groups_rrelu(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)
            def forward(self, x):
                x = self.linear(x)
                x = torch.nn.functional.rrelu(x, lower=0.3, upper=0.7)
                return x

        model = TestModel()
        input_data = torch.randn(5, 5)
        onnx_file = os.path.join(self.temp_folder, 'test_model.onnx')
        Parser.export_onnx(model, input_data, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        ada_round_groups = get_ada_round_groups(graph, self.quant_config)
        self.assertEqual(ada_round_groups[0][0], 'linear')
        if not version_higher_than(torch.__version__, '2.1.0'):
            self.assertTrue(isinstance(ada_round_groups[0][1], torch.nn.RReLU))
        else:
            self.assertTrue(isinstance(ada_round_groups[0][1], torch.nn.LeakyReLU))

    def test_get_ada_round_groups_leaky_relu(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)
            def forward(self, x):
                x = self.linear(x)
                x = torch.nn.functional.leaky_relu(x, negative_slope=0.123)
                return x

        model = TestModel()
        input_data = torch.randn(5, 5)
        onnx_file = os.path.join(self.temp_folder, 'test_model.onnx')
        Parser.export_onnx(model, input_data, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        ada_round_groups = get_ada_round_groups(graph, self.quant_config)
        self.assertEqual(ada_round_groups[0][0], 'linear')
        self.assertTrue(isinstance(ada_round_groups[0][1], torch.nn.LeakyReLU))

    def test_get_ada_round_groups_prelu(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Conv2d(5, 5, kernel_size=3)
                self.prelu = torch.nn.PReLU(num_parameters=5, init=1)
            def forward(self, x):
                x = self.linear(x)
                x = self.prelu(x)
                return x

        model = TestModel()
        input_data = torch.randn(1, 5, 16, 16)
        onnx_file = os.path.join(self.temp_folder, 'test_model.onnx')
        Parser.export_onnx(model, input_data, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        ada_round_groups = get_ada_round_groups(graph, self.quant_config)
        self.assertEqual(ada_round_groups[0][0], 'linear')
        self.assertTrue(isinstance(ada_round_groups[0][1], torch.nn.PReLU))
        except_weight = torch.ones(5)
        self.assertTrue((ada_round_groups[0][1].weight.data == except_weight).all())
        self.assertFalse(ada_round_groups[0][1].weight.requires_grad)

    def test_get_ada_round_groups_gelu(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)
            def forward(self, x):
                x = self.linear(x)
                x = torch.nn.functional.gelu(x)
                return x

        model = TestModel()
        input_data = torch.randn(5, 5)
        onnx_file = os.path.join(self.temp_folder, 'test_model.onnx')
        Parser.export_onnx(model, input_data, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        ada_round_groups = get_ada_round_groups(graph, self.quant_config)
        self.assertEqual(ada_round_groups[0][0], 'linear')
        self.assertTrue(isinstance(ada_round_groups[0][1], torch.nn.GELU))

    def test_get_ada_round_groups_relu6(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)
            def forward(self, x):
                x = self.linear(x)
                x = torch.nn.functional.relu6(x)
                return x

        model = TestModel()
        input_data = torch.randn(5, 5)
        onnx_file = os.path.join(self.temp_folder, 'test_model.onnx')
        Parser.export_onnx(model, input_data, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        ada_round_groups = get_ada_round_groups(graph, self.quant_config)
        self.assertEqual(ada_round_groups[0][0], 'linear')
        self.assertTrue(isinstance(ada_round_groups[0][1], torch.nn.ReLU6))

    def test_get_ada_round_groups_sigmoid(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)
            def forward(self, x):
                x = self.linear(x)
                x = torch.nn.functional.sigmoid(x)
                return x

        model = TestModel()
        input_data = torch.randn(5, 5)
        onnx_file = os.path.join(self.temp_folder, 'test_model.onnx')
        Parser.export_onnx(model, input_data, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        ada_round_groups = get_ada_round_groups(graph, self.quant_config)
        self.assertEqual(ada_round_groups[0][0], 'linear')
        self.assertTrue(isinstance(ada_round_groups[0][1], torch.nn.Sigmoid))

    def test_get_ada_round_groups_tanh(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)
            def forward(self, x):
                x = self.linear(x)
                x = torch.nn.functional.tanh(x)
                return x

        model = TestModel()
        input_data = torch.randn(5, 5)
        onnx_file = os.path.join(self.temp_folder, 'test_model.onnx')
        Parser.export_onnx(model, input_data, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        ada_round_groups = get_ada_round_groups(graph, self.quant_config)
        self.assertEqual(ada_round_groups[0][0], 'linear')
        self.assertTrue(isinstance(ada_round_groups[0][1], torch.nn.Tanh))

    def test_is_weight_ada_round_not_quant(self):
        quant_config = dict()
        ret = _is_weight_ada_round(quant_config, 'name')
        self.assertFalse(ret)

    def test_not_weight_ada_round(self):
        quant_config = {
            'name':{
                'weight_quant_params': {
                    'wts_algo': 'arq_quantize'
                }
            }
        }
        ret = _is_weight_ada_round(quant_config, 'name')
        self.assertFalse(ret)

    def test_match_gelu_01(self):
        input_nodes = list()
        ret = _match_gelu_subgraph(input_nodes)
        self.assertFalse(ret)

    def test_match_gelu_02(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5,5)
            def forward(self, x):
                x0 = self.linear(x)
                x1 = x0 / 2
                x2 = torch.erf(x1) + x1
                x3 = x0 * x2
                return x3

        model = TestModel()
        input_data = torch.randn(5, 5)
        onnx_file = os.path.join(self.temp_folder, 'test_model.onnx')
        Parser.export_onnx(model, input_data, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        ada_round_groups = get_ada_round_groups(graph, self.quant_config)
        except_groups = [
            ['linear', None]
        ]
        self.assertEqual(ada_round_groups, except_groups)

    def test_match_gelu_03(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5,5)
            def forward(self, x):
                x0 = self.linear(x)
                x1 = x0 / 2
                x2 = torch.erf(x1)
                x3 = x0 * x2
                return x3

        model = TestModel()
        input_data = torch.randn(5, 5)
        onnx_file = os.path.join(self.temp_folder, 'test_model.onnx')
        Parser.export_onnx(model, input_data, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        ada_round_groups = get_ada_round_groups(graph, self.quant_config)
        except_groups = [
            ['linear', None]
        ]
        self.assertEqual(ada_round_groups, except_groups)

    def test_match_gelu_04(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5,5)
            def forward(self, x):
                x0 = self.linear(x)
                x1 = x0 / 1.4142135381698608
                x2 = torch.erf(x1)
                x3 = x2 + x2
                x4 = x0 * x3
                return x4

        model = TestModel()
        input_data = torch.randn(5, 5)
        onnx_file = os.path.join(self.temp_folder, 'test_model.onnx')
        Parser.export_onnx(model, input_data, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        ada_round_groups = get_ada_round_groups(graph, self.quant_config)
        except_groups = [
            ['linear', None]
        ]
        self.assertEqual(ada_round_groups, except_groups)

    def test_match_gelu_05(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5,5)
            def forward(self, x):
                x0 = self.linear(x)
                x1 = x0 / 1.4142135381698608
                x2 = torch.erf(x1)
                x3 = x2 + 2
                x4 = x3 * x3
                x5 = x0 * x4
                return x5

        model = TestModel()
        input_data = torch.randn(5, 5)
        onnx_file = os.path.join(self.temp_folder, 'test_model.onnx')
        Parser.export_onnx(model, input_data, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        ada_round_groups = get_ada_round_groups(graph, self.quant_config)
        except_groups = [
            ['linear', None]
        ]
        self.assertEqual(ada_round_groups, except_groups)

    def test_match_gelu_06(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5,5)
            def forward(self, x):
                x0 = self.linear(x)
                x1 = x0 / 1.4142135381698608
                x2 = torch.erf(x1)
                x3 = x2 + 2
                x4 = x0 * x3
                return x4

        model = TestModel()
        input_data = torch.randn(5, 5)
        onnx_file = os.path.join(self.temp_folder, 'test_model.onnx')
        Parser.export_onnx(model, input_data, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        ada_round_groups = get_ada_round_groups(graph, self.quant_config)
        except_groups = [
            ['linear', None]
        ]
        self.assertEqual(ada_round_groups, except_groups)

    def test_match_gelu_07(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5,5)
            def forward(self, x):
                x0 = self.linear(x)
                x1 = x0 / 1.4142135381698608
                x2 = torch.erf(x1)
                x3 = x2 + 1
                x4 = x0 * x3
                x5 = x4 * x4
                return x5

        model = TestModel()
        input_data = torch.randn(5, 5)
        onnx_file = os.path.join(self.temp_folder, 'test_model.onnx')
        Parser.export_onnx(model, input_data, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        ada_round_groups = get_ada_round_groups(graph, self.quant_config)
        except_groups = [
            ['linear', None]
        ]
        self.assertEqual(ada_round_groups, except_groups)

    def test_match_gelu_08(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5,5)
            def forward(self, x):
                x0 = self.linear(x)
                x1 = x0 / 1.4142135381698608
                x2 = torch.erf(x1)
                x3 = x2 + 1
                x4 = x0 * x3
                x5 = x4 - 1
                return x5

        model = TestModel()
        input_data = torch.randn(5, 5)
        onnx_file = os.path.join(self.temp_folder, 'test_model.onnx')
        Parser.export_onnx(model, input_data, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        ada_round_groups = get_ada_round_groups(graph, self.quant_config)
        except_groups = [
            ['linear', None]
        ]
        self.assertEqual(ada_round_groups, except_groups)

    def test_match_gelu_09(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5,5)
            def forward(self, x):
                x0 = self.linear(x)
                x1 = x0 / 1.4142135381698608
                x2 = torch.erf(x1)
                x3 = x2 + 1
                x4 = x0 * x3
                x5 = x4 * 0.6
                return x5

        model = TestModel()
        input_data = torch.randn(5, 5)
        onnx_file = os.path.join(self.temp_folder, 'test_model.onnx')
        Parser.export_onnx(model, input_data, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        ada_round_groups = get_ada_round_groups(graph, self.quant_config)
        except_groups = [
            ['linear', None]
        ]
        self.assertEqual(ada_round_groups, except_groups)

    def test_match_gelu_tanh_01(self):
        input_nodes = list()
        ret = _match_gelu_tanh_subgraph(input_nodes)
        self.assertFalse(ret)

    def test_match_gelu_tanh_02(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5,5)
            def forward(self, x):
                x0 = self.linear(x)
                x1 = x0 * x0
                x2 = x1 * x1
                x3 = torch.mul(2, x2)
                x4 = x3 + x0
                x5 = torch.mul(2, x4)
                x6 = torch.tanh(x5)
                x7 = torch.add(2, x6)
                x8 = x7 * x0
                x9 = torch.mul(0.6, x8)
                return x9

        model = TestModel()
        input_data = torch.randn(5, 5)
        onnx_file = os.path.join(self.temp_folder, 'test_model.onnx')
        Parser.export_onnx(model, input_data, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        ada_round_groups = get_ada_round_groups(graph, self.quant_config)
        except_groups = [
            ['linear', None]
        ]
        self.assertEqual(ada_round_groups, except_groups)

    def test_match_gelu_tanh_03(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5,5)
            def forward(self, x):
                x0 = self.linear(x)
                x1 = x0 * x0
                x2 = x1 * x1
                x3 = torch.mul(2, x2)
                x4 = x3 + x0
                x5 = torch.mul(2, x4)
                x6 = torch.tanh(x5)
                x7 = torch.add(2, x6)
                x8 = x7 * x0
                x9 = torch.mul(0.6, x8)
                return x9

        model = TestModel()
        input_data = torch.randn(5, 5)
        onnx_file = os.path.join(self.temp_folder, 'test_model.onnx')
        Parser.export_onnx(model, input_data, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        ada_round_groups = get_ada_round_groups(graph, self.quant_config)
        except_groups = [
            ['linear', None]
        ]
        self.assertEqual(ada_round_groups, except_groups)

    def test_match_gelu_tanh_04(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5,5)
            def forward(self, x):
                x0 = self.linear(x)
                x1 = x0 * x0
                x2 = x1 * x0
                x3 = torch.mul(2, x2)
                x4 = x3 + x3 + x0
                x5 = torch.mul(2, x4)
                x6 = torch.tanh(x5)
                x7 = torch.add(2, x6)
                x8 = x7 * x0
                x9 = torch.mul(0.6, x8)
                return x9

        model = TestModel()
        input_data = torch.randn(5, 5)
        onnx_file = os.path.join(self.temp_folder, 'test_model.onnx')
        Parser.export_onnx(model, input_data, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        ada_round_groups = get_ada_round_groups(graph, self.quant_config)
        except_groups = [
            ['linear', None]
        ]
        self.assertEqual(ada_round_groups, except_groups)

    def test_match_gelu_tanh_05(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5,5)
            def forward(self, x):
                x0 = self.linear(x)
                x1 = x0 * x0
                x2 = x1 * x0
                x3 = torch.mul(2, x2)
                x4 = x3 + x0
                x5 = torch.mul(2, x4)
                x6 = torch.tanh(x5)
                x7 = torch.add(2, x6)
                x8 = x7 * x0
                x9 = torch.mul(0.6, x8)
                return x9

        model = TestModel()
        input_data = torch.randn(5, 5)
        onnx_file = os.path.join(self.temp_folder, 'test_model.onnx')
        Parser.export_onnx(model, input_data, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        ada_round_groups = get_ada_round_groups(graph, self.quant_config)
        except_groups = [
            ['linear', None]
        ]
        self.assertEqual(ada_round_groups, except_groups)

    def test_match_gelu_tanh_06(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5,5)
            def forward(self, x):
                x0 = self.linear(x)
                x1 = x0 * x0
                x2 = x1 * x0
                x3 = torch.mul(0.044714998453855515, x2)
                x4 = x3 + x0
                x5 = x4 * x4
                x6 = torch.tanh(x5)
                x7 = torch.add(2, x6)
                x8 = x7 * x0
                x9 = torch.mul(0.6, x8)
                return x9

        model = TestModel()
        input_data = torch.randn(5, 5)
        onnx_file = os.path.join(self.temp_folder, 'test_model.onnx')
        Parser.export_onnx(model, input_data, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        ada_round_groups = get_ada_round_groups(graph, self.quant_config)
        except_groups = [
            ['linear', None]
        ]
        self.assertEqual(ada_round_groups, except_groups)

    def test_match_gelu_tanh_07(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5,5)
            def forward(self, x):
                x0 = self.linear(x)
                x1 = x0 * x0
                x2 = x1 * x0
                x3 = torch.mul(0.044714998453855515, x2)
                x4 = x3 + x0
                x5 = torch.mul(2, x4)
                x6 = torch.tanh(x5) * x5
                x7 = torch.add(2, x6)
                x8 = x7 * x0
                x9 = torch.mul(0.6, x8)
                return x9

        model = TestModel()
        input_data = torch.randn(5, 5)
        onnx_file = os.path.join(self.temp_folder, 'test_model.onnx')
        Parser.export_onnx(model, input_data, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        ada_round_groups = get_ada_round_groups(graph, self.quant_config)
        except_groups = [
            ['linear', None]
        ]
        self.assertEqual(ada_round_groups, except_groups)

    def test_match_gelu_tanh_08(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5,5)
            def forward(self, x):
                x0 = self.linear(x)
                x1 = x0 * x0
                x2 = x1 * x0
                x3 = torch.mul(0.044714998453855515, x2)
                x4 = x3 + x0
                x5 = torch.mul(2, x4)
                x6 = torch.tanh(x5)
                x7 = torch.add(2, x6)
                x8 = x7 * x0
                x9 = torch.mul(0.6, x8)
                return x9

        model = TestModel()
        input_data = torch.randn(5, 5)
        onnx_file = os.path.join(self.temp_folder, 'test_model.onnx')
        Parser.export_onnx(model, input_data, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        ada_round_groups = get_ada_round_groups(graph, self.quant_config)
        except_groups = [
            ['linear', None]
        ]
        self.assertEqual(ada_round_groups, except_groups)

    def test_match_gelu_tanh_09(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5,5)
            def forward(self, x):
                x0 = self.linear(x)
                x1 = x0 * x0
                x2 = x1 * x0
                x3 = torch.mul(0.044714998453855515, x2)
                x4 = x3 + x0
                x5 = torch.mul(0.7978845834732056, x4)
                x6 = torch.tanh(x5)
                x7 = x6 + x6
                x8 = x7 * x0
                x9 = torch.mul(0.6, x8)
                return x9

        model = TestModel()
        input_data = torch.randn(5, 5)
        onnx_file = os.path.join(self.temp_folder, 'test_model.onnx')
        Parser.export_onnx(model, input_data, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        ada_round_groups = get_ada_round_groups(graph, self.quant_config)
        except_groups = [
            ['linear', None]
        ]
        self.assertEqual(ada_round_groups, except_groups)

    def test_match_gelu_tanh_10(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5,5)
            def forward(self, x):
                x0 = self.linear(x)
                x1 = x0 * x0
                x2 = x1 * x0
                x3 = torch.mul(0.044714998453855515, x2)
                x4 = x3 + x0
                x5 = torch.mul(0.7978845834732056, x4)
                x6 = torch.tanh(x5)
                x7 = torch.add(2, x6)
                x8 = x7 * x0 * x7
                x9 = torch.mul(0.6, x8)
                return x9

        model = TestModel()
        input_data = torch.randn(5, 5)
        onnx_file = os.path.join(self.temp_folder, 'test_model.onnx')
        Parser.export_onnx(model, input_data, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        ada_round_groups = get_ada_round_groups(graph, self.quant_config)
        except_groups = [
            ['linear', None]
        ]
        self.assertEqual(ada_round_groups, except_groups)

    def test_match_gelu_tanh_11(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5,5)
            def forward(self, x):
                x0 = self.linear(x)
                x1 = x0 * x0
                x2 = x1 * x0
                x3 = torch.mul(0.044714998453855515, x2)
                x4 = x3 + x0
                x5 = torch.mul(0.7978845834732056, x4)
                x6 = torch.tanh(x5)
                x7 = torch.add(2, x6)
                x8 = x7 * x0
                x9 = torch.mul(0.6, x8)
                return x9

        model = TestModel()
        input_data = torch.randn(5, 5)
        onnx_file = os.path.join(self.temp_folder, 'test_model.onnx')
        Parser.export_onnx(model, input_data, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        ada_round_groups = get_ada_round_groups(graph, self.quant_config)
        except_groups = [
            ['linear', None]
        ]
        self.assertEqual(ada_round_groups, except_groups)

    def test_match_gelu_tanh_12(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5,5)
            def forward(self, x):
                x0 = self.linear(x)
                x1 = x0 * x0
                x2 = x1 * x0
                x3 = torch.mul(0.044714998453855515, x2)
                x4 = x3 + x0
                x5 = torch.mul(0.7978845834732056, x4)
                x6 = torch.tanh(x5)
                x7 = torch.add(1, x6)
                x8 = x7 * x0
                x9 = x8 * x8
                return x9

        model = TestModel()
        input_data = torch.randn(5, 5)
        onnx_file = os.path.join(self.temp_folder, 'test_model.onnx')
        Parser.export_onnx(model, input_data, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        ada_round_groups = get_ada_round_groups(graph, self.quant_config)
        except_groups = [
            ['linear', None]
        ]
        self.assertEqual(ada_round_groups, except_groups)

    def test_match_gelu_tanh_13(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5,5)
            def forward(self, x):
                x0 = self.linear(x)
                x1 = x0 * x0
                x2 = x1 * x0
                x3 = torch.mul(0.044714998453855515, x2)
                x4 = x3 + x0
                x5 = torch.mul(0.7978845834732056, x4)
                x6 = torch.tanh(x5)
                x7 = torch.add(1, x6)
                x8 = x7 * x0
                x9 = x8 - 1
                return x9

        model = TestModel()
        input_data = torch.randn(5, 5)
        onnx_file = os.path.join(self.temp_folder, 'test_model.onnx')
        Parser.export_onnx(model, input_data, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        ada_round_groups = get_ada_round_groups(graph, self.quant_config)
        except_groups = [
            ['linear', None]
        ]
        self.assertEqual(ada_round_groups, except_groups)

    def test_match_gelu_tanh_14(self):
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5,5)
            def forward(self, x):
                x0 = self.linear(x)
                x1 = x0 * x0
                x2 = x1 * x0
                x3 = torch.mul(0.044714998453855515, x2)
                x4 = x3 + x0
                x5 = torch.mul(0.7978845834732056, x4)
                x6 = torch.tanh(x5)
                x7 = torch.add(1, x6)
                x8 = x7 * x0
                x9 = torch.mul(0.6, x8)
                return x9

        model = TestModel()
        input_data = torch.randn(5, 5)
        onnx_file = os.path.join(self.temp_folder, 'test_model.onnx')
        Parser.export_onnx(model, input_data, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        ada_round_groups = get_ada_round_groups(graph, self.quant_config)
        except_groups = [
            ['linear', None]
        ]
        self.assertEqual(ada_round_groups, except_groups)

    def test_match_rrelu_01(self):
        input_nodes = list()
        ret = _match_rrelu_subgraph(input_nodes)
        self.assertFalse(ret)

    def test_match_rrelu_02(self):
        model_proto = onnx_pb.ModelProto()
        graph_input0 = model_proto.graph.input.add()
        graph_input0.name = 'input'
        rand = model_proto.graph.node.add()
        rand.name = 'rand'
        rand.op_type = 'RandomUniformLike'
        rand.input[:] = ['input']
        rand.output[:] = ['rand_out', 'rand_out1']
        graph = Graph(model_proto)

        input_nodes = [graph.nodes[0], graph.nodes[0]]
        ret = _match_rrelu_subgraph(input_nodes)
        self.assertFalse(ret)

    def test_match_rrelu_03(self):
        model_proto = onnx_pb.ModelProto()
        graph_input0 = model_proto.graph.input.add()
        graph_input0.name = 'input'
        rand = model_proto.graph.node.add()
        rand.name = 'rand'
        rand.op_type = 'RandomUniformLike'
        rand.input[:] = ['input']
        rand.output[:] = ['rand_out']
        sub = model_proto.graph.node.add()
        sub.name = 'sub'
        sub.op_type = 'Sub'
        sub.input[:] = ['rand_out']
        sub.output[:] = ['sub_out']
        graph = Graph(model_proto)

        input_nodes = [graph.nodes[0], graph.nodes[0]]
        ret = _match_rrelu_subgraph(input_nodes)
        self.assertFalse(ret)

    def test_get_other_activation_clip_no_min_max(self):
        model_proto = onnx_pb.ModelProto()
        graph_input0 = model_proto.graph.input.add()
        graph_input0.name = 'input'
        clip = model_proto.graph.node.add()
        clip.name = 'clip'
        clip.op_type = 'Clip'
        clip.input[:] = ['input']
        clip.output[:] = ['clip_out']
        graph = Graph(model_proto)

        consumers = [graph.nodes[0]]
        ret = _get_other_activation(consumers)
        self.assertIsNone(ret)

    def test_get_other_activation_clip_not_relu6(self):
        model_proto = onnx_pb.ModelProto()
        graph_input0 = model_proto.graph.input.add()
        graph_input0.name = 'input'

        clip_min_value = model_proto.graph.initializer.add()
        clip_min_value.name = 'clip_min'
        clip_min_value.data_type = onnx_pb.TensorProto.DataType.FLOAT
        clip_min_value.float_data[:] = [1]
        clip_min_value.dims[:] = []
 
        clip_max_value = model_proto.graph.initializer.add()
        clip_max_value.name = 'clip_max'
        clip_max_value.data_type = onnx_pb.TensorProto.DataType.FLOAT
        clip_max_value.float_data[:] = [5]
        clip_max_value.dims[:] = []
 
        clip = model_proto.graph.node.add()
        clip.name = 'clip'
        clip.op_type = 'Clip'
        clip.input[:] = ['input', 'clip_min', 'clip_max']
        clip.output[:] = ['clip_out']
        graph = Graph(model_proto)
 
        consumers = [graph.nodes[0]]
        ret = _get_other_activation(consumers)
        self.assertIsNone(ret)
