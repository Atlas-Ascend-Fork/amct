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
from io import BytesIO
from unittest.mock import patch

import numpy as np
import torch
from onnx import TensorProto, helper, numpy_helper

from amct_pytorch.classic.graph_based.amct_pytorch.graph.graph import Graph
from amct_pytorch.classic.graph_based.amct_pytorch.optimizer.graph_optimizer import (
    GraphOptimizer,
)
from amct_pytorch.classic.graph_based.amct_pytorch.optimizer.insert_weight_quant_pass import (
    InsertWeightQuantPass,
)
from amct_pytorch.classic.graph_based.amct_pytorch.parser.parser import Parser
from amct_pytorch.classic.graph_based.amct_pytorch.utils.onnx_initializer_util import (
    TensorProtoHelper,
)
from amct_pytorch.classic.graph_based.amct_pytorch.utils.quant_node import QuantOpInfo
from amct_pytorch.classic.graph_based.amct_pytorch.utils.vars import (
    QUANTIZABLE_TYPES,
)

from .utils import models, record_file_utils

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]
# 原生 INT4 需 onnx>=1.16；旧版 onnx 无 INT4 枚举，INT4 相关用例跳过
_INT4_SUPPORTED = 'INT4' in TensorProtoHelper.data_type_maps
_SKIP_INT4_MSG = 'onnx version too old for native INT4'


class TestWeightQuantPass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        QUANTIZABLE_TYPES.extend(['ConvTranspose2d', 'AvgPool2d'])
        cls.temp_folder = os.path.join(CUR_DIR, 'test_weight_quant_pass')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.model_001 = models.Net001().to(torch.device("cpu"))
        cls.args_shape = [(1, 2, 28, 28)]
        cls.args = list()
        for input_shape in cls.args_shape:
            cls.args.append(torch.randn(input_shape))
        cls.args = tuple(cls.args)

        tmp_onnx = BytesIO()
        Parser.export_onnx(cls.model_001, cls.args, tmp_onnx)
        cls._base_graph = Parser.parse_net_to_graph(tmp_onnx)

        cls.records = record_file_utils.generate_records(
            layers_length={"layer1.0": 16, "layer2.0": 16, "fc.2": 1}
        )

    def setUp(self):
        # Tests run passes/set_data that mutate the graph in place; give each test
        # a fresh copy so weight-packing/quantization state does not leak across tests.
        self.graph = type(self)._base_graph.deep_copy()

    @classmethod
    def tearDownClass(cls):
        QUANTIZABLE_TYPES.remove('ConvTranspose2d')
        QUANTIZABLE_TYPES.remove('AvgPool2d')
        os.popen('rm -r ' + cls.temp_folder)

    def test_quant_weight(self):
        passer = InsertWeightQuantPass(self.records)
        optimizer = GraphOptimizer()
        optimizer.add_pass(passer)
        optimizer.do_optimizer(self.graph, None)
        weight_dtype = (
            TensorProtoHelper(
                self.graph.get_node_by_name('layer1.0.sub_module.weight').proto
            )
            .get_data()
            .dtype
        )
        self.assertEqual(weight_dtype, 'int8')

    @unittest.skipUnless(_INT4_SUPPORTED, _SKIP_INT4_MSG)
    def test_quant_weight_int4(self):
        target_node = self.graph.get_node_by_name('fc.2')
        weight_node = QuantOpInfo.get_weight_node(target_node)
        original_dims = list(weight_node.proto.dims)
        element_count = int(np.prod(original_dims))

        with patch(
            'amct_pytorch.classic.graph_based.amct_pytorch.utils.quant_node.'
            'QuantOpInfo.get_dst_num_bits',
            return_value=4,
        ):
            passer = InsertWeightQuantPass(self.records)
            before_nodes = len(self.graph.nodes)
            optimizer = GraphOptimizer()
            optimizer.add_pass(passer)
            optimizer.do_optimizer(self.graph, None)
            after_nodes = len(self.graph.nodes)
            self.assertEqual(after_nodes - before_nodes, 0)

        self.assertEqual(
            weight_node.proto.data_type,
            TensorProtoHelper.data_type_maps['INT4'][0],
        )
        self.assertEqual(list(weight_node.proto.dims), original_dims)
        self.assertEqual(len(weight_node.proto.raw_data), (element_count + 1) // 2)

    def test_matmul_weight_quantizes_per_output_channel(self):
        class LinearModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 4, bias=False)
                self.linear.weight.data = torch.tensor(
                    [[1, 2], [10, 20], [100, 200], [1000, 2000]],
                    dtype=torch.float32,
                )

            def forward(self, inputs):
                return self.linear(inputs)

        model = LinearModel()
        tmp_onnx = BytesIO()
        Parser.export_onnx(model, torch.ones(1, 2), tmp_onnx)
        graph = Parser.parse_net_to_graph(tmp_onnx)
        node = graph.get_node_by_name('linear')
        self.assertEqual(node.type, 'MatMul')
        records = {
            'linear': {
                'weight_scale': np.array([1, 10, 100, 1000], dtype=np.float32),
                'weight_offset': np.zeros(4, dtype=np.int8),
                'wts_type': 'INT8',
            }
        }

        InsertWeightQuantPass(records).do_pass(graph, node)

        weight_node = QuantOpInfo.get_weight_node(node)
        quantized = TensorProtoHelper(weight_node.proto).get_data()
        np.testing.assert_array_equal(
            quantized,
            np.array([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=np.int8),
        )

    def test_matmul_multidimensional_weight_quantizes_per_output_channel(self):
        weight = np.stack(
            [
                np.full((2, 3), 1, dtype=np.float32),
                np.full((2, 3), 10, dtype=np.float32),
                np.full((2, 3), 100, dtype=np.float32),
                np.full((2, 3), 1000, dtype=np.float32),
            ],
            axis=-1,
        )
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        'MatMul', ['inputs', 'weight'], ['output'], name='linear'
                    )
                ],
                'linear_graph',
                [helper.make_tensor_value_info('inputs', TensorProto.FLOAT, [1, 2, 3])],
                [helper.make_tensor_value_info('output', TensorProto.FLOAT, None)],
                [numpy_helper.from_array(weight, name='weight')],
            )
        )
        graph = Graph(model)
        node = graph.get_node_by_name('linear')
        records = {
            'linear': {
                'weight_scale': np.array([1, 10, 100, 1000], dtype=np.float32),
                'weight_offset': np.zeros(4, dtype=np.int8),
                'wts_type': 'INT8',
            }
        }

        InsertWeightQuantPass(records).do_pass(graph, node)

        quantized = TensorProtoHelper(graph.get_node_by_name('weight').proto).get_data()
        self.assertEqual(list(quantized.shape), [2, 3, 4])
        np.testing.assert_array_equal(quantized, np.ones((2, 3, 4), dtype=np.int8))

    def test_matmul_transposed_weight_quantizes_per_output_channel(self):
        weight = np.array(
            [[1, 2], [10, 20], [100, 200], [1000, 2000]], dtype=np.float32
        )
        model = helper.make_model(
            helper.make_graph(
                [
                    helper.make_node(
                        'Transpose', ['weight'], ['weight_t'], name='weight_trans'
                    ),
                    helper.make_node(
                        'MatMul', ['inputs', 'weight_t'], ['output'], name='linear'
                    ),
                ],
                'linear_graph',
                [helper.make_tensor_value_info('inputs', TensorProto.FLOAT, [1, 2])],
                [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])],
                [numpy_helper.from_array(weight, name='weight')],
            )
        )
        graph = Graph(model)
        node = graph.get_node_by_name('linear')
        records = {
            'linear': {
                'weight_scale': np.array([1, 10, 100, 1000], dtype=np.float32),
                'weight_offset': np.zeros(4, dtype=np.int8),
                'wts_type': 'INT8',
            }
        }

        InsertWeightQuantPass(records).do_pass(graph, node)

        weight_node = graph.get_node_by_name('weight')
        quantized = TensorProtoHelper(weight_node.proto).get_data()
        self.assertEqual(list(weight_node.proto.dims), [4, 2])
        np.testing.assert_array_equal(
            quantized,
            np.array([[1, 2], [1, 2], [1, 2], [1, 2]], dtype=np.int8),
        )

    def test_rnn_weight_quant_success(self):
        class RNNModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(10, 20, 1)

            def forward(self, input_data, hx):
                x = self.lstm(input_data, hx)
                return x

        model = RNNModule()
        tmp_onnx = BytesIO()
        Parser.export_onnx(
            model,
            (torch.randn(1, 1, 10), (torch.randn(1, 1, 20), torch.randn(1, 1, 20))),
            tmp_onnx,
        )
        graph = Parser.parse_net_to_graph(tmp_onnx)
        node_name = 'lstm'
        node = graph.get_node_by_name(node_name)

        records = {
            node_name: {
                'weight_scale': np.array([1.0] * 4, dtype=np.float32),
                'weight_offset': np.array([0] * 4, dtype=np.int8),
                'recurrence_weight_scale': np.array([1.0] * 4, dtype=np.float32),
                'recurrence_weight_offset': np.array([0] * 4, dtype=np.int8),
                'wts_type': 'INT8',
            }
        }

        passer = InsertWeightQuantPass(records)
        passer.quant_recurrence_weight(node)

    def build_lstm_int4_case(self):
        """Build the LSTM graph node and INT4 records for recurrence-weight UT."""
        model = models.LSTMNet(10, 20, 1)
        tmp_onnx = BytesIO()
        Parser.export_onnx(
            model,
            (torch.randn(1, 1, 10), (torch.randn(1, 1, 20), torch.randn(1, 1, 20))),
            tmp_onnx,
        )
        graph = Parser.parse_net_to_graph(tmp_onnx)
        node = graph.get_node_by_name('lstm')

        scale = np.array([1.0] * 4, dtype=np.float32)
        records = {
            'lstm': {
                'weight_scale': scale,
                'weight_offset': np.array([0] * 4, dtype=np.int8),
                'recurrence_weight_scale': scale,
                'recurrence_weight_offset': np.array([0] * 4, dtype=np.int8),
                'wts_type': 'INT4',
            }
        }
        return node, records

    def run_recurrence_weight_with_spy(self, node, records):
        """Run quant_recurrence_weight while spying on weight_quant_np / set_data.

        Returns (mock_wqnp, set_data_type_args) so callers can assert the
        num_bits actually passed and the stored dtype.
        """
        passer = InsertWeightQuantPass(records)

        # Before fix: hardcoded 8 is passed regardless of get_dst_num_bits.
        # After fix: get_dst_num_bits returns 4 and 4 is passed.
        weight_quant_module = (
            'amct_pytorch.classic.graph_based.amct_pytorch.'
            'optimizer.insert_weight_quant_pass.weight_quant_np'
        )
        set_data_module = (
            'amct_pytorch.classic.graph_based.amct_pytorch.'
            'optimizer.insert_weight_quant_pass.TensorProtoHelper.set_data'
        )
        # Capture set_data calls via a closure that delegates to the real method.
        real_set_data = (
            TensorProtoHelper.set_data.__wrapped__
            if hasattr(TensorProtoHelper.set_data, '__wrapped__')
            else TensorProtoHelper.set_data
        )
        set_data_type_args = []

        def spy_set_data(self_inner, data, type_string=None, dims=None):
            set_data_type_args.append(type_string)
            return real_set_data(self_inner, data, type_string, dims)

        with (
            patch(
                'amct_pytorch.classic.graph_based.amct_pytorch.utils.quant_node.'
                'QuantOpInfo.get_dst_num_bits',
                return_value=4,
            ),
            patch(
                weight_quant_module,
                wraps=__import__(
                    'amct_pytorch.classic.graph_based.amct_pytorch.custom_op.arq.arq',
                    fromlist=['weight_quant_np'],
                ).weight_quant_np,
            ) as mock_wqnp,
            patch(set_data_module, spy_set_data),
        ):
            passer.quant_recurrence_weight(node)
        return mock_wqnp, set_data_type_args

    @unittest.skipUnless(_INT4_SUPPORTED, _SKIP_INT4_MSG)
    def test_recurrence_weight_int4(self):
        """wts_type=INT4: quant_recurrence_weight must call weight_quant_np with num_bits=4, not hardcoded 8."""
        node, records = self.build_lstm_int4_case()
        mock_wqnp, set_data_type_args = self.run_recurrence_weight_with_spy(
            node, records
        )

        # Assert weight_quant_np was called with num_bits=4, not 8
        self.assertTrue(mock_wqnp.called, 'weight_quant_np was never called')
        args, kwargs = mock_wqnp.call_args
        actual_num_bits = args[3] if len(args) > 3 else kwargs.get('num_bits')
        self.assertEqual(
            actual_num_bits,
            4,
            msg='expected weight_quant_np called with num_bits=4, got {}'.format(
                actual_num_bits
            ),
        )

        # Assert recurrence_weight is stored as INT4 when num_bits==4 (dtype label
        # matches the actual [-8,7] value range)
        self.assertTrue(
            set_data_type_args, 'TensorProtoHelper.set_data was never called'
        )
        actual_dtype = set_data_type_args[-1]
        self.assertEqual(
            actual_dtype,
            'INT4',
            msg='expected recurrence_weight stored as INT4, got {}'.format(
                actual_dtype
            ),
        )
