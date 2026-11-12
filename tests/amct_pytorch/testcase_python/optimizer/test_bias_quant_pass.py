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
from unittest.mock import patch

import numpy as np
import torch

from amct_pytorch.classic.graph_based.amct_pytorch.optimizer.graph_optimizer import (
    GraphOptimizer,
)
from amct_pytorch.classic.graph_based.amct_pytorch.optimizer.insert_bias_quant_pass import (
    InsertBiasQuantPass,
)
from amct_pytorch.classic.graph_based.amct_pytorch.optimizer.replace_bias_quant_pass import (
    ReplaceBiasQuantPass,
)
from amct_pytorch.classic.graph_based.amct_pytorch.parser.parser import Parser
from amct_pytorch.classic.graph_based.amct_pytorch.utils.onnx_initializer_util import (
    TensorProtoHelper,
)

from .utils import models, record_file_utils

CUR_DIR = os.path.split(os.path.realpath(__file__))[0]


class TestInsertBiasQuantPass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_bias_quant_pass')
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

        cls.records = record_file_utils.generate_records(
            layers_length={
                "layer1.0": 16,
                "layer2.0": 16,
                "fc.2": 1,
            }
        )

    @classmethod
    def tearDownClass(cls):
        os.popen('rm -r ' + cls.temp_folder)
        pass

    def test_quant_bias(self):
        optimizer = GraphOptimizer()
        optimizer.add_pass(InsertBiasQuantPass(self.records))
        optimizer.do_optimizer(self.graph, None)
        bias_dtype = (
            TensorProtoHelper(
                self.graph.get_node_by_name('layer2.0.sub_module.bias').proto
            )
            .get_data()
            .dtype
        )
        self.assertEqual(bias_dtype, 'int32')

    def test_quant_bias_int4(self):
        optimizer = GraphOptimizer()
        before_nodes = len(self.graph.nodes)
        with patch(
            'amct_pytorch.classic.graph_based.amct_pytorch.utils.quant_node.'
            'QuantOpInfo.get_dst_num_bits',
            return_value=4,
        ):
            optimizer.add_pass(InsertBiasQuantPass(self.records))
            optimizer.do_optimizer(self.graph, None)
            after_nodes = len(self.graph.nodes)
            self.assertEqual(after_nodes - before_nodes, 1)

    def test_replace_bias_quant_int4(self):
        optimizer = GraphOptimizer()
        before_nodes = len(self.graph.nodes)
        with patch(
            'amct_pytorch.classic.graph_based.amct_pytorch.utils.quant_node.'
            'QuantOpInfo.get_dst_num_bits',
            return_value=4,
        ):
            optimizer.add_pass(ReplaceBiasQuantPass(self.records))
            optimizer.do_optimizer(self.graph, None)
            after_nodes = len(self.graph.nodes)
            self.assertEqual(before_nodes - after_nodes, 1)

    def test_bias_exceed_int32(self):
        bias = np.array(
            [[0.0, 1.0, 2.0**31, -(2.0**31)], [0.0, 1.0, 2**31 - 1, -(2**31) - 1]],
            dtype=np.float32,
        )
        scale_w = np.array([0.1])
        scale_d = np.array(1.0)
        self.assertRaises(
            RuntimeError,
            InsertBiasQuantPass.quant_bias,
            bias,
            scale_w,
            scale_d,
            'conv1',
        )

    def _build_shared_bias_graph(self, temp_dir):
        """Build an onnx model whose bias initializer is shared by two MatMul
        layers, then parse it into an amct graph.

        Returns the graph, in which only ``layer_shared_a`` and
        ``layer_shared_b`` consume the shared ``shared_bias`` initializer.
        """
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        bias = numpy_helper.from_array(
            np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float32), name='shared_bias'
        )
        weight_a = numpy_helper.from_array(np.eye(4, dtype=np.float32), name='weight_a')
        weight_b = numpy_helper.from_array(np.eye(4, dtype=np.float32), name='weight_b')
        graph_input = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 4])
        out_a = helper.make_tensor_value_info('y_a', TensorProto.FLOAT, [1, 4])
        out_b = helper.make_tensor_value_info('y_b', TensorProto.FLOAT, [1, 4])
        nodes = [
            helper.make_node(
                'MatMul', ['x', 'weight_a'], ['mm_a_out'], name='layer_shared_a'
            ),
            helper.make_node(
                'Add', ['mm_a_out', 'shared_bias'], ['add_a_out'], name='AddA'
            ),
            helper.make_node(
                'MatMul', ['x', 'weight_b'], ['mm_b_out'], name='layer_shared_b'
            ),
            helper.make_node('Add', ['mm_b_out', 'shared_bias'], ['y_b'], name='AddB'),
            helper.make_node('Identity', ['add_a_out'], ['y_a'], name='OutA'),
        ]
        graph = helper.make_graph(
            nodes,
            'shared_bias_graph',
            [graph_input],
            [out_a, out_b],
            initializer=[weight_a, weight_b, bias],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
        onnx_file = os.path.join(temp_dir, 'shared_bias.onnx')
        onnx.save(model, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        return graph

    def test_shared_bias_not_polluted(self):
        """A shared bias initializer must keep FLOAT for the non-quantized
        consumer: the quantized layer works on a dedicated split copy.
        """
        temp_dir = os.path.join(CUR_DIR, 'test_bias_quant_shared')
        os.makedirs(temp_dir, exist_ok=True)
        try:
            graph = self._build_shared_bias_graph(temp_dir)
            records = {
                'layer_shared_a': {
                    'weight_scale': np.array([0.001] * 4),
                    'data_scale': np.array(1.0),
                    'dst_type': 'INT8',
                }
            }
            optimizer = GraphOptimizer()
            optimizer.add_pass(InsertBiasQuantPass(records))
            optimizer.do_optimizer(graph, None)

            from onnx import TensorProto

            bias_node = graph.get_node_by_name('shared_bias')
            self.assertEqual(bias_node.proto.data_type, TensorProto.FLOAT)
            kept_value = TensorProtoHelper(bias_node.proto).get_data()
            self.assertTrue(np.allclose(kept_value, [1.1, 2.2, 3.3, 4.4]))

            # The quantized layer consumes a dedicated INT32 copy instead.
            split_node = graph.get_node_by_name('shared_bias_layer_shared_a')
            self.assertEqual(split_node.proto.data_type, TensorProto.INT32)
            quant_value = TensorProtoHelper(split_node.proto).get_data()
            self.assertEqual(quant_value.dtype, np.int32)
            add_a = graph.get_node_by_name('AddA')
            bias_anchor = add_a.get_input_anchor(1)
            self.assertEqual(
                bias_anchor.get_peer_output_anchor().node.name,
                'shared_bias_layer_shared_a',
            )
            add_b = graph.get_node_by_name('AddB')
            bias_anchor = add_b.get_input_anchor(1)
            self.assertEqual(
                bias_anchor.get_peer_output_anchor().node.name, 'shared_bias'
            )
        finally:
            os.popen('rm -r ' + temp_dir)

    def _build_shared_conv_bias_graph(self, temp_dir):
        """Build an onnx model whose bias initializer is shared by two Conv
        layers (bias is the third input of Conv itself, not of a follower
        Add), then parse it into an amct graph.

        Returns the graph, in which only ``conv_shared_a`` and
        ``conv_shared_b`` consume the shared ``shared_conv_bias``
        initializer.
        """
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        bias = numpy_helper.from_array(
            np.array([1.1, 2.2], dtype=np.float32), name='shared_conv_bias'
        )
        weight_a = numpy_helper.from_array(
            np.full([2, 1, 1, 1], 1.0, dtype=np.float32), name='conv_weight_a'
        )
        weight_b = numpy_helper.from_array(
            np.full([2, 1, 1, 1], 1.0, dtype=np.float32), name='conv_weight_b'
        )
        graph_input = helper.make_tensor_value_info(
            'x', TensorProto.FLOAT, [1, 1, 2, 2]
        )
        out_a = helper.make_tensor_value_info('y_a', TensorProto.FLOAT, [1, 2, 2, 2])
        out_b = helper.make_tensor_value_info('y_b', TensorProto.FLOAT, [1, 2, 2, 2])
        nodes = [
            helper.make_node(
                'Conv',
                ['x', 'conv_weight_a', 'shared_conv_bias'],
                ['y_a'],
                name='conv_shared_a',
            ),
            helper.make_node(
                'Conv',
                ['x', 'conv_weight_b', 'shared_conv_bias'],
                ['y_b'],
                name='conv_shared_b',
            ),
        ]
        graph = helper.make_graph(
            nodes,
            'shared_conv_bias_graph',
            [graph_input],
            [out_a, out_b],
            initializer=[weight_a, weight_b, bias],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
        onnx_file = os.path.join(temp_dir, 'shared_conv_bias.onnx')
        onnx.save(model, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        return graph

    def test_shared_conv_bias_not_polluted(self):
        """A shared Conv bias initializer (re-linked via the bias_index of the
        conv node itself, not via a follower Add) must keep FLOAT for the
        non-quantized consumer: the quantized layer works on a dedicated
        split copy.
        """
        temp_dir = os.path.join(CUR_DIR, 'test_bias_quant_shared_conv')
        os.makedirs(temp_dir, exist_ok=True)
        try:
            graph = self._build_shared_conv_bias_graph(temp_dir)
            records = {
                'conv_shared_a': {
                    'weight_scale': np.array([0.001] * 2),
                    'data_scale': np.array(1.0),
                    'dst_type': 'INT8',
                }
            }
            optimizer = GraphOptimizer()
            optimizer.add_pass(InsertBiasQuantPass(records))
            optimizer.do_optimizer(graph, None)

            from onnx import TensorProto

            bias_node = graph.get_node_by_name('shared_conv_bias')
            self.assertEqual(bias_node.proto.data_type, TensorProto.FLOAT)
            kept_value = TensorProtoHelper(bias_node.proto).get_data()
            self.assertTrue(np.allclose(kept_value, [1.1, 2.2]))

            # The quantized conv consumes a dedicated INT32 copy instead.
            split_node = graph.get_node_by_name('shared_conv_bias_conv_shared_a')
            self.assertEqual(split_node.proto.data_type, TensorProto.INT32)
            quant_value = TensorProtoHelper(split_node.proto).get_data()
            self.assertEqual(quant_value.dtype, np.int32)
            self.assertTrue(np.allclose(quant_value, [1100, 2200]))

            conv_a = graph.get_node_by_name('conv_shared_a')
            bias_anchor = conv_a.get_input_anchor(2)
            self.assertEqual(
                bias_anchor.get_peer_output_anchor().node.name,
                'shared_conv_bias_conv_shared_a',
            )
            conv_b = graph.get_node_by_name('conv_shared_b')
            bias_anchor = conv_b.get_input_anchor(2)
            self.assertEqual(
                bias_anchor.get_peer_output_anchor().node.name, 'shared_conv_bias'
            )
        finally:
            os.popen('rm -r ' + temp_dir)

    def test_shared_bias_both_quantized_use_own_scale(self):
        """When both consumers are quantized, each one gets its own copy
        quantized with its own scale instead of overwriting each other.
        """
        temp_dir = os.path.join(CUR_DIR, 'test_bias_quant_shared_both')
        os.makedirs(temp_dir, exist_ok=True)
        try:
            graph = self._build_shared_bias_graph(temp_dir)
            records = {
                'layer_shared_a': {
                    'weight_scale': np.array([0.001] * 4),
                    'data_scale': np.array(1.0),
                    'dst_type': 'INT8',
                },
                'layer_shared_b': {
                    'weight_scale': np.array([0.01] * 4),
                    'data_scale': np.array(1.0),
                    'dst_type': 'INT8',
                },
            }
            optimizer = GraphOptimizer()
            optimizer.add_pass(InsertBiasQuantPass(records))
            optimizer.do_optimizer(graph, None)

            split_a = graph.get_node_by_name('shared_bias_layer_shared_a')
            quant_a = TensorProtoHelper(split_a.proto).get_data()
            self.assertTrue(np.allclose(quant_a, [1100, 2200, 3300, 4400]))
            # layer_shared_b quantizes the original node directly: after the
            # split for layer A its fan-out is 1, so no second copy is needed.
            bias_node = graph.get_node_by_name('shared_bias')
            quant_b = TensorProtoHelper(bias_node.proto).get_data()
            self.assertEqual(quant_b.dtype, np.int32)
            self.assertTrue(np.allclose(quant_b, [110, 220, 330, 440]))
        finally:
            os.popen('rm -r ' + temp_dir)

    def test_rnn_bias_quant_success(self):
        layer_name = 'lstm'
        records = {
            layer_name: {
                'data_scale': np.array(1.0, dtype=np.float32),
                'h_scale': np.array(1.0, dtype=np.float32),
                'weight_scale': np.array([1.0] * 4, dtype=np.float32),
                'weight_offset': np.array([0] * 4, dtype=np.int8),
                'recurrence_weight_scale': np.array([1.0] * 4, dtype=np.float32),
                'recurrence_weight_offset': np.array([0] * 4, dtype=np.int8),
            }
        }
        bias = np.random.random([1, 160]).astype(np.float32)
        passer = InsertBiasQuantPass(records)
        quant_bias = passer.bias_quant_rnn(bias, layer_name)
        self.assertEqual(quant_bias.dtype, np.int32)

    def build_a8w4_bias_case(self):
        """Build records/bias and run bias_quant_rnn for the A8W4 half-split case.

        Returns a dict with the scales, the raw bias, the split length and the
        quantized bias so callers can assert the per-half deq_scale.
        """
        layer_name = 'lstm_a8w4'
        # Use distinct non-unit scales so we can verify the correct scale pair
        # is applied to each half.
        scale_w = np.array([0.5] * 4, dtype=np.float32)  # weight scale (4 gates)
        scale_d = np.array(0.25, dtype=np.float32)  # input activation scale
        scale_r = np.array([0.125] * 4, dtype=np.float32)  # recurrence weight scale
        scale_h = np.array(0.0625, dtype=np.float32)  # hidden activation scale

        records = {
            layer_name: {
                'data_scale': scale_d,
                'h_scale': scale_h,
                'weight_scale': scale_w,
                'weight_offset': np.array([0] * 4, dtype=np.int8),
                'recurrence_weight_scale': scale_r,
                'recurrence_weight_offset': np.array([0] * 4, dtype=np.int8),
                # A8W4 marker — bias_quant_rnn does not inspect wts_type,
                # so deq_scale must be identical regardless of this value.
                'wts_type': 'INT4',
                'act_type': 'INT8',
            }
        }

        # bias shape [1, 2*bias_len]; split is at bias_len = 80
        bias_len = 80
        np.random.seed(7)
        bias = (np.random.random([1, bias_len * 2]) * 0.01).astype(np.float32)

        passer = InsertBiasQuantPass(records)
        quant_bias = passer.bias_quant_rnn(bias, layer_name)
        return {
            'scale_w': scale_w,
            'scale_d': scale_d,
            'scale_r': scale_r,
            'scale_h': scale_h,
            'bias': bias,
            'bias_len': bias_len,
            'quant_bias': quant_bias,
        }

    def test_rnn_bias_quant_a8w4_halfsplit_deqscale(self):
        """A8W4 bias: output is int32; first half deq_scale=scale_w*scale_d, second=scale_r*scale_h."""
        case = self.build_a8w4_bias_case()
        scale_w = case['scale_w']
        scale_d = case['scale_d']
        scale_r = case['scale_r']
        scale_h = case['scale_h']
        bias = case['bias']
        bias_len = case['bias_len']
        quant_bias = case['quant_bias']

        # 1. Output dtype must be int32 regardless of weight bit-width
        self.assertEqual(
            quant_bias.dtype, np.int32, 'A8W4 bias quant must produce int32 output'
        )

        # 2. Output length equals input length
        self.assertEqual(
            quant_bias.size,
            bias_len * 2,
            'quantized bias length must match original bias length',
        )

        # 3. Verify first half: deq_scale = scale_w * scale_d (broadcast to bias_len)
        flat_bias = bias.flatten()
        first_half = flat_bias[:bias_len]
        # scale_w has 4 gates; broadcast to bias_len elements
        sw = np.repeat(scale_w, bias_len // scale_w.size)
        expected_first = np.round(first_half / np.multiply(sw, scale_d)).astype(
            np.int32
        )
        np.testing.assert_array_equal(
            quant_bias[:bias_len],
            expected_first,
            err_msg='First-half deq_scale must use scale_w * scale_d (A8W4)',
        )

        # 4. Verify second half: deq_scale = scale_r * scale_h (broadcast to bias_len)
        second_half = flat_bias[bias_len:]
        sr = np.repeat(scale_r, bias_len // scale_r.size)
        expected_second = np.round(second_half / np.multiply(sr, scale_h)).astype(
            np.int32
        )
        np.testing.assert_array_equal(
            quant_bias[bias_len:],
            expected_second,
            err_msg='Second-half deq_scale must use scale_r * scale_h (A8W4)',
        )
