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
