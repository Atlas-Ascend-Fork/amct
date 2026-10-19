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
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

import copy
import logging
import sys
import math
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from mock_torch_npu import mock_npu_dynamic_quant
from mock_torch_npu import mock_npu_quant_matmul
from mock_torch_npu import mock_npu_quantize

from amct_pytorch import convert
from amct_pytorch import quantize
from amct_pytorch.algorithms import AlgorithmRegistry
from amct_pytorch.classic.deploy_op.npu_hif8_quantization_linear import NpuHIF8Linear


LOGGER = logging.getLogger(__name__)


class FP8Linear(nn.Module):
    def __init__(self, in_features, out_features, block_size=None, has_bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if has_bias else None
        self.block_size = block_size
        scale_shape = self._get_scale_shape(out_features, in_features, block_size)
        self.register_buffer('weight_scale_inv', torch.ones(scale_shape))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @staticmethod
    def _get_scale_shape(out_features, in_features, block_size):
        if block_size is None:
            return (out_features, in_features)
        block_h, block_w = block_size
        return (math.ceil(out_features / block_h), math.ceil(in_features / block_w))

    def forward(self, x):
        weight_scale_inv = self.weight_scale_inv
        if self.block_size is not None:
            block_h, block_w = self.block_size
            weight_scale_inv = torch.repeat_interleave(weight_scale_inv, block_h, dim=0)
            weight_scale_inv = torch.repeat_interleave(weight_scale_inv, block_w, dim=1)
            weight_scale_inv = weight_scale_inv[
                : self.weight.shape[0], : self.weight.shape[1]
            ]
        weight = self.weight.to(torch.float32) / weight_scale_inv.to(torch.float32)
        return torch.nn.functional.linear(x, weight.to(x.dtype), self.bias)


AlgorithmRegistry.register('cast', None, FP8Linear, NpuHIF8Linear)

torch.manual_seed(0)

logger = logging.getLogger(__name__)


class FP8Model(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, block_size=None):
        super().__init__()
        # 第一层 FP8 Linear
        self.layer1 = FP8Linear(in_dim, hidden_dim, has_bias=True)
        # 第二层 FP8 Linear
        if block_size is None:
            self.layer2 = FP8Linear(hidden_dim, out_dim, has_bias=False)
        else:
            block_size = (block_size, block_size)
            self.layer2 = FP8Linear(
                hidden_dim, out_dim, block_size=block_size, has_bias=False
            )

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)  # 中间加个激活函数演示
        x = self.layer2(x)
        return x


class TestFP8HIF8(unittest.TestCase):
    '''ST FOR FP8HIF8 ALGORITHM'''

    @classmethod
    def setUpClass(cls):
        input_dim, hidden_dim, output_dim, block_size = 128, 256, 64, 10
        batch = 8
        cls.test_model = FP8Model(input_dim, hidden_dim, output_dim).to(torch.bfloat16)
        cls.test_block_model = FP8Model(
            input_dim, hidden_dim, output_dim, block_size
        ).to(torch.bfloat16)
        cls.test_inputs = torch.randn(batch, input_dim).to(torch.bfloat16)
        cls.ori_out = cls.test_model(cls.test_inputs)
        LOGGER.info('TestFP8HIF8 START!')

    @classmethod
    def tearDownClass(cls):
        LOGGER.info('TestFP8HIF8 END!')

    def setUp(self):
        mock_torch_npu = MagicMock()
        sys.modules['torch_npu'] = mock_torch_npu

    def tearDown(self):
        del sys.modules['torch_npu']

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_dynamic_quant', wraps=mock_npu_dynamic_quant)
    @patch(
        'amct_pytorch.classic.deploy_op.npu_hif8_quantization_linear.check_parameters_in_schema',
        MagicMock(return_value=True),
    )
    def test_fp8_hif8_success(self, mock_1, mock_2, mock_3):
        model = copy.deepcopy(self.test_model)
        quantize(model)
        LOGGER.info("%s", model)
        self.assertEqual(
            list(model.state_dict().keys()), list(self.test_model.state_dict().keys())
        )
        self.assertEqual(type(model.layer1).__name__, 'FP8Linear')
        self.assertEqual(type(model.layer2).__name__, 'FP8Linear')
        convert(model)
        self.assertEqual(type(model.layer1).__name__, 'NpuHIF8Linear')
        self.assertEqual(type(model.layer2).__name__, 'NpuHIF8Linear')
        LOGGER.info("%s", model)
        quant_out = model(self.test_inputs)
        self.assertIsNotNone(quant_out)

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_dynamic_quant', wraps=mock_npu_dynamic_quant)
    @patch(
        'amct_pytorch.classic.deploy_op.npu_hif8_quantization_linear.check_parameters_in_schema',
        MagicMock(return_value=True),
    )
    def test_block_fp8_hif8_success(self, mock_1, mock_2, mock_3):
        model = copy.deepcopy(self.test_block_model)
        quantize(model)
        LOGGER.info("%s", model)
        self.assertEqual(
            list(model.state_dict().keys()), list(self.test_model.state_dict().keys())
        )
        self.assertEqual(type(model.layer1).__name__, 'FP8Linear')
        self.assertEqual(type(model.layer2).__name__, 'FP8Linear')
        convert(model)
        self.assertEqual(type(model.layer1).__name__, 'NpuHIF8Linear')
        self.assertEqual(type(model.layer2).__name__, 'NpuHIF8Linear')
        LOGGER.info("%s", model)
        quant_out = model(self.test_inputs)
        self.assertIsNotNone(quant_out)


class TestHIF8DeployDeqScale(unittest.TestCase):
    '''ST FOR THE DEPLOY-SIDE DEQUANT SCALE OF FP8HIF8 (issue #222)

    NpuHIF8Linear._init_weight_quant recomputes the HiF8 weight scale from the
    dequantized FP8 weight instead of reusing the fake-quant scale, so a bug
    there stays invisible in the fake-quant accuracy results. These cases pin
    the exported deq_scale to max(abs(weight)) / 16 per output channel, which
    only holds when the scale covers the negative absolute extreme too.
    '''

    HIF8_SCOPE = 16.0

    # Every value is exactly representable in bfloat16, so the expected scale
    # stays free of rounding noise.
    LAYER1_WEIGHT = torch.tensor(
        [
            [-8.0, 0.5, 1.0, 0.25],  # negative extreme dominates
            [1.0, 2.0, 0.5, -0.125],  # positive extreme dominates
            [-4.0, -2.0, -1.0, -0.5],  # all negative, no positive max at all
            [0.5, -0.25, 0.125, 0.0625],  # mixed, |min| > max
        ]
    )
    LAYER2_WEIGHT = torch.tensor(
        [
            [-16.0, 1.0, 0.5, 0.25],
            [0.5, 0.25, -0.125, 0.0625],
        ]
    )

    def setUp(self):
        mock_torch_npu = MagicMock()
        sys.modules['torch_npu'] = mock_torch_npu

    def tearDown(self):
        del sys.modules['torch_npu']

    def _build_model(self, block_size=None, weight_scale_inv=1.0):
        model = FP8Model(4, 4, 2, block_size=block_size).to(torch.bfloat16)
        with torch.no_grad():
            for layer, weight in (
                (model.layer1, self.LAYER1_WEIGHT),
                (model.layer2, self.LAYER2_WEIGHT),
            ):
                layer.weight.copy_(weight.to(torch.bfloat16))
                layer.weight_scale_inv.fill_(weight_scale_inv)
        return model

    def _expected_deq_scale(self, weight_scale_inv=1.0):
        scales = []
        for weight in (self.LAYER1_WEIGHT, self.LAYER2_WEIGHT):
            abs_max = weight.abs().max(dim=1).values
            scales.append(abs_max / weight_scale_inv / self.HIF8_SCOPE)
        return torch.cat(scales)

    def _convert(self, model):
        quantize(model)
        convert(model)
        for layer in (model.layer1, model.layer2):
            self.assertEqual(type(layer).__name__, 'NpuHIF8Linear')
        return model

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_dynamic_quant', wraps=mock_npu_dynamic_quant)
    @patch(
        'amct_pytorch.classic.deploy_op.npu_hif8_quantization_linear.check_parameters_in_schema',
        MagicMock(return_value=True),
    )
    def test_deq_scale_covers_negative_extreme(self, mock_1, mock_2, mock_3):
        '''per-channel (block_size=None) export path'''
        model = self._convert(self._build_model())
        expected = self._expected_deq_scale()
        torch.testing.assert_close(model.layer1.deq_scale, expected[:4])
        torch.testing.assert_close(model.layer2.deq_scale, expected[4:])
        self.assertTrue(torch.all(model.layer1.deq_scale > 0))
        self.assertTrue(torch.all(model.layer2.deq_scale > 0))
        # The all-negative row must not degenerate into the unscaled 1.0 fallback.
        self.assertNotIn(1.0, model.layer1.deq_scale.tolist())

        quant_out = model(torch.randn(2, 4).to(torch.bfloat16))
        self.assertIsNotNone(quant_out)
        self.assertTrue(torch.all(torch.isfinite(quant_out)))

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_dynamic_quant', wraps=mock_npu_dynamic_quant)
    @patch(
        'amct_pytorch.classic.deploy_op.npu_hif8_quantization_linear.check_parameters_in_schema',
        MagicMock(return_value=True),
    )
    def test_block_deq_scale_covers_negative_extreme(self, mock_1, mock_2, mock_3):
        '''block quantize export path: the FP8 scale must be divided out first'''
        weight_scale_inv = 2.0
        model = self._convert(
            self._build_model(block_size=2, weight_scale_inv=weight_scale_inv)
        )
        expected = self._expected_deq_scale(weight_scale_inv)
        torch.testing.assert_close(model.layer1.deq_scale, expected[:4])
        torch.testing.assert_close(model.layer2.deq_scale, expected[4:])

        quant_out = model(torch.randn(2, 4).to(torch.bfloat16))
        self.assertIsNotNone(quant_out)
        self.assertTrue(torch.all(torch.isfinite(quant_out)))
