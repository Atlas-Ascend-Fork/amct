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
import copy
import logging
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from mock_torch_npu import (
    mock_npu,
    mock_npu_dtype_cast,
    mock_npu_quant_matmul,
    mock_npu_quantize,
    mock_npu_weight_quant_batchmatmul,
)
from utils import TestModel, TestModelBias

from amct_pytorch import convert, quantize

NPU_HIF8_CAST_LINEAR = 'NpuHIF8CastLinear'
HIF8_CAST_QUANT = 'HIF8CastQuant'
HIFLOAT8 = 'hifloat8'
CAST_ALGO = 'cast'

torch.manual_seed(0)

logger = logging.getLogger(__name__)


class TestCast(unittest.TestCase):
    '''
    ST FOR CAST ALGORITHM
    '''
    @classmethod
    def setUpClass(cls):
        cls.test_model = TestModel().to(torch.bfloat16)
        cls.inputs = torch.randn(64, 64).to(torch.bfloat16)
        cls.ori_out = cls.test_model(cls.inputs)
        logger.info('TestCast START!')

    @classmethod
    def tearDownClass(cls):
        logger.info('TestCast END!')

    def setUp(self):
        mock_torch_npu = MagicMock()
        sys.modules['torch_npu'] = mock_torch_npu
 
    def tearDown(self):
        del sys.modules['torch_npu']

    @patch('torch_npu.npu_dtype_cast', wraps=mock_npu_dtype_cast)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    def test_hif8_weights_tensor_sym_cast_success(self, mock_1, mock_2, mock_3, mock_4):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': HIFLOAT8,
                    'symmetric': True,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {CAST_ALGO},
        }
        torch.Tensor.npu = mock_npu
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs)
        self.assertEqual(type(model.linear3).__name__, HIF8_CAST_QUANT)
        convert(model)
        self.assertEqual(type(model.linear3).__name__, NPU_HIF8_CAST_LINEAR)
        quant_out = model(self.inputs.npu())

    @patch('torch_npu.npu_dtype_cast', wraps=mock_npu_dtype_cast)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    def test_hif8_weights_channel_sym_cast_success(self, mock_1, mock_2, mock_3, mock_4):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': HIFLOAT8,
                    'symmetric': True,
                    'strategy': 'channel',
                },
            },
            'algorithm': {CAST_ALGO},
        }
        torch.Tensor.npu = mock_npu
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs)
        self.assertEqual(type(model.linear3).__name__, HIF8_CAST_QUANT)
        convert(model)
        self.assertEqual(type(model.linear3).__name__, NPU_HIF8_CAST_LINEAR)
        quant_out = model(self.inputs.npu())

    @patch('torch_npu.npu_dtype_cast', wraps=mock_npu_dtype_cast)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    def test_hif8_tensor_tensor_sym_cast_success(self, mock_1, mock_2, mock_3, mock_4):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': HIFLOAT8,
                    'symmetric': True,
                    'strategy': 'tensor',
                },
                'inputs': {
                    'type': HIFLOAT8,
                    'symmetric': True,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {CAST_ALGO},
        }
        torch.Tensor.npu = mock_npu
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs.to(torch.bfloat16))
        self.assertEqual(type(model.linear1).__name__, HIF8_CAST_QUANT)
        self.assertEqual(type(model.linear2).__name__, HIF8_CAST_QUANT)
        self.assertEqual(type(model.linear3).__name__, HIF8_CAST_QUANT)
        convert(model)
        self.assertEqual(type(model.linear1).__name__, NPU_HIF8_CAST_LINEAR)
        self.assertEqual(type(model.linear2).__name__, NPU_HIF8_CAST_LINEAR)
        self.assertEqual(type(model.linear3).__name__, NPU_HIF8_CAST_LINEAR)
        quant_out = model(self.inputs.npu())

    @patch('torch_npu.npu_dtype_cast', wraps=mock_npu_dtype_cast)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    def test_hif8_channel_tensor_sym_cast_success(self, mock_1, mock_2, mock_3, mock_4):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': HIFLOAT8,
                    'symmetric': True,
                    'strategy': 'channel',
                },
                'inputs': {
                    'type': HIFLOAT8,
                    'symmetric': True,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {CAST_ALGO},
        }
        torch.Tensor.npu = mock_npu
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs.to(torch.bfloat16))
        self.assertEqual(type(model.linear1).__name__, HIF8_CAST_QUANT)
        self.assertEqual(type(model.linear2).__name__, HIF8_CAST_QUANT)
        self.assertEqual(type(model.linear3).__name__, HIF8_CAST_QUANT)
        convert(model)
        self.assertEqual(type(model.linear1).__name__, NPU_HIF8_CAST_LINEAR)
        self.assertEqual(type(model.linear2).__name__, NPU_HIF8_CAST_LINEAR)
        self.assertEqual(type(model.linear3).__name__, NPU_HIF8_CAST_LINEAR)
        quant_out = model(self.inputs.npu())
