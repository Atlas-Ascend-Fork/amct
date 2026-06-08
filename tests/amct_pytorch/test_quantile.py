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
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from mock_torch_npu import (
    mock_npu,
    mock_npu_dtype_cast,
    mock_npu_dynamic_quant,
    mock_npu_quant_matmul,
    mock_npu_quantize,
    mock_npu_weight_quant_batchmatmul,
)
from utils import TestModel

from amct_pytorch import HIFP8_QUANTILE_CFG, convert, quantize
from amct_pytorch.algorithms import AlgorithmRegistry

logger = logging.getLogger(__name__)

QUANTILEQUANT = 'QuantileQuant'


class TestQuantileQuant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = TestModel().to(torch.bfloat16)
        cls.inputs = torch.randn(64, 64).to(torch.bfloat16)
        cls.ori_out = cls.model(cls.inputs)
        logger.info('TestQuantile START!')

    @classmethod
    def tearDownClass(cls):
        logger.info('TestQuantile END!')

    def setUp(self):
        mock_torch_npu = MagicMock()
        sys.modules['torch_npu'] = mock_torch_npu

    def tearDown(self):
        del sys.modules['torch_npu']
    
    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_dtype_cast', wraps=mock_npu_dtype_cast)
    @patch(
        'amct_pytorch.classic.deploy_op.weight_npu_quant_module.check_parameters_in_schema',
        MagicMock(return_value=True),
    )
    def test_quantile_weight_only(self, mock_1, mock_2, mock_3, mock_4):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'hifloat8',
                    'symmetric': True,
                    'strategy': 'channel',
                },
            },
            'algorithm': {'quantile'}
        }
        
        model = copy.deepcopy(self.model).to(torch.bfloat16)
        quantize(model, cfg)

        model(self.inputs)
        self.assertEqual(model.linear1.scale_w.shape[0], 64)
        self.assertIsNotNone(model.linear1.scale_w)
        self.assertIsNone(model.linear2.offset_w)
        self.assertEqual(type(model.linear1).__name__, QUANTILEQUANT)
        self.assertEqual(type(model.linear2).__name__, QUANTILEQUANT)
        self.assertEqual(type(model.linear3).__name__, QUANTILEQUANT)
        torch.Tensor.npu = mock_npu
        convert(model)
        self.assertEqual(type(model.linear1).__name__, 'NpuWeightQuantizedLinear')
        self.assertEqual(type(model.linear2).__name__, 'NpuWeightQuantizedLinear')


    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch(
        'amct_pytorch.classic.deploy_op.npu_quantization_linear.check_parameters_in_schema',
        MagicMock(return_value=True),
    )
    def test_quantile_full_quant_with_convert(self, mock_1, mock_2, mock_3):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'hifloat8',
                    'symmetric': True,
                    'strategy': 'channel',
                },
                'inputs': {
                    'type': 'hifloat8',
                    'symmetric': True,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {'quantile'}
        }
        
        model = copy.deepcopy(self.model).to(torch.bfloat16)
        quantize(model, cfg)

        model(self.inputs)
        self.assertIsNotNone(model.linear1.scale_w)
        self.assertIsNotNone(model.linear1.scale_d)
        self.assertEqual(type(model.linear1).__name__, QUANTILEQUANT)
        torch.Tensor.npu = mock_npu
        convert(model)
        self.assertEqual(type(model.linear1).__name__, 'NpuQuantizationLinear')

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_dtype_cast', wraps=mock_npu_dtype_cast)
    @patch(
        'amct_pytorch.classic.deploy_op.weight_npu_quant_module.check_parameters_in_schema',
        MagicMock(return_value=True),
    )
    def test_quantile_with_bias(self, mock_1, mock_2, mock_3, mock_4):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'hifloat8',
                    'symmetric': True,
                    'strategy': 'channel',
                },
                'bias': {
                    'type': 'hifloat8',
                    'symmetric': True,
                },
            },
            'algorithm': {'quantile'}
        }
        
        model = copy.deepcopy(self.model).to(torch.bfloat16)
        quantize(model, cfg)

        model(self.inputs)
        self.assertIsNotNone(model.linear1.scale_w)
        self.assertEqual(type(model.linear1).__name__, QUANTILEQUANT)
        torch.Tensor.npu = mock_npu
        convert(model)
        self.assertEqual(type(model.linear1).__name__, 'NpuWeightQuantizedLinear')

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch(
        'amct_pytorch.classic.deploy_op.npu_quantization_linear.check_parameters_in_schema',
        MagicMock(return_value=True),
    )
    def test_quantile_with_token(self, mock_1, mock_2, mock_3):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'hifloat8',
                    'symmetric': True,
                    'strategy': 'tensor',
                },
                'inputs': {
                    'type': 'hifloat8',
                    'symmetric': True,
                    'strategy': 'token',
                },
            },
            'algorithm': {'quantile'},
            'skip_layers': {'lm_head'}
        }
        
        model = copy.deepcopy(self.model).to(torch.bfloat16)
        quantize(model, cfg)

        model(self.inputs)
        self.assertIsNotNone(model.linear1.scale_w)
        self.assertEqual(type(model.linear1).__name__, QUANTILEQUANT)
        torch.Tensor.npu = mock_npu
        convert(model)
        self.assertEqual(type(model.linear1).__name__, 'NpuQuantizationLinear')

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_dtype_cast', wraps=mock_npu_dtype_cast)
    @patch(
        'amct_pytorch.classic.deploy_op.weight_npu_quant_module.check_parameters_in_schema',
        MagicMock(return_value=True),
    )
    def test_quantile_calibration(self, mock_1, mock_2, mock_3, mock_4):
        cfg = {
            'batch_num': 4,
            'quant_cfg': {
                'weights': {
                    'type': 'hifloat8',
                    'symmetric': True,
                    'strategy': 'channel',
                },
            },
            'algorithm': {'quantile'}
        }
        
        model = copy.deepcopy(self.model).to(torch.bfloat16)
        quantize(model, cfg)

        for _ in range(4):
            inputs = torch.randn(64, 64).to(torch.bfloat16)
            model(inputs)
        
        self.assertIsNotNone(model.linear1.scale_w)
        self.assertEqual(type(model.linear1).__name__, QUANTILEQUANT)
        torch.Tensor.npu = mock_npu
        convert(model)
        self.assertEqual(type(model.linear1).__name__, 'NpuWeightQuantizedLinear')

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_dtype_cast', wraps=mock_npu_dtype_cast)
    @patch(
        'amct_pytorch.classic.deploy_op.npu_quantization_linear.check_parameters_in_schema',
        MagicMock(return_value=True),
    )
    def test_quantile_with_config_object(self, mock_1, mock_2, mock_3, mock_4):
        model = copy.deepcopy(self.model).to(torch.bfloat16)
        quantize(model, HIFP8_QUANTILE_CFG)

        for _ in range(4):
            inputs = torch.randn(64, 64).to(torch.bfloat16)
            model(inputs)

        self.assertIsNotNone(model.linear1.scale_w)
        self.assertIsNotNone(model.linear1.scale_d)
        self.assertEqual(type(model.linear1).__name__, QUANTILEQUANT)
        torch.Tensor.npu = mock_npu
        convert(model)
        self.assertEqual(type(model.linear1).__name__, 'NpuQuantizationLinear')

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_dynamic_quant', wraps=mock_npu_dynamic_quant)
    @patch('torch_npu.npu_dtype_cast', wraps=mock_npu_dtype_cast)
    @patch(
        'amct_pytorch.classic.deploy_op.npu_quantization_linear.check_parameters_in_schema',
        MagicMock(return_value=True),
    )
    def test_quantile_with_dynamic_token(self, mock_1, mock_2, mock_3, mock_4, mock_5):
        model = copy.deepcopy(self.model).to(torch.bfloat16)
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'hifloat8',
                    'symmetric': True,
                    'strategy': 'tensor',
                },
                'inputs': {
                    'type': 'hifloat8',
                    'symmetric': True,
                    'strategy': 'token',
                    'dynamic': True,
                },
            },
            'algorithm': {'quantile'},
            'skip_layers': {'lm_head'}
        }
        quantize(model, cfg)

        for _ in range(4):
            inputs = torch.randn(64, 64).to(torch.bfloat16)
            model(inputs)
        
        self.assertIsNotNone(model.linear1.scale_w)
        self.assertIsNone(model.linear1.scale_d)
        self.assertEqual(type(model.linear1).__name__, QUANTILEQUANT)
        torch.Tensor.npu = mock_npu
        convert(model)
        self.assertEqual(type(model.linear1).__name__, 'NpuQuantizationLinear')

if __name__ == '__main__':
    unittest.main()

