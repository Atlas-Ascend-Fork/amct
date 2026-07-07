# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

import numpy as np
import torch
import torch.nn as nn
from mock_torch_npu import (
    mock_npu_convert_weight_to_int4pack,
    mock_npu_quant_matmul,
    mock_npu_quantize,
    mock_npu_weight_quant_batchmatmul,
    mock_npu
)

from amct_pytorch import (
    INT4_AWQ_WEIGHT_QUANT_CFG,
    INT4_GPTQ_WEIGHT_QUANT_CFG,
    INT8_MINMAX_WEIGHT_QUANT_CFG,
    INT8_SMOOTHQUANT_CFG,
    convert,
    quantize,
)
from amct_pytorch.quantize_op.base_quant_module import BaseQuantizeModule
from amct_pytorch.common.config import parse_config

torch.manual_seed(0)

logger = logging.getLogger(__name__)


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64, 64)
        self.linear2 = nn.Linear(64, 32, bias=True)
        self.linear3 = nn.Linear(32, 1)

    def forward(self, inputs):
        x = self.linear1(inputs)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


class TestDefQuantize(unittest.TestCase):
    '''
    ST FOR QUANTIZATION
    '''
    @classmethod
    def setUpClass(cls):
        cls.test_model = TestModel().to(torch.bfloat16)
        cls.inputs = torch.randn(64, 64).to(torch.bfloat16)
        cls.ori_out = cls.test_model(cls.inputs).to(torch.float32).detach().to('cpu').numpy().astype(np.float32)
        logger.info('TestDefQuantize START!')

    @classmethod
    def tearDownClass(cls):
        logger.info('TestDefQuantize END!')

    def setUp(self):
        mock_torch_npu = MagicMock()
        sys.modules['torch_npu'] = mock_torch_npu

    def tearDown(self):
        del sys.modules['torch_npu']

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch(
        'amct_pytorch.classic.deploy_op.weight_npu_quant_module.check_parameters_in_schema',
        MagicMock(return_value=True),
    )
    def test_default_success(self, mock_1, mock_2, mock_3, mock_4):
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model)
        model(self.inputs)
        self.assertIsNotNone(model.linear1.scale_w)
        self.assertIsNotNone(model.linear2.scale_w)
        self.assertIsNotNone(model.linear3.scale_w)
        self.assertEqual(type(model.linear3).__name__, 'MinMaxQuant')
        torch.Tensor.npu = mock_npu
        convert(model)
        self.assertEqual(model.linear3.quantized_weight.dtype, torch.int8)
        quant_out = model(self.inputs.npu())

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch(
        'amct_pytorch.classic.deploy_op.weight_npu_quant_module.check_parameters_in_schema',
        MagicMock(return_value=True),
    )
    def test_int4_awq_quant_success(self, mock_1, mock_2, mock_3, mock_4):
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, INT4_AWQ_WEIGHT_QUANT_CFG)
        model(self.inputs)
        self.assertIsNotNone(model.linear1.scale_w)
        self.assertIsNotNone(model.linear2.scale_w)
        self.assertEqual(type(model.linear2).__name__, 'LinearAWQuant')
        self.assertEqual(type(model.linear3).__name__, 'Linear')
        torch.Tensor.npu = mock_npu
        convert(model)
        self.assertEqual(model.linear2.quantized_weight.dtype, torch.int32)
        quant_out = model(self.inputs.npu())

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch(
        'amct_pytorch.classic.deploy_op.weight_npu_quant_module.check_parameters_in_schema',
        MagicMock(return_value=True),
    )
    def test_int4_gptq_success(self, mock_1, mock_2, mock_3, mock_4):
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, INT4_GPTQ_WEIGHT_QUANT_CFG)
        model(self.inputs)
        self.assertIsNotNone(model.linear1.scale_w)
        self.assertIsNotNone(model.linear2.scale_w)
        self.assertEqual(type(model.linear2).__name__, 'GPTQuant')
        self.assertEqual(type(model.linear3).__name__, 'Linear')
        torch.Tensor.npu = mock_npu
        convert(model)
        self.assertEqual(model.linear2.quantized_weight.dtype, torch.int32)
        quant_out = model(self.inputs.npu())


    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch(
        'amct_pytorch.classic.deploy_op.weight_npu_quant_module.check_parameters_in_schema',
        MagicMock(return_value=True),
    )
    def test_int8_minmax_success(self, mock_1, mock_2, mock_3, mock_4):
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, INT8_MINMAX_WEIGHT_QUANT_CFG)
        model(self.inputs)
        self.assertIsNotNone(model.linear1.scale_w)
        self.assertIsNotNone(model.linear2.scale_w)
        self.assertEqual(type(model.linear3).__name__, 'MinMaxQuant')
        torch.Tensor.npu = mock_npu
        convert(model)
        self.assertEqual(model.linear3.quantized_weight.dtype, torch.int8)
        quant_out = model(self.inputs.npu())

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch(
        'amct_pytorch.classic.deploy_op.npu_quantization_linear.check_parameters_in_schema',
        MagicMock(return_value=True),
    )
    def test_int8_smooth_success(self, mock_1, mock_2, mock_3, mock_4):
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, INT8_SMOOTHQUANT_CFG)
        model(self.inputs)
        self.assertIsNotNone(model.linear1.scale_w)
        self.assertIsNotNone(model.linear2.scale_w)
        self.assertIsNotNone(model.linear1.scale_d)
        self.assertIsNotNone(model.linear2.scale_d)
        self.assertEqual(type(model.linear3).__name__, 'SmoothQuant')
        torch.Tensor.npu = mock_npu
        convert(model)
        self.assertEqual(model.linear3.quantized_weight.dtype, torch.int8)
        quant_out = model(self.inputs.npu())

