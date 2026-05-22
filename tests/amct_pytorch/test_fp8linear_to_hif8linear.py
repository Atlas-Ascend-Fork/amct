import copy
import logging
import math
import sys
import unittest

from unittest.mock import MagicMock
from unittest.mock import patch

import torch
import torch.nn as nn

from mock_torch_npu import mock_npu_dynamic_quant
from mock_torch_npu import mock_npu_quant_matmul
from mock_torch_npu import mock_npu_quantize

from amct_pytorch import convert
from amct_pytorch import quantize


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
            weight_scale_inv = weight_scale_inv[:self.weight.shape[0], :self.weight.shape[1]]
        weight = self.weight.to(torch.float32) / weight_scale_inv.to(torch.float32)
        return torch.nn.functional.linear(x, weight.to(x.dtype), self.bias)

torch.manual_seed(0)


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
            self.layer2 = FP8Linear(hidden_dim, out_dim, block_size=block_size, has_bias=False)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x) # 中间加个激活函数演示
        x = self.layer2(x)
        return x


class TestFP8HIF8(unittest.TestCase):
    '''    ST FOR FP8HIF8 ALGORITHM    '''
    @classmethod
    def setUpClass(cls):
        input_dim, hidden_dim, output_dim, block_size = 128, 256, 64, 10
        batch = 8
        cls.test_model = FP8Model(input_dim, hidden_dim, output_dim).to(torch.bfloat16)
        cls.test_block_model = FP8Model(input_dim, hidden_dim, output_dim,
                                        block_size).to(torch.bfloat16)
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
    @patch('amct_pytorch.deploy_op.npu_hif8_quantization_linear.check_parameters_in_schema',
           MagicMock(return_value=True))
    def test_fp8_hif8_success(self, mock_1, mock_2, mock_3):
        model = copy.deepcopy(self.test_model)
        quantize(model)
        LOGGER.info("%s", model)
        self.assertEqual(list(model.state_dict().keys()), list(self.test_model.state_dict().keys()))
        self.assertEqual(type(model.layer1).__name__, 'FP8Linear')
        self.assertEqual(type(model.layer2).__name__, 'FP8Linear')
        convert(model)
        self.assertEqual(type(model.layer1).__name__, 'NpuHIF8Linear')
        self.assertEqual(type(model.layer2).__name__, 'NpuHIF8Linear')
        LOGGER.info("%s", model)
        quant_out = model(self.test_inputs)

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_dynamic_quant', wraps=mock_npu_dynamic_quant)
    @patch('amct_pytorch.deploy_op.npu_hif8_quantization_linear.check_parameters_in_schema',
           MagicMock(return_value=True))
    def test_block_fp8_hif8_success(self, mock_1, mock_2, mock_3):
        model = copy.deepcopy(self.test_block_model)
        quantize(model)
        LOGGER.info("%s", model)
        self.assertEqual(list(model.state_dict().keys()), list(self.test_model.state_dict().keys()))
        self.assertEqual(type(model.layer1).__name__, 'FP8Linear')
        self.assertEqual(type(model.layer2).__name__, 'FP8Linear')
        convert(model)
        self.assertEqual(type(model.layer1).__name__, 'NpuHIF8Linear')
        self.assertEqual(type(model.layer2).__name__, 'NpuHIF8Linear')
        LOGGER.info("%s", model)
        quant_out = model(self.test_inputs)
