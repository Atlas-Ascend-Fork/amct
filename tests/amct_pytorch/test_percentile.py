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
import unittest
import sys
import torch
import torch.nn as nn
from utils import TestModel, TestModelBias
from mock_torch_npu import *
from unittest.mock import MagicMock
from unittest.mock import patch

from amct_pytorch import quantize, convert
from amct_pytorch import HIFP8_PERCENTILE_WEIGHT_ONLY_TENSOR_CFG
from amct_pytorch import HIFP8_PERCENTILE_WEIGHT_ONLY_CHANNEL_CFG
from amct_pytorch import HIFP8_PERCENTILE_ACT_TENSOR_WEIGHT_TENSOR_CFG
from amct_pytorch import HIFP8_PERCENTILE_ACT_TENSOR_WEIGHT_CHANNEL_CFG
from amct_pytorch import HIFP8_PERCENTILE_ACT_TOKEN_STATIC_WEIGHT_TENSOR_CFG
from amct_pytorch import HIFP8_PERCENTILE_ACT_TOKEN_STATIC_WEIGHT_CHANNEL_CFG

torch.manual_seed(0)

class TestPercentile(unittest.TestCase):
    '''
    ST FOR PERCENTILE ALGORITHM
    '''
    @classmethod
    def setUpClass(cls):
        cls.test_model = TestModel().to(torch.bfloat16)
        cls.inputs = torch.randn(64, 64).to(torch.bfloat16)
        cls.ori_out = cls.test_model(cls.inputs)
        print('TestPercentile START!')

    @classmethod
    def tearDownClass(cls):
        print('TestPercentile END!')

    def setUp(self):
        mock_torch_npu = MagicMock()
        sys.modules['torch_npu'] = mock_torch_npu
 
    def tearDown(self):
        del sys.modules['torch_npu']

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch('amct_pytorch.deploy_op.weight_npu_quant_module.check_parameters_in_schema', MagicMock(return_value=True))
    def test_hifp8_percentile_weight_only_tensor_success(self, mock_1, mock_2, mock_3, mock_4):
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, HIFP8_PERCENTILE_WEIGHT_ONLY_TENSOR_CFG)
        model(self.inputs)
        self.assertIsNotNone(model.linear1.scale_w)
        self.assertIsNotNone(model.linear2.scale_w)
        self.assertIsNotNone(model.linear3.scale_w)
        self.assertEqual(type(model.linear3).__name__, 'PercentileQuant')
        torch.Tensor.npu = mock_npu
        convert(model)
        quant_out = model(self.inputs.npu())

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch('amct_pytorch.deploy_op.weight_npu_quant_module.check_parameters_in_schema', MagicMock(return_value=True))
    def test_hifp8_percentile_weight_only_channel_success(self, mock_1, mock_2, mock_3, mock_4):
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, HIFP8_PERCENTILE_WEIGHT_ONLY_CHANNEL_CFG)
        model(self.inputs)
        self.assertIsNotNone(model.linear1.scale_w)
        self.assertIsNotNone(model.linear2.scale_w)
        self.assertIsNotNone(model.linear3.scale_w)
        self.assertEqual(type(model.linear3).__name__, 'PercentileQuant')
        torch.Tensor.npu = mock_npu
        convert(model)
        quant_out = model(self.inputs.npu())

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch('amct_pytorch.deploy_op.weight_npu_quant_module.check_parameters_in_schema', MagicMock(return_value=True))
    def test_hifp8_percentile_act_tensor_weight_tensor_success(self, mock_1, mock_2, mock_3, mock_4):
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, HIFP8_PERCENTILE_ACT_TENSOR_WEIGHT_TENSOR_CFG)
        model(self.inputs)
        self.assertIsNotNone(model.linear1.scale_w)
        self.assertIsNotNone(model.linear1.scale_d)
        self.assertIsNotNone(model.linear2.scale_w)
        self.assertIsNotNone(model.linear2.scale_d)
        self.assertIsNotNone(model.linear3.scale_w)
        self.assertIsNotNone(model.linear3.scale_d)
        self.assertEqual(type(model.linear3).__name__, 'PercentileQuant')
        torch.Tensor.npu = mock_npu
        convert(model)
        quant_out = model(self.inputs.npu())

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch('amct_pytorch.deploy_op.weight_npu_quant_module.check_parameters_in_schema', MagicMock(return_value=True))
    def test_hifp8_percentile_act_tensor_weight_channel_success(self, mock_1, mock_2, mock_3, mock_4):
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, HIFP8_PERCENTILE_ACT_TENSOR_WEIGHT_CHANNEL_CFG)
        model(self.inputs)
        self.assertIsNotNone(model.linear1.scale_w)
        self.assertIsNotNone(model.linear1.scale_d)
        self.assertIsNotNone(model.linear2.scale_w)
        self.assertIsNotNone(model.linear2.scale_d)
        self.assertIsNotNone(model.linear3.scale_w)
        self.assertIsNotNone(model.linear3.scale_d)
        self.assertEqual(type(model.linear3).__name__, 'PercentileQuant')
        torch.Tensor.npu = mock_npu
        convert(model)
        quant_out = model(self.inputs.npu())

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch('amct_pytorch.deploy_op.weight_npu_quant_module.check_parameters_in_schema', MagicMock(return_value=True))
    def test_hifp8_percentile_act_token_static_weight_tensor_success(self, mock_1, mock_2, mock_3, mock_4):
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, HIFP8_PERCENTILE_ACT_TOKEN_STATIC_WEIGHT_TENSOR_CFG)
        model(self.inputs)
        self.assertIsNotNone(model.linear1.scale_w)
        self.assertIsNotNone(model.linear1.scale_d)
        self.assertIsNotNone(model.linear2.scale_w)
        self.assertIsNotNone(model.linear2.scale_d)
        self.assertIsNotNone(model.linear3.scale_w)
        self.assertIsNotNone(model.linear3.scale_d)
        self.assertEqual(type(model.linear3).__name__, 'PercentileQuant')
        torch.Tensor.npu = mock_npu
        convert(model)
        quant_out = model(self.inputs.npu())

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch('amct_pytorch.deploy_op.weight_npu_quant_module.check_parameters_in_schema', MagicMock(return_value=True))
    def test_hifp8_percentile_act_token_static_weight_channel_success(self, mock_1, mock_2, mock_3, mock_4):
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, HIFP8_PERCENTILE_ACT_TOKEN_STATIC_WEIGHT_CHANNEL_CFG)
        model(self.inputs)
        self.assertIsNotNone(model.linear1.scale_w)
        self.assertIsNotNone(model.linear1.scale_d)
        self.assertIsNotNone(model.linear2.scale_w)
        self.assertIsNotNone(model.linear2.scale_d)
        self.assertIsNotNone(model.linear3.scale_w)
        self.assertIsNotNone(model.linear3.scale_d)
        self.assertEqual(type(model.linear3).__name__, 'PercentileQuant')
        torch.Tensor.npu = mock_npu
        convert(model)
        quant_out = model(self.inputs.npu())


if __name__ == '__main__':
    unittest.main()
