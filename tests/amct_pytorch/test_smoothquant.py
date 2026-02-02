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

torch.manual_seed(0)

class TestSmoothQuant(unittest.TestCase):
    '''
    ST FOR SMOOTH ALGORITHM
    '''
    @classmethod
    def setUpClass(cls):
        cls.test_model = TestModel().to(torch.bfloat16)
        cls.inputs = torch.randn(64, 64).to(torch.bfloat16)
        cls.ori_out = cls.test_model(cls.inputs)
        print('TestSmoothQuant START!')

    @classmethod
    def tearDownClass(cls):
        print('TestSmoothQuant END!')

    def setUp(self):
        mock_torch_npu = MagicMock()
        sys.modules['torch_npu'] = mock_torch_npu
 
    def tearDown(self):
        del sys.modules['torch_npu']

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    def test_int8_int8_tensor_sym_smooth_success(self, mock_1, mock_2, mock_3, mock_4):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'tensor',
                },
                'inputs': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {'smoothquant': {'smooth_strength': 0.5}}
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs)
        torch.Tensor.npu = mock_npu
        self.assertEqual(type(model.linear1).__name__, 'SmoothQuant')
        self.assertEqual(type(model.linear2).__name__, 'SmoothQuant')
        self.assertEqual(type(model.linear3).__name__, 'SmoothQuant')
        convert(model)
        quant_out = model(self.inputs.npu())
        self.assertEqual(type(model.linear1).__name__, 'NpuQuantizationLinear')
        self.assertEqual(type(model.linear2).__name__, 'NpuQuantizationLinear')
        self.assertEqual(type(model.linear3).__name__, 'NpuQuantizationLinear')

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    def test_int8_int8_tensor_asym_smooth_success(self, mock_1, mock_2, mock_3, mock_4):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int8',
                    'symmetric': False,
                    'strategy': 'channel',
                },
                'inputs': {
                    'type': 'int8',
                    'symmetric': False,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {'smoothquant': {'smooth_strength': 0.4}}
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs)
        torch.Tensor.npu = mock_npu
        self.assertEqual(type(model.linear1).__name__, 'SmoothQuant')
        self.assertEqual(type(model.linear2).__name__, 'SmoothQuant')
        self.assertEqual(type(model.linear3).__name__, 'SmoothQuant')
        convert(model)
        quant_out = model(self.inputs.npu())
        self.assertEqual(type(model.linear1).__name__, 'NpuQuantizationLinear')
        self.assertEqual(type(model.linear2).__name__, 'NpuQuantizationLinear')
        self.assertEqual(type(model.linear3).__name__, 'NpuQuantizationLinear')


    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    def test_int8_int8_token_sym_smooth_success(self, mock_1, mock_2, mock_3, mock_4):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int8',
                    'symmetric': False,
                    'strategy': 'channel',
                },
                'inputs': {
                    'type': 'int8',
                    'symmetric': True,
                    'strategy': 'token',
                },
            },
            'algorithm': {'smoothquant': {'smooth_strength': 0.4}}
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs)
        torch.Tensor.npu = mock_npu
        self.assertEqual(type(model.linear1).__name__, 'SmoothQuant')
        self.assertEqual(type(model.linear2).__name__, 'SmoothQuant')
        self.assertEqual(type(model.linear3).__name__, 'SmoothQuant')
        self.assertIsNotNone(model.linear1.offset_w)
        self.assertIsNotNone(model.linear1.scale_d)
        self.assertIsNone(model.linear1.offset_d)
        convert(model)
        quant_out = model(self.inputs.npu())
        self.assertEqual(type(model.linear1).__name__, 'NpuQuantizationLinear')
        self.assertEqual(type(model.linear2).__name__, 'NpuQuantizationLinear')
        self.assertEqual(type(model.linear3).__name__, 'NpuQuantizationLinear')

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize) 
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul) 
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul) 
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack) 
    @patch('torch_npu.npu_format_cast', wraps=mock_npu_format_cast) 
    @patch('torch_npu.npu_dtype_cast', wraps=mock_npu_dtype_cast) 
    @patch('torch_npu.npu_dynamic_mx_quant', wraps=mock_npu_dynamic_mx_quant) 
    @patch('torch_npu.npu_trans_quant_param', wraps=mock_npu_trans_quant_param) 
    def test_fp8_fp4_group_sym_smooth_success(self, mock_1, mock_2, mock_3, mock_4, mock_5, mock_6, mock_7, mock_8): 
        cfg = { 
            'batch_num': 1, 
            'quant_cfg': { 
                'weights': { 
                    'type': 'float4_e2m1', 
                    'symmetric': True, 
                    'strategy': 'group', 
                    'group_size': 32 
                }, 
                'inputs': { 
                    'type': 'float8_e4m3fn', 
                    'symmetric': True, 
                    'strategy': 'tensor', 
                }, 
            }, 
            'algorithm': {'smoothquant': {'smooth_strength': 0.5}} 
        } 
        model = copy.deepcopy(TestModelBias()).to(torch.bfloat16) 
        quantize(model, cfg) 
        model(self.inputs) 
        torch.Tensor.npu = mock_npu 
        self.assertEqual(type(model.linear1).__name__, 'Linear') 
        self.assertEqual(type(model.linear2).__name__, 'SmoothQuant') 
        self.assertEqual(type(model.linear3).__name__, 'Linear') 
        self.assertIsNotNone(model.linear2.scale_w1) 
        self.assertIsNotNone(model.linear2.scale_d) 
        convert(model) 
        quant_out = model(self.inputs.npu()) 
        self.assertEqual(type(model.linear1).__name__, 'Linear') 
        self.assertEqual(type(model.linear2).__name__, 'NpuQuantizationLinear') 
        self.assertEqual(type(model.linear3).__name__, 'Linear')