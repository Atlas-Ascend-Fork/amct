# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache License for more details at
# http://www.apache.org/licenses/LICENSE-2.0
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


class TestMinMax(unittest.TestCase):
    '''
    ST FOR MINMAX ALGORITHM
    '''
    @classmethod
    def setUpClass(cls):
        cls.test_model = TestModel().to(torch.bfloat16)
        cls.inputs = torch.randn(64, 64).to(torch.bfloat16)
        cls.ori_out = cls.test_model(cls.inputs)
        print('TestMinMax START!')

    @classmethod
    def tearDownClass(cls):
        print('TestMinMax END!')

    def setUp(self):
        mock_torch_npu = MagicMock()
        sys.modules['torch_npu'] = mock_torch_npu

    def tearDown(self):
        del sys.modules['torch_npu']

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch('torch_npu.npu_format_cast', wraps=mock_npu_format_cast)
    @patch('torch_npu.npu_dtype_cast', wraps=mock_npu_dtype_cast)
    @patch('torch_npu.npu_dynamic_mx_quant', wraps=mock_npu_dynamic_mx_quant)
    def test_mxfp4_group_sym_mxquant_success(self, mock_1, mock_2, mock_3, mock_4, mock_5, mock_6, mock_7):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'mxfp4_e2m1',
                    'symmetric': True,
                    'strategy': 'group',
                    'group_size': 32
                },
            },
            'algorithm': {'mxquant'},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        torch.Tensor.npu = mock_npu
        quantize(model, cfg)
        model(self.inputs.npu())
        self.assertEqual(type(model.linear1).__name__, 'NpuMXQuantizationLinear')
        self.assertEqual(type(model.linear2).__name__, 'Linear')
        self.assertEqual(type(model.linear3).__name__, 'Linear')
        self.assertTrue(model.linear1.weight_compress_only)
        self.assertIsNotNone(model.linear1.scale_w)
        convert(model)
        quant_out = model(self.inputs.npu())

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch('torch_npu.npu_format_cast', wraps=mock_npu_format_cast)
    @patch('torch_npu.npu_dtype_cast', wraps=mock_npu_dtype_cast)
    @patch('torch_npu.npu_dynamic_mx_quant', wraps=mock_npu_dynamic_mx_quant)
    def test_mxfp8_mxfp8_group_sym_mxquant_success(self, mock_1, mock_2, mock_3, mock_4, mock_5, mock_6, mock_7):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'mxfp8_e4m3fn',
                    'symmetric': True,
                    'strategy': 'group',
                    'group_size': 32
                },
                'inputs': {
                    'type': 'mxfp8_e4m3fn',
                    'symmetric': True,
                    'strategy': 'group',
                },
            },
            'algorithm': {'mxquant'},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        torch.Tensor.npu = mock_npu
        quantize(model, cfg)
        model(self.inputs.npu())
        self.assertEqual(type(model.linear1).__name__, 'NpuMXQuantizationLinear')
        self.assertEqual(type(model.linear2).__name__, 'NpuMXQuantizationLinear')
        self.assertEqual(type(model.linear3).__name__, 'Linear')
        self.assertFalse(model.linear1.weight_compress_only)
        self.assertIsNotNone(model.linear1.scale_w)
        convert(model)
        quant_out = model(self.inputs.npu())