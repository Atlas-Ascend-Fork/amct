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
import logging
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from mock_torch_npu import (
    mock_npu,
    mock_npu_convert_weight_to_int4pack,
    mock_npu_dtype_cast,
    mock_npu_dynamic_mx_quant,
    mock_npu_format_cast,
    mock_npu_quant_matmul,
    mock_npu_quantize,
    mock_npu_trans_quant_param,
    mock_npu_weight_quant_batchmatmul,
    mocked_npu_quant_conv2d,
)
from utils import TestModel, TestModelConv2d

from amct_pytorch import convert, quantize

logger = logging.getLogger(__name__)

OFMR_QUANT = 'OfmrQuant'
NPU_QUANTIZATION_LINEAR = 'NpuQuantizationLinear'
NPU_WEIGHT_QUANTIZED_LINEAR = 'NpuWeightQuantizedLinear'
NPU_QUANTIZATION_CONV2D = 'NpuQuantizationConv2d'

torch.manual_seed(0)


class TestOFMR(unittest.TestCase):
    '''
    ST FOR OFMR ALGORITHM
    '''
    @classmethod
    def setUpClass(cls):
        cls.test_model = TestModel().to(torch.bfloat16)
        cls.inputs = torch.randn(64, 64).to(torch.bfloat16)
        cls.ori_out = cls.test_model(cls.inputs)
        logger.info('TestOFMR START!')

    @classmethod
    def tearDownClass(cls):
        logger.info('TestOFMR END!')

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
    @patch(
        'amct_pytorch.classic.deploy_op.weight_npu_quant_module.check_parameters_in_schema',
        MagicMock(return_value=True),
    )
    def test_hif8_weight_tensor_sym_ofmr_success(self, mock_1, mock_2, mock_3, mock_4, mock_5, mock_6):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'hifloat8',
                    'symmetric': True,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {'ofmr'},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs.to(torch.bfloat16))
        torch.Tensor.npu = mock_npu
        self.assertEqual(type(model.linear1).__name__, OFMR_QUANT)
        self.assertEqual(type(model.linear2).__name__, OFMR_QUANT)
        self.assertEqual(type(model.linear3).__name__, OFMR_QUANT)
        self.assertIsNotNone(model.linear1.scale_w)
        self.assertIsNone(model.linear1.offset_w)
        self.assertIsNone(model.linear1.scale_d)
        self.assertIsNone(model.linear1.offset_d)
        convert(model)
        model(self.inputs.npu())
        self.assertEqual(type(model.linear1).__name__, NPU_WEIGHT_QUANTIZED_LINEAR)
        self.assertEqual(type(model.linear2).__name__, NPU_WEIGHT_QUANTIZED_LINEAR)
        self.assertEqual(type(model.linear3).__name__, NPU_WEIGHT_QUANTIZED_LINEAR)

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch('torch_npu.npu_format_cast', wraps=mock_npu_format_cast)
    @patch('torch_npu.npu_dtype_cast', wraps=mock_npu_dtype_cast)
    @patch(
        'amct_pytorch.classic.deploy_op.weight_npu_quant_module.check_parameters_in_schema',
        MagicMock(return_value=True),
    )
    def test_hif8_weight_channel_sym_ofmr_success(self, mock_1, mock_2, mock_3, mock_4, mock_5, mock_6):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'hifloat8',
                    'symmetric': True,
                    'strategy': 'channel',
                },
            },
            'algorithm': {'ofmr'},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs.to(torch.bfloat16))
        torch.Tensor.npu = mock_npu
        self.assertEqual(type(model.linear1).__name__, OFMR_QUANT)
        self.assertEqual(type(model.linear2).__name__, OFMR_QUANT)
        self.assertEqual(type(model.linear3).__name__, OFMR_QUANT)
        self.assertIsNotNone(model.linear1.scale_w)
        self.assertIsNone(model.linear1.offset_w)
        self.assertIsNone(model.linear1.scale_d)
        self.assertIsNone(model.linear1.offset_d)
        convert(model)
        model(self.inputs.npu())
        self.assertEqual(type(model.linear1).__name__, NPU_WEIGHT_QUANTIZED_LINEAR)
        self.assertEqual(type(model.linear2).__name__, NPU_WEIGHT_QUANTIZED_LINEAR)
        self.assertEqual(type(model.linear3).__name__, NPU_WEIGHT_QUANTIZED_LINEAR)

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch('torch_npu.npu_format_cast', wraps=mock_npu_format_cast)
    @patch('torch_npu.npu_dtype_cast', wraps=mock_npu_dtype_cast)
    @patch('torch_npu.npu_dynamic_mx_quant', wraps=mock_npu_dynamic_mx_quant)
    @patch(
        'amct_pytorch.classic.deploy_op.weight_npu_quant_module.check_parameters_in_schema',
        MagicMock(return_value=True),
    )
    def test_fp8_weight_tensor_sym_ofmr_success(self, mock_1, mock_2, mock_3, mock_4, mock_5, mock_6, mock_7):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'float8_e4m3fn',
                    'symmetric': True,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {'ofmr'},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs.to(torch.bfloat16))
        torch.Tensor.npu = mock_npu
        self.assertEqual(type(model.linear1).__name__, OFMR_QUANT)
        self.assertEqual(type(model.linear2).__name__, OFMR_QUANT)
        self.assertEqual(type(model.linear3).__name__, OFMR_QUANT)
        self.assertIsNotNone(model.linear1.scale_w)
        self.assertIsNone(model.linear1.offset_w)
        self.assertIsNone(model.linear1.scale_d)
        self.assertIsNone(model.linear1.offset_d)
        convert(model)
        model(self.inputs.npu())
        self.assertEqual(type(model.linear1).__name__, NPU_WEIGHT_QUANTIZED_LINEAR)
        self.assertEqual(type(model.linear2).__name__, NPU_WEIGHT_QUANTIZED_LINEAR)
        self.assertEqual(type(model.linear3).__name__, NPU_WEIGHT_QUANTIZED_LINEAR)

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch('torch_npu.npu_format_cast', wraps=mock_npu_format_cast)
    @patch('torch_npu.npu_dtype_cast', wraps=mock_npu_dtype_cast)
    @patch('torch_npu.npu_dynamic_mx_quant', wraps=mock_npu_dynamic_mx_quant)
    @patch(
        'amct_pytorch.classic.deploy_op.weight_npu_quant_module.check_parameters_in_schema',
        MagicMock(return_value=True),
    )
    def test_fp8_weight_channel_sym_ofmr_success(self, mock_1, mock_2, mock_3, mock_4, mock_5, mock_6, mock_7):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'float8_e4m3fn',
                    'symmetric': True,
                    'strategy': 'channel',
                },
            },
            'algorithm': {'ofmr'},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs.to(torch.bfloat16))
        torch.Tensor.npu = mock_npu
        self.assertEqual(type(model.linear1).__name__, OFMR_QUANT)
        self.assertEqual(type(model.linear2).__name__, OFMR_QUANT)
        self.assertEqual(type(model.linear3).__name__, OFMR_QUANT)
        self.assertIsNotNone(model.linear1.scale_w)
        self.assertIsNone(model.linear1.offset_w)
        self.assertIsNone(model.linear1.scale_d)
        self.assertIsNone(model.linear1.offset_d)
        convert(model)
        model(self.inputs.npu())
        self.assertEqual(type(model.linear1).__name__, NPU_WEIGHT_QUANTIZED_LINEAR)
        self.assertEqual(type(model.linear2).__name__, NPU_WEIGHT_QUANTIZED_LINEAR)
        self.assertEqual(type(model.linear3).__name__, NPU_WEIGHT_QUANTIZED_LINEAR)

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch(
        'amct_pytorch.classic.deploy_op.npu_quantization_linear.check_parameters_in_schema',
        MagicMock(return_value=True),
    )
    def test_fp8_sym_ofmr_success(self, mock_1, mock_2, mock_3, mock_4):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'float8_e4m3fn',
                    'symmetric': True,
                    'strategy': 'channel',
                },
                'inputs': {
                    'type': 'float8_e4m3fn',
                    'symmetric': True,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {'ofmr'},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs.to(torch.bfloat16))
        torch.Tensor.npu = mock_npu
        self.assertEqual(type(model.linear1).__name__, OFMR_QUANT)
        self.assertEqual(type(model.linear2).__name__, OFMR_QUANT)
        self.assertEqual(type(model.linear3).__name__, OFMR_QUANT)
        self.assertIsNotNone(model.linear1.scale_w)
        self.assertIsNone(model.linear1.offset_w)
        self.assertIsNotNone(model.linear1.scale_d)
        self.assertIsNone(model.linear1.offset_d)
        convert(model)
        model(self.inputs.npu())
        self.assertEqual(type(model.linear1).__name__, NPU_QUANTIZATION_LINEAR)
        self.assertEqual(type(model.linear2).__name__, NPU_QUANTIZATION_LINEAR)
        self.assertEqual(type(model.linear3).__name__, NPU_QUANTIZATION_LINEAR)


    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch('torch_npu.npu_format_cast', wraps=mock_npu_format_cast)
    @patch('torch_npu.npu_dtype_cast', wraps=mock_npu_dtype_cast)
    @patch('torch_npu.npu_dynamic_mx_quant', wraps=mock_npu_dynamic_mx_quant)
    @patch(
        'amct_pytorch.classic.deploy_op.npu_quantization_linear.check_parameters_in_schema',
        MagicMock(return_value=True),
    )
    def test_hif8_sym_ofmr_success(self, mock_1, mock_2, mock_3, mock_4, mock_5, mock_6, mock_7):
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
            'algorithm': {'ofmr'},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs.to(torch.bfloat16))
        torch.Tensor.npu = mock_npu
        self.assertEqual(type(model.linear1).__name__, OFMR_QUANT)
        self.assertEqual(type(model.linear2).__name__, OFMR_QUANT)
        self.assertEqual(type(model.linear3).__name__, OFMR_QUANT)
        self.assertIsNotNone(model.linear1.scale_w)
        self.assertIsNone(model.linear1.offset_w)
        self.assertIsNotNone(model.linear1.scale_d)
        self.assertIsNone(model.linear1.offset_d)
        convert(model)
        model(self.inputs.npu())
        self.assertEqual(type(model.linear1).__name__, NPU_QUANTIZATION_LINEAR)
        self.assertEqual(type(model.linear2).__name__, NPU_QUANTIZATION_LINEAR)
        self.assertEqual(type(model.linear3).__name__, NPU_QUANTIZATION_LINEAR)


class TestOFMRConv2d(unittest.TestCase):
    '''
    ST FOR OFMR ALGORITHM
    '''
    @classmethod
    def setUpClass(cls):
        cls.test_model = TestModelConv2d().to(torch.bfloat16)
        cls.inputs = torch.randn(1, 32, 32, 32).to(torch.bfloat16)
        cls.ori_out = cls.test_model(cls.inputs)
        logger.info('TestOFMR START!')

    @classmethod
    def tearDownClass(cls):
        logger.info('TestOFMR END!')

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
    @patch('torch_npu.npu_quant_conv2d', wraps=mocked_npu_quant_conv2d)
    @patch('torch_npu.npu_trans_quant_param', wraps=mock_npu_trans_quant_param)
    def test_hif8_sym_ofmr_conv2d_success(self, mock_1, mock_2, mock_3, mock_4, mock_5, mock_6, mock_7, mock_8, mock_9):
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
            'algorithm': {'ofmr'},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs.to(torch.bfloat16))
        torch.Tensor.npu = mock_npu
        self.assertEqual(type(model.conv2d1).__name__, OFMR_QUANT)
        self.assertEqual(type(model.conv2d2).__name__, OFMR_QUANT)
        self.assertEqual(type(model.conv2d3).__name__, OFMR_QUANT)
        self.assertIsNotNone(model.conv2d1.scale_w)
        self.assertIsNone(model.conv2d1.offset_w)
        self.assertIsNotNone(model.conv2d1.scale_d)
        self.assertIsNone(model.conv2d1.offset_d)
        convert(model)
        model(self.inputs.npu())
        self.assertEqual(type(model.conv2d1).__name__, NPU_QUANTIZATION_CONV2D)
        self.assertEqual(type(model.conv2d2).__name__, NPU_QUANTIZATION_CONV2D)
        self.assertEqual(type(model.conv2d3).__name__, NPU_QUANTIZATION_CONV2D)


    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch('torch_npu.npu_quant_conv2d', wraps=mocked_npu_quant_conv2d)
    @patch('torch_npu.npu_trans_quant_param', wraps=mock_npu_trans_quant_param)
    def test_fp8_sym_ofmr_conv2d_success(self, mock_1, mock_2, mock_3, mock_4, mock_5, mock_6):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'float8_e4m3fn',
                    'symmetric': True,
                    'strategy': 'channel',
                },
                'inputs': {
                    'type': 'float8_e4m3fn',
                    'symmetric': True,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {'ofmr'},
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs.to(torch.bfloat16))
        torch.Tensor.npu = mock_npu
        self.assertEqual(type(model.conv2d1).__name__, OFMR_QUANT)
        self.assertEqual(type(model.conv2d2).__name__, OFMR_QUANT)
        self.assertEqual(type(model.conv2d3).__name__, OFMR_QUANT)
        self.assertIsNotNone(model.conv2d1.scale_w)
        self.assertIsNone(model.conv2d1.offset_w)
        self.assertIsNotNone(model.conv2d1.scale_d)
        self.assertIsNone(model.conv2d1.offset_d)
        convert(model)
        model(self.inputs.npu())
        self.assertEqual(type(model.conv2d1).__name__, NPU_QUANTIZATION_CONV2D)
        self.assertEqual(type(model.conv2d2).__name__, NPU_QUANTIZATION_CONV2D)
        self.assertEqual(type(model.conv2d3).__name__, NPU_QUANTIZATION_CONV2D)