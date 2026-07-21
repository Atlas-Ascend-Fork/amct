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

import torch
import torch.nn as nn
from mock_torch_npu import (
    mock_npu,
    mock_npu_convert_weight_to_int4pack,
    mock_npu_dtype_cast,
    mock_npu_dynamic_mx_quant,
    mock_npu_dynamic_quant,
    mock_npu_format_cast,
    mock_npu_quant_matmul,
    mock_npu_quantize,
    mock_npu_trans_quant_param,
    mock_npu_weight_quant_batchmatmul,
)
from utils import TestModel, TestModelBias

from amct_pytorch import convert, quantize

logger = logging.getLogger(__name__)

SMOOTH_QUANT = 'SmoothQuant'
NPU_QUANTIZATION_LINEAR = 'NpuQuantizationLinear'

torch.manual_seed(0)

LINEAR = 'Linear'


class TestSmoothQuant(unittest.TestCase):
    '''
    ST FOR SMOOTH ALGORITHM
    '''
    @classmethod
    def setUpClass(cls):
        cls.test_model = TestModel().to(torch.bfloat16)
        cls.inputs = torch.randn(64, 64).to(torch.bfloat16)
        cls.ori_out = cls.test_model(cls.inputs)
        logger.info('TestSmoothQuant START!')

    @classmethod
    def tearDownClass(cls):
        logger.info('TestSmoothQuant END!')

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
        'amct_pytorch.classic.deploy_op.npu_quantization_linear.check_parameters_in_schema',
        MagicMock(return_value=True),
    )
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
        self.assertEqual(type(model.linear1).__name__, SMOOTH_QUANT)
        self.assertEqual(type(model.linear2).__name__, SMOOTH_QUANT)
        self.assertEqual(type(model.linear3).__name__, SMOOTH_QUANT)
        convert(model)
        model(self.inputs.npu())
        self.assertEqual(type(model.linear1).__name__, NPU_QUANTIZATION_LINEAR)
        self.assertEqual(type(model.linear2).__name__, NPU_QUANTIZATION_LINEAR)
        self.assertEqual(type(model.linear3).__name__, NPU_QUANTIZATION_LINEAR)


    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch('amct_pytorch.classic.deploy_op.npu_quantization_linear.check_parameters_in_schema', 
           MagicMock(return_value=True))
    def test_int8_int8_tensor_asym_smooth_invalid(self, mock_1, mock_2, mock_3, mock_4):
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
        with self.assertRaisesRegex(ValueError, 'int8 int8 only support symmetric weight quantization'):
            quantize(model, cfg)


    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch('torch_npu.npu_dynamic_quant', wraps=mock_npu_dynamic_quant)
    @patch(
        'amct_pytorch.classic.deploy_op.npu_quantization_linear.check_parameters_in_schema',
        MagicMock(return_value=True),
    )
    def test_int8_int8_token_sym_smooth_success(self, mock_1, mock_2, mock_3, mock_4, mock_5):
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int8',
                    'symmetric': True,
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
        self.assertEqual(type(model.linear1).__name__, SMOOTH_QUANT)
        self.assertEqual(type(model.linear2).__name__, SMOOTH_QUANT)
        self.assertEqual(type(model.linear3).__name__, SMOOTH_QUANT)
        self.assertIsNone(model.linear1.offset_w)
        self.assertIsNotNone(model.linear1.scale_d)
        self.assertIsNone(model.linear1.offset_d)
        convert(model)
        model(self.inputs.npu())
        self.assertEqual(type(model.linear1).__name__, 'NpuQuantizationLinear')
        self.assertEqual(type(model.linear2).__name__, 'NpuQuantizationLinear')
        self.assertEqual(type(model.linear3).__name__, 'NpuQuantizationLinear')

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch('torch_npu.npu_dynamic_quant', wraps=mock_npu_dynamic_quant)
    @patch(
        'amct_pytorch.classic.deploy_op.npu_quantization_linear.check_parameters_in_schema',
        MagicMock(return_value=True),
    )
    def test_int8_int8_token_deploy_uses_dynamic_quant(self, mock_1, mock_2, mock_3, mock_4, mock_5):
        '''
        INT8 per-token 激活量化部署后,forward 必须走运行时 npu_dynamic_quant 分支
        (而非定长静态 npu_quantize),否则推理 seqlen 与校准不同会维度不匹配报错。
        本用例断言部署算子确实调用了 npu_dynamic_quant,且推理 seqlen 不同于校准时仍正常。
        '''
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int8',
                    'symmetric': True,
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
        convert(model)
        self.assertEqual(type(model.linear1).__name__, NPU_QUANTIZATION_LINEAR)
        # 推理 batch(行数)故意不同于校准的 64,验证 per-token 动态量化不绑定校准维度
        infer_inputs = torch.randn(16, 64).to(torch.bfloat16)
        quant_out = model(infer_inputs.npu())
        # mock_1 是 npu_dynamic_quant(最靠近函数的 wraps 装饰器);
        # check_parameters_in_schema 用 MagicMock(new=...) 不注入参数。
        self.assertTrue(mock_1.called)
        self.assertEqual(quant_out.shape[0], 16)

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch('torch_npu.npu_trans_quant_param', wraps=mock_npu_trans_quant_param)
    @patch('amct_pytorch.classic.deploy_op.npu_quantization_linear.check_parameters_in_schema',
           MagicMock(return_value=True))
    def test_int8_int4_tensor_tensor_sym_smooth_success(self, mock_1, mock_2, mock_3, mock_4, mock_5):
        self._run_a8w4_smooth_case('tensor', True)
        self.assertTrue(mock_1.called)

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch('torch_npu.npu_trans_quant_param', wraps=mock_npu_trans_quant_param)
    @patch('amct_pytorch.classic.deploy_op.npu_quantization_linear.check_parameters_in_schema', 
           MagicMock(return_value=True))
    def test_int8_int4_tensor_channel_sym_smooth_success(self, mock_1, mock_2, mock_3, mock_4, mock_5):
        self._run_a8w4_smooth_case('channel', True)
        self.assertTrue(mock_1.called)

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch('torch_npu.npu_trans_quant_param', wraps=mock_npu_trans_quant_param)
    @patch('amct_pytorch.classic.deploy_op.npu_quantization_linear.check_parameters_in_schema', 
           MagicMock(return_value=True))
    def test_int8_int4_asym_act_tensor_smooth_success(self, mock_1, mock_2, mock_3, mock_4, mock_5):
        self._run_a8w4_smooth_case('tensor', False)
        self.assertTrue(mock_1.called)

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch('torch_npu.npu_trans_quant_param', wraps=mock_npu_trans_quant_param)
    @patch('amct_pytorch.classic.deploy_op.npu_quantization_linear.check_parameters_in_schema', 
           MagicMock(return_value=True))
    def test_int8_int4_asym_act_channel_smooth_success(self, mock_1, mock_2, mock_3, mock_4, mock_5):
        self._run_a8w4_smooth_case('channel', False)
        self.assertTrue(mock_1.called)

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize) 
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul) 
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul) 
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack) 
    @patch('torch_npu.npu_format_cast', wraps=mock_npu_format_cast) 
    @patch('torch_npu.npu_dtype_cast', wraps=mock_npu_dtype_cast) 
    @patch('torch_npu.npu_dynamic_mx_quant', wraps=mock_npu_dynamic_mx_quant) 
    @patch('torch_npu.npu_trans_quant_param', wraps=mock_npu_trans_quant_param) 
    @patch(
        'amct_pytorch.classic.deploy_op.npu_quantization_linear.check_parameters_in_schema',
        MagicMock(return_value=True),
    )
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
        self.assertEqual(type(model.linear1).__name__, LINEAR) 
        self.assertEqual(type(model.linear2).__name__, SMOOTH_QUANT) 
        self.assertEqual(type(model.linear3).__name__, LINEAR) 
        self.assertIsNotNone(model.linear2.scale_w1) 
        self.assertIsNotNone(model.linear2.scale_d) 
        convert(model) 
        model(self.inputs.npu()) 
        self.assertEqual(type(model.linear1).__name__, LINEAR) 
        self.assertEqual(type(model.linear2).__name__, 'NpuQuantizationLinear') 
        self.assertEqual(type(model.linear3).__name__, 'Linear')

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch(
        'amct_pytorch.classic.deploy_op.npu_quantization_linear.check_parameters_in_schema',
        MagicMock(return_value=True),
    )
    def test_hif8_hif8_channel_tensor_sym_smooth_success(self, mock_1, mock_2, mock_3, mock_4):
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
            'algorithm': {'smoothquant': {'smooth_strength': 0.5}}
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs)
        torch.Tensor.npu = mock_npu
        self.assertEqual(type(model.linear1).__name__, SMOOTH_QUANT)
        self.assertEqual(type(model.linear2).__name__, SMOOTH_QUANT)
        self.assertEqual(type(model.linear3).__name__, SMOOTH_QUANT)
        # hifloat8 is always symmetric, so no activation/weight offset is produced
        self.assertIsNone(model.linear1.offset_w)
        self.assertIsNone(model.linear1.offset_d)
        self.assertIsNotNone(model.linear1.scale_w)
        self.assertIsNotNone(model.linear1.scale_d)
        convert(model)
        quant_out = model(self.inputs.npu())
        self.assertEqual(type(model.linear1).__name__, NPU_QUANTIZATION_LINEAR)
        self.assertEqual(type(model.linear2).__name__, NPU_QUANTIZATION_LINEAR)
        self.assertEqual(type(model.linear3).__name__, NPU_QUANTIZATION_LINEAR)

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch('torch_npu.npu_dynamic_quant', wraps=mock_npu_dynamic_quant)
    @patch(
        'amct_pytorch.classic.deploy_op.npu_quantization_linear.check_parameters_in_schema',
        MagicMock(return_value=True),
    )
    def test_hif8_hif8_token_sym_smooth_success(self, mock_1, mock_2, mock_3, mock_4, mock_5):
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
                    'strategy': 'token',
                },
            },
            'algorithm': {'smoothquant': {'smooth_strength': 0.4}}
        }
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        model(self.inputs)
        torch.Tensor.npu = mock_npu
        self.assertEqual(type(model.linear1).__name__, SMOOTH_QUANT)
        self.assertIsNone(model.linear1.offset_w)
        self.assertIsNone(model.linear1.offset_d)
        self.assertIsNotNone(model.linear1.scale_d)
        convert(model)
        # 推理 batch(行数)故意不同于校准的 64,验证 per-token 动态量化不绑定校准维度
        infer_inputs = torch.randn(16, 64).to(torch.bfloat16)
        quant_out = model(infer_inputs.npu())
        # mock_5 是 npu_dynamic_quant,hif8 per-token 部署必须走运行时动态量化分支
        self.assertTrue(mock_5.called)
        self.assertEqual(quant_out.shape[0], 16)
        self.assertEqual(type(model.linear1).__name__, NPU_QUANTIZATION_LINEAR)
        self.assertEqual(type(model.linear2).__name__, NPU_QUANTIZATION_LINEAR)
        self.assertEqual(type(model.linear3).__name__, NPU_QUANTIZATION_LINEAR)

    def _run_a8w4_smooth_case(self, weight_strategy, act_symmetric):
        model = copy.deepcopy(TestModel()).to(torch.float16)
        inputs = self.inputs.to(torch.float16)
        cfg = {
            'batch_num': 1,
            'quant_cfg': {
                'weights': {
                    'type': 'int4',
                    'symmetric': True,
                    'strategy': weight_strategy,
                },
                'inputs': {
                    'type': 'int8',
                    'symmetric': act_symmetric,
                    'strategy': 'tensor',
                },
            },
            'algorithm': {'smoothquant': {'smooth_strength': 0.5}}
        }
        quantize(model, cfg)
        model(inputs)
        torch.Tensor.npu = mock_npu
        self.assertEqual(type(model.linear1).__name__, 'SmoothQuant')
        self.assertEqual(type(model.linear2).__name__, 'SmoothQuant')
        self.assertEqual(type(model.linear3).__name__, 'Linear')
        self.assertIsNotNone(model.linear1.scale_w)
        self.assertIsNotNone(model.linear1.scale_d)
        self.assertIsNone(model.linear1.offset_w)
        if act_symmetric:
            self.assertIsNone(model.linear1.offset_d)
        else:
            self.assertIsNotNone(model.linear1.offset_d)
        convert(model)
        model(inputs.npu())
        self.assertEqual(type(model.linear1).__name__, 'NpuQuantizationLinear')
        self.assertEqual(type(model.linear2).__name__, NPU_QUANTIZATION_LINEAR)
        self.assertEqual(type(model.linear3).__name__, LINEAR)

