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
from amct_pytorch.classic.quantize_op.quantile_module import QuantileQuant
from amct_pytorch.common.utils import quant_util

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
        # 部署后调用 forward,覆盖 NpuQuantizationLinear HIF8 per-token 动态量化分支
        deploy_out = model(torch.randn(64, 64).to(torch.bfloat16).npu())
        self.assertTrue(mock_4.called)
        self.assertIsNotNone(deploy_out)


def _guard_cfg(weight_type, enable_act, act_type=None):
    """Minimal internal quant_config dict consumed by QuantileQuant.__init__."""
    inputs_cfg = {'enable_quant': enable_act}
    if enable_act:
        inputs_cfg.update({'quant_type': act_type, 'symmetric': True,
                           'strategy': 'tensor', 'dynamic': False})
    return {
        'batch_num': 1,
        'inputs_cfg': inputs_cfg,
        'weights_cfg': {'quant_type': weight_type, 'strategy': 'channel'},
    }


class TestQuantileGuards(unittest.TestCase):
    """The two ValueError guards (quantile_module.py:68-72)."""

    def setUp(self):
        sys.modules['torch_npu'] = MagicMock()
        self.linear = nn.Linear(64, 64).to(torch.bfloat16)

    def tearDown(self):
        sys.modules.pop('torch_npu', None)

    def test_weight_type_must_be_hifloat8(self):
        # weight-only, int8 weight -> guard at line 69 fires before any scale calc.
        cfg = _guard_cfg('int8', enable_act=False)
        with self.assertRaises(ValueError):
            QuantileQuant(self.linear, 'linear', cfg)

    def test_activation_type_must_be_hifloat8(self):
        # hifloat8 weight (passes line 68) but int8 activation -> guard at line 72.
        cfg = _guard_cfg('hifloat8', enable_act=True, act_type='int8')
        with self.assertRaises(ValueError):
            QuantileQuant(self.linear, 'linear', cfg)


class TestQuantileHelpers(unittest.TestCase):
    """None-scale corner and the unknown-granularity batch-max branch."""

    def setUp(self):
        sys.modules['torch_npu'] = MagicMock()
        # A valid weight-only instance; we then probe its helper methods directly.
        cfg = _guard_cfg('hifloat8', enable_act=False)
        self.qq = QuantileQuant(nn.Linear(64, 64).to(torch.bfloat16), 'linear', cfg)

    def tearDown(self):
        sys.modules.pop('torch_npu', None)

    def test_calculate_hif8_scale_none_returns_none(self):
        # quantile_module.py:114 -- tensor_max is None short-circuit.
        self.assertIsNone(self.qq.calculate_hif8_scale(None))

    def test_compute_batch_max_unknown_granularity_returns_none(self):
        # quantile_module.py:154 -- granularity neither 'tensor' nor 'token'.
        self.qq.act_granularity = 'channel'
        self.assertIsNone(self.qq._compute_batch_max(torch.randn(4, 64)))

    def test_ema_update_on_second_batch(self):
        # quantile_module.py:160 -- previous_max already set -> EMA path.
        self.qq.previous_max = torch.ones(1)
        self.qq._update_act_scale_tensor(torch.full((1,), 2.0))
        # EMA blends old and new, so result is strictly between 1 and 2.
        self.assertTrue(torch.all(self.qq.previous_max > 1.0))
        self.assertTrue(torch.all(self.qq.previous_max <= 2.0))


class TestQuantileDynamicFallback(unittest.TestCase):
    """Dynamic per-token activation amct_ops fallback (quantile_module.py:136-140)."""

    def setUp(self):
        sys.modules['torch_npu'] = MagicMock()
        torch.Tensor.npu = lambda self: self

    def tearDown(self):
        sys.modules.pop('torch_npu', None)

    def test_dynamic_token_without_native_hif8_uses_fake_quant(self):
        qq = self._make_dynamic_module()
        self.assertTrue(qq.dynamic)
        # Force the no-native-hifloat8 branch (else at line 133->136) and keep the
        # HiF8 round trip on CPU (identity) instead of hitting a real NPU op.
        with patch.object(quant_util, 'hifloat8_supported', return_value=False), \
             patch.object(quant_util, 'hifloat8_fake_quant', side_effect=lambda t: t):
            out = qq(torch.randn(64, 64).to(torch.bfloat16))
        self.assertEqual(out.shape[0], 64)
        # fake-quant cache built on first forward, so the second call reuses it.
        self.assertTrue(qq.fake_quant_cache_ready)

    def _make_dynamic_module(self):
        cfg = {
            'batch_num': 1,
            'inputs_cfg': {'enable_quant': True, 'quant_type': 'hifloat8',
                           'symmetric': True, 'strategy': 'token', 'dynamic': True},
            'weights_cfg': {'quant_type': 'hifloat8', 'strategy': 'tensor'},
        }
        return QuantileQuant(nn.Linear(64, 64).to(torch.bfloat16), 'linear', cfg)


if __name__ == '__main__':
    unittest.main()

