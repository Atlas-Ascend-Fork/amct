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
import sys
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

import torch

from amct_pytorch import quantize, convert, prepare_sinq
from amct_pytorch.config import parse_config
from amct_pytorch.algorithm import AlgorithmRegistry
from amct_pytorch.algorithm.sinq import generate_sinq_params
from amct_pytorch.utils.quant_util import quant_dequant_tensor
from mock_torch_npu import *
from utils import TestModel

torch.manual_seed(0)

INT4_SINQ_WEIGHT_QUANT_CFG = {
    'batch_num': 1,
    'quant_cfg': {
        'weights': {
            'type': 'int4',
            'symmetric': False,
            'strategy': 'group',
            'group_size': 32
        },
    },
    'algorithm': {'sinq': {'order': 16}},
    'skip_layers': {'lm_head'}
}

INT8_SINQ_WEIGHT_QUANT_CFG = {
    'batch_num': 1,
    'quant_cfg': {
        'weights': {
            'type': 'int8',
            'symmetric': False,
            'strategy': 'group',
            'group_size': 32
        },
    },
    'algorithm': {'sinq': {'order': 16}},
    'skip_layers': {'lm_head'}
}


class TestSinq(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_model = TestModel().to(torch.bfloat16)
        cls.inputs = torch.randn(64, 64).to(torch.bfloat16)
        print('TestSinq START!')

    @classmethod
    def tearDownClass(cls):
        print('TestSinq END!')

    def setUp(self):
        mock_torch_npu = MagicMock()
        sys.modules['torch_npu'] = mock_torch_npu

    def tearDown(self):
        del sys.modules['torch_npu']

    def _run_prepare_forward_success(self, cfg):
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        before_scale = model.linear1.scale.detach().clone()
        self.assertFalse(model.linear1.calibration_applied)

        prepare_sinq(model, self.inputs)
        self.assertTrue(model.linear1.calibration_applied)
        self.assertFalse(torch.allclose(model.linear1.scale, before_scale))

        torch.Tensor.npu = mock_npu
        convert(model)
        self.assertEqual(type(model.linear1).__name__, 'NpuWeightQuantizedLinear')
        _ = model(self.inputs.npu())

    def test_int4_sinq_cfg(self):
        detail_config = parse_config(self.test_model, INT4_SINQ_WEIGHT_QUANT_CFG, AlgorithmRegistry)

        self.assertEqual(len(detail_config.keys()), 2)
        self.assertEqual(detail_config.get('linear1').get('weights_cfg').get('strategy'), 'group')
        self.assertEqual(detail_config.get('linear1').get('weights_cfg').get('group_size'), 32)
        self.assertEqual(detail_config.get('linear1').get('algorithm').get('sinq').get('order'), 16)

    def test_int8_sinq_cfg(self):
        detail_config = parse_config(self.test_model, INT8_SINQ_WEIGHT_QUANT_CFG, AlgorithmRegistry)

        self.assertEqual(len(detail_config.keys()), 2)
        self.assertEqual(detail_config.get('linear1').get('weights_cfg').get('quant_type'), 'int8')
        self.assertEqual(detail_config.get('linear1').get('weights_cfg').get('strategy'), 'group')
        self.assertEqual(detail_config.get('linear1').get('weights_cfg').get('group_size'), 32)
        self.assertEqual(detail_config.get('linear1').get('algorithm').get('sinq').get('order'), 16)

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch('amct_pytorch.deploy_op.weight_npu_quant_module.check_parameters_in_schema', MagicMock(return_value=True))
    def test_int4_sinq_success(self, mock_1, mock_2, mock_3, mock_4):
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, INT4_SINQ_WEIGHT_QUANT_CFG)
        self.assertEqual(type(model.linear1).__name__, 'LinearSINQQuant')
        self.assertEqual(type(model.linear2).__name__, 'LinearSINQQuant')
        self.assertEqual(type(model.linear3).__name__, 'Linear')
        self.assertEqual(list(model.linear1.scale.shape), [1, 64])
        self.assertEqual(list(model.linear1.scale_w.shape), [64, 2, 1])
        self.assertIsNone(model.linear1.scale_d)

        torch.Tensor.npu = mock_npu
        convert(model)
        self.assertEqual(type(model.linear1).__name__, 'NpuWeightQuantizedLinear')
        self.assertEqual(model.linear1.quantized_weight.dtype, torch.int32)
        _ = model(self.inputs.npu())

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch('amct_pytorch.deploy_op.weight_npu_quant_module.check_parameters_in_schema', MagicMock(return_value=True))
    def test_int8_sinq_asym_success(self, mock_1, mock_2, mock_3, mock_4):
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, INT8_SINQ_WEIGHT_QUANT_CFG)
        self.assertEqual(type(model.linear1).__name__, 'LinearSINQQuant')
        self.assertEqual(type(model.linear2).__name__, 'LinearSINQQuant')
        self.assertEqual(type(model.linear3).__name__, 'Linear')
        self.assertEqual(list(model.linear1.scale.shape), [1, 64])
        self.assertEqual(list(model.linear1.scale_w.shape), [64, 2, 1])
        self.assertEqual(model.linear1.wts_type, 'int8')
        self.assertTrue(torch.all(model.linear1.offset_w >= -128))
        self.assertTrue(torch.all(model.linear1.offset_w <= 127))

        torch.Tensor.npu = mock_npu
        convert(model)
        self.assertEqual(type(model.linear1).__name__, 'NpuWeightQuantizedLinear')
        self.assertEqual(model.linear1.quantized_weight.dtype, torch.int8)
        _ = model(self.inputs.npu())

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch('amct_pytorch.deploy_op.weight_npu_quant_module.check_parameters_in_schema', MagicMock(return_value=True))
    def test_int8_sinq_sym_success(self, mock_1, mock_2, mock_3, mock_4):
        cfg = copy.deepcopy(INT8_SINQ_WEIGHT_QUANT_CFG)
        cfg['quant_cfg']['weights']['symmetric'] = True
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, cfg)
        self.assertEqual(type(model.linear1).__name__, 'LinearSINQQuant')
        self.assertTrue(torch.all(model.linear1.offset_w == 0))

        torch.Tensor.npu = mock_npu
        convert(model)
        self.assertEqual(type(model.linear1).__name__, 'NpuWeightQuantizedLinear')
        self.assertEqual(model.linear1.quantized_weight.dtype, torch.int8)
        _ = model(self.inputs.npu())

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch('amct_pytorch.deploy_op.weight_npu_quant_module.check_parameters_in_schema', MagicMock(return_value=True))
    def test_scale2_mapping_matches_reference(self, mock_1, mock_2, mock_3, mock_4):
        model = copy.deepcopy(self.test_model).to(torch.bfloat16)
        quantize(model, INT4_SINQ_WEIGHT_QUANT_CFG)
        layer = model.linear1
        expected_scale = layer.scale.detach().clone()
        scaled_weight = layer.weight / layer.scale
        dq_weight = quant_dequant_tensor(
            scaled_weight, layer.wts_type, layer.scale_w, layer.offset_w, layer.group_size
        )
        ref = torch.nn.functional.linear(self.inputs * layer.scale, dq_weight, layer.bias)
        self.assertEqual(ref.dtype, self.inputs.dtype)

        torch.Tensor.npu = mock_npu
        convert(model)
        self.assertTrue(torch.allclose(model.linear1.scale_factor.cpu(), expected_scale.cpu()))

    @patch('amct_pytorch.algorithm.sinq.tiled_quant_rectangle')
    @patch('amct_pytorch.algorithm.sinq.kurtosis', return_value=torch.tensor(6.0))
    def test_sinq_kurtosis_gate_falls_back_to_rtn(self, _, mock_tiled):
        mock_tiled.return_value = (
            torch.zeros(128, 32, dtype=torch.int32),
            torch.ones(128, 1, dtype=torch.float32),
            torch.ones(1, 64, dtype=torch.float32),
            torch.zeros(128, 1, dtype=torch.int32),
        )
        weight = torch.randn(64, 64, dtype=torch.float32)
        cfg = copy.deepcopy(INT4_SINQ_WEIGHT_QUANT_CFG)
        cfg['quant_cfg']['weights']['symmetric'] = False
        generate_sinq_params(weight, parse_config(self.test_model, cfg, AlgorithmRegistry)['linear1'])
        self.assertEqual(mock_tiled.call_args.kwargs['method'], 'rtn')

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch('amct_pytorch.deploy_op.weight_npu_quant_module.check_parameters_in_schema', MagicMock(return_value=True))
    @patch('amct_pytorch.algorithm.sinq.compute_awq_scale', return_value=torch.full((64,), 1.5, dtype=torch.float32))
    def test_int4_sinq_prepare_forward_enables_calibration(self, mock_1, mock_2, mock_3, mock_4, _mock_awq):
        self._run_prepare_forward_success(INT4_SINQ_WEIGHT_QUANT_CFG)

    @patch('torch_npu.npu_quantize', wraps=mock_npu_quantize)
    @patch('torch_npu.npu_quant_matmul', wraps=mock_npu_quant_matmul)
    @patch('torch_npu.npu_weight_quant_batchmatmul', wraps=mock_npu_weight_quant_batchmatmul)
    @patch('torch_npu.npu_convert_weight_to_int4pack', wraps=mock_npu_convert_weight_to_int4pack)
    @patch('amct_pytorch.deploy_op.weight_npu_quant_module.check_parameters_in_schema', MagicMock(return_value=True))
    @patch('amct_pytorch.algorithm.sinq.compute_awq_scale', return_value=torch.full((64,), 1.5, dtype=torch.float32))
    def test_int8_sinq_prepare_forward_enables_calibration(self, mock_1, mock_2, mock_3, mock_4, _mock_awq):
        self._run_prepare_forward_success(INT8_SINQ_WEIGHT_QUANT_CFG)
