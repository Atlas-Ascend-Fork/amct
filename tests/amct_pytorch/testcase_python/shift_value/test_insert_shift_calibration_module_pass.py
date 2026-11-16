#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
"""Unit tests of InsertShiftCalibrationModulePass."""

import copy
import os
import shutil
import unittest

import pytest
from unittest.mock import patch

import torch

from amct_pytorch.classic.graph_based.amct_pytorch.configuration.configuration import (
    Configuration,
)
from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.quantization.shift_calibration import (
    ShiftCalibrationModule,
)
from amct_pytorch.classic.graph_based.amct_pytorch.optimizer.insert_shift_calibration_module_pass import (
    InsertShiftCalibrationModulePass,
)
from amct_pytorch.classic.graph_based.amct_pytorch.optimizer.model_optimizer import (
    ModelOptimizer,
)
from amct_pytorch.classic.graph_based.amct_pytorch.parser.parser import Parser

pytestmark = pytest.mark.cpu


CUR_DIR = os.path.split(os.path.realpath(__file__))[0]


class TinyModel(torch.nn.Module):
    """conv + fc, both are FIXED_SHIFT_VALUE_TYPES modules."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.fc = torch.nn.Linear(8 * 8 * 8, 4)

    def forward(self, x):
        return self.fc(torch.flatten(self.conv(x), 1))


QUANT_LAYER_CONFIG = {
    'quant_enable': True,
    'activation_quant_params': {
        'act_algo': 'ifmr',
        'batch_num': 7,
        'num_bits': 8,
    },
    'weight_quant_params': {
        'wts_algo': 'arq',
        'channel_wise': True,
        'num_bits': 8,
    },
}

WEIGHT_ONLY_LAYER_CONFIG = {
    'quant_enable': True,
    'activation_quant_params': {
        'act_algo': 'ifmr',
        'batch_num': 7,
        'num_bits': 16,
    },
    'weight_quant_params': {
        'wts_algo': 'arq',
        'channel_wise': True,
        'num_bits': 8,
    },
}


def _fake_global_config(key):
    return {'enable_shift_value': 'auto', 'batch_num': 5}.get(key)


def _fake_layer_config_skip_all(name):
    return None


def _fake_layer_config_conv_quantized(name):
    return QUANT_LAYER_CONFIG if name == 'conv' else None


def _fake_layer_config_conv_weight_only(name):
    if name == 'conv':
        return WEIGHT_ONLY_LAYER_CONFIG
    return QUANT_LAYER_CONFIG


class TestInsertShiftCalibrationModulePass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(
            CUR_DIR, 'test_insert_shift_calibration_module_pass'
        )
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

        cls.model = TinyModel().to(torch.device('cpu'))
        cls.input_data = torch.randn(2, 3, 8, 8)
        cls.onnx_file = os.path.join(cls.temp_folder, 'tiny.onnx')
        Parser.export_onnx(cls.model, cls.input_data, cls.onnx_file)
        cls.graph = Parser.parse_net_to_graph(cls.onnx_file)
        cls.graph.add_model(cls.model)
        cls.record_file = os.path.join(cls.temp_folder, 'record.txt')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_folder, ignore_errors=True)

    def _run_pass(self):
        model = copy.deepcopy(self.model)
        optimizer = ModelOptimizer()
        optimizer.add_pass(InsertShiftCalibrationModulePass(self.record_file))
        optimizer.do_optimizer(model, self.graph)
        return model

    @patch.object(Configuration, 'get_layer_config')
    @patch.object(Configuration, 'get_global_config')
    def test_off_matches_nothing(self, mock_global, mock_layer):
        mock_global.side_effect = lambda key: {'enable_shift_value': 'off'}.get(key)
        mock_layer.return_value = QUANT_LAYER_CONFIG
        model = self._run_pass()
        named = dict(model.named_modules())
        self.assertNotIsInstance(named['conv'], ShiftCalibrationModule)
        self.assertNotIsInstance(named['fc'], ShiftCalibrationModule)

    @patch.object(Configuration, 'get_layer_config')
    @patch.object(Configuration, 'get_global_config')
    def test_all_wraps_every_target(self, mock_global, mock_layer):
        mock_global.side_effect = lambda key: {
            'enable_shift_value': 'all',
            'batch_num': 5,
        }.get(key)
        mock_layer.return_value = QUANT_LAYER_CONFIG
        model = self._run_pass()
        named = dict(model.named_modules())
        self.assertIsInstance(named['conv'], ShiftCalibrationModule)
        self.assertIsInstance(named['fc'], ShiftCalibrationModule)
        self.assertEqual(named['conv'].quant_regime, 'fp16_fp16')
        self.assertEqual(named['fc'].quant_regime, 'fp16_fp16')

    @patch.object(Configuration, 'get_layer_config')
    @patch.object(Configuration, 'get_global_config')
    def test_auto_all_layers_skipped(self, mock_global, mock_layer):
        """Layers absent from the quant config run the FP16 path."""
        mock_global.side_effect = _fake_global_config
        mock_layer.side_effect = _fake_layer_config_skip_all
        model = self._run_pass()
        named = dict(model.named_modules())
        self.assertIsInstance(named['conv'], ShiftCalibrationModule)
        self.assertIsInstance(named['fc'], ShiftCalibrationModule)
        self.assertEqual(named['conv'].quant_regime, 'fp16_fp16')
        self.assertEqual(named['fc'].quant_regime, 'fp16_fp16')

    @patch.object(Configuration, 'get_layer_config')
    @patch.object(Configuration, 'get_global_config')
    def test_auto_skip_and_quantized_mix(self, mock_global, mock_layer):
        """conv quantized (INT8 act), fc skipped: only fc is wrapped."""
        mock_global.side_effect = _fake_global_config
        mock_layer.side_effect = _fake_layer_config_conv_quantized
        model = self._run_pass()
        named = dict(model.named_modules())
        self.assertNotIsInstance(named['conv'], ShiftCalibrationModule)
        self.assertIsInstance(named['fc'], ShiftCalibrationModule)
        self.assertEqual(named['fc'].quant_regime, 'fp16_fp16')

    @patch.object(Configuration, 'get_layer_config')
    @patch.object(Configuration, 'get_global_config')
    def test_auto_weight_only_layer_fp16_int8(self, mock_global, mock_layer):
        """An act num_bits=16 layer runs the FP16*INT8 weight-only path."""
        mock_global.side_effect = _fake_global_config
        mock_layer.side_effect = _fake_layer_config_conv_weight_only
        model = self._run_pass()
        named = dict(model.named_modules())
        self.assertIsInstance(named['conv'], ShiftCalibrationModule)
        self.assertEqual(named['conv'].quant_regime, 'fp16_int8')
        self.assertNotIsInstance(named['fc'], ShiftCalibrationModule)

    @patch.object(Configuration, 'get_layer_config')
    @patch.object(Configuration, 'get_global_config')
    def test_batch_num_fallback_to_global(self, mock_global, mock_layer):
        """A skipped layer has no layer config: batch_num falls back."""
        mock_global.side_effect = _fake_global_config
        mock_layer.side_effect = _fake_layer_config_conv_quantized
        model = self._run_pass()
        named = dict(model.named_modules())
        self.assertEqual(named['fc'].batch_num, 5)

    @patch.object(Configuration, 'get_layer_config')
    @patch.object(Configuration, 'get_global_config')
    def test_batch_num_from_layer_config(self, mock_global, mock_layer):
        """A weight-only layer takes batch_num of its activation config."""
        mock_global.side_effect = _fake_global_config
        mock_layer.side_effect = _fake_layer_config_conv_weight_only
        model = self._run_pass()
        named = dict(model.named_modules())
        self.assertEqual(named['conv'].batch_num, 7)

    @patch.object(Configuration, 'get_layer_config')
    @patch.object(Configuration, 'get_global_config')
    def test_non_target_type_not_wrapped(self, mock_global, mock_layer):
        """A non FIXED_SHIFT_VALUE_TYPES module is never wrapped."""

        class MixedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(3)
                self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)

            def forward(self, x):
                return self.conv(self.bn(x))

        mock_global.side_effect = lambda key: {
            'enable_shift_value': 'all',
            'batch_num': 5,
        }.get(key)
        mock_layer.return_value = QUANT_LAYER_CONFIG

        model = MixedModel().to(torch.device('cpu'))
        input_data = torch.randn(2, 3, 8, 8)
        onnx_file = os.path.join(self.temp_folder, 'mixed.onnx')
        Parser.export_onnx(model, input_data, onnx_file)
        graph = Parser.parse_net_to_graph(onnx_file)
        graph.add_model(model)

        optimizer = ModelOptimizer()
        optimizer.add_pass(InsertShiftCalibrationModulePass(self.record_file))
        optimizer.do_optimizer(model, graph)
        named = dict(model.named_modules())
        self.assertIsInstance(named['conv'], ShiftCalibrationModule)
        self.assertNotIsInstance(named['bn'], ShiftCalibrationModule)


if __name__ == '__main__':
    unittest.main()
