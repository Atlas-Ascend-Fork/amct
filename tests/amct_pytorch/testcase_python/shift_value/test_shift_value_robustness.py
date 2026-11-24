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
"""Robustness tests of the shift value feature.

Covers the gaps left by the scenario/exception suites (see
test_shift_value_e2e.py S1-S12):
- determinism: the same seed must give bit-identical search results
  across two full flows (fp16_fp16 and fp16_int8 regimes);
- corrupted record files: truncated / garbage / empty records must
  raise a clear RuntimeError from save_model instead of silently
  producing a deploy model;
- minimal spatial input: a (1, 3, 4, 4) input with batch_num=1 still
  completes the search and the deploy export;
- AdaRound + Linear: the fp16_int8 search also works when the
  weight-only layer is an AdaRound Linear (S6 only covers ConvT).
"""

import os
import re
import shutil
import unittest

import pytest
import torch

import amct_pytorch
from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.quantization.shift_calibration import (
    SingletonScaleOffsetRecord,
)

pytestmark = pytest.mark.cpu

CUR_DIR = os.path.realpath(os.path.dirname(__file__))


class TinyModel(torch.nn.Module):
    """conv(3->4, 3x3, pad=1) + fc(256->4), input (N, 3, 8, 8)."""

    def __init__(self):
        super(TinyModel, self).__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.fc = torch.nn.Linear(4 * 8 * 8, 4)

    def forward(self, x):
        return self.fc(torch.flatten(self.conv(x), 1))


class TinyModel4x4(torch.nn.Module):
    """Minimal spatial dims: conv(3->4, 3x3, pad=1) + fc(64->4), input (1, 3, 4, 4)."""

    def __init__(self):
        super(TinyModel4x4, self).__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.fc = torch.nn.Linear(4 * 4 * 4, 4)

    def forward(self, x):
        return self.fc(torch.flatten(self.conv(x), 1))


def _deploy_attrs(onnx_file):
    attrs = {}
    import onnx

    model = onnx.load(onnx_file)
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.name == 'fixed_shift_value':
                attrs[node.name] = attr.i
    return attrs


def _deploy_int8_weight(onnx_file, node_name):
    """The int8 weight initializer of the deploy node (torch layout)."""
    import onnx

    model = onnx.load(onnx_file)
    weight_name = None
    for node in model.graph.node:
        if node.name == node_name:
            weight_name = node.input[1]
            break
    if weight_name is None:
        return None
    for initializer in model.graph.initializer:
        if initializer.name == weight_name:
            return torch.from_numpy(onnx.numpy_helper.to_array(initializer))
    return None


class TestShiftValueRobustness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_shift_value_robustness')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_folder, ignore_errors=True)

    def setUp(self):
        SingletonScaleOffsetRecord().reset_record()

    def _run_flow(self, tag, simple_cfg_text, model=None, input_shape=(2, 3, 8, 8)):
        """One full PTQ flow; the RNG is reseeded so two calls are comparable."""
        work_dir = os.path.join(self.temp_folder, tag)
        if not os.path.isdir(work_dir):
            os.makedirs(work_dir)

        torch.manual_seed(1234)
        model = TinyModel().eval() if model is None else model.eval()
        torch.manual_seed(1234)
        input_data = torch.randn(*input_shape)
        simple_cfg = os.path.join(work_dir, 'simple.cfg')
        with open(simple_cfg, 'w') as fid:
            fid.write(simple_cfg_text)
        config_file = os.path.join(work_dir, 'config.json')
        amct_pytorch.create_quant_config(
            config_file=config_file,
            model=model,
            input_data=input_data,
            config_defination=simple_cfg,
        )

        record_file = os.path.join(work_dir, 'record.txt')
        modified_onnx = os.path.join(work_dir, 'modified.onnx')
        calib_model = amct_pytorch.quantize_model(
            config_file=config_file,
            modfied_onnx_file=modified_onnx,
            record_file=record_file,
            model=model,
            input_data=input_data,
        )
        calib_model.eval()
        with torch.no_grad():
            calib_model(input_data)

        with open(record_file) as fid:
            record_text = fid.read()

        amct_pytorch.save_model(
            modfied_onnx_file=modified_onnx,
            record_file=record_file,
            save_path=os.path.join(work_dir, tag),
        )
        deploy_file = os.path.join(work_dir, tag + '_deploy_model.onnx')
        return {
            'calib_model': calib_model,
            'record_text': record_text,
            'deploy_attrs': _deploy_attrs(deploy_file),
            'deploy_file': deploy_file,
        }

    @staticmethod
    def _shift_values(record_text):
        """Every (layer, fixed_shift_value) pair in the record, in file order."""
        return re.findall(r'key: "(\w+)".*?fixed_shift_value: (\d+)', record_text, re.S)

    # ------------------------------------------------------------------
    # determinism
    # ------------------------------------------------------------------
    def test_fp16_flow_bit_deterministic(self):
        """Same seed twice: identical searched values and deploy attrs.

        In auto mode only the skipped layer stays on the unquantized
        fp16_fp16 path and is searched (the S1 semantics).
        """
        cfg = (
            'batch_num: 1\nenable_shift_value: "auto"\nskip_layers: "fc"\n'
            'common_config {\n}\n'
        )
        first = self._run_flow('det_fp16_run1', cfg)
        second = self._run_flow('det_fp16_run2', cfg)

        values_first = self._shift_values(first['record_text'])
        values_second = self._shift_values(second['record_text'])
        self.assertNotEqual(values_first, [])
        self.assertEqual(values_first, values_second)
        self.assertEqual(first['deploy_attrs'], second['deploy_attrs'])

    def test_int8_flow_bit_deterministic(self):
        """Same seed twice on the weight-only INT8 path: identical searched value."""
        cfg = (
            'batch_num: 1\nenable_shift_value: "auto"\n'
            'override_layer_configs {\n  layer_name: "conv"\n'
            '  calibration_config {\n'
            '    arq_quantize { }\n  }\n}\ncommon_config {\n}\n'
        )
        first = self._run_flow('det_int8_run1', cfg)
        second = self._run_flow('det_int8_run2', cfg)

        def conv_value(result):
            match = re.search(
                r'key: "conv".*?fixed_shift_value: (\d+)', result['record_text'], re.S
            )
            self.assertIsNotNone(match)
            return int(match.group(1))

        first_value = conv_value(first)
        # fp16_int8 does not search: the pinned regime default is written
        self.assertEqual(first_value, 13)
        self.assertEqual(conv_value(second), first_value)
        self.assertEqual(first['deploy_attrs'], second['deploy_attrs'])

    # ------------------------------------------------------------------
    # corrupted record files
    # ------------------------------------------------------------------
    def _corrupt_then_save(self, tag, corrupted_text):
        cfg = 'batch_num: 1\nenable_shift_value: "auto"\ncommon_config {\n}\n'
        result = self._run_flow(tag, cfg)
        record_file = os.path.join(self.temp_folder, tag, 'record.txt')
        modified_onnx = os.path.join(self.temp_folder, tag, 'modified.onnx')
        with open(record_file, 'w') as fid:
            fid.write(corrupted_text(result['record_text']))
        with self.assertRaises(RuntimeError):
            amct_pytorch.save_model(
                modfied_onnx_file=modified_onnx,
                record_file=record_file,
                save_path=os.path.join(self.temp_folder, tag, 'save'),
            )

    def test_truncated_record_raises(self):
        """A half-written record must fail loudly, not deploy silently."""

        def truncate(text):
            return text[: len(text) // 2]

        self._corrupt_then_save('corrupt_trunc', truncate)

    def test_garbage_record_raises(self):
        """Non-proto garbage in the record must fail loudly."""

        def garbage(_text):
            return '\x00\x01garbage not proto \xff\xfe\n'

        self._corrupt_then_save('corrupt_garbage', garbage)

    def test_empty_record_raises(self):
        """An empty record (calibration never finished) must fail loudly."""

        def empty(_text):
            return ''

        self._corrupt_then_save('corrupt_empty', empty)

    # ------------------------------------------------------------------
    # minimal input boundary
    # ------------------------------------------------------------------
    def test_minimal_spatial_input_batch_one(self):
        """(1, 3, 4, 4) input with batch_num=1 still searches and deploys."""
        result = self._run_flow(
            'tiny_input',
            'batch_num: 1\nenable_shift_value: "auto"\nskip_layers: "fc"\n'
            'common_config {\n}\n',
            model=TinyModel4x4(),
            input_shape=(1, 3, 4, 4),
        )
        record_text = result['record_text']
        match = re.search(r'key: "fc".*?fixed_shift_value: (\d+)', record_text, re.S)
        self.assertIsNotNone(match)
        self.assertIn(int(match.group(1)), range(34, 44))
        self.assertIn('fc', result['deploy_attrs'])

    # ------------------------------------------------------------------
    # AdaRound + Linear (S6 only covers ConvTranspose2d)
    # ------------------------------------------------------------------
    def test_adaround_linear_searched(self):
        """Weight-only AdaRound Linear: fp16_int8 search + bit-exact rebuild."""
        from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.quantization import (
            shift_value_kernel,
        )

        tag = 'ada_linear'
        result = self._run_flow(
            tag,
            'batch_num: 1\nenable_shift_value: "auto"\n'
            'override_layer_configs {\n  layer_name: "fc"\n'
            '  calibration_config {\n'
            '    ada_quantize { num_iteration: 100 }\n  }\n}\ncommon_config {\n}\n',
        )
        record_text = result['record_text']

        match = re.search(r'key: "fc".*?fixed_shift_value: (\d+)', record_text, re.S)
        self.assertIsNotNone(match)
        fc_shift_value = int(match.group(1))
        # fp16_int8 does not search: the pinned regime default is written
        self.assertEqual(fc_shift_value, 13)
        self.assertEqual(result['deploy_attrs'], {'fc': fc_shift_value})

        # weight-only: the module IS the bare ShiftCalibrationModule
        shift_module = result['calib_model'].fc
        self.assertEqual(shift_module.quant_regime, 'fp16_int8')
        weight_quant_info = shift_module._get_weight_quant_info()
        self.assertIsNotNone(weight_quant_info)
        rebuilt, _ = shift_value_kernel._rebuild_int8_weight(
            shift_module.object_module, weight_quant_info
        )
        deploy_weight = _deploy_int8_weight(result['deploy_file'], 'fc')
        self.assertIsNotNone(deploy_weight)
        self.assertEqual(deploy_weight.dtype, torch.int8)
        # the Linear deploys as Gemm: the ONNX initializer carries the
        # transposed (in, out) layout, compare after transposing back
        self.assertEqual(tuple(deploy_weight.shape), (256, 4))
        self.assertTrue(
            torch.equal(deploy_weight.t(), rebuilt),
            'the rebuilt int8 weight differs from the deploy weight',
        )
