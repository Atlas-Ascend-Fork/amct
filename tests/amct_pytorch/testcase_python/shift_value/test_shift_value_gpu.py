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
"""Accelerator tests of the shift value feature.

The CPU suites (test_shift_value_e2e / _robustness / _contract) run on
CPU only; this file covers the accelerator side:

- the search runs on the accelerator when the calibration model is
  on the accelerator; the fp64->fp16 cast then takes the torch two-step rounding path (see
  conv2d._to_fp16), so GPU determinism and value-domain checks
  apply, but NOT bit parity between CPU and GPU searched values;
- with fp16 inputs the engine math lives in the integer / float64
  domain, so the engine output must be bit-identical with the CPU
  engine output;
- the fp64->fp16 two-step rounding (accelerator) vs the numpy single
  rounding (CPU) must stay within 1 fp16 ulp on random samples (the
  200-seed experiment during development measured 20.5% argmin
  flips when the paths disagreed; keep the sampled equivalence).

Run on a machine with a GPU:

    CUDA_VISIBLE_DEVICES=0 python -m pytest \
        tests/amct_pytorch/testcase_python/shift_value/test_shift_value_gpu.py -v
"""

import os
import re
import shutil
import unittest

import pytest
import torch

import amct_pytorch
from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.fp16_fixed_point.conv2d import (
    Fp16FixedConv2d,
)

pytestmark = pytest.mark.gpu

CUR_DIR = os.path.realpath(os.path.dirname(__file__))


class TinyModel(torch.nn.Module):
    """conv(3->4, 3x3, pad=1) + fc(256->4), input (N, 3, 8, 8)."""

    def __init__(self):
        super(TinyModel, self).__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.fc = torch.nn.Linear(4 * 8 * 8, 4)

    def forward(self, x):
        return self.fc(torch.flatten(self.conv(x), 1))


def _make_conv2d(seed=0):
    """A Fp16FixedConv2d engine with deterministic random weights (CPU)."""
    torch.manual_seed(seed)
    module = torch.nn.Conv2d(3, 4, 3, padding=1)
    return Fp16FixedConv2d.from_module(module, fixed_shift_value=42)


@unittest.skipIf(not torch.cuda.is_available(), 'CUDA device not available')
class TestShiftValueGpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_folder = os.path.join(CUR_DIR, 'test_shift_value_gpu')
        if not os.path.isdir(cls.temp_folder):
            os.makedirs(cls.temp_folder)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_folder, ignore_errors=True)

    # ------------------------------------------------------------------
    # engine-level: with fp16 inputs the math is integer / float64, the
    # accelerator engine must be bit-identical with the CPU engine
    # ------------------------------------------------------------------
    def test_gpu_engine_bit_parity_with_cpu(self):
        engine = _make_conv2d(seed=7)
        torch.manual_seed(7)
        x = torch.randn(2, 3, 8, 8)

        with torch.no_grad():
            cpu_out = engine(x)
            gpu_out = engine.cuda()(x.cuda())

        self.assertEqual(gpu_out.device.type, 'cuda')
        self.assertEqual(cpu_out.dtype, gpu_out.dtype)
        self.assertTrue(
            torch.equal(cpu_out, gpu_out.cpu()),
            'accelerator engine output differs from the CPU engine bit pattern',
        )

    def test_gpu_to_fp16_rounding_within_one_ulp(self):
        """torch two-step cast (accelerator) vs numpy single cast (CPU) <= 1 ulp."""
        from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.fp16_fixed_point.conv2d import (
            _to_fp16_single_round,
        )

        torch.manual_seed(11)
        values = (
            torch.randn(4096, dtype=torch.float64)
            * 10 ** torch.randn(4096, dtype=torch.float64)
            * 4
        )

        cpu_cast = _to_fp16_single_round(values)
        gpu_cast = _to_fp16_single_round(values.cuda())

        self.assertEqual(gpu_cast.device.type, 'cuda')
        self.assertEqual(cpu_cast.dtype, torch.float16)
        diff = (cpu_cast.cuda().view(torch.int16) - gpu_cast.view(torch.int16)).abs()
        # the two-step cast may land 1 fp16 ulp off the single cast
        self.assertTrue(
            bool((diff <= 1).all()),
            f'max bit-pattern distance {int(diff.max())} > 1 ulp',
        )

    # ------------------------------------------------------------------
    # search-level: calibration on the accelerator (device.type !=
    # 'cpu' -> torch cast),
    # determinism and the value domain apply, not CPU bit parity
    # ------------------------------------------------------------------
    def _run_flow_on_gpu(self, tag, simple_cfg_text, batches=1):
        """CPU config/export + accelerator calibration forward (search on device)."""
        work_dir = os.path.join(self.temp_folder, tag)
        if not os.path.isdir(work_dir):
            os.makedirs(work_dir)

        torch.manual_seed(1234)
        model = TinyModel().eval()
        torch.manual_seed(1234)
        input_data = torch.randn(2, 3, 8, 8)
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
        calib_model.eval().cuda()
        with torch.no_grad():
            for _ in range(batches):
                calib_model(input_data.cuda())

        with open(record_file) as fid:
            record_text = fid.read()

        amct_pytorch.save_model(
            modfied_onnx_file=modified_onnx,
            record_file=record_file,
            save_path=os.path.join(work_dir, tag),
        )
        return record_text

    @staticmethod
    def _fc_shift_value(record_text):
        match = re.search(r'key: "fc".*?fixed_shift_value: (\d+)', record_text, re.S)
        assert match is not None, 'no fixed_shift_value for fc in record'
        return int(match.group(1))

    def test_gpu_fp16_search_deterministic(self):
        """Two identical accelerator runs must give the identical searched value."""
        cfg = (
            'batch_num: 1\nenable_shift_value: "auto"\nskip_layers: "fc"\n'
            'common_config {\n}\n'
        )
        first = self._run_flow_on_gpu('gpu_det_run1', cfg)
        second = self._run_flow_on_gpu('gpu_det_run2', cfg)

        value_first = self._fc_shift_value(first)
        value_second = self._fc_shift_value(second)
        self.assertIn(value_first, range(34, 44))
        self.assertEqual(value_first, value_second)

    def test_gpu_int8_search_deterministic(self):
        """Two identical accelerator runs on the weight-only INT8 path.

        The fp16_int8 entry is weight-only: the layer's calibration_config
        carries a wts algo and NO act algo (the activation stays
        unquantized); the former act INT16 + wts INT8 combination is
        activation-quantized and no longer joins the sv search."""
        cfg = (
            'batch_num: 1\nenable_shift_value: "auto"\n'
            'override_layer_configs {\n  layer_name: "conv"\n'
            '  calibration_config {\n    ada_quantize { num_iteration: 100 }\n'
            '  }\n}\ncommon_config {\n}\n'
        )
        first = self._run_flow_on_gpu('gpu_int8_run1', cfg)
        second = self._run_flow_on_gpu('gpu_int8_run2', cfg)

        match_first = re.search(r'key: "conv".*?fixed_shift_value: (\d+)', first, re.S)
        match_second = re.search(
            r'key: "conv".*?fixed_shift_value: (\d+)', second, re.S
        )
        self.assertIsNotNone(match_first)
        self.assertIsNotNone(match_second)
        # the fp16_int8 search is disabled (board kernel pins sv=13):
        # both runs carry the regime default directly
        value_first = int(match_first.group(1))
        self.assertEqual(value_first, 13)
        self.assertEqual(value_first, int(match_second.group(1)))

    def test_gpu_conv2d_fp16_search_flow(self):
        """The conv2d fp16 search also runs on the accelerator (skip conv, not fc)."""
        cfg = (
            'batch_num: 1\nenable_shift_value: "auto"\nskip_layers: "conv"\n'
            'common_config {\n}\n'
        )
        record_text = self._run_flow_on_gpu('gpu_conv_search', cfg)
        match = re.search(r'key: "conv".*?fixed_shift_value: (\d+)', record_text, re.S)
        self.assertIsNotNone(match)
        self.assertIn(int(match.group(1)), range(34, 44))
