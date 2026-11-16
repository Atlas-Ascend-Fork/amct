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
"""Contract tests pinning the documented numeric / behaviour contracts.

Covers the design-doc contracts that had no dedicated test after the
precision-first single-path rework:

- ConvTranspose over-padding (> dilated kernel - 1) falls back to the
  default fixed_shift_value instead of crashing the calibration (design
  doc 10.1, commit d5325c4);
- the S8 operands are registered buffers so .to(device) moves them
  together with the parameters (design doc C2, commit 37cd112);
- the bias type-chain contract (design doc 8.1.1): the fp16 round-trip
  is load-bearing (dropping it changes the output), and an integer
  carrier (int64 MAC + int32 saturation) is bit-exact with the float64
  engine accumulator.
"""

import os
import re
import tempfile
import unittest

import pytest

import torch

from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.fp16_fixed_point import (
    Fp16FixedLinearS8,
)

from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.fp16_fixed_point.matmul import (
    _bias_to_int64,
    f16_matmul_f16,
    int32_dequant_to_fp16,
)
from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.quantization.shift_calibration import (  # noqa: E501
    ShiftCalibrationModule,
)

pytestmark = pytest.mark.cpu


class TestConvTransposeOverPaddingFallback(unittest.TestCase):
    """Over-padding ConvT falls back to the default value (no crash)."""

    def setUp(self):
        """Reset the process-level record singleton so the test only
        sees its own entries (other tests in the same process may have
        left theirs in the shared singleton)."""
        from amct_pytorch.classic.graph_based.amct_pytorch.utils.singleton_record import (
            SingletonScaleOffsetRecord,
        )

        SingletonScaleOffsetRecord().reset_record()

    def test_over_padding_writes_default_not_crash(self):
        # k=3, dilation=1 -> max legal padding is 2; padding=3 exceeds it
        module = torch.nn.ConvTranspose2d(4, 6, 3, stride=1, padding=3)
        with tempfile.TemporaryDirectory() as work_dir:
            record = os.path.join(work_dir, 'record.txt')
            wrapper = ShiftCalibrationModule(
                module,
                'ct_over_pad',
                record,
                batch_num=1,
                quant_regime='fp16_fp16',
            )
            x = torch.randn(1, 4, 8, 8)
            with torch.no_grad():
                wrapper(x)  # must not raise
            record_text = open(record).read()
            match = re.search(
                r'key: "ct_over_pad".*?fixed_shift_value: (\d+)', record_text, re.S
            )
            self.assertIsNotNone(match, 'record must carry the value')
            self.assertEqual(int(match.group(1)), 42)  # regime default

    def test_legal_padding_still_searches(self):
        # same kernel with legal padding really searches (no regression
        # of the guard: it must only trigger on over-padding)
        module = torch.nn.ConvTranspose2d(4, 6, 3, stride=1, padding=2)
        with tempfile.TemporaryDirectory() as work_dir:
            record = os.path.join(work_dir, 'record.txt')
            wrapper = ShiftCalibrationModule(
                module,
                'ct_ok_pad',
                record,
                batch_num=1,
                quant_regime='fp16_fp16',
            )
            with torch.no_grad():
                wrapper(torch.randn(1, 4, 8, 8))
            record_text = open(record).read()
            value = int(
                re.search(
                    r'key: "ct_ok_pad".*?fixed_shift_value: (\d+)', record_text, re.S
                ).group(1)
            )
            self.assertIn(value, range(34, 44))


class TestS8OperandsAreBuffers(unittest.TestCase):
    """int8_weight / weight_scale move with .to(device) (register_buffer)."""

    def test_buffers_registered_and_moved(self):
        torch.manual_seed(31)
        module = torch.nn.Linear(64, 32)
        weight_fp16 = module.weight.detach().to(torch.float16)
        scale = (weight_fp16.abs().amax(dim=1) / 127.0).clamp(min=1e-8)
        int8_weight = (
            torch.round(weight_fp16 / scale.view(-1, 1)).clamp(-127, 127).to(torch.int8)
        )

        fp16fixed = Fp16FixedLinearS8.from_module(module, int8_weight, scale)
        self.assertIn('int8_weight', fp16fixed._buffers)
        self.assertIn('weight_scale', fp16fixed._buffers)
        # a .to() round trip keeps them attached to the module
        moved = fp16fixed.to(torch.device('cpu'))
        self.assertTrue(torch.equal(moved.int8_weight, fp16fixed.int8_weight))


class TestBiasTypeChainContract(unittest.TestCase):
    """Design doc 8.1.1: the fp16 round-trip is load-bearing; an integer
    carrier is bit-exact with the float64 engine."""

    def _engine(self, a, w, bias, sv, bias_roundtrip):
        if bias_roundtrip:
            bias_int = _bias_to_int64(bias, w.shape[0], 1 << (58 - sv))
        else:
            # variant without the round-trip: raw fp32 -> f64 scaling
            scaled = torch.round(bias.to(torch.float64) * float(1 << (58 - sv))).to(
                torch.int64
            )
            bias_int = scaled
        m, k = a.shape
        n = w.shape[0]
        # drive the engine with the pre-scaled seed via a wrapper engine:
        # reuse f16_matmul_f16 for the roundtrip case and compare against
        # a manual pipeline for the no-roundtrip case
        if bias_roundtrip:
            return f16_matmul_f16(a, w, bias, sv)
        # manual: same pipeline but bias seed without round-trip
        from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.fp16_fixed_point.matmul import (  # noqa: E502
            _split_fp16_bits,
            _to_fixed_point_all_tiles,
            _scale_by_shift,
            INT32_MAX,
            INT32_MIN,
        )

        k_pad = (k + 15) // 16 * 16
        if k_pad != k:
            a = torch.nn.functional.pad(a, (0, k_pad - k))
            w = torch.nn.functional.pad(w, (0, k_pad - k))
        num_tiles = k_pad // 16
        a_sign, a_exp, a_man = _split_fp16_bits(a)
        w_sign, w_exp, w_man = _split_fp16_bits(w)
        a_nrm, w_nrm = a_exp > 0, w_exp > 0
        a_man = torch.where(a_nrm, a_man + 1024, a_man).to(torch.int64)
        w_man = torch.where(w_nrm, w_man + 1024, w_man).to(torch.int64)
        a_fixed, a_tmax = _to_fixed_point_all_tiles(
            a_exp, a_man, a_nrm, a_sign, num_tiles
        )
        w_fixed, w_tmax = _to_fixed_point_all_tiles(
            w_exp, w_man, w_nrm, w_sign, num_tiles
        )
        acc = bias_int.to(torch.float64).reshape(1, n).repeat(m, 1)
        for t in range(num_tiles):
            cols = slice(t * 16, (t + 1) * 16)
            mac = a_fixed[:, cols] @ w_fixed[:, cols].T
            shift = (sv - a_tmax[:, t].unsqueeze(1) - w_tmax[:, t].unsqueeze(0)).clamp(
                -26, 36
            )
            acc = torch.round(acc + _scale_by_shift(mac, shift)).clamp(
                INT32_MIN, INT32_MAX
            )
        return acc.to(torch.int32)

    def test_roundtrip_is_load_bearing(self):
        """Dropping the fp16 round-trip changes the fp16 output."""
        torch.manual_seed(41)
        a = torch.randn(16, 128).to(torch.float16)
        w = torch.randn(24, 128).to(torch.float16)
        bias = torch.randn(24) * 0.7  # values with fp16-representable loss
        with_rt = int32_dequant_to_fp16(self._engine(a, w, bias, 42, True), 42)
        no_rt = int32_dequant_to_fp16(self._engine(a, w, bias, 42, False), 42)
        self.assertFalse(
            torch.equal(with_rt, no_rt),
            'the fp16 round-trip must influence the output for the '
            'documented contract to be load-bearing',
        )

    def test_integer_carrier_bit_exact(self):
        """int64 MAC + int32 saturation == float64 engine (contract)."""
        torch.manual_seed(43)
        a = torch.randn(12, 96).to(torch.float16)
        w = torch.randn(16, 96).to(torch.float16)
        sv = 42
        m, k = a.shape
        n = w.shape[0]
        k_pad = (k + 15) // 16 * 16
        if k_pad != k:
            a = torch.nn.functional.pad(a, (0, k_pad - k))
            w = torch.nn.functional.pad(w, (0, k_pad - k))
        num_tiles = k_pad // 16
        from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.fp16_fixed_point.matmul import (  # noqa: E502
            _split_fp16_bits,
            _to_fixed_point_all_tiles,
        )

        a_sign, a_exp, a_man = _split_fp16_bits(a)
        w_sign, w_exp, w_man = _split_fp16_bits(w)
        # NOTE: _to_fixed_point_all_tiles takes the RAW mantissa and the
        # normal flag (it restores the implicit leading 1 internally)
        a_fixed, a_tmax = _to_fixed_point_all_tiles(
            a_exp, a_man, a_exp > 0, a_sign, num_tiles
        )
        w_fixed, w_tmax = _to_fixed_point_all_tiles(
            w_exp, w_man, w_exp > 0, w_sign, num_tiles
        )
        bias_int = _bias_to_int64(None, n, 1 << (58 - sv), device=a.device)
        # integer carrier: exact int64 tile MAC, integer half-even shift
        # onto the grid, int32 saturation
        acc = bias_int.clone().reshape(1, n).repeat(m, 1)
        for t in range(num_tiles):
            cols = slice(t * 16, (t + 1) * 16)
            mac = (a_fixed[:, cols] @ w_fixed[:, cols].T).to(torch.int64)
            shift = (sv - a_tmax[:, t].unsqueeze(1) - w_tmax[:, t].unsqueeze(0)).clamp(
                -26, 36
            )

            # integer half-even: positive shift rounds half to even,
            # negative shift is an exact left shift
            def _int_round(x, s):
                pow2 = torch.ones_like(x) << s.clamp_min(0)
                rem = x & (pow2 - 1)
                half = pow2 >> 1
                lsb = (x >> s.clamp_min(0)) & 1
                carry = (rem > half) | ((rem == half) & (lsb == 1))
                return torch.where(
                    s >= 0, (x >> s.clamp_min(0)) + carry, x << (-s).clamp_min(0)
                )

            # bit-exact integer form of round(f64(acc) + mac * 2^-s):
            # round((acc << s) + mac) >> s -- the whole addition happens
            # on the fine grid before one half-even step back, exactly
            # like the float64 engine folds acc + scaled and rounds once
            contribution = _int_round(
                (acc << shift.clamp_min(0)) + (mac << (-shift).clamp_min(0)), shift
            )
            acc = torch.clamp(contribution, -(2**31), 2**31 - 1)
        engine = f16_matmul_f16(
            torch.randn(1).to(torch.float16) * 0 + a[:, :k],  # keep dtype
            torch.randn(1).to(torch.float16) * 0 + w[:, :k],
            None,
            sv,
        )
        # rebuild the engine accumulator for the same bias seed
        self.assertTrue(
            torch.equal(acc.to(torch.int32), engine),
            'integer carrier (int64 MAC + int32 saturation) must be '
            'bit-exact with the float64 engine',
        )


if __name__ == '__main__':
    unittest.main()
