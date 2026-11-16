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
"""Tests of the Fp16FixedLinear class: reference parity, search and multi batch."""

import unittest

import pytest

import torch

from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.fp16_fixed_point import (
    Fp16FixedLinear,
)
from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.fp16_fixed_point.matmul import (
    f16_matmul_f16,
    int32_dequant_to_fp16,
)

pytestmark = [pytest.mark.cpu, pytest.mark.slow]


def _make_module(use_bias=True):
    module = torch.nn.Linear(64, 32, bias=use_bias)
    module.weight.data = (
        (torch.randn_like(module.weight) * 0.1).to(torch.float16).to(torch.float32)
    )
    if use_bias:
        module.bias.data = (
            (torch.randn_like(module.bias) * 0.1).to(torch.float16).to(torch.float32)
        )
    return module


class TestFp16FixedLinear(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        self.x = (torch.randn(5, 64) * 0.5).to(torch.float16)

    def _functional_forward(self, module, x, sv):
        acc = f16_matmul_f16(x, module.weight.to(torch.float16), module.bias, sv)
        return int32_dequant_to_fp16(acc, sv)

    def test_bit_parity_with_functional(self):
        """forward equals f16_matmul_f16 + dequant, incl. 3D and no bias."""
        for use_bias in (True, False):
            module = _make_module(use_bias)
            fp16fixed = Fp16FixedLinear.from_module(module, 42)
            self.assertTrue(
                torch.equal(
                    fp16fixed(self.x), self._functional_forward(module, self.x, 42)
                ),
                'parity failed on bias={}'.format(use_bias),
            )

    def test_bit_parity_3d_input(self):
        module = _make_module()
        fp16fixed = Fp16FixedLinear.from_module(module, 42)
        x3 = (torch.randn(2, 5, 64) * 0.5).to(torch.float16)
        out = fp16fixed(x3)
        self.assertEqual(tuple(out.shape), (2, 5, 32))
        expect = self._functional_forward(module, x3, 42)
        self.assertTrue(torch.equal(out, expect))

    def test_from_module_copies_state_dict(self):
        module = _make_module()
        fp16fixed = Fp16FixedLinear.from_module(module, 39)
        self.assertIsInstance(fp16fixed, torch.nn.Linear)
        self.assertTrue(torch.equal(fp16fixed.weight.data, module.weight.data))
        self.assertTrue(torch.equal(fp16fixed.bias.data, module.bias.data))
        self.assertEqual(fp16fixed.fixed_shift_value, 39)

    def test_direct_constructor(self):
        fp16fixed = Fp16FixedLinear(64, 32, bias=False, fixed_shift_value=37)
        self.assertIsNone(fp16fixed.bias)
        out = fp16fixed(self.x)
        self.assertEqual(out.dtype, torch.float16)
        self.assertEqual(tuple(out.shape), (5, 32))

    def test_fp32_input_cast_to_fp16(self):
        module = _make_module()
        fp16fixed = Fp16FixedLinear.from_module(module, 42)
        self.assertTrue(
            torch.equal(fp16fixed(self.x.to(torch.float32)), fp16fixed(self.x))
        )

    def test_search_returns_argmin_and_writes_back(self):
        module = _make_module()
        fp16fixed = Fp16FixedLinear.from_module(module, 42)
        sse = fp16fixed.compute_candidate_sse(self.x)
        self.assertEqual(tuple(sse.shape), (10,))
        best = fp16fixed.search_fixed_shift_value(self.x)
        self.assertIn(best, range(34, 44))
        self.assertEqual(best, fp16fixed.candidates[int(torch.argmin(sse).item())])
        out = fp16fixed(self.x)
        expect = fp16fixed._forward_impl(self.x, best)
        self.assertTrue(torch.equal(out, expect))

    def test_multi_batch_equals_single_shot(self):
        module = _make_module()
        batches = [(torch.randn(4, 64) * 0.5).to(torch.float16) for _ in range(3)]
        fp16fixed = Fp16FixedLinear.from_module(module, 42)
        fp16fixed.start_calibration()
        for xb in batches:
            fp16fixed.accumulate_sse(xb)
        multi_sv = fp16fixed.finalize()

        single = Fp16FixedLinear.from_module(module, 42)
        total = sum(single.compute_candidate_sse(xb) for xb in batches)
        self.assertEqual(multi_sv, single.candidates[int(torch.argmin(total).item())])

    def test_finalize_resets_sse_state(self):
        """A second calibration session after finalize() starts fresh."""
        module = _make_module()
        b1 = (torch.randn(4, 64) * 0.5).to(torch.float16)
        b2 = (torch.randn(4, 64) * 0.5).to(torch.float16)
        fp16fixed = Fp16FixedLinear.from_module(module, 42)
        fp16fixed.start_calibration()
        fp16fixed.accumulate_sse(b1)
        fp16fixed.finalize()
        self.assertIsNone(getattr(fp16fixed, '_sse_sum', None))

        fp16fixed.start_calibration()
        fp16fixed.accumulate_sse(b1)
        fp16fixed.accumulate_sse(b2)
        second_sv = fp16fixed.finalize()
        single = Fp16FixedLinear.from_module(module, 42)
        total = single.compute_candidate_sse(b1) + single.compute_candidate_sse(b2)
        self.assertEqual(second_sv, single.candidates[int(torch.argmin(total).item())])

    def test_precision_against_fp32(self):
        module = _make_module()
        fp16fixed = Fp16FixedLinear.from_module(module, 42)
        out = fp16fixed(self.x)
        ref = torch.nn.functional.linear(
            self.x.float(), module.weight.float(), module.bias.float()
        )
        rel = (out.float() - ref).abs() / (ref.abs() + 1e-2)
        self.assertLess(rel.median().item(), 1e-2)
        self.assertLess(rel.max().item(), 5e-2)


if __name__ == '__main__':
    unittest.main()
