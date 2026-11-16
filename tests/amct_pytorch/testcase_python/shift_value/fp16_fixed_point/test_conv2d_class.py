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
"""Tests of the Fp16FixedConv2d class: legacy parity, search and multi batch."""

import unittest

import pytest

import torch

from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.fp16_fixed_point import (
    Fp16FixedConv2d,
)
from .reference.conv2d import (
    conv_f16f16 as legacy_conv_f16f16,
)

pytestmark = [pytest.mark.cpu, pytest.mark.slow]


def _make_module(stride, padding, dilation, groups, use_bias):
    module = torch.nn.Conv2d(
        8,
        16,
        3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=use_bias,
    )
    module.weight.data = (
        (torch.randn_like(module.weight) * 0.1).to(torch.float16).to(torch.float32)
    )
    if use_bias:
        module.bias.data = (
            (torch.randn_like(module.bias) * 0.1).to(torch.float16).to(torch.float32)
        )
    return module


class TestFp16FixedConv2d(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        self.x = (torch.randn(2, 8, 8, 8) * 0.5).to(torch.float16)

    def _legacy_forward(self, module, x, sv):
        padding = module.padding
        if isinstance(padding, int):
            padding = (padding, padding)
        pads = (padding[0], padding[0], padding[1], padding[1])
        return legacy_conv_f16f16(
            x,
            module.weight.to(torch.float16),
            module.bias,
            pads,
            tuple(module.stride),
            tuple(module.dilation),
            module.groups,
            sv,
        )

    def test_bit_parity_with_legacy(self):
        """forward equals the legacy reference on the parameter matrix."""
        combos = [
            (1, 1, 1, 1, True),
            (1, 0, 1, 1, False),
            (2, 1, 1, 1, True),
            (1, 2, 2, 1, True),
            (1, 1, 1, 2, True),
            (1, 1, 1, 4, False),
            (1, 1, 2, 2, True),
        ]
        for stride, padding, dilation, groups, use_bias in combos:
            module = _make_module(stride, padding, dilation, groups, use_bias)
            fp16fixed = Fp16FixedConv2d.from_module(module, 42)
            out = fp16fixed(self.x)
            expect = self._legacy_forward(module, self.x, 42)
            self.assertTrue(
                torch.equal(out, expect),
                'parity failed on {}'.format(
                    (stride, padding, dilation, groups, use_bias)
                ),
            )

    def test_from_module_copies_state_dict(self):
        module = _make_module(2, 1, 1, 1, True)
        fp16fixed = Fp16FixedConv2d.from_module(module, 40)
        self.assertTrue(torch.equal(fp16fixed.weight.data, module.weight.data))
        self.assertTrue(torch.equal(fp16fixed.bias.data, module.bias.data))
        self.assertEqual(fp16fixed.stride, module.stride)
        self.assertEqual(fp16fixed.padding, module.padding)
        self.assertEqual(fp16fixed.groups, module.groups)
        self.assertEqual(fp16fixed.fixed_shift_value, 40)
        # the source module is untouched
        self.assertIsInstance(module, torch.nn.Conv2d)

    def test_direct_constructor(self):
        fp16fixed = Fp16FixedConv2d(8, 16, 3, stride=2, padding=1, fixed_shift_value=38)
        self.assertIsInstance(fp16fixed, torch.nn.Conv2d)
        self.assertEqual(fp16fixed.fixed_shift_value, 38)
        out = fp16fixed(self.x)
        self.assertEqual(out.dtype, torch.float16)
        self.assertEqual(tuple(out.shape), (2, 16, 4, 4))

    def test_fp32_input_cast_to_fp16(self):
        """An fp32 input is cast onto the fp16 operator path."""
        module = _make_module(1, 1, 1, 1, True)
        fp16fixed = Fp16FixedConv2d.from_module(module, 42)
        out_fp32 = fp16fixed(self.x.to(torch.float32))
        out_fp16 = fp16fixed(self.x)
        self.assertTrue(torch.equal(out_fp32, out_fp16))

    def test_tuple_padding(self):
        """_pads() handles the (pad_h, pad_w) tuple form."""
        module = _make_module(1, (1, 2), 1, 1, True)
        fp16fixed = Fp16FixedConv2d.from_module(module, 42)
        out = fp16fixed(self.x)
        self.assertEqual(tuple(out.shape), (2, 16, 8, 10))
        expect = self._legacy_forward(module, self.x, 42)
        self.assertTrue(torch.equal(out, expect))
        ref = torch.nn.functional.conv2d(
            self.x.float(),
            module.weight.float(),
            bias=module.bias.float(),
            stride=module.stride,
            padding=module.padding,
        )
        rel = (out.float() - ref).abs() / (ref.abs() + 1e-2)
        self.assertLess(rel.median().item(), 1e-2)
        self.assertLess(rel.max().item(), 5e-2)

    def test_search_returns_argmin_and_writes_back(self):
        module = _make_module(1, 1, 1, 1, True)
        fp16fixed = Fp16FixedConv2d.from_module(module, 42)
        sse = fp16fixed.compute_candidate_sse(self.x)
        self.assertEqual(tuple(sse.shape), (10,))
        best = fp16fixed.search_fixed_shift_value(self.x)
        self.assertIn(best, range(34, 44))
        self.assertEqual(best, fp16fixed.candidates[int(torch.argmin(sse).item())])
        # written back: forward now runs with the searched value
        out = fp16fixed(self.x)
        expect = fp16fixed._forward_impl(self.x, best)
        self.assertTrue(torch.equal(out, expect))

    def test_multi_batch_equals_single_shot(self):
        module = _make_module(1, 1, 1, 1, True)
        batches = [(torch.randn(2, 8, 8, 8) * 0.5).to(torch.float16) for _ in range(3)]
        fp16fixed = Fp16FixedConv2d.from_module(module, 42)
        fp16fixed.start_calibration()
        for xb in batches:
            fp16fixed.accumulate_sse(xb)
        multi_sv = fp16fixed.finalize()

        single = Fp16FixedConv2d.from_module(module, 42)
        total = sum(single.compute_candidate_sse(xb) for xb in batches)
        self.assertEqual(multi_sv, single.candidates[int(torch.argmin(total).item())])
        self.assertTrue(hasattr(fp16fixed, 'fixed_shift_value'))

    def test_finalize_without_accumulation_raises(self):
        fp16fixed = Fp16FixedConv2d(8, 16, 3)
        fp16fixed.start_calibration()
        with self.assertRaises(RuntimeError):
            fp16fixed.finalize()

    def test_precision_against_fp32(self):
        module = _make_module(1, 1, 1, 1, True)
        fp16fixed = Fp16FixedConv2d.from_module(module, 42)
        out = fp16fixed(self.x)
        ref = torch.nn.functional.conv2d(
            self.x.float(),
            module.weight.float(),
            bias=module.bias.float(),
            stride=module.stride,
            padding=module.padding,
        )
        rel = (out.float() - ref).abs() / (ref.abs() + 1e-2)
        self.assertLess(rel.median().item(), 1e-2)
        self.assertLess(rel.max().item(), 5e-2)


if __name__ == '__main__':
    unittest.main()
