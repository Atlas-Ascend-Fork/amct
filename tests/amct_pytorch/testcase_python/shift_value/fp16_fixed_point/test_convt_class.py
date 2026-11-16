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
"""Tests of the Fp16FixedConvTranspose2d class: legacy parity, search and multi batch."""

import unittest

import pytest

import torch

from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.fp16_fixed_point import (
    Fp16FixedConvTranspose2d,
)
from .reference.convtranspose2d import (
    convt_f16f16 as legacy_convt_f16f16,
)

pytestmark = [pytest.mark.cpu, pytest.mark.slow]


def _make_module(stride, padding, dilation, groups, use_bias, output_padding=0):
    module = torch.nn.ConvTranspose2d(
        8,
        16,
        3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=use_bias,
        output_padding=output_padding,
    )
    module.weight.data = (
        (torch.randn_like(module.weight) * 0.1).to(torch.float16).to(torch.float32)
    )
    if use_bias:
        module.bias.data = (
            (torch.randn_like(module.bias) * 0.1).to(torch.float16).to(torch.float32)
        )
    return module


class TestFp16FixedConvTranspose2d(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        self.x = (torch.randn(2, 8, 8, 8) * 0.5).to(torch.float16)

    def _legacy_forward(self, module, x, sv):
        padding = module.padding
        if isinstance(padding, int):
            padding = (padding, padding)
        pads = (padding[0], padding[0], padding[1], padding[1])
        n, _, h_in, w_in = x.shape
        h_out = (
            (h_in - 1) * module.stride[0]
            - 2 * module.padding[0]
            + module.dilation[0] * (module.kernel_size[0] - 1)
            + module.output_padding[0]
            + 1
        )
        w_out = (
            (w_in - 1) * module.stride[1]
            - 2 * module.padding[1]
            + module.dilation[1] * (module.kernel_size[1] - 1)
            + module.output_padding[1]
            + 1
        )
        return legacy_convt_f16f16(
            x,
            module.weight.to(torch.float16),
            module.bias,
            pads,
            tuple(module.stride),
            tuple(module.dilation),
            module.groups,
            (n, module.out_channels, h_out, w_out),
            sv,
        )

    def test_bit_parity_with_legacy(self):
        """forward equals the legacy reference on the parameter matrix."""
        combos = [
            (1, 0, 1, 1, True, 0),
            (2, 1, 1, 1, True, 0),
            (1, 0, 2, 1, True, 0),
            (1, 1, 1, 2, False, 0),
            (2, 0, 1, 4, True, 0),
            (1, 0, 2, 2, True, 0),
            (2, 0, 1, 1, True, 1),
        ]
        for stride, padding, dilation, groups, use_bias, opad in combos:
            module = _make_module(
                stride, padding, dilation, groups, use_bias, output_padding=opad
            )
            fp16fixed = Fp16FixedConvTranspose2d.from_module(module, 42)
            out = fp16fixed(self.x)
            expect = self._legacy_forward(module, self.x, 42)
            self.assertTrue(
                torch.equal(out, expect),
                'parity failed on {}'.format(
                    (stride, padding, dilation, groups, use_bias, opad)
                ),
            )

    def test_from_module_copies_state_dict(self):
        module = _make_module(2, 1, 1, 1, True)
        fp16fixed = Fp16FixedConvTranspose2d.from_module(module, 41)
        self.assertTrue(torch.equal(fp16fixed.weight.data, module.weight.data))
        self.assertTrue(torch.equal(fp16fixed.bias.data, module.bias.data))
        self.assertEqual(fp16fixed.stride, module.stride)
        self.assertEqual(fp16fixed.output_padding, module.output_padding)
        self.assertEqual(fp16fixed.fixed_shift_value, 41)

    def test_output_shape_matches_torch_rule(self):
        """The class computes output_shape with the ConvTranspose2d rule."""
        module = _make_module(2, 1, 1, 1, True, output_padding=1)
        fp16fixed = Fp16FixedConvTranspose2d.from_module(module, 42)
        out = fp16fixed(self.x)
        ref_shape = torch.nn.functional.conv_transpose2d(
            self.x.float(),
            module.weight.float(),
            stride=module.stride,
            padding=module.padding,
            output_padding=module.output_padding,
        ).shape
        self.assertEqual(tuple(out.shape), tuple(ref_shape))

    def test_tuple_padding_and_output_padding(self):
        """Tuple forms of padding / output_padding flow through the class."""
        module = _make_module(2, (1, 2), 1, 1, True, output_padding=(1, 0))
        fp16fixed = Fp16FixedConvTranspose2d.from_module(module, 42)
        out = fp16fixed(self.x)
        expect = self._legacy_forward(module, self.x, 42)
        self.assertTrue(torch.equal(out, expect))
        ref = torch.nn.functional.conv_transpose2d(
            self.x.float(),
            module.weight.float(),
            bias=module.bias.float(),
            stride=module.stride,
            padding=module.padding,
            output_padding=module.output_padding,
        )
        self.assertEqual(tuple(out.shape), tuple(ref.shape))
        rel = (out.float() - ref).abs() / (ref.abs() + 1e-2)
        self.assertLess(rel.median().item(), 1e-2)
        self.assertLess(rel.max().item(), 5e-2)

    def test_search_returns_argmin_and_writes_back(self):
        module = _make_module(2, 1, 1, 1, True)
        fp16fixed = Fp16FixedConvTranspose2d.from_module(module, 42)
        sse = fp16fixed.compute_candidate_sse(self.x)
        self.assertEqual(tuple(sse.shape), (10,))
        best = fp16fixed.search_fixed_shift_value(self.x)
        self.assertIn(best, range(34, 44))
        self.assertEqual(best, fp16fixed.candidates[int(torch.argmin(sse).item())])

    def test_multi_batch_equals_single_shot(self):
        module = _make_module(2, 1, 1, 1, True)
        batches = [(torch.randn(2, 8, 8, 8) * 0.5).to(torch.float16) for _ in range(3)]
        fp16fixed = Fp16FixedConvTranspose2d.from_module(module, 42)
        fp16fixed.start_calibration()
        for xb in batches:
            fp16fixed.accumulate_sse(xb)
        multi_sv = fp16fixed.finalize()

        single = Fp16FixedConvTranspose2d.from_module(module, 42)
        total = sum(single.compute_candidate_sse(xb) for xb in batches)
        self.assertEqual(multi_sv, single.candidates[int(torch.argmin(total).item())])

    def test_precision_against_fp32(self):
        module = _make_module(1, 0, 1, 1, True)
        fp16fixed = Fp16FixedConvTranspose2d.from_module(module, 42)
        out = fp16fixed(self.x)
        ref = torch.nn.functional.conv_transpose2d(
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
