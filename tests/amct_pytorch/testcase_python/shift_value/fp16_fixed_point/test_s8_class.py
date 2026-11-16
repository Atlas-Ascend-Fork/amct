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
"""Class level tests of the fp16fixed F16xS8 (weight-only) operator classes.

``Fp16FixedLinearS8`` / ``Fp16FixedConv2dS8`` / ``Fp16FixedConvTranspose2dS8`` wrap the
validated F16xS8 functional entries for the ``ShiftValueSearchMixin``
based per-layer shift value search:

- the class derives the accumulator-LSB antiquant scale
  ``weight_scale * 2 ** (sv - 29)`` per shift value (pinned against the
  manual engine pipeline);
- forward stays close to the fp32 torch op run with the dequantized
  int8 weight;
- the search runs over the fp16_int8 candidate set {5, ..., 14} and
  picks the manual per-candidate SSE argmin;
- from_module validates the int8 weight dtype / shape.
"""

import unittest

import pytest

import torch
import torch.nn.functional as F

from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.fp16_fixed_point import (
    Fp16FixedConv2dS8,
)
from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.fp16_fixed_point import (
    Fp16FixedConvTranspose2dS8,
)
from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.fp16_fixed_point import (
    Fp16FixedLinearS8,
)
from .reference.conv2d_convt_fp16s8 import (
    s8_bias_to_int32,
)
from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.fp16_fixed_point.conv2d_s8 import (
    conv_f16s8,
)
from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.fp16_fixed_point.convtranspose2d_s8 import (
    convt_f16s8,
)
from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.fp16_fixed_point.matmul import (
    f16_matmul_s8,
)
from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.fp16_fixed_point.matmul import (
    int32_dequant_to_fp16_s8,
)

pytestmark = [pytest.mark.cpu, pytest.mark.slow]


SV_S8 = 13


def _quant_weight_symmetric(weight_fp16, axis):
    """Per-axis symmetric int8 quantization of a fp16 weight."""
    weight_f32 = weight_fp16.to(torch.float32)
    reduce_dims = tuple(d for d in range(weight_f32.dim()) if d != axis)
    amax = weight_f32.abs().amax(dim=reduce_dims)
    weight_scale = (amax / 127.0).clamp_min(2.0**-30).to(torch.float32)
    scale_shape = [1] * weight_f32.dim()
    scale_shape[axis] = -1
    int8_weight = (
        torch.round(weight_f32 / weight_scale.reshape(scale_shape))
        .clamp(-127, 127)
        .to(torch.int8)
    )
    return int8_weight, weight_scale


def _assert_precision(test_case, out, ref):
    rel = (out.float() - ref).abs() / (ref.abs() + 1e-2)
    test_case.assertLess(rel.median().item(), 1e-2)
    test_case.assertLess(rel.max().item(), 5e-2)


def _manual_sse(fp16fixed_module, x):
    """Per-candidate SSE computed by hand, ordered as the candidates."""
    reference = fp16fixed_module._reference_output(x)
    sse = torch.zeros(len(fp16fixed_module.candidates), dtype=torch.float64)
    for idx, candidate in enumerate(fp16fixed_module.candidates):
        out = fp16fixed_module._forward_impl(x, candidate)
        diff = out.to(torch.float32) - reference
        sse[idx] = diff.pow(2).sum().to(torch.float64)
    return sse


class TestFp16FixedLinearS8(unittest.TestCase):
    """Fp16FixedLinearS8 forward, scale derivation and search."""

    def setUp(self):
        torch.manual_seed(2026)

    def _make(self, in_features=64, out_features=32, bias=True):
        module = torch.nn.Linear(in_features, out_features, bias=bias)
        int8_weight, weight_scale = _quant_weight_symmetric(
            module.weight.detach().to(torch.float16), axis=0
        )
        fp16fixed = Fp16FixedLinearS8.from_module(module, int8_weight, weight_scale)
        return module, fp16fixed, int8_weight, weight_scale

    def test_from_module_forward_precision(self):
        module, fp16fixed, int8_weight, weight_scale = self._make()
        x = torch.randn(4, 64)
        ref = F.linear(
            x,
            int8_weight.to(torch.float32) * weight_scale.reshape(-1, 1),
            module.bias,
        )
        _assert_precision(self, fp16fixed(x), ref)

    def test_batched_input(self):
        module, fp16fixed, int8_weight, weight_scale = self._make()
        x = torch.randn(2, 5, 64)
        ref = F.linear(
            x,
            int8_weight.to(torch.float32) * weight_scale.reshape(-1, 1),
            module.bias,
        )
        out = fp16fixed(x)
        self.assertEqual(tuple(out.shape), (2, 5, 32))
        _assert_precision(self, out, ref)

    def test_accumulator_scale_derived_per_shift_value(self):
        # the class must derive the accumulator-LSB real value
        # weight_scale * 2 ** (sv - 29) for every candidate shift value
        module, fp16fixed, int8_weight, weight_scale = self._make()
        x = torch.randn(4, 64)
        for sv in (5, SV_S8, 14):
            ascale = weight_scale * float(2.0 ** (sv - 29))
            bias_int = s8_bias_to_int32(module.bias, ascale)
            accumulator = f16_matmul_s8(
                x.to(torch.float16), int8_weight, bias_int, sv, None
            )
            expect = int32_dequant_to_fp16_s8(accumulator, ascale)
            self.assertTrue(torch.equal(fp16fixed._forward_impl(x, sv), expect))

    def test_candidates_are_int8_range(self):
        self.assertEqual(Fp16FixedLinearS8.candidates, list(range(5, 15)))

    def test_search_returns_argmin(self):
        module, fp16fixed, int8_weight, weight_scale = self._make()
        x = torch.randn(8, 64)
        best = int(torch.argmin(_manual_sse(fp16fixed, x)).item())
        self.assertEqual(
            fp16fixed.search_fixed_shift_value(x), fp16fixed.candidates[best]
        )

    def test_multi_batch_accumulate_finalize(self):
        module, fp16fixed, int8_weight, weight_scale = self._make()
        batches = [torch.randn(4, 64), torch.randn(4, 64)]
        fp16fixed.start_calibration()
        for batch in batches:
            fp16fixed.accumulate_sse(batch)
        value = fp16fixed.finalize()
        self.assertIsNone(fp16fixed._sse_sum)
        combined = _manual_sse(fp16fixed, batches[0]) + _manual_sse(
            fp16fixed, batches[1]
        )
        self.assertEqual(
            value, fp16fixed.candidates[int(torch.argmin(combined).item())]
        )

    def test_scalar_weight_scale_broadcast(self):
        module = torch.nn.Linear(64, 32)
        int8_weight, _ = _quant_weight_symmetric(
            module.weight.detach().to(torch.float16), axis=0
        )
        scalar = Fp16FixedLinearS8.from_module(module, int8_weight, torch.tensor(0.003))
        vector = Fp16FixedLinearS8.from_module(
            module, int8_weight, torch.full((32,), 0.003)
        )
        x = torch.randn(4, 64)
        self.assertTrue(torch.equal(scalar(x), vector(x)))

    def test_int8_weight_dtype_validated(self):
        module = torch.nn.Linear(64, 32)
        fp16_weight = module.weight.detach().to(torch.float16)
        with self.assertRaises(ValueError):
            Fp16FixedLinearS8.from_module(module, fp16_weight, torch.tensor(0.003))

    def test_int8_weight_shape_validated(self):
        module = torch.nn.Linear(64, 32)
        int8_weight = torch.randint(-127, 128, (32, 16), dtype=torch.int8)
        with self.assertRaises(ValueError):
            Fp16FixedLinearS8.from_module(module, int8_weight, torch.tensor(0.003))

    def test_no_bias_module(self):
        module, fp16fixed, int8_weight, weight_scale = self._make(bias=False)
        x = torch.randn(4, 64)
        ref = F.linear(x, int8_weight.to(torch.float32) * weight_scale.reshape(-1, 1))
        _assert_precision(self, fp16fixed(x), ref)
        self.assertIn(fp16fixed.search_fixed_shift_value(x), range(5, 15))

    def test_tie_breaks_to_first_candidate(self):
        """Equal SSE across candidates: the first (smallest) sv wins.

        A bias-free module with an all-zero input produces exactly the
        same output for every candidate, so all SSE entries tie and
        ``torch.argmin`` returns the first index -- the smallest sv of
        the candidate list (the coarsest accumulator grid, i.e. the one
        with the most INT32 headroom).
        """
        module, fp16fixed, int8_weight, weight_scale = self._make(bias=False)
        sse = fp16fixed.compute_candidate_sse(torch.zeros(4, 64))
        self.assertEqual(sse.unique().numel(), 1)
        self.assertEqual(fp16fixed.search_fixed_shift_value(torch.zeros(4, 64)), 5)


class TestFp16FixedConv2dS8(unittest.TestCase):
    """Fp16FixedConv2dS8 forward, functional parity and search."""

    def setUp(self):
        torch.manual_seed(2026)

    def _make(self, **kwargs):
        params = dict(
            in_channels=8,
            out_channels=12,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=True,
        )
        params.update(kwargs)
        module = torch.nn.Conv2d(
            params['in_channels'],
            params['out_channels'],
            params['kernel_size'],
            stride=params['stride'],
            padding=params['padding'],
            dilation=params['dilation'],
            groups=params['groups'],
            bias=params['bias'],
        )
        int8_weight, weight_scale = _quant_weight_symmetric(
            module.weight.detach().to(torch.float16), axis=0
        )
        fp16fixed = Fp16FixedConv2dS8.from_module(module, int8_weight, weight_scale)
        return module, fp16fixed, int8_weight, weight_scale

    def test_from_module_forward_precision(self):
        module, fp16fixed, int8_weight, weight_scale = self._make()
        x = torch.randn(2, 8, 8, 8)
        ref = F.conv2d(
            x,
            int8_weight.to(torch.float32) * weight_scale.reshape(-1, 1, 1, 1),
            module.bias,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )
        _assert_precision(self, fp16fixed(x), ref)

    def test_matches_functional_entry(self):
        module, fp16fixed, int8_weight, weight_scale = self._make()
        x = torch.randn(2, 8, 8, 8)
        for sv in (5, SV_S8, 14):
            ascale = weight_scale * float(2.0 ** (sv - 29))
            expect = conv_f16s8(
                x.to(torch.float16),
                int8_weight,
                module.bias,
                ascale,
                (1, 1, 1, 1),
                (1, 1),
                (1, 1),
                1,
                sv,
            )
            self.assertTrue(torch.equal(fp16fixed._forward_impl(x, sv), expect))

    def test_groups(self):
        module, fp16fixed, int8_weight, weight_scale = self._make(
            in_channels=6, out_channels=12, groups=2
        )
        x = torch.randn(2, 6, 8, 8)
        out = fp16fixed(x)
        self.assertEqual(tuple(out.shape), (2, 12, 8, 8))
        ref = F.conv2d(
            x,
            int8_weight.to(torch.float32) * weight_scale.reshape(-1, 1, 1, 1),
            module.bias,
            stride=module.stride,
            padding=module.padding,
            groups=module.groups,
        )
        _assert_precision(self, out, ref)

    def test_tuple_padding(self):
        module, fp16fixed, int8_weight, weight_scale = self._make(padding=(1, 2))
        x = torch.randn(2, 8, 8, 8)
        out = fp16fixed(x)
        self.assertEqual(tuple(out.shape), (2, 12, 8, 10))
        ref = F.conv2d(
            x,
            int8_weight.to(torch.float32) * weight_scale.reshape(-1, 1, 1, 1),
            module.bias,
            padding=module.padding,
        )
        _assert_precision(self, out, ref)

    def test_search_returns_value_in_range(self):
        module, fp16fixed, int8_weight, weight_scale = self._make()
        x = torch.randn(2, 8, 8, 8)
        value = fp16fixed.search_fixed_shift_value(x)
        self.assertIn(value, range(5, 15))
        self.assertEqual(value, fp16fixed.fixed_shift_value)

    def test_search_returns_argmin(self):
        module, fp16fixed, int8_weight, weight_scale = self._make()
        x = torch.randn(2, 8, 8, 8)
        best = int(torch.argmin(_manual_sse(fp16fixed, x)).item())
        self.assertEqual(
            fp16fixed.search_fixed_shift_value(x), fp16fixed.candidates[best]
        )

    def test_int8_weight_dtype_validated(self):
        module = torch.nn.Conv2d(8, 12, 3, padding=1)
        with self.assertRaises(ValueError):
            Fp16FixedConv2dS8.from_module(
                module, module.weight.detach().to(torch.float16), torch.tensor(0.003)
            )

    def test_pads_accepts_int_padding(self):
        # torch normalizes int padding to a tuple in __init__; the int
        # form is still supported when reassigned after construction
        module, fp16fixed, int8_weight, weight_scale = self._make()
        fp16fixed.padding = 1
        self.assertEqual(fp16fixed._pads(), (1, 1, 1, 1))


class TestFp16FixedConvTranspose2dS8(unittest.TestCase):
    """Fp16FixedConvTranspose2dS8 forward, functional parity and search."""

    def setUp(self):
        torch.manual_seed(2026)

    def _make(self, **kwargs):
        params = dict(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=0,
            output_padding=0,
            dilation=1,
            groups=1,
            bias=True,
        )
        params.update(kwargs)
        module = torch.nn.ConvTranspose2d(
            params['in_channels'],
            params['out_channels'],
            params['kernel_size'],
            stride=params['stride'],
            padding=params['padding'],
            output_padding=params['output_padding'],
            dilation=params['dilation'],
            groups=params['groups'],
            bias=params['bias'],
        )
        int8_weight, weight_scale = _quant_weight_symmetric(
            module.weight.detach().to(torch.float16), axis=1
        )
        # the weight dim 1 holds out_channels / groups entries shared by
        # every group: the class scale needs one entry per output channel
        weight_scale = weight_scale.repeat(module.groups)
        fp16fixed = Fp16FixedConvTranspose2dS8.from_module(
            module, int8_weight, weight_scale
        )
        return module, fp16fixed, int8_weight, weight_scale

    def _ref(self, module, int8_weight, weight_scale, x, groups=1):
        weight = int8_weight.to(torch.float32)
        if groups > 1:
            in_channels, channels_per_group, kernel_h, kernel_w = weight.shape
            channels_in_group = in_channels // groups
            weight = (
                weight.reshape(
                    groups, channels_in_group, channels_per_group, kernel_h, kernel_w
                )
                * weight_scale.reshape(groups, 1, channels_per_group, 1, 1)
            ).reshape(in_channels, channels_per_group, kernel_h, kernel_w)
        else:
            weight = weight * weight_scale.reshape(1, -1, 1, 1)
        return F.conv_transpose2d(
            x,
            weight,
            module.bias,
            stride=module.stride,
            padding=module.padding,
            output_padding=module.output_padding,
            dilation=module.dilation,
            groups=module.groups,
        )

    def test_from_module_forward_precision(self):
        module, fp16fixed, int8_weight, weight_scale = self._make()
        x = torch.randn(2, 8, 8, 8)
        out = fp16fixed(x)
        self.assertEqual(tuple(out.shape), (2, 16, 10, 10))
        _assert_precision(self, out, self._ref(module, int8_weight, weight_scale, x))

    def test_matches_functional_entry(self):
        module, fp16fixed, int8_weight, weight_scale = self._make()
        x = torch.randn(2, 8, 8, 8)
        for sv in (5, SV_S8, 14):
            ascale = weight_scale * float(2.0 ** (sv - 29))
            expect = convt_f16s8(
                x.to(torch.float16),
                int8_weight,
                module.bias,
                ascale,
                (0, 0, 0, 0),
                (1, 1),
                (1, 1),
                1,
                (2, 16, 10, 10),
                sv,
            )
            self.assertTrue(torch.equal(fp16fixed._forward_impl(x, sv), expect))

    def test_stride_padding_geometry(self):
        module, fp16fixed, int8_weight, weight_scale = self._make(
            stride=2, padding=1, output_padding=1
        )
        x = torch.randn(2, 8, 8, 8)
        out = fp16fixed(x)
        # out = (8 - 1) * 2 - 2 * 1 + 2 + 1 + 1 = 16
        self.assertEqual(tuple(out.shape), (2, 16, 16, 16))
        _assert_precision(self, out, self._ref(module, int8_weight, weight_scale, x))

    def test_groups(self):
        module, fp16fixed, int8_weight, weight_scale = self._make(groups=2)
        x = torch.randn(2, 8, 8, 8)
        out = fp16fixed(x)
        self.assertEqual(tuple(out.shape), (2, 16, 10, 10))
        _assert_precision(self, out, self._ref(module, int8_weight, weight_scale, x, 2))

    def test_search_returns_value_in_range(self):
        module, fp16fixed, int8_weight, weight_scale = self._make()
        x = torch.randn(2, 8, 8, 8)
        value = fp16fixed.search_fixed_shift_value(x)
        self.assertIn(value, range(5, 15))
        self.assertEqual(value, fp16fixed.fixed_shift_value)

    def test_groups_search_covers_reference(self):
        """Search with groups exercises the per-group reference expansion."""
        module, fp16fixed, int8_weight, weight_scale = self._make(groups=2)
        x = torch.randn(2, 8, 8, 8)
        value = fp16fixed.search_fixed_shift_value(x)
        self.assertIn(value, range(5, 15))

    def test_pads_accepts_int_padding(self):
        # torch normalizes int padding to a tuple in __init__; the int
        # form is still supported when reassigned after construction
        module, fp16fixed, int8_weight, weight_scale = self._make()
        fp16fixed.padding = 1
        self.assertEqual(fp16fixed._pads(), (1, 1, 1, 1))


if __name__ == '__main__':
    unittest.main()
