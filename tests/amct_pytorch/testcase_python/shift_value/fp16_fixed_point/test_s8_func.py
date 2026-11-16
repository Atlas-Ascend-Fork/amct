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
"""Functional tests of the fp16fixed F16xS8 (weight-only) references.

``tests/amct_pytorch/testcase_python/shift_value/fp16_fixed_point/reference/conv2d_convt_fp16s8.py`` is the fp16-activation x int8-weight
reference of the weight-only shift value path (regime ``fp16_int8``,
candidates {5, ..., 14}). It has no other consumer yet, so these are its
first direct tests:

- ``f16xs8_gemm`` is checked bit-exactly against the refactored
  ``matmul.f16_matmul_s8``: both implement the same fixed point
  pipeline (per-tile max-exponent alignment of the fp16 side, 16-element
  MAC with the raw int8 weight, shift ``sv - input_max_exp`` without
  clamp, per-tile round + int32 saturation);
- ``conv_f16s8`` / ``convt_f16s8`` are checked against the fp32 torch
  ops run with the dequantized int8 weight over the geometry matrix of
  the F16xF16 suite (non square kernels, asymmetric pads, non square
  strides / dilations, groups) plus subnormal inputs, saturating
  extremes and scalar scales;
- the refactored copies ``conv2d_s8.py`` / ``convtranspose2d_s8.py``
  (engine and im2col machinery reused from the F16xF16 family) are
  checked bit-exactly against the legacy reference over the same matrix.

Antiquant scale convention (pinned by ``test_matmul``):
``antiquant_scale`` is the accumulator-LSB real value,
``weight_scale * 2 ** (sv - 29)`` -- it depends on the shift value.
"""

import unittest

import pytest

import torch
import torch.nn.functional as F

from .reference.conv2d_convt_fp16s8 import (
    conv_f16s8,
    convt_f16s8,
    f16xs8_gemm,
    int32_dequant_to_fp16_s8 as legacy_dequant_s8,
    s8_bias_to_int32,
)
from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.fp16_fixed_point.conv2d_s8 import (
    conv_f16s8 as new_conv_f16s8,
)
from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.fp16_fixed_point.convtranspose2d_s8 import (
    convt_f16s8 as new_convt_f16s8,
)
from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.fp16_fixed_point.matmul import (
    _check_int8_weight,
)
from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.fp16_fixed_point.matmul import (
    _normalize_ascale,
)
from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.fp16_fixed_point.matmul import (
    _s8_bias_to_int32 as shared_s8_bias_to_int32,
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
    """Per-axis symmetric int8 quantization of a fp16 weight.

    Returns ``(int8_weight, weight_scale)``; the scale has one entry per
    position of ``axis`` (the output channel axis: 0 for Conv, 1 for
    ConvTranspose).
    """
    weight_f32 = weight_fp16.to(torch.float32)
    reduce_dims = tuple(d for d in range(weight_f32.dim()) if d != axis)
    amax = weight_f32.abs().amax(dim=reduce_dims)
    # clamp so an all zero channel never yields a zero scale
    weight_scale = (amax / 127.0).clamp_min(2.0**-30).to(torch.float32)
    scale_shape = [1] * weight_f32.dim()
    scale_shape[axis] = -1
    int8_weight = (
        torch.round(weight_f32 / weight_scale.reshape(scale_shape))
        .clamp(-127, 127)
        .to(torch.int8)
    )
    return int8_weight, weight_scale


def _dequant_weight(int8_weight, weight_scale, axis):
    """int8 weight * per-output-channel scale, as fp32."""
    scale_shape = [1] * int8_weight.dim()
    scale_shape[axis] = -1
    return int8_weight.to(torch.float32) * weight_scale.reshape(scale_shape)


def _accumulator_scale(weight_scale, sv):
    """Accumulator-LSB real value of the F16xS8 domain: ws * 2 ** (sv - 29)."""
    return weight_scale.to(torch.float32) * float(2.0 ** (sv - 29))


class TestF16xS8GemmParity(unittest.TestCase):
    """f16xs8_gemm (legacy) vs f16_matmul_s8 (refactored), bit-exact."""

    def setUp(self):
        torch.manual_seed(2026)

    def _run_pair(self, m, k, n, sv, bias=None, ascale=None, value_scale=0.5):
        a = (torch.randn(m, k) * value_scale).to(torch.float16)
        b = torch.randint(-128, 128, (n, k), dtype=torch.int8)
        if bias is None:
            acc0 = None
        elif ascale is not None:
            # raw bias: the legacy caller pre-scales it via s8_bias_to_int32
            bias_int = s8_bias_to_int32(bias, ascale)
            acc0 = bias_int.unsqueeze(0).expand(m, n)
        else:
            # bias already in the accumulator domain
            acc0 = (
                torch.round(bias.to(torch.float64))
                .to(torch.int64)
                .unsqueeze(0)
                .expand(m, n)
            )
        legacy_out = f16xs8_gemm(a, b, sv, acc0=acc0)
        new_out = f16_matmul_s8(a, b, bias, sv, ascale)
        return legacy_out, new_out

    def test_random_k_multiple_of_16(self):
        legacy_out, new_out = self._run_pair(7, 48, 5, SV_S8)
        self.assertTrue(torch.equal(legacy_out, new_out))

    def test_k_padded_to_multiple_of_16(self):
        legacy_out, new_out = self._run_pair(6, 37, 4, SV_S8)
        self.assertTrue(torch.equal(legacy_out, new_out))

    def test_raw_bias_with_antiquant_scale(self):
        bias = (torch.randn(5) * 0.1).to(torch.float16)
        ascale = (torch.rand(5) * 1e-3 + 1e-5).to(torch.float32)
        legacy_out, new_out = self._run_pair(7, 48, 5, SV_S8, bias=bias, ascale=ascale)
        self.assertTrue(torch.equal(legacy_out, new_out))

    def test_pre_scaled_bias_without_antiquant_scale(self):
        bias = torch.randn(5)
        legacy_out, new_out = self._run_pair(7, 48, 5, SV_S8, bias=bias, ascale=None)
        self.assertTrue(torch.equal(legacy_out, new_out))

    def test_subnormal_inputs(self):
        legacy_out, new_out = self._run_pair(7, 48, 5, SV_S8, value_scale=2.0**-20)
        self.assertTrue(torch.equal(legacy_out, new_out))

    def test_all_zero_input(self):
        a = torch.zeros(6, 32, dtype=torch.float16)
        b = torch.randint(-128, 128, (4, 32), dtype=torch.int8)
        legacy_out = f16xs8_gemm(a, b, SV_S8)
        new_out = f16_matmul_s8(a, b, None, SV_S8)
        self.assertTrue(torch.equal(legacy_out, new_out))
        self.assertTrue(torch.all(legacy_out == 0))

    def test_extreme_values_saturation(self):
        a = torch.full((6, 32), 65504.0, dtype=torch.float16)
        b = torch.full((4, 32), 127, dtype=torch.int8)
        legacy_out = f16xs8_gemm(a, b, SV_S8)
        new_out = f16_matmul_s8(a, b, None, SV_S8)
        self.assertTrue(torch.equal(legacy_out, new_out))
        self.assertTrue(torch.all(legacy_out == 2**31 - 1))
        a_neg = torch.full((6, 32), -65504.0, dtype=torch.float16)
        legacy_neg = f16xs8_gemm(a_neg, b, SV_S8)
        new_neg = f16_matmul_s8(a_neg, b, None, SV_S8)
        self.assertTrue(torch.equal(legacy_neg, new_neg))
        self.assertTrue(torch.all(legacy_neg == -(2**31)))

    def test_candidate_range_extremes(self):
        for sv in (5, 14):
            legacy_out, new_out = self._run_pair(7, 48, 5, sv)
            self.assertTrue(torch.equal(legacy_out, new_out))

    def test_broadcast_batch_input_expansion(self):
        """A smaller input batch is expanded onto the weight batch."""
        a = (torch.randn(1, 6, 48) * 0.5).to(torch.float16)
        b = torch.randint(-128, 128, (3, 5, 48), dtype=torch.int8)
        out = f16_matmul_s8(a, b, None, SV_S8)
        self.assertEqual(tuple(out.shape), (3, 6, 5))
        expanded = f16_matmul_s8(a.expand(3, 6, 48), b, None, SV_S8)
        self.assertTrue(torch.equal(out, expanded))

    def test_dequant_parity_per_channel_and_scalar(self):
        acc = torch.randint(-(2**31), 2**31 - 1, (5, 8), dtype=torch.int32)
        ascale = (torch.rand(8) * 0.01).to(torch.float32)
        self.assertTrue(
            torch.equal(
                legacy_dequant_s8(acc, ascale), int32_dequant_to_fp16_s8(acc, ascale)
            )
        )
        scalar = torch.tensor(0.01)
        self.assertTrue(
            torch.equal(
                legacy_dequant_s8(acc, scalar),
                int32_dequant_to_fp16_s8(acc, scalar),
            )
        )


class TestS8BiasHelper(unittest.TestCase):
    """Direct tests of s8_bias_to_int32."""

    def setUp(self):
        torch.manual_seed(2026)

    def test_formula(self):
        bias = torch.randn(8)
        ascale = (torch.rand(8) * 0.01 + 1e-4).to(torch.float32)
        out = s8_bias_to_int32(bias, ascale)
        expect = torch.round(bias.to(torch.float16).to(torch.float32) / ascale).to(
            torch.int64
        )
        self.assertTrue(torch.equal(out, expect))

    def test_zero_scale_guard(self):
        # a zero antiquant scale is treated as one (defensive guard of the
        # legacy reference; the refactored f16_matmul_s8 keeps the numpy
        # behaviour and does not guard)
        bias = torch.tensor([0.5, -1.5, 2.0])
        out = s8_bias_to_int32(bias, torch.zeros(3))
        expect = torch.round(bias.to(torch.float16).to(torch.float32)).to(torch.int64)
        self.assertTrue(torch.equal(out, expect))

    def test_none_bias(self):
        self.assertIsNone(s8_bias_to_int32(None, torch.ones(1)))


class TestS8SharedHelpers(unittest.TestCase):
    """Direct tests of the S8 family shared helpers of ``matmul``.

    These helpers moved next to the S8 engine when the combined conv /
    convt module was split into per-operator files; the tests pin their
    contract (and, for the bias scaling, identity with the legacy copy).
    """

    def setUp(self):
        torch.manual_seed(2026)

    def test_s8_bias_to_int32_matches_legacy(self):
        bias = torch.randn(8)
        ascale = (torch.rand(8) * 0.01 + 1e-4).to(torch.float32)
        self.assertTrue(
            torch.equal(
                shared_s8_bias_to_int32(bias, ascale),
                s8_bias_to_int32(bias, ascale),
            )
        )

    def test_s8_bias_to_int32_none_and_zero_scale_guard(self):
        self.assertIsNone(shared_s8_bias_to_int32(None, torch.ones(1)))
        bias = torch.tensor([0.5, -1.5, 2.0])
        # a zero antiquant scale (dead output channel) is treated as one
        out = shared_s8_bias_to_int32(bias, torch.zeros(3))
        expect = torch.round(bias.to(torch.float16).to(torch.float32)).to(torch.int64)
        self.assertTrue(torch.equal(out, expect))

    def test_normalize_ascale_scalar_and_mismatch(self):
        normalized = _normalize_ascale(torch.tensor(0.01), 4)
        self.assertTrue(torch.equal(normalized, torch.full((4,), 0.01)))
        per_channel = torch.tensor([0.1, 0.2, 0.3])
        self.assertTrue(torch.equal(_normalize_ascale(per_channel, 3), per_channel))
        with self.assertRaises(ValueError):
            _normalize_ascale(per_channel, 4)

    def test_check_int8_weight(self):
        reference = torch.randn(4, 8)
        int8_weight = torch.randint(-127, 128, (4, 8), dtype=torch.int8)
        _check_int8_weight(int8_weight, reference)
        with self.assertRaises(ValueError):
            _check_int8_weight(reference.to(torch.float16), reference)
        with self.assertRaises(ValueError):
            _check_int8_weight(
                torch.randint(-127, 128, (4, 6), dtype=torch.int8), reference
            )


class TestConvF16S8(unittest.TestCase):
    """conv_f16s8 vs the fp32 conv run with the dequantized int8 weight."""

    def setUp(self):
        torch.manual_seed(2026)

    def _run(
        self,
        cin,
        cout,
        kernel,
        pads,
        strides,
        dilations,
        groups=1,
        use_bias=True,
        value_scale=0.5,
        weight_scale=0.1,
        sv=SV_S8,
    ):
        kh, kw = kernel
        x = (torch.randn(2, cin, 8, 8) * value_scale).to(torch.float16)
        weight = (torch.randn(cout, cin // groups, kh, kw) * weight_scale).to(
            torch.float16
        )
        int8_weight, w_scale = _quant_weight_symmetric(weight, axis=0)
        bias = (torch.randn(cout) * 0.1).to(torch.float16) if use_bias else None
        out = conv_f16s8(
            x,
            int8_weight,
            bias,
            _accumulator_scale(w_scale, sv),
            pads,
            strides,
            dilations,
            groups,
            sv,
        )
        return out, x, int8_weight, w_scale, bias

    def _assert_precision(self, out, ref):
        rel = (out.float() - ref).abs() / (ref.abs() + 1e-2)
        self.assertLess(rel.median().item(), 1e-2)
        self.assertLess(rel.max().item(), 5e-2)

    def _ref(self, x, int8_weight, w_scale, bias, pads, strides, dilations, groups):
        pad_t, pad_b, pad_l, pad_r = pads
        padded = F.pad(x.float(), (pad_l, pad_r, pad_t, pad_b))
        return F.conv2d(
            padded,
            _dequant_weight(int8_weight, w_scale, 0),
            bias=bias.float() if bias is not None else None,
            stride=strides,
            dilation=dilations,
            groups=groups,
        )

    def test_basic_geometry(self):
        out, x, w8, ws, bias = self._run(8, 12, (3, 3), (1, 1, 1, 1), (1, 1), (1, 1))
        self.assertEqual(tuple(out.shape), (2, 12, 8, 8))
        self._assert_precision(
            out, self._ref(x, w8, ws, bias, (1, 1, 1, 1), (1, 1), (1, 1), 1)
        )

    def test_non_square_kernel(self):
        out, x, w8, ws, bias = self._run(8, 12, (3, 5), (1, 1, 1, 1), (1, 1), (1, 1))
        self.assertEqual(tuple(out.shape), (2, 12, 8, 6))
        self._assert_precision(
            out, self._ref(x, w8, ws, bias, (1, 1, 1, 1), (1, 1), (1, 1), 1)
        )

    def test_asymmetric_pads_non_square_geometry(self):
        pads, strides, dilations = (1, 0, 2, 0), (2, 1), (1, 2)
        out, x, w8, ws, bias = self._run(6, 10, (3, 5), pads, strides, dilations)
        # out_h = (8 + 1 + 0 - 3) // 2 + 1 = 4, out_w = (8 + 2 + 0 - 9) + 1 = 2
        self.assertEqual(tuple(out.shape), (2, 10, 4, 2))
        self._assert_precision(
            out, self._ref(x, w8, ws, bias, pads, strides, dilations, 1)
        )

    def test_groups(self):
        out, x, w8, ws, bias = self._run(
            6, 12, (3, 3), (1, 1, 1, 1), (1, 1), (1, 1), groups=2
        )
        self.assertEqual(tuple(out.shape), (2, 12, 8, 8))
        self._assert_precision(
            out, self._ref(x, w8, ws, bias, (1, 1, 1, 1), (1, 1), (1, 1), 2)
        )

    def test_subnormal_inputs(self):
        out, x, w8, ws, bias = self._run(
            8, 16, (3, 3), (1, 1, 1, 1), (1, 1), (1, 1), value_scale=1e-5
        )
        self.assertTrue(torch.any(out != 0))
        self._assert_precision(
            out, self._ref(x, w8, ws, bias, (1, 1, 1, 1), (1, 1), (1, 1), 1)
        )

    def test_extreme_values_saturation(self):
        x = torch.full((2, 8, 8, 8), 65504.0, dtype=torch.float16)
        int8_weight = torch.full((12, 8, 3, 3), 127, dtype=torch.int8)
        # accumulator LSB of 2 ** -20 maps the saturated INT32_MAX to 2048
        ascale = torch.full((12,), 2.0**-20, dtype=torch.float32)
        out = conv_f16s8(
            x, int8_weight, None, ascale, (1, 1, 1, 1), (1, 1), (1, 1), 1, SV_S8
        )
        self.assertTrue(torch.isfinite(out.float()).all())
        self.assertTrue(torch.all(out == 2048.0))
        x_neg = torch.full((2, 8, 8, 8), -65504.0, dtype=torch.float16)
        out_neg = conv_f16s8(
            x_neg,
            int8_weight,
            None,
            ascale,
            (1, 1, 1, 1),
            (1, 1),
            (1, 1),
            1,
            SV_S8,
        )
        self.assertTrue(torch.all(out_neg == -2048.0))

    def test_scalar_ascale_matches_per_channel(self):
        x = (torch.randn(2, 8, 8, 8) * 0.5).to(torch.float16)
        int8_weight = torch.randint(-127, 128, (12, 8, 3, 3), dtype=torch.int8)
        w_scale = 0.003
        bias = (torch.randn(12) * 0.1).to(torch.float16)
        out_scalar = conv_f16s8(
            x,
            int8_weight,
            bias,
            torch.tensor(w_scale * 2.0 ** (SV_S8 - 29)),
            (1, 1, 1, 1),
            (1, 1),
            (1, 1),
            1,
            SV_S8,
        )
        out_vector = conv_f16s8(
            x,
            int8_weight,
            bias,
            _accumulator_scale(torch.full((12,), w_scale), SV_S8),
            (1, 1, 1, 1),
            (1, 1),
            (1, 1),
            1,
            SV_S8,
        )
        self.assertTrue(torch.equal(out_scalar, out_vector))
        ref = F.conv2d(
            x.float(), int8_weight.float() * w_scale, bias=bias.float(), padding=1
        )
        self._assert_precision(out_scalar, ref)

    def test_default_shift_value_is_13(self):
        torch.manual_seed(11)
        x = (torch.randn(2, 8, 8, 8) * 0.5).to(torch.float16)
        weight = (torch.randn(12, 8, 3, 3) * 0.1).to(torch.float16)
        int8_weight, w_scale = _quant_weight_symmetric(weight, axis=0)
        bias = (torch.randn(12) * 0.1).to(torch.float16)
        ascale = _accumulator_scale(w_scale, 13)
        out_default = conv_f16s8(
            x, int8_weight, bias, ascale, (1, 1, 1, 1), (1, 1), (1, 1), 1
        )
        out_explicit = conv_f16s8(
            x, int8_weight, bias, ascale, (1, 1, 1, 1), (1, 1), (1, 1), 1, 13
        )
        self.assertTrue(torch.equal(out_default, out_explicit))


class TestConvtF16S8(unittest.TestCase):
    """convt_f16s8 vs the fp32 conv transpose run with the dequantized weight."""

    def setUp(self):
        torch.manual_seed(2026)

    def _run(
        self,
        cin,
        cout,
        kernel,
        pads,
        strides,
        dilations,
        groups=1,
        use_bias=True,
        output_padding=(0, 0),
        value_scale=0.5,
        weight_scale=0.1,
        sv=SV_S8,
    ):
        kh, kw = kernel
        x = (torch.randn(2, cin, 8, 8) * value_scale).to(torch.float16)
        weight = (torch.randn(cin, cout // groups, kh, kw) * weight_scale).to(
            torch.float16
        )
        int8_weight, w_scale = _quant_weight_symmetric(weight, axis=1)
        bias = (torch.randn(cout) * 0.1).to(torch.float16) if use_bias else None
        pad_t, pad_b, pad_l, pad_r = pads
        stride_h, stride_w = strides
        dilation_h, dilation_w = dilations
        opad_h, opad_w = output_padding
        h_out = (8 - 1) * stride_h + (kh - 1) * dilation_h + 1 - pad_t - pad_b + opad_h
        w_out = (8 - 1) * stride_w + (kw - 1) * dilation_w + 1 - pad_l - pad_r + opad_w
        output_shape = (2, cout, h_out, w_out)
        # the weight dim 1 holds out_channels / groups entries shared by
        # every group, so the accumulator scale is tiled per group
        ascale = _accumulator_scale(w_scale, sv).repeat(groups)
        out = convt_f16s8(
            x,
            int8_weight,
            bias,
            ascale,
            pads,
            strides,
            dilations,
            groups,
            output_shape,
            sv,
        )
        return out, x, int8_weight, w_scale, bias, output_shape

    def _assert_precision(self, out, ref):
        rel = (out.float() - ref).abs() / (ref.abs() + 1e-2)
        self.assertLess(rel.median().item(), 1e-2)
        self.assertLess(rel.max().item(), 5e-2)

    def _ref(
        self,
        x,
        int8_weight,
        w_scale,
        bias,
        pads,
        strides,
        dilations,
        groups,
        output_shape,
        output_padding=(0, 0),
    ):
        pad_t, pad_b, pad_l, pad_r = pads
        dequant = _dequant_weight(int8_weight, w_scale, 1)
        if pad_t == pad_b and pad_l == pad_r:
            ref = F.conv_transpose2d(
                x.float(),
                dequant,
                stride=strides,
                dilation=dilations,
                groups=groups,
                padding=(pad_t, pad_l),
                output_padding=output_padding,
            )
        else:
            # asymmetric pads: full conv transpose then crop
            full = F.conv_transpose2d(
                x.float(), dequant, stride=strides, dilation=dilations, groups=groups
            )
            _, _, h_out, w_out = output_shape
            ref = full[:, :, pad_t : pad_t + h_out, pad_l : pad_l + w_out]
        if bias is not None:
            ref = ref + bias.float().view(1, -1, 1, 1)
        return ref

    def test_basic_geometry(self):
        out, x, w8, ws, bias, shape = self._run(
            8, 16, (3, 3), (0, 0, 0, 0), (1, 1), (1, 1)
        )
        self.assertEqual(tuple(out.shape), (2, 16, 10, 10))
        self._assert_precision(
            out, self._ref(x, w8, ws, bias, (0, 0, 0, 0), (1, 1), (1, 1), 1, shape)
        )

    def test_non_square_kernel(self):
        out, x, w8, ws, bias, shape = self._run(
            8, 16, (2, 4), (0, 0, 0, 0), (1, 1), (1, 1)
        )
        self.assertEqual(tuple(out.shape), (2, 16, 9, 11))
        self._assert_precision(
            out, self._ref(x, w8, ws, bias, (0, 0, 0, 0), (1, 1), (1, 1), 1, shape)
        )

    def test_non_square_stride_dilation_with_output_padding(self):
        pads, strides, dilations, opad = (1, 1, 0, 0), (2, 1), (1, 2), (1, 0)
        out, x, w8, ws, bias, shape = self._run(
            8, 16, (3, 3), pads, strides, dilations, output_padding=opad
        )
        # h_out = 7 * 2 + 2 + 1 - 1 - 1 + 1 = 16, w_out = 7 + 4 + 1 = 12
        self.assertEqual(tuple(out.shape), (2, 16, 16, 12))
        self._assert_precision(
            out,
            self._ref(x, w8, ws, bias, pads, strides, dilations, 1, shape, opad),
        )

    def test_asymmetric_pads(self):
        pads, strides, dilations = (1, 0, 2, 0), (1, 1), (1, 1)
        out, x, w8, ws, bias, shape = self._run(8, 16, (3, 3), pads, strides, dilations)
        # h_out = 7 + 2 + 1 - 1 - 0 = 9, w_out = 7 + 2 + 1 - 2 - 0 = 8
        self.assertEqual(tuple(out.shape), (2, 16, 9, 8))
        self._assert_precision(
            out, self._ref(x, w8, ws, bias, pads, strides, dilations, 1, shape)
        )

    def test_groups(self):
        out, x, w8, ws, bias, shape = self._run(
            8, 16, (3, 3), (0, 0, 0, 0), (1, 1), (1, 1), groups=2
        )
        self.assertEqual(tuple(out.shape), (2, 16, 10, 10))
        self._assert_precision(
            out, self._ref(x, w8, ws, bias, (0, 0, 0, 0), (1, 1), (1, 1), 2, shape)
        )

    def test_subnormal_inputs(self):
        out, x, w8, ws, bias, shape = self._run(
            8, 16, (3, 3), (0, 0, 0, 0), (1, 1), (1, 1), value_scale=1e-5
        )
        self.assertTrue(torch.any(out != 0))
        self._assert_precision(
            out, self._ref(x, w8, ws, bias, (0, 0, 0, 0), (1, 1), (1, 1), 1, shape)
        )

    def test_extreme_values_saturation(self):
        x = torch.full((2, 8, 8, 8), 65504.0, dtype=torch.float16)
        int8_weight = torch.full((8, 16, 3, 3), 127, dtype=torch.int8)
        ascale = torch.full((16,), 2.0**-20, dtype=torch.float32)
        output_shape = (2, 16, 10, 10)
        out = convt_f16s8(
            x,
            int8_weight,
            None,
            ascale,
            (0, 0, 0, 0),
            (1, 1),
            (1, 1),
            1,
            output_shape,
            SV_S8,
        )
        self.assertTrue(torch.isfinite(out.float()).all())
        self.assertTrue(torch.all(out == 2048.0))
        x_neg = torch.full((2, 8, 8, 8), -65504.0, dtype=torch.float16)
        out_neg = convt_f16s8(
            x_neg,
            int8_weight,
            None,
            ascale,
            (0, 0, 0, 0),
            (1, 1),
            (1, 1),
            1,
            output_shape,
            SV_S8,
        )
        self.assertTrue(torch.all(out_neg == -2048.0))


class TestNewConvS8ParityWithLegacy(unittest.TestCase):
    """Bit-exact parity of the refactored conv_f16s8 vs the legacy reference.

    The combinations mirror the legacy precision matrix: non square
    kernels, asymmetric pads, non square strides / dilations, groups,
    subnormal inputs, saturating extremes and the candidate range ends.
    """

    def setUp(self):
        torch.manual_seed(2026)

    def _run(
        self,
        cin,
        cout,
        kernel,
        pads,
        strides,
        dilations,
        groups=1,
        use_bias=True,
        value_scale=0.5,
        weight_scale=0.1,
        sv=SV_S8,
    ):
        kh, kw = kernel
        x = (torch.randn(2, cin, 8, 8) * value_scale).to(torch.float16)
        weight = (torch.randn(cout, cin // groups, kh, kw) * weight_scale).to(
            torch.float16
        )
        int8_weight, w_scale = _quant_weight_symmetric(weight, axis=0)
        bias = (torch.randn(cout) * 0.1).to(torch.float16) if use_bias else None
        args = (
            x,
            int8_weight,
            bias,
            _accumulator_scale(w_scale, sv),
            pads,
            strides,
            dilations,
            groups,
            sv,
        )
        return new_conv_f16s8(*args), conv_f16s8(*args)

    def test_basic_geometry(self):
        new_out, legacy_out = self._run(8, 12, (3, 3), (1, 1, 1, 1), (1, 1), (1, 1))
        self.assertEqual(tuple(new_out.shape), (2, 12, 8, 8))
        self.assertTrue(torch.equal(new_out, legacy_out))

    def test_non_square_kernel(self):
        new_out, legacy_out = self._run(8, 12, (3, 5), (1, 1, 1, 1), (1, 1), (1, 1))
        self.assertEqual(tuple(new_out.shape), (2, 12, 8, 6))
        self.assertTrue(torch.equal(new_out, legacy_out))

    def test_asymmetric_pads_non_square_geometry(self):
        pads, strides, dilations = (1, 0, 2, 0), (2, 1), (1, 2)
        new_out, legacy_out = self._run(6, 10, (3, 5), pads, strides, dilations)
        self.assertEqual(tuple(new_out.shape), (2, 10, 4, 2))
        self.assertTrue(torch.equal(new_out, legacy_out))

    def test_groups(self):
        new_out, legacy_out = self._run(
            6, 12, (3, 3), (1, 1, 1, 1), (1, 1), (1, 1), groups=2
        )
        self.assertEqual(tuple(new_out.shape), (2, 12, 8, 8))
        self.assertTrue(torch.equal(new_out, legacy_out))

    def test_no_bias(self):
        new_out, legacy_out = self._run(
            8, 12, (3, 3), (1, 1, 1, 1), (1, 1), (1, 1), use_bias=False
        )
        self.assertTrue(torch.equal(new_out, legacy_out))

    def test_subnormal_inputs(self):
        new_out, legacy_out = self._run(
            8, 16, (3, 3), (1, 1, 1, 1), (1, 1), (1, 1), value_scale=1e-5
        )
        self.assertTrue(torch.equal(new_out, legacy_out))
        self.assertTrue(torch.any(new_out != 0))

    def test_extreme_values_saturation(self):
        x = torch.full((2, 8, 8, 8), 65504.0, dtype=torch.float16)
        int8_weight = torch.full((12, 8, 3, 3), 127, dtype=torch.int8)
        ascale = torch.full((12,), 2.0**-20, dtype=torch.float32)
        args = (
            x,
            int8_weight,
            None,
            ascale,
            (1, 1, 1, 1),
            (1, 1),
            (1, 1),
            1,
            SV_S8,
        )
        new_out = new_conv_f16s8(*args)
        legacy_out = conv_f16s8(*args)
        self.assertTrue(torch.equal(new_out, legacy_out))
        self.assertTrue(torch.isfinite(new_out.float()).all())
        self.assertTrue(torch.all(new_out == 2048.0))

    def test_scalar_antiquant_scale(self):
        torch.manual_seed(3)
        x = (torch.randn(2, 8, 8, 8) * 0.5).to(torch.float16)
        int8_weight = torch.randint(-127, 128, (12, 8, 3, 3), dtype=torch.int8)
        bias = (torch.randn(12) * 0.1).to(torch.float16)
        ascale = torch.tensor(0.003 * 2.0 ** (SV_S8 - 29))
        args = (
            x,
            int8_weight,
            bias,
            ascale,
            (1, 1, 1, 1),
            (1, 1),
            (1, 1),
            1,
            SV_S8,
        )
        self.assertTrue(torch.equal(new_conv_f16s8(*args), conv_f16s8(*args)))

    def test_candidate_range_extremes(self):
        for sv in (5, 14):
            new_out, legacy_out = self._run(
                8, 12, (3, 3), (1, 1, 1, 1), (1, 1), (1, 1), sv=sv
            )
            self.assertTrue(torch.equal(new_out, legacy_out))


class TestNewConvtS8ParityWithLegacy(unittest.TestCase):
    """Bit-exact parity of the refactored convt_f16s8 vs the legacy reference."""

    def setUp(self):
        torch.manual_seed(2026)

    def _run(
        self,
        cin,
        cout,
        kernel,
        pads,
        strides,
        dilations,
        groups=1,
        use_bias=True,
        output_padding=(0, 0),
        value_scale=0.5,
        weight_scale=0.1,
        sv=SV_S8,
    ):
        kh, kw = kernel
        x = (torch.randn(2, cin, 8, 8) * value_scale).to(torch.float16)
        weight = (torch.randn(cin, cout // groups, kh, kw) * weight_scale).to(
            torch.float16
        )
        int8_weight, w_scale = _quant_weight_symmetric(weight, axis=1)
        bias = (torch.randn(cout) * 0.1).to(torch.float16) if use_bias else None
        pad_t, pad_b, pad_l, pad_r = pads
        stride_h, stride_w = strides
        dilation_h, dilation_w = dilations
        opad_h, opad_w = output_padding
        h_out = (8 - 1) * stride_h + (kh - 1) * dilation_h + 1 - pad_t - pad_b + opad_h
        w_out = (8 - 1) * stride_w + (kw - 1) * dilation_w + 1 - pad_l - pad_r + opad_w
        output_shape = (2, cout, h_out, w_out)
        args = (
            x,
            int8_weight,
            bias,
            _accumulator_scale(w_scale, sv).repeat(groups),
            pads,
            strides,
            dilations,
            groups,
            output_shape,
            sv,
        )
        return new_convt_f16s8(*args), convt_f16s8(*args), output_shape

    def test_basic_geometry(self):
        new_out, legacy_out, shape = self._run(
            8, 16, (3, 3), (0, 0, 0, 0), (1, 1), (1, 1)
        )
        self.assertEqual(tuple(new_out.shape), (2, 16, 10, 10))
        self.assertTrue(torch.equal(new_out, legacy_out))

    def test_non_square_kernel(self):
        new_out, legacy_out, shape = self._run(
            8, 16, (2, 4), (0, 0, 0, 0), (1, 1), (1, 1)
        )
        self.assertEqual(tuple(new_out.shape), (2, 16, 9, 11))
        self.assertTrue(torch.equal(new_out, legacy_out))

    def test_non_square_stride_dilation_with_output_padding(self):
        pads, strides, dilations, opad = (1, 1, 0, 0), (2, 1), (1, 2), (1, 0)
        new_out, legacy_out, shape = self._run(
            8, 16, (3, 3), pads, strides, dilations, output_padding=opad
        )
        self.assertEqual(tuple(new_out.shape), (2, 16, 16, 12))
        self.assertTrue(torch.equal(new_out, legacy_out))

    def test_asymmetric_pads(self):
        pads, strides, dilations = (1, 0, 2, 0), (1, 1), (1, 1)
        new_out, legacy_out, shape = self._run(8, 16, (3, 3), pads, strides, dilations)
        self.assertEqual(tuple(new_out.shape), (2, 16, 9, 8))
        self.assertTrue(torch.equal(new_out, legacy_out))

    def test_groups(self):
        new_out, legacy_out, shape = self._run(
            8, 16, (3, 3), (0, 0, 0, 0), (1, 1), (1, 1), groups=2
        )
        self.assertEqual(tuple(new_out.shape), (2, 16, 10, 10))
        self.assertTrue(torch.equal(new_out, legacy_out))

    def test_no_bias(self):
        new_out, legacy_out, shape = self._run(
            8, 16, (3, 3), (0, 0, 0, 0), (1, 1), (1, 1), use_bias=False
        )
        self.assertTrue(torch.equal(new_out, legacy_out))

    def test_subnormal_inputs(self):
        new_out, legacy_out, shape = self._run(
            8, 16, (3, 3), (0, 0, 0, 0), (1, 1), (1, 1), value_scale=1e-5
        )
        self.assertTrue(torch.equal(new_out, legacy_out))
        self.assertTrue(torch.any(new_out != 0))

    def test_extreme_values_saturation(self):
        x = torch.full((2, 8, 8, 8), 65504.0, dtype=torch.float16)
        int8_weight = torch.full((8, 16, 3, 3), 127, dtype=torch.int8)
        ascale = torch.full((16,), 2.0**-20, dtype=torch.float32)
        output_shape = (2, 16, 10, 10)
        args = (
            x,
            int8_weight,
            None,
            ascale,
            (0, 0, 0, 0),
            (1, 1),
            (1, 1),
            1,
            output_shape,
            SV_S8,
        )
        new_out = new_convt_f16s8(*args)
        legacy_out = convt_f16s8(*args)
        self.assertTrue(torch.equal(new_out, legacy_out))
        self.assertTrue(torch.isfinite(new_out.float()).all())
        self.assertTrue(torch.all(new_out == 2048.0))


if __name__ == '__main__':
    unittest.main()
