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
"""Functional and precision tests of the fp16fixed convtranspose2d references.

Covers the legacy reference directly and checks the refactored
functional entry (``convtranspose2d.convt_f16f16``) bit-exactly
against it on the geometry / dynamic range combinations the class level
tests do not reach (non square kernels, asymmetric pads, non square
strides / dilations, subnormal inputs, saturating extremes).
"""

import unittest

import pytest

import torch
import torch.nn.functional as F

from .reference.convtranspose2d import (
    convt_f16f16,
)
from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.fp16_fixed_point.convtranspose2d import (
    convt_f16f16 as new_convt_f16f16,
)

pytestmark = [pytest.mark.cpu, pytest.mark.slow]


def _convt_ref(x, w, bias, stride, padding, dilation, groups):
    return F.conv_transpose2d(
        x.float(),
        w.float(),
        bias=bias.float() if bias is not None else None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


class TestLegacyConvtF16F16(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)

    def _run(
        self,
        n,
        cin,
        cout,
        k,
        h,
        w,
        stride,
        padding,
        dilation,
        groups,
        use_bias,
        output_padding=0,
    ):
        x = (torch.randn(n, cin, h, w) * 0.5).to(torch.float16)
        # ConvTranspose weight layout: (in, out // groups, kh, kw)
        weight = (torch.randn(cin, cout // groups, k, k) * 0.1).to(torch.float16)
        bias = (torch.randn(cout) * 0.1).to(torch.float16) if use_bias else None
        h_out = (h - 1) * stride - 2 * padding + dilation * (k - 1) + output_padding + 1
        w_out = (w - 1) * stride - 2 * padding + dilation * (k - 1) + output_padding + 1
        pads = (padding, padding, padding, padding)
        out = convt_f16f16(
            x,
            weight,
            bias,
            pads,
            (stride, stride),
            (dilation, dilation),
            groups,
            (n, cout, h_out, w_out),
            42,
        )
        return out, x, weight, bias

    def _assert_precision(self, out, ref):
        rel = (out.float() - ref).abs() / (ref.abs() + 1e-2)
        self.assertLess(rel.median().item(), 1e-2)
        self.assertLess(rel.max().item(), 5e-2)

    def test_basic_convt(self):
        out, x, w, b = self._run(2, 8, 16, 3, 8, 8, 1, 0, 1, 1, True)
        self.assertEqual(out.dtype, torch.float16)
        self.assertEqual(tuple(out.shape), (2, 16, 10, 10))
        self._assert_precision(out, _convt_ref(x, w, b, 1, 0, 1, 1))

    def test_stride_padding_dilation_combos(self):
        for stride, padding, dilation, h_out in (
            (2, 1, 1, 15),
            (1, 0, 2, 12),
            (2, 0, 1, 17),
            (1, 2, 1, 6),
        ):
            out, x, w, b = self._run(
                2, 8, 16, 3, 8, 8, stride, padding, dilation, 1, True
            )
            self.assertEqual(tuple(out.shape), (2, 16, h_out, h_out))
            self._assert_precision(
                out, _convt_ref(x, w, b, stride, padding, dilation, 1)
            )

    def test_groups(self):
        for groups in (2, 4):
            out, x, w, b = self._run(2, 8, 16, 3, 8, 8, 1, 0, 1, groups, True)
            self.assertEqual(tuple(out.shape), (2, 16, 10, 10))
            self._assert_precision(out, _convt_ref(x, w, b, 1, 0, 1, groups))

    def test_without_bias(self):
        out, x, w, _ = self._run(2, 8, 16, 3, 8, 8, 1, 0, 1, 1, False)
        self._assert_precision(out, _convt_ref(x, w, None, 1, 0, 1, 1))

    def test_fixed_shift_value_sweep(self):
        for sv in (34, 38, 42, 43):
            x = (torch.randn(2, 8, 8, 8) * 0.5).to(torch.float16)
            w = (torch.randn(8, 16, 3, 3) * 0.1).to(torch.float16)
            b = (torch.randn(16) * 0.1).to(torch.float16)
            out = convt_f16f16(
                x,
                w,
                b,
                (0, 0, 0, 0),
                (1, 1),
                (1, 1),
                1,
                (2, 16, 10, 10),
                sv,
            )
            ref = _convt_ref(x, w, b, 1, 0, 1, 1)
            rel = (out.float() - ref).abs() / (ref.abs() + 1e-2)
            self.assertLess(rel.median().item(), 1e-2)


class TestNewConvtF16F16ParityWithLegacy(unittest.TestCase):
    """Bit-exact parity of the refactored entry vs the legacy reference.

    The combinations here extend the class level parity matrix: non
    square kernels, asymmetric pads, non square strides / dilations
    with output padding, subnormal inputs and saturating extremes.
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
        output_padding=(0, 0),
        value_scale=0.5,
        weight_scale=0.1,
    ):
        kh, kw = kernel
        x = (torch.randn(2, cin, 8, 8) * value_scale).to(torch.float16)
        weight = (torch.randn(cin, cout // groups, kh, kw) * weight_scale).to(
            torch.float16
        )
        bias = (torch.randn(cout) * 0.1).to(torch.float16) if use_bias else None
        pad_top, pad_bottom, pad_left, pad_right = pads
        stride_h, stride_w = strides
        dilation_h, dilation_w = dilations
        opad_h, opad_w = output_padding
        h_out = (
            (8 - 1) * stride_h
            + (kh - 1) * dilation_h
            + 1
            - pad_top
            - pad_bottom
            + opad_h
        )
        w_out = (
            (8 - 1) * stride_w
            + (kw - 1) * dilation_w
            + 1
            - pad_left
            - pad_right
            + opad_w
        )
        output_shape = (2, cout, h_out, w_out)
        new_out = new_convt_f16f16(
            x, weight, bias, pads, strides, dilations, groups, output_shape, 42
        )
        legacy_out = convt_f16f16(
            x, weight, bias, pads, strides, dilations, groups, output_shape, 42
        )
        return new_out, legacy_out, x, weight, bias, output_shape

    def _assert_precision(self, out, ref):
        rel = (out.float() - ref).abs() / (ref.abs() + 1e-2)
        self.assertLess(rel.median().item(), 1e-2)
        self.assertLess(rel.max().item(), 5e-2)

    def _convt_ref(
        self, x, weight, bias, pads, strides, dilations, groups, output_shape
    ):
        """Asymmetric-pad reference: full conv transpose + cropping."""
        pad_top, _, pad_left, _ = pads
        _, _, h_out, w_out = output_shape
        full = F.conv_transpose2d(
            x.float(),
            weight.float(),
            stride=strides,
            dilation=dilations,
            groups=groups,
        )
        out = full[:, :, pad_top : pad_top + h_out, pad_left : pad_left + w_out]
        if bias is not None:
            out = out + bias.float().view(1, -1, 1, 1)
        return out

    def test_non_square_kernel(self):
        new_out, legacy_out, x, w, b, shape = self._run(
            8, 16, (2, 4), (0, 0, 0, 0), (1, 1), (1, 1)
        )
        self.assertEqual(tuple(new_out.shape), (2, 16, 9, 11))
        self.assertTrue(torch.equal(new_out, legacy_out))
        self._assert_precision(
            new_out, self._convt_ref(x, w, b, (0, 0, 0, 0), (1, 1), (1, 1), 1, shape)
        )

    def test_non_square_stride_dilation_with_output_padding(self):
        pads, strides, dilations, opad = (1, 1, 0, 0), (2, 1), (1, 2), (1, 0)
        new_out, legacy_out, x, w, b, shape = self._run(
            8, 16, (3, 3), pads, strides, dilations, output_padding=opad
        )
        # h_out = 7 * 2 + 2 + 1 - 1 - 1 + 1 = 16, w_out = 7 + 4 + 1 = 12
        self.assertEqual(tuple(new_out.shape), (2, 16, 16, 12))
        self.assertTrue(torch.equal(new_out, legacy_out))
        ref = F.conv_transpose2d(
            x.float(),
            w.float(),
            bias=b.float(),
            stride=strides,
            padding=(1, 0),
            dilation=dilations,
            output_padding=opad,
        )
        self._assert_precision(new_out, ref)

    def test_asymmetric_pads(self):
        pads, strides, dilations = (1, 0, 2, 0), (1, 1), (1, 1)
        new_out, legacy_out, x, w, b, shape = self._run(
            8, 16, (3, 3), pads, strides, dilations
        )
        # h_out = 7 + 2 + 1 - 1 - 0 = 9, w_out = 7 + 2 + 1 - 2 - 0 = 8
        self.assertEqual(tuple(new_out.shape), (2, 16, 9, 8))
        self.assertTrue(torch.equal(new_out, legacy_out))
        self._assert_precision(
            new_out, self._convt_ref(x, w, b, pads, strides, dilations, 1, shape)
        )

    def test_subnormal_inputs(self):
        new_out, legacy_out, x, w, b, shape = self._run(
            8, 16, (3, 3), (0, 0, 0, 0), (1, 1), (1, 1), value_scale=1e-5
        )
        self.assertTrue(torch.equal(new_out, legacy_out))
        self.assertTrue(torch.any(new_out != 0))
        self._assert_precision(
            new_out, self._convt_ref(x, w, b, (0, 0, 0, 0), (1, 1), (1, 1), 1, shape)
        )

    def test_extreme_values_saturation(self):
        x = torch.full((2, 8, 8, 8), 65504.0, dtype=torch.float16)
        weight = torch.full((8, 16, 3, 3), 65504.0, dtype=torch.float16)
        bias = torch.zeros(16, dtype=torch.float16)
        output_shape = (2, 16, 10, 10)
        new_out = new_convt_f16f16(
            x, weight, bias, (0, 0, 0, 0), (1, 1), (1, 1), 1, output_shape, 42
        )
        legacy_out = convt_f16f16(
            x, weight, bias, (0, 0, 0, 0), (1, 1), (1, 1), 1, output_shape, 42
        )
        self.assertTrue(torch.equal(new_out, legacy_out))
        self.assertTrue(torch.isfinite(new_out.float()).all())


if __name__ == '__main__':
    unittest.main()
