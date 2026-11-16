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
"""Functional and precision tests of the fp16fixed conv2d references.

Covers the legacy reference directly and checks the refactored
functional entry (``conv2d.conv_f16f16``) bit-exactly against it
on the geometry / dynamic range combinations the class level tests do
not reach (non square kernels, asymmetric pads, subnormal inputs,
saturating extremes), plus unit tests of the fixed point helpers.
"""

import unittest

import pytest

import numpy as np

import torch
import torch.nn.functional as F

from .reference.conv2d import (
    _rint_half_even,
    conv_f16f16,
)
from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.fp16_fixed_point.conv2d import (
    _saturate_to_int32,
    to_fixed_point_source_with_conv_clamp as _to_fixed_point_source,
)
from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.fp16_fixed_point.conv2d import (
    conv_f16f16 as new_conv_f16f16,
)
from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.fp16_fixed_point.conv2d import (
    _matmul_f16f16_int32,
    _to_fp16_single_round,
)

pytestmark = [pytest.mark.cpu, pytest.mark.slow]


def _conv_ref(x, w, bias, stride, padding, dilation, groups):
    return F.conv2d(
        x.float(),
        w.float(),
        bias=bias.float() if bias is not None else None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


class TestLegacyConvF16F16(unittest.TestCase):
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
    ):
        x = (torch.randn(n, cin, h, w) * 0.5).to(torch.float16)
        weight = (torch.randn(cout, cin // groups, k, k) * 0.1).to(torch.float16)
        bias = (torch.randn(cout) * 0.1).to(torch.float16) if use_bias else None
        pads = (padding, padding, padding, padding)
        out = conv_f16f16(
            x,
            weight,
            bias,
            pads,
            (stride, stride),
            (dilation, dilation),
            groups,
            42,
        )
        return out, x, weight, bias

    def _assert_precision(self, out, ref):
        rel = (out.float() - ref).abs() / (ref.abs() + 1e-2)
        self.assertLess(rel.median().item(), 1e-2)
        self.assertLess(rel.max().item(), 5e-2)

    def test_basic_conv(self):
        out, x, w, b = self._run(2, 8, 16, 3, 8, 8, 1, 1, 1, 1, True)
        self.assertEqual(out.dtype, torch.float16)
        self.assertEqual(tuple(out.shape), (2, 16, 8, 8))
        self._assert_precision(out, _conv_ref(x, w, b, 1, 1, 1, 1))

    def test_stride_padding_dilation_combos(self):
        for stride, padding, dilation, h_out in (
            (1, 0, 1, 6),
            (2, 1, 1, 4),
            (1, 2, 2, 8),
            (2, 0, 1, 3),
        ):
            out, x, w, b = self._run(
                2, 8, 16, 3, 8, 8, stride, padding, dilation, 1, True
            )
            self.assertEqual(tuple(out.shape), (2, 16, h_out, h_out))
            self._assert_precision(
                out, _conv_ref(x, w, b, stride, padding, dilation, 1)
            )

    def test_groups(self):
        for groups in (2, 4):
            out, x, w, b = self._run(2, 8, 16, 3, 8, 8, 1, 1, 1, groups, True)
            self.assertEqual(tuple(out.shape), (2, 16, 8, 8))
            self._assert_precision(out, _conv_ref(x, w, b, 1, 1, 1, groups))

    def test_k_padding_cin_not_multiple_of_16(self):
        out, x, w, b = self._run(2, 6, 12, 3, 8, 8, 1, 1, 1, 1, True)
        self.assertEqual(tuple(out.shape), (2, 12, 8, 8))
        self._assert_precision(out, _conv_ref(x, w, b, 1, 1, 1, 1))

    def test_without_bias(self):
        out, x, w, _ = self._run(2, 8, 16, 3, 8, 8, 1, 1, 1, 1, False)
        self._assert_precision(out, _conv_ref(x, w, None, 1, 1, 1, 1))

    def test_fixed_shift_value_sweep(self):
        for sv in (34, 38, 42, 43):
            x = (torch.randn(2, 8, 8, 8) * 0.5).to(torch.float16)
            w = (torch.randn(16, 8, 3, 3) * 0.1).to(torch.float16)
            b = (torch.randn(16) * 0.1).to(torch.float16)
            out = conv_f16f16(x, w, b, (1, 1, 1, 1), (1, 1), (1, 1), 1, sv)
            ref = _conv_ref(x, w, b, 1, 1, 1, 1)
            rel = (out.float() - ref).abs() / (ref.abs() + 1e-2)
            self.assertLess(rel.median().item(), 1e-2)


class TestNewConvF16F16ParityWithLegacy(unittest.TestCase):
    """Bit-exact parity of the refactored entry vs the legacy reference.

    The combinations here extend the class level parity matrix: non
    square kernels, asymmetric pads, non square strides / dilations,
    subnormal inputs and saturating extreme values.
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
    ):
        kh, kw = kernel
        x = (torch.randn(2, cin, 8, 8) * value_scale).to(torch.float16)
        weight = (torch.randn(cout, cin // groups, kh, kw) * weight_scale).to(
            torch.float16
        )
        bias = (torch.randn(cout) * 0.1).to(torch.float16) if use_bias else None
        new_out = new_conv_f16f16(x, weight, bias, pads, strides, dilations, groups, 42)
        legacy_out = conv_f16f16(x, weight, bias, pads, strides, dilations, groups, 42)
        return new_out, legacy_out, x, weight, bias

    def _assert_precision(self, out, ref):
        rel = (out.float() - ref).abs() / (ref.abs() + 1e-2)
        self.assertLess(rel.median().item(), 1e-2)
        self.assertLess(rel.max().item(), 5e-2)

    def _conv_ref(self, x, weight, bias, pads, strides, dilations, groups):
        pad_top, pad_bottom, pad_left, pad_right = pads
        padded = F.pad(x.float(), (pad_left, pad_right, pad_top, pad_bottom))
        return F.conv2d(
            padded,
            weight.float(),
            bias=bias.float() if bias is not None else None,
            stride=strides,
            dilation=dilations,
            groups=groups,
        )

    def test_non_square_kernel(self):
        new_out, legacy_out, x, w, b = self._run(
            8, 12, (3, 5), (1, 1, 1, 1), (1, 1), (1, 1)
        )
        self.assertEqual(tuple(new_out.shape), (2, 12, 8, 6))
        self.assertTrue(torch.equal(new_out, legacy_out))
        self._assert_precision(
            new_out, self._conv_ref(x, w, b, (1, 1, 1, 1), (1, 1), (1, 1), 1)
        )

    def test_asymmetric_pads_non_square_geometry(self):
        pads, strides, dilations = (1, 0, 2, 0), (2, 1), (1, 2)
        new_out, legacy_out, x, w, b = self._run(
            6, 10, (3, 5), pads, strides, dilations
        )
        # out_h = (8 + 1 + 0 - 3) // 2 + 1 = 4, out_w = (8 + 2 + 0 - 9) + 1 = 2
        self.assertEqual(tuple(new_out.shape), (2, 10, 4, 2))
        self.assertTrue(torch.equal(new_out, legacy_out))
        self._assert_precision(
            new_out, self._conv_ref(x, w, b, pads, strides, dilations, 1)
        )

    def test_groups_with_partial_channel_block(self):
        new_out, legacy_out, x, w, b = self._run(
            6, 12, (3, 3), (1, 1, 1, 1), (1, 1), (1, 1), groups=2
        )
        self.assertEqual(tuple(new_out.shape), (2, 12, 8, 8))
        self.assertTrue(torch.equal(new_out, legacy_out))
        self._assert_precision(
            new_out, self._conv_ref(x, w, b, (1, 1, 1, 1), (1, 1), (1, 1), 2)
        )

    def test_subnormal_inputs(self):
        new_out, legacy_out, x, w, b = self._run(
            8, 16, (3, 3), (1, 1, 1, 1), (1, 1), (1, 1), value_scale=1e-5
        )
        self.assertTrue(torch.equal(new_out, legacy_out))
        self.assertTrue(torch.any(new_out != 0))
        self._assert_precision(
            new_out, self._conv_ref(x, w, b, (1, 1, 1, 1), (1, 1), (1, 1), 1)
        )

    def test_extreme_values_saturation(self):
        x = torch.full((2, 8, 8, 8), 65504.0, dtype=torch.float16)
        weight = torch.full((12, 8, 3, 3), 65504.0, dtype=torch.float16)
        bias = torch.zeros(12, dtype=torch.float16)
        new_out = new_conv_f16f16(x, weight, bias, (1, 1, 1, 1), (1, 1), (1, 1), 1, 42)
        legacy_out = conv_f16f16(x, weight, bias, (1, 1, 1, 1), (1, 1), (1, 1), 1, 42)
        self.assertTrue(torch.equal(new_out, legacy_out))
        self.assertTrue(torch.isfinite(new_out.float()).all())
        # the saturated accumulator dequantizes onto the largest fp16 value
        self.assertTrue(torch.all(new_out == 32768.0))


class TestConvHelpers(unittest.TestCase):
    """Unit tests of the fixed point helpers of the refactored module."""

    def _fixed_point_of(self, values):
        bits = values.contiguous().view(torch.int16).to(torch.int32)
        sign = (bits >> 15) & 1
        exponent = (bits >> 10) & 0x1F
        mantissa = bits & 0x3FF
        is_normal = exponent > 0
        tile_max = exponent.amax().to(torch.int64)
        fixed_point = _to_fixed_point_source(
            exponent, mantissa, sign, is_normal, tile_max
        )
        return fixed_point, tile_max

    def test_rint_half_even_ties_round_to_even(self):
        product = torch.tensor([5, 7, 3, 1, 6, 14, 13, 11], dtype=torch.int64)
        shift = torch.tensor([1, 1, 1, 1, 2, 2, 2, 2])
        expect = torch.tensor([2, 4, 2, 0, 2, 4, 3, 3], dtype=torch.int64)
        self.assertTrue(torch.equal(_rint_half_even(product, shift), expect))

    def test_rint_half_even_zero_shift_behaviour(self):
        # faithful to the reference: at shift 0 every value is a
        # degenerate exact tie (remainder == half == 0), so odd values
        # carry up to the next even number
        product = torch.arange(-5, 6, dtype=torch.int64)
        shift = torch.zeros_like(product)
        expect = product + (product & 1)
        self.assertTrue(torch.equal(_rint_half_even(product, shift), expect))

    def test_saturate_to_int32_collapses_to_min(self):
        values = torch.tensor(
            [2**31, -(2**31) - 1, 2**31 - 1, -(2**31), 5, -5], dtype=torch.int64
        )
        expect = torch.tensor(
            [-(2**31), -(2**31), 2**31 - 1, -(2**31), 5, -5], dtype=torch.int64
        )
        self.assertTrue(torch.equal(_saturate_to_int32(values), expect))

    def test_to_fixed_point_source_grid_normals(self):
        # every element lands on the grid value = fixed_point *
        # 2 ** (tile_max_exponent - 29); the exponents stay close enough
        # that no alignment clamp kicks in
        values = torch.tensor(
            [
                1.0,
                2.0,
                0.5,
                -1.5,
                -0.25,
                6.0,
                0.75,
                -3.5,
                1.25,
                -0.625,
                4.0,
                0.125,
                -2.75,
                8.0,
                0.0625,
                -5.5,
            ],
            dtype=torch.float16,
        )
        fixed_point, tile_max = self._fixed_point_of(values)
        recovered = fixed_point.to(torch.float64) * 2.0 ** (tile_max - 29)
        self.assertTrue(
            torch.equal(recovered.to(torch.float16), values),
            'grid relation value = fixed_point * 2 ** (E - 29) broken',
        )

    def test_to_fixed_point_source_grid_subnormals(self):
        # subnormal value = m * 2 ** -24 lands on m << 5 of the grid
        values = torch.tensor(
            [2.0**-24, 2.0**-23, 3.0 * 2.0**-24, 0.0, 5.0 * 2.0**-24],
            dtype=torch.float16,
        )
        fixed_point, tile_max = self._fixed_point_of(values)
        self.assertEqual(int(tile_max), 0)
        self.assertTrue(
            torch.equal(
                fixed_point, torch.tensor([32, 64, 96, 0, 160], dtype=torch.int64)
            )
        )
        recovered = fixed_point.to(torch.float64) * 2.0**-29
        self.assertTrue(torch.equal(recovered.to(torch.float16), values))

    def test_to_fixed_point_source_clamps_tiny_elements_to_zero(self):
        # a subnormal sitting next to the fp16 max in one tile loses all
        # its mantissa bits to the [-11, 5] alignment clamp
        values = torch.tensor([65504.0, 2.0**-20], dtype=torch.float16)
        fixed_point, tile_max = self._fixed_point_of(values)
        self.assertEqual(int(tile_max), 30)
        self.assertEqual(int(fixed_point[0]), 2047 << 4)
        self.assertEqual(int(fixed_point[1]), 0)


class TestConv2dZeroShiftAccumulatorParity(unittest.TestCase):
    """Accumulator level parity of the R5 fast path against the exact
    pre-R5 integer pipeline (PR review Major).

    An ODD product at shift == 0 rounds UP to the even neighbour in the
    frozen contract (an exact tie of the half-even shifter, see
    ``_rint_half_even``), while ``round()`` leaves an integer unchanged;
    the fp16 output cannot show the +/-1 accumulator step (its ulp is
    far coarser at the shift-0 output scale), so the parity is pinned on
    the INT32 accumulator itself."""

    @staticmethod
    def _zero_shift_matrices():
        torch.manual_seed(3)
        # one tile of 16: column 0 carries the tile max (raw exponent
        # 21, value ~2**6), the rest stay tiny (raw exponent 15) so the
        # tiny-tiny products keep odd-capable aligned fixed points
        # (aligned_exp 4 - 6 == -2, mantissa >> 2 can be odd)
        im = torch.empty(9, 16, dtype=torch.float16)
        im[:, 0] = (64.0 + torch.rand(9) * 0.5).to(torch.float16)
        im[:, 1:] = (1.0 + torch.rand(9, 15)).to(torch.float16)
        wm = torch.empty(7, 16, dtype=torch.float16)
        wm[:, 0] = (64.0 + torch.rand(7) * 0.5).to(torch.float16)
        wm[:, 1:] = (1.0 + torch.rand(7, 15)).to(torch.float16)
        # every (m, n) shift: 42 - 21 - 21 == 0
        return im, wm

    def test_zero_shift_accumulator_parity(self):
        im, wm = self._zero_shift_matrices()
        bias = torch.zeros(wm.shape[0], dtype=torch.float32)
        fast = _matmul_f16f16_int32(im, wm, bias, 42)
        truth = _pre_r5_matmul_f16f16_int32(im, wm, bias, 42)
        self.assertTrue(torch.equal(fast, truth))
        # non-vacuous: the reference carry maps ODD products to the
        # even neighbour, so both fast and truth are all even by
        # construction -- the guard must assert that odd raw products
        # exist (decode + one tile MAC, mirroring the engine decode)
        bits_i = im.view(torch.int16).to(torch.int32)
        bits_w = wm.view(torch.int16).to(torch.int32)

        def _fixed(bits):
            exp = (bits >> 10) & 0x1F
            man = (bits & 0x3FF).to(torch.int64) + 1024
            tile_max = exp.amax(1, keepdim=True)
            aligned = (4 - (tile_max - exp)).clamp(-11, 5)
            return torch.where(
                aligned >= 0,
                man << aligned.clamp_min(0),
                man >> (-aligned).clamp_min(0),
            ).to(torch.float64)

        mac = _fixed(bits_i) @ _fixed(bits_w).T
        self.assertTrue(bool(((mac % 2) != 0).any()))

    def test_accumulator_parity_random_sweep(self):
        # the reviewer's differential methodology: random shapes /
        # scales / sv, fast path vs the pre-R5 integer pipeline
        for seed in range(30):
            g = torch.Generator().manual_seed(seed)
            rows = int(torch.randint(1, 12, (1,), generator=g))
            n_out = int(torch.randint(1, 12, (1,), generator=g))
            tiles = int(torch.randint(1, 4, (1,), generator=g))
            scale_a = 2.0 ** int(torch.randint(-8, 8, (1,), generator=g))
            scale_b = 2.0 ** int(torch.randint(-8, 8, (1,), generator=g))
            im = (torch.randn(rows, tiles * 16, generator=g) * scale_a).to(
                torch.float16
            )
            wm = (torch.randn(n_out, tiles * 16, generator=g) * scale_b).to(
                torch.float16
            )
            bias = torch.randn(n_out, generator=g).to(torch.float32)
            sv = int(torch.randint(34, 44, (1,), generator=g))
            fast = _matmul_f16f16_int32(im, wm, bias, sv)
            truth = _pre_r5_matmul_f16f16_int32(im, wm, bias, sv)
            self.assertTrue(
                torch.equal(fast, truth),
                "seed {} sv {} accumulator mismatch".format(seed, sv),
            )


INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1


def _pre_r5_matmul_f16f16_int32(
    input_matrix, weight_matrix, bias, fixed_shift_value=42
):
    """F16xF16 matmul of one im2col block, returns the INT32 accumulator.

    ``input_matrix`` is ``[rows, reduction]``, ``weight_matrix`` is
    ``[output_channels, reduction]``, both fp16; the K axis is processed
    in tiles of 16.
    """
    num_rows, reduction_size = input_matrix.shape
    num_output_channels, _ = weight_matrix.shape
    if reduction_size % 16:
        pad_size = 16 - reduction_size % 16
        input_matrix = torch.nn.functional.pad(input_matrix, (0, pad_size))
        weight_matrix = torch.nn.functional.pad(weight_matrix, (0, pad_size))
        reduction_size = input_matrix.shape[-1]

    # decode the fp16 bit patterns: sign(1) | raw exponent(5) | mantissa(10)
    input_bits = input_matrix.contiguous().view(torch.int16).to(torch.int32)
    weight_bits = weight_matrix.contiguous().view(torch.int16).to(torch.int32)
    input_sign = (input_bits >> 15) & 1
    input_exponent = (input_bits >> 10) & 0x1F
    input_mantissa = input_bits & 0x3FF
    weight_sign = (weight_bits >> 15) & 1
    weight_exponent = (weight_bits >> 10) & 0x1F
    weight_mantissa = weight_bits & 0x3FF
    input_is_normal = input_exponent > 0
    weight_is_normal = weight_exponent > 0

    # pre-scale the bias onto the accumulator grid (LSB = 2**(sv-58) of
    # real value); this op adds it after the last tile
    scaled_bias = torch.round(
        bias.to(torch.float32) * float(2 ** (58 - fixed_shift_value))
    ).to(torch.int64)
    bias_int32 = _saturate_to_int32(scaled_bias)

    accumulator = torch.zeros(
        (num_rows, num_output_channels),
        dtype=torch.int64,
        device=input_matrix.device,
    )

    # vectorized: compute all tile fixed points and max exponents at once
    # (replaces the per-tile _to_fixed_point_source calls in the loop)
    num_tiles = reduction_size // 16

    # input side: reshape K -> (tiles, 16), compute per-tile amax, align all
    input_exp_tiles = input_exponent.reshape(num_rows, num_tiles, 16)
    input_tile_max_all = input_exp_tiles.amax(-1).to(torch.int64)  # (rows, T)

    input_exp_base_all = (
        torch.where(input_exponent > 0, 4, 5)
        .to(torch.int64)
        .reshape(num_rows, num_tiles, 16)
    )
    input_man_all = torch.where(
        input_is_normal,
        input_mantissa.to(torch.int64) + 1024,
        input_mantissa.to(torch.int64),
    ).reshape(num_rows, num_tiles, 16)
    input_sign_all = (
        torch.where(input_sign == 1, -1, 1)
        .to(torch.int64)
        .reshape(num_rows, num_tiles, 16)
    )

    input_delta = input_tile_max_all.unsqueeze(-1) - input_exp_tiles.to(torch.int64)
    # the per-element exponent clamp of the conv engine is [-11, 5]
    # (the frozen reference contract, see _to_fixed_point_source and
    # the numpy reference) -- NOT the [-16, 15] of the matmul / convt
    # engines. With the truncating alignment shift the two bounds are
    # observationally equivalent (every element whose unclamped
    # exponent lands in [-16, -12] decodes to zero under either bound
    # because the adjusted mantissa never exceeds 2047 < 2**11);
    # carrying the reference bounds keeps the engine auditable against
    # its frozen contract.
    input_aligned_exp = (input_exp_base_all - input_delta).clamp(-11, 5)
    input_fixed = torch.where(
        input_aligned_exp >= 0,
        input_man_all << input_aligned_exp.clamp_min(0),
        input_man_all >> (-input_aligned_exp).clamp_min(0),
    )
    input_all_fp = (
        input_fixed.to(torch.float64) * input_sign_all.to(torch.float64)
    ).reshape(num_rows, -1)  # (rows, K)

    # weight side: same vectorized alignment
    weight_exp_tiles = weight_exponent.reshape(num_output_channels, num_tiles, 16)
    weight_tile_max_all = weight_exp_tiles.amax(-1).to(torch.int64)  # (N, T)

    weight_exp_base_all = (
        torch.where(weight_exponent > 0, 4, 5)
        .to(torch.int64)
        .reshape(num_output_channels, num_tiles, 16)
    )
    weight_man_all = torch.where(
        weight_is_normal,
        weight_mantissa.to(torch.int64) + 1024,
        weight_mantissa.to(torch.int64),
    ).reshape(num_output_channels, num_tiles, 16)
    weight_sign_all = (
        torch.where(weight_sign == 1, -1, 1)
        .to(torch.int64)
        .reshape(num_output_channels, num_tiles, 16)
    )

    weight_delta = weight_tile_max_all.unsqueeze(-1) - weight_exp_tiles.to(torch.int64)
    weight_aligned_exp = (weight_exp_base_all - weight_delta).clamp(-11, 5)
    weight_fixed = torch.where(
        weight_aligned_exp >= 0,
        weight_man_all << weight_aligned_exp.clamp_min(0),
        weight_man_all >> (-weight_aligned_exp).clamp_min(0),
    )
    weight_all_fp = (
        weight_fixed.to(torch.float64) * weight_sign_all.to(torch.float64)
    ).reshape(num_output_channels, -1)  # (N, K)

    # one Cube instruction processes 16 consecutive K elements of both
    # operands (K was zero padded to a multiple of 16 above)
    for tile_index in range(num_tiles):
        col_start = 16 * tile_index
        col_end = 16 * tile_index + 16

        # pre-computed per-tile alignment anchors
        input_tmax = input_tile_max_all[:, tile_index]  # (rows,)
        weight_tmax = weight_tile_max_all[:, tile_index]  # (N,)

        # pre-computed fixed points for this tile
        input_fp = input_all_fp[:, col_start:col_end]  # (rows, 16)
        weight_fp = weight_all_fp[:, col_start:col_end]  # (N, 16)

        # 16-element integer MAC using BLAS matmul (exact in float64
        # for these magnitudes), in units of real_value * 2**(58 - Ea - Eb)
        tile_mac = (input_fp @ weight_fp.T).to(torch.int64)  # (rows, N), exact
        # rescale onto the global accumulator grid (LSB = 2**(sv-58)):
        # tile_mac * 2**(Ea + Eb - sv); the clamp is the hardware
        # shifter range
        tile_shift = torch.clamp(
            fixed_shift_value - input_tmax.unsqueeze(1) - weight_tmax.unsqueeze(0),
            -26,
            36,
        )
        zero_shift = torch.zeros_like(tile_shift)
        left_shift_amount = torch.where(tile_shift >= 0, tile_shift, zero_shift)
        right_shift_amount = torch.where(tile_shift < 0, -tile_shift, zero_shift)
        # a positive shift rounds half to even, a negative shift is an
        # exact left shift
        tile_result = torch.where(
            tile_shift >= 0,
            _rint_half_even(tile_mac, left_shift_amount),
            tile_mac << right_shift_amount,
        )
        # saturate to INT32 after every tile (hardware behaviour)
        accumulator = torch.clamp(accumulator + tile_result, INT32_MIN, INT32_MAX)
    # the bias joins only after the last tile (unlike convt / matmul)
    accumulator = torch.clamp(
        accumulator + bias_int32.unsqueeze(0), INT32_MIN, INT32_MAX
    )
    return accumulator


class TestDequantSingleRound(unittest.TestCase):
    """The dequant cast must round ONCE (f64 -> fp16), like the frozen
    references' numpy astype. torch's cast routes through float32, so
    values adjacent to an fp16 midpoint double round (the reviewer's
    seed 91 / sv 34 example: accumulator 2100822015, real value
    125.21874994..., just below the 125.21875 midpoint)."""

    def test_midpoint_neighbour_single_rounds_down(self):
        k = torch.tensor([2100822015], dtype=torch.int64)
        v = k.to(torch.float64) * 2.0**-24
        self.assertEqual(_to_fp16_single_round(v).item(), 125.1875)
        # and the double rounded torch cast crosses the midpoint
        self.assertEqual(v.to(torch.float16).item(), 125.25)

    def test_midpoint_zone_random(self):
        g = torch.Generator().manual_seed(91)
        k = torch.randint(
            2100822015 - 500000, 2100822015 + 500000, (1000000,), generator=g
        )
        v = k.to(torch.float64) * 2.0**-24
        single = _to_fp16_single_round(v)
        ref = torch.from_numpy(v.numpy().astype(np.float16))
        self.assertTrue(torch.equal(single, ref))
        # the zone really separates the two casts (non-vacuous)
        self.assertFalse(torch.equal(single, v.to(torch.float16)))


class TestConv2dAlignmentClampBoundary(unittest.TestCase):
    """Boundary parity of the vectorized decode against the frozen
    reference on adversarial tiles.

    The conv alignment clamp contract is [-11, 5] (the reference and
    numpy bounds), NOT the [-16, 15] of the matmul / convt engines.
    With the truncating alignment shift both bounds are observationally
    equivalent -- every element whose unclamped exponent lands in
    [-16, -12] decodes to zero under either bound because the adjusted
    mantissa never exceeds 2047 < 2**11 -- so these parity cases pin
    the equivalence: any future change that makes the bounds observable
    (for example a rounding alignment shift) fails here.
    """

    def _adversarial_channels(self):
        """16 channel values whose exponent deltas against the tile max
        cover 15..20: the clamped zone boundary on the [-11, 5] side."""
        deltas = [0, 15, 16, 17, 18, 19, 20, 16, 17, 18, 19, 20, 15, 16, 17, 18]
        values = []
        for delta in deltas:
            # tile max exponent 30 (65504); a mantissa-rich value whose
            # raw exponent is 30 - delta keeps the largest possible
            # mantissa against the alignment shift
            values.append(1.999 * 2.0 ** (15 - delta))
        return torch.tensor(values, dtype=torch.float16).view(1, 16, 1, 1)

    def test_activation_side_boundary_parity(self):
        torch.manual_seed(11)
        x = self._adversarial_channels()
        weight = torch.full((4, 16, 1, 1), 1.0, dtype=torch.float16)
        bias = torch.zeros(4, dtype=torch.float16)
        new_out = new_conv_f16f16(x, weight, bias, (0, 0, 0, 0), (1, 1), (1, 1), 1, 42)
        ref_out = conv_f16f16(x, weight, bias, (0, 0, 0, 0), (1, 1), (1, 1), 1, 42)
        self.assertTrue(torch.equal(new_out, ref_out))

    def test_weight_side_boundary_parity(self):
        torch.manual_seed(12)
        x = torch.full((1, 16, 1, 1), 1.0, dtype=torch.float16)
        weight = self._adversarial_channels().expand(4, 16, 1, 1).contiguous()
        bias = torch.zeros(4, dtype=torch.float16)
        new_out = new_conv_f16f16(x, weight, bias, (0, 0, 0, 0), (1, 1), (1, 1), 1, 42)
        ref_out = conv_f16f16(x, weight, bias, (0, 0, 0, 0), (1, 1), (1, 1), 1, 42)
        self.assertTrue(torch.equal(new_out, ref_out))


if __name__ == '__main__':
    unittest.main()
