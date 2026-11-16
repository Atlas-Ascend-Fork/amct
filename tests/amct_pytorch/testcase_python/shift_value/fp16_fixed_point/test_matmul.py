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
"""Full test suite of the fp16fixed matmul PyTorch implementation.

Every compute path is checked bit-exactly against the numpy reference
``reference/matmul_np.py``, plus fp32-reference sanity checks and
interface contract checks.
"""

import unittest

import pytest

import numpy as np
import torch

from .reference.matmul_np import (
    F16_matmul_F16,
    F16_matmul_S8_v2,
    int32_dequant_to_fp16 as np_dequant_f16f16,
    int32_dequant_to_fp16_s8 as np_dequant_s8,
)
from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.fp16_fixed_point.matmul import (
    f16_matmul_f16,
    f16_matmul_s8,
    int32_dequant_to_fp16,
    int32_dequant_to_fp16_s8,
)

pytestmark = [pytest.mark.cpu, pytest.mark.slow]


def _randn_fp16(*shape, scale=3.0):
    return (torch.randn(*shape) * scale).to(torch.float16)


def _parity_f16f16(a, b, bias, sv):
    torch_out = f16_matmul_f16(a, b, bias, sv)
    np_out = F16_matmul_F16(
        a.numpy(),
        b.numpy(),
        bias.numpy() if bias is not None else None,
        sv,
    )
    return torch_out, torch.from_numpy(np_out.astype(np.int64))


def _parity_s8(a, b, bias, sv, ascale):
    torch_out = f16_matmul_s8(a, b, bias, sv, ascale)
    np_out = F16_matmul_S8_v2(
        a.numpy(),
        b.numpy(),
        bias.numpy() if bias is not None else None,
        sv,
        ascale.numpy() if ascale is not None else None,
    )
    return torch_out, torch.from_numpy(np_out.astype(np.int64))


class TestF16MatmulF16Parity(unittest.TestCase):
    """Bit-exact parity with the numpy F16_matmul_F16 reference."""

    def setUp(self):
        torch.manual_seed(2026)

    def _assert_parity(self, a, b, bias, sv):
        torch_out, np_out = _parity_f16f16(a, b, bias, sv)
        self.assertEqual(torch_out.dtype, torch.int32)
        self.assertTrue(
            torch.equal(torch_out.to(torch.int64), np_out),
            'max diff {}'.format(
                (torch_out.to(torch.int64) - np_out).abs().max().item()
            ),
        )

    def test_2d_single_tile(self):
        a = _randn_fp16(8, 16)
        b = _randn_fp16(4, 16)
        self._assert_parity(a, b, None, 42)
        bias = (torch.randn(4) * 2).to(torch.float16)
        self._assert_parity(a, b, bias, 42)

    def test_2d_multi_tile(self):
        a = _randn_fp16(8, 32)
        b = _randn_fp16(4, 32)
        self._assert_parity(a, b, None, 42)

    def test_k_padding_variants(self):
        for k in (1, 5, 17, 20, 33):
            a = _randn_fp16(5, k)
            b = _randn_fp16(3, k)
            self._assert_parity(a, b, None, 42)

    def test_m_n_edges(self):
        a = _randn_fp16(1, 16)
        b = _randn_fp16(1, 16)
        self._assert_parity(a, b, None, 42)
        a = _randn_fp16(7, 16)
        b = _randn_fp16(1, 16)
        self._assert_parity(a, b, None, 42)

    def test_3d_batch(self):
        a = _randn_fp16(2, 5, 32)
        b = _randn_fp16(2, 4, 32)
        self._assert_parity(a, b, None, 42)

    def test_batch_broadcast(self):
        a = _randn_fp16(2, 5, 16)
        b = _randn_fp16(4, 16)
        self._assert_parity(a, b, None, 42)

    def test_batch_bias(self):
        """Batch bias of shape (batch..., N) and its broadcast variants."""
        a = _randn_fp16(2, 5, 16)
        b = _randn_fp16(2, 4, 16)
        # per-batch bias (B, N)
        bias = (torch.randn(2, 4) * 2).to(torch.float16)
        self._assert_parity(a, b, bias, 42)
        # size-1 batch dim broadcasts to every batch
        bias_b1 = (torch.randn(1, 4) * 2).to(torch.float16)
        self._assert_parity(a, b, bias_b1, 42)
        # multi-dim batch
        a2 = _randn_fp16(2, 3, 5, 16)
        b2 = _randn_fp16(2, 3, 4, 16)
        bias2 = (torch.randn(2, 3, 4) * 2).to(torch.float16)
        self._assert_parity(a2, b2, bias2, 42)
        # batch bias together with a broadcast matmul batch (a 3D, b 2D)
        a3 = _randn_fp16(2, 5, 16)
        b3 = _randn_fp16(4, 16)
        bias3 = (torch.randn(2, 4) * 2).to(torch.float16)
        self._assert_parity(a3, b3, bias3, 42)

    def test_fixed_shift_value_sweep(self):
        a = _randn_fp16(5, 33)
        b = _randn_fp16(7, 33)
        for sv in (34, 37, 40, 42, 43):
            self._assert_parity(a, b, None, sv)

    def test_subnormal_inputs(self):
        a = (torch.randn(6, 16) * 1e-5).to(torch.float16)
        b = (torch.randn(3, 16) * 3).to(torch.float16)
        self._assert_parity(a, b, None, 42)
        # both subnormal
        b2 = (torch.randn(3, 16) * 1e-5).to(torch.float16)
        self._assert_parity(a, b2, None, 42)

    def test_mixed_signs(self):
        a = _randn_fp16(8, 16, scale=1.0)
        b = _randn_fp16(4, 16, scale=1.0)
        self._assert_parity(a, b, None, 42)

    def test_extreme_values_saturation(self):
        a = torch.full((4, 16), 65504.0, dtype=torch.float16)
        b = torch.full((3, 16), 65504.0, dtype=torch.float16)
        torch_out, np_out = _parity_f16f16(a, b, None, 42)
        self.assertTrue(torch.equal(torch_out.to(torch.int64), np_out))
        self.assertTrue(torch.all(torch_out.to(torch.int64) <= 2**31 - 1))
        self.assertTrue(torch.all(torch_out.to(torch.int64) >= -(2**31)))

    def test_zero_inputs(self):
        a = torch.zeros(4, 16, dtype=torch.float16)
        b = _randn_fp16(3, 16)
        torch_out, _ = _parity_f16f16(a, b, None, 42)
        self.assertTrue(torch.all(torch_out == 0))

    def test_bias_enters_accumulator_domain(self):
        a = _randn_fp16(8, 16)
        b = _randn_fp16(4, 16)
        sv = 42
        bias = torch.tensor([0.5, -0.5, 1.0, -1.0], dtype=torch.float16)
        out_bias = f16_matmul_f16(a, b, bias, sv)
        out_none = f16_matmul_f16(a, b, None, sv)
        # bias 0.5 * 2 ** (58 - 42) rounds to 32768
        expected_delta = torch.round(bias.to(torch.float32) * float(2 ** (58 - sv))).to(
            torch.int64
        )
        delta = out_bias.to(torch.int64) - out_none.to(torch.int64)
        self.assertTrue(
            torch.equal(delta, expected_delta.unsqueeze(0).expand_as(delta))
        )


class TestF16MatmulS8Parity(unittest.TestCase):
    """Bit-exact parity with the numpy F16_matmul_S8_v2 reference."""

    def setUp(self):
        torch.manual_seed(2026)

    def _randn_ascale(self, n):
        return (torch.randn(n) * 0.01).to(torch.float32)

    def _assert_parity(self, a, b, bias, sv, ascale):
        torch_out, np_out = _parity_s8(a, b, bias, sv, ascale)
        self.assertEqual(torch_out.dtype, torch.int32)
        self.assertTrue(
            torch.equal(torch_out.to(torch.int64), np_out),
            'max diff {}'.format(
                (torch_out.to(torch.int64) - np_out).abs().max().item()
            ),
        )

    def test_basic_with_antiquant_scale(self):
        a = _randn_fp16(8, 16)
        b = torch.randint(-128, 128, (4, 16), dtype=torch.int8)
        bias = (torch.randn(4) * 0.5).to(torch.float16)
        self._assert_parity(a, b, bias, 13, self._randn_ascale(4))

    def test_pre_scaled_bias(self):
        a = _randn_fp16(8, 16)
        b = torch.randint(-128, 128, (4, 16), dtype=torch.int8)
        ascale = self._randn_ascale(4)
        bias = torch.round(
            (torch.randn(4) * 0.5).to(torch.float16).float() / ascale
        ).to(torch.int64)
        self._assert_parity(a, b, bias, 13, None)

    def test_batch_and_k_padding(self):
        a = _randn_fp16(2, 3, 20)
        b = torch.randint(-128, 128, (2, 5, 20), dtype=torch.int8)
        bias = (torch.randn(5) * 0.3).to(torch.float16)
        self._assert_parity(a, b, bias, 9, self._randn_ascale(5))

    def test_fixed_shift_value_sweep(self):
        a = _randn_fp16(8, 16)
        b = torch.randint(-128, 128, (4, 16), dtype=torch.int8)
        bias = (torch.randn(4) * 0.5).to(torch.float16)
        for sv in (5, 9, 13, 14):
            self._assert_parity(a, b, bias, sv, self._randn_ascale(4))

    def test_subnormal_inputs(self):
        a = (torch.randn(6, 16) * 1e-5).to(torch.float16)
        b = torch.randint(-128, 128, (4, 16), dtype=torch.int8)
        bias = (torch.randn(4) * 0.5).to(torch.float16)
        self._assert_parity(a, b, bias, 13, self._randn_ascale(4))

    def test_batch_broadcast(self):
        a = _randn_fp16(2, 6, 16)
        b = torch.randint(-128, 128, (5, 16), dtype=torch.int8)
        bias = (torch.randn(5) * 0.5).to(torch.float16)
        self._assert_parity(a, b, bias, 13, self._randn_ascale(5))

    def test_batch_bias(self):
        """Batch bias of shape (batch..., N) on both bias paths."""
        a = _randn_fp16(2, 5, 16)
        b = torch.randint(-128, 128, (2, 4, 16), dtype=torch.int8)
        ascale = self._randn_ascale(4)
        # raw fp32 bias, scaled internally through antiquant_scale
        bias = (torch.randn(2, 4) * 0.5).to(torch.float16)
        self._assert_parity(a, b, bias, 13, ascale)
        # per-batch bias already scaled to the accumulator domain
        bias_int = torch.round(
            (torch.randn(2, 4) * 0.5).to(torch.float16).float() / ascale
        ).to(torch.int64)
        self._assert_parity(a, b, bias_int, 13, None)

    def test_int8_extremes(self):
        a = _randn_fp16(4, 16)
        b = torch.zeros(3, 16, dtype=torch.int8)
        b[0], b[1], b[2] = -128, 127, 64
        bias = (torch.randn(3) * 0.5).to(torch.float16)
        self._assert_parity(a, b, bias, 13, self._randn_ascale(3))

    def test_none_bias_equals_zero_bias(self):
        a = _randn_fp16(8, 16)
        b = torch.randint(-128, 128, (4, 16), dtype=torch.int8)
        ascale = self._randn_ascale(4)
        out_none = f16_matmul_s8(a, b, None, 13, ascale)
        out_zero = f16_matmul_s8(a, b, torch.zeros(4), 13, ascale)
        self.assertTrue(torch.equal(out_none, out_zero))


class TestDequantParity(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)

    def test_dequant_f16f16(self):
        acc = torch.randint(-(2**31), 2**31 - 1, (5, 8), dtype=torch.int64)
        for sv in (34, 42, 43):
            torch_out = int32_dequant_to_fp16(acc, sv)
            np_out = np_dequant_f16f16(acc.numpy(), sv)
            self.assertTrue(torch.equal(torch_out, torch.from_numpy(np_out)))

    def test_dequant_s8_per_channel(self):
        acc = torch.randint(-(2**31), 2**31 - 1, (5, 8), dtype=torch.int32)
        ascale = (torch.randn(8) * 0.01).to(torch.float32)
        torch_out = int32_dequant_to_fp16_s8(acc, ascale)
        np_out = np_dequant_s8(acc.numpy(), ascale.numpy())
        self.assertTrue(torch.equal(torch_out, torch.from_numpy(np_out)))

    def test_dequant_s8_scalar_scale(self):
        """A 0-dim antiquant scale broadcasts over the whole output."""
        acc = torch.randint(-(2**31), 2**31 - 1, (5, 8), dtype=torch.int32)
        ascale = torch.tensor(0.01, dtype=torch.float32)
        out = int32_dequant_to_fp16_s8(acc, ascale)
        expect = (acc.to(torch.float32) * 0.01).to(torch.float16)
        self.assertEqual(tuple(out.shape), (5, 8))
        self.assertTrue(torch.equal(out, expect))


class TestNumericalSanity(unittest.TestCase):
    """Independent fp32 reference checks (not relying on the numpy code)."""

    def setUp(self):
        torch.manual_seed(2026)

    def test_f16f16_close_to_fp32_reference(self):
        a = _randn_fp16(16, 48, scale=2.0)
        b = _randn_fp16(12, 48, scale=2.0)
        out = int32_dequant_to_fp16(f16_matmul_f16(a, b, None, 42), 42)
        ref = a.float() @ b.float().t()
        rel = (out.float() - ref).abs() / (ref.abs() + 1e-2)
        self.assertLess(rel.median().item(), 1e-2)
        self.assertLess(rel.max().item(), 5e-2)

    def test_s8_close_to_dequantized_reference(self):
        # the S8 accumulator lives on the 2 ** (29 - fsv) fixed point scale,
        # so the accumulator-LSB antiquant scale is the weight scale divided
        # by 2 ** (29 - fsv)
        fsv = 13
        a = _randn_fp16(16, 48, scale=0.5)
        b = torch.randint(-128, 128, (12, 48), dtype=torch.int8)
        w_scale = (torch.rand(12) * 2e-4 + 1e-5).to(torch.float32)
        ascale = w_scale / float(1 << (29 - fsv))
        out = int32_dequant_to_fp16_s8(f16_matmul_s8(a, b, None, fsv), ascale)
        ref = a.float() @ (b.float() * w_scale[:, None]).t()
        rel = (out.float() - ref).abs() / (ref.abs() + 1e-2)
        self.assertLess(rel.median().item(), 1e-2)
        self.assertLess(rel.max().item(), 5e-2)

    def test_bias_preserved_in_output(self):
        a = _randn_fp16(4, 16)
        b = _randn_fp16(3, 16)
        bias = torch.tensor([1.0, -2.0, 0.25], dtype=torch.float16)
        out = int32_dequant_to_fp16(f16_matmul_f16(a, b, bias, 42), 42)
        ref = a.float() @ b.float().t() + bias.float()
        self.assertTrue(torch.allclose(out.float(), ref, rtol=0.05, atol=0.05))


class TestContract(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)

    def test_k_mismatch_f16f16(self):
        a = _randn_fp16(8, 16)
        b = _randn_fp16(4, 32)
        with self.assertRaises(RuntimeError):
            f16_matmul_f16(a, b, None, 42)

    def test_k_mismatch_s8(self):
        a = _randn_fp16(8, 16)
        b = torch.randint(-128, 128, (4, 32), dtype=torch.int8)
        with self.assertRaises(RuntimeError):
            f16_matmul_s8(a, b, None, 13)

    def test_output_dtype_int32(self):
        a = _randn_fp16(4, 16)
        b = _randn_fp16(3, 16)
        self.assertEqual(f16_matmul_f16(a, b, None, 42).dtype, torch.int32)
        b8 = torch.randint(-128, 128, (3, 16), dtype=torch.int8)
        self.assertEqual(f16_matmul_s8(a, b8, None, 13).dtype, torch.int32)

    def test_output_shape(self):
        a = _randn_fp16(2, 5, 33)
        b = _randn_fp16(4, 33)
        out = f16_matmul_f16(a, b, None, 42)
        self.assertEqual(tuple(out.shape), (2, 5, 4))


if __name__ == '__main__':
    unittest.main()
