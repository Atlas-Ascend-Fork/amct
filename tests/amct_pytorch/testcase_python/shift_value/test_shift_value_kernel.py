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
"""Unit tests of the fp16fixed shift value kernel plug-in point."""

import unittest

import pytest

import torch

from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.quantization import (
    shift_value_kernel,
)

pytestmark = pytest.mark.cpu


class TestShiftValueKernel(unittest.TestCase):
    def test_get_candidates(self):
        self.assertEqual(
            shift_value_kernel.get_candidates('fp16_fp16'), list(range(34, 44))
        )
        self.assertEqual(
            shift_value_kernel.get_candidates('fp16_int8'), list(range(5, 15))
        )

    def test_get_default_value(self):
        self.assertEqual(shift_value_kernel.get_default_value('fp16_fp16'), 42)
        self.assertEqual(shift_value_kernel.get_default_value('fp16_int8'), 13)

    def test_compute_shift_value_sse_unsupported_module(self):
        """A module type without a fp16fixed implementation raises."""
        module = torch.nn.BatchNorm1d(4)
        with self.assertRaises(NotImplementedError):
            shift_value_kernel.compute_shift_value_sse(
                torch.randn(2, 4), torch.randn(4, 4), 'fp16_fp16', module
            )

    def test_compute_shift_value_sse_int8_missing_quant_info(self):
        """The fp16_int8 regime without weight quant factors raises."""
        module = torch.nn.Linear(4, 2)
        with self.assertRaises(NotImplementedError):
            shift_value_kernel.compute_shift_value_sse(
                torch.randn(3, 4), module.weight, 'fp16_int8', module
            )

    @staticmethod
    def _fake_quant_weight(module, axis):
        """Fake-quantize the module weight per output channel (ARQ style).

        Mirrors what ``WeightsCalibrationPass`` leaves behind: the module
        weight is replaced by the on-grid fake-quant weight and the
        quant factors are what the record file would carry.
        """
        weight = module.weight.detach()
        reduce_dims = tuple(d for d in range(weight.dim()) if d != axis)
        scale = (weight.abs().amax(dim=reduce_dims) / 127.0).clamp_min(2.0**-30)
        scale_shape = [1] * weight.dim()
        scale_shape[axis] = -1
        int8 = (
            torch.round(weight / scale.reshape(scale_shape))
            .clamp(-127, 127)
            .to(torch.int8)
        )
        module.weight.data = int8.to(torch.float32) * scale.reshape(scale_shape)
        quant_info = {
            'scale_w': scale.tolist(),
            'offset_w': [0] * scale.numel(),
            'wts_type': 'INT8',
        }
        return int8, quant_info

    def test_int8_rebuild_recovers_deployed_weight(self):
        """The rebuilt int8 weight is bit-exact with the deployed one.

        The module weight carries the on-grid fake-quant weight, so the
        round trip through weight_quant_np recovers the original int8
        values -- the premise of the fp16_int8 search.
        """
        modules = [
            (torch.nn.Linear(64, 32), 0),
            (torch.nn.Conv2d(8, 16, 3, padding=1), 0),
            (torch.nn.ConvTranspose2d(8, 16, 3), 1),
        ]
        for module, axis in modules:
            int8_weight, quant_info = self._fake_quant_weight(module, axis)
            rebuilt, weight_scale = shift_value_kernel._rebuild_int8_weight(
                module, quant_info
            )
            self.assertEqual(rebuilt.dtype, torch.int8)
            self.assertTrue(
                torch.equal(rebuilt, int8_weight),
                'rebuild mismatch for {}'.format(type(module).__name__),
            )
            self.assertTrue(
                torch.allclose(
                    weight_scale, torch.tensor(quant_info['scale_w']), atol=0
                )
            )

    def test_compute_shift_value_sse_int8_dispatch(self):
        """The fp16_int8 regime dispatches to the S8 operator classes."""
        cases = [
            (torch.nn.Linear(64, 32), (5, 64), 0),
            (torch.nn.Conv2d(8, 16, 3, padding=1), (2, 8, 8, 8), 0),
            (torch.nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1), (2, 8, 8, 8), 1),
        ]
        for module, shape, axis in cases:
            _, quant_info = self._fake_quant_weight(module, axis)
            inputs = (torch.randn(*shape) * 0.5).to(torch.float16)
            sse = shift_value_kernel.compute_shift_value_sse(
                inputs, module.weight, 'fp16_int8', module, quant_info
            )
            self.assertEqual(tuple(sse.shape), (10,))
            self.assertEqual(sse.dtype, torch.float64)

    def test_compute_shift_value_sse_int8_matches_operator_class(self):
        """The dispatched SSE equals the S8 operator class computation."""
        module = torch.nn.Linear(64, 32)
        int8_weight, quant_info = self._fake_quant_weight(module, 0)
        rebuilt, weight_scale = shift_value_kernel._rebuild_int8_weight(
            module, quant_info
        )
        inputs = (torch.randn(5, 64) * 0.5).to(torch.float16)
        sse = shift_value_kernel.compute_shift_value_sse(
            inputs, module.weight, 'fp16_int8', module, quant_info
        )
        from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.fp16_fixed_point import (  # noqa: E501
            Fp16FixedLinearS8,
        )

        expect = Fp16FixedLinearS8.from_module(module, rebuilt, weight_scale)
        self.assertTrue(torch.equal(sse, expect.compute_candidate_sse(inputs)))

    def test_int8_asymmetric_offset_fallback(self):
        """Asymmetric weight quantization (offset_w != 0) falls back."""
        module = torch.nn.Linear(64, 32)
        _, quant_info = self._fake_quant_weight(module, 0)
        quant_info['offset_w'] = [1] * len(quant_info['offset_w'])
        with self.assertRaises(NotImplementedError):
            shift_value_kernel.compute_shift_value_sse(
                torch.randn(5, 64), module.weight, 'fp16_int8', module, quant_info
            )

    def test_int8_wrong_wts_type_fallback(self):
        """A non INT8 weight (e.g. INT4 weight-only) falls back."""
        module = torch.nn.Linear(64, 32)
        _, quant_info = self._fake_quant_weight(module, 0)
        quant_info['wts_type'] = 'INT4'
        with self.assertRaises(NotImplementedError):
            shift_value_kernel.compute_shift_value_sse(
                torch.randn(5, 64), module.weight, 'fp16_int8', module, quant_info
            )

    def test_int8_unsupported_module(self):
        module = torch.nn.BatchNorm1d(4)
        quant_info = {'scale_w': [0.1], 'offset_w': [0], 'wts_type': 'INT8'}
        with self.assertRaises(NotImplementedError):
            shift_value_kernel.compute_shift_value_sse(
                torch.randn(2, 4), torch.randn(4, 4), 'fp16_int8', module, quant_info
            )

    def test_compute_shift_value_sse_dispatch(self):
        """The fp16_fp16 regime dispatches to the fp16fixed operator classes."""
        cases = [
            (torch.nn.Linear(64, 32), (5, 64)),
            (torch.nn.Conv2d(8, 16, 3, padding=1), (2, 8, 8, 8)),
            (torch.nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1), (2, 8, 8, 8)),
        ]
        for module, shape in cases:
            module.weight.data = (
                (torch.randn_like(module.weight) * 0.1)
                .to(torch.float16)
                .to(torch.float32)
            )
            inputs = (torch.randn(*shape) * 0.5).to(torch.float16)
            sse = shift_value_kernel.compute_shift_value_sse(
                inputs, module.weight, 'fp16_fp16', module
            )
            self.assertEqual(tuple(sse.shape), (10,))
            self.assertEqual(sse.dtype, torch.float64)

    def test_compute_shift_value_sse_matches_operator_class(self):
        """The dispatched SSE equals the operator class computation."""
        module = torch.nn.Linear(64, 32)
        module.weight.data = (
            (torch.randn_like(module.weight) * 0.1).to(torch.float16).to(torch.float32)
        )
        inputs = (torch.randn(5, 64) * 0.5).to(torch.float16)
        sse = shift_value_kernel.compute_shift_value_sse(
            inputs, module.weight, 'fp16_fp16', module
        )
        from amct_pytorch.classic.graph_based.amct_pytorch.nn.module.fp16_fixed_point import (  # noqa: E501
            Fp16FixedLinear,
        )

        expect = Fp16FixedLinear.from_module(module).compute_candidate_sse(inputs)
        self.assertTrue(torch.equal(sse, expect))
