#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for OfmrQuant and LinearAWQuant modules (CPU, no NPU required)."""
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from amct_pytorch.classic.quantize_op.ofmr_quant_module import OfmrQuant
from amct_pytorch.classic.quantize_op.linear_awq_module import LinearAWQuant
from amct_pytorch.common.utils.vars import INT8


def _make_ofmr_config(batch_num=2, strategy="tensor", weight_compress_only=False):
    return {
        "batch_num": batch_num,
        "weights_cfg": {"quant_type": INT8, "strategy": strategy},
        "inputs_cfg": {
            "quant_type": INT8,
            "enable_quant": not weight_compress_only,
        },
    }


def _make_awq_config(group_size=None):
    cfg = {
        "weights_cfg": {
            "quant_type": INT8,
            "symmetric": True,
            "group_size": group_size,
        },
        "inputs_cfg": {"enable_quant": True, "strategy": "tensor"},
    }
    return cfg


def _make_linear(in_f=8, out_f=4, bias=True):
    return nn.Linear(in_f, out_f, bias=bias)


class TestOfmrQuantInit(unittest.TestCase):
    """Test OfmrQuant __init__ paths."""

    def test_init_per_tensor_weight(self):
        mod = _make_linear()
        cfg = _make_ofmr_config(strategy="tensor")
        q = OfmrQuant(mod, "fc", cfg)
        self.assertEqual(q.cout, 1)
        self.assertEqual(q.batch_num, 2)

    def test_init_per_channel_weight(self):
        mod = _make_linear(8, 4)
        cfg = _make_ofmr_config(strategy="channel")
        q = OfmrQuant(mod, "fc", cfg)
        self.assertEqual(q.cout, mod.weight.shape[0])

    def test_init_weight_compress_only(self):
        mod = _make_linear()
        cfg = _make_ofmr_config(weight_compress_only=True)
        q = OfmrQuant(mod, "fc", cfg)
        self.assertTrue(q.weight_compress_only)


class TestOfmrQuantForward(unittest.TestCase):
    """Test OfmrQuant forward and calibration paths."""

    def setUp(self):
        self.mod = nn.Linear(8, 4, bias=False)
        self.cfg = _make_ofmr_config(batch_num=2, strategy="tensor")
        self.q = OfmrQuant(self.mod, "fc", self.cfg)

    def test_forward_calibration_batches(self):
        x = torch.randn(2, 8)
        out1 = self.q(x)
        self.assertEqual(out1.shape, (2, 4))
        out2 = self.q(x)
        self.assertEqual(out2.shape, (2, 4))

    def test_forward_after_calibration_uses_fake_quant(self):
        x = torch.randn(2, 8)
        self.q(x)
        self.q(x)
        # after batch_num batches, scale_w is set; next call uses fake_quant_forward
        out = self.q(x)
        self.assertEqual(out.shape, (2, 4))

    def test_compute_mse_no_reduce(self):
        a = torch.ones(4)
        b = torch.zeros(4)
        loss = OfmrQuant.compute_mse(a, b)
        self.assertAlmostEqual(loss.item(), 1.0, places=5)

    def test_compute_mse_with_reduce_axis(self):
        a = torch.ones(2, 3, 4)
        b = torch.zeros(2, 3, 4)
        loss = OfmrQuant.compute_mse(a, b, reduce_axis=(0, 2))
        self.assertEqual(loss.shape, (3,))

    def test_raises_on_invalid_linear_input_dim(self):
        x = torch.randn(2)  # 1D input — invalid for Linear
        with self.assertRaises(RuntimeError):
            self.q(x)

    def test_weight_compress_only_skips_act_loss(self):
        cfg = _make_ofmr_config(batch_num=1, weight_compress_only=True)
        q = OfmrQuant(self.mod, "fc", cfg)
        x = torch.randn(2, 8)
        out = q(x)
        self.assertEqual(out.shape, (2, 4))
        # scale_d should remain None (no act quant)
        self.assertIsNone(q.scale_d)


class TestOfmrQuantFakeQuantForward(unittest.TestCase):
    """Test fake_quant_forward caching."""

    def test_linear_fake_quant_cached(self):
        mod = nn.Linear(4, 2, bias=True)
        cfg = _make_ofmr_config(batch_num=1)
        q = OfmrQuant(mod, "fc", cfg)
        x = torch.randn(2, 4)
        q(x)  # calibration batch
        # directly exercise fake_quant_forward
        q.scale_w = torch.tensor([1.0])
        q.offset_w = None
        q.scale_d = None
        out = q.fake_quant_forward(x)
        self.assertEqual(out.shape, (2, 2))
        # second call hits cache
        out2 = q.fake_quant_forward(x)
        self.assertEqual(out2.shape, (2, 2))

    def test_conv2d_fake_quant_path(self):
        """Exercise the Conv2d branch in fake_quant_forward (line 137)."""
        mod = nn.Conv2d(2, 4, kernel_size=3, padding=1, bias=True)
        cfg = _make_ofmr_config(batch_num=1, strategy="tensor")
        q = OfmrQuant(mod, "conv", cfg)
        x = torch.randn(1, 2, 8, 8)
        q(x)  # calibration
        q.scale_w = torch.ones(1)
        q.offset_w = None
        q.scale_d = None
        out = q.fake_quant_forward(x)
        self.assertEqual(out.shape, (1, 4, 8, 8))

    def test_conv2d_calibration_with_act_quant(self):
        """Exercise Conv2d branches in _calc_weight/act_quant_loss (lines 202,211,214,240)."""
        mod = nn.Conv2d(2, 4, kernel_size=3, padding=1, bias=False)
        cfg = _make_ofmr_config(batch_num=2, strategy="tensor", weight_compress_only=False)
        q = OfmrQuant(mod, "conv", cfg)
        x = torch.randn(1, 2, 8, 8)
        q(x)
        q(x)  # both batches done → scale_w and scale_d computed
        self.assertIsNotNone(q.scale_w)
        self.assertIsNotNone(q.scale_d)

    def test_conv2d_per_channel_calibration_cout_axis(self):
        """Per-channel Conv2d hits the cout>1 axis-reduction path (lines 214-218)."""
        mod = nn.Conv2d(2, 4, kernel_size=3, padding=1, bias=False)
        cfg = _make_ofmr_config(batch_num=1, strategy="channel", weight_compress_only=True)
        q = OfmrQuant(mod, "conv", cfg)
        self.assertEqual(q.cout, 4)
        x = torch.randn(1, 2, 8, 8)
        q(x)  # single batch → _calc_weight_quant_loss with cout>1 → scale_w computed
        self.assertIsNotNone(q.scale_w)


class TestLinearAWQuantInit(unittest.TestCase):
    """Test LinearAWQuant __init__ paths."""

    def test_init_with_group_size(self):
        mod = _make_linear(bias=False)
        cfg = _make_awq_config(group_size=4)
        q = LinearAWQuant(mod, "fc", cfg)
        self.assertEqual(q.group_size, 4)

    def test_init_without_group_size(self):
        mod = _make_linear(bias=False)
        cfg = _make_awq_config(group_size=None)
        q = LinearAWQuant(mod, "fc", cfg)
        self.assertFalse(hasattr(q, "group_size") and q.group_size is not None)

    def test_init_act_quant_disabled(self):
        mod = _make_linear(bias=False)
        cfg = {
            "weights_cfg": {"quant_type": INT8, "symmetric": True, "group_size": None},
            "inputs_cfg": {"enable_quant": False, "strategy": "tensor"},
        }
        q = LinearAWQuant(mod, "fc", cfg)
        self.assertFalse(hasattr(q, "act_granularity"))


class TestLinearAWQuantForward(unittest.TestCase):
    """Test LinearAWQuant forward and fake_quant_forward."""

    def setUp(self):
        self.mod = nn.Linear(8, 4, bias=False)
        self.cfg = _make_awq_config()

    @patch("amct_pytorch.classic.quantize_op.linear_awq_module.search_scale")
    @patch("amct_pytorch.classic.quantize_op.linear_awq_module.apply_scale")
    @patch(
        "amct_pytorch.classic.quantize_op.linear_awq_module.calculate_scale_offset_by_granularity"
    )
    def test_forward_first_pass(self, mock_calc, mock_apply, mock_search):
        mock_search.return_value = torch.ones(8)
        mock_calc.return_value = (torch.ones(4, 1), torch.zeros(4, 1))
        q = LinearAWQuant(self.mod, "fc", self.cfg)
        x = torch.randn(2, 8)
        out = q(x)
        self.assertEqual(out.shape, (2, 4))
        self.assertTrue(q.calc_done)

    @patch("amct_pytorch.classic.quantize_op.linear_awq_module.search_scale")
    @patch("amct_pytorch.classic.quantize_op.linear_awq_module.apply_scale")
    @patch(
        "amct_pytorch.classic.quantize_op.linear_awq_module.calculate_scale_offset_by_granularity"
    )
    def test_forward_second_pass_uses_fake_quant(self, mock_calc, mock_apply, mock_search):
        mock_search.return_value = torch.ones(8)
        mock_calc.return_value = (torch.ones(4, 1), torch.zeros(4, 1))
        q = LinearAWQuant(self.mod, "fc", self.cfg)
        x = torch.randn(2, 8)
        q(x)  # first pass sets calc_done=True
        out = q(x)  # second pass uses fake_quant_forward
        self.assertEqual(out.shape, (2, 4))

    @patch("amct_pytorch.classic.quantize_op.linear_awq_module.search_scale")
    @patch("amct_pytorch.classic.quantize_op.linear_awq_module.apply_scale")
    @patch(
        "amct_pytorch.classic.quantize_op.linear_awq_module.calculate_scale_offset_by_granularity"
    )
    def test_fake_quant_forward_cached(self, mock_calc, mock_apply, mock_search):
        mock_search.return_value = torch.ones(8)
        mock_calc.return_value = (torch.ones(4, 1), torch.zeros(4, 1))
        q = LinearAWQuant(self.mod, "fc", self.cfg)
        x = torch.randn(2, 8)
        q(x)
        # call fake_quant_forward directly twice to test cache path
        out1 = q.fake_quant_forward(x)
        out2 = q.fake_quant_forward(x)  # hits cache
        self.assertEqual(out1.shape, out2.shape)


if __name__ == "__main__":
    unittest.main()
