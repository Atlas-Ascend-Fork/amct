# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
# ----------------------------------------------------------------------------
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


class TestBaseQuantizeFakeQuant(unittest.TestCase):
    """_fake_quant_forward produces quant-dequant output, not raw float output."""

    def _make_weight_only_module(self):
        from amct_pytorch.quantize_op.minmax_module import MinMaxQuant

        linear = nn.Linear(8, 4, bias=True).to(torch.bfloat16)
        cfg = {
            "batch_num": 1,
            "weights_cfg": {
                "quant_type": "int8",
                "symmetric": True,
                "strategy": "channel",
                "group_size": None,
            },
            "inputs_cfg": {
                "quant_type": "int8",
                "symmetric": True,
                "strategy": "tensor",
                "enable_quant": False,
            },
        }
        return MinMaxQuant(linear, "test_layer", cfg), linear

    def test_fake_quant_forward_differs_from_fp(self):
        """_fake_quant_forward output must differ from raw float output."""
        mod, linear = self._make_weight_only_module()
        x = torch.randn(2, 8).to(torch.bfloat16)
        fp_out = linear(x)
        fake_out = mod._fake_quant_forward(x)
        self.assertFalse(
            torch.allclose(fp_out, fake_out, atol=0),
            "fake quant output must differ from float output",
        )

    def test_fake_quant_forward_matches_manual_quant_dequant(self):
        """_fake_quant_forward must equal manual quant_dequant(weight) @ x."""
        from amct_pytorch.utils.quant_util import quant_dequant_tensor

        mod, _ = self._make_weight_only_module()
        x = torch.randn(2, 8).to(torch.bfloat16)
        fake_out = mod._fake_quant_forward(x)
        dq_w = quant_dequant_tensor(
            mod.weight.data, "int8", mod.scale_w, mod.offset_w, mod.group_size
        )
        expected = F.linear(x, dq_w, mod.bias)
        self.assertTrue(
            torch.allclose(fake_out, expected),
            f"max diff: {(fake_out - expected).abs().max()}",
        )


class TestMinMaxFakeQuant(unittest.TestCase):
    """MinMaxQuant forward uses fake quant after calibration."""

    def _cfg(self, weight_only=True):
        return {
            "batch_num": 2,
            "weights_cfg": {
                "quant_type": "int8",
                "symmetric": True,
                "strategy": "channel",
                "group_size": None,
            },
            "inputs_cfg": {
                "quant_type": "int8",
                "symmetric": True,
                "strategy": "tensor",
                "enable_quant": not weight_only,
            },
        }

    def test_during_calibration_output_equals_fp(self):
        """weight+act path: output equals fp_out while cur_batch <= batch_num."""
        from amct_pytorch.quantize_op.minmax_module import MinMaxQuant

        linear = nn.Linear(8, 4).to(torch.bfloat16)
        mod = MinMaxQuant(linear, "l", self._cfg(weight_only=False))
        x = torch.randn(2, 8).to(torch.bfloat16)
        fp_ref = linear(x)
        out = mod(x)  # cur_batch=1, batch_num=2 → calibrating
        self.assertTrue(
            torch.allclose(fp_ref, out),
            "During calibration, output should equal float output",
        )

    def test_weight_compress_only_applies_fake_quant_immediately(self):
        """weight_compress_only=True: every forward applies fake quant."""
        from amct_pytorch.quantize_op.minmax_module import MinMaxQuant
        from amct_pytorch.utils.quant_util import quant_dequant_tensor

        linear = nn.Linear(8, 4).to(torch.bfloat16)
        mod = MinMaxQuant(linear, "l", self._cfg(weight_only=True))
        x = torch.randn(2, 8).to(torch.bfloat16)
        out = mod(x)
        dq_w = quant_dequant_tensor(mod.weight.data, "int8", mod.scale_w, mod.offset_w)
        expected = F.linear(x, dq_w, mod.bias)
        self.assertTrue(
            torch.allclose(out, expected), f"max diff: {(out - expected).abs().max()}"
        )

    def test_after_calibration_weight_act_fake_quant(self):
        """weight+act: after batch_num rounds, both are fake-quanted."""
        from amct_pytorch.quantize_op.minmax_module import MinMaxQuant
        from amct_pytorch.utils.quant_util import quant_dequant_tensor

        linear = nn.Linear(8, 4).to(torch.bfloat16)
        mod = MinMaxQuant(linear, "l", self._cfg(weight_only=False))
        x = torch.randn(2, 8).to(torch.bfloat16)
        mod(x)  # cur_batch=1 (calibrating)
        mod(x)  # cur_batch=2 == batch_num → scale_d computed, returns fp_out
        out = mod(x)  # cur_batch=3 > batch_num → fake quant
        dq_w = quant_dequant_tensor(mod.weight.data, "int8", mod.scale_w, mod.offset_w)
        dq_x = quant_dequant_tensor(x, "int8", mod.scale_d, mod.offset_d)
        expected = F.linear(dq_x, dq_w, mod.bias)
        self.assertTrue(
            torch.allclose(out, expected), f"max diff: {(out - expected).abs().max()}"
        )


class TestSmoothQuantFakeQuant(unittest.TestCase):
    """SmoothQuant forward applies smooth_factor-aware fake quant after calibration."""

    def _cfg(self):
        return {
            "batch_num": 1,
            "weights_cfg": {
                "quant_type": "int8",
                "symmetric": True,
                "strategy": "channel",
                "group_size": None,
            },
            "inputs_cfg": {
                "quant_type": "int8",
                "symmetric": True,
                "strategy": "tensor",
            },
            "algorithm": {"smoothquant": {"smooth_strength": 0.5}},
        }

    def test_after_calibration_uses_smooth_factor(self):
        """Fake quant uses weight * smooth_factor and input / smooth_factor."""
        from amct_pytorch.quantize_op.smooth_quant_module import SmoothQuant
        from amct_pytorch.utils.quant_util import quant_dequant_tensor

        linear = nn.Linear(8, 4).to(torch.bfloat16)
        mod = SmoothQuant(linear, "l", self._cfg())
        x = torch.randn(2, 8).to(torch.bfloat16)
        mod(x)  # cur_batch=1 == batch_num → params computed, returns raw output
        out = mod(x)  # cur_batch=2 > batch_num → fake quant

        smooth = mod.scale  # smooth_factor, shape [1, cin]
        dq_w = quant_dequant_tensor(
            mod.weight.data * smooth.to(dtype=mod.weight.dtype),
            "int8",
            mod.scale_w,
            mod.offset_w,
        )
        dq_x = quant_dequant_tensor(
            x / smooth.to(dtype=x.dtype), "int8", mod.scale_d, mod.offset_d
        )
        expected = F.linear(dq_x, dq_w, mod.bias)
        self.assertTrue(
            torch.allclose(out, expected), f"max diff: {(out - expected).abs().max()}"
        )

    def test_quantizable_weight_applies_smooth_factor(self):
        """_quantizable_weight returns weight * smooth_factor (not raw weight)."""
        from amct_pytorch.quantize_op.smooth_quant_module import SmoothQuant

        linear = nn.Linear(8, 4).to(torch.bfloat16)
        mod = SmoothQuant(linear, "l", self._cfg())
        x = torch.randn(2, 8).to(torch.bfloat16)
        mod(x)  # calibration completes, mod.scale = smooth_factor
        smooth = mod.scale
        expected_qw = mod.weight.data * smooth.to(dtype=mod.weight.dtype)
        self.assertTrue(
            torch.allclose(mod._quantizable_weight(), expected_qw),
            "_quantizable_weight must return weight * smooth_factor",
        )


class TestOfmrFakeQuant(unittest.TestCase):
    """OfmrQuant forward applies fake quant after batch_num calibration rounds."""

    def _cfg(self):
        return {
            "batch_num": 1,
            "weights_cfg": {
                "quant_type": "int8",
                "symmetric": True,
                "strategy": "channel",
                "group_size": None,
            },
            "inputs_cfg": {
                "quant_type": "int8",
                "symmetric": True,
                "strategy": "tensor",
                "enable_quant": False,
            },
        }

    def test_during_calibration_output_equals_fp(self):
        """During calibration (cur_batch == batch_num), output equals fp_out."""
        from amct_pytorch.quantize_op.ofmr_quant_module import OfmrQuant

        linear = nn.Linear(8, 4).to(torch.bfloat16)
        mod = OfmrQuant(linear, "l", self._cfg())
        x = torch.randn(2, 8).to(torch.bfloat16)
        fp_ref = linear(x)
        out = mod(x)  # cur_batch=1 == batch_num → calibrates, returns fp_out
        self.assertTrue(
            torch.allclose(fp_ref, out),
            "During calibration, output should equal float output",
        )

    def test_after_calibration_applies_fake_quant(self):
        """After calibration, forward applies fake quant to weight."""
        from amct_pytorch.quantize_op.ofmr_quant_module import OfmrQuant
        from amct_pytorch.utils.quant_util import quant_dequant_tensor

        linear = nn.Linear(8, 4).to(torch.bfloat16)
        mod = OfmrQuant(linear, "l", self._cfg())
        x = torch.randn(2, 8).to(torch.bfloat16)
        mod(x)  # calibration: cur_batch=1 == batch_num → scale_w computed
        out = mod(x)  # fake quant: cur_batch=2 > batch_num

        dq_w = quant_dequant_tensor(mod.weight.data, "int8", mod.scale_w, mod.offset_w)
        expected = F.linear(x, dq_w, mod.bias)
        self.assertTrue(
            torch.allclose(out, expected), f"max diff: {(out - expected).abs().max()}"
        )


class TestAWQFakeQuant(unittest.TestCase):
    """LinearAWQuant fake quant uses ori_module.weight (AWQ-scaled), not self.weight."""

    def _make_cfg(self):
        return {
            "batch_num": 1,
            "weights_cfg": {
                "quant_type": "int8",
                "symmetric": True,
                "strategy": "channel",
                "group_size": None,
            },
            "inputs_cfg": {
                "quant_type": None,
                "symmetric": True,
                "strategy": "tensor",
                "enable_quant": False,
            },
            "algorithm": {"awq": {"grids_num": 20, "duo_scaling": False}},
        }

    def test_quantizable_weight_override_returns_ori_module_weight(self):
        """_quantizable_weight must return ori_module.weight.data (post-apply_scale)."""
        from amct_pytorch.quantize_op.linear_awq_module import LinearAWQuant

        linear = nn.Linear(8, 4).to(torch.bfloat16)
        mod = LinearAWQuant(linear, "l", self._make_cfg())
        x = torch.randn(2, 8).to(torch.bfloat16)
        mod(x)  # calibration: apply_scale modifies ori_module.weight, calc_done=True
        self.assertTrue(
            torch.allclose(mod._quantizable_weight(), mod.ori_module.weight.data),
            "_quantizable_weight must return ori_module.weight.data",
        )

    def test_after_calibration_output_matches_manual(self):
        """After calc_done, forward equals quant_dequant(ori_module.weight) @ x."""
        from amct_pytorch.quantize_op.linear_awq_module import LinearAWQuant
        from amct_pytorch.utils.quant_util import quant_dequant_tensor

        linear = nn.Linear(8, 4).to(torch.bfloat16)
        mod = LinearAWQuant(linear, "l", self._make_cfg())
        x = torch.randn(2, 8).to(torch.bfloat16)
        mod(x)  # calibration pass
        out = mod(x)  # fake quant pass

        dq_w = quant_dequant_tensor(
            mod.ori_module.weight.data, "int8", mod.scale_w, mod.offset_w
        )
        expected = F.linear(x, dq_w, mod.bias)
        self.assertTrue(
            torch.allclose(out, expected), f"max diff: {(out - expected).abs().max()}"
        )


class TestFakeQuantForwardConv2dDispatch(unittest.TestCase):
    """_fake_quant_forward must use F.conv2d (with stride/padding/etc.) for Conv2d modules.

    OfmrQuant registers for both Linear and Conv2d, but the cann/master
    quant_dequant_tensor has a 4-D reshape bug that breaks Conv2d calibration
    end-to-end (independent of this feature). Use mocks to verify the op
    dispatch layout without going through the calibration path.
    """

    def _make_conv2d_module_with_scale(self):
        from amct_pytorch.quantize_op.ofmr_quant_module import OfmrQuant

        conv = nn.Conv2d(4, 8, kernel_size=3, padding=1, stride=1, dilation=1, groups=1)
        conv = conv.to(torch.bfloat16)
        # Bypass calibration to dodge the unrelated 4-D reshape bug in
        # quant_dequant_tensor; populate just the fields _fake_quant_forward needs.
        mod = OfmrQuant.__new__(OfmrQuant)
        nn.Module.__init__(mod)
        mod.ori_module = conv
        mod.weight = conv.weight
        mod.bias = conv.bias
        mod.scale = None
        mod.scale_w = torch.ones(8).to(torch.bfloat16)
        mod.scale_d = None
        mod.offset_w = None
        mod.offset_d = None
        mod.group_size = None
        mod.wts_type = "int8"
        mod.act_type = "int8"
        return mod, conv

    def test_conv2d_dispatch_uses_F_conv2d_with_conv_attrs(self):
        mod, conv = self._make_conv2d_module_with_scale()
        x = torch.randn(2, 4, 8, 8).to(torch.bfloat16)
        sentinel_dq_w = torch.randn(8, 4, 3, 3).to(torch.bfloat16)

        with patch(
            "amct_pytorch.utils.quant_util.quant_dequant_tensor",
            return_value=sentinel_dq_w,
        ):
            out = mod._fake_quant_forward(x)

        expected = F.conv2d(
            x,
            sentinel_dq_w,
            conv.bias,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
        )
        self.assertEqual(out.shape, expected.shape)
        self.assertTrue(
            torch.allclose(out, expected), f"max diff: {(out - expected).abs().max()}"
        )


class TestApplyProgressiveQuantDequant(unittest.TestCase):
    """apply_progressive_quant_dequant does two-level quant+dequant with correct layout."""

    def test_rejects_wrong_scale_w1_shape(self):
        from amct_pytorch.quantize_op.utils import apply_progressive_quant_dequant

        weight = torch.randn(8, 64).to(torch.bfloat16)
        bad_scale_w1 = torch.ones(4).to(torch.bfloat16)  # should be [8]
        scale_w2 = torch.ones(int(8 * 64 / 32), 1).to(torch.bfloat16)
        with self.assertRaises(RuntimeError):
            apply_progressive_quant_dequant(weight, bad_scale_w1, scale_w2)

    def test_rejects_wrong_scale_w2_shape(self):
        from amct_pytorch.quantize_op.utils import apply_progressive_quant_dequant

        weight = torch.randn(8, 64).to(torch.bfloat16)
        scale_w1 = torch.ones(8).to(torch.bfloat16)
        bad_scale_w2 = torch.ones(4, 1).to(
            torch.bfloat16
        )  # should be [8*64/32, 1] = [16,1]
        with self.assertRaises(RuntimeError):
            apply_progressive_quant_dequant(weight, scale_w1, bad_scale_w2)

    def test_dispatches_to_quant_dequant_tensor_twice(self):
        """Two-level fake quant: FP8 on weight^T then FP4 on the result."""
        from amct_pytorch.quantize_op import utils as quant_utils
        from amct_pytorch.utils.vars import FLOAT8_E4M3FN, FLOAT4_E2M1

        weight = torch.randn(8, 64).to(torch.bfloat16)
        scale_w1 = torch.ones(8).to(torch.bfloat16)
        scale_w2 = torch.ones(int(8 * 64 / 32), 1).to(torch.bfloat16)

        # Side effect returns a tensor of the same shape as input — preserves
        # downstream reshapes/transposes so we can assert on call args.
        def fake_qd(tensor, dst_dtype, scale=None, offset=None, group_size=None):
            return tensor.clone()

        with patch.object(
            quant_utils, "quant_dequant_tensor", side_effect=fake_qd
        ) as m:
            result = quant_utils.apply_progressive_quant_dequant(
                weight, scale_w1, scale_w2
            )

        self.assertEqual(m.call_count, 2)
        first_call, second_call = m.call_args_list
        # First call: per-channel FP8 on weight^T
        self.assertEqual(first_call.args[0].shape, weight.transpose(-1, -2).shape)
        self.assertEqual(first_call.args[1], FLOAT8_E4M3FN)
        self.assertIs(first_call.args[2], scale_w1)
        # Second call: per-group FP4 on reshaped FP8-dequant
        self.assertEqual(second_call.args[0].shape, torch.Size([8 * 64 // 32, 32]))
        self.assertEqual(second_call.args[1], FLOAT4_E2M1)
        self.assertIs(second_call.args[2], scale_w2)
        # Output shape matches the input weight
        self.assertEqual(result.shape, weight.shape)
        self.assertEqual(result.dtype, weight.dtype)


class TestFakeQuantForwardProgressiveDispatch(unittest.TestCase):
    """_fake_quant_forward routes to apply_progressive_quant_dequant when scale_w1/scale_w2 are set."""

    def _make_module_with_progressive_scales(self):
        from amct_pytorch.quantize_op.minmax_module import MinMaxQuant

        linear = nn.Linear(8, 4, bias=True).to(torch.bfloat16)
        cfg = {
            "batch_num": 1,
            "weights_cfg": {
                "quant_type": "int8",  # placeholder; we override scale fields below
                "symmetric": True,
                "strategy": "channel",
                "group_size": None,
            },
            "inputs_cfg": {
                "quant_type": "int8",
                "symmetric": True,
                "strategy": "tensor",
                "enable_quant": False,
            },
        }
        mod = MinMaxQuant(linear, "l", cfg)
        # Force the progressive branch: clear scale_w and inject scale_w1/scale_w2.
        mod.scale_w = None
        mod.scale_w1 = torch.ones(4).to(torch.bfloat16)
        mod.scale_w2 = torch.ones(1, 1).to(torch.bfloat16)
        return mod

    def test_progressive_branch_calls_progressive_helper(self):
        """When scale_w is None and scale_w1/scale_w2 are set, dispatch to progressive helper."""
        from amct_pytorch.quantize_op import utils as quant_utils

        mod = self._make_module_with_progressive_scales()
        x = torch.randn(2, 8).to(torch.bfloat16)
        sentinel_dq_w = torch.randn(4, 8).to(torch.bfloat16)

        with patch.object(
            quant_utils, "apply_progressive_quant_dequant", return_value=sentinel_dq_w
        ) as m:
            out = mod._fake_quant_forward(x)

        self.assertEqual(m.call_count, 1)
        self.assertIs(m.call_args.args[1], mod.scale_w1)
        self.assertIs(m.call_args.args[2], mod.scale_w2)
        # Output should be F.linear(x, sentinel_dq_w, bias)
        expected = F.linear(x, sentinel_dq_w, mod.bias)
        self.assertTrue(torch.allclose(out, expected))
