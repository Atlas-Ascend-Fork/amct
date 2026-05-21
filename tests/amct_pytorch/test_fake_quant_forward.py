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
            'batch_num': 1,
            'weights_cfg': {'quant_type': 'int8', 'symmetric': True,
                            'strategy': 'channel', 'group_size': None},
            'inputs_cfg': {'quant_type': 'int8', 'symmetric': True,
                           'strategy': 'tensor', 'enable_quant': False},
        }
        return MinMaxQuant(linear, 'test_layer', cfg), linear

    def test_fake_quant_forward_differs_from_fp(self):
        mod, linear = self._make_weight_only_module()
        x = torch.randn(2, 8).to(torch.bfloat16)
        fp_out = linear(x)
        fake_out = mod._fake_quant_forward(x)
        self.assertFalse(torch.allclose(fp_out, fake_out, atol=0),
                         'fake quant output must differ from float output')

    def test_fake_quant_forward_matches_manual_quant_dequant(self):
        from amct_pytorch.utils.quant_util import quant_dequant_tensor
        mod, _ = self._make_weight_only_module()
        x = torch.randn(2, 8).to(torch.bfloat16)
        fake_out = mod._fake_quant_forward(x)
        dq_w = quant_dequant_tensor(mod.weight.data, 'int8', mod.scale_w, mod.offset_w, mod.group_size)
        expected = F.linear(x, dq_w, mod.bias)
        self.assertTrue(torch.allclose(fake_out, expected),
                        f'max diff: {(fake_out - expected).abs().max()}')


class TestMinMaxFakeQuant(unittest.TestCase):
    """MinMaxQuant forward uses fake quant after calibration."""

    def _cfg(self, weight_only=True):
        return {
            'batch_num': 2,
            'weights_cfg': {'quant_type': 'int8', 'symmetric': True,
                            'strategy': 'channel', 'group_size': None},
            'inputs_cfg': {'quant_type': 'int8', 'symmetric': True,
                           'strategy': 'tensor', 'enable_quant': not weight_only},
        }

    def test_during_calibration_output_equals_fp(self):
        from amct_pytorch.quantize_op.minmax_module import MinMaxQuant
        linear = nn.Linear(8, 4).to(torch.bfloat16)
        mod = MinMaxQuant(linear, 'l', self._cfg(weight_only=False))
        x = torch.randn(2, 8).to(torch.bfloat16)
        fp_ref = linear(x)
        out = mod(x)
        self.assertTrue(torch.allclose(fp_ref, out),
                        'During calibration, output should equal float output')

    def test_weight_compress_only_applies_fake_quant_immediately(self):
        from amct_pytorch.quantize_op.minmax_module import MinMaxQuant
        from amct_pytorch.utils.quant_util import quant_dequant_tensor
        linear = nn.Linear(8, 4).to(torch.bfloat16)
        mod = MinMaxQuant(linear, 'l', self._cfg(weight_only=True))
        x = torch.randn(2, 8).to(torch.bfloat16)
        out = mod(x)
        dq_w = quant_dequant_tensor(mod.weight.data, 'int8', mod.scale_w, mod.offset_w)
        expected = F.linear(x, dq_w, mod.bias)
        self.assertTrue(torch.allclose(out, expected),
                        f'max diff: {(out - expected).abs().max()}')

    def test_after_calibration_weight_act_fake_quant(self):
        from amct_pytorch.quantize_op.minmax_module import MinMaxQuant
        from amct_pytorch.utils.quant_util import quant_dequant_tensor
        linear = nn.Linear(8, 4).to(torch.bfloat16)
        mod = MinMaxQuant(linear, 'l', self._cfg(weight_only=False))
        x = torch.randn(2, 8).to(torch.bfloat16)
        mod(x)
        mod(x)
        out = mod(x)
        dq_w = quant_dequant_tensor(mod.weight.data, 'int8', mod.scale_w, mod.offset_w)
        dq_x = quant_dequant_tensor(x, 'int8', mod.scale_d, mod.offset_d)
        expected = F.linear(dq_x, dq_w, mod.bias)
        self.assertTrue(torch.allclose(out, expected),
                        f'max diff: {(out - expected).abs().max()}')


class TestSmoothQuantFakeQuant(unittest.TestCase):
    """SmoothQuant forward applies smooth_factor-aware fake quant after calibration."""

    def _cfg(self):
        return {
            'batch_num': 1,
            'weights_cfg': {'quant_type': 'int8', 'symmetric': True,
                            'strategy': 'channel', 'group_size': None},
            'inputs_cfg': {'quant_type': 'int8', 'symmetric': True, 'strategy': 'tensor'},
            'algorithm': {'smoothquant': {'smooth_strength': 0.5}},
        }

    def test_after_calibration_uses_smooth_factor(self):
        from amct_pytorch.quantize_op.smooth_quant_module import SmoothQuant
        from amct_pytorch.utils.quant_util import quant_dequant_tensor
        linear = nn.Linear(8, 4).to(torch.bfloat16)
        mod = SmoothQuant(linear, 'l', self._cfg())
        x = torch.randn(2, 8).to(torch.bfloat16)
        mod(x)
        out = mod(x)

        smooth = mod.scale
        dq_w = quant_dequant_tensor(
            mod.weight.data * smooth.to(dtype=mod.weight.dtype),
            'int8', mod.scale_w, mod.offset_w)
        dq_x = quant_dequant_tensor(
            x / smooth.to(dtype=x.dtype),
            'int8', mod.scale_d, mod.offset_d)
        expected = F.linear(dq_x, dq_w, mod.bias)
        self.assertTrue(torch.allclose(out, expected),
                        f'max diff: {(out - expected).abs().max()}')

    def test_quantizable_weight_applies_smooth_factor(self):
        from amct_pytorch.quantize_op.smooth_quant_module import SmoothQuant
        linear = nn.Linear(8, 4).to(torch.bfloat16)
        mod = SmoothQuant(linear, 'l', self._cfg())
        x = torch.randn(2, 8).to(torch.bfloat16)
        mod(x)
        smooth = mod.scale
        expected_qw = mod.weight.data * smooth.to(dtype=mod.weight.dtype)
        self.assertTrue(torch.allclose(mod._quantizable_weight(), expected_qw),
                        '_quantizable_weight must return weight * smooth_factor')


class TestOfmrFakeQuant(unittest.TestCase):
    """OfmrQuant forward applies fake quant after batch_num calibration rounds."""

    def _cfg(self):
        return {
            'batch_num': 1,
            'weights_cfg': {'quant_type': 'int8', 'symmetric': True,
                            'strategy': 'channel', 'group_size': None},
            'inputs_cfg': {'quant_type': 'int8', 'symmetric': True,
                           'strategy': 'tensor', 'enable_quant': False},
        }

    def test_during_calibration_output_equals_fp(self):
        from amct_pytorch.quantize_op.ofmr_quant_module import OfmrQuant
        linear = nn.Linear(8, 4).to(torch.bfloat16)
        mod = OfmrQuant(linear, 'l', self._cfg())
        x = torch.randn(2, 8).to(torch.bfloat16)
        fp_ref = linear(x)
        out = mod(x)
        self.assertTrue(torch.allclose(fp_ref, out),
                        'During calibration, output should equal float output')

    def test_after_calibration_applies_fake_quant(self):
        from amct_pytorch.quantize_op.ofmr_quant_module import OfmrQuant
        from amct_pytorch.utils.quant_util import quant_dequant_tensor
        linear = nn.Linear(8, 4).to(torch.bfloat16)
        mod = OfmrQuant(linear, 'l', self._cfg())
        x = torch.randn(2, 8).to(torch.bfloat16)
        mod(x)
        out = mod(x)

        dq_w = quant_dequant_tensor(mod.weight.data, 'int8', mod.scale_w, mod.offset_w)
        expected = F.linear(x, dq_w, mod.bias)
        self.assertTrue(torch.allclose(out, expected),
                        f'max diff: {(out - expected).abs().max()}')


class TestAWQFakeQuant(unittest.TestCase):
    """LinearAWQuant fake quant uses ori_module.weight (AWQ-scaled), not self.weight."""

    def _make_cfg(self):
        return {
            'batch_num': 1,
            'weights_cfg': {'quant_type': 'int8', 'symmetric': True,
                            'strategy': 'channel', 'group_size': None},
            'inputs_cfg': {'quant_type': None, 'symmetric': True,
                           'strategy': 'tensor', 'enable_quant': False},
            'algorithm': {'awq': {'grids_num': 20, 'duo_scaling': False}},
        }

    def test_quantizable_weight_override_returns_ori_module_weight(self):
        from amct_pytorch.quantize_op.linear_awq_module import LinearAWQuant
        linear = nn.Linear(8, 4).to(torch.bfloat16)
        mod = LinearAWQuant(linear, 'l', self._make_cfg())
        x = torch.randn(2, 8).to(torch.bfloat16)
        mod(x)
        self.assertTrue(
            torch.allclose(mod._quantizable_weight(), mod.ori_module.weight.data),
            '_quantizable_weight must return ori_module.weight.data')

    def test_after_calibration_output_matches_manual(self):
        from amct_pytorch.quantize_op.linear_awq_module import LinearAWQuant
        from amct_pytorch.utils.quant_util import quant_dequant_tensor
        linear = nn.Linear(8, 4).to(torch.bfloat16)
        mod = LinearAWQuant(linear, 'l', self._make_cfg())
        x = torch.randn(2, 8).to(torch.bfloat16)
        mod(x)
        out = mod(x)

        dq_w = quant_dequant_tensor(mod.ori_module.weight.data, 'int8', mod.scale_w, mod.offset_w)
        expected = F.linear(x, dq_w, mod.bias)
        self.assertTrue(torch.allclose(out, expected),
                        f'max diff: {(out - expected).abs().max()}')


class TestApplyProgressiveQuantDequant(unittest.TestCase):
    """apply_progressive_quant_dequant does two-level quant+dequant with correct layout."""

    def test_rejects_wrong_scale_w1_shape(self):
        from amct_pytorch.quantize_op.utils import apply_progressive_quant_dequant
        weight = torch.randn(8, 64).to(torch.bfloat16)
        bad_scale_w1 = torch.ones(4).to(torch.bfloat16)
        scale_w2 = torch.ones(int(8 * 64 / 32), 1).to(torch.bfloat16)
        with self.assertRaises(RuntimeError):
            apply_progressive_quant_dequant(weight, bad_scale_w1, scale_w2)

    def test_rejects_wrong_scale_w2_shape(self):
        from amct_pytorch.quantize_op.utils import apply_progressive_quant_dequant
        weight = torch.randn(8, 64).to(torch.bfloat16)
        scale_w1 = torch.ones(8).to(torch.bfloat16)
        bad_scale_w2 = torch.ones(4, 1).to(torch.bfloat16)
        with self.assertRaises(RuntimeError):
            apply_progressive_quant_dequant(weight, scale_w1, bad_scale_w2)

    def test_dispatches_to_quant_dequant_tensor_twice(self):
        from amct_pytorch.quantize_op import utils as quant_utils
        from amct_pytorch.utils.vars import FLOAT8_E4M3FN, FLOAT4_E2M1

        weight = torch.randn(8, 64).to(torch.bfloat16)
        scale_w1 = torch.ones(8).to(torch.bfloat16)
        scale_w2 = torch.ones(int(8 * 64 / 32), 1).to(torch.bfloat16)

        def fake_qd(tensor, dst_dtype, scale=None, offset=None, group_size=None):
            return tensor.clone()

        with patch.object(quant_utils, 'quant_dequant_tensor', side_effect=fake_qd) as m:
            result = quant_utils.apply_progressive_quant_dequant(weight, scale_w1, scale_w2)

        self.assertEqual(m.call_count, 2)
        first_call, second_call = m.call_args_list
        self.assertEqual(first_call.args[0].shape, weight.transpose(-1, -2).shape)
        self.assertEqual(first_call.args[1], FLOAT8_E4M3FN)
        self.assertIs(first_call.args[2], scale_w1)
        self.assertEqual(second_call.args[0].shape, torch.Size([8 * 64 // 32, 32]))
        self.assertEqual(second_call.args[1], FLOAT4_E2M1)
        self.assertIs(second_call.args[2], scale_w2)
        self.assertEqual(result.shape, weight.shape)
        self.assertEqual(result.dtype, weight.dtype)


class TestFakeQuantForwardProgressiveDispatch(unittest.TestCase):
    """_fake_quant_forward routes to apply_progressive_quant_dequant when scale_w1/scale_w2 are set."""

    def _make_module_with_progressive_scales(self):
        from amct_pytorch.quantize_op.minmax_module import MinMaxQuant
        linear = nn.Linear(8, 4, bias=True).to(torch.bfloat16)
        cfg = {
            'batch_num': 1,
            'weights_cfg': {'quant_type': 'int8', 'symmetric': True,
                            'strategy': 'channel', 'group_size': None},
            'inputs_cfg': {'quant_type': 'int8', 'symmetric': True,
                           'strategy': 'tensor', 'enable_quant': False},
        }
        mod = MinMaxQuant(linear, 'l', cfg)
        mod.scale_w = None
        mod.scale_w1 = torch.ones(4).to(torch.bfloat16)
        mod.scale_w2 = torch.ones(1, 1).to(torch.bfloat16)
        return mod

    def test_progressive_branch_calls_progressive_helper(self):
        from amct_pytorch.quantize_op import utils as quant_utils
        mod = self._make_module_with_progressive_scales()
        x = torch.randn(2, 8).to(torch.bfloat16)
        sentinel_dq_w = torch.randn(4, 8).to(torch.bfloat16)

        with patch.object(quant_utils, 'apply_progressive_quant_dequant',
                          return_value=sentinel_dq_w) as m:
            out = mod._fake_quant_forward(x)

        self.assertEqual(m.call_count, 1)
        self.assertIs(m.call_args.args[1], mod.scale_w1)
        self.assertIs(m.call_args.args[2], mod.scale_w2)
        expected = F.linear(x, sentinel_dq_w, mod.bias)
        self.assertTrue(torch.allclose(out, expected))


class TestFakeQuantForwardConv2dDispatch(unittest.TestCase):
    """_fake_quant_forward must use F.conv2d (with stride/padding/etc.) for Conv2d modules.

    OfmrQuant registers for both Linear and Conv2d; verify the op dispatch
    layout via mock to avoid depending on NPU hardware for FP8/HIFLOAT8 ops.
    """

    def _make_conv2d_module_with_scale(self):
        from amct_pytorch.quantize_op.ofmr_quant_module import OfmrQuant
        conv = nn.Conv2d(4, 8, kernel_size=3, padding=1, stride=1, dilation=1, groups=1).to(torch.bfloat16)
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
        mod.wts_type = 'int8'
        mod.act_type = 'int8'
        return mod, conv

    def test_conv2d_dispatch_uses_F_conv2d_with_conv_attrs(self):
        mod, conv = self._make_conv2d_module_with_scale()
        x = torch.randn(2, 4, 8, 8).to(torch.bfloat16)
        sentinel_dq_w = torch.randn(8, 4, 3, 3).to(torch.bfloat16)

        with patch('amct_pytorch.utils.quant_util.quant_dequant_tensor',
                   return_value=sentinel_dq_w):
            out = mod._fake_quant_forward(x)

        expected = F.conv2d(x, sentinel_dq_w, conv.bias,
                            conv.stride, conv.padding, conv.dilation, conv.groups)
        self.assertEqual(out.shape, expected.shape)
        self.assertTrue(torch.allclose(out, expected),
                        f'max diff: {(out - expected).abs().max()}')


class TestFakeQuantBias(unittest.TestCase):
    """_fake_quant_bias quantizes bias only on INT8 + PER_TENSOR; otherwise raw."""

    def _make_int8_per_tensor_module(self, weight_only=False):
        from amct_pytorch.quantize_op.minmax_module import MinMaxQuant
        linear = nn.Linear(8, 4, bias=True).to(torch.bfloat16)
        cfg = {
            'batch_num': 1,
            'weights_cfg': {'quant_type': 'int8', 'symmetric': True,
                            'strategy': 'channel', 'group_size': None},
            'inputs_cfg': {'quant_type': 'int8', 'symmetric': True,
                           'strategy': 'tensor', 'enable_quant': not weight_only},
        }
        return MinMaxQuant(linear, 'l', cfg)

    def test_returns_none_when_bias_none(self):
        mod = self._make_int8_per_tensor_module()
        mod.bias = None
        deq_scale = torch.ones(4).to(torch.bfloat16)
        self.assertIsNone(mod._fake_quant_bias(deq_scale))

    def test_returns_raw_bias_when_deq_scale_none(self):
        mod = self._make_int8_per_tensor_module()
        self.assertIs(mod._fake_quant_bias(None), mod.bias)

    def test_returns_raw_bias_for_per_token_granularity(self):
        mod = self._make_int8_per_tensor_module()
        mod.act_granularity = 'token'
        deq_scale = torch.ones(4).to(torch.bfloat16)
        self.assertIs(mod._fake_quant_bias(deq_scale), mod.bias)

    def test_returns_raw_bias_for_non_int8_act(self):
        mod = self._make_int8_per_tensor_module()
        mod.act_type = 'float8_e4m3fn'
        deq_scale = torch.ones(4).to(torch.bfloat16)
        self.assertIs(mod._fake_quant_bias(deq_scale), mod.bias)

    def test_int8_per_tensor_round_trip_through_int32(self):
        from amct_pytorch.utils.vars import INT32_MIN, INT32_MAX
        mod = self._make_int8_per_tensor_module()
        deq_scale = torch.tensor([0.25, 0.5, 0.125, 1.0]).to(mod.bias.dtype)
        result = mod._fake_quant_bias(deq_scale)
        expected_int = (mod.bias.data / deq_scale).round().clamp(INT32_MIN, INT32_MAX)
        expected = expected_int * deq_scale
        self.assertTrue(torch.allclose(result, expected),
                        f'max diff: {(result - expected).abs().max()}')


if __name__ == '__main__':
    unittest.main()
