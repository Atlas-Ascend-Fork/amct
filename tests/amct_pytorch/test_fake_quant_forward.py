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
import sys
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


class TestBaseQuantizeFakeQuant(unittest.TestCase):
    """_fake_quant_forward produces quant-dequant output, not raw float output."""

    def setUp(self):
        sys.modules.pop('torch_npu', None)

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
        """_fake_quant_forward output must differ from raw float output."""
        mod, linear = self._make_weight_only_module()
        x = torch.randn(2, 8).to(torch.bfloat16)
        fp_out = linear(x)
        fake_out = mod._fake_quant_forward(x)
        self.assertFalse(torch.allclose(fp_out, fake_out, atol=0),
                         "fake quant output must differ from float output")

    def test_fake_quant_forward_matches_manual_quant_dequant(self):
        """_fake_quant_forward must equal manual quant_dequant(weight) @ x."""
        from amct_pytorch.utils.quant_util import quant_dequant_tensor
        mod, _ = self._make_weight_only_module()
        x = torch.randn(2, 8).to(torch.bfloat16)
        fake_out = mod._fake_quant_forward(x)
        dq_w = quant_dequant_tensor(mod.weight.data, 'int8', mod.scale_w, mod.offset_w, mod.group_size)
        expected = F.linear(x, dq_w, mod.bias)
        self.assertTrue(torch.allclose(fake_out, expected),
                        f"max diff: {(fake_out - expected).abs().max()}")


class TestMinMaxFakeQuant(unittest.TestCase):
    """MinMaxQuant forward uses fake quant after calibration."""

    def setUp(self):
        sys.modules.pop('torch_npu', None)

    def _cfg(self, weight_only=True):
        return {
            'batch_num': 2,
            'weights_cfg': {'quant_type': 'int8', 'symmetric': True,
                            'strategy': 'channel', 'group_size': None},
            'inputs_cfg': {'quant_type': 'int8', 'symmetric': True,
                           'strategy': 'tensor', 'enable_quant': not weight_only},
        }

    def test_during_calibration_output_equals_fp(self):
        """weight+act path: output equals fp_out while cur_batch <= batch_num."""
        from amct_pytorch.quantize_op.minmax_module import MinMaxQuant
        linear = nn.Linear(8, 4).to(torch.bfloat16)
        mod = MinMaxQuant(linear, 'l', self._cfg(weight_only=False))
        x = torch.randn(2, 8).to(torch.bfloat16)
        fp_ref = linear(x)
        out = mod(x)   # cur_batch=1, batch_num=2 → calibrating
        self.assertTrue(torch.allclose(fp_ref, out),
                        "During calibration, output should equal float output")

    def test_weight_compress_only_applies_fake_quant_immediately(self):
        """weight_compress_only=True: every forward applies fake quant."""
        from amct_pytorch.quantize_op.minmax_module import MinMaxQuant
        from amct_pytorch.utils.quant_util import quant_dequant_tensor
        linear = nn.Linear(8, 4).to(torch.bfloat16)
        mod = MinMaxQuant(linear, 'l', self._cfg(weight_only=True))
        x = torch.randn(2, 8).to(torch.bfloat16)
        out = mod(x)
        dq_w = quant_dequant_tensor(mod.weight.data, 'int8', mod.scale_w, mod.offset_w)
        expected = F.linear(x, dq_w, mod.bias)
        self.assertTrue(torch.allclose(out, expected),
                        f"max diff: {(out - expected).abs().max()}")

    def test_after_calibration_weight_act_fake_quant(self):
        """weight+act: after batch_num rounds, both are fake-quanted."""
        from amct_pytorch.quantize_op.minmax_module import MinMaxQuant
        from amct_pytorch.utils.quant_util import quant_dequant_tensor
        linear = nn.Linear(8, 4).to(torch.bfloat16)
        mod = MinMaxQuant(linear, 'l', self._cfg(weight_only=False))
        x = torch.randn(2, 8).to(torch.bfloat16)
        mod(x)   # cur_batch=1 (calibrating)
        mod(x)   # cur_batch=2 == batch_num → scale_d computed, returns fp_out
        out = mod(x)  # cur_batch=3 > batch_num → fake quant
        dq_w = quant_dequant_tensor(mod.weight.data, 'int8', mod.scale_w, mod.offset_w)
        dq_x = quant_dequant_tensor(x, 'int8', mod.scale_d, mod.offset_d)
        expected = F.linear(dq_x, dq_w, mod.bias)
        self.assertTrue(torch.allclose(out, expected),
                        f"max diff: {(out - expected).abs().max()}")


class TestSmoothQuantFakeQuant(unittest.TestCase):
    """SmoothQuant forward applies smooth_factor-aware fake quant after calibration."""

    def setUp(self):
        sys.modules.pop('torch_npu', None)

    def _cfg(self):
        return {
            'batch_num': 1,
            'weights_cfg': {'quant_type': 'int8', 'symmetric': True,
                            'strategy': 'channel', 'group_size': None},
            'inputs_cfg': {'quant_type': 'int8', 'symmetric': True, 'strategy': 'tensor'},
            'algorithm': {'smoothquant': {'smooth_strength': 0.5}},
        }

    def test_after_calibration_uses_smooth_factor(self):
        """Fake quant uses weight * smooth_factor and input / smooth_factor."""
        from amct_pytorch.quantize_op.smooth_quant_module import SmoothQuant
        from amct_pytorch.utils.quant_util import quant_dequant_tensor
        linear = nn.Linear(8, 4).to(torch.bfloat16)
        mod = SmoothQuant(linear, 'l', self._cfg())
        x = torch.randn(2, 8).to(torch.bfloat16)
        mod(x)           # cur_batch=1 == batch_num → params computed, returns raw output
        out = mod(x)     # cur_batch=2 > batch_num → fake quant

        smooth = mod.scale  # smooth_factor, shape [1, cin]
        dq_w = quant_dequant_tensor(
            mod.weight.data * smooth.to(dtype=mod.weight.dtype),
            'int8', mod.scale_w, mod.offset_w)
        dq_x = quant_dequant_tensor(
            x / smooth.to(dtype=x.dtype),
            'int8', mod.scale_d, mod.offset_d)
        expected = F.linear(dq_x, dq_w, mod.bias)
        self.assertTrue(torch.allclose(out, expected),
                        f"max diff: {(out - expected).abs().max()}")

    def test_quantizable_weight_applies_smooth_factor(self):
        """_quantizable_weight returns weight * smooth_factor (not raw weight)."""
        from amct_pytorch.quantize_op.smooth_quant_module import SmoothQuant
        linear = nn.Linear(8, 4).to(torch.bfloat16)
        mod = SmoothQuant(linear, 'l', self._cfg())
        x = torch.randn(2, 8).to(torch.bfloat16)
        mod(x)   # calibration completes, mod.scale = smooth_factor
        smooth = mod.scale
        expected_qw = mod.weight.data * smooth.to(dtype=mod.weight.dtype)
        self.assertTrue(torch.allclose(mod._quantizable_weight(), expected_qw),
                        "_quantizable_weight must return weight * smooth_factor")
