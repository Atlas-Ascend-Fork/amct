# ----------------------------------------------------------------------------
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
# ----------------------------------------------------------------------------
"""Regression tests for NpuQuantizationLinear._init_bias dtype correctness.

aclnnQuantMatmulV5 requires DT_FLOAT (float32) bias in all non-INT8-tensor
modes (HIF8×HIF8, FP8×FP8, per-token, dynamic). The mock in mock_torch_npu
silently casts to float32 at line 113, so only an explicit dtype assertion
catches this defect before it reaches real hardware.
"""
import unittest
from unittest.mock import MagicMock

import torch

from amct_pytorch.classic.deploy_op.npu_quantization_linear import NpuQuantizationLinear
from amct_pytorch.common.utils.vars import FLOAT8_E4M3FN, HIFLOAT8, INT8


def _make_shell():
    """Construct a NpuQuantizationLinear bypassing __init__."""
    obj = object.__new__(NpuQuantizationLinear)
    torch.nn.Module.__init__(obj)
    return obj


class _ScaleDOnDifferentDevice:
    def __init__(self, tensor):
        self.tensor = tensor

    @staticmethod
    def __mul__(other):
        raise RuntimeError(
            "Expected all tensors to be on the same device. Expected NPU tensor, "
            "please check whether the input tensor device is correct.")

    def to(self, *args, **kwargs):
        return self.tensor.to(*args, **kwargs)


def _call_init_bias(act_type, wts_type, bias_tensor, act_granularity='tensor',
                    dynamic=None, offset_bias=None, scale_d=None, scale_w=None):
    linear = _make_shell()
    linear.act_granularity = act_granularity
    linear.dynamic = dynamic
    linear.act_type = act_type
    linear.wts_type = wts_type
    linear.scale_w_tensor = scale_w if scale_w is not None else torch.ones(bias_tensor.shape[0])
    linear.offset_bias = offset_bias

    module = MagicMock()
    module.bias = bias_tensor
    module.scale_d = scale_d if scale_d is not None else torch.ones(1)
    linear._init_bias(module)
    return linear


class TestInitBiasDtype(unittest.TestCase):

    def test_hif8x_hif8_bias_is_float32(self):
        linear = _call_init_bias(HIFLOAT8, HIFLOAT8, torch.randn(32).to(torch.float16))
        self.assertEqual(linear.bias.dtype, torch.float32,
                         "HIF8×HIF8 bias must be float32 for aclnnQuantMatmulV5")

    def test_hif8x_hif8_tensor_channel_bias_uses_x2scale_domain(self):
        bias = torch.tensor([2.0, 12.0, -18.0], dtype=torch.float16)
        scale_d = torch.tensor([0.5], dtype=torch.float32)
        scale_w = torch.tensor([2.0, 3.0, 6.0], dtype=torch.float32)
        linear = _call_init_bias(HIFLOAT8, HIFLOAT8, bias, scale_d=scale_d, scale_w=scale_w)

        expected = bias.to(torch.float32) / (scale_d * scale_w)
        self.assertTrue(torch.allclose(linear.bias, expected))
        self.assertEqual(linear.bias.dtype, torch.float32)

    def test_hif8x_hif8_tensor_channel_bias_moves_scale_d_to_weight_device(self):
        bias = torch.tensor([2.0, 12.0, -18.0], dtype=torch.float16)
        scale_d = _ScaleDOnDifferentDevice(torch.tensor([0.5], dtype=torch.float32))
        scale_w = torch.tensor([2.0, 3.0, 6.0], dtype=torch.float32)
        linear = _call_init_bias(HIFLOAT8, HIFLOAT8, bias, scale_d=scale_d, scale_w=scale_w)

        expected = bias.to(torch.float32) / (scale_d.tensor * scale_w)
        self.assertTrue(torch.allclose(linear.bias, expected))
        self.assertEqual(linear.bias.dtype, torch.float32)

    def test_hif8x_hif8_token_bias_keeps_original_fp32(self):
        bias = torch.tensor([2.0, 12.0, -18.0], dtype=torch.float16)
        scale_d = torch.tensor([0.5], dtype=torch.float32)
        scale_w = torch.tensor([2.0, 3.0, 6.0], dtype=torch.float32)
        linear = _call_init_bias(HIFLOAT8, HIFLOAT8, bias, act_granularity='token',
                                 scale_d=scale_d, scale_w=scale_w)

        self.assertTrue(torch.allclose(linear.bias, bias.to(torch.float32)))
        self.assertEqual(linear.bias.dtype, torch.float32)

    def test_fp8xfp8_bias_is_float32(self):
        linear = _call_init_bias(FLOAT8_E4M3FN, FLOAT8_E4M3FN, torch.randn(32).to(torch.float16))
        self.assertEqual(linear.bias.dtype, torch.float32,
                         "FP8×FP8 bias must be float32 for aclnnQuantMatmulV5")

    def test_fp8xfp8_tensor_channel_bias_uses_x2scale_domain(self):
        bias = torch.tensor([4.0, -10.0, 24.0], dtype=torch.float16)
        scale_d = torch.tensor([0.25], dtype=torch.float32)
        scale_w = torch.tensor([2.0, 5.0, 8.0], dtype=torch.float32)
        linear = _call_init_bias(FLOAT8_E4M3FN, FLOAT8_E4M3FN, bias, scale_d=scale_d, scale_w=scale_w)

        expected = bias.to(torch.float32) / (scale_d * scale_w)
        self.assertTrue(torch.allclose(linear.bias, expected))
        self.assertEqual(linear.bias.dtype, torch.float32)

    def test_fp8xfp8_dynamic_bias_keeps_original_fp32(self):
        bias = torch.tensor([4.0, -10.0, 24.0], dtype=torch.float16)
        scale_d = torch.tensor([0.25], dtype=torch.float32)
        scale_w = torch.tensor([2.0, 5.0, 8.0], dtype=torch.float32)
        linear = _call_init_bias(FLOAT8_E4M3FN, FLOAT8_E4M3FN, bias, dynamic=True,
                                 scale_d=scale_d, scale_w=scale_w)

        self.assertTrue(torch.allclose(linear.bias, bias.to(torch.float32)))
        self.assertEqual(linear.bias.dtype, torch.float32)

    def test_int8_pertoken_bias_is_float32(self):
        linear = _call_init_bias(INT8, INT8, torch.randn(32).to(torch.float16),
                                 act_granularity='token')
        self.assertEqual(linear.bias.dtype, torch.float32,
                         "INT8 per-token bias must be float32 for aclnnQuantMatmulV5")

    def test_int8_dynamic_bias_is_float32(self):
        linear = _call_init_bias(INT8, INT8, torch.randn(32).to(torch.float16),
                                 dynamic=True)
        self.assertEqual(linear.bias.dtype, torch.float32,
                         "INT8 dynamic bias must be float32 for aclnnQuantMatmulV5")

    def test_int8_tensor_static_bias_is_int32(self):
        # INT8 per-tensor non-dynamic takes the quantize-to-int32 branch — unchanged.
        linear = _call_init_bias(INT8, INT8, torch.randn(32).to(torch.float32))
        self.assertEqual(linear.bias.dtype, torch.int32,
                         "INT8 per-tensor static bias must stay int32")

    def test_none_bias_stays_none(self):
        linear = _make_shell()
        linear.act_granularity = 'tensor'
        linear.dynamic = None
        linear.act_type = HIFLOAT8
        linear.wts_type = HIFLOAT8
        linear.scale_w_tensor = torch.ones(1)
        linear.offset_bias = None
        module = MagicMock()
        module.bias = None
        linear._init_bias(module)
        self.assertIsNone(linear.bias)


if __name__ == '__main__':
    unittest.main()
