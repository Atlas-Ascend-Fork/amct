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
# See the License for the specific language governing permissions or
# limitations under the License.
# ----------------------------------------------------------------------------
"""Regression tests for NpuQuantizationLinear._init_bias correctness.

Two invariants are verified:
  1. dtype: aclnnQuantMatmulV5 requires DT_FLOAT (float32) bias in all
     non-INT8-tensor-static modes.  The mock silently casts to float32,
     so only an explicit assertion catches this before real hardware.
  2. scale: npu_quant_matmul adds bias BEFORE applying deq_scale in the
     pertensor/perchannel 8-bit-float path (HIF8*HIF8, FP8*FP8).  The
     bias must therefore be pre-divided by deq_scale so that
     (x@w + bias/deq_scale) * deq_scale == (x@w)*deq_scale + bias.
     Per-token/dynamic paths add bias after scaling, so no pre-division.
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


def _call_init_bias(act_type, wts_type, bias_tensor, act_granularity='tensor',
                    dynamic=None, offset_bias=None, deq_scale=None):
    linear = _make_shell()
    linear.act_granularity = act_granularity
    linear.dynamic = dynamic
    linear.act_type = act_type
    linear.wts_type = wts_type
    n = bias_tensor.shape[0]
    linear.scale_w_tensor = torch.ones(n)
    linear.offset_bias = offset_bias
    if deq_scale is None:
        linear.deq_scale = torch.ones(n)
    else:
        linear.deq_scale = deq_scale

    module = MagicMock()
    module.bias = bias_tensor
    module.scale_d = torch.ones(1)
    linear._init_bias(module)
    return linear


class TestInitBiasDtype(unittest.TestCase):

    def test_hif8x_hif8_bias_is_float32(self):
        linear = _call_init_bias(HIFLOAT8, HIFLOAT8, torch.randn(32).to(torch.float16))
        self.assertEqual(linear.bias.dtype, torch.float32,
                         "HIF8*HIF8 bias must be float32 for aclnnQuantMatmulV5")

    def test_fp8xfp8_bias_is_float32(self):
        linear = _call_init_bias(FLOAT8_E4M3FN, FLOAT8_E4M3FN, torch.randn(32).to(torch.float16))
        self.assertEqual(linear.bias.dtype, torch.float32,
                         "FP8*FP8 bias must be float32 for aclnnQuantMatmulV5")

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
        linear.deq_scale = torch.ones(1)
        linear.offset_bias = None
        module = MagicMock()
        module.bias = None
        linear._init_bias(module)
        self.assertIsNone(linear.bias)


class TestInitBiasPreDivision(unittest.TestCase):

    def test_hif8_pertensor_bias_predivided_by_deq_scale(self):
        bias = torch.randn(32)
        deq_scale = torch.rand(32) + 0.5
        linear = _call_init_bias(HIFLOAT8, HIFLOAT8, bias.clone(), deq_scale=deq_scale)
        expected = bias.to(torch.float32) / deq_scale.to(torch.float32)
        self.assertEqual(linear.bias.dtype, torch.float32)
        torch.testing.assert_close(linear.bias, expected, rtol=1e-6, atol=1e-7)

    def test_fp8_pertensor_bias_predivided_by_deq_scale(self):
        bias = torch.randn(16)
        deq_scale = torch.rand(16) + 0.5
        linear = _call_init_bias(FLOAT8_E4M3FN, FLOAT8_E4M3FN, bias.clone(), deq_scale=deq_scale)
        expected = bias.to(torch.float32) / deq_scale.to(torch.float32)
        torch.testing.assert_close(linear.bias, expected, rtol=1e-6, atol=1e-7)

    def test_int8_pertoken_bias_not_predivided(self):
        bias = torch.randn(32)
        deq_scale = torch.rand(32) + 0.5
        linear = _call_init_bias(INT8, INT8, bias.clone(),
                                 act_granularity='token', deq_scale=deq_scale)
        torch.testing.assert_close(linear.bias, bias.to(torch.float32), rtol=0, atol=0)

    def test_int8_dynamic_bias_not_predivided(self):
        bias = torch.randn(32)
        deq_scale = torch.rand(32) + 0.5
        linear = _call_init_bias(INT8, INT8, bias.clone(),
                                 dynamic=True, deq_scale=deq_scale)
        torch.testing.assert_close(linear.bias, bias.to(torch.float32), rtol=0, atol=0)

    def test_hif8_pertoken_bias_not_predivided(self):
        bias = torch.randn(32)
        deq_scale = torch.rand(32) + 0.5
        linear = _call_init_bias(HIFLOAT8, HIFLOAT8, bias.clone(),
                                 act_granularity='token', deq_scale=deq_scale)
        torch.testing.assert_close(linear.bias, bias.to(torch.float32), rtol=0, atol=0)

    def test_hif8_unit_deq_scale_bias_unchanged(self):
        bias = torch.randn(32)
        linear = _call_init_bias(HIFLOAT8, HIFLOAT8, bias.clone(),
                                 deq_scale=torch.ones(32))
        torch.testing.assert_close(linear.bias, bias.to(torch.float32), rtol=1e-6, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
