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
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""Real CPU quantizers: FP16 scale arithmetic and actual QuantLinear consumption."""

from types import SimpleNamespace
import pytest
import torch
from amct_pytorch.quantization.dtypes.fp import QuantDequantFP
from amct_pytorch.quantization.dtypes.int import QuantDequantInt
from amct_pytorch.quantization.dtypes.mxfp import QuantDequantMx
from amct_pytorch.quantization.modules.quant_linear import QuantLinear

pytestmark = pytest.mark.cpu

CASES = [
    ("mx", 4, "fp32"),
    ("mx", 8, "fp32"),
    ("fp", 4, "e8m0"),
    ("fp", 8, "e8m0"),
    ("fp", 8, "fp32"),
]


def make_quant(kind, bits, scale_dtype):
    if kind == "mx":
        return QuantDequantMx(bits=bits)
    return QuantDequantFP(bits=bits, scale_dtype=scale_dtype)


@pytest.mark.parametrize("kind,bits,scale_dtype", CASES)
@pytest.mark.parametrize("value", [0.0, 1e-6, 1.0])
def test_fp16_fake_quant_and_payload_match_float32_reference(
    kind, bits, scale_dtype, value
):
    x = torch.full((2, 32), value, dtype=torch.float16)
    quant = make_quant(kind, bits, scale_dtype)
    actual = quant(x)
    reference = quant(x.float()).half()
    assert actual.dtype == x.dtype
    assert actual.shape == x.shape
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    payload, expected = quant.export_deploy(x), quant.export_deploy(x.float())
    for key in ["qweight", "weight_scale"]:
        left, right = payload[key], expected[key]
        if left.dtype == torch.float8_e4m3fn:
            left, right = left.float(), right.float()
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("kind,bits,scale_dtype", CASES)
def test_fp16_zero_weight_keeps_finite_ste_gradient(kind, bits, scale_dtype):
    x = torch.zeros((2, 32), dtype=torch.float16, requires_grad=True)
    make_quant(kind, bits, scale_dtype)(x).sum().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("kind,bits,scale_dtype", CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_other_input_dtypes_and_passthrough_keep_shape_and_dtype(
    kind, bits, scale_dtype, dtype
):
    x = torch.zeros((2, 32), dtype=dtype)
    quant = make_quant(kind, bits, scale_dtype)
    y = quant(x)
    assert y.dtype == x.dtype
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    torch.testing.assert_close(y, x, rtol=0, atol=0)
    quant.is_observe = True
    assert quant(x) is x
    assert make_quant(kind, 16, scale_dtype)(x) is x


@pytest.mark.parametrize("kind", ["int", "mxfp", "fp"])
def test_quantlinear_consumes_fp16_zero_weight_in_train_and_eval(kind):
    linear = torch.nn.Linear(32, 2, bias=False, dtype=torch.float16)
    with torch.no_grad():
        linear.weight.zero_()
    args = SimpleNamespace(
        quant_dtype=kind, w_bits=8, algos=[], block_size_col=128, scale_dtype="fp32"
    )
    module = QuantLinear(args, linear)
    assert isinstance(
        module.weight_quantizer.quant_obj,
        {"int": QuantDequantInt, "mxfp": QuantDequantMx, "fp": QuantDequantFP}[kind],
    )
    x = torch.ones((1, 32), dtype=torch.float16)
    for mode in [False, True]:
        module.eval_mode = mode
        y = module(x)
        assert y.dtype == x.dtype
        torch.testing.assert_close(
            y, torch.zeros((1, 2), dtype=x.dtype), rtol=0, atol=0
        )
