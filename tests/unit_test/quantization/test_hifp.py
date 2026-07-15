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
import sys
from unittest.mock import patch

import pytest

import torch

from amct_pytorch.cli.llm.args import parser_gen
from amct_pytorch.quantization.dtypes import DTYPE_REGISTRY, hifp as hifp_dtype
from amct_pytorch.quantization.dtypes.hifp import QuantDequantHifp


def test_hifp_is_registered_under_hifp_name_only():
    assert DTYPE_REGISTRY.get("hifp") is QuantDequantHifp
    with pytest.raises(KeyError):
        DTYPE_REGISTRY.get("hif")


def test_hifp_quant_dtype_cli_option_replaces_hif():
    with patch.object(sys, "argv", ["amct", "--quant_dtype", "hifp"]):
        assert parser_gen().quant_dtype == "hifp"

    with patch.object(sys, "argv", ["amct", "--quant_dtype", "hif"]):
        with pytest.raises(SystemExit):
            parser_gen()


def test_hifp_quant_passes_gradient_through_ste():
    x = torch.randn(2, 8, dtype=torch.float32, requires_grad=True)
    quantized = torch.full_like(x, 0.25)

    with patch.object(
        hifp_dtype, "hifloat8_fake_quant", return_value=quantized
    ):
        out = QuantDequantHifp(bits=8)(x)

    assert torch.equal(out, quantized)
    out.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_hifp_passthrough_fallback_preserves_gradient():
    x = torch.randn(2, 8, dtype=torch.float32, requires_grad=True)

    with patch.object(
        hifp_dtype,
        "hifloat8_fake_quant",
        side_effect=lambda tensor: tensor.detach(),
    ):
        out = QuantDequantHifp(bits=8)(x)

    assert torch.equal(out, x)
    out.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_hifp_forward_bits16_is_passthrough():
    x = torch.randn(2, 8, requires_grad=True)

    out = QuantDequantHifp(bits=16)(x)

    assert out is x
