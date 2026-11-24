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
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

import amct_pytorch


_DEEPSEEK_V32_ROOT = (
    Path(amct_pytorch.__file__).resolve().parent
    / "experimental"
    / "quantization"
    / "DeepSeekV3.2"
)


def _import_quant_linear():
    root = str(_DEEPSEEK_V32_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from cores.quantization.linear import QuantLinear

    return QuantLinear


def _build_quant_linear(linear):
    QuantLinear = _import_quant_linear()
    args = SimpleNamespace(w_bits=8, w_asym=False, lwc=False)
    quant_linear = QuantLinear(args, linear)
    quant_linear._eval_mode = True
    return quant_linear


def test_eval_forward_uses_wrapped_linear_weight_and_bias():
    linear = nn.Linear(4, 3)
    quant_linear = _build_quant_linear(linear)
    hidden_states = torch.randn(2, 4)

    output = quant_linear(hidden_states)
    expected = torch.nn.functional.linear(hidden_states, linear.weight, linear.bias)

    assert torch.allclose(output, expected)


def test_eval_forward_handles_missing_bias():
    linear = nn.Linear(4, 3, bias=False)
    quant_linear = _build_quant_linear(linear)
    hidden_states = torch.randn(2, 4)

    output = quant_linear(hidden_states)
    expected = torch.nn.functional.linear(hidden_states, linear.weight, None)

    assert torch.allclose(output, expected)
