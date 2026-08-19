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

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from amct_pytorch.algorithms.quant.classic_llm import (
    LlmAwq,
    LlmDiagonalGptq,
    LlmMinMax,
)
from amct_pytorch.quantization.dtypes.int import QuantDequantInt


def _args(rows=3, columns=4):
    return SimpleNamespace(quant_dtype="int", w_size=(rows, columns))


def _round_trip(algorithm, params):
    restored = type(algorithm)(_args(algorithm.rows, algorithm.columns), algorithm.bits)
    restored.load_ptq_params(params)
    return restored


def test_minmax_persists_the_scale_that_controls_quantization():
    weight = torch.tensor(
        [[-4.0, -0.8, 0.3, 3.1], [-0.7, -0.2, 0.4, 0.9]],
        dtype=torch.float32,
    )
    algorithm = LlmMinMax(_args(2, 4), bits=4)
    algorithm.calib_forward(weight)
    params = algorithm.export_ptq_params()
    restored = _round_trip(algorithm, params)
    quant_obj = QuantDequantInt(bits=4)

    expected = algorithm.quantize(weight, quant_obj)
    actual = restored.quantize(weight, quant_obj)

    assert torch.equal(actual, expected)
    assert torch.equal(params["weight_scale"], weight.abs().amax(1, keepdim=True) / 7)
    altered = dict(params)
    altered["weight_scale"] = params["weight_scale"] * 0.5
    changed = _round_trip(algorithm, altered).quantize(weight, quant_obj)
    assert not torch.equal(changed, expected)


def test_awq_compensates_the_activation_and_weight_transforms():
    weight = torch.tensor(
        [[0.2, -2.0, 0.5, 4.0], [0.7, 1.0, -3.0, 0.1]],
        dtype=torch.float32,
    )
    inputs = torch.tensor(
        [[100.0, 1.0, 0.1, 4.0], [80.0, -2.0, 0.2, -3.0]],
        dtype=torch.float32,
    )
    algorithm = LlmAwq(_args(2, 4), bits=4)
    algorithm.observe_input(inputs, weight)
    params = algorithm.export_ptq_params()

    scale = params["channel_scale"].to(inputs)
    assert not torch.equal(scale, torch.ones_like(scale))
    assert torch.allclose(
        F.linear(inputs / scale, weight * scale),
        F.linear(inputs, weight),
        atol=1e-5,
        rtol=1e-5,
    )


def test_awq_artifact_round_trip_reproduces_fake_quant_output():
    torch.manual_seed(7)
    weight = torch.randn(3, 4)
    inputs = torch.randn(12, 4) * torch.tensor([20.0, 1.0, 0.1, 4.0])
    algorithm = LlmAwq(_args(), bits=4)
    algorithm.observe_input(inputs, weight)
    restored = _round_trip(algorithm, algorithm.export_ptq_params())
    quant_obj = QuantDequantInt(bits=4)

    expected = F.linear(
        algorithm.transform_input(inputs), algorithm.quantize(weight, quant_obj)
    )
    actual = F.linear(
        restored.transform_input(inputs), restored.quantize(weight, quant_obj)
    )
    assert torch.equal(actual, expected)


def test_gptq_diagonal_hessian_ignores_low_importance_outlier_when_clipping():
    weight = torch.tensor([[10.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    inputs = torch.tensor(
        [[1e-4, 1.0, 1.0, 1.0], [-1e-4, -2.0, 2.0, -1.0]],
        dtype=torch.float32,
    )
    algorithm = LlmDiagonalGptq(_args(1, 4), bits=4)
    algorithm.observe_input(inputs, weight)
    params = algorithm.export_ptq_params()

    assert params["clip_factor"].item() < 1.0
    restored = _round_trip(algorithm, params)
    quant_obj = QuantDequantInt(bits=4)
    assert torch.equal(
        restored.quantize(weight, quant_obj), algorithm.quantize(weight, quant_obj)
    )


@pytest.mark.parametrize("algorithm_cls", [LlmMinMax, LlmAwq, LlmDiagonalGptq])
def test_statistical_algorithms_reject_non_int_dtype(algorithm_cls):
    args = _args()
    args.quant_dtype = "mxfp"
    with pytest.raises(ValueError, match="quant_dtype='int'"):
        algorithm_cls(args, bits=4)


def test_awq_and_gptq_require_observed_activations_before_export():
    for algorithm_cls in (LlmAwq, LlmDiagonalGptq):
        algorithm = algorithm_cls(_args(), bits=4)
        with pytest.raises(RuntimeError, match="observed weight"):
            algorithm.export_ptq_params()


@pytest.mark.parametrize("algorithm_cls", [LlmMinMax, LlmDiagonalGptq])
@pytest.mark.parametrize("bits", [4, 8])
def test_minmax_and_gptq_deploy_reconstruct_fake_quant(algorithm_cls, bits):
    torch.manual_seed(11)
    weight = torch.randn(3, 256, dtype=torch.bfloat16)
    inputs = torch.randn(4, 256, dtype=torch.bfloat16)
    algorithm = algorithm_cls(_args(3, 256), bits=bits)
    if algorithm_cls is LlmMinMax:
        algorithm.calib_forward(weight)
    else:
        algorithm.observe_input(inputs, weight)
        algorithm.export_ptq_params()
    restored = _round_trip(algorithm, algorithm.export_ptq_params())
    quant_obj = QuantDequantInt(bits=bits)

    expected = restored.quantize(weight, quant_obj)
    if bits == 4:
        payload = restored.export_dense_linear_deploy(weight, quant_obj)
    else:
        payload = restored.export_deploy(weight, quant_obj)
    actual = payload["qweight"].to(weight) * payload["weight_scale"].to(weight)

    assert torch.equal(actual, expected)


def test_awq_deploy_refuses_missing_runtime_input_transform():
    algorithm = LlmAwq(_args(2, 256), bits=4)
    weight = torch.randn(2, 256)
    inputs = torch.randn(4, 256)
    algorithm.observe_input(inputs, weight)
    algorithm.export_ptq_params()

    with pytest.raises(NotImplementedError, match="input-channel transformation"):
        algorithm.export_dense_linear_deploy(weight, QuantDequantInt(bits=4))
