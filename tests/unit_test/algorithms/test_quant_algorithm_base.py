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

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

import importlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from amct_pytorch.algorithms.quant import (
    AlgoBuildContext,
    QuantAlgorithmBase,
    register_algorithms,
)
from amct_pytorch.algorithms.registry_factory import ALGO_REGISTRY


register_algorithms()


def _make_registered_algorithm(name):
    item = ALGO_REGISTRY.get_item(name)
    args = SimpleNamespace(
        algos=[],
        is_per_tensor=True,
        quant_dtype="int",
        w_bits=8,
        w_size=(4, 4),
    )
    targets = item.metadata["targets"]
    if "structure" in targets:
        return item.target(args, AlgoBuildContext(matrix_size=2, dim_size=4))
    if "weight" in targets:
        return item.target(args, args.w_bits)
    return item.target(args)


def test_quant_package_exports_algorithm_base():
    quant_module = importlib.import_module("amct_pytorch.algorithms.quant")
    base_module = importlib.import_module("amct_pytorch.algorithms.quant.base")

    assert quant_module.QuantAlgorithmBase is base_module.QuantAlgorithmBase
    assert "QuantAlgorithmBase" in quant_module.__all__


def _make_test_algorithm(dtype=torch.float32):
    base_module = importlib.import_module("amct_pytorch.algorithms.quant.base")

    class _TestAlgorithm(base_module.QuantAlgorithmBase):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([1.0], dtype=dtype))

        def forward(self, x):
            return x * self.weight

    return _TestAlgorithm()


def test_default_calib_forward_returns_input_object():
    algo = _make_test_algorithm()
    x = torch.randn(2, 4)
    snapshot = x.clone()

    output = algo.calib_forward(x, object(), name="activation")

    assert algo.is_observe is False
    assert output is x
    assert torch.equal(output, snapshot)


def test_base_common_parameter_interfaces_round_trip():
    source = _make_test_algorithm(dtype=torch.float32)
    with torch.no_grad():
        source.weight.fill_(2.5)

    params = source.export_ptq_params()
    target = _make_test_algorithm(dtype=torch.float64)
    target.load_ptq_params({**params, "unknown": torch.tensor([9.0])})

    trainable_params = source.trainable_params()
    assert len(trainable_params) == 1
    assert trainable_params[0] is source.weight
    assert params["weight"].device.type == "cpu"
    assert not params["weight"].requires_grad
    assert target.weight.dtype == torch.float64
    assert target.weight.item() == pytest.approx(2.5)


def test_base_requires_subclass_to_implement_forward():
    base_module = importlib.import_module("amct_pytorch.algorithms.quant.base")

    class _IncompleteAlgorithm(base_module.QuantAlgorithmBase):
        pass

    with pytest.raises(TypeError, match="abstract"):
        _IncompleteAlgorithm()


@pytest.mark.parametrize("name", ALGO_REGISTRY.list_all())
def test_registered_algorithm_satisfies_calibration_contract(name):
    expected_algorithms = {"autoround", "flatquant", "lac", "lwc", "omniquant"}
    assert set(ALGO_REGISTRY.list_all()) == expected_algorithms

    item = ALGO_REGISTRY.get_item(name)
    assert issubclass(item.target, QuantAlgorithmBase)

    algorithm = _make_registered_algorithm(name)
    algorithm.is_observe = True
    x = torch.randn(2, 4)
    snapshot = x.clone()

    output = algorithm.calib_forward(x)

    assert output is x
    assert torch.equal(output, snapshot)
