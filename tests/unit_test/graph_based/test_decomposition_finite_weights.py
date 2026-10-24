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
"""Exercise real NumPy validation, isolating the external native filter boundary."""

import importlib.util
from pathlib import Path
import types
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
GRAPH = ROOT / "amct_pytorch/classic/graph_based"


@pytest.fixture
def decomposition_api(monkeypatch):
    # Load the entire tracked Python module without the CANN package initializer.
    # The external filter is the only controlled execution boundary below.
    package_name = "_amct_finite_weight_host"
    for suffix in ("", ".common", ".common.decomposition", ".lib"):
        package = types.ModuleType(package_name + suffix)
        directory = GRAPH.joinpath(*suffix.strip(".").split("."))
        package.__path__ = [str(directory)]
        package.__file__ = str(directory / "__init__.py")
        monkeypatch.setitem(sys.modules, package_name + suffix, package)
    module_name = package_name + ".common.decomposition.decomposition"
    spec = importlib.util.spec_from_file_location(
        module_name, GRAPH / "common/decomposition/decomposition.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def unchanged_filter(info):
        calls.append(info.conv_info_ctype)
        return 0

    # The real C++ FastFilterConv returns UNCHANGE for these 1x1 kernels.
    # Native loading, rank estimation, SVD, and device execution are not tested.
    monkeypatch.setattr(module, "fast_filter_conv", unchanged_filter)
    return module, calls


@pytest.mark.parametrize(
    "values",
    [
        [1e308, 1e308, 1e308, 1e308],
        [-1e308, -1e308, -1e308, -1e308],
        [1e308, 1e308, -1e308, -1e308],
    ],
)
def test_finite_weights_are_not_rejected_by_sum_overflow(decomposition_api, values):
    module, calls = decomposition_api
    weight = np.array(values, dtype=np.float64).reshape(2, 2, 1, 1)
    assert np.isfinite(weight).all()
    before = weight.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        result = module.tensor_decomposition(weight)
    assert result == {
        "mode": module.DecomposeMode.UNCHANGE,
        "first": None,
        "last": None,
    }
    assert len(calls) == 1
    info = calls[0]
    assert (
        info.in_channel,
        info.out_channel,
        info.kernel_size_h,
        info.kernel_size_w,
    ) == (2, 2, 1, 1)
    np.testing.assert_array_equal(weight, before)


@pytest.mark.parametrize("value", [0.0, 0.125])
def test_ordinary_finite_weights_keep_unchanged_result(decomposition_api, value):
    module, calls = decomposition_api
    result = module.tensor_decomposition(np.full((2, 2, 1, 1), value))
    assert result == {
        "mode": module.DecomposeMode.UNCHANGE,
        "first": None,
        "last": None,
    }
    assert len(calls) == 1


@pytest.mark.parametrize(
    "values",
    [[np.inf, 0, 0, 0], [-np.inf, 0, 0, 0], [np.nan, 0, 0, 0], [np.inf, -np.inf, 0, 0]],
)
def test_nonfinite_elements_still_fail_before_native_filter(decomposition_api, values):
    module, calls = decomposition_api
    weight = np.array(values).reshape(2, 2, 1, 1)
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="weight tensor contains invalid number"):
            module.tensor_decomposition(weight)
    assert calls == []


def test_wrong_weight_dimensions_still_fail_before_native_filter(decomposition_api):
    module, calls = decomposition_api
    with pytest.raises(ValueError, match="weight tensor must have shape"):
        module.tensor_decomposition(np.zeros((2, 2, 1)))
    assert calls == []
