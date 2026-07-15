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
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from amct_pytorch.algorithms.registry_factory import ALGO_REGISTRY
# Importing higptq triggers its ALGO_REGISTRY.register(name="higptq", ...) decorator
# regardless of whether the lazy register_algorithms() has run.
from amct_pytorch.algorithms.quant.higptq import HiGPTQuant  # noqa: F401
from amct_pytorch.common.optimization.calibration_solver import CalibrationSolver
from amct_pytorch.quantization.dtypes import register_dtype
from amct_pytorch.quantization.modules.quant_linear import QuantLinear

sys.modules["torch_npu"] = MagicMock()

register_dtype()


def _args(algos):
    return SimpleNamespace(
        algos=list(algos),
        quant_dtype="int",
        w_bits=8,
        quant_target=[],
        device=torch.device("cpu"),
    )


class _Block(nn.Module):
    """Mirrors how a PTQ unit's module wraps one or more QuantLinears -- CalibrationSolver
    scans for QuantLinear submodules, and PtqParamHandler.export_module skips the root
    itself, so a bare QuantLinear passed as `self.model` would export nothing."""

    def __init__(self, quant_linear):
        super().__init__()
        self.proj = quant_linear

    def forward(self, x, **kwargs):
        return self.proj(x)


def _build_block(in_features=16, out_features=16, algos=("higptq",)):
    linear = nn.Linear(in_features, out_features)
    args = _args(algos)
    ql = QuantLinear(args, linear, w_bits=8, name="proj")
    return args, linear, _Block(ql)


def _build_loader(n=8, dim=16, batch_size=4):
    return DataLoader(TensorDataset(torch.randn(n, dim)), batch_size=batch_size)


class _registered_calibration_algo:
    """Temporarily registers a dummy solver="calibrate" entry in ALGO_REGISTRY (no targets,
    matching higptq's real registration -- calibrate-class algos are looked up from
    args.algos directly, not via get_algo_names_by_target)."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        ALGO_REGISTRY.register(name=self.name, solver="calibrate")(type(self.name, (), {}))
        return self.name

    def __exit__(self, *exc_info):
        ALGO_REGISTRY._items.pop(self.name, None)


# ---- algo resolution -------------------------------------------------------


def test_solve_raises_when_no_calibration_algo_in_args():
    args, _, block = _build_block(algos=())
    solver = CalibrationSolver(args, layer_idx=0, model=block)
    with pytest.raises(ValueError, match="supports exactly one calibration algo"):
        solver.solve(_build_loader())


def test_plain_gptq_name_is_not_registered_since_only_higptq_is_wired_up():
    # "gptq" (plain int/mxfp/hifloat8 GPTQuant) and "higptq" (HiGPTQuant, HiFloat4-only)
    # are two distinct algorithms; only "higptq" is registered in ALGO_REGISTRY today, so
    # passing "gptq" must fail loudly instead of silently being treated as an alias. Any
    # unregistered name in args.algos is rejected as soon as get_algo_names_by_target runs
    # for ANY target -- here that's WeightQuantizer's own lookup inside QuantLinear's
    # constructor, before CalibrationSolver.solve() is ever reached.
    with pytest.raises(KeyError, match="'gptq' is not registered"):
        _build_block(algos=("gptq",))


def test_solve_raises_when_multiple_calibration_algos_matched():
    with _registered_calibration_algo("dummy_calib"):
        args, _, block = _build_block(algos=("higptq", "dummy_calib"))
        solver = CalibrationSolver(args, layer_idx=0, model=block)
        with pytest.raises(ValueError, match="supports exactly one calibration algo"):
            solver.solve(_build_loader())


def test_solve_raises_for_calibration_algo_with_no_dispatch_branch():
    with _registered_calibration_algo("not_dispatched"):
        args, _, block = _build_block(algos=("not_dispatched",))
        solver = CalibrationSolver(args, layer_idx=0, model=block)
        with pytest.raises(ValueError, match="no dispatch branch for algo 'not_dispatched'"):
            solver.solve(_build_loader())


# ---- early-return guards ----------------------------------------------------


def test_solve_returns_none_when_no_quant_linear_in_model():
    args = _args(("higptq",))
    solver = CalibrationSolver(args, layer_idx=0, model=nn.Linear(4, 4))
    assert solver.solve(_build_loader()) is None


def test_solve_returns_none_when_data_loader_empty():
    args, _, block = _build_block()
    solver = CalibrationSolver(args, layer_idx=0, model=block)
    empty_loader = DataLoader(TensorDataset(torch.empty(0, 16)), batch_size=4)
    assert solver.solve(empty_loader) is None


# ---- end-to-end higptq dispatch ----------------------------------------------


def test_solve_runs_higptq_and_mutates_weight_in_place():
    args, linear, block = _build_block()
    original_weight = linear.weight.detach().clone()

    solver = CalibrationSolver(args, layer_idx=0, model=block)
    assert solver.solve(_build_loader()) is None

    assert not torch.allclose(linear.weight.detach(), original_weight)


def test_finalize_exports_quant_linear_weight():
    args, linear, block = _build_block()
    solver = CalibrationSolver(args, layer_idx=0, model=block)
    solver.solve(_build_loader())

    params = solver.finalize()
    assert "proj" in params
    assert torch.equal(params["proj"]["weight"], linear.weight.detach().cpu())
