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
# Importing gptq triggers its ALGO_REGISTRY.register(name="gptq"/"higptq", ...) decorators
# regardless of whether the lazy register_algorithms() has run.
from amct_pytorch.algorithms.quant.gptq import build_gptq  # noqa: F401
from amct_pytorch.common.optimization.calibration_solver import (
    CalibrationSolver, _build_gptq_weights_cfg)
from amct_pytorch.quantization.dtypes import register_dtype
from amct_pytorch.quantization.modules.quant_linear import QuantLinear

sys.modules["torch_npu"] = MagicMock()

register_dtype()


def _args(algos, quant_dtype="int", w_bits=8):
    return SimpleNamespace(
        algos=list(algos),
        quant_dtype=quant_dtype,
        w_bits=w_bits,
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


def _build_block(in_features=16, out_features=16, algos=("gptq",), quant_dtype="int", w_bits=8):
    linear = nn.Linear(in_features, out_features)
    args = _args(algos, quant_dtype=quant_dtype, w_bits=w_bits)
    ql = QuantLinear(args, linear, w_bits=w_bits, name="proj")
    return args, linear, _Block(ql)


def _build_loader(n=8, dim=16, batch_size=4):
    return DataLoader(TensorDataset(torch.randn(n, dim)), batch_size=batch_size)


class _registered_calibration_algo:
    """Temporarily registers a dummy solver="calibrate" entry in ALGO_REGISTRY (no targets,
    matching gptq's real registration -- calibrate-class algos are looked up from
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


def test_gptq_and_higptq_alias_resolve_to_the_same_dispatcher():
    # "gptq" is the canonical registered name; "higptq" is kept as a backward-compat alias
    # pointing at the exact same build_gptq target (amct_pytorch/algorithms/quant/gptq.py),
    # not a second, distinct algorithm.
    assert ALGO_REGISTRY.get("gptq") is build_gptq
    assert ALGO_REGISTRY.get("higptq") is build_gptq


# ---- _build_gptq_weights_cfg (per-layer dtype/bit-width -> weights_cfg) -----


def test_build_gptq_weights_cfg_selects_quant_type_by_dtype_and_bits():
    assert _build_gptq_weights_cfg("int", 4) == {
        "quant_type": "int4", "symmetric": True, "strategy": "channel"}
    assert _build_gptq_weights_cfg("int", 8) == {
        "quant_type": "int8", "symmetric": True, "strategy": "channel"}
    assert _build_gptq_weights_cfg("mxfp", 4) == {
        "quant_type": "mxfp4_e2m1", "symmetric": True, "strategy": "group", "group_size": 32}
    assert _build_gptq_weights_cfg("mxfp", 8) == {
        "quant_type": "mxfp8_e4m3fn", "symmetric": True, "strategy": "group", "group_size": 32}
    assert _build_gptq_weights_cfg("hifp", 4) == {
        "quant_type": "hifloat4", "symmetric": True, "strategy": "channel"}


def test_build_gptq_weights_cfg_rejects_unsupported_bit_width():
    with pytest.raises(ValueError, match="unsupported int w_bits"):
        _build_gptq_weights_cfg("int", 16)
    with pytest.raises(ValueError, match="unsupported mxfp w_bits"):
        _build_gptq_weights_cfg("mxfp", 16)


def test_solve_raises_when_multiple_calibration_algos_matched():
    with _registered_calibration_algo("dummy_calib"):
        args, _, block = _build_block(algos=("gptq", "dummy_calib"))
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
    args = _args(("gptq",))
    solver = CalibrationSolver(args, layer_idx=0, model=nn.Linear(4, 4))
    assert solver.solve(_build_loader()) is None


def test_solve_returns_none_when_data_loader_empty():
    args, _, block = _build_block()
    solver = CalibrationSolver(args, layer_idx=0, model=block)
    empty_loader = DataLoader(TensorDataset(torch.empty(0, 16)), batch_size=4)
    assert solver.solve(empty_loader) is None


# ---- end-to-end gptq dispatch ----------------------------------------------


def test_solve_runs_gptq_and_mutates_weight_in_place():
    args, linear, block = _build_block()
    original_weight = linear.weight.detach().clone()

    solver = CalibrationSolver(args, layer_idx=0, model=block)
    assert solver.solve(_build_loader()) is None

    assert not torch.allclose(linear.weight.detach(), original_weight)


def test_solve_runs_via_higptq_alias_and_mutates_weight_in_place():
    # quant_dtype must be set to "hifp" explicitly here: dispatch is now dtype-driven (see
    # _build_gptq_weights_cfg), so without this the alias would just retest the int8 path.
    args, linear, block = _build_block(algos=("higptq",), quant_dtype="hifp", w_bits=4)
    original_weight = linear.weight.detach().clone()

    solver = CalibrationSolver(args, layer_idx=0, model=block)
    assert solver.solve(_build_loader()) is None

    assert not torch.allclose(linear.weight.detach(), original_weight)


def test_solve_runs_gptq_with_int4_and_mutates_weight_in_place():
    args, linear, block = _build_block(quant_dtype="int", w_bits=4)
    original_weight = linear.weight.detach().clone()

    solver = CalibrationSolver(args, layer_idx=0, model=block)
    assert solver.solve(_build_loader()) is None

    assert not torch.allclose(linear.weight.detach(), original_weight)


def test_solve_runs_gptq_with_mxfp_and_mutates_weight_in_place():
    args, linear, block = _build_block(quant_dtype="mxfp", w_bits=4)
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
