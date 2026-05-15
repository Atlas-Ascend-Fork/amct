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
from types import SimpleNamespace

import pytest

from amct_pytorch.common.optimization import SOLVER_REGISTRY
from amct_pytorch.common.optimization.global_solver import GlobalSolver


class _ConcreteSolver(GlobalSolver):
    def solve(self, calibration_data):
        pass


def _make_model(quant_target=None):
    model = SimpleNamespace()
    model.quant_target = quant_target or ["mlp"]
    return model


def test_global_solver_registered_in_solver_registry():
    cls = SOLVER_REGISTRY.get("global")
    assert cls is GlobalSolver


def test_global_solver_init_sets_granularity_to_model():
    model = _make_model()
    solver = _ConcreteSolver(model=model, optimizer_fn=lambda _: None, max_iters=50)
    assert solver.granularity == "model"


def test_global_solver_init_stores_args_and_model():
    model = _make_model(["attn-linear"])
    solver = _ConcreteSolver(model=model, optimizer_fn=lambda _: None)
    assert solver.args is model
    assert solver.quant_target == ["attn-linear"]


def test_global_solver_init_uses_default_max_iters():
    model = _make_model()
    solver = _ConcreteSolver(model=model, optimizer_fn=lambda _: None)
    assert solver.max_iters == 100
    assert solver.current_iter == 0


def test_global_solver_init_without_lr_scheduler():
    model = _make_model()
    solver = _ConcreteSolver(model=model, optimizer_fn=lambda _: None)
    assert solver.lr_scheduler is None


def test_global_solver_init_with_block_size():
    model = _make_model()
    solver = _ConcreteSolver(model=model, optimizer_fn=lambda _: None, block_size=64)
    assert solver.args is model
