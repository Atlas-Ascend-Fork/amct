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
import torch.nn as nn

from amct_pytorch.common.optimization.base_solver import BaseSolver


class _ConcreteSolver(BaseSolver):
    def solve(self, calibration_data):
        return calibration_data


def _args(quant_target=None):
    return SimpleNamespace(quant_target=quant_target or ["mlp"])


def test_cannot_instantiate_abstract_base_solver():
    with pytest.raises(TypeError):
        BaseSolver(_args(), 0, nn.Linear(4, 4))


def test_concrete_solver_records_attributes():
    solver = _ConcreteSolver(args=_args(["attn-linear"]), layer_idx=3, model=nn.Linear(4, 4))
    assert solver.layer_idx == 3
    assert solver.quant_target == ["attn-linear"]
    assert solver.optimizer is None and solver.lr_scheduler is None
    assert solver.max_iters == 100
    assert solver.current_iter == 0
    assert solver.granularity == "block"


def test_finalize_uses_export_ptq_params_when_available():
    class _Exportable(nn.Module):
        def export_ptq_params(self):
            return {"k": torch.tensor([1.0])}

    solver = _ConcreteSolver(args=_args(), layer_idx=0, model=_Exportable())
    out = solver.finalize()
    assert torch.equal(out["k"], torch.tensor([1.0]))


def test_finalize_falls_back_to_trainable_params():
    m = nn.Module()
    m.trainable = nn.Parameter(torch.ones(2))
    m.frozen = nn.Parameter(torch.zeros(2), requires_grad=False)
    solver = _ConcreteSolver(args=_args(), layer_idx=0, model=m)
    out = solver.finalize()
    assert set(out) == {"trainable"}
    assert torch.equal(out["trainable"], torch.ones(2))


def test_step_calls_optimizer_and_scheduler_and_increments_iter():
    class _StubOpt:
        def __init__(self):
            self.stepped = 0
            self.zeroed = 0

        def step(self):
            self.stepped += 1

        def zero_grad(self):
            self.zeroed += 1

    class _StubSched:
        def __init__(self):
            self.stepped = 0

        def step(self):
            self.stepped += 1

    opt = _StubOpt()
    sched = _StubSched()
    solver = _ConcreteSolver(args=_args(), layer_idx=0, model=nn.Linear(4, 4))
    solver.optimizer = opt
    solver.lr_scheduler = sched
    solver.step()
    solver.step()
    assert opt.stepped == 2
    assert opt.zeroed == 2
    assert sched.stepped == 2
    assert solver.current_iter == 2


def test_step_no_op_when_optimizer_and_scheduler_unset():
    solver = _ConcreteSolver(args=_args(), layer_idx=0, model=nn.Linear(4, 4))
    solver.step()
    assert solver.current_iter == 1


def test_unimplemented_hooks_default_to_no_op():
    solver = _ConcreteSolver(args=_args(), layer_idx=0, model=nn.Linear(4, 4))
    assert solver.forward_once() is None
    assert solver.origin_forward() is None
    assert solver.quant_forward() is None


def test_base_solver_granularity_class_attribute():
    from amct_pytorch.common.optimization.base_solver import (
        BaseSolver as base_solver_cls,
    )
    assert base_solver_cls.granularity == "block"
