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

from amct_pytorch.common.optimization.blockwise_solver import BlockwiseSolver

sys.modules["torch_npu"] = MagicMock()


def _args(epochs=1, base_lr=1e-3, optimizer="adam"):
    return SimpleNamespace(
        quant_target=["mlp"],
        device=torch.device("cpu"),
        epochs=epochs,
        base_lr=base_lr,
        optimizer=optimizer,
        weight_decay=0.0,
        momentum=0.9,
        lr_scheduler="none",
    )


# ---- _reconstruction_loss ------------------------------------------------


def test_reconstruction_loss_is_mse():
    solver = BlockwiseSolver(_args(), layer_idx=0, model=nn.Linear(4, 4))
    out = torch.tensor([[1.0, 2.0]])
    tgt = torch.tensor([[2.0, 4.0]])
    expected = torch.nn.functional.mse_loss(out, tgt)
    assert torch.allclose(solver._reconstruction_loss(out, tgt), expected)


# ---- _collect_trainable_param_groups -------------------------------------


class _ModuleWithTrainable(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4))
        self.bias = nn.Parameter(torch.zeros(4))

    def trainable_params(self):
        return [self.weight, self.bias]


def test_collect_trainable_param_groups_marks_grad_and_dedups():
    layer = nn.Module()
    layer.a = _ModuleWithTrainable()
    layer.b = _ModuleWithTrainable()
    # Add a duplicate of layer.a.weight via shared parameter.
    layer.shared_alias = layer.a  # same module reachable twice; dedup must apply.

    solver = BlockwiseSolver(_args(), layer_idx=0, model=layer)
    groups = solver._collect_trainable_param_groups(layer)

    assert len(groups) == 1
    params = groups[0]["params"]
    # 4 unique params: a.weight, a.bias, b.weight, b.bias
    assert len(params) == 4
    assert all(p.requires_grad for p in params)
    assert groups[0]["lr"] == pytest.approx(_args().base_lr * 10)


def test_collect_trainable_param_groups_returns_empty_when_no_trainable():
    layer = nn.Linear(4, 4)
    solver = BlockwiseSolver(_args(), layer_idx=0, model=layer)
    assert not solver._collect_trainable_param_groups(layer)


def test_collect_trainable_param_groups_freezes_other_params_first():
    # nn.Linear has weight + bias with requires_grad=True by default.
    # Without trainable_params hook, _collect should freeze them and return [].
    layer = nn.Linear(4, 4)
    solver = BlockwiseSolver(_args(), layer_idx=0, model=layer)
    solver._collect_trainable_param_groups(layer)
    assert all(not p.requires_grad for p in layer.parameters())


def test_collect_trainable_param_groups_skips_none_and_duplicates():
    class _ModuleWithFlawedTrainable(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(4))
            self.b = nn.Parameter(torch.zeros(4))

        def trainable_params(self):
            return [None, self.w, self.w]

    layer = nn.Module()
    layer.flawed = _ModuleWithFlawedTrainable()
    solver = BlockwiseSolver(_args(), layer_idx=0, model=layer)
    groups = solver._collect_trainable_param_groups(layer)
    assert len(groups) == 1
    assert len(groups[0]["params"]) == 1


# ---- solve / _optimize_block --------------------------------------------


class _TrainableLinear(nn.Linear):
    def trainable_params(self):
        return [self.weight, self.bias]


def _build_loader(in_dim=4, out_dim=4, n=4):
    inps = torch.randn(n, in_dim)
    targets = torch.randn(n, out_dim)
    return DataLoader(TensorDataset(inps, targets), batch_size=2)


def test_solve_runs_one_epoch_and_advances_iter():
    layer = _TrainableLinear(4, 4)
    solver = BlockwiseSolver(_args(epochs=1), layer_idx=0, model=layer)
    solver.solve(_build_loader())
    # 4 samples / batch_size=2 -> 2 batches per epoch.
    assert solver.current_iter == 2
    assert solver.optimizer is not None
    # No scheduler when lr_scheduler="none".
    assert solver.lr_scheduler is None


def test_solve_returns_model_unchanged_when_no_trainable_params():
    # A plain Linear without trainable_params hook: solver should early-return.
    layer = nn.Linear(4, 4)
    solver = BlockwiseSolver(_args(), layer_idx=0, model=layer)
    out = solver.solve(_build_loader())
    assert out is layer
    # Optimizer should not have been built.
    assert solver.optimizer is None


def test_optimize_block_raises_on_uninitialized_optimizer():
    solver = BlockwiseSolver(_args(), layer_idx=0, model=nn.Linear(4, 4))
    with pytest.raises(RuntimeError, match="Optimizer has not been initialized"):
        solver._optimize_block(_build_loader())


def test_optimize_block_raises_on_loader_yielding_single_tensor():
    layer = _TrainableLinear(4, 4)
    solver = BlockwiseSolver(_args(), layer_idx=0, model=layer)
    # Bootstrap the optimizer manually without invoking solve().
    groups = solver._collect_trainable_param_groups(layer)
    from amct_pytorch.common.optimization.factory import build_optimizer
    solver.optimizer = build_optimizer(_args(), groups)

    bad_loader = DataLoader(TensorDataset(torch.randn(2, 4)), batch_size=1)
    with pytest.raises(ValueError, match="Expected PTQ dataloader to yield"):
        solver._optimize_block(bad_loader)


def test_optimize_block_returns_zero_when_loader_empty():
    layer = _TrainableLinear(4, 4)
    solver = BlockwiseSolver(_args(), layer_idx=0, model=layer)
    groups = solver._collect_trainable_param_groups(layer)
    from amct_pytorch.common.optimization.factory import build_optimizer
    solver.optimizer = build_optimizer(_args(), groups)

    empty_loader = DataLoader(
        TensorDataset(torch.empty(0, 4), torch.empty(0, 4)), batch_size=1
    )
    assert solver._optimize_block(empty_loader) == 0.0


def test_optimize_block_handles_tuple_outputs_from_model():
    class _TupleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, x):
            return self.linear(x), "metadata"

        def trainable_params(self):
            return [self.linear.weight, self.linear.bias]

    layer = _TupleModel()
    solver = BlockwiseSolver(_args(), layer_idx=0, model=layer)
    groups = solver._collect_trainable_param_groups(layer)
    from amct_pytorch.common.optimization.factory import build_optimizer
    solver.optimizer = build_optimizer(_args(), groups)

    solver._optimize_block(_build_loader())
    assert solver.current_iter == 2
