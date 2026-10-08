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
"""Calibration, batch transfer, and temporary mode edge cases."""

from collections import UserDict

import pytest
import torch
from torch import nn

from amct_pytorch.pruning.calib import calib_nll
from amct_pytorch.pruning.context import PruneContext
from amct_pytorch.pruning.finetune import prune_finetune
from amct_pytorch.pruning.prune_op.dense_reconstruct import _collect_consumer_acts
from amct_pytorch.pruning.prune_op.moe_output_merge import _collect_block_inputs
from amct_pytorch.pruning.utils import move_batch_to_device, run_calibration_epoch


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.head = nn.Linear(4, 8)

    def forward(self, ids):
        return self.head(self.embed(ids))


@pytest.mark.parametrize("shape", [(1, 1), (0, 3), (1, 0)])
def test_calibration_skips_batches_without_targets(shape):
    class RejectEmptyTargets(TinyLM):
        def forward(self, ids):
            if ids[:, 1:].numel() == 0:
                raise RuntimeError("empty forward")
            return super().forward(ids)

    model = RejectEmptyTargets()
    valid = torch.tensor([[1, 2, 3]])
    empty = torch.zeros(shape, dtype=torch.long)
    assert calib_nll(model, [empty, valid]) == pytest.approx(calib_nll(model, [valid]))


@pytest.mark.parametrize("case", ["valid", "floating_point"])
def test_calibration_restores_mixed_module_modes(case):
    model = TinyLM()
    model.train()
    model.head.eval()
    before = [module.training for module in model.modules()]
    ids = torch.tensor([[1, 2, 3]])
    if case == "floating_point":
        with pytest.raises(ValueError, match="integer token-id"):
            calib_nll(model, [ids.float()])
    else:
        calib_nll(model, [ids])
    assert [module.training for module in model.modules()] == before


def test_recursive_transfer_accepts_general_mapping():
    batch = UserDict({"ids": torch.ones(2), "nested": UserDict({"states": [torch.zeros(2)]})})
    (moved,), _ = move_batch_to_device((batch,), {}, torch.device("meta"))
    assert moved["ids"].device.type == "meta"
    assert moved["nested"]["states"][0].device.type == "meta"
    assert batch["ids"].device.type == "cpu"


@pytest.mark.parametrize("collector", ["dense", "moe"])
def test_collectors_restore_mixed_module_modes(collector):
    model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.2))
    model.train()
    model[1].eval()
    before = [module.training for module in model.modules()]
    context = PruneContext(data=[torch.randn(2, 4)])
    if collector == "dense":
        _collect_consumer_acts(model, [], context, torch.device("cpu"))
    else:
        _collect_block_inputs(model, [model[0]], context)
    assert [module.training for module in model.modules()] == before


@pytest.mark.parametrize("operation", ["nll", "dense", "moe"])
def test_mode_switch_failure_restores_submodules(operation):
    class FailingModeSwitch(TinyLM):
        fail_mode_switch = False

        def train(self, mode=True):
            super().train(mode)
            if self.fail_mode_switch:
                raise RuntimeError("mode switch failed")
            return self

    model = FailingModeSwitch()
    model.train()
    model.head.eval()
    before = [module.training for module in model.modules()]
    model.fail_mode_switch = True
    ids = torch.tensor([[1, 2, 3]])
    context = PruneContext(data=[ids])
    with pytest.raises(RuntimeError, match="mode switch failed"):
        if operation == "nll":
            calib_nll(model, [ids])
        elif operation == "dense":
            _collect_consumer_acts(model, [], context, torch.device("cpu"))
        else:
            _collect_block_inputs(model, [model.embed], context)
    assert [module.training for module in model.modules()] == before


@pytest.mark.parametrize("operation", ["calibration", "finetune"])
def test_epoch_mode_switch_failure_restores_modes(operation):
    class FailingModeSwitch(TinyLM):
        fail_mode_switch = False

        def train(self, mode=True):
            super().train(mode)
            if self.fail_mode_switch:
                raise RuntimeError("mode switch failed")
            return self

    model = FailingModeSwitch()
    if operation == "finetune":
        model.eval()
    before = [module.training for module in model.modules()]
    model.fail_mode_switch = True
    ids = torch.tensor([[1, 2, 3]])
    with pytest.raises(RuntimeError, match="mode switch failed"):
        if operation == "calibration":
            run_calibration_epoch(model, PruneContext(data=[ids]))
        else:
            prune_finetune(model, [ids], steps=1)
    assert [module.training for module in model.modules()] == before
