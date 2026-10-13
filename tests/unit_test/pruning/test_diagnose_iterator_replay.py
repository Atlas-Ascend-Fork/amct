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
"""Host-only public diagnosis checks for replaying one-shot batch iterators."""

import importlib
from pathlib import Path
import sys
import types

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def pruning_api():
    # Only the hardware-dependent package initializer is bypassed. Pruning,
    # calibration, model forward, parameter edits, and search remain real.
    previous = {
        name: module
        for name, module in sys.modules.items()
        if name == "amct_pytorch" or name.startswith("amct_pytorch.")
    }
    for name in previous:
        sys.modules.pop(name, None)
    package = types.ModuleType("amct_pytorch")
    package.__path__ = [str(ROOT / "amct_pytorch")]
    sys.modules.update({"amct_pytorch": package})
    try:
        api = importlib.import_module("amct_pytorch.pruning")
        config = importlib.import_module("amct_pytorch.pruning.config")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            yield api.prune_diagnose, config.PruneConfig
    finally:
        names = [
            name
            for name in sys.modules
            if name == "amct_pytorch" or name.startswith("amct_pytorch.")
        ]
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(previous)


class TinyFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(8, 3)

    def forward(self, inputs):
        return self.fc2(self.activation(self.fc1(inputs)))


class ReusableBatches:
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)


@pytest.mark.parametrize("kind", ["iterator", "generator"])
def test_diagnose_one_shot_batches_keep_forward_and_search(pruning_api, kind):
    diagnose, config_class = pruning_api
    model = TinyFFN().eval()
    batches = [torch.randn(16, 4), torch.randn(16, 4)]
    config = config_class(
        methods={"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}},
        min_neurons=1,
    )
    state_before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    reference = diagnose(model, data=list(batches), config=config)
    data = iter(batches) if kind == "iterator" else (batch for batch in batches)

    report = diagnose(model, data=data, config=config)

    assert report.targets == {"cnn": 0, "dense": 1, "moe": 0}
    assert report.prune_works is True
    assert report.prune_reduction == pytest.approx(32 / 67)
    assert report.prune_forward_ok is True
    assert report.search_works is True
    assert report.search_chosen_ratio == reference.search_chosen_ratio
    assert report.notes == []
    for name, tensor in model.state_dict().items():
        expected = state_before.get(name)
        assert expected is not None
        torch.testing.assert_close(tensor, expected, rtol=0, atol=0)


def test_diagnose_reusable_batches_keep_existing_behavior(pruning_api):
    diagnose, config_class = pruning_api
    model = TinyFFN().eval()
    batches = ReusableBatches([torch.randn(16, 4), torch.randn(16, 4)])
    config = config_class(
        methods={"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}},
        min_neurons=1,
    )

    report = diagnose(model, data=batches, config=config)

    assert report.prune_forward_ok is True
    assert report.search_works is True
    assert report.prune_reduction == pytest.approx(32 / 67)
    assert report.notes == []
