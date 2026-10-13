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

"""Public dry-run diagnosis must preserve the caller's model state."""

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


class EmbeddingFFN(nn.Module):
    def __init__(self, max_norm):
        super().__init__()
        self.embedding = nn.Embedding(16, 4, max_norm=max_norm)
        self.fc1 = nn.Linear(4, 8)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(8, 16)

    def forward(self, inputs):
        hidden = self.embedding(inputs)
        return self.fc2(self.activation(self.fc1(hidden)))


class CountingFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(8, 3)
        self.register_buffer("forward_calls", torch.zeros((), dtype=torch.int64))

    def forward(self, inputs):
        self.forward_calls.add_(1)
        return self.fc2(self.activation(self.fc1(inputs)))


def assert_model_unchanged(model, before, modes):
    for name, tensor in model.state_dict().items():
        expected = before.get(name)
        assert expected is not None
        torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
    assert [(name, module.training) for name, module in model.named_modules()] == modes


@pytest.mark.parametrize("max_norm", [1.0, None])
def test_diagnosis_keeps_embedding_weights(pruning_api, max_norm):
    diagnose, config_class = pruning_api
    model = EmbeddingFFN(max_norm).eval()
    batches = [torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])]
    config = config_class(
        methods={"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}},
        min_neurons=1,
    )
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    modes = [(name, module.training) for name, module in model.named_modules()]
    assert model.embedding.weight[:9].norm(dim=1).max().item() > 1.0

    report = diagnose(model, data=batches, config=config)

    assert report.prune_works is True
    assert report.prune_forward_ok is True
    assert report.search_works is True
    assert report.search_chosen_ratio == 0.8
    assert report.notes == []
    assert_model_unchanged(model, before, modes)


def test_diagnosis_keeps_persistent_buffers_and_mixed_modes(pruning_api):
    diagnose, config_class = pruning_api
    model = CountingFFN().train()
    model.fc1.eval()
    config = config_class(
        methods={"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}},
        min_neurons=1,
    )
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    modes = [(name, module.training) for name, module in model.named_modules()]

    report = diagnose(model, data=[torch.randn(16, 4)], config=config)

    assert report.prune_works is True
    assert report.prune_forward_ok is True
    assert report.search_works is True
    assert report.notes == []
    assert_model_unchanged(model, before, modes)
