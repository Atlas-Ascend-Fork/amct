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
from collections import namedtuple
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from amct_pytorch.pruning.config import PruneConfig
from amct_pytorch.pruning.diagnostics import _forward_ok, prune_diagnose


class DiagnosticDenseModel(nn.Module):
    def __init__(self, output_kind, nonfinite=False):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(8, 3)
        self.output_kind = output_kind
        self.nonfinite = nonfinite

    def forward(self, inputs):
        logits = self.fc2(self.activation(self.fc1(inputs)))
        if self.nonfinite:
            logits = logits * torch.nan
        if self.output_kind == "tuple":
            return logits, "auxiliary output"
        if self.output_kind == "list":
            return [logits, "auxiliary output"]
        if self.output_kind == "object":
            return SimpleNamespace(logits=logits)
        if self.output_kind == "empty_tuple":
            return ()
        return logits


@pytest.mark.parametrize("kind", ["tensor", "object", "tuple", "list"])
def test_forward_diagnostic_accepts_supported_model_output(kind):
    model = DiagnosticDenseModel(kind).eval()
    assert _forward_ok(model, torch.ones(3, 4), None)


@pytest.mark.parametrize("kind", ["tensor", "object", "tuple", "list"])
def test_forward_diagnostic_keeps_nonfinite_logits_rejection(kind):
    model = DiagnosticDenseModel(kind, nonfinite=True).eval()
    assert not _forward_ok(model, torch.ones(3, 4), None)


def test_forward_diagnostic_keeps_invalid_output_rejection():
    assert not _forward_ok(DiagnosticDenseModel("empty_tuple"), torch.ones(3, 4), None)


@pytest.mark.parametrize("kind", ["tuple", "list", "tensor"])
def test_public_diagnose_does_not_report_valid_pruned_output_as_failure(kind):
    torch.manual_seed(7)
    model = DiagnosticDenseModel(kind).eval()
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    config = PruneConfig(
        methods={"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}},
        min_neurons=1,
    )
    report = prune_diagnose(
        model, data=[torch.randn(16, 4)], config=config, prune_ratio=0.5
    )
    assert report.targets["dense"] > 0
    assert report.prune_works and report.prune_reduction > 0
    assert report.prune_forward_ok is True
    assert "forward OK" in report.summary()
    assert list(state) == list(model.state_dict())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)


@pytest.mark.parametrize("aux_finite", [True, False])
@pytest.mark.parametrize("logits_finite", [True, False])
def test_namedtuple_diagnostic_prefers_named_logits(aux_finite, logits_finite):
    output_type = namedtuple("DiagnosticOutput", ["aux", "logits"])

    class NamedOutput(nn.Module):
        def forward(self, inputs):
            aux = inputs if aux_finite else inputs * torch.nan
            logits = inputs if logits_finite else inputs * torch.nan
            return output_type(aux, logits)

    assert _forward_ok(NamedOutput(), torch.ones(3, 4), None) is logits_finite
