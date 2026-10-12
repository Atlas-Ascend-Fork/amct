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
"""A rolled-back stage must not publish its discarded model's sparsity."""

import pytest
import torch.nn as nn

from amct_pytorch.pruning.config import MethodSpec, PruneConfig
from amct_pytorch.pruning.context import PruneContext
from amct_pytorch.pruning.domains.base import BasePruningDomain
from amct_pytorch.pruning.prune_op.base import BasePruningMethod
from amct_pytorch.pruning.pruner import PruningStage
from amct_pytorch.pruning.report import PruneReport


class Domain(BasePruningDomain):
    def find_targets(self, model, config):
        return [model]


class Method(BasePruningMethod):
    def __init__(self, fail):
        self.fail = fail

    def apply(self, model, domain, targets, context, report, config, spec):
        model.fc = nn.Linear(4, 2)
        report.add("dense", "test", "fc", "removed channels")
        report.record_layer_sparsity("fc", 20, 10)
        report.add_warning("intermediate diagnostic")
        if self.fail:
            raise RuntimeError("after partial prune")


@pytest.mark.parametrize("fail", [False, True])
def test_stage_report_matches_the_model_that_survived(fail):
    model = nn.Sequential()
    model.add_module("fc", nn.Linear(4, 4))
    report = PruneReport(per_layer_sparsity={"earlier": 0.25})
    report.add("cnn", "previous", "earlier", "already applied")
    report.add_warning("earlier warning")
    stage = PruningStage("dense", Domain(), Method(fail), MethodSpec("test"))
    result = stage.apply(
        model, PruneContext(), report, PruneConfig(stage_error_policy="warn_skip")
    )
    assert result.fc.out_features == (4 if fail else 2)
    assert report.per_layer_sparsity["earlier"] == 0.25
    if fail:
        assert "fc" not in report.per_layer_sparsity
        assert len(report.events) == 1
        assert "intermediate diagnostic" not in report.warnings
        assert "rolled back" in report.warnings[-1]
    else:
        assert report.per_layer_sparsity["fc"] == 0.5
        assert len(report.events) == 2
        assert "intermediate diagnostic" in report.warnings


def test_raise_policy_still_propagates_stage_errors():
    model = nn.Sequential(nn.Linear(4, 4))
    stage = PruningStage("dense", Domain(), Method(True), MethodSpec("test"))
    with pytest.raises(RuntimeError, match="after partial prune"):
        stage.apply(
            model,
            PruneContext(),
            PruneReport(),
            PruneConfig(stage_error_policy="raise"),
        )
