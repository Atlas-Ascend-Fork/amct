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
"""Size-budget reachability tests: on an embedding-dominated toy LM, a budget below the unprunable
floor must warn and set budget_unreachable, while a reachable budget does not.
"""

import logging

import pytest
import torch
from torch import nn

from amct_pytorch.pruning.accuracy_based_auto_prune import (
    _size_budget_prune as size_budget_prune,
)


EMBED_PARAMS = 32000
FFN_PARAMS = 552
TOTAL_PARAMS = EMBED_PARAMS + FFN_PARAMS


class EmbeddingHeavyLM(nn.Module):
    """Embedding-dominated toy LM: embed 32000 params >> FFN(fc1+fc2) 552 params."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(4000, 8)
        self.fc1 = nn.Linear(8, 32)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(32, 8)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(self.embed(x))))


class EmbeddingOnlyLM(nn.Module):
    """Extreme case with no prunable targets: embedding only."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(1000, 8)

    def forward(self, x):
        return self.embed(x)


def make_batches(n=2):
    torch.manual_seed(0)
    return [torch.randint(0, 4000, (2, 6)) for _ in range(n)]


def budget_warnings(records):
    return [
        r
        for r in records
        if r.levelno == logging.WARNING and "unreachable" in r.getMessage()
    ]


def test_unreachable_budget_warns_and_flags_no_exception(caplog):
    model = EmbeddingHeavyLM()
    with caplog.at_level(logging.WARNING, logger="Log"):
        res = size_budget_prune(
            model, data=make_batches(), target_keep_ratio=0.5, apply=False
        )
    assert budget_warnings(caplog.records), "an unreachable budget must emit a warning"
    assert res.budget_unreachable is True
    assert res.chosen_ratio is None and res.applied is False
    assert res.prunable_fraction == pytest.approx(FFN_PARAMS / TOTAL_PARAMS, rel=1e-6)


def test_reachable_budget_no_warning_no_flag(caplog):
    model = EmbeddingHeavyLM()
    with caplog.at_level(logging.WARNING, logger="Log"):
        res = size_budget_prune(
            model, data=make_batches(), target_keep_ratio=0.999, apply=False
        )
    assert not budget_warnings(caplog.records), (
        "a reachable budget must not emit a budget warning"
    )
    assert res.budget_unreachable is False
    assert res.chosen_ratio is not None
    assert res.prunable_fraction == pytest.approx(FFN_PARAMS / TOTAL_PARAMS, rel=1e-6)


def test_prunable_scope_makes_same_keep_ratio_reachable(caplog):
    model = EmbeddingHeavyLM()
    with caplog.at_level(logging.WARNING, logger="Log"):
        res = size_budget_prune(
            model,
            data=make_batches(),
            target_keep_ratio=0.6,
            budget_scope="prunable",
            apply=False,
        )
    assert not budget_warnings(caplog.records)
    assert res.budget_unreachable is False
    assert res.chosen_ratio is not None
    assert res.params_after <= EMBED_PARAMS + round(FFN_PARAMS * 0.6)


def test_default_scope_total_semantics_unchanged():
    res_a = size_budget_prune(
        EmbeddingHeavyLM(), data=make_batches(), target_keep_ratio=0.999, apply=False
    )
    res_b = size_budget_prune(
        EmbeddingHeavyLM(),
        data=make_batches(),
        target_keep_ratio=0.999,
        budget_scope="total",
        apply=False,
    )
    assert res_a.chosen_ratio == res_b.chosen_ratio
    assert res_a.params_after == res_b.params_after


def test_invalid_budget_scope_raises():
    with pytest.raises(ValueError, match="budget_scope"):
        size_budget_prune(
            EmbeddingHeavyLM(),
            data=make_batches(),
            target_keep_ratio=0.9,
            budget_scope="bogus",
            apply=False,
        )
