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
"""预算诚实性测试：embedding 主导的 toy LM 上，预算低于不可剪下限时须中文告警并置 budget_unreachable，预算可达时无告警。"""

import warnings

import pytest
import torch
from torch import nn

from amct_pytorch.experimental.pruning.accuracy_based_auto_prune import (
    accuracy_based_auto_prune_with_budget,
    size_budget_prune,
)


#: toy LM 参数核算：embed 4000*8=32000（不可剪）>> FFN fc1(8*32+32)+fc2(32*8+8)=552（可剪）
EMBED_PARAMS = 32000
FFN_PARAMS = 552
TOTAL_PARAMS = EMBED_PARAMS + FFN_PARAMS


class EmbeddingHeavyLM(nn.Module):
    """embedding 占主导的 toy LM：embed 32000 参数 >> FFN(fc1+fc2) 552 参数。"""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(4000, 8)
        self.fc1 = nn.Linear(8, 32)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(32, 8)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(self.embed(x))))


class EmbeddingOnlyLM(nn.Module):
    """无任何可剪目标的极端情况：仅 embedding。"""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(1000, 8)

    def forward(self, x):
        return self.embed(x)


def make_batches(n=2):
    # 校准/评估批：LongTensor token id
    torch.manual_seed(0)
    return [torch.randint(0, 4000, (2, 6)) for _ in range(n)]


def budget_warnings(records):
    # 只挑预算诚实性告警（UserWarning，含"预算不可达"），排除试探期 RuntimeWarning
    return [
        w
        for w in records
        if issubclass(w.category, UserWarning) and "预算不可达" in str(w.message)
    ]


def test_unreachable_budget_warns_and_flags_no_exception():
    # 预算 50% << 不可剪 embedding 质量 → 中文告警 + 标志，不抛异常
    model = EmbeddingHeavyLM()
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        res = size_budget_prune(
            model, data=make_batches(), target_keep_ratio=0.5, apply=False
        )
    assert budget_warnings(records), "预算不可达时必须发出中文告警"
    assert res.budget_unreachable is True
    assert res.chosen_ratio is None and res.applied is False
    # prunable_fraction 被记录：fc1+fc2 / 总参数
    assert res.prunable_fraction == pytest.approx(FFN_PARAMS / TOTAL_PARAMS, rel=1e-6)


def test_reachable_budget_no_warning_no_flag():
    # 预算 99.9%（削 33 参数即可）→ 可达：无告警、无标志、选出剪枝率
    model = EmbeddingHeavyLM()
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        res = size_budget_prune(
            model, data=make_batches(), target_keep_ratio=0.999, apply=False
        )
    assert not budget_warnings(records), "可达预算不应发预算告警"
    assert res.budget_unreachable is False
    assert res.chosen_ratio is not None
    assert res.prunable_fraction == pytest.approx(FFN_PARAMS / TOTAL_PARAMS, rel=1e-6)


def test_prunable_scope_makes_same_keep_ratio_reachable():
    # 同一 keep_ratio=0.6：'total' 口径不可达，'prunable' 只对可剪质量计 → 可达
    model = EmbeddingHeavyLM()
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        res = size_budget_prune(
            model,
            data=make_batches(),
            target_keep_ratio=0.6,
            budget_scope="prunable",
            apply=False,
        )
    assert not budget_warnings(records)
    assert res.budget_unreachable is False
    assert res.chosen_ratio is not None
    # 目标 = 不可剪 32000 + round(552*0.6)，应剪到 ≤ 目标
    assert res.params_after <= EMBED_PARAMS + round(FFN_PARAMS * 0.6)


def test_default_scope_total_semantics_unchanged():
    # 默认 budget_scope='total' 与显式 'total' 完全同语义（不改变现行行为）
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
    with pytest.raises(ValueError, match="budget_scope"):
        accuracy_based_auto_prune_with_budget(
            EmbeddingHeavyLM(),
            data=make_batches(),
            total_tolerance=0.1,
            budget_scope="bogus",
            apply=False,
        )


def test_joint_budget_records_prunable_fraction():
    # 联合预算入口：prunable_fraction 总被记录；有可剪质量时无"预算不可达"告警
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        jb = accuracy_based_auto_prune_with_budget(
            EmbeddingHeavyLM(),
            data=make_batches(),
            total_tolerance=1.0,
            budget_scope="prunable",
            apply=False,
        )
    assert not budget_warnings(records)
    assert jb.prune_result.budget_unreachable is False
    assert jb.prune_result.prunable_fraction == pytest.approx(
        FFN_PARAMS / TOTAL_PARAMS, rel=1e-6
    )


def test_joint_budget_zero_prunable_mass_warns_and_flags():
    # 无任何可剪目标 → 剪枝预算永远花不出去：中文告警 + 标志，不抛异常
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        jb = accuracy_based_auto_prune_with_budget(
            EmbeddingOnlyLM(),
            data=[torch.randint(0, 1000, (2, 6))],
            total_tolerance=0.1,
            apply=False,
        )
    assert budget_warnings(records), "零可剪质量时必须发出中文告警"
    assert jb.prune_result.budget_unreachable is True
    assert jb.prune_result.prunable_fraction == 0.0
