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
import math

import pytest
import torch
from torch import nn

from amct_pytorch.pruning import prune
from amct_pytorch.pruning.accuracy_based_auto_prune import _make_ppl_quality


class TokenModel(nn.Module):
    def __init__(self, magnitude):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(1, 2), nn.ReLU(), nn.Linear(2, 2, bias=False)
        )
        with torch.no_grad():
            self.ffn[0].weight.copy_(torch.tensor([[0.0], [1.0]]))
            self.ffn[0].bias.copy_(torch.tensor([1.0, 0.0]))
            self.ffn[2].weight.copy_(torch.tensor([[0.0, 0.0], [magnitude, 0.0]]))

    def forward(self, input_ids):
        return self.ffn(input_ids.float().unsqueeze(-1))


def batches():
    ids = torch.tensor([[0, 1, 0, 1]], dtype=torch.long)
    return ids, [{"input_ids": ids, "labels": torch.zeros_like(ids)}]


@pytest.mark.parametrize("magnitude", [800.0, 1000.0, 10000.0])
def test_large_finite_nll_improvement_has_finite_quality(magnitude):
    ids, evaluation = batches()
    baseline, improved = TokenModel(magnitude), TokenModel(0.0)
    assert torch.isfinite(baseline(ids)).all()
    quality = _make_ppl_quality(baseline, evaluation, None)
    assert quality(baseline) == pytest.approx(1.0)
    score = quality(improved)
    assert math.isfinite(score) and score > 1.0


@pytest.mark.parametrize("evaluator", ["ppl", "causal_lm"])
@pytest.mark.parametrize("copy_trial", [False, True])
def test_public_prune_accepts_an_actual_large_improvement(evaluator, copy_trial):
    ids, evaluation = batches()
    model = TokenModel(1000.0)
    before = sum(parameter.numel() for parameter in model.parameters())
    callback = (lambda candidate: None) if copy_trial else None
    result = prune(
        model,
        {"methods": {"dense": "low_variance"}, "min_neurons": 1},
        data=[ids],
        eval_data=evaluation,
        evaluator=evaluator,
        ratio_grid=[0.5],
        tolerance=0.0,
        finetune_fn=callback,
    )
    assert result is None
    assert model.ffn[0].out_features == model.ffn[2].in_features == 1
    assert sum(parameter.numel() for parameter in model.parameters()) < before
    # The discarded constant neuron supplied the wrong-token logit of 1000.
    assert torch.equal(model(ids), torch.zeros(1, 4, 2))


@pytest.mark.parametrize(
    "baseline_magnitude,candidate_magnitude",
    [(2.0, 0.0), (0.0, 2.0), (2.0, 2.0), (0.0, 1000.0)],
)
def test_representable_quality_and_underflow_keep_original_formula(
    baseline_magnitude, candidate_magnitude
):
    ids, evaluation = batches()
    baseline, candidate = (
        TokenModel(baseline_magnitude),
        TokenModel(candidate_magnitude),
    )

    def nll(model):
        logits = model(ids)[:, :-1].reshape(-1, 2)
        return float(
            nn.functional.cross_entropy(logits, torch.zeros(3, dtype=torch.long))
        )

    expected = math.exp(nll(baseline) - nll(candidate))
    assert _make_ppl_quality(baseline, evaluation, None)(candidate) == pytest.approx(
        expected
    )
