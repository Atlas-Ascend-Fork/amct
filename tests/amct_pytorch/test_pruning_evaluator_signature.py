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
import inspect

import pytest
import torch
from torch import nn

from amct_pytorch.pruning import prune
from amct_pytorch.pruning.accuracy_based_auto_prune import _make_evaluator_quality


class ScoreEvaluator:
    def __init__(self):
        self.calls = []

    def score(self, model):
        with torch.no_grad():
            output = model(torch.ones(3, 4))
            self.calls.append(tuple(output.shape))
            return float(output.mean())


class ModelOnly(ScoreEvaluator):
    def evaluate(self, model):
        return self.score(model)


class KeywordOption(ScoreEvaluator):
    def evaluate(self, model, *, normalize=False):
        assert normalize is False
        return self.score(model)


class KeywordIterations(ScoreEvaluator):
    def evaluate(self, model, *, iterations=1):
        assert iterations == 3
        return self.score(model)


class KeywordDictionary(ScoreEvaluator):
    def evaluate(self, model, **options):
        assert options == {}
        return self.score(model)


class PositionalIterations(ScoreEvaluator):
    def evaluate(self, model, iterations):
        assert iterations == 3
        return self.score(model)


class VariadicIterations(ScoreEvaluator):
    def evaluate(self, model, *iterations):
        assert iterations == (3,)
        return self.score(model)


KINDS = [
    ModelOnly,
    KeywordOption,
    KeywordIterations,
    KeywordDictionary,
    PositionalIterations,
    VariadicIterations,
]


@pytest.mark.parametrize("kind", KINDS)
def test_evaluator_signature_uses_one_callback_with_actual_model(kind):
    model = nn.Linear(4, 2)
    evaluator = kind()
    result = _make_evaluator_quality(evaluator, 3)(model)
    assert result == pytest.approx(float(model(torch.ones(3, 4)).detach().mean()))
    assert evaluator.calls == [(3, 2)]


@pytest.mark.parametrize("kind", KINDS)
def test_public_prune_uses_supported_evaluator(kind):
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)).eval()
    evaluator = kind()
    before = sum(parameter.numel() for parameter in model.parameters())
    result = prune(
        model,
        {"methods": {"dense": "low_variance"}, "min_neurons": 1},
        data=[torch.randn(16, 4)],
        tolerance=100.0,
        evaluator=evaluator,
        eval_iterations=3,
        ratio_grid=[0.5],
    )
    assert result is None
    assert len(evaluator.calls) >= 2
    assert sum(parameter.numel() for parameter in model.parameters()) < before
    assert model(torch.ones(3, 4)).shape == (3, 2)


def test_evaluator_internal_typeerror_is_not_retried():
    class Broken:
        calls = 0

        def evaluate(self, model, *, iterations=1):
            self.calls += 1
            raise TypeError("actual evaluator failure")

    evaluator = Broken()
    with pytest.raises(TypeError, match="actual evaluator failure"):
        _make_evaluator_quality(evaluator, 3)(nn.Linear(4, 2))
    assert evaluator.calls == 1


def test_callable_evaluator_is_preserved():
    def evaluate(model):
        return float(model(torch.ones(3, 4)).mean())

    assert _make_evaluator_quality(evaluate, 3) is evaluate


def test_none_evaluator_result_remains_rejected():
    class NoneEvaluator:
        def evaluate(self, model):
            return None

    with pytest.raises(TypeError, match="returned None"):
        _make_evaluator_quality(NoneEvaluator(), 3)(nn.Linear(4, 2))


def test_uninspectable_builtin_evaluator_keeps_positional_fallback():
    class HashCountModel(nn.Identity):
        def __init__(self):
            super().__init__()
            self.hash_calls = 0

        def __hash__(self):
            self.hash_calls += 1
            return object.__hash__(self)

    model = HashCountModel()
    scores = {"existing unrelated score": 0.5}

    class BuiltinEvaluator:
        evaluate = scores.pop

    evaluator = BuiltinEvaluator()
    with pytest.raises(ValueError):
        inspect.signature(evaluator.evaluate)
    # Real dict.pop(model, iterations) uses iterations as the default value.
    # Omitting iterations raises KeyError; repeated calls increment hash_calls.
    assert _make_evaluator_quality(evaluator, 3)(model) == 3.0
    assert model.hash_calls == 1
    assert scores == {"existing unrelated score": 0.5}
