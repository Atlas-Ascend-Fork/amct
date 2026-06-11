#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
"""Guarded variance-HYBRID tests: criterion MENU picks best on val, never worse than fallback, params drop,
forward OK; also covers the mass_variance ``variance_score`` kwarg (cond/peak/cvxpeak).
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(__file__))

from mini_models import MiniMoEConfig, MiniMoEModel
from amct_pytorch.pruning import (
    guarded_variance_hybrid_prune,
    prune,
)
from amct_pytorch.pruning.variance_hybrid import DEFAULT_MENU

VOCAB = 64


def _toy_config():
    """Small multi-expert MoE (16 experts x 4 layers, top-4): enough choices for variance criteria."""
    return MiniMoEConfig(
        hidden_size=32,
        intermediate_size=48,
        num_experts=16,
        num_experts_per_tok=4,
        num_hidden_layers=4,
        vocab_size=VOCAB,
    )


def _toy_model(seed=0):
    torch.manual_seed(seed)
    return MiniMoEModel(_toy_config()).eval()


def _tokens(seed, n=6):
    torch.manual_seed(seed)
    return [torch.randint(0, VOCAB, (2, 16)) for _ in range(n)]


def _num_params(model):
    return sum(p.numel() for p in model.parameters())


def _imbalanced_model(seed=0):
    """Imbalanced-routing MoE: gate weights favor low-index experts so mass concentrates on a few,
    making peak/cond variance criteria keep a *different* expert set than mass (deterministic divergence).
    """
    model = _toy_model(seed)
    with torch.no_grad():
        for layer in model.layers:
            w = layer.gate.weight  # [num_experts, hidden]
            w.zero_()
            num_experts, hidden = w.shape
            for e in range(num_experts):
                w[e, e % hidden] = (num_experts - e) * 0.5
    return model


def _expert_fingerprints(model):
    """Per-layer fingerprint set of kept experts (router-independent; used to compare which experts survived prune)."""
    return [
        frozenset(round(float(e.fc1.weight.sum()), 5) for e in layer.experts)
        for layer in model.layers
    ]


def _prune_variance(variance_score, model, data):
    """Prune MoE with a single variance_score (boundary=-1 so every layer uses variance scoring); return model."""
    prune(
        model,
        {
            "methods": {
                "moe": {
                    "name": "mass_variance",
                    "kwargs": {
                        "prune_ratio": 0.5,
                        "top_k": 4,
                        "boundary": -1,
                        "variance_score": variance_score,
                    },
                }
            },
            "missing_data_policy": "warn_skip",
        },
        data=data,
    )
    return model


class TestGuardedVarianceHybrid(unittest.TestCase):
    def test_never_worse_than_fallback_prunes_and_forwards(self):
        model = _toy_model(0)
        before = _num_params(model)
        res = guarded_variance_hybrid_prune(
            model,
            data=_tokens(7),
            prune_ratio=0.5,
            top_k=4,
            eval_data=_tokens(11, n=3),
            apply=True,
        )
        # guard: HYBRID never worse than fallback on val
        self.assertGreaterEqual(res.gain_over_fallback, 0.0)
        self.assertEqual(res.fallback_criterion, "mass")
        self.assertIn(res.chosen_criterion, dict(DEFAULT_MENU))
        # chosen criterion's val quality >= fallback's val quality
        self.assertGreaterEqual(res.chosen_quality, res.fallback_quality)
        # actually pruned: params drop, and all 4 criteria evaluated
        self.assertLess(_num_params(model), before)
        self.assertEqual(set(res.criterion_quality), {n for n, _ in DEFAULT_MENU})
        # forward works after pruning
        out = model(_tokens(11, n=1)[0])
        self.assertEqual(out.shape[-1], VOCAB)

    def test_non_fallback_criterion_can_strictly_win(self):
        # peak criterion's kept-expert set (fingerprints) on the imbalanced model, used as "known-better" reward target
        peak_keep = set()
        for fps in _expert_fingerprints(
            _prune_variance("peak", _imbalanced_model(0), _tokens(7))
        ):
            peak_keep |= set(fps)

        def reward_peak_keep(candidate):
            # higher quality the more peak-selected experts the candidate keeps -- rewards variance criterion's choice
            return float(
                sum(
                    round(float(e.fc1.weight.sum()), 5) in peak_keep
                    for layer in candidate.layers
                    for e in layer.experts
                )
            )

        res = guarded_variance_hybrid_prune(
            _imbalanced_model(0),
            data=_tokens(7),
            prune_ratio=0.5,
            top_k=4,
            quality_fn=reward_peak_keep,
            apply=False,
        )
        # a non-fallback criterion strictly wins so the guard switches (not the tautological >=0)
        self.assertNotEqual(res.chosen_criterion, "mass")
        self.assertGreater(res.gain_over_fallback, 0.0)
        # the switched criterion's val quality is strictly above fallback(mass)
        self.assertGreater(res.chosen_quality, res.criterion_quality["mass"])

    def test_apply_false_leaves_model_unchanged(self):
        model = _toy_model(0)
        before = _num_params(model)
        res = guarded_variance_hybrid_prune(
            model,
            data=_tokens(7),
            prune_ratio=0.5,
            top_k=4,
            eval_data=_tokens(11, n=3),
            apply=False,
        )
        self.assertFalse(res.applied)
        self.assertEqual(_num_params(model), before)  # original model untouched
        self.assertLess(res.params_after, before)  # but reports chosen criterion's pruned size

    def test_degenerate_menu_falls_back_to_mass(self):
        # MENU with only the fallback must pick mass with gain exactly 0
        model = _toy_model(1)
        res = guarded_variance_hybrid_prune(
            model,
            data=_tokens(7),
            prune_ratio=0.5,
            top_k=4,
            eval_data=_tokens(11, n=3),
            menu=[("mass", {"name": "activation_count", "kwargs": {}})],
            apply=True,
        )
        self.assertEqual(res.chosen_criterion, "mass")
        self.assertEqual(res.gain_over_fallback, 0.0)

    def test_invalid_prune_ratio_raises(self):
        model = _toy_model(0)
        with self.assertRaises(ValueError):
            guarded_variance_hybrid_prune(model, data=_tokens(7), prune_ratio=1.0)


class TestVarianceScoreKwarg(unittest.TestCase):
    def test_each_variance_score_prunes(self):
        for vs in ("cond", "peak", "cvxpeak"):
            with self.subTest(variance_score=vs):
                model = _toy_model(0)
                before = _num_params(model)
                prune(model, self._cfg(vs), data=_tokens(7))
                self.assertLess(_num_params(model), before)
                out = model(_tokens(11, n=1)[0])
                self.assertEqual(out.shape[-1], VOCAB)

    def test_variance_scores_keep_different_experts(self):
        cond = _expert_fingerprints(
            _prune_variance("cond", _imbalanced_model(0), _tokens(7))
        )
        peak = _expert_fingerprints(
            _prune_variance("peak", _imbalanced_model(0), _tokens(7))
        )
        cvxpeak = _expert_fingerprints(
            _prune_variance("cvxpeak", _imbalanced_model(0), _tokens(7))
        )
        # same expert count kept per layer (same prune ratio), but which experts differ
        self.assertEqual([len(s) for s in cond], [len(s) for s in peak])
        # peak differs from both cond and cvxpeak (>=2 criteria keep distinct sets)
        self.assertTrue(
            any(peak[i] != cond[i] for i in range(len(peak))),
            "peak and cond kept the same expert set -- variance_score did not change the selection",
        )
        self.assertTrue(
            any(peak[i] != cvxpeak[i] for i in range(len(peak))),
            "peak and cvxpeak kept the same expert set -- variance_score did not change the selection",
        )

    def test_default_is_cond_backward_compatible(self):
        m_default = _toy_model(0)
        prune(
            m_default,
            {
                "methods": {
                    "moe": {
                        "name": "mass_variance",
                        "kwargs": {"prune_ratio": 0.5, "top_k": 4, "boundary": -1},
                    }
                },
                "missing_data_policy": "warn_skip",
            },
            data=_tokens(7),
        )
        m_cond = _toy_model(0)
        prune(m_cond, self._cfg("cond"), data=_tokens(7))
        self.assertEqual(_num_params(m_default), _num_params(m_cond))

    def test_invalid_variance_score_raises(self):
        model = _toy_model(0)
        with self.assertRaises(ValueError):
            prune(model, self._cfg("bogus"), data=_tokens(7))

    def _cfg(self, variance_score):
        return {
            "methods": {
                "moe": {
                    "name": "mass_variance",
                    "kwargs": {
                        "prune_ratio": 0.5,
                        "top_k": 4,
                        "boundary": -1,  # every layer uses variance scoring
                        "variance_score": variance_score,
                    },
                },
            },
            "missing_data_policy": "warn_skip",
        }


if __name__ == "__main__":
    unittest.main()
