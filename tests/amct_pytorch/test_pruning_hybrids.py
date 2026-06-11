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
"""Config-driven menu mode of accuracy_based_auto_prune (folds in the former guarded
HYBRIDs): the criterion / recovery MENU picks the best variant on a validation set,
guarded to be never worse than the item-0 fallback; the chosen variant is exposed via
the returned AutoTuneResult + get_last_report(). Also covers the mass_variance
``variance_score`` kwarg (cond/peak/cvxpeak) and the reconstruct ``recovery`` kwarg.
"""

import copy
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(__file__))

from mini_models import MiniMoEConfig, MiniMoEModel
from mini_models import (
    MiniMLPConfig,
    MiniMLPModel,
    MiniCNNConfig,
    MiniCNNModel,
)
from amct_pytorch.pruning import (
    accuracy_based_auto_prune,
    prune,
    MOE_VARIANCE_MENU_CFG,
    DENSE_RECOVERY_MENU_CFG,
)
from amct_pytorch.pruning.presets import _MOE_CRITERION_MENU

VOCAB = 64


def _moe_menu_cfg(prune_ratio=0.5, top_k=4, menu=None):
    """A MoE variance-criterion menu cfg at the given fixed prune_ratio/top_k."""
    cfg = copy.deepcopy(MOE_VARIANCE_MENU_CFG)
    cfg["methods"]["moe"]["kwargs"] = {"prune_ratio": prune_ratio, "top_k": top_k}
    if menu is not None:
        cfg["methods"]["moe"]["menu"] = menu
    return cfg


def _dense_recovery_cfg(prune_ratio=0.5, menu=None):
    cfg = copy.deepcopy(DENSE_RECOVERY_MENU_CFG)
    cfg["methods"]["dense"]["kwargs"] = {"prune_ratio": prune_ratio, "ridge": 1e-2}
    if menu is not None:
        cfg["methods"]["dense"]["menu"] = menu
    return cfg


def _cnn_recovery_cfg(prune_ratio=0.3, menu=None):
    base_menu = menu if menu is not None else (
        ("none", {"name": "reconstruct", "kwargs": {"recovery": "none"}}),
        ("bias", {"name": "reconstruct", "kwargs": {"recovery": "bias"}}),
        ("ls", {"name": "reconstruct", "kwargs": {"recovery": "ls"}}),
    )
    return {
        "methods": {
            "cnn": {
                "name": "reconstruct",
                "kwargs": {"prune_ratio": prune_ratio, "ridge": 1e-2},
                "menu": base_menu,
            },
        },
        "missing_data_policy": "warn_skip",
    }


def _chosen_variant(res, menu_names=None):
    """Read the winning menu variant name.

    When applied, it is recorded in the report (a menu-select StageEvent); when
    apply=False there is no report, so fall back to the accepted trial (trials are in
    menu order, the winner marked accepted).
    """
    if res.report is not None:
        for ev in res.report.events:
            if ev.stage == "menu-select":
                # detail: "chosen variant 'NAME' (fallback 'FB') at prune_ratio=..."
                return ev.detail.split("chosen variant ", 1)[1].split(" ", 1)[0].strip("'")
    if menu_names is not None:
        for name, t in zip(menu_names, res.trials):
            if t.accepted:
                return name
    return None


def _variant_quality(res, name, menu_names):
    """Per-variant validation quality from res.trials (trials are in menu order)."""
    return res.trials[menu_names.index(name)].quality


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


_MOE_MENU_NAMES = [n for n, _ in _MOE_CRITERION_MENU]


class TestGuardedVarianceHybrid(unittest.TestCase):
    def test_never_worse_than_fallback_prunes_and_forwards(self):
        model = _toy_model(0)
        before = _num_params(model)
        res = accuracy_based_auto_prune(
            model,
            _moe_menu_cfg(prune_ratio=0.5, top_k=4),
            data=_tokens(7),
            eval_data=_tokens(11, n=3),
            apply=True,
        )
        # returns a normal AutoTuneResult at the fixed prune_ratio
        self.assertTrue(res.applied)
        self.assertEqual(res.chosen_ratio, 0.5)
        # menu-mode summary() reports the applied point WITHOUT a bogus "<= tolerance -1" clause
        self.assertIn("prune_ratio=0.50", res.summary())
        self.assertNotIn("tolerance -", res.summary())
        # guard: chosen variant's val quality >= fallback(mass)'s val quality
        self.assertGreaterEqual(
            res.pruned_quality, _variant_quality(res, "mass", _MOE_MENU_NAMES)
        )
        # chosen variant is one of the menu, recorded in the report
        self.assertIn(_chosen_variant(res), _MOE_MENU_NAMES)
        # actually pruned: params drop, weight reduction > 0, and all 4 criteria evaluated
        self.assertLess(_num_params(model), before)
        self.assertGreater(res.weight_reduction, 0.0)
        self.assertEqual(len(res.trials), len(_MOE_CRITERION_MENU))
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

        res = accuracy_based_auto_prune(
            _imbalanced_model(0),
            _moe_menu_cfg(prune_ratio=0.5, top_k=4),
            data=_tokens(7),
            evaluator=reward_peak_keep,
            apply=False,
        )
        # a non-fallback criterion strictly wins so the guard switches (not the tautological >=)
        self.assertNotEqual(_chosen_variant(res, _MOE_MENU_NAMES), "mass")
        # the switched criterion's val quality is strictly above fallback(mass)
        self.assertGreater(
            res.pruned_quality, _variant_quality(res, "mass", _MOE_MENU_NAMES)
        )

    def test_apply_false_leaves_model_unchanged(self):
        model = _toy_model(0)
        before = _num_params(model)
        res = accuracy_based_auto_prune(
            model,
            _moe_menu_cfg(prune_ratio=0.5, top_k=4),
            data=_tokens(7),
            eval_data=_tokens(11, n=3),
            apply=False,
        )
        self.assertFalse(res.applied)
        self.assertEqual(_num_params(model), before)  # original model untouched
        self.assertLess(res.params_after, before)  # but reports chosen criterion's pruned size

    def test_degenerate_menu_falls_back_to_mass(self):
        # MENU with only the fallback must pick mass
        model = _toy_model(1)
        res = accuracy_based_auto_prune(
            model,
            _moe_menu_cfg(
                prune_ratio=0.5, top_k=4,
                menu=(("mass", {"name": "activation_count", "kwargs": {}}),),
            ),
            data=_tokens(7),
            eval_data=_tokens(11, n=3),
            apply=True,
        )
        self.assertEqual(_chosen_variant(res), "mass")
        # only the fallback evaluated; it is the winner
        self.assertEqual(len(res.trials), 1)

    def test_invalid_prune_ratio_raises(self):
        model = _toy_model(0)
        with self.assertRaises(ValueError):
            accuracy_based_auto_prune(
                model, _moe_menu_cfg(prune_ratio=1.0, top_k=4), data=_tokens(7)
            )


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


def _mlp():
    torch.manual_seed(0)
    cfg = MiniMLPConfig(
        hidden_size=64, intermediate_size=256, num_hidden_layers=3, vocab_size=128
    )
    return MiniMLPModel(cfg).eval()


def _mlp_tokens(seed, n=6):
    torch.manual_seed(seed)
    return [torch.randint(0, 128, (2, 16)) for _ in range(n)]


def _cnn():
    torch.manual_seed(0)
    return MiniCNNModel(MiniCNNConfig()).eval()


def _images(seed, n=3, batch_size=8):
    torch.manual_seed(seed)
    return [torch.randn(batch_size, 3, 32, 32) for _ in range(n)]


def _trained_mlp(seed=0, steps=40):
    """Lightly trained mini-MLP: a randomly initialized toy model's output is nearly saturated
    (fidelity ~unchanged after any pruning) and cannot show a recovery quality ladder; only after
    training in nontrivial input-dependent structure does ls reconstruction truly beat none.
    """
    model = _mlp()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    torch.manual_seed(123)
    data = [
        (torch.randint(0, 128, (4, 16)), torch.randint(0, 128, (4, 16)))
        for _ in range(8)
    ]
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(steps):
        for x, y in data:
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out.reshape(-1, out.shape[-1]), y.reshape(-1))
            loss.backward()
            optimizer.step()
    return model.eval()


_RECOVERY_NAMES = ["none", "bias", "ls"]


class TestGuardedRecoveryHybrid(unittest.TestCase):
    def test_dense_never_worse_prunes_and_forwards(self):
        model = _mlp()
        before = _num_params(model)
        res = accuracy_based_auto_prune(
            model,
            _dense_recovery_cfg(prune_ratio=0.5),
            data=_mlp_tokens(7),
            eval_data=_mlp_tokens(11, n=3),
            apply=True,
        )
        # never worse than none(fallback) on val
        self.assertGreaterEqual(
            res.pruned_quality, _variant_quality(res, "none", _RECOVERY_NAMES)
        )
        self.assertEqual(len(res.trials), 3)
        self.assertIn(_chosen_variant(res), _RECOVERY_NAMES)
        self.assertLess(_num_params(model), before)
        self.assertGreater(res.weight_reduction, 0.0)
        out = model(_mlp_tokens(11, n=1)[0])
        self.assertEqual(out.shape[-1], 128)

    def test_dense_recovery_quality_ladder_ls_best(self):
        # apply=False: only compare each MENU recovery's val quality, don't touch the original model
        res = accuracy_based_auto_prune(
            _trained_mlp(0),
            _dense_recovery_cfg(prune_ratio=0.5),
            data=_mlp_tokens(7, n=8),
            eval_data=_mlp_tokens(11, n=4),
            apply=False,
        )
        q_none = _variant_quality(res, "none", _RECOVERY_NAMES)
        q_bias = _variant_quality(res, "bias", _RECOVERY_NAMES)
        q_ls = _variant_quality(res, "ls", _RECOVERY_NAMES)
        # ls is the best recovery: >= none and >= bias (so the guard picks ls over fallback).
        # Honest scope: only assert the partial order impl guarantees (ls>=none, ls>=bias); not bias>=none,
        # since under small calibration bias-folding overfits and can fall below none.
        # keep = round(256*(1-0.5)) = 128; calib rows = n*2*16 = 8*32 = 256 >= 128, so LS is determined.
        self.assertGreaterEqual(q_ls, q_none)
        self.assertGreaterEqual(q_ls, q_bias)
        # ladder really takes effect: ls is chosen and strictly above fallback(none)
        self.assertEqual(_chosen_variant(res, _RECOVERY_NAMES), "ls")
        self.assertGreater(res.pruned_quality, q_none)

    def test_apply_false_leaves_model_unchanged(self):
        model = _mlp()
        before = _num_params(model)
        res = accuracy_based_auto_prune(
            model,
            _dense_recovery_cfg(prune_ratio=0.5),
            data=_mlp_tokens(7),
            eval_data=_mlp_tokens(11, n=3),
            apply=False,
        )
        self.assertFalse(res.applied)
        self.assertEqual(_num_params(model), before)
        self.assertLess(res.params_after, before)

    def test_safety_menu_drops_bias(self):
        # small-calibration safety knob: a menu of only none/ls evaluates just these two
        model = _mlp()
        res = accuracy_based_auto_prune(
            model,
            _dense_recovery_cfg(
                prune_ratio=0.5,
                menu=(
                    ("none", {"name": "reconstruct", "kwargs": {"recovery": "none"}}),
                    ("ls", {"name": "reconstruct", "kwargs": {"recovery": "ls"}}),
                ),
            ),
            data=_mlp_tokens(7),
            eval_data=_mlp_tokens(11, n=3),
        )
        self.assertEqual(len(res.trials), 2)
        self.assertIn(_chosen_variant(res), ("none", "ls"))

    def test_cnn_domain_prunes_and_forwards(self):
        model = _cnn()
        before = _num_params(model)
        res = accuracy_based_auto_prune(
            model,
            _cnn_recovery_cfg(prune_ratio=0.3),
            data=_images(3),
            eval_data=_images(5, n=2),
            apply=True,
        )
        self.assertGreaterEqual(
            res.pruned_quality, _variant_quality(res, "none", _RECOVERY_NAMES)
        )
        self.assertLess(_num_params(model), before)
        self.assertGreater(res.weight_reduction, 0.0)
        out = model(_images(5, n=1)[0])
        self.assertEqual(out.shape[0], 8)

    def test_cnn_apply_false_leaves_model_unchanged(self):
        model = _cnn()
        before = _num_params(model)
        res = accuracy_based_auto_prune(
            model,
            _cnn_recovery_cfg(prune_ratio=0.3),
            data=_images(3),
            eval_data=_images(5, n=2),
            apply=False,
        )
        self.assertFalse(res.applied)
        self.assertEqual(_num_params(model), before)  # original model untouched
        self.assertLess(res.params_after, before)

    def test_invalid_args_raise(self):
        with self.assertRaises(ValueError):
            accuracy_based_auto_prune(
                _mlp(), _dense_recovery_cfg(prune_ratio=1.0), data=_mlp_tokens(7)
            )


class TestRecoveryKwarg(unittest.TestCase):
    def test_dense_each_recovery_prunes(self):
        for rec in ("none", "bias", "ls"):
            with self.subTest(recovery=rec):
                model = _mlp()
                before = _num_params(model)
                prune(model, self._dense_cfg(rec), data=_mlp_tokens(7))
                self.assertLess(_num_params(model), before)
                out = model(_mlp_tokens(11, n=1)[0])
                self.assertEqual(out.shape[-1], 128)

    def test_dense_recovery_keeps_same_neurons(self):
        # recovery only changes weights/biases, not which neurons are kept -> none and ls have equal param counts
        m_none = _mlp()
        prune(m_none, self._dense_cfg("none"), data=_mlp_tokens(7))
        m_ls = _mlp()
        prune(m_ls, self._dense_cfg("ls"), data=_mlp_tokens(7))
        self.assertEqual(_num_params(m_none), _num_params(m_ls))

    def test_dense_default_is_ls_backward_compatible(self):
        m_default = _mlp()
        prune(
            m_default,
            {
                "methods": {
                    "dense": {"name": "reconstruct", "kwargs": {"prune_ratio": 0.5}}
                },
                "missing_data_policy": "warn_skip",
            },
            data=_mlp_tokens(7),
        )
        m_ls = _mlp()
        prune(m_ls, self._dense_cfg("ls"), data=_mlp_tokens(7))
        self.assertEqual(_num_params(m_default), _num_params(m_ls))

    def test_recovery_is_not_a_no_op(self):

        def consumer_w_b(rec):
            m = _mlp()
            prune(m, self._dense_cfg(rec), data=_mlp_tokens(7))
            c = m.layers[0].dense2
            return (
                c.weight.detach().clone(),
                (c.bias.detach().clone() if c.bias is not None else None),
            )

        w_none, b_none = consumer_w_b("none")
        w_ls, _ = consumer_w_b("ls")
        _, b_bias = consumer_w_b("bias")
        self.assertFalse(torch.allclose(w_ls, w_none))  # ls reconstructed the consumer weights
        self.assertIsNotNone(b_bias)
        self.assertFalse(torch.allclose(b_bias, b_none))  # bias folding changed the consumer bias

    def test_dense_ls_underdetermined_warns_and_falls_back(self):
        import warnings as _w

        tiny = _mlp_tokens(7, n=1)  # 1 batch [2,16]=32 rows << keep(=128) -> underdetermined
        model = _mlp()
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            prune(model, self._dense_cfg("ls"), data=tiny)
        msgs = " ".join(str(c.message) for c in caught)
        self.assertIn("recovery=ls not executed", msgs)
        self.assertIn("underdetermined", msgs)
        # falls back to naive slicing: consumer weights must exactly match recovery=none
        # (not overfit-rewritten by underdetermined ls)
        m_none = _mlp()
        prune(m_none, self._dense_cfg("none"), data=tiny)
        self.assertTrue(
            torch.allclose(
                model.layers[0].dense2.weight, m_none.layers[0].dense2.weight
            )
        )

    def test_invalid_recovery_raises(self):
        with self.assertRaises(ValueError):
            prune(_mlp(), self._dense_cfg("bogus"), data=_mlp_tokens(7))

    def test_cnn_each_recovery_prunes(self):
        for rec in ("none", "bias", "ls"):
            with self.subTest(recovery=rec):
                model = _cnn()
                before = _num_params(model)
                prune(
                    model,
                    {
                        "methods": {
                            "cnn": {
                                "name": "reconstruct",
                                "kwargs": {"prune_ratio": 0.3, "recovery": rec},
                            }
                        },
                        "missing_data_policy": "warn_skip",
                    },
                    data=_images(3),
                )
                self.assertLess(_num_params(model), before)
                out = model(_images(5, n=1)[0])
                self.assertEqual(out.shape[0], 8)

    def test_cnn_recovery_is_not_a_no_op(self):
        # batch_size=16, n=6 -> conv1's consumer conv2 reconstruction is determined
        # (measured: conv2 doesn't fall back at this scale)
        data = _images(3, n=6, batch_size=16)
        w_none = self._cnn_recovery("none", data).conv2.weight.detach().clone()
        w_ls = self._cnn_recovery("ls", data).conv2.weight.detach().clone()
        self.assertEqual(w_ls.shape, w_none.shape)  # same prune ratio -> same shape
        # ls really reconstructed conv2 weights (vs none's naive slicing) -> compensation runs, not a silent no-op
        self.assertFalse(torch.allclose(w_ls, w_none))

    def _dense_cfg(self, recovery):
        return {
            "methods": {
                "dense": {
                    "name": "reconstruct",
                    "kwargs": {"prune_ratio": 0.5, "recovery": recovery},
                }
            },
            "missing_data_policy": "warn_skip",
        }

    def _cnn_recovery(self, recovery, data):
        """Prune CNN with a single recovery (sequential targets producer conv1 -> consumer conv2 etc.); return model."""
        model = _cnn()
        prune(
            model,
            {
                "methods": {
                    "cnn": {
                        "name": "reconstruct",
                        "kwargs": {"prune_ratio": 0.3, "recovery": recovery},
                    }
                },
                "missing_data_policy": "warn_skip",
            },
            data=data,
        )
        return model


if __name__ == "__main__":
    unittest.main()
