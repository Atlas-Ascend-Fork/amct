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
"""Tests for accuracy_based_auto_prune: tolerance-driven prune-ratio search, size-budget mode,
and config-driven menu mode (the criterion / recovery MENU picks the best variant on a validation
set, falling back to the item-0 default; the chosen variant is exposed via the returned
AutoTuneResult / the report sink). Also covers the mass_variance ``variance_score`` kwarg
(cond/peak/cvxpeak) and the reconstruct ``recovery`` kwarg.
"""

import copy
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(__file__))

from mini_models import (
    create_mini_mlp,
    MiniMoEConfig,
    MiniMoEModel,
    MiniMLPConfig,
    MiniMLPModel,
    MiniCNNConfig,
    MiniCNNModel,
)
from amct_pytorch.pruning import (
    prune,
    MOE_VARIANCE_MENU_CFG,
    DENSE_RECOVERY_MENU_CFG,
)
from amct_pytorch.pruning.presets import _MOE_CRITERION_MENU
from amct_pytorch.pruning.accuracy_based_auto_prune import (
    _accuracy_based_auto_prune as accuracy_based_auto_prune,
    _size_budget_prune as size_budget_prune,
)

try:
    import transformers

    _HAS_TF = True
except Exception:
    _HAS_TF = False


def _calib(n=6):
    return [torch.randint(0, 1000, (4, 20)) for _ in range(n)]


def _output_quality(model, probe):
    """Quality read from the model's outputs: how close a pruned model stays to the original.

    Parameter counts make a tempting stand-in for quality in a test, but they measure the
    model's structure rather than its behaviour, and a search is free to measure a
    candidate without physically resizing anything. Size targets belong to ``size_budget``.
    """
    with torch.no_grad():
        ref = model(probe).detach().clone()
    scale = ref.abs().mean().clamp_min(1e-12)

    def quality(m):
        with torch.no_grad():
            out = m(probe)
        return -float((out - ref).abs().mean() / scale)

    return quality


DENSE_CFG = {
    "methods": {"dense": {"name": "low_variance"}},
    "missing_data_policy": "warn_skip",
}


class TestSizeBudget(unittest.TestCase):
    def test_hits_size_budget(self):
        model, _ = create_mini_mlp()
        model.eval()
        p0 = sum(p.numel() for p in model.parameters())
        res = size_budget_prune(
            model, DENSE_CFG, data=_calib(), target_keep_ratio=0.85, apply=True
        )
        self.assertIsNotNone(res.chosen_ratio)
        p1 = sum(p.numel() for p in model.parameters())
        self.assertLessEqual(p1, int(round(p0 * 0.85)))
        self.assertGreaterEqual(res.weight_reduction, 0.15 - 1e-9)

    def test_unreachable_budget_returns_none(self):
        model, _ = create_mini_mlp()
        model.eval()
        p0 = sum(p.numel() for p in model.parameters())
        res = size_budget_prune(
            model, DENSE_CFG, data=_calib(), target_keep_ratio=0.01, apply=True
        )
        self.assertIsNone(res.chosen_ratio)
        self.assertEqual(sum(p.numel() for p in model.parameters()), p0)

    def test_prune_size_budget_mode(self):
        model, _ = create_mini_mlp()
        model.eval()
        p0 = sum(p.numel() for p in model.parameters())
        prune(model, DENSE_CFG, data=_calib(), size_budget=0.85)
        p1 = sum(p.numel() for p in model.parameters())
        self.assertLessEqual(p1, int(round(p0 * 0.85)))

    def test_tolerance_and_size_budget_mutually_exclusive(self):
        model, _ = create_mini_mlp()
        with self.assertRaises(ValueError):
            prune(model, DENSE_CFG, data=_calib(), tolerance=0.02, size_budget=0.7)


class TestToleranceSelection(unittest.TestCase):
    def test_higher_tolerance_allows_more_pruning(self):
        calib = _calib()
        chosen = {}
        reductions = {}

        probe = calib[0]

        for tol in (0.05, 0.15, 0.40):
            model, _ = create_mini_mlp()
            model.eval()
            res = accuracy_based_auto_prune(
                model,
                DENSE_CFG,
                data=calib,
                tolerance=tol,
                evaluator=_output_quality(model, probe),
            )
            chosen[tol] = res.chosen_ratio if res.chosen_ratio is not None else 0.0
            reductions[tol] = res.weight_reduction
            if res.chosen_ratio is not None:
                self.assertLessEqual(res.quality_drop, tol + 1e-9)
            with torch.no_grad():
                out = model(torch.randint(0, 1000, (2, 20)))
            self.assertTrue(torch.isfinite(out).all())
        self.assertLessEqual(chosen[0.05], chosen[0.15])
        self.assertLessEqual(chosen[0.15], chosen[0.40])
        self.assertLessEqual(reductions[0.05], reductions[0.40])

    def test_zero_tolerance_is_conservative(self):
        calib = _calib()
        model, _ = create_mini_mlp()
        model.eval()
        p_before = sum(p.numel() for p in model.parameters())
        res = accuracy_based_auto_prune(
            model, DENSE_CFG, data=calib, tolerance=0.0, eval_data=_calib(4)
        )
        p_after = sum(p.numel() for p in model.parameters())
        if res.chosen_ratio is not None:
            self.assertEqual(res.quality_drop, 0.0)
        else:
            self.assertFalse(res.applied)
            self.assertEqual(p_after, p_before)

    def test_apply_false_leaves_model_untouched(self):
        calib = _calib()
        model, _ = create_mini_mlp()
        model.eval()
        p_before = sum(p.numel() for p in model.parameters())
        res = accuracy_based_auto_prune(
            model,
            DENSE_CFG,
            data=calib,
            tolerance=0.5,
            eval_data=_calib(4),
            apply=False,
        )
        self.assertEqual(sum(p.numel() for p in model.parameters()), p_before)
        self.assertFalse(res.applied)
        self.assertIsNotNone(res.chosen_ratio)
        self.assertGreater(res.weight_reduction, 0.0)

    def test_custom_quality_fn(self):
        calib = _calib()
        model, _ = create_mini_mlp()
        model.eval()

        res = accuracy_based_auto_prune(
            model,
            DENSE_CFG,
            data=calib,
            tolerance=0.2,
            evaluator=_output_quality(model, calib[0]),
        )
        self.assertIsNotNone(res.chosen_ratio)
        self.assertLessEqual(res.quality_drop, 0.2 + 1e-9)
        self.assertGreater(res.weight_reduction, 0.0)


class TestPruneToleranceMode(unittest.TestCase):
    def test_prune_tolerance_applies_and_records_autotune(self):
        calib = _calib()
        model, _ = create_mini_mlp()
        model.eval()
        p_before = sum(p.numel() for p in model.parameters())

        res = accuracy_based_auto_prune(
            model,
            data=calib,
            tolerance=0.30,
            evaluator=_output_quality(model, calib[0]),
        )

        self.assertIsNotNone(res)
        self.assertTrue(res.applied)
        p_after = sum(p.numel() for p in model.parameters())
        self.assertLess(p_after, p_before)
        self.assertLessEqual(res.quality_drop, 0.30 + 1e-9)
        self.assertGreater(res.weight_reduction, 0.0)
        with torch.no_grad():
            out = model(torch.randint(0, 1000, (2, 20)))
        self.assertTrue(torch.isfinite(out).all())


@unittest.skipUnless(
    _HAS_TF and hasattr(transformers, "Qwen3ForCausalLM"),
    "transformers/Qwen3 unavailable",
)
class TestToleranceRealModel(unittest.TestCase):
    def test_loose_vs_tight_tolerance(self):
        calib = [torch.randint(0, 128, (2, 16)) for _ in range(6)]
        eval_data = [torch.randint(0, 128, (4, 24)) for _ in range(4)]
        cfg = {
            "methods": {"dense": {"name": "low_variance"}},
            "missing_data_policy": "warn_skip",
        }

        tight = accuracy_based_auto_prune(
            self._model(), cfg, data=calib, tolerance=0.05, eval_data=eval_data
        )
        loose = accuracy_based_auto_prune(
            self._model(), cfg, data=calib, tolerance=0.20, eval_data=eval_data
        )

        self.assertGreaterEqual(loose.chosen_ratio or 0.0, tight.chosen_ratio or 0.0)
        self.assertGreaterEqual(loose.weight_reduction, tight.weight_reduction)
        for res in (tight, loose):
            if res.chosen_ratio is not None:
                self.assertLessEqual(res.quality_drop, res.tolerance + 1e-9)

    @unittest.skipUnless(
        _HAS_TF and hasattr(transformers, "LlamaForCausalLM"), "Llama unavailable"
    )
    def test_llama_auto_skips_attention_no_crash(self):
        torch.manual_seed(0)
        model = transformers.LlamaForCausalLM(
            transformers.LlamaConfig(
                vocab_size=128,
                hidden_size=64,
                intermediate_size=256,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=64,
            )
        ).eval()
        q_before = model.model.layers[0].self_attn.q_proj.out_features
        calib = [torch.randint(0, 128, (2, 16)) for _ in range(6)]
        eval_data = [torch.randint(0, 128, (4, 24)) for _ in range(4)]

        res = accuracy_based_auto_prune(
            model, data=calib, tolerance=0.15, eval_data=eval_data
        )

        self.assertEqual(model.model.layers[0].self_attn.q_proj.out_features, q_before)
        with torch.no_grad():
            out = model(eval_data[0])
        self.assertTrue(torch.isfinite(out.logits).all())
        if res.chosen_ratio is not None:
            self.assertLessEqual(res.quality_drop, 0.15 + 1e-9)

    def _model(self):
        torch.manual_seed(0)
        return transformers.Qwen3ForCausalLM(
            transformers.Qwen3Config(
                vocab_size=128,
                hidden_size=64,
                intermediate_size=256,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=64,
            )
        ).eval()


class TestFinetuneInSearch(unittest.TestCase):
    def test_finetune_fn_changes_chosen_ratio(self):
        model, _ = create_mini_mlp()
        model.eval()

        base = _output_quality(model, _calib()[0])

        def quality(m):
            # The callback stands in for a recovery that more than restores the lost
            # accuracy, so with it every candidate clears even a zero tolerance.
            return base(m) + (10.0 if getattr(m, "recovered_by_test", False) else 0.0)

        def recover(m):
            m.recovered_by_test = True

        r_a = accuracy_based_auto_prune(
            model,
            DENSE_CFG,
            data=_calib(),
            tolerance=0.0,
            evaluator=quality,
            ratio_grid=(0.1, 0.2, 0.3),
            apply=False,
        )
        self.assertIsNone(r_a.chosen_ratio)

        model2, _ = create_mini_mlp()
        model2.eval()
        r_b = accuracy_based_auto_prune(
            model2,
            DENSE_CFG,
            data=_calib(),
            tolerance=0.0,
            evaluator=quality,
            ratio_grid=(0.1, 0.2, 0.3),
            finetune_fn=recover,
            apply=False,
        )
        self.assertEqual(r_b.chosen_ratio, 0.3)


class TestQuantAwareSearch(unittest.TestCase):
    def test_quant_fn_lowers_measured_quality_and_search_not_more_aggressive(self):
        reference, _ = create_mini_mlp()
        reference.eval()
        probe = _calib()[0]
        base = _output_quality(reference, probe)

        def quality(m):
            # Quantisation costs accuracy on top of what pruning already cost, so the
            # same cut measures worse once the quantise callback has run.
            return base(m) * (3.0 if getattr(m, "quantized_by_test", False) else 1.0)

        def quantize_sim(m):
            m.quantized_by_test = True

        grid = (0.1, 0.3, 0.5)
        from amct_pytorch.pruning import prune as _prune

        m_fp, _ = create_mini_mlp()
        m_fp.eval()
        _prune(
            m_fp,
            {
                "methods": {
                    "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.3}}
                }
            },
            data=_calib(),
        )
        q_fp = quality(m_fp)
        import copy as _copy

        m_q = _copy.deepcopy(m_fp)
        quantize_sim(m_q)
        self.assertLess(quality(m_q), q_fp)

        m1, _ = create_mini_mlp()
        m1.eval()
        rfp = accuracy_based_auto_prune(
            m1,
            DENSE_CFG,
            data=_calib(),
            tolerance=0.20,
            evaluator=quality,
            ratio_grid=grid,
            apply=False,
        )
        m2, _ = create_mini_mlp()
        m2.eval()
        rqa = accuracy_based_auto_prune(
            m2,
            DENSE_CFG,
            data=_calib(),
            tolerance=0.20,
            evaluator=quality,
            ratio_grid=grid,
            quant_fn=quantize_sim,
            apply=False,
        )
        self.assertLessEqual(rqa.weight_reduction, rfp.weight_reduction + 1e-9)


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
    base_menu = (
        menu
        if menu is not None
        else (
            ("none", {"name": "reconstruct", "kwargs": {"recovery": "none"}}),
            ("bias", {"name": "reconstruct", "kwargs": {"recovery": "bias"}}),
            ("ls", {"name": "reconstruct", "kwargs": {"recovery": "ls"}}),
        )
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
                return (
                    ev.detail.split("chosen variant ", 1)[1].split(" ", 1)[0].strip("'")
                )
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
            w = layer.gate.weight
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


class TestGuardedCriterionMenu(unittest.TestCase):
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
        self.assertTrue(res.applied)
        self.assertEqual(res.chosen_ratio, 0.5)
        self.assertIn("prune_ratio=0.50", res.summary())
        self.assertNotIn("tolerance -", res.summary())
        self.assertGreaterEqual(
            res.pruned_quality, _variant_quality(res, "mass", _MOE_MENU_NAMES)
        )
        self.assertIn(_chosen_variant(res), _MOE_MENU_NAMES)
        self.assertLess(_num_params(model), before)
        self.assertGreater(res.weight_reduction, 0.0)
        self.assertEqual(len(res.trials), len(_MOE_CRITERION_MENU))
        out = model(_tokens(11, n=1)[0])
        self.assertEqual(out.shape[-1], VOCAB)

    def test_non_fallback_criterion_can_strictly_win(self):
        peak_keep = set()
        for fps in _expert_fingerprints(
            _prune_variance("peak", _imbalanced_model(0), _tokens(7))
        ):
            peak_keep |= set(fps)

        def reward_peak_keep(candidate):
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
        self.assertNotEqual(_chosen_variant(res, _MOE_MENU_NAMES), "mass")
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
        self.assertEqual(_num_params(model), before)
        self.assertLess(res.params_after, before)

    def test_degenerate_menu_falls_back_to_mass(self):
        model = _toy_model(1)
        res = accuracy_based_auto_prune(
            model,
            _moe_menu_cfg(
                prune_ratio=0.5,
                top_k=4,
                menu=(("mass", {"name": "activation_count", "kwargs": {}}),),
            ),
            data=_tokens(7),
            eval_data=_tokens(11, n=3),
            apply=True,
        )
        self.assertEqual(_chosen_variant(res), "mass")
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
        self.assertEqual([len(s) for s in cond], [len(s) for s in peak])
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
                        "boundary": -1,
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


class TestGuardedRecoveryMenu(unittest.TestCase):
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
        self.assertGreaterEqual(q_ls, q_none)
        self.assertGreaterEqual(q_ls, q_bias)
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
        self.assertEqual(_num_params(model), before)
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
        self.assertFalse(torch.allclose(w_ls, w_none))
        self.assertIsNotNone(b_bias)
        self.assertFalse(torch.allclose(b_bias, b_none))

    def test_dense_ls_underdetermined_warns_and_falls_back(self):
        import logging as _logging

        tiny = _mlp_tokens(7, n=1)
        model = _mlp()
        records = []

        class _Rec(_logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = _Rec(level=_logging.WARNING)
        logger = _logging.getLogger("Log")
        logger.addHandler(handler)
        try:
            prune(model, self._dense_cfg("ls"), data=tiny)
        finally:
            logger.removeHandler(handler)
        msgs = " ".join(r.getMessage() for r in records)
        self.assertIn("recovery=ls not executed", msgs)
        self.assertIn("underdetermined", msgs)
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
        data = _images(3, n=6, batch_size=16)
        w_none = self._cnn_recovery("none", data).conv2.weight.detach().clone()
        w_ls = self._cnn_recovery("ls", data).conv2.weight.detach().clone()
        self.assertEqual(w_ls.shape, w_none.shape)
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
    sys.exit(unittest.main(verbosity=2))
