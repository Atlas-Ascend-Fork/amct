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
"""Tests for tolerance-driven automatic prune-ratio search (accuracy_based_auto_prune)."""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(__file__))

from mini_models import create_mini_mlp
from amct_pytorch.experimental.pruning import (
    prune,
    accuracy_based_auto_prune,
    accuracy_based_auto_prune_with_budget,
    size_budget_prune,
    get_last_auto_prune,
)

# the result dataclass is not in the package public namespace; import by submodule path.
from amct_pytorch.experimental.pruning.accuracy_based_auto_prune import JointBudget

try:
    import transformers

    _HAS_TF = True
except Exception:  # pragma: no cover
    _HAS_TF = False


def _calib(n=6):
    return [torch.randint(0, 1000, (4, 20)) for _ in range(n)]


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
        # budget met: params <= target (keep 85%)
        self.assertLessEqual(p1, int(round(p0 * 0.85)))
        # no over-pruning: reduction should not far exceed the budget (stop when enough)
        self.assertGreaterEqual(res.weight_reduction, 0.15 - 1e-9)

    def test_unreachable_budget_returns_none(self):
        model, _ = create_mini_mlp()
        model.eval()
        # keeping only 1% is far beyond grid capability -> not found, model untouched
        p0 = sum(p.numel() for p in model.parameters())
        res = size_budget_prune(
            model, DENSE_CFG, data=_calib(), target_keep_ratio=0.01, apply=True
        )
        self.assertIsNone(res.chosen_ratio)
        self.assertEqual(sum(p.numel() for p in model.parameters()), p0)


class TestToleranceSelection(unittest.TestCase):

    def test_higher_tolerance_allows_more_pruning(self):
        calib = _calib()
        chosen = {}
        reductions = {}

        def quality_by_params(p0):
            return lambda m: sum(p.numel() for p in m.parameters()) / p0

        for tol in (0.05, 0.15, 0.40):
            model, _ = create_mini_mlp()
            model.eval()
            p0 = sum(p.numel() for p in model.parameters())
            res = accuracy_based_auto_prune(
                model,
                DENSE_CFG,
                data=calib,
                tolerance=tol,
                quality_fn=quality_by_params(p0),
            )
            chosen[tol] = res.chosen_ratio if res.chosen_ratio is not None else 0.0
            reductions[tol] = res.weight_reduction
            if res.chosen_ratio is not None:
                self.assertLessEqual(res.quality_drop, tol + 1e-9)
            with torch.no_grad():
                out = model(torch.randint(0, 1000, (2, 20)))
            self.assertTrue(torch.isfinite(out).all())
        # monotonic non-decreasing: wider tolerance must not shrink ratio/reduction
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
        # the chosen point (if any) must have zero loss
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
        # search only, original model unchanged
        self.assertEqual(sum(p.numel() for p in model.parameters()), p_before)
        self.assertFalse(res.applied)
        # but a candidate ratio and estimated reduction were indeed found
        self.assertIsNotNone(res.chosen_ratio)
        self.assertGreater(res.weight_reduction, 0.0)

    def test_custom_quality_fn(self):
        calib = _calib()
        model, _ = create_mini_mlp()
        model.eval()
        p0 = sum(p.numel() for p in model.parameters())

        def quality(m):
            # normalize to [0,1]; closer to original param count is "better"
            return sum(p.numel() for p in m.parameters()) / p0

        # allow at most 0.2 loss (params drop to at most 80% of original)
        res = accuracy_based_auto_prune(
            model, DENSE_CFG, data=calib, tolerance=0.2, quality_fn=quality
        )
        self.assertIsNotNone(res.chosen_ratio)
        self.assertLessEqual(res.quality_drop, 0.2 + 1e-9)
        # equivalently: weight reduction <= tolerance
        self.assertLessEqual(res.weight_reduction, 0.2 + 1e-9)


class TestFacadeToleranceMode(unittest.TestCase):

    def test_facade_tolerance_applies_and_records_autotune(self):
        calib = _calib()
        model, _ = create_mini_mlp()
        model.eval()
        p_before = sum(p.numel() for p in model.parameters())

        # deterministic "param count" quality (top-1 fidelity of a random model is noise);
        # cap loss at 30% and verify tolerance mode: model changed + AutoTuneResult recorded + within tolerance.
        def quality(m):
            return sum(p.numel() for p in m.parameters()) / p_before

        prune(model, data=calib, tolerance=0.30, quality_fn=quality)

        res = get_last_auto_prune()
        self.assertIsNotNone(res)
        self.assertTrue(res.applied)
        p_after = sum(p.numel() for p in model.parameters())
        self.assertLess(p_after, p_before)
        self.assertLessEqual(res.quality_drop, 0.30 + 1e-9)
        self.assertLessEqual(res.weight_reduction, 0.30 + 1e-9)
        with torch.no_grad():
            out = model(torch.randint(0, 1000, (2, 20)))
        self.assertTrue(torch.isfinite(out).all())

    def test_facade_fixed_ratio_mode_does_not_set_autotune(self):
        calib = _calib()
        model, _ = create_mini_mlp()
        model.eval()
        prune(model, DENSE_CFG, data=calib)  # no tolerance -> fixed-ratio mode
        self.assertIsNone(get_last_auto_prune())


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

        # looser tolerance should allow ratio/reduction no smaller than tight tolerance
        self.assertGreaterEqual(loose.chosen_ratio or 0.0, tight.chosen_ratio or 0.0)
        self.assertGreaterEqual(loose.weight_reduction, tight.weight_reduction)
        # each chosen point satisfies its own tolerance
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

        # one-shot: no config, only tolerance
        res = accuracy_based_auto_prune(
            model, data=calib, tolerance=0.15, eval_data=eval_data
        )

        # attention projection dim unchanged (protected by safe_skip_attention)
        self.assertEqual(model.model.layers[0].self_attn.q_proj.out_features, q_before)
        # model forwards normally
        with torch.no_grad():
            out = model(eval_data[0])
        self.assertTrue(torch.isfinite(out.logits).all())
        # chosen point satisfies tolerance
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
        # deterministically reproduce "crashes right after pruning, usable after recovery":
        # unrecovered pruned model has quality=0; original (full params) or finetune-recovered model has quality=1
        model, _ = create_mini_mlp()
        model.eval()
        p0 = sum(p.numel() for p in model.parameters())

        def quality(m):
            if getattr(m, "recovered_by_test", False):
                return 1.0
            cur = sum(p.numel() for p in m.parameters())
            return 1.0 if cur >= p0 else 0.0  # any "unrecovered pruning" -> crash to 0

        def recover(m):
            m.recovered_by_test = True  # simulate finetune recovery

        # (A) no finetune_fn: every candidate crashes (0.0), drop=1.0 > tol -> none chosen
        r_a = accuracy_based_auto_prune(
            model,
            DENSE_CFG,
            data=_calib(),
            tolerance=0.1,
            quality_fn=quality,
            ratio_grid=(0.1, 0.2, 0.3),
            apply=False,
        )
        self.assertIsNone(r_a.chosen_ratio)

        # (B) with finetune_fn: candidates "recover" and score high (drop=0) -> largest ratio chosen
        model2, _ = create_mini_mlp()
        model2.eval()
        r_b = accuracy_based_auto_prune(
            model2,
            DENSE_CFG,
            data=_calib(),
            tolerance=0.1,
            quality_fn=quality,
            ratio_grid=(0.1, 0.2, 0.3),
            finetune_fn=recover,
            apply=False,
        )
        self.assertEqual(r_b.chosen_ratio, 0.3)


class TestQuantAwareSearch(unittest.TestCase):

    def test_quant_fn_lowers_measured_quality_and_search_not_more_aggressive(self):
        p0 = sum(p.numel() for p in create_mini_mlp()[0].parameters())

        def quality(m):
            red = 1.0 - sum(p.numel() for p in m.parameters()) / p0
            # quantization amplifies pruning loss: once quant_fn-marked, the same reduction yields lower quality.
            return 1.0 - red * (3.0 if getattr(m, "quantized_by_test", False) else 1.0)

        def quantize_sim(m):
            m.quantized_by_test = True  # simulate applying the target quant scheme (fake-quant)

        grid = (0.1, 0.3, 0.5)
        # (1) mechanism: quant_fn is applied before eval -> quantized quality < float quality (same pruned model)
        from amct_pytorch.experimental.pruning import prune as _prune

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

        # (2) invariant: quant-aware search is never more aggressive than quant-blind search
        # (weight_reduction not bigger)
        m1, _ = create_mini_mlp()
        m1.eval()
        rfp = accuracy_based_auto_prune(
            m1,
            DENSE_CFG,
            data=_calib(),
            tolerance=0.20,
            quality_fn=quality,
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
            quality_fn=quality,
            ratio_grid=grid,
            quant_fn=quantize_sim,
            apply=False,
        )
        self.assertLessEqual(rqa.weight_reduction, rfp.weight_reduction + 1e-9)


class TestJointBudget(unittest.TestCase):

    def test_budget_split_reserves_remaining_for_quant(self):
        model, _ = create_mini_mlp()
        model.eval()
        total = 0.20
        jb = accuracy_based_auto_prune_with_budget(
            model, DENSE_CFG, total_tolerance=total, prune_fraction=0.5, data=_calib()
        )
        self.assertIsInstance(jb, JointBudget)
        # prune upper bound = total budget * fraction
        self.assertAlmostEqual(jb.prune_tolerance, total * 0.5, places=9)
        # remaining budget = total - measured prune loss; non-negative and not over total
        self.assertGreaterEqual(jb.remaining_tolerance, 0.0)
        self.assertLessEqual(jb.remaining_tolerance, total + 1e-9)
        self.assertAlmostEqual(
            jb.remaining_tolerance, max(0.0, total - jb.prune_drop), places=6
        )
        # prune loss should not exceed its allotted upper bound (under monotonicity)
        self.assertLessEqual(jb.prune_drop, jb.prune_tolerance + 1e-9)

    def test_budget_no_prune_keeps_full_budget_for_quant(self):
        model, _ = create_mini_mlp()
        model.eval()
        # tiny budget -> pruning mostly does nothing -> remaining ~= total budget
        jb = accuracy_based_auto_prune_with_budget(
            model, DENSE_CFG, total_tolerance=0.0, prune_fraction=1.0, data=_calib()
        )
        self.assertEqual(jb.prune_drop, 0.0)
        self.assertEqual(jb.remaining_tolerance, 0.0)

    def test_budget_validates_inputs(self):
        model, _ = create_mini_mlp()
        with self.assertRaises(ValueError):
            accuracy_based_auto_prune_with_budget(
                model, DENSE_CFG, total_tolerance=-0.1, data=_calib()
            )
        with self.assertRaises(ValueError):
            accuracy_based_auto_prune_with_budget(
                model, DENSE_CFG, total_tolerance=0.1, prune_fraction=1.5, data=_calib()
            )


if __name__ == "__main__":
    sys.exit(unittest.main(verbosity=2))
