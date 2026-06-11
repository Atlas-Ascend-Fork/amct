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
"""Auto-generated coverage tests for the amct_pytorch.pruning package.

Exercises edge/branch paths that the main behavioural suites do not reach:
summaries, helper validation, warn/raise policies, device inference, budget
distribution residuals, menu fallback. Everything is tiny, CPU-only and
deterministic (no HF downloads, no NPU, no real finetuning).
"""

import os
import sys
import unittest
import warnings
from unittest import mock

import torch
import torch.nn.functional as F
from torch import nn

# The local test helpers and the amct_pytorch.pruning package live alongside this
# test file; make them importable before the local imports below.
sys.path.insert(0, os.path.dirname(__file__))

from mini_models import create_mini_mlp
from amct_pytorch.pruning import (
    prune,
    accuracy_based_auto_prune,
    accuracy_based_auto_prune_with_budget,
    size_budget_prune,
    guarded_variance_hybrid_prune,
    PrunePass,
)
from amct_pytorch.pruning import allocation as alloc_mod
from amct_pytorch.pruning import variance_hybrid as vh
from amct_pytorch.pruning.accuracy_based_auto_prune import (
    AutoTuneResult,
    JointBudget,
    _config_to_dict,
    _with_ratio,
    _prunable_param_count,
    _make_fidelity_quality,
    _make_evaluator_quality,
    _model_logits,
)
from amct_pytorch.pruning.allocation import (
    layer_prefixes,
    measure_layer_sensitivity,
    water_fill_ratios,
    _redistribute_residual,
    _isolation_skips,
)
from amct_pytorch.pruning.utils import (
    RunningVariance,
    topk_keep_indices,
    infer_model_device,
    move_batch_to_device,
    prune_conv2d_in_channels,
    expand_flatten_channel_indices,
)
from amct_pytorch.pruning.config import MethodSpec, PruneConfig
from amct_pytorch.pruning.variance_hybrid import (
    HybridResult,
    _select_chosen,
    _prune_with,
)

torch.set_num_threads(2)

# NB: the package __init__ re-exports the function ``accuracy_based_auto_prune``,
# which *shadows* the submodule attribute of the same name on the package. To patch
# module-level names (e.g. AutoPruner) we must fetch the real module from sys.modules.
ap_module = sys.modules["amct_pytorch.pruning.accuracy_based_auto_prune"]


def _calib(n=4):
    torch.manual_seed(0)
    return [torch.randint(0, 1000, (4, 20)) for _ in range(n)]


DENSE_CFG = {
    "methods": {"dense": {"name": "low_variance"}},
    "missing_data_policy": "warn_skip",
}


def _params(m):
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------------------
# utils.py edge cases
# ---------------------------------------------------------------------------
class TestUtilsEdges(unittest.TestCase):

    def test_running_variance_empty_update_is_noop(self):
        rv = RunningVariance()
        rv.update(torch.empty(0, 5))  # numel()==0 -> early return (line 40)
        self.assertEqual(rv.count, 0)
        self.assertIsNone(rv.sum)

    def test_running_variance_no_samples_raises(self):
        rv = RunningVariance()
        with self.assertRaises(RuntimeError):  # line 53
            rv.variance()

    def test_running_variance_accumulates(self):
        rv = RunningVariance()
        rv.update(torch.randn(3, 4))
        rv.update(torch.randn(2, 4))
        var = rv.variance()
        self.assertEqual(var.shape, (4,))
        self.assertTrue((var >= 0).all())

    def test_topk_keep_indices_requires_1d(self):
        with self.assertRaises(ValueError):  # line 71
            topk_keep_indices(torch.randn(2, 3), 0.5, 1)

    def test_topk_keep_indices_clamps_min_keep(self):
        keep = topk_keep_indices(torch.randn(8), prune_ratio=0.99, min_keep=1)
        self.assertEqual(len(keep), 1)
        self.assertEqual(keep, sorted(keep))

    def test_conv2d_in_channels_grouped_raises(self):
        grouped = nn.Conv2d(4, 4, kernel_size=3, groups=2)
        with self.assertRaises(ValueError):  # line 155
            prune_conv2d_in_channels(grouped, [0, 1])

    def test_expand_flatten_indivisible_raises(self):
        with self.assertRaises(ValueError):  # line 306
            expand_flatten_channel_indices([0, 1], old_channels=3, linear_in_features=10)

    def test_expand_flatten_ok(self):
        out = expand_flatten_channel_indices([0, 2], old_channels=4, linear_in_features=8)
        # span = 8//4 = 2; idx 0 -> {0,1}; idx 2 -> {4,5}
        self.assertEqual(out, [0, 1, 4, 5])

    def test_infer_device_on_empty_module(self):
        empty = nn.Module()  # no parameters -> StopIteration -> cpu (lines 320-321)
        self.assertEqual(infer_model_device(empty), torch.device("cpu"))

    def test_infer_device_on_real_model(self):
        model, _ = create_mini_mlp()
        self.assertEqual(infer_model_device(model).type, "cpu")

    def test_move_batch_to_device_nested(self):
        # exercise tensor / tuple / list / dict / passthrough branches (lines 332-338)
        dev = torch.device("cpu")
        t = torch.randn(2)
        args, kwargs = move_batch_to_device(
            (t, (t, t), [t, t], "scalar"),
            {"d": {"x": t}, "n": 7},
            dev,
        )
        self.assertTrue(torch.is_tensor(args[0]))
        self.assertIsInstance(args[1], tuple)
        self.assertIsInstance(args[2], list)
        self.assertEqual(args[3], "scalar")
        self.assertTrue(torch.is_tensor(kwargs["d"]["x"]))
        self.assertEqual(kwargs["n"], 7)


# ---------------------------------------------------------------------------
# pruner.py policy / skip / warn branches
# ---------------------------------------------------------------------------
class TestPrunerPolicies(unittest.TestCase):

    def test_missing_data_raise_policy(self):
        # low_variance requires data; missing_data_policy="raise" -> ValueError (lines 101-102)
        model, _ = create_mini_mlp()
        cfg = {
            "methods": {"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}},
            "missing_data_policy": "raise",
        }
        with self.assertRaises(ValueError):
            prune(model, cfg, data=None)

    def test_missing_data_warn_skip_policy(self):
        # warn_skip -> _warn (lines 65-66, 103-104), no exception, model untouched
        model, _ = create_mini_mlp()
        p0 = _params(model)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            prune(model, DENSE_CFG, data=None)
        self.assertTrue(any("data was not provided" in str(w.message) for w in caught))
        self.assertEqual(_params(model), p0)

    def test_skip_layers_filters_target(self):
        # skip_layers matching a target path -> _is_skipped returns True (line 56);
        # everything skipped -> no params change.
        model, _ = create_mini_mlp()
        p0 = _params(model)
        cfg = {
            "methods": {"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}},
            "skip_layers": ["dense1", "dense2"],
            "missing_data_policy": "warn_skip",
        }
        prune(model, cfg, data=_calib())
        self.assertEqual(_params(model), p0)

    def test_stage_error_warn_skip(self):
        # force the method.apply to raise; stage_error_policy="warn_skip" -> _warn + return (line 111)
        model, _ = create_mini_mlp()
        cfg = {
            "methods": {"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}},
            "missing_data_policy": "warn_skip",
            "stage_error_policy": "warn_skip",
        }
        boom = mock.Mock(side_effect=RuntimeError("boom"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with mock.patch(
                "amct_pytorch.pruning.methods.dense_low_variance.LowVarianceDensePruningMethod.apply",
                boom,
            ):
                prune(model, cfg, data=_calib())
        self.assertTrue(any("after error" in str(w.message) for w in caught))

    def test_stage_error_raise_propagates(self):
        # stage_error_policy="raise" -> re-raise (line 115)
        model, _ = create_mini_mlp()
        cfg = {
            "methods": {"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}},
            "missing_data_policy": "warn_skip",
            "stage_error_policy": "raise",
        }
        boom = mock.Mock(side_effect=RuntimeError("boom"))
        with mock.patch(
            "amct_pytorch.pruning.methods.dense_low_variance.LowVarianceDensePruningMethod.apply",
            boom,
        ):
            with self.assertRaises(RuntimeError):
                prune(model, cfg, data=_calib())

    def test_prune_pass_runs_and_type_checks(self):
        model, _ = create_mini_mlp()
        p0 = _params(model)
        ph = PrunePass(DENSE_CFG, data=_calib())
        ph.run(model)
        self.assertIsNotNone(ph.last_report)
        self.assertLess(_params(model), p0)
        with self.assertRaises(TypeError):
            ph.run("not a module")


# ---------------------------------------------------------------------------
# accuracy_based_auto_prune.py: summaries, helpers, error trials, dict batches
# ---------------------------------------------------------------------------
class TestAutoPruneHelpers(unittest.TestCase):

    def test_summary_no_ratio(self):
        # chosen_ratio None branch (lines 107-114)
        res = AutoTuneResult(
            chosen_ratio=None, tolerance=0.02, baseline_quality=1.0,
            pruned_quality=None, quality_drop=None, params_before=100,
            params_after=100, weight_reduction=0.0, applied=False, report=None,
        )
        self.assertIn("找不到", res.summary())

    def test_summary_with_ratio(self):
        res = AutoTuneResult(
            chosen_ratio=0.3, tolerance=0.1, baseline_quality=1.0,
            pruned_quality=0.95, quality_drop=0.05, params_before=200,
            params_after=140, weight_reduction=0.3, applied=True, report=None,
        )
        self.assertIn("已应用", res.summary())
        res.applied = False
        self.assertIn("建议", res.summary())

    def test_config_to_dict_variants(self):
        # None -> 3-domain default (lines 126-134)
        d = _config_to_dict(None)
        self.assertEqual(set(d["methods"]), {"cnn", "dense", "moe"})
        # PruneConfig instance branch (lines 135-137)
        d2 = _config_to_dict(PruneConfig(methods={"dense": {"name": "low_variance"}}))
        self.assertIn("methods", d2)
        # Mapping branch
        d3 = _config_to_dict({"methods": {"dense": "low_variance"}})
        self.assertEqual(d3["methods"]["dense"], "low_variance")
        # bad type (line 140)
        with self.assertRaises(TypeError):
            _config_to_dict(12345)

    def test_with_ratio_methodspec_and_str(self):
        # MethodSpec branch (lines 148-149)
        cfg_ms = {"methods": {"dense": MethodSpec("low_variance", {})}}
        out = _with_ratio(cfg_ms, 0.4)
        self.assertEqual(out["methods"]["dense"].kwargs["prune_ratio"], 0.4)
        # str shorthand branch (lines 154-156)
        cfg_str = {"methods": {"dense": "low_variance"}}
        out2 = _with_ratio(cfg_str, 0.25)
        self.assertEqual(out2["methods"]["dense"]["kwargs"]["prune_ratio"], 0.25)

    def test_tolerance_negative_raises(self):
        model, _ = create_mini_mlp()
        with self.assertRaises(ValueError):  # line 693
            accuracy_based_auto_prune(model, DENSE_CFG, data=_calib(), tolerance=-0.1)

    def test_trial_error_is_marked_unacceptable(self):
        # make AutoPruner blow up inside the trial -> warn + return -inf trial (lines 455-464)
        model, _ = create_mini_mlp()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with mock.patch.object(
                ap_module, "AutoPruner", side_effect=RuntimeError("trial boom")
            ):
                res = accuracy_based_auto_prune(
                    model, DENSE_CFG, data=_calib(), tolerance=0.5,
                    ratio_grid=(0.2, 0.4), apply=False,
                )
        self.assertIsNone(res.chosen_ratio)
        self.assertTrue(any("试探发生错误" in str(w.message) for w in caught))

    def test_model_logits_dict_batch(self):
        # Mapping batch branch (lines 284-286) of _model_logits via fidelity quality
        model, _ = create_mini_mlp()
        model.eval()
        batches = [{"input_ids": torch.randint(0, 1000, (2, 10))}]
        qf = _make_fidelity_quality(model, batches, None)
        self.assertAlmostEqual(qf(model), 1.0, places=6)

    def test_model_logits_tuple_and_unextractable(self):
        model, _ = create_mini_mlp()
        ids = torch.randint(0, 1000, (2, 10))
        # tuple/list branch (lines 287-288): wrap input in a list
        out = _model_logits(model, [ids], None)
        self.assertTrue(torch.is_tensor(out))

        class WeirdOut(nn.Module):
            def forward(self, x):
                return {"not_logits": x}  # dict, no .logits, not tensor/tuple -> raises (lines 296-298)

        with self.assertRaises(TypeError):
            _model_logits(WeirdOut(), ids, None)

    def test_fidelity_requires_eval_data(self):
        model, _ = create_mini_mlp()
        with self.assertRaises(ValueError):  # line 311
            _make_fidelity_quality(model, [], None)

    def test_evaluator_quality_paths(self):
        # evaluator without .evaluate -> TypeError (lines 365-366 region)
        with self.assertRaises(TypeError):
            _make_evaluator_quality(object(), 1)

        class Eval:
            @staticmethod
            def evaluate(m, iters=1):
                return 0.5

        qf = _make_evaluator_quality(Eval(), 2)
        model, _ = create_mini_mlp()
        self.assertEqual(qf(model), 0.5)

        class EvalNone:
            @staticmethod
            def evaluate(m):
                return None

        qf2 = _make_evaluator_quality(EvalNone(), 1)
        with self.assertRaises(TypeError):
            qf2(model)

    def test_auto_prune_uses_evaluator(self):
        # drive _resolve_quality evaluator branch end-to-end
        class Eval:
            @staticmethod
            def evaluate(m, iters=1):
                return _params(m) / 1.0

        model, _ = create_mini_mlp()
        p0 = _params(model)

        class EvalNorm:
            @staticmethod
            def evaluate(m, iters=1):
                return _params(m) / p0

        res = accuracy_based_auto_prune(
            model, DENSE_CFG, data=_calib(), tolerance=0.2,
            evaluator=EvalNorm(), ratio_grid=(0.1, 0.3), apply=False,
        )
        self.assertLessEqual(res.weight_reduction, 0.2 + 1e-9)

    def test_apply_chosen_finetune_called(self):
        # exercise _apply_chosen_ratio finetune branch (lines 534-536) + apply path
        model, _ = create_mini_mlp()
        p0 = _params(model)
        calls = {"n": 0}

        def fin(m):
            calls["n"] += 1

        def quality(m):
            return _params(m) / p0

        res = accuracy_based_auto_prune(
            model, DENSE_CFG, data=_calib(), tolerance=0.3,
            quality_fn=quality, ratio_grid=(0.1, 0.2), finetune_fn=fin, apply=True,
        )
        self.assertTrue(res.applied)
        self.assertGreaterEqual(calls["n"], 1)

    def test_prunable_param_count_warns_on_domain_failure(self):
        # force a domain.find_targets to raise -> warn branch (lines 208-215)
        model, _ = create_mini_mlp()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with mock.patch(
                "amct_pytorch.pruning.domains.dense.DensePruningDomain.find_targets",
                side_effect=RuntimeError("detect boom"),
            ):
                n = _prunable_param_count(model, _config_to_dict(DENSE_CFG))
        self.assertGreaterEqual(n, 0)
        self.assertTrue(any("find_targets" in str(w.message) for w in caught))


class TestJointBudgetExtra(unittest.TestCase):

    def test_joint_budget_summary(self):
        jb = JointBudget(
            total_tolerance=0.2, prune_tolerance=0.1, prune_drop=0.05,
            remaining_tolerance=0.15, prune_result=mock.Mock(),
        )
        self.assertIn("joint-budget", jb.summary())

    def test_joint_budget_validation(self):
        model, _ = create_mini_mlp()
        with self.assertRaises(ValueError):
            accuracy_based_auto_prune_with_budget(
                model, DENSE_CFG, total_tolerance=-1.0, data=_calib()
            )
        with self.assertRaises(ValueError):
            accuracy_based_auto_prune_with_budget(
                model, DENSE_CFG, total_tolerance=0.1, prune_fraction=2.0, data=_calib()
            )

    def test_joint_budget_runs(self):
        model, _ = create_mini_mlp()
        p0 = _params(model)
        jb = accuracy_based_auto_prune_with_budget(
            model, DENSE_CFG, total_tolerance=0.2, prune_fraction=0.5,
            quality_fn=lambda m: _params(m) / p0, data=_calib(),
        )
        self.assertIsInstance(jb, JointBudget)
        self.assertGreaterEqual(jb.remaining_tolerance, 0.0)


# ---------------------------------------------------------------------------
# allocation.py: water-fill residual, isolation skips, whole-model fallback,
# guard revert paths.
# ---------------------------------------------------------------------------
class _FFNBlock(nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.dense1 = nn.Linear(hidden, inter)
        self.dense2 = nn.Linear(inter, hidden)

    def forward(self, x):
        return self.dense2(F.gelu(self.dense1(x)))


class _PlainMLP(nn.Module):
    """No numbered layers -> keys<2 triggers whole-model uniform fallback."""

    def __init__(self, hidden=32, inter=48):
        super().__init__()
        self.embeddings = nn.Embedding(1000, hidden)
        self.ffn = _FFNBlock(hidden, inter)
        self.classifier = nn.Linear(hidden, 1000)

    def forward(self, input_ids):
        return self.classifier(self.ffn(self.embeddings(input_ids)))


SENS_CFG = {
    "methods": {"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}},
    "allocation": {"strategy": "sensitivity"},
    "missing_data_policy": "warn_skip",
}


class TestAllocationEdges(unittest.TestCase):

    def test_water_fill_redistributes_residual(self):
        # extreme spread + clip forces residual redistribution (lines 195 region)
        ratios = water_fill_ratios([0.0, 1.0], 0.85, floor=0.05, ceil=0.9)
        self.assertAlmostEqual(sum(ratios), 0.85 * 2, places=6)

    def test_water_fill_empty_raises(self):
        with self.assertRaises(ValueError):
            water_fill_ratios([], 0.5)

    def test_water_fill_infeasible_raises(self):
        with self.assertRaises(ValueError):  # line 215
            water_fill_ratios([0.0, 1.0], 0.95, floor=0.05, ceil=0.9)

    def test_redistribute_residual_converges_when_all_clamped(self):
        # all at ceil, positive residual, no free index -> early return (line 195)
        ratios = [0.9, 0.9]
        _redistribute_residual(ratios, total=2.0, floor=0.05, ceil=0.9)
        self.assertEqual(ratios, [0.9, 0.9])

    def test_isolation_skips_excludes_other_layers(self):
        keys = ["model.layers.0", "model.layers.1", "model.layers.10"]
        skips = _isolation_skips(keys, "model.layers.1", ["userskip"],
                                 ["model.head.dense1", "model.layers.1.dense1"])
        self.assertIn("userskip", skips)
        # other layer prefixes carry a trailing dot so layers.1 != layers.10
        self.assertIn("model.layers.0.", skips)
        self.assertIn("model.layers.10.", skips)
        # out-of-keep target path retained; in-keep target dropped
        self.assertIn("model.head.dense1", skips)
        self.assertNotIn("model.layers.1.dense1", skips)

    def test_layer_prefixes_empty_for_plain_model(self):
        self.assertEqual(layer_prefixes(_PlainMLP()), [])

    def test_whole_model_uniform_fallback(self):
        # keys<2 -> _uniform_whole_model warns + actually prunes (lines 276-288, 330-333 area)
        torch.manual_seed(0)
        model = _PlainMLP().eval()
        inter = model.ffn.dense1.out_features
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            prune(model, SENS_CFG, data=_calib(4))
        self.assertTrue(any("uniform" in str(w.message) for w in caught))
        self.assertLess(model.ffn.dense1.out_features, inter)

    def test_measure_layer_sensitivity_no_data_raises(self):
        model, _ = create_mini_mlp()
        with self.assertRaises(ValueError):  # line 154
            measure_layer_sensitivity(model, [], ref_ratio=0.5)

    def test_measure_layer_sensitivity_no_layers_raises(self):
        # plain model has no layer prefixes -> ValueError (lines 156-160)
        with self.assertRaises(ValueError):
            measure_layer_sensitivity(_PlainMLP(), _calib(2), ref_ratio=0.5)

    def test_sensitivity_probe_failure_marks_max(self):
        # force per-layer prune to raise inside the probe -> max sensitivity + warn (lines 173-180)
        model, _ = create_mini_mlp()
        model.eval()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with mock.patch.object(
                alloc_mod, "_prune_single_layer",
                side_effect=RuntimeError("probe boom"),
            ):
                sens = measure_layer_sensitivity(model, _calib(2), ref_ratio=0.5)
        self.assertTrue(all(v == 1.0 for v in sens.values()))
        self.assertTrue(any("maximally sensitive" in str(w.message) for w in caught))

    def test_guard_calib_nll_unavailable_reverts_uniform(self):
        # calib_nll raising (TypeError) -> revert to uniform (lines 243-251)
        torch.manual_seed(0)
        model, _ = create_mini_mlp()
        model.eval()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with mock.patch.object(
                alloc_mod, "calib_nll", side_effect=TypeError("no nll")
            ):
                prune(model, SENS_CFG, data=_calib(4))
        self.assertTrue(
            any("calib NLL guard unavailable" in str(w.message) for w in caught)
        )

    def test_guard_none_skips_nll(self):
        # allocation guard="none" -> _apply_layer_ratios branch (lines 330-333)
        torch.manual_seed(0)
        model, _ = create_mini_mlp()
        model.eval()
        p0 = _params(model)
        cfg = {
            "methods": {"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}},
            "allocation": {"strategy": "sensitivity", "guard": "none"},
            "missing_data_policy": "warn_skip",
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prune(model, cfg, data=_calib(4))
        from amct_pytorch.pruning import get_last_report
        rep = get_last_report()
        self.assertEqual(rep.allocation_choice, "sensitivity")
        self.assertLess(rep.params_after, p0)

    def test_allocation_requires_data(self):
        model, _ = create_mini_mlp()
        with self.assertRaises(ValueError):  # lines 303-306
            prune(model, SENS_CFG, data=None)


# ---------------------------------------------------------------------------
# variance_hybrid.py: summary, select_chosen, menu validation, fallback
# ---------------------------------------------------------------------------
class TestVarianceHybridEdges(unittest.TestCase):

    def test_hybrid_summary(self):
        res = HybridResult(
            chosen_criterion="mass", fallback_criterion="mass",
            criterion_quality={"mass": 0.5, "peak": 0.4},
            chosen_quality=0.5, fallback_quality=0.5, gain_over_fallback=0.0,
            prune_ratio=0.5, params_before=100, params_after=70,
            weight_reduction=0.3, applied=True, report=None,
        )
        s = res.summary()
        self.assertIn("variance-hybrid", s)
        res.applied = False
        self.assertIn("建议", res.summary())

    def test_select_chosen_keeps_fallback_when_not_better(self):
        # no criterion strictly beats fallback -> stays fallback, gain 0 (lines 239-241)
        chosen, fb_q, gain = _select_chosen({"mass": 0.5, "peak": 0.5}, "mass")
        self.assertEqual(chosen, "mass")
        self.assertEqual(gain, 0.0)

    def test_select_chosen_switches_when_better(self):
        chosen, fb_q, gain = _select_chosen({"mass": 0.3, "peak": 0.7}, "mass")
        self.assertEqual(chosen, "peak")
        self.assertAlmostEqual(gain, 0.4, places=6)

    def test_invalid_prune_ratio_raises(self):
        with self.assertRaises(ValueError):  # line 335 region (prune_ratio guard 331-332)
            guarded_variance_hybrid_prune(self._moe(), data=self._tokens(), prune_ratio=1.0)

    def test_empty_menu_raises(self):
        with self.assertRaises(ValueError):  # lines 334-335
            guarded_variance_hybrid_prune(
                self._moe(), data=self._tokens(), prune_ratio=0.5, menu=[]
            )

    def test_degenerate_menu_picks_fallback(self):
        # single-entry menu -> _finalize fallback path; gain 0
        res = guarded_variance_hybrid_prune(
            self._moe(), data=self._tokens(), prune_ratio=0.5, top_k=4,
            menu=[("mass", {"name": "activation_count", "kwargs": {}})],
            apply=False,
        )
        self.assertEqual(res.chosen_criterion, "mass")
        self.assertEqual(res.gain_over_fallback, 0.0)

    def test_trial_failure_records_neg_inf(self):
        # force a criterion to fail in the trial loop -> -inf + warn (lines 169-177)
        orig = _prune_with
        calls = {"n": 0}

        def flaky(model, spec, *a, **k):
            calls["n"] += 1
            if calls["n"] == 2:  # fail the second criterion only
                raise RuntimeError("criterion boom")
            return orig(model, spec, *a, **k)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with mock.patch.object(vh, "_prune_with", side_effect=flaky):
                res = guarded_variance_hybrid_prune(
                    self._moe(), data=self._tokens(), prune_ratio=0.5, top_k=4,
                    menu=[
                        ("mass", {"name": "activation_count", "kwargs": {}}),
                        ("cond_var", {"name": "mass_variance",
                                      "kwargs": {"boundary": -1, "variance_score": "cond"}}),
                    ],
                    apply=False,
                )
        self.assertTrue(any("试探失败" in str(w.message) for w in caught))
        self.assertEqual(res.criterion_quality["cond_var"], float("-inf"))

    def _moe(self, seed=0):
        from mini_models import MiniMoEConfig, MiniMoEModel

        torch.manual_seed(seed)
        cfg = MiniMoEConfig(
            hidden_size=32, intermediate_size=48, num_experts=8,
            num_experts_per_tok=4, num_hidden_layers=2, vocab_size=64,
        )
        return MiniMoEModel(cfg).eval()

    def _tokens(self, seed=3, n=3):
        torch.manual_seed(seed)
        return [torch.randint(0, 64, (2, 16)) for _ in range(n)]


# ---------------------------------------------------------------------------
# Remaining reachable branches: size-budget path, model-logits variants,
# evaluator signature fallback, layer-ratio skip, single-layer uniform fallback.
# ---------------------------------------------------------------------------
class _OneLayerMLP(nn.Module):
    """Exactly ONE numbered layer -> keys==1 (<2): whole-model uniform fallback
    with a non-empty key list (drives the report.add loop)."""

    def __init__(self, hidden=32, inter=48):
        super().__init__()
        self.embeddings = nn.Embedding(1000, hidden)
        self.layers = nn.ModuleList([_FFNBlock(hidden, inter)])
        self.classifier = nn.Linear(hidden, 1000)

    def forward(self, input_ids):
        h = self.embeddings(input_ids)
        for layer in self.layers:
            h = layer(h)
        return self.classifier(h)


class _TupleOutModel(nn.Module):
    """Returns a (logits,) tuple so _model_logits hits the tuple/list out[0] path."""

    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(1000, 16)
        self.classifier = nn.Linear(16, 1000)

    def forward(self, input_ids):
        return (self.classifier(self.embeddings(input_ids)),)


class TestRemainingBranches(unittest.TestCase):

    def test_target_module_paths_skips_non_path_fields(self):
        # _target_module_paths: a non-"_path" field is skipped (line 180); empty *_path skipped too.
        from dataclasses import dataclass as _dc
        from amct_pytorch.pruning.accuracy_based_auto_prune import _target_module_paths

        @_dc
        class _Tgt:
            producer_path: str
            consumer_path: str
            score: float  # non-path field -> continue (line 180)

        paths = _target_module_paths(_Tgt("a.b", "", 1.0))
        self.assertEqual(paths, ["a.b"])  # empty consumer_path dropped; score skipped

    def test_size_budget_resolves_target_and_prunes(self):
        # real size_budget_prune run: drives _target_module_paths (180),
        # _resolve_size_target keep-ratio branch (242, 247-248) + _prunable_param_count.
        model, _ = create_mini_mlp()
        model.eval()
        p0 = _params(model)
        res = size_budget_prune(
            model, DENSE_CFG, data=_calib(), target_keep_ratio=0.85, apply=True
        )
        self.assertIsNotNone(res.chosen_ratio)
        self.assertLessEqual(_params(model), int(round(p0 * 0.85)))

    def test_size_budget_prunable_scope(self):
        model, _ = create_mini_mlp()
        model.eval()
        res = size_budget_prune(
            model, DENSE_CFG, data=_calib(), target_keep_ratio=0.85,
            budget_scope="prunable", apply=False,
        )
        self.assertIsInstance(res, AutoTuneResult)

    def test_size_budget_requires_a_target(self):
        # neither target_keep_ratio nor target_params -> ValueError (line 238)
        model, _ = create_mini_mlp()
        with self.assertRaises(ValueError):
            size_budget_prune(model, DENSE_CFG, data=_calib(), apply=False)

    def test_size_budget_invalid_keep_ratio(self):
        model, _ = create_mini_mlp()
        with self.assertRaises(ValueError):  # target_keep_ratio out of (0,1]
            size_budget_prune(
                model, DENSE_CFG, data=_calib(), target_keep_ratio=1.5, apply=False
            )

    def test_size_budget_with_target_params(self):
        model, _ = create_mini_mlp()
        p0 = _params(model)
        res = size_budget_prune(
            model, DENSE_CFG, data=_calib(), target_params=int(p0 * 0.9), apply=False
        )
        self.assertIsInstance(res, AutoTuneResult)

    def test_size_budget_target_params_prunable_scope(self):
        # target_params + budget_scope='prunable' -> non_prunable + int(target_params) (line 248)
        model, _ = create_mini_mlp()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = size_budget_prune(
                model, DENSE_CFG, data=_calib(), target_params=200,
                budget_scope="prunable", apply=False,
            )
        self.assertIsInstance(res, AutoTuneResult)

    def test_is_skipped_direct(self):
        # pruner._is_skipped inner return True (line 56) + empty skip list false path
        from amct_pytorch.pruning.pruner import _is_skipped
        from amct_pytorch.pruning.config import PruneConfig as _PC
        from amct_pytorch.pruning.domains.dense import DensePruningDomain

        model, _ = create_mini_mlp()
        cfg = _PC(methods={"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}})
        targets = DensePruningDomain().find_targets(model, cfg)
        self.assertTrue(_is_skipped(targets[0], ["dense1"]))   # matches -> line 56
        self.assertFalse(_is_skipped(targets[0], ["nomatch"]))
        self.assertFalse(_is_skipped(targets[0], []))

    def test_evaluator_unintrospectable_signature(self):
        # signature() raises -> except (TypeError, ValueError): accepts_iterations=True (lines 365-366)
        model, _ = create_mini_mlp()

        class Eval2:
            @staticmethod
            def evaluate(m, *a):
                return 0.9

        # patch inspect.signature to raise during _make_evaluator_quality's introspection,
        # forcing the unparseable-signature fallback (accepts_iterations=True).
        with mock.patch(
            "inspect.signature", side_effect=TypeError("unparseable")
        ):
            qf = _make_evaluator_quality(Eval2(), 2)
        self.assertEqual(qf(model), 0.9)

    def test_empty_ratio_grid_raises(self):
        # _prepare_search_inputs: no valid ratio in [0,1) -> ValueError (line 406)
        model, _ = create_mini_mlp()
        with self.assertRaises(ValueError):
            accuracy_based_auto_prune(
                model, DENSE_CFG, data=_calib(), tolerance=0.1, ratio_grid=(1.0, 1.5)
            )

    def test_model_logits_tuple_output(self):
        # tuple/list output branch -> out[0] (lines 296-297)
        model = _TupleOutModel().eval()
        out = _model_logits(model, torch.randint(0, 1000, (2, 8)), None)
        self.assertTrue(torch.is_tensor(out))

    def test_model_logits_batch_adapter(self):
        # batch_adapter branch (line 284)
        model, _ = create_mini_mlp()
        model.eval()
        ids = torch.randint(0, 1000, (2, 8))

        def adapter(batch):
            return (batch,), {}

        out = _model_logits(model, ids, adapter)
        self.assertTrue(torch.is_tensor(out))

    def test_evaluator_unparseable_signature(self):
        # signature inspection fails -> accepts_iterations=True fallback (lines 365-366)
        class WeirdEval:
            # a builtin-like callable whose signature can't be introspected
            evaluate = staticmethod(len)

        # len(model) raises TypeError when called, but signature parse path is what we target;
        # use a wrapper whose .evaluate is an un-introspectable builtin partial.
        import functools

        class Eval2:
            def __init__(self):
                self.evaluate = functools.partial(self._impl)

            @staticmethod
            def _impl(m, *a):
                return 0.7

        qf = _make_evaluator_quality(Eval2(), 1)
        model, _ = create_mini_mlp()
        self.assertEqual(qf(model), 0.7)

    def test_one_layer_uniform_fallback_records_keys(self):
        # exactly one numbered layer -> keys==1 (<2) -> _uniform_whole_model with
        # a non-empty key loop (allocation lines 283-284 / report.add).
        torch.manual_seed(0)
        model = _OneLayerMLP().eval()
        inter = model.layers[0].dense1.out_features
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            prune(model, SENS_CFG, data=_calib(4))
        self.assertTrue(any("uniform" in str(w.message) for w in caught))
        from amct_pytorch.pruning import get_last_report

        rep = get_last_report()
        self.assertEqual(rep.allocation_choice, "uniform")
        self.assertLess(model.layers[0].dense1.out_features, inter)

    def test_apply_layer_ratios_skips_zero(self):
        # _apply_layer_ratios: ratio<=0 entries are skipped (line 135)
        from amct_pytorch.pruning.allocation import (
            _apply_layer_ratios,
            _base_config_dict,
        )

        torch.manual_seed(0)
        from mini_models import create_mini_mlp as _cm

        model, _ = _cm()
        model.eval()
        keys = layer_prefixes(model)
        cfg = PruneConfig(methods={"dense": {"name": "low_variance"}})
        # first layer ratio 0 (skipped), rest pruned
        ratios = [0.0] + [0.5] * (len(keys) - 1)
        out = _apply_layer_ratios(model, _base_config_dict(cfg), keys, ratios, _calib(2), None)
        self.assertIs(out, model)

    def test_measure_sensitivity_with_skip_layers(self):
        # skip_layers extension branch (line 163)
        model, _ = create_mini_mlp()
        model.eval()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sens = measure_layer_sensitivity(
                model, _calib(2), ref_ratio=0.5, skip_layers=["classifier"]
            )
        self.assertEqual(sorted(sens), sorted(layer_prefixes(model)))

    def test_prunable_target_paths_warns_on_domain_failure(self):
        # allocation._prunable_target_paths domain-failure warn (lines 93-100, 100=continue).
        # Call the helper directly so the raising mock only affects accounting, not the
        # real prune pass (which would re-raise under stage_error_policy='raise').
        from amct_pytorch.pruning.allocation import _prunable_target_paths, _base_config_dict

        model, _ = create_mini_mlp()
        model.eval()
        base_cfg = _base_config_dict(PruneConfig(methods={"dense": {"name": "low_variance"}}))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with mock.patch(
                "amct_pytorch.pruning.domains.cnn.CNNPruningDomain.find_targets",
                side_effect=RuntimeError("cnn boom"),
            ):
                paths = _prunable_target_paths(model, base_cfg)
        # dense/moe still detected despite the CNN failure
        self.assertIsInstance(paths, list)
        self.assertTrue(
            any("isolation may be incomplete" in str(w.message) for w in caught)
        )

    def test_finalize_missing_fallback_raises(self):
        # variance_hybrid._finalize: fallback absent from evaluation -> RuntimeError (line 199)
        model = self._moe()
        # drop the fallback ("mass") from the trial result so _finalize can't find it
        def trimmed_menu(*args, **kwargs):
            return ({"cond_var": 0.5}, {"cond_var": _params(model)})

        with mock.patch.object(vh, "_trial_menu", side_effect=trimmed_menu):
            with self.assertRaises(RuntimeError):
                guarded_variance_hybrid_prune(
                    model, data=self._tokens(), prune_ratio=0.5, top_k=4, apply=False
                )

    def _moe(self, seed=0):
        from mini_models import MiniMoEConfig, MiniMoEModel

        torch.manual_seed(seed)
        cfg = MiniMoEConfig(
            hidden_size=32, intermediate_size=48, num_experts=8,
            num_experts_per_tok=4, num_hidden_layers=2, vocab_size=64,
        )
        return MiniMoEModel(cfg).eval()

    def _tokens(self, seed=3, n=3):
        torch.manual_seed(seed)
        return [torch.randint(0, 64, (2, 16)) for _ in range(n)]


if __name__ == "__main__":
    unittest.main(verbosity=2)
