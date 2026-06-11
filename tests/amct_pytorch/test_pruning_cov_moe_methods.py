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
"""Diff-coverage tests for the MoE pruning *method* modules.

Targets the uncovered branches of mass / mass_variance / output_merge:
- variance_score in {cond, peak, cvxpeak}; boundary in {int, -1, list[int], "auto"};
- missing-data / wrong-domain / prune_ratio<=0 / empty-dataset fallbacks;
- the router-hook internal early-returns and the output-merge drop / bake / NLL edges.

Everything runs on tiny CPU tensors -- no HF downloads, no NPU.
"""

import copy
import os
import sys
import unittest
import warnings
from unittest import mock

import torch

torch.set_num_threads(2)

import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from mini_models import MiniMoEConfig, MiniMoEModel
from amct_pytorch.pruning import prune, get_last_report
from amct_pytorch.pruning.config import MethodSpec, PruneConfig
from amct_pytorch.pruning.context import PruneContext
from amct_pytorch.pruning.domains.base import BasePruningDomain
from amct_pytorch.pruning.domains.moe import MoEPruningDomain
from amct_pytorch.pruning.report import PruneReport

from amct_pytorch.pruning.methods import (
    moe_mass_pruning,
    moe_mass_variance_pruning,
    moe_output_merge,
)
from amct_pytorch.pruning.methods.moe_mass_pruning import (
    ActivationCounter,
    MassMoEPruningMethod,
)
from amct_pytorch.pruning.methods.moe_mass_variance_pruning import (
    LayerRoutingStats,
    MassVarianceMoEPruningMethod,
    _auto_select_score,
    _dispersion,
    _make_router_hook,
    _variance_regime_score,
)
from amct_pytorch.pruning.methods.moe_output_merge import (
    OutputMergeMoEPruningMethod,
    _calib_nll,
    _last_linear,
    _solve_least_squares,
)

VOCAB = 48


def _toy_config(num_experts=4, num_hidden_layers=2):
    return MiniMoEConfig(
        hidden_size=16,
        intermediate_size=24,
        num_experts=num_experts,
        num_experts_per_tok=2,
        num_hidden_layers=num_hidden_layers,
        vocab_size=VOCAB,
    )


def _toy_model(seed=0, **kw):
    torch.manual_seed(seed)
    return MiniMoEModel(_toy_config(**kw)).eval()


def _calib(n=4, batch=2, seqlen=12):
    torch.manual_seed(123)
    return [torch.randint(0, VOCAB, (batch, seqlen)) for _ in range(n)]


def _num_params(model):
    return sum(p.numel() for p in model.parameters())


class _NonMoEDomain(BasePruningDomain):
    """Concrete (non-MoE) domain so the methods' isinstance(domain, MoEPruningDomain) guard
    is the thing that raises -- not the abstract-class instantiation."""

    name = "dense"

    def find_targets(self, model, config):
        return []


# ============================================================================
# mass_variance: variance_score options + boundary variants (end-to-end prune)
# ============================================================================
class TestMassVarianceScoresAndBoundaries(unittest.TestCase):
    def _run(self, kwargs):
        model = _toy_model(1)
        cfg = {
            "methods": {"moe": {"name": "mass_variance", "kwargs": kwargs}},
            "missing_data_policy": "warn_skip",
        }
        prune(model, cfg, data=_calib())
        return model, get_last_report()

    def test_variance_score_cond_default(self):
        # boundary=-1 forces every layer into the variance regime (line 226-227, -1 int path)
        model, report = self._run({"prune_ratio": 0.5, "boundary": -1})
        self.assertLess(report.params_after, report.params_before)
        self.assertTrue(
            any("variance-based" in e.detail for e in report.events)
        )

    def test_variance_score_peak(self):
        model, report = self._run(
            {"prune_ratio": 0.5, "boundary": -1, "variance_score": "peak"}
        )
        self.assertLess(report.params_after, report.params_before)

    def test_variance_score_cvxpeak(self):
        model, report = self._run(
            {"prune_ratio": 0.5, "boundary": -1, "variance_score": "cvxpeak"}
        )
        self.assertLess(report.params_after, report.params_before)

    def test_boundary_auto(self):
        # boundary="auto" -> _auto_select_score / _dispersion (lines 315, 215-216, 183-196)
        model, report = self._run({"prune_ratio": 0.5, "boundary": "auto"})
        self.assertLess(report.params_after, report.params_before)
        self.assertTrue(any("auto" in e.detail for e in report.events))

    def test_boundary_list_explicit(self):
        # variance on layer idx 1 only (in-list), mass on idx 0 (out-of-list): lines 217-221
        model, report = self._run({"prune_ratio": 0.5, "boundary": [1]})
        self.assertLess(report.params_after, report.params_before)
        details = " ".join(e.detail for e in report.events)
        self.assertIn("variance(explicit)", details)
        self.assertIn("mass(explicit)", details)

    def test_boundary_int_mass_regime(self):
        # large int boundary -> all layers use mass (line 222-224, "mass" desc)
        model, report = self._run({"prune_ratio": 0.5, "boundary": 100})
        self.assertLess(report.params_after, report.params_before)
        self.assertTrue(any("mass-based" in e.detail for e in report.events))

    def test_forward_after_prune(self):
        model, _ = self._run({"prune_ratio": 0.5, "boundary": "auto"})
        with torch.no_grad():
            out = model(torch.randint(0, VOCAB, (2, 8)))
        self.assertTrue(torch.isfinite(out).all())


# ============================================================================
# mass_variance: apply() guard branches (lines 305,307,323-324,333-337)
# ============================================================================
class TestMassVarianceApplyGuards(unittest.TestCase):
    def _method_call(self, model, domain, context, spec, config=None):
        config = config or PruneConfig(min_experts=1)
        domain_obj = MoEPruningDomain()
        report = PruneReport(backend="test", params_before=_num_params(model))
        targets = domain_obj.find_targets(model, config)
        method = MassVarianceMoEPruningMethod()
        method.apply(model, domain, targets, context, report, config, spec)
        return report

    def test_raises_without_data(self):
        # line 304-305
        model = _toy_model(2)
        with self.assertRaises(ValueError):
            self._method_call(
                model,
                MoEPruningDomain(),
                PruneContext(data=None),
                MethodSpec("mass_variance", {}),
            )

    def test_raises_on_wrong_domain(self):
        # line 306-307
        model = _toy_model(2)
        with self.assertRaises(TypeError):
            MassVarianceMoEPruningMethod().apply(
                model,
                _NonMoEDomain(),  # not a MoEPruningDomain
                [],
                PruneContext(data=_calib()),
                PruneReport(backend="test", params_before=0),
                PruneConfig(min_experts=1),
                MethodSpec("mass_variance", {}),
            )

    def test_invalid_variance_score_raises(self):
        # line 319-322
        model = _toy_model(2)
        with self.assertRaises(ValueError):
            self._method_call(
                model,
                MoEPruningDomain(),
                PruneContext(data=_calib()),
                MethodSpec("mass_variance", {"variance_score": "bogus"}),
            )

    def test_prune_ratio_zero_is_noop(self):
        # line 323-324: prune_ratio <= 0 -> early return, no pruning
        model = _toy_model(2)
        before = _num_params(model)
        report = self._method_call(
            model,
            MoEPruningDomain(),
            PruneContext(data=_calib()),
            MethodSpec("mass_variance", {"prune_ratio": 0.0}),
        )
        self.assertEqual(_num_params(model), before)
        self.assertEqual(report.events, [])

    def test_empty_dataset_warns_and_skips(self):
        # lines 333-337: num_batches == 0 -> warn + add_warning + return
        model = _toy_model(2)
        before = _num_params(model)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = self._method_call(
                model,
                MoEPruningDomain(),
                PruneContext(data=[]),  # empty iterable -> 0 batches
                MethodSpec("mass_variance", {"prune_ratio": 0.5}),
            )
        self.assertEqual(_num_params(model), before)
        self.assertTrue(any("Empty dataset" in str(w.message) for w in caught))
        self.assertTrue(any("Empty dataset" in w for w in report.warnings))


# ============================================================================
# mass_variance: small-unit helpers (hook internals, dispersion, select)
# ============================================================================
class TestMassVarianceUnits(unittest.TestCase):
    def _stats(self, num_experts=4, k=2):
        return LayerRoutingStats(
            mass=torch.zeros(num_experts, dtype=torch.float64),
            expert_var_sum=torch.zeros(num_experts, dtype=torch.float64),
            expert_select_count=torch.zeros(num_experts, dtype=torch.long),
            expert_var=torch.zeros(num_experts, dtype=torch.float32),
            expert_max=torch.zeros(num_experts, dtype=torch.float32),
            num_experts=num_experts,
            top_k=k,
        )

    def test_hook_returns_on_unparseable_output(self):
        # line 86-87: route_topk_from_output -> None -> hook returns early
        stats = self._stats()
        hook = _make_router_hook(stats, 2)
        hook(nn.Identity(), None, "not-a-tensor")
        self.assertEqual(float(stats.mass.sum()), 0.0)

    def test_hook_returns_when_no_valid_indices(self):
        # lines 94-96: all indices out of [0, num_experts) -> valid.any() False -> return
        stats = self._stats(num_experts=4)
        hook = _make_router_hook(stats, 2)
        out_idx = torch.full((3, 2), 99, dtype=torch.long)  # all >= num_experts
        out_wt = torch.ones(3, 2)
        hook(nn.Identity(), None, (out_idx, out_wt))
        self.assertEqual(float(stats.mass.sum()), 0.0)

    def test_hook_accumulates_valid(self):
        stats = self._stats(num_experts=4)
        hook = _make_router_hook(stats, 2)
        idx = torch.tensor([[0, 1], [0, 2]], dtype=torch.long)
        wt = torch.tensor([[0.6, 0.4], [0.7, 0.3]])
        hook(nn.Identity(), None, (idx, wt))
        self.assertGreater(float(stats.mass[0]), 0.0)
        self.assertGreater(float(stats.expert_select_count.sum()), 0)

    def test_dispersion_nonpositive_mean_returns_zero(self):
        # lines 185-186: mean <= 0 -> 0.0
        self.assertEqual(_dispersion(torch.zeros(4)), 0.0)
        # line 187: positive mean -> std/mean
        self.assertGreater(_dispersion(torch.tensor([1.0, 2.0, 3.0, 4.0])), 0.0)

    def test_variance_regime_score_modes(self):
        st = self._stats()
        st.expert_var = torch.tensor([0.1, 0.2, 0.3, 0.4])
        st.expert_max = torch.tensor([1.0, 2.0, 3.0, 4.0])
        torch.testing.assert_close(_variance_regime_score(st, "peak"), st.expert_max)
        torch.testing.assert_close(
            _variance_regime_score(st, "cvxpeak"), st.expert_var * st.expert_max
        )
        torch.testing.assert_close(_variance_regime_score(st, "cond"), st.expert_var)

    def test_auto_select_prefers_more_dispersed(self):
        # lines 192-196: pick variance when its CV beats mass's CV (and vice versa)
        st = self._stats()
        st.mass = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)  # CV 0
        st.expert_var = torch.tensor([0.0, 1.0, 2.0, 3.0])  # high CV
        st.expert_max = torch.tensor([1.0, 1.0, 1.0, 1.0])
        _, desc = _auto_select_score(st, "cond")
        self.assertEqual(desc, "variance(auto)")

        st.mass = torch.tensor([0.0, 1.0, 2.0, 9.0], dtype=torch.float64)  # high CV
        st.expert_var = torch.tensor([1.0, 1.0, 1.0, 1.0])  # CV 0
        _, desc = _auto_select_score(st, "cond")
        self.assertEqual(desc, "mass(auto)")

    def test_select_keep_returns_none_when_all_kept(self):
        # lines 228-230: keep == num_experts (prune_ratio 0) -> (None, None, num_experts)
        from amct_pytorch.pruning.methods.moe_mass_variance_pruning import (
            _select_keep_indices,
        )

        st = self._stats(num_experts=4)
        st.mass = torch.tensor([4.0, 3.0, 2.0, 1.0], dtype=torch.float64)
        st.expert_var = torch.tensor([0.4, 0.3, 0.2, 0.1])
        layer_stats = {"r": st}
        target = mock.Mock(router_path="r")
        config = PruneConfig(min_experts=1)
        keep, desc, n = _select_keep_indices(
            0, target, layer_stats, config, 0.0, 100, "cond"
        )
        self.assertIsNone(keep)
        self.assertEqual(n, 4)

    def test_select_keep_min_experts_guard(self):
        # lines 212-213: num_experts <= config.min_experts -> (None, None, num_experts)
        from amct_pytorch.pruning.methods.moe_mass_variance_pruning import (
            _select_keep_indices,
        )

        st = self._stats(num_experts=2)
        layer_stats = {"r": st}
        target = mock.Mock(router_path="r")
        config = PruneConfig(min_experts=4)  # min > num_experts
        keep, desc, n = _select_keep_indices(
            0, target, layer_stats, config, 0.5, 5, "cond"
        )
        self.assertIsNone(keep)
        self.assertEqual(n, 2)

    def test_select_keep_missing_router(self):
        # lines 208-209: router_path None / not in stats -> (None, None, 0)
        from amct_pytorch.pruning.methods.moe_mass_variance_pruning import (
            _select_keep_indices,
        )

        target = mock.Mock(router_path=None)
        keep, desc, n = _select_keep_indices(0, target, {}, PruneConfig(), 0.5, 10, "cond")
        self.assertIsNone(keep)
        self.assertEqual(n, 0)

    def test_register_hooks_skips_routerless_target(self):
        # line 119-120: target.router_path is None -> continue (mass_variance variant)
        from amct_pytorch.pruning.methods.moe_mass_variance_pruning import (
            _register_router_hooks,
        )

        domain = MoEPruningDomain()
        model = _toy_model(1)
        targets = domain.find_targets(model, PruneConfig(min_experts=1))
        targets[0].router_path = None
        layer_stats, hooks = _register_router_hooks(model, domain, targets, 2)
        for h in hooks:
            h.remove()
        self.assertNotIn(None, layer_stats)

    def test_prune_layers_skips_none_keep(self):
        # line 252-253: _select_keep_indices None for one layer -> continue, other layer prunes
        from amct_pytorch.pruning.methods.moe_mass_variance_pruning import _prune_layers

        domain = MoEPruningDomain()
        model = _toy_model(1)
        config = PruneConfig(min_experts=1)
        targets = domain.find_targets(model, config)
        report = PruneReport(backend="test", params_before=_num_params(model))
        # stats only for the second target's router -> first target yields None (continue)
        st = self._stats(num_experts=4)
        st.mass = torch.tensor([8.0, 4.0, 2.0, 1.0], dtype=torch.float64)
        st.expert_var = torch.tensor([0.4, 0.3, 0.2, 0.1])
        layer_stats = {targets[1].router_path: st}
        last = _prune_layers(
            "mass_variance", model, domain, targets, layer_stats, config, report,
            0.5, 2, 100, "cond",
        )
        self.assertEqual(last, 2)  # only the second layer was pruned


# ============================================================================
# activation_count (mass) method: apply guards + units
# ============================================================================
class TestMassMethod(unittest.TestCase):
    def test_end_to_end_prunes(self):
        model = _toy_model(3)
        cfg = {
            "methods": {
                "moe": {"name": "activation_count", "kwargs": {"prune_ratio": 0.5}}
            },
            "missing_data_policy": "warn_skip",
        }
        prune(model, cfg, data=_calib())
        report = get_last_report()
        self.assertLess(report.params_after, report.params_before)
        self.assertTrue(any("Mass-based prune" in e.detail for e in report.events))

    def _method_call(self, model, domain, context, spec, config=None):
        config = config or PruneConfig(min_experts=1)
        report = PruneReport(backend="test", params_before=_num_params(model))
        targets = MoEPruningDomain().find_targets(model, config)
        MassMoEPruningMethod().apply(model, domain, targets, context, report, config, spec)
        return report

    def test_raises_without_data(self):
        # lines 209-212
        with self.assertRaises(ValueError):
            self._method_call(
                _toy_model(3),
                MoEPruningDomain(),
                PruneContext(data=None),
                MethodSpec("activation_count", {}),
            )

    def test_raises_on_wrong_domain(self):
        # lines 213-214
        with self.assertRaises(TypeError):
            MassMoEPruningMethod().apply(
                _toy_model(3),
                _NonMoEDomain(),
                [],
                PruneContext(data=_calib()),
                PruneReport(backend="test", params_before=0),
                PruneConfig(min_experts=1),
                MethodSpec("activation_count", {}),
            )

    def test_prune_ratio_zero_noop(self):
        # lines 218-219
        model = _toy_model(3)
        before = _num_params(model)
        report = self._method_call(
            model,
            MoEPruningDomain(),
            PruneContext(data=_calib()),
            MethodSpec("activation_count", {"prune_ratio": 0.0}),
        )
        self.assertEqual(_num_params(model), before)
        self.assertEqual(report.events, [])

    def test_empty_dataset_warns(self):
        # lines 228-232
        model = _toy_model(3)
        before = _num_params(model)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = self._method_call(
                model,
                MoEPruningDomain(),
                PruneContext(data=[]),
                MethodSpec("activation_count", {"prune_ratio": 0.5}),
            )
        self.assertEqual(_num_params(model), before)
        self.assertTrue(any("empty dataset" in str(w.message) for w in caught))
        self.assertTrue(any("empty dataset" in w for w in report.warnings))

    def test_counter_update_returns_on_unparseable(self):
        # line 71-72: route returns None -> update returns early without allocating counts
        c = ActivationCounter()
        c.num_experts = 4
        c.update("not-a-tensor", top_k=2)
        self.assertIsNone(c.counts)

    def test_counter_update_counts(self):
        c = ActivationCounter()
        c.num_experts = 4
        idx = torch.tensor([[0, 1], [0, 3]], dtype=torch.long)
        wt = torch.ones(2, 2)
        c.update((idx, wt), top_k=2)
        self.assertEqual(int(c.counts[0]), 2)
        self.assertEqual(int(c.counts[3]), 1)

    def test_select_keep_units(self):
        # lines 133-134 (missing router), 136-137 (counts None),
        # 139-140 (min_experts), 143-144 (keep == num_experts)
        from amct_pytorch.pruning.methods.moe_mass_pruning import _select_keep_indices

        domain = MoEPruningDomain()
        model = _toy_model(3, num_experts=4)
        config = PruneConfig(min_experts=1)
        targets = domain.find_targets(model, config)
        target = targets[0]

        # missing router path
        keep, n = _select_keep_indices(
            mock.Mock(router_path=None), {}, domain, model, config, 0.5
        )
        self.assertIsNone(keep)

        # counts None
        coll_none = ActivationCounter()
        keep, n = _select_keep_indices(
            target, {target.router_path: coll_none}, domain, model, config, 0.5
        )
        self.assertIsNone(keep)

        # populate counts for the remaining branches
        coll = ActivationCounter()
        coll.num_experts = 4
        coll.counts = torch.tensor([10, 7, 4, 1], dtype=torch.long)
        collectors = {target.router_path: coll}

        # min_experts >= num_experts
        keep, n = _select_keep_indices(
            target, collectors, domain, model, PruneConfig(min_experts=4), 0.5
        )
        self.assertIsNone(keep)
        self.assertEqual(n, 4)

        # keep == num_experts (prune_ratio 0)
        keep, n = _select_keep_indices(target, collectors, domain, model, config, 0.0)
        self.assertIsNone(keep)
        self.assertEqual(n, 4)

        # normal prune
        keep, n = _select_keep_indices(target, collectors, domain, model, config, 0.5)
        self.assertIsNotNone(keep)
        self.assertEqual(len(keep), 2)

    def test_register_hooks_skips_routerless_target(self):
        # line 110-111: target.router_path is None -> continue
        from amct_pytorch.pruning.methods.moe_mass_pruning import _register_router_hooks

        domain = MoEPruningDomain()
        model = _toy_model(3)
        targets = domain.find_targets(model, PruneConfig(min_experts=1))
        targets[0].router_path = None  # force the skip branch
        collectors, hooks = _register_router_hooks(model, domain, targets, 2)
        for h in hooks:
            h.remove()
        self.assertNotIn(None, collectors)

    def test_register_hooks_num_experts_typeerror(self):
        # lines 117-119: domain.num_experts raises -> except pass, collector kept with default
        from amct_pytorch.pruning.methods.moe_mass_pruning import _register_router_hooks

        domain = MoEPruningDomain()
        model = _toy_model(3)
        targets = domain.find_targets(model, PruneConfig(min_experts=1))
        with mock.patch.object(
            MoEPruningDomain, "num_experts", side_effect=ValueError("boom")
        ):
            collectors, hooks = _register_router_hooks(model, domain, targets, 2)
        for h in hooks:
            h.remove()
        self.assertTrue(len(collectors) >= 1)

    def test_prune_layers_skips_none_keep(self):
        # line 164-165: _select_keep_indices None for one layer -> continue, other prunes
        from amct_pytorch.pruning.methods.moe_mass_pruning import _prune_layers

        domain = MoEPruningDomain()
        model = _toy_model(3)
        config = PruneConfig(min_experts=1)
        targets = domain.find_targets(model, config)
        report = PruneReport(backend="test", params_before=_num_params(model))
        coll = ActivationCounter()
        coll.num_experts = 4
        coll.counts = torch.tensor([9, 5, 3, 1], dtype=torch.long)
        # collector only for the second layer's router -> first layer yields None (continue)
        collectors = {targets[1].router_path: coll}
        last = _prune_layers(
            "activation_count", model, domain, targets, collectors, config, report, 0.5, 2
        )
        self.assertEqual(last, 2)


# ============================================================================
# output_merge: end-to-end + apply guards + helper edges
# ============================================================================
class TestOutputMerge(unittest.TestCase):
    def test_prune_ratio_alias(self):
        # lines 459-460: prune_ratio overrides keep_ratio
        model = _toy_model(5)
        cfg = {
            "methods": {
                "moe": {
                    "name": "output_merge",
                    "kwargs": {"prune_ratio": 0.5, "selector": "none"},
                }
            },
            "missing_data_policy": "warn_skip",
        }
        prune(model, cfg, data=_calib())
        report = get_last_report()
        self.assertLess(report.params_after, report.params_before)
        for layer in model.layers:
            self.assertEqual(len(layer.experts), 2)

    def test_raises_without_data(self):
        # lines 452-455
        with self.assertRaises(ValueError):
            OutputMergeMoEPruningMethod().apply(
                _toy_model(5),
                MoEPruningDomain(),
                [],
                PruneContext(data=None),
                PruneReport(backend="test", params_before=0),
                PruneConfig(min_experts=1),
                MethodSpec("output_merge", {}),
            )

    def test_raises_on_wrong_domain(self):
        # lines 456-457
        with self.assertRaises(TypeError):
            OutputMergeMoEPruningMethod().apply(
                _toy_model(5),
                _NonMoEDomain(),
                [],
                PruneContext(data=_calib()),
                PruneReport(backend="test", params_before=0),
                PruneConfig(min_experts=1),
                MethodSpec("output_merge", {}),
            )

    def test_selector_calib_nll_runs(self):
        # exercises the full calib_nll selector path on a deep toy MoE
        model = _toy_model(6, num_experts=4, num_hidden_layers=3)
        cfg = {
            "methods": {
                "moe": {
                    "name": "output_merge",
                    "kwargs": {"keep_ratio": 0.5, "selector": "calib_nll"},
                }
            },
            "missing_data_policy": "warn_skip",
        }
        prune(model, cfg, data=_calib())
        report = get_last_report()
        self.assertLess(report.params_after, report.params_before)
        self.assertTrue(
            any("calib_nll selector" in e.detail for e in report.events)
        )

    def test_last_linear_name_then_children_fallback(self):
        # line 107-110 (named hit) and 111-112 (children fallback / empty)
        class NamedExpert(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(8, 8)
                self.down_proj = nn.Linear(8, 8)

        self.assertIsInstance(_last_linear(NamedExpert()), nn.Linear)

        class GenericExpert(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(8, 8)
                self.b = nn.Linear(8, 8)

            def forward(self, x):
                return self.b(self.a(x))

        ge = GenericExpert()
        self.assertIs(_last_linear(ge), ge.b)  # last direct child Linear

        class NoLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.act = nn.ReLU()

        self.assertIsNone(_last_linear(NoLinear()))  # line 112 [] -> None

    def test_solve_least_squares_cpu_fallback(self):
        # lines 138-141: device solver raises RuntimeError -> CPU fallback
        gram = torch.eye(3)
        rhs = torch.randn(3, 1)
        calls = {"n": 0}
        real_solve = torch.linalg.solve

        def flaky(a, b):
            if calls["n"] == 0:
                calls["n"] += 1
                raise RuntimeError("no solver on device")
            return real_solve(a, b)

        with mock.patch("torch.linalg.solve", side_effect=flaky):
            out = _solve_least_squares(gram, rhs)
        torch.testing.assert_close(out, rhs, rtol=1e-4, atol=1e-4)

    def test_calib_nll_none_when_no_token_labels(self):
        # line 350-351: ids None or seq < 2 -> None
        model = _toy_model(5)
        # float inputs -> _batch_token_ids returns None
        ctx = PruneContext(data=[torch.randn(2, 12, 16)])
        # model forward needs int ids; wrap with a model that accepts the float path is awkward,
        # so test the helper's None branch directly via a model that returns logits but no token ids.
        single_tok = PruneContext(data=[torch.randint(0, VOCAB, (2, 1))])
        self.assertIsNone(_calib_nll(model, single_tok))  # seq length 1 < 2

    def test_calib_nll_handles_tuple_logits(self):
        # lines 352-354: model returns a tuple -> take element 0
        class TupleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(VOCAB, 8)
                self.head = nn.Linear(8, VOCAB)

            def forward(self, input_ids):
                return (self.head(self.emb(input_ids)),)

        m = TupleModel().eval()
        ctx = PruneContext(data=[torch.randint(0, VOCAB, (2, 6))])
        val = _calib_nll(m, ctx)
        self.assertIsNotNone(val)
        self.assertTrue(val > 0)


# ============================================================================
# output_merge: internal merge-block branches (drop, bake bias, NLL-missing)
# ============================================================================
class TestOutputMergeInternals(unittest.TestCase):
    def test_merge_target_returns_none_when_no_room(self):
        # line 258-260: n_keep >= e_n -> None (keep_ratio 1.0 keeps everything)
        domain = MoEPruningDomain()
        model = _toy_model(5, num_experts=4)
        targets = domain.find_targets(model, PruneConfig(min_experts=1))
        x = torch.randn(8, 16)
        out = moe_output_merge._merge_target(
            model, domain, targets[0], x, keep_ratio=1.0, top_k=2,
            config=PruneConfig(min_experts=1),
        )
        self.assertIsNone(out)

    def test_merge_target_returns_none_non_linear_gate(self):
        # line 259-260: gate is not nn.Linear -> None
        domain = MoEPruningDomain()
        model = _toy_model(5, num_experts=4)
        targets = domain.find_targets(model, PruneConfig(min_experts=1))
        # replace the gate with a non-Linear module
        model.layers[0].gate = nn.Identity()
        x = torch.randn(8, 16)
        out = moe_output_merge._merge_target(
            model, domain, targets[0], x, keep_ratio=0.5, top_k=2,
            config=PruneConfig(min_experts=1),
        )
        self.assertIsNone(out)

    def test_expert_hidden_outs_none_on_missing_linear(self):
        # line 119-121: an expert with no last linear -> _expert_hidden_outs returns None
        experts = nn.ModuleList([nn.ReLU()])  # no linear inside
        out = moe_output_merge._expert_hidden_outs(experts, torch.randn(4, 8))
        self.assertIsNone(out)

    def test_input_hook_stops_at_token_cap(self):
        # line 74-75: when accumulated tokens >= _TOKEN_CAP, the hook returns without appending
        feats = {7: [torch.zeros(moe_output_merge._TOKEN_CAP, 4)]}
        hook = moe_output_merge._make_input_hook(feats, 7)
        hook(nn.Identity(), (torch.randn(3, 4),))
        self.assertEqual(len(feats[7]), 1)  # nothing appended past the cap

    def test_merge_target_returns_none_when_experts_lack_linear(self):
        # line 263-264: _expert_hidden_outs returns None -> _merge_target returns None
        domain = MoEPruningDomain()
        model = _toy_model(5, num_experts=4)
        targets = domain.find_targets(model, PruneConfig(min_experts=1))
        # replace experts with linear-free modules so _expert_hidden_outs -> None
        model.layers[0].experts = nn.ModuleList([nn.ReLU() for _ in range(4)])
        x = torch.randn(8, 16)
        out = moe_output_merge._merge_target(
            model, domain, targets[0], x, keep_ratio=0.5, top_k=2,
            config=PruneConfig(min_experts=1),
        )
        self.assertIsNone(out)

    def test_greedy_grouping_drop_branch_and_break(self):
        # lines 206 (best None -> break) and 209 (do_drop -> dropped.extend)
        # Force every action to be a drop by making drop_cost tiny vs LS residual.
        torch.manual_seed(0)
        e_n, n_tok, h = 4, 32, 6
        probs = torch.softmax(torch.randn(n_tok, e_n), -1)
        outs = torch.randn(e_n, n_tok, h)
        hidden_list = [torch.randn(n_tok, h) for _ in range(e_n)]
        biases = [None] * e_n
        usage = probs.sum(0)
        drop_cost = [0.0] * e_n  # zero drop cost -> always drop
        groups, dropped = moe_output_merge._greedy_grouping(
            probs, outs, hidden_list, biases, usage, n_merge=2, drop_cost=drop_cost
        )
        self.assertEqual(len(dropped), 2)  # two experts dropped via the do_drop branch

        # n_merge larger than groups can support -> best becomes None -> break (line 205-206)
        groups2, dropped2 = moe_output_merge._greedy_grouping(
            probs, outs, hidden_list, biases, usage, n_merge=99, drop_cost=drop_cost
        )
        self.assertTrue(len(groups2) >= 1)

    def test_bake_with_bias_writes_gate_bias(self):
        # lines 246-247: gate.bias is not None -> bias row written
        torch.manual_seed(0)
        h, e_n, n_tok = 6, 4, 40
        gate = nn.Linear(h, e_n, bias=True)
        experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, h)) for _ in range(e_n)]
        )
        x = torch.randn(n_tok, h)
        fwd = moe_output_merge._expert_hidden_outs(experts, x)
        self.assertIsNotNone(fwd)
        outs, hidden_list, last_list = fwd
        probs = torch.softmax(gate(x).float(), -1)
        biases = [None if l.bias is None else l.bias.detach().float() for l in last_list]
        groups = {0: [0, 1], 2: [2], 3: [3]}  # group 0 absorbs 1 -> baked
        bias_before = gate.bias.data.clone()
        moe_output_merge._bake_merged_weights(
            gate, groups, hidden_list, biases, probs, outs, last_list, x
        )
        # the absorbing row 0's bias should have been updated
        self.assertFalse(torch.allclose(bias_before[0], gate.bias.data[0]))

    def test_apply_merge_all_skips_block_without_tokens(self):
        # lines 310-314: result None (no calib tokens collected) -> warn + add_warning + continue
        domain = MoEPruningDomain()
        model = _toy_model(5, num_experts=4)
        targets = domain.find_targets(model, PruneConfig(min_experts=1))
        report = PruneReport(backend="test", params_before=_num_params(model))
        # _collect_block_inputs returns None feats for every block (empty data)
        with mock.patch.object(
            moe_output_merge,
            "_collect_block_inputs",
            return_value={id(model.layers[i]): None for i in range(len(model.layers))},
        ), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            merged = moe_output_merge._apply_merge_all(
                "output_merge", model, domain, targets,
                PruneContext(data=_calib()), report, PruneConfig(min_experts=1),
                keep_ratio=0.5, top_k=2,
            )
        self.assertFalse(merged)
        self.assertTrue(any("output_merge skipped" in str(w.message) for w in caught))
        self.assertTrue(any("output_merge skipped" in w for w in report.warnings))

    def test_selector_keeps_merge_when_nll_unavailable(self):
        # lines 396-400: calib NLL None for both -> keep merge + warn
        domain = MoEPruningDomain()
        model = _toy_model(6, num_experts=4)
        targets = domain.find_targets(model, PruneConfig(min_experts=1))
        report = PruneReport(backend="test", params_before=_num_params(model))
        with mock.patch.object(
            moe_output_merge, "_calib_nll", return_value=None
        ), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            moe_output_merge._select_by_calib_nll(
                "output_merge", model, domain, targets,
                PruneContext(data=_calib()), report, PruneConfig(min_experts=1),
                keep_ratio=0.5, top_k=2,
            )
        self.assertTrue(
            any("calib NLL unavailable" in str(w.message) for w in caught)
        )

    def test_selector_picks_activation_count_when_lower_nll(self):
        # line 403-404: nll_act < nll_merge -> activation_count wins
        domain = MoEPruningDomain()
        model = _toy_model(6, num_experts=4)
        targets = domain.find_targets(model, PruneConfig(min_experts=1))
        report = PruneReport(backend="test", params_before=_num_params(model))
        seq = iter([5.0, 1.0])  # first call (merge) high, second (act) low

        def fake_nll(*_a, **_k):
            return next(seq)

        with mock.patch.object(moe_output_merge, "_calib_nll", side_effect=fake_nll):
            moe_output_merge._select_by_calib_nll(
                "output_merge", model, domain, targets,
                PruneContext(data=_calib()), report, PruneConfig(min_experts=1),
                keep_ratio=0.5, top_k=2,
            )
        self.assertTrue(
            any("selected activation_count" in e.detail for e in report.events)
        )


if __name__ == "__main__":
    unittest.main()
