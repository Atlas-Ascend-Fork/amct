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
"""A masked trial must measure exactly what pruning a copy would measure, and must
leave the model untouched. Everything else about low_memory_search follows from that."""

import copy
import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from mini_models import create_mini_mlp, create_mini_cnn, create_mini_moe

from amct_pytorch.pruning import prune, PruneReport
from amct_pytorch.pruning import simulate
from amct_pytorch.pruning.config import PruneConfig
from amct_pytorch.pruning.pruner import AutoPruner
from amct_pytorch.pruning.utils import count_parameters

torch.set_num_threads(2)


def _tok(n=6):
    return [torch.randint(0, 1000, (4, 20)) for _ in range(n)]


def _images(n=8):
    return [torch.randn(2, 3, 32, 32) for _ in range(n)]


def _state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _meta(model):
    return copy.deepcopy(getattr(model, "_amct_prune_meta", None))


def _assert_same_state(case, before, model):
    after = model.state_dict()
    case.assertEqual(sorted(before), sorted(after), "module set changed")
    for name, ref in before.items():
        case.assertTrue(
            torch.equal(ref, after[name]), f"{name} was modified by a masked trial"
        )


def _masked_run(model, cfg_dict, data):
    """Run one trial through the simulation session; returns (output, params_after)."""
    cfg = PruneConfig(**copy.deepcopy(cfg_dict))
    cfg.copy_model = False
    params_before = count_parameters(model)
    with simulate.SimulationSession(model) as session:
        AutoPruner(cfg)(model, data=data)
        probe = data[0]
        with torch.no_grad():
            out = model(probe).detach().clone()
        return out, params_before - session.removed_params


def _real_run(model, cfg_dict, data):
    """Prune a copy for real; returns (output, params_after)."""
    clone = copy.deepcopy(model)
    report = PruneReport()
    prune(clone, copy.deepcopy(cfg_dict), data=data, report=report)
    with torch.no_grad():
        out = clone(data[0]).detach().clone()
    return out, report.params_after


class MaskedTrialMatchesRealPrune(unittest.TestCase):
    """The whole design rests on this: mask-in == cut-out, output and param count."""

    def _check(self, model, cfg, data, rtol=1e-6):
        """Params must match exactly; outputs only up to float reassociation.

        A masked trial keeps the matmul at its original width and zeroes the removed
        columns, while a real cut makes the matmul narrower. Same terms, different
        accumulation order, so the two agree to a few ulp rather than bit-for-bit.
        """
        before = _state(model)
        with torch.no_grad():
            pristine_out = model(data[0]).detach().clone()
        masked_out, masked_params = _masked_run(model, cfg, data)
        _assert_same_state(self, before, model)
        real_out, real_params = _real_run(model, cfg, data)
        # Without this the whole comparison passes when nothing happened on either side:
        # a mask that never took effect matches a cut that never landed.
        moved = (masked_out - pristine_out).abs().max().item()
        self.assertGreater(
            moved / (pristine_out.abs().max().item() or 1.0),
            1e-6,
            "the mask did not change the forward -- it never took effect",
        )
        self.assertEqual(masked_params, real_params, "param accounting differs")
        scale = real_out.abs().max().item() or 1.0
        gap = (masked_out - real_out).abs().max().item()
        self.assertLessEqual(
            gap / scale, rtol, f"masked output differs by {gap} (scale {scale})"
        )

    def test_dense_low_variance(self):
        model, _ = create_mini_mlp()
        model.eval()
        cfg = {
            "methods": {
                "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}
            },
            "missing_data_policy": "warn_skip",
        }
        self._check(model, cfg, _tok())

    def test_cnn_variance_channel(self):
        model, _ = create_mini_cnn()
        model.eval()
        cfg = {
            "methods": {
                "cnn": {"name": "variance_channel", "kwargs": {"prune_ratio": 0.4}}
            },
            "missing_data_policy": "warn_skip",
        }
        self._check(model, cfg, _images())

    def test_moe_mass_variance(self):
        model, _ = create_mini_moe()
        model.eval()
        cfg = {
            "methods": {
                "moe": {
                    "name": "mass_variance",
                    "kwargs": {"prune_ratio": 0.5, "boundary": 10},
                }
            },
            "missing_data_policy": "warn_skip",
        }
        self._check(model, cfg, _tok())


class MaskedSearchIsNonDestructive(unittest.TestCase):
    def test_search_leaves_the_model_bit_identical_until_it_applies(self):
        model, _ = create_mini_mlp()
        model.eval()
        before = _state(model)
        cfg = PruneConfig(
            methods={"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}},
            missing_data_policy="warn_skip",
        )
        cfg.copy_model = False
        with simulate.SimulationSession(model) as session:
            AutoPruner(cfg)(model, data=_tok())
            self.assertGreater(session.removed_params, 0)
        _assert_same_state(self, before, model)

    def test_trials_do_not_accumulate_prune_metadata(self):
        """Two ratios on one model must not leave two widths behind: the final config
        sync reads that as a non-uniform cut and then refuses to write the count."""
        model, _ = create_mini_moe()
        model.eval()
        cfg = {
            "methods": {
                "moe": {
                    "name": "mass_variance",
                    "kwargs": {"prune_ratio": 0.25, "boundary": 10},
                }
            },
            "missing_data_policy": "warn_skip",
        }
        start = _meta(model)
        for ratio in (0.25, 0.5):
            spec = copy.deepcopy(cfg)
            spec["methods"]["moe"]["kwargs"]["prune_ratio"] = ratio
            _masked_run(model, spec, _tok())
            self.assertEqual(_meta(model), start, f"ratio {ratio} left metadata behind")

    def test_nested_session_restores_in_order(self):
        model, _ = create_mini_mlp()
        model.eval()
        before = _state(model)
        with simulate.SimulationSession(model):
            with simulate.SimulationSession(model) as inner:
                self.assertIs(simulate.active(), inner)
            self.assertIsNot(simulate.active(), inner)
        self.assertIsNone(simulate.active())
        _assert_same_state(self, before, model)


class SearchResultIsUnchanged(unittest.TestCase):
    """The masked trial is an implementation detail of the search: same answer, either way.

    A no-op ``finetune_fn`` is enough to force the copy path, since a callback that may
    modify the model rules masking out.
    """

    CFG = {
        "methods": {"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}},
        "missing_data_policy": "warn_skip",
    }

    def test_tolerance_search_matches_the_copy_search(self):
        """Both paths must land on the same mid-grid ratio.

        The tolerance is calibrated between the drop at 0.3 and the drop at 0.7, so a
        search that measures nothing (a mask that never took effect measures every
        candidate at zero drop) picks 0.7 and fails -- this test must stay
        discriminating, it once passed with every mask hook disabled.
        """
        torch.manual_seed(0)
        data = _tok()
        reference, _ = create_mini_mlp()
        reference.eval()
        probe = data[0]
        with torch.no_grad():
            ref = reference(probe).detach().clone()
        scale = ref.abs().mean().clamp_min(1e-12)

        def ev(m):
            with torch.no_grad():
                return -float((m(probe) - ref).abs().mean() / scale)

        drops = {}
        for ratio in (0.3, 0.7):
            trial_model = copy.deepcopy(reference)
            cfg = copy.deepcopy(self.CFG)
            cfg["methods"]["dense"]["kwargs"]["prune_ratio"] = ratio
            prune(trial_model, cfg, data=data)
            drops[ratio] = -ev(trial_model)
        self.assertLess(drops[0.3], drops[0.7], "grid does not discriminate")
        tolerance = (drops[0.3] + drops[0.7]) / 2

        results = {}
        for label, extra in (("masked", {}), ("copy", {"finetune_fn": lambda m: None})):
            model = copy.deepcopy(reference)
            report = PruneReport()
            prune(
                model,
                copy.deepcopy(self.CFG),
                data=data,
                tolerance=tolerance,
                evaluator=ev,
                ratio_grid=(0.3, 0.7),
                report=report,
                **extra,
            )
            results[label] = report.params_after
        self.assertEqual(results["masked"], results["copy"])
        # The mid tolerance admits 0.3 and rejects 0.7: the model must shrink, but
        # not by what 0.7 would have removed.
        base = sum(p.numel() for p in reference.parameters())
        self.assertLess(results["masked"], base)
        self.assertGreater(results["masked"], base * 0.5)


class TrialsMustNotMutateTheModel(unittest.TestCase):
    """Findings from the adversarial review: state that escaped the session."""

    def _moe_cfg(self, ratio):
        return {
            "methods": {
                "moe": {
                    "name": "mass_variance",
                    "kwargs": {"prune_ratio": ratio, "boundary": 10},
                }
            },
            "missing_data_policy": "warn_skip",
        }

    def test_router_attributes_survive_a_trial(self):
        """One aggressive masked trial used to lower gate.top_k for good, changing what
        every later trial measured."""
        model, _ = create_mini_moe()
        model.eval()
        gate = model.layers[0].gate
        gate.top_k = 2  # the attribute real HF routers carry
        _masked_run(model, self._moe_cfg(0.875), _tok())
        self.assertEqual(gate.top_k, 2, "trial permanently lowered router.top_k")

    def test_model_config_survives_a_trial(self):
        """patch_common_config used to run inside the trial; _clamp_int_attrs only ever
        lowers num_experts_per_tok, so nothing could restore it afterwards."""
        model, _ = create_mini_moe()
        model.eval()
        before_experts = model.config.num_experts
        before_topk = model.config.num_experts_per_tok
        _masked_run(model, self._moe_cfg(0.875), _tok())
        self.assertEqual(model.config.num_experts, before_experts)
        self.assertEqual(model.config.num_experts_per_tok, before_topk)

    def test_warn_skip_config_is_not_maskable(self):
        """warn_skip deep-copies per stage; the hooks would land on the copy while
        quality reads the original, so every candidate would measure unpruned."""
        from amct_pytorch.pruning.accuracy_based_auto_prune import _maskable

        cfg = self._moe_cfg(0.5)
        self.assertTrue(_maskable(cfg))
        cfg["stage_error_policy"] = "warn_skip"
        self.assertFalse(_maskable(cfg))


class MethodsThatRewriteWeightsKeepCopying(unittest.TestCase):
    """A mask can express "this channel is gone", not "these weights were re-solved"."""

    def test_reconstruct_is_not_maskable(self):
        from amct_pytorch.pruning.accuracy_based_auto_prune import _maskable

        selection = {"methods": {"dense": {"name": "low_variance", "kwargs": {}}}}
        rewriting = {"methods": {"dense": {"name": "reconstruct", "kwargs": {}}}}
        self.assertTrue(_maskable(selection))
        self.assertFalse(_maskable(rewriting))

    def test_output_merge_is_not_maskable(self):
        from amct_pytorch.pruning.accuracy_based_auto_prune import _maskable

        self.assertFalse(
            _maskable({"methods": {"moe": {"name": "output_merge", "kwargs": {}}}})
        )


class _ScatterRouter(nn.Module):
    """Top-k router that scatters softmax-over-top-k back into a full-width tensor.

    Mirrors the HF pattern where the block receives (tokens, num_experts) routing
    weights whose non-selected entries are exactly zero, plus the top-k indices. The
    logits are born from a bare weight (no nn.Linear anywhere), so the trial cannot
    poison a bias and must rebuild the returned tuple itself.
    """

    def __init__(self, hidden_size, num_experts, top_k):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_experts, hidden_size))
        self.top_k = top_k

    def forward(self, hidden_states):
        logits = nn.functional.linear(hidden_states, self.weight)
        k = min(self.top_k, logits.shape[-1])
        top_vals, top_idx = torch.topk(logits, k, dim=-1)
        scores = torch.zeros_like(logits)
        scores = scores.scatter(-1, top_idx, torch.softmax(top_vals, dim=-1))
        return scores, top_idx


class _ScatterMoELayer(nn.Module):
    """MoE block that dispatches on the indices but weights by the FULL-WIDTH scores."""

    def __init__(self, hidden_size, intermediate_size, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.gate = _ScatterRouter(hidden_size, num_experts, top_k)
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_size, intermediate_size),
                nn.GELU(),
                nn.Linear(intermediate_size, hidden_size),
            )
            for _ in range(num_experts)
        )

    def forward(self, hidden_states):
        batch, seq, hidden = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden)
        scores, top_idx = self.gate(flat)
        out = torch.zeros_like(flat)
        for expert_id in range(self.num_experts):
            token_mask = (top_idx == expert_id).any(dim=-1)
            if token_mask.any():
                weight = scores[token_mask, expert_id].unsqueeze(-1)
                out[token_mask] += weight * self.experts[expert_id](flat[token_mask])
        return out.view(batch, seq, hidden)


class _ScatterMoEModel(nn.Module):
    def __init__(self, num_experts=8, top_k=2, hidden_size=16, intermediate_size=32):
        super().__init__()
        vocab = num_experts * num_experts
        self.embeddings = nn.Embedding(vocab, hidden_size)
        self.layers = nn.ModuleList(
            [_ScatterMoELayer(hidden_size, intermediate_size, num_experts, top_k)]
        )
        self.classifier = nn.Linear(hidden_size, vocab)

    def forward(self, input_ids):
        hidden_states = self.embeddings(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.classifier(hidden_states)


def _scatter_moe_model_and_data():
    """Build the scatter model with routing engineered for an exact comparison.

    The router weight is a basis projection, so a token's logits are its first
    ``num_experts`` embedding dims: each token id encodes a pair (a, b) routed at
    logits 30 and 29 with every other logit 0. That pins three things down: which
    experts the mass criterion removes (6 and 7 appear in one pair each, the rest in
    many), that no token routes to two removed experts at once, and that the softmax
    mass a real prune promotes into a freed slot is ~e^-29 -- far below the output
    tolerance, so reselection-without-promotion is exact to float precision.
    """
    torch.manual_seed(7)
    num_experts, top_k, hidden_size = 8, 2, 16
    model = _ScatterMoEModel(num_experts, top_k, hidden_size)
    with torch.no_grad():
        gate = model.layers[0].gate
        gate.weight.zero_()
        gate.weight[:, :num_experts] = torch.eye(num_experts)
        emb = model.embeddings.weight
        emb[:, :num_experts] = 0.0
        emb[:, num_experts:] = 0.1 * torch.randn(
            emb.shape[0], hidden_size - num_experts
        )
        for a in range(num_experts):
            for b in range(num_experts):
                if a != b:
                    emb[a * num_experts + b, a] = 30.0
                    emb[a * num_experts + b, b] = 29.0
    heavy = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    heavy += [(b, a) for a, b in heavy]
    light = [(6, 0), (7, 1)]
    ids = [a * num_experts + b for a, b in heavy * 2 + light]
    return model, [torch.tensor(ids, dtype=torch.long).view(2, -1)]


class ScatterRouterMaskMatchesRealPrune(unittest.TestCase):
    """Finding: scatter-pattern routers were silently mismeasured.

    The (tokens, k) replacements alone are not enough for this router: the block
    consumes the FULL-WIDTH scores tensor, and masking it leaves the survivors with
    their unrenormalised values -- the removed experts' probability mass is lost
    instead of redistributed, while a real prune renormalises over the survivors.
    The hook must scatter the reselected top-k back into a zero canvas.
    """

    def test_masked_trial_matches_real_prune_on_scatter_router(self):
        model, data = _scatter_moe_model_and_data()
        model.eval()
        cfg = {
            "methods": {
                "moe": {
                    "name": "mass_variance",
                    "kwargs": {"prune_ratio": 0.25, "boundary": 10},
                }
            },
            "missing_data_policy": "warn_skip",
        }
        before = _state(model)
        with torch.no_grad():
            pristine_out = model(data[0]).detach().clone()
        masked_out, masked_params = _masked_run(model, cfg, data)
        _assert_same_state(self, before, model)
        real_out, real_params = _real_run(model, cfg, data)
        moved = (masked_out - pristine_out).abs().max().item()
        self.assertGreater(
            moved / (pristine_out.abs().max().item() or 1.0),
            1e-6,
            "the mask did not change the forward -- it never took effect",
        )
        self.assertEqual(masked_params, real_params, "param accounting differs")
        scale = real_out.abs().max().item() or 1.0
        gap = (masked_out - real_out).abs().max().item()
        self.assertLessEqual(
            gap / scale,
            1e-5,
            f"masked output differs from the real prune by {gap} (scale {scale}): "
            "survivor mass was not redistributed into the full-width scores",
        )


class PrunedBatchNormKeepsItsMode(unittest.TestCase):
    """Rebuilding a BatchNorm must preserve its mode. An eval model that comes back in
    training mode normalises by batch statistics and overwrites the running stats that
    were just copied over -- which is also why a masked trial and a real cut disagreed."""

    CFG = {
        "methods": {
            "cnn": {"name": "variance_channel", "kwargs": {"prune_ratio": 0.4}}
        },
        "missing_data_policy": "warn_skip",
    }

    def test_eval_model_stays_in_eval(self):
        model, _ = create_mini_cnn()
        model.eval()
        prune(model, self.CFG, data=_images())
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                self.assertFalse(module.training, f"{name} came back in training mode")

    def test_running_stats_survive_a_forward(self):
        model, _ = create_mini_cnn()
        model.eval()
        prune(model, self.CFG, data=_images())
        before = model.bn1.running_mean.detach().clone()
        with torch.no_grad():
            model(torch.randn(2, 3, 32, 32))
        self.assertTrue(torch.equal(before, model.bn1.running_mean))


class CombinedDenseAndMoeAccounting(unittest.TestCase):
    """Dense find_targets matches the FFN pair inside every expert, so a combined
    dense+moe masked search records axis cuts on modules the MoE stage then drops
    wholesale. Without the containment rule in removed_params, each dropped expert
    contributes its full (uncut) numel AND its internal fc1/fc2 cuts, so the masked
    params_after lands below what the real prune reports by exactly the dropped
    experts' dense savings; this test fails without that fix."""

    def test_masked_params_match_real_prune_with_both_domains(self):
        torch.manual_seed(0)
        model, _ = create_mini_moe()
        model.eval()
        data = _tok()
        cfg = {
            "methods": {
                "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}},
                "moe": {
                    "name": "mass_variance",
                    "kwargs": {"prune_ratio": 0.5, "boundary": 10},
                },
            },
            "missing_data_policy": "warn_skip",
        }
        before = _state(model)
        params_before = count_parameters(model)
        _, masked_params = _masked_run(model, cfg, data)
        _assert_same_state(self, before, model)
        _, real_params = _real_run(model, cfg, data)
        # Guard against a vacuous pass: both stages must actually have landed.
        self.assertLess(real_params, params_before)
        self.assertEqual(
            masked_params,
            real_params,
            "combined dense+moe accounting double-counts the internal FFN cuts "
            "of the experts the MoE stage drops",
        )


class GroupCollapseScopeMirrorsRealPrune(unittest.TestCase):
    """Group-attr collapse must land exactly where the real prune lands it:
    update_moe_attributes collapses n_group/topk_group on the BLOCK object and
    _prune_fused_router collapses them on the ROUTER; nested descendants are never
    touched. Without the fix this test fails twice mid-trial: the old code skipped
    the router entirely (a fused-style router kept doing grouped top-k among
    poisoned logits, so kept experts in decimated groups were unroutable in the
    trial only), and it walked block.modules() recursively (collapsing group attrs
    on nested descendants the real prune never touches)."""

    def test_collapse_hits_block_and_router_only_and_restores(self):
        from amct_pytorch.pruning.domains.moe import MoETarget

        model, _ = create_mini_moe()
        model.eval()
        block = model.layers[0]
        router = block.gate
        nested = block.experts[0]
        block.n_group = 2
        block.topk_group = 2
        router.n_group = 2
        router.topk_group = 2
        nested.n_group = 4
        target = MoETarget(
            module_path="layers.0",
            router_path="layers.0.gate",
            experts_path="layers.0.experts",
            fused=False,
        )
        with simulate.SimulationSession(model) as session:
            session.record_moe(model, target, 8, [0, 1, 2, 3])
            # During the trial the block is collapsed, as the real prune's
            # update_moe_attributes would do on the block object...
            self.assertEqual(block.n_group, 1)
            self.assertEqual(block.topk_group, 1)
            # ...and so is the router, as _prune_fused_router would do; the old
            # code skipped the router, so these two fail without the fix.
            self.assertEqual(router.n_group, 1)
            self.assertEqual(router.topk_group, 1)
            # A nested descendant stays untouched: the real prune never walks
            # into it. The old recursive walk collapsed it, failing this one.
            self.assertEqual(nested.n_group, 4)
        # Leaving the session restores every collapsed attribute.
        self.assertEqual(block.n_group, 2)
        self.assertEqual(block.topk_group, 2)
        self.assertEqual(router.n_group, 2)
        self.assertEqual(router.topk_group, 2)
        self.assertEqual(nested.n_group, 4)


class _BiasBlindGate(nn.Linear):
    """An nn.Linear subclass whose forward never reads self.bias.

    Real routers do this: a custom forward that calls F.linear(x, self.weight)
    directly. The bias-injection rung then poisons a tensor the forward never
    consumes, and the mask is silently inert.
    """

    def forward(self, x):
        return nn.functional.linear(x, self.weight)


def _calibrated_tolerance(case, reference, cfg, data, ev, ratios):
    """Midpoint tolerance between the real quality drops at the two grid ratios."""
    drops = {}
    for ratio in ratios:
        trial_model = copy.deepcopy(reference)
        spec = copy.deepcopy(cfg)
        domain = next(iter(spec["methods"]))
        spec["methods"][domain]["kwargs"]["prune_ratio"] = ratio
        prune(trial_model, spec, data=data)
        drops[ratio] = -ev(trial_model)
    case.assertLess(drops[ratios[0]], drops[ratios[1]], "grid does not discriminate")
    return (drops[ratios[0]] + drops[ratios[1]]) / 2


def _fidelity_evaluator(reference, probe):
    with torch.no_grad():
        ref = reference(probe).detach().clone()
    scale = ref.abs().mean().clamp_min(1e-12)

    def ev(m):
        with torch.no_grad():
            return -float((m(probe) - ref).abs().mean() / scale)

    return ev


class RouterThatIgnoresItsBias(unittest.TestCase):
    """The bias-injection rung must verify its poison took, not assume it.

    Without the tripwire hook in _reroute, the poisoned bias on a bias-blind
    router changes nothing: every masked candidate measures zero quality drop,
    the search accepts the max grid ratio, and the copy search (which measures
    real drops) picks the smaller one -- so the equality below fails.
    """

    CFG = {
        "methods": {
            "moe": {
                "name": "mass_variance",
                "kwargs": {"prune_ratio": 0.5, "boundary": 10},
            }
        },
        "missing_data_policy": "warn_skip",
    }

    def test_masked_search_matches_copy_search_on_bias_blind_router(self):
        torch.manual_seed(0)
        data = _tok()
        reference, _ = create_mini_moe()
        for layer in reference.layers:
            blind = _BiasBlindGate(
                layer.gate.in_features, layer.gate.out_features, bias=False
            )
            with torch.no_grad():
                blind.weight.copy_(layer.gate.weight)
            layer.gate = blind
        reference.eval()
        ev = _fidelity_evaluator(reference, data[0])
        tolerance = _calibrated_tolerance(
            self, reference, self.CFG, data, ev, (0.25, 0.75)
        )

        results = {}
        for label, extra in (("masked", {}), ("copy", {"finetune_fn": lambda m: None})):
            model = copy.deepcopy(reference)
            report = PruneReport()
            prune(
                model,
                copy.deepcopy(self.CFG),
                data=data,
                tolerance=tolerance,
                evaluator=ev,
                ratio_grid=(0.25, 0.75),
                report=report,
                **extra,
            )
            results[label] = report.params_after
        self.assertEqual(
            results["masked"],
            results["copy"],
            "an inert router mask let the search accept the max grid ratio",
        )


class MaskedApplyFailureFallsBackToCopy(unittest.TestCase):
    """A generic exception from the masked path is a simulation failure, not a verdict.

    Without the _TRIAL_ERRORS fallback in _run_one, a RuntimeError raised
    mid-apply under a session reaches trial()'s handler and every candidate is
    scored as a rejected ratio: the sabotaged search prunes nothing while the
    pure copy search finds a ratio, so both assertions below fail.
    """

    CFG = {
        "methods": {"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}},
        "missing_data_policy": "warn_skip",
    }

    def test_runtime_error_mid_apply_reruns_on_a_copy(self):
        from amct_pytorch.pruning.domains.dense import DensePruningDomain

        torch.manual_seed(0)
        data = _tok()
        reference, _ = create_mini_mlp()
        reference.eval()
        ev = _fidelity_evaluator(reference, data[0])
        tolerance = _calibrated_tolerance(
            self, reference, self.CFG, data, ev, (0.3, 0.7)
        )

        original = DensePruningDomain.apply_keep_indices

        def sabotaged(domain, model, target, keep_idx):
            # Only the masked path runs under an active session; copy trials and
            # the final apply must keep working.
            if simulate.active() is not None:
                raise RuntimeError("injected failure in the masked apply path")
            return original(domain, model, target, keep_idx)

        def _search(extra):
            model = copy.deepcopy(reference)
            report = PruneReport()
            prune(
                model,
                copy.deepcopy(self.CFG),
                data=data,
                tolerance=tolerance,
                evaluator=ev,
                ratio_grid=(0.3, 0.7),
                report=report,
                **extra,
            )
            return report.params_after

        DensePruningDomain.apply_keep_indices = sabotaged
        try:
            masked_result = _search({})
        finally:
            DensePruningDomain.apply_keep_indices = original
        copy_result = _search({"finetune_fn": lambda m: None})

        self.assertEqual(
            masked_result,
            copy_result,
            "a masked-path failure was scored as a rejected ratio",
        )
        base = sum(p.numel() for p in reference.parameters())
        self.assertLess(masked_result, base, "the search pruned nothing")


if __name__ == "__main__":
    unittest.main()
