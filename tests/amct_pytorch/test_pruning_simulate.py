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


if __name__ == "__main__":
    unittest.main()
