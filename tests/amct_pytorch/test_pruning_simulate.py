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
        masked_out, masked_params = _masked_run(model, cfg, data)
        _assert_same_state(self, before, model)
        real_out, real_params = _real_run(model, cfg, data)
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

    def test_dense_reconstruct_keeps_the_least_squares_result(self):
        """reconstruct rewrites the consumer, which no mask expresses: the delta must land."""
        model, _ = create_mini_mlp()
        model.eval()
        cfg = {
            "methods": {
                "dense": {"name": "reconstruct", "kwargs": {"prune_ratio": 0.4}}
            },
            "missing_data_policy": "warn_skip",
        }
        self._check(model, cfg, _tok(), rtol=1e-4)


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


class LowMemorySearchSurface(unittest.TestCase):
    def test_tolerance_search_picks_the_same_ratio_as_the_copy_search(self):
        cfg = {
            "methods": {
                "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}
            },
            "missing_data_policy": "warn_skip",
        }
        data = _tok()
        ev = lambda m: -float(m(data[0]).float().abs().mean())  # noqa: E731

        masked_model, _ = create_mini_mlp()
        masked_model.eval()
        copy_model = copy.deepcopy(masked_model)
        r_masked, r_copy = PruneReport(), PruneReport()
        prune(
            masked_model,
            copy.deepcopy(cfg),
            data=data,
            tolerance=0.5,
            evaluator=ev,
            ratio_grid=(0.2, 0.4),
            low_memory_search=True,
            report=r_masked,
        )
        prune(
            copy_model,
            copy.deepcopy(cfg),
            data=data,
            tolerance=0.5,
            evaluator=ev,
            ratio_grid=(0.2, 0.4),
            report=r_copy,
        )
        self.assertEqual(r_masked.params_after, r_copy.params_after)

    def test_callbacks_that_mutate_the_model_are_refused(self):
        model, _ = create_mini_mlp()
        model.eval()
        with self.assertRaises(ValueError) as ctx:
            prune(
                model,
                data=_tok(),
                tolerance=0.5,
                evaluator=lambda m: 1.0,
                low_memory_search=True,
                quant_fn=lambda m: None,
            )
        self.assertIn("low_memory_search", str(ctx.exception))

    def test_rejected_outside_a_search(self):
        model, _ = create_mini_mlp()
        model.eval()
        with self.assertRaises(ValueError) as ctx:
            prune(model, data=_tok(), low_memory_search=True)
        self.assertIn("low_memory_search", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()


class PrunedBatchNormKeepsItsMode(unittest.TestCase):
    """A rebuilt BatchNorm must not come back in training mode: an eval model that
    starts normalising by batch statistics gives different outputs and corrupts the
    running stats that were just copied over."""

    def test_eval_model_stays_in_eval_after_pruning(self):
        model, _ = create_mini_cnn()
        model.eval()
        cfg = {
            "methods": {
                "cnn": {"name": "variance_channel", "kwargs": {"prune_ratio": 0.4}}
            },
            "missing_data_policy": "warn_skip",
        }
        prune(model, cfg, data=_images())
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                self.assertFalse(module.training, f"{name} came back in training mode")

    def test_running_stats_survive_a_forward_after_pruning(self):
        model, _ = create_mini_cnn()
        model.eval()
        cfg = {
            "methods": {
                "cnn": {"name": "variance_channel", "kwargs": {"prune_ratio": 0.4}}
            },
            "missing_data_policy": "warn_skip",
        }
        prune(model, cfg, data=_images())
        before = model.bn1.running_mean.detach().clone()
        with torch.no_grad():
            model(torch.randn(2, 3, 32, 32))
        self.assertTrue(torch.equal(before, model.bn1.running_mean))
