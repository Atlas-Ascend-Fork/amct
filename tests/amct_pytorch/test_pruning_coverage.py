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
"""Covers the public surface that was implemented but previously untested: default config path,
activation_count(MoE), FULL_STRUCTURED preset, PrunePass, per_layer_sparsity, and the evaluator
path sharing the quantization accuracy abstraction.
"""

import os
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.dirname(__file__))

from mini_models import create_mini_mlp, create_mini_moe
import torch.nn as nn
import torch.nn.functional as F

from amct_pytorch.pruning import (
    prune,
    accuracy_based_auto_prune,
    get_last_report,
    PrunePass,
    FULL_STRUCTURED_PRUNE_CFG,
    prune_diagnose,
    prune_finetune,
)

# internal helper/result classes aren't in the package-level public namespace; import via submodule path
from amct_pytorch.pruning.diagnostics import DiagnosisReport


def _tok(n=6):
    return [torch.randint(0, 1000, (4, 20)) for _ in range(n)]


def _params(m):
    return sum(p.numel() for p in m.parameters())


class TestFinetune(unittest.TestCase):
    def test_finetune_reduces_loss(self):
        data, d, c = self._toy_task()
        model = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, c))
        hist = prune_finetune(
            model,
            data,
            loss_fn=lambda m, b: F.cross_entropy(m(b[0]), b[1]),
            steps=200,
            lr=1e-2,
        )
        self.assertEqual(hist["steps"], 200)
        # training should significantly reduce loss
        self.assertLess(hist["final_loss"], hist["initial_loss"])
        self.assertLess(hist["final_loss"], 0.5)

    def test_finetune_default_loss_xy(self):
        # default loss infers cross-entropy from (x, y) without an explicit loss_fn
        data, d, c = self._toy_task()
        model = nn.Sequential(nn.Linear(d, c))
        hist = prune_finetune(model, data, steps=50, lr=1e-2)
        self.assertEqual(hist["steps"], 50)
        self.assertIsNotNone(hist["final_loss"])

    def test_finetune_zero_steps_noop(self):
        data, d, c = self._toy_task()
        model = nn.Sequential(nn.Linear(d, c))
        before = [p.clone() for p in model.parameters()]
        hist = prune_finetune(model, data, steps=0)
        self.assertEqual(hist["steps"], 0)
        for a, b in zip(before, model.parameters()):
            self.assertTrue(torch.equal(a, b))  # unchanged

    def _toy_task(self, n=256, d=16, c=4):
        torch.manual_seed(0)
        w_mat = torch.randn(d, c)
        x_in = torch.randn(n, d)
        y_lbl = (x_in @ w_mat).argmax(-1)  # deterministic learnable labels
        batches = []
        for i in range(0, n, 32):
            sl = slice(i, i + 32)
            batches.append((x_in[sl], y_lbl[sl]))
        return batches, d, c


class TestDefaultAndMoEMethods(unittest.TestCase):
    def test_default_config_prunes(self):
        # prune(model) with no config: uses PruneConfig() default three-domain methods (moe=activation_count)
        model, config = create_mini_moe()
        model.eval()
        p_before = _params(model)
        prune(model, data=_tok())  # default config
        self.assertLess(_params(model), p_before)
        with torch.no_grad():
            out = model(torch.randint(0, 1000, (2, 20)))
        self.assertEqual(out.shape, (2, 20, config.vocab_size))

    def test_moe_activation_count_method(self):
        # explicit call to MoE activation_count (MassMoEPruningMethod, the default method)
        model, config = create_mini_moe()
        model.eval()
        cfg = {
            "methods": {
                "moe": {"name": "activation_count", "kwargs": {"prune_ratio": 0.5}}
            },
            "missing_data_policy": "warn_skip",
        }
        prune(model, cfg, data=_tok())
        report = get_last_report()
        self.assertLess(report.params_after, report.params_before)
        with torch.no_grad():
            out = model(torch.randint(0, 1000, (2, 20)))
        self.assertEqual(out.shape, (2, 20, config.vocab_size))

    def test_full_structured_preset(self):
        # FULL_STRUCTURED_PRUNE_CFG (all three domains at once) takes effect on a MoE model
        model, _ = create_mini_moe()
        model.eval()
        p_before = _params(model)
        prune(model, FULL_STRUCTURED_PRUNE_CFG, data=_tok())
        self.assertLess(_params(model), p_before)


class TestReportAndPass(unittest.TestCase):
    def test_per_layer_sparsity_recorded(self):
        model, _ = create_mini_mlp()
        model.eval()
        prune(
            model,
            {
                "methods": {
                    "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}
                },
                "missing_data_policy": "warn_skip",
            },
            data=_tok(),
        )
        report = get_last_report()
        self.assertGreater(len(report.per_layer_sparsity), 0)
        for sparsity in report.per_layer_sparsity.values():
            self.assertGreater(sparsity, 0.0)
        # serializable as_dict includes this field
        self.assertIn("per_layer_sparsity", report.as_dict())

    def test_prune_pass_run(self):
        # PrunePass (ModelOptimizer style, the entry point chained with the quantization pass) runs standalone
        model, _ = create_mini_mlp()
        model.eval()
        p_before = _params(model)
        pass_ = PrunePass(
            {
                "methods": {
                    "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}
                },
                "missing_data_policy": "warn_skip",
            },
            data=_tok(),
        )
        pass_.run(model)
        self.assertLess(_params(model), p_before)
        self.assertIsNotNone(pass_.last_report)

    def test_prune_pass_rejects_non_module(self):
        pass_ = PrunePass()
        with self.assertRaises(TypeError):
            pass_.run("not a module")



class _FakeEvaluator:
    """Mimics the quantization-side ModelEvaluator: evaluate(model, iterations) -> accuracy (here a param ratio)."""

    def __init__(self, p0):
        self.p0 = p0
        self.calls = 0

    def evaluate(self, model, iterations):
        self.calls += 1
        return sum(p.numel() for p in model.parameters()) / self.p0


class TestEvaluatorUnification(unittest.TestCase):
    def test_evaluator_drives_search(self):
        model, _ = create_mini_mlp()
        model.eval()
        p0 = _params(model)
        ev = _FakeEvaluator(p0)
        res = accuracy_based_auto_prune(
            model,
            {
                "methods": {"dense": {"name": "low_variance"}},
                "missing_data_policy": "warn_skip",
            },
            data=_tok(),
            tolerance=0.2,
            evaluator=ev,
        )
        # evaluator is actually called (baseline + each trial)
        self.assertGreater(ev.calls, 1)
        self.assertIsNotNone(res.chosen_ratio)
        # chosen point meets tolerance (under the param-ratio metric, equivalent to weight reduction <= tolerance)
        self.assertLessEqual(res.quality_drop, 0.2 + 1e-9)
        self.assertLessEqual(res.weight_reduction, 0.2 + 1e-9)

    def test_bad_evaluator_raises(self):
        model, _ = create_mini_mlp()
        model.eval()
        with self.assertRaises(TypeError):
            accuracy_based_auto_prune(
                model, data=_tok(), tolerance=0.2, evaluator=object()
            )  # no .evaluate

    def test_single_arg_evaluator_works(self):
        # an AutoCalibrationEvaluatorBase-style single-arg evaluate(model) can also drive the search
        model, _ = create_mini_mlp()
        model.eval()
        p0 = _params(model)

        class _OneArgEvaluator:  # same shape as quantization's AutoCalibrationEvaluatorBase.evaluate(self, model)
            def __init__(self):
                self.calls = 0

            def evaluate(self, model):
                self.calls += 1
                return sum(p.numel() for p in model.parameters()) / p0

        ev = _OneArgEvaluator()
        res = accuracy_based_auto_prune(
            model,
            {
                "methods": {"dense": {"name": "low_variance"}},
                "missing_data_policy": "warn_skip",
            },
            data=_tok(),
            tolerance=0.2,
            evaluator=ev,
        )
        self.assertGreater(ev.calls, 1)
        self.assertIsNotNone(res.chosen_ratio)

    def test_none_returning_evaluator_raises(self):
        # graph-version ModelEvaluator (a data feeder whose evaluate returns None) raises a clear error
        model, _ = create_mini_mlp()
        model.eval()

        class _FeederEvaluator:  # mimics graph ModelEvaluator: returns None
            @staticmethod
            def evaluate(model, iterations):
                return None

        with self.assertRaises(TypeError):
            accuracy_based_auto_prune(
                model, data=_tok(), tolerance=0.2, evaluator=_FeederEvaluator()
            )


class TestDiagnose(unittest.TestCase):
    def test_diagnose_functional_model(self):
        model, _ = create_mini_moe()
        model.eval()
        p_before = _params(model)
        rep = prune_diagnose(model, data=_tok())
        self.assertIsInstance(rep, DiagnosisReport)
        # on a toy MoE it should detect dense + moe targets
        self.assertTrue(rep.any_domain_detected)
        self.assertGreater(rep.targets["dense"], 0)
        self.assertGreater(rep.targets["moe"], 0)
        # fixed-ratio pruning actually takes effect and can forward
        self.assertTrue(rep.prune_works)
        self.assertGreater(rep.prune_reduction, 0.0)
        self.assertTrue(rep.prune_forward_ok)
        # search check completed (bool; whether it selects under random weights is not required)
        self.assertIsInstance(rep.search_works, bool)
        self.assertIsInstance(rep.summary(), str)
        # dry-run doesn't change the original model
        self.assertEqual(_params(model), p_before)

    def test_diagnose_unsupported_model(self):
        # no prunable structure (Embedding only) -> 0 targets, pruning ineffective, reported honestly
        model = nn.Sequential(nn.Embedding(50, 8))
        rep = prune_diagnose(model, data=None)
        self.assertFalse(rep.any_domain_detected)
        self.assertFalse(rep.prune_works)
        self.assertTrue(any("in any domain" in n for n in rep.notes))


if __name__ == "__main__":
    sys.exit(unittest.main(verbosity=2))
