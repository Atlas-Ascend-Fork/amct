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
"""Tests for per-layer sensitivity allocation pruning: water-fill math, guards, default no-op."""

import copy
import logging
import os
import sys
import unittest
import warnings
from unittest import mock

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, os.path.dirname(__file__))


from mini_models import create_mini_mlp
from amct_pytorch.pruning import (
    DENSE_LOWVAR_PRUNE_CFG,
    SENSITIVITY_ALLOC_PRUNE_CFG,
    prune,
    PruneReport,
)
from amct_pytorch.pruning.allocation import (
    layer_prefixes,
    measure_layer_sensitivity,
    water_fill_ratios,
)
from amct_pytorch.pruning.calib import calib_nll
from amct_pytorch.pruning.config import PruneConfig


class _RecordingHandler(logging.Handler):
    """Capture LOGGER (logger name 'Log') records emitted during a block."""

    def __init__(self, sink):
        super().__init__(level=logging.WARNING)
        self._sink = sink

    def emit(self, record):
        self._sink.append(record)


def _calib_tokens(n=6):
    torch.manual_seed(0)
    return [torch.randint(0, 1000, (4, 20)) for _ in range(n)]


class _FFNBlock(nn.Module):
    """Two-layer FFN (dense1->gelu->dense2) recognizable by the dense domain."""

    def __init__(self, hidden, inter):
        super().__init__()
        self.dense1 = nn.Linear(hidden, inter)
        self.dense2 = nn.Linear(inter, hidden)

    def forward(self, x):
        return self.dense2(F.gelu(self.dense1(x)))


class _HeadedMLP(nn.Module):
    """Top-level head FFN outside the layers.{i} blocks: checks per-layer isolation skips out-of-layer targets."""

    def __init__(self, n_layers=3, hidden=64, inter=128):
        super().__init__()
        self.embeddings = nn.Embedding(1000, hidden)
        self.layers = nn.ModuleList([_FFNBlock(hidden, inter) for _ in range(n_layers)])
        self.head_mlp = _FFNBlock(hidden, inter)
        self.classifier = nn.Linear(hidden, 1000)

    def forward(self, input_ids):
        h = self.embeddings(input_ids)
        for layer in self.layers:
            h = layer(h)
        return self.classifier(self.head_mlp(h))


class _PlainMLP(nn.Module):
    """Plain MLP with no numbered layers/blocks/h: keys<2 must do real uniform pruning, not a no-op."""

    def __init__(self, hidden=64, inter=128):
        super().__init__()
        self.embeddings = nn.Embedding(1000, hidden)
        self.ffn = _FFNBlock(hidden, inter)
        self.classifier = nn.Linear(hidden, 1000)

    def forward(self, input_ids):
        return self.classifier(self.ffn(self.embeddings(input_ids)))


class TestWaterFill(unittest.TestCase):
    def test_sum_preserved_and_ordering(self):
        sens = [0.1, 0.3, 0.5, 0.7]
        cut = 0.5
        ratios = water_fill_ratios(sens, cut)
        self.assertAlmostEqual(sum(ratios), cut * len(sens), places=4)
        self.assertGreater(ratios[0], ratios[-1])
        for r in ratios:
            self.assertGreaterEqual(r, 0.05)
            self.assertLessEqual(r, 0.9)

    def test_floor_ceil_clipping(self):
        ratios = water_fill_ratios([0.0, 0.0, 1.0, 1.0], 0.6, floor=0.1, ceil=0.7)
        for r in ratios:
            self.assertGreaterEqual(r, 0.1 * 0.6 / 1.0 - 1e-9)
            self.assertLessEqual(r, 0.7)
        self.assertLessEqual(max(ratios), 0.7)

    def test_dict_input_and_empty(self):
        ratios = water_fill_ratios({"a": 0.2, "b": 0.4}, 0.3)
        self.assertEqual(len(ratios), 2)
        with self.assertRaises(ValueError):
            water_fill_ratios([], 0.5)

    def test_budget_preserved_under_clipping(self):
        ratios = water_fill_ratios([0.0, 1.0], 0.85)
        self.assertAlmostEqual(sum(ratios), 0.85 * 2, places=6)
        for r in ratios:
            self.assertGreaterEqual(r, 0.05 - 1e-9)
            self.assertLessEqual(r, 0.9 + 1e-9)
        self.assertGreater(ratios[0], ratios[1])

    def test_infeasible_budget_raises(self):
        with self.assertRaises(ValueError):
            water_fill_ratios([0.0, 1.0], 0.95, floor=0.05, ceil=0.9)


class TestSensitivityAllocation(unittest.TestCase):
    def test_guard_never_worse_than_uniform(self):
        torch.manual_seed(0)
        calib = _calib_tokens()
        model, _ = create_mini_mlp()
        model.eval()
        baseline = copy.deepcopy(model)

        report = PruneReport()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prune(model, SENSITIVITY_ALLOC_PRUNE_CFG, data=calib, report=report)
        self.assertIsNotNone(report)
        self.assertIn(report.allocation_choice, {"uniform", "sensitivity"})
        self.assertLess(report.params_after, report.params_before)

        from amct_pytorch.pruning.allocation import (
            _apply_layer_ratios,
            _base_config_dict,
        )

        cfg = PruneConfig(**{k: v for k, v in SENSITIVITY_ALLOC_PRUNE_CFG.items()})
        keys = layer_prefixes(baseline)
        uni = _apply_layer_ratios(
            copy.deepcopy(baseline),
            _base_config_dict(cfg),
            keys,
            [0.5] * len(keys),
            calib,
            None,
        )
        self.assertLessEqual(calib_nll(model, calib), calib_nll(uni, calib) + 1e-6)

    def test_measure_layer_sensitivity_shape(self):
        calib = _calib_tokens(4)
        model, _ = create_mini_mlp()
        model.eval()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sens = measure_layer_sensitivity(model, calib, ref_ratio=0.5)
        keys = layer_prefixes(model)
        self.assertEqual(sorted(sens), sorted(keys))
        for drop in sens.values():
            self.assertGreaterEqual(drop, 0.0)
            self.assertLessEqual(drop, 1.0)

    def test_requires_data(self):
        model, _ = create_mini_mlp()
        with self.assertRaises(ValueError):
            prune(model, SENSITIVITY_ALLOC_PRUNE_CFG, data=None)


class TestIsolationOutsideLayers(unittest.TestCase):
    def test_head_mlp_untouched(self):
        torch.manual_seed(0)
        calib = _calib_tokens(4)
        model = _HeadedMLP().eval()
        inter = model.head_mlp.dense1.out_features
        report = PruneReport()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prune(model, SENSITIVITY_ALLOC_PRUNE_CFG, data=calib, report=report)
        self.assertEqual(model.head_mlp.dense1.out_features, inter)
        self.assertEqual(model.head_mlp.dense2.in_features, inter)
        self.assertLess(model.layers[0].dense1.out_features, inter)
        self.assertLess(report.params_after, report.params_before)


class TestNoLayerUniformFallback(unittest.TestCase):
    def test_plain_mlp_actually_pruned(self):
        torch.manual_seed(0)
        calib = _calib_tokens(4)
        model = _PlainMLP().eval()
        inter = model.ffn.dense1.out_features
        report = PruneReport()
        records = []
        handler = _RecordingHandler(records)
        logger = logging.getLogger("Log")
        logger.addHandler(handler)
        try:
            prune(model, SENSITIVITY_ALLOC_PRUNE_CFG, data=calib, report=report)
        finally:
            logger.removeHandler(handler)
        self.assertTrue(any("uniform" in r.getMessage() for r in records))
        self.assertEqual(report.allocation_choice, "uniform")
        self.assertLess(report.params_after, report.params_before)
        self.assertLess(model.ffn.dense1.out_features, inter)


class TestGuardEdgeCases(unittest.TestCase):
    def test_nan_nll_warns_and_falls_back_uniform(self):
        report, messages = self._run_with_mock_nll([float("nan"), float("nan")])
        self.assertEqual(report.allocation_choice, "uniform")
        self.assertTrue(any("NaN" in m for m in messages))

    def test_sensitivity_worse_picks_uniform(self):
        report, messages = self._run_with_mock_nll([5.0, 1.0])
        self.assertEqual(report.allocation_choice, "uniform")
        self.assertTrue(any("reverting to uniform" in m for m in messages))
        self.assertLess(report.params_after, report.params_before)

    def _run_with_mock_nll(self, side_effect):
        torch.manual_seed(0)
        calib = _calib_tokens(4)
        model, _ = create_mini_mlp()
        model.eval()
        report = PruneReport()
        records = []
        handler = _RecordingHandler(records)
        logger = logging.getLogger("Log")
        logger.addHandler(handler)
        try:
            with mock.patch(
                "amct_pytorch.pruning.allocation.calib_nll",
                side_effect=side_effect,
            ):
                prune(model, SENSITIVITY_ALLOC_PRUNE_CFG, data=calib, report=report)
        finally:
            logger.removeHandler(handler)
        return report, [r.getMessage() for r in records]


class TestCalibNLLInputGuard(unittest.TestCase):
    def test_float_batch_raises(self):
        model, _ = create_mini_mlp()
        model.eval()
        with self.assertRaises(ValueError):
            calib_nll(model, [torch.randn(4, 20)])


class TestDefaultUnchanged(unittest.TestCase):
    def test_default_config_has_no_allocation(self):
        self.assertIsNone(PruneConfig().allocation)

    def test_default_prune_path_unchanged(self):
        calib = _calib_tokens(4)
        model, _ = create_mini_mlp()
        model.eval()
        report = PruneReport()
        prune(model, DENSE_LOWVAR_PRUNE_CFG, data=calib, report=report)
        self.assertIsNone(report.allocation_choice)
        self.assertLess(report.params_after, report.params_before)

    def test_allocation_validation(self):
        with self.assertRaises(ValueError):
            PruneConfig(allocation={"strategy": "bogus"}).validate()
        with self.assertRaises(ValueError):
            PruneConfig(
                allocation={"strategy": "sensitivity", "guard": "bogus"}
            ).validate()
        with self.assertRaises(ValueError):
            PruneConfig(
                allocation={
                    "strategy": "sensitivity",
                    "min_ratio": 0.8,
                    "max_ratio": 0.2,
                }
            ).validate()
        PruneConfig(allocation={"strategy": "sensitivity"}).validate()


if __name__ == "__main__":
    unittest.main()
