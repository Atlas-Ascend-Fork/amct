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
"""Tests for MoE output-space expert merging (output_merge): selector picks the best, params drop, on a deep toy MoE."""

import copy
import logging
import os
import sys
import unittest
from unittest import mock

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from mini_models import MiniMoEConfig, MiniMoEModel
from amct_pytorch.pruning import (
    MOE_OUTPUT_MERGE_PRUNE_CFG,
    prune,
    PruneReport,
)
from amct_pytorch.pruning.prune_op import moe_output_merge
from amct_pytorch.pruning.prune_op.moe_output_merge import (
    _batch_token_ids,
    _fit_gate_row,
)

VOCAB = 64


class _RecordingHandler(logging.Handler):
    """Capture LOGGER (logger name 'Log') records emitted during a block."""

    def __init__(self, sink):
        super().__init__(level=logging.WARNING)
        self._sink = sink

    def emit(self, record):
        self._sink.append(record)


def _toy_config():
    return MiniMoEConfig(
        hidden_size=32,
        intermediate_size=48,
        num_experts=6,
        num_experts_per_tok=2,
        num_hidden_layers=6,
        vocab_size=VOCAB,
    )


def _toy_model(seed):
    torch.manual_seed(seed)
    return MiniMoEModel(_toy_config()).eval()


def _calib_tokens():
    torch.manual_seed(123)
    return [torch.randint(0, VOCAB, (2, 16)) for _ in range(6)]


def _num_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def _calib_nll(model, calib):
    total = 0.0
    for batch in calib:
        logits = model(batch)
        total += float(
            nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1)
            )
        )
    return total / len(calib)


class _FusedExperts(nn.Module):
    """Modern fused-expert tensor structure (experts batched on dim 0)."""

    def __init__(self, num_experts=8, hidden=16):
        super().__init__()
        self.num_experts = num_experts
        self.weight = nn.Parameter(torch.randn(num_experts, hidden, hidden))

    def forward(self, x):
        return x


class _FusedMoEModel(nn.Module):
    def __init__(self, num_experts=8, hidden=16):
        super().__init__()
        self.embeddings = nn.Embedding(VOCAB, hidden)
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        self.experts = _FusedExperts(num_experts, hidden)
        self.classifier = nn.Linear(hidden, VOCAB)

    def forward(self, input_ids):
        return self.classifier(self.experts(self.embeddings(input_ids)))


class TestOutputMergeSelector(unittest.TestCase):
    def test_selector_reduces_params_and_never_worse(self):
        calib = _calib_tokens()
        merged = _toy_model(0)
        report = PruneReport()
        prune(merged, MOE_OUTPUT_MERGE_PRUNE_CFG, data=calib, report=report)
        self.assertLess(report.params_after, report.params_before)
        events = [e.detail for e in report.events if "calib_nll selector" in e.detail]
        self.assertEqual(len(events), 1)
        self.assertIn("selected", events[0])

        act = _toy_model(0)
        act_cfg = {
            "methods": {
                "moe": {"name": "activation_count", "kwargs": {"prune_ratio": 0.5}}
            },
            "missing_data_policy": "warn_skip",
        }
        prune(act, act_cfg, data=calib)
        for layer_m, layer_a in zip(merged.layers, act.layers):
            self.assertEqual(len(layer_m.experts), len(layer_a.experts))
            self.assertEqual(layer_m.gate.out_features, layer_a.gate.out_features)
        self.assertLessEqual(_calib_nll(merged, calib), _calib_nll(act, calib) + 1e-6)


class TestOutputMergeOnly(unittest.TestCase):
    def test_merge_only_prunes_and_forwards(self):
        model = _toy_model(1)
        cfg = {
            "methods": {
                "moe": {
                    "name": "output_merge",
                    "kwargs": {"keep_ratio": 0.5, "selector": "none"},
                }
            },
            "missing_data_policy": "warn_skip",
        }
        report = PruneReport()
        prune(model, cfg, data=_calib_tokens(), report=report)
        self.assertLess(report.params_after, report.params_before)
        for layer in model.layers:
            self.assertEqual(len(layer.experts), 3)
            self.assertEqual(layer.gate.out_features, 3)
        details = [e.detail for e in report.events]
        self.assertTrue(any("Output-merge experts 6 -> 3" in d for d in details))
        with torch.no_grad():
            logits = model(torch.randint(0, VOCAB, (2, 8)))
        self.assertEqual(logits.shape, (2, 8, VOCAB))
        self.assertTrue(torch.isfinite(logits).all())


class TestOutputMergeFusedSkip(unittest.TestCase):
    def test_fused_target_loud_skip(self):
        torch.manual_seed(2)
        model = _FusedMoEModel().eval()
        before = _num_params(model)
        report = PruneReport()
        log_records = []
        handler = _RecordingHandler(log_records)
        logger = logging.getLogger("Log")
        logger.addHandler(handler)
        try:
            prune(
                model, MOE_OUTPUT_MERGE_PRUNE_CFG, data=_calib_tokens(), report=report
            )
        finally:
            logger.removeHandler(handler)
        self.assertEqual(_num_params(model), before)
        messages = [r.getMessage() for r in log_records]
        self.assertTrue(any("skipping fused" in m for m in messages))
        self.assertTrue(any("skipping fused" in w for w in report.warnings))


class TestRouterMergeMath(unittest.TestCase):
    def test_ls_fit_beats_logaddexp(self):
        torch.manual_seed(7)
        hidden, e_n, grp = 32, 6, [0, 1]
        x = torch.randn(512, hidden)
        for use_bias in (False, True):
            gate = nn.Linear(hidden, e_n, bias=use_bias)
            probs = torch.softmax(gate(x).float(), -1)
            target = probs[:, grp].sum(1)

            gate_la = copy.deepcopy(gate)
            gate_la.weight.data[0] = torch.logaddexp(
                gate.weight.data[0], gate.weight.data[1]
            )
            la_probs, _ = self._probs_after(gate_la, x, grp)
            err_la = float((la_probs[:, 0] - target).abs().mean())

            gate_ls = copy.deepcopy(gate)
            row, bias_val = _fit_gate_row(gate, x, grp, probs)
            gate_ls.weight.data[0] = row
            if use_bias:
                self.assertIsNotNone(bias_val)
                gate_ls.bias.data[0] = bias_val
            else:
                self.assertIsNone(bias_val)
            ls_probs, _ = self._probs_after(gate_ls, x, grp)
            err_ls = float((ls_probs[:, 0] - target).abs().mean())

            self.assertLess(err_ls, 0.5 * err_la)
            if use_bias:
                self.assertLess(err_ls, 0.1 * err_la)
                self.assertLess(err_ls, 0.03)

    def _probs_after(self, gate, x, grp):
        keep = [i for i in range(gate.out_features) if i not in grp[1:]]
        return torch.softmax(gate(x).float()[:, keep], -1), keep


class TestSelectorWhitelist(unittest.TestCase):
    def test_invalid_selector_raises(self):
        model = _toy_model(4)
        cfg = {
            "methods": {
                "moe": {"name": "output_merge", "kwargs": {"selector": "bogus"}}
            },
            "missing_data_policy": "warn_skip",
        }
        with self.assertRaises(ValueError):
            prune(model, cfg, data=_calib_tokens())


class TestTokenIdExtraction(unittest.TestCase):
    def test_prefers_input_ids_never_attention_mask(self):
        ids = torch.randint(0, VOCAB, (2, 16))
        mask = torch.ones(2, 16, dtype=torch.long)
        got = _batch_token_ids((), {"attention_mask": mask, "input_ids": ids})
        self.assertIs(got, ids)
        self.assertIsNone(_batch_token_ids((), {"attention_mask": mask}))
        self.assertIs(_batch_token_ids((ids,), {"attention_mask": mask}), ids)
        self.assertIsNone(_batch_token_ids((torch.randn(2, 16),), {}))


class TestMergeMutation(unittest.TestCase):
    def test_skipping_bake_changes_outputs(self):
        calib = _calib_tokens()
        cfg = {
            "methods": {
                "moe": {
                    "name": "output_merge",
                    "kwargs": {"keep_ratio": 0.5, "selector": "none"},
                }
            },
            "missing_data_policy": "warn_skip",
        }
        baked = _toy_model(5)
        unbaked = copy.deepcopy(baked)
        prune(baked, cfg, data=calib)
        with mock.patch.object(
            moe_output_merge, "_bake_merged_weights", lambda *args: None
        ):
            prune(unbaked, cfg, data=calib)
        self.assertEqual(_num_params(baked), _num_params(unbaked))
        x = torch.randint(0, VOCAB, (2, 16), generator=torch.Generator().manual_seed(9))
        with torch.no_grad():
            self.assertFalse(torch.allclose(baked(x), unbaked(x)))


if __name__ == "__main__":
    unittest.main()
