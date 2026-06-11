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
"""Diff-coverage tests for the dense/CNN pruning *method* modules.

Targets the guard / edge / recovery branches in
``prune_op/{dense_low_variance,dense_reconstruct,cnn_variance,cnn_reconstruct}.py``
that the comprehensive suite happy-paths skip: empty-data warn-skip, wrong-domain
type errors, prune_ratio<=0 no-op, keep-all (no prune) short-circuit, the
``recovery in {ls, bias, none}`` post-prune compensation arms, underdetermined-LS
fallbacks, the tuple/fused/non-Linear hook helper guards, and quant-aware saliency.

Tiny CPU tensors only; no HF downloads, no NPU.
"""

import os
import sys
import unittest
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from amct_pytorch.pruning import prune, get_last_report
from amct_pytorch.pruning.config import MethodSpec, PruneConfig
from amct_pytorch.pruning.context import PruneContext
from amct_pytorch.pruning.report import PruneReport
from amct_pytorch.pruning.domains.dense import (
    DensePruningDomain,
    GatedDenseTarget,
    TwoLayerDenseTarget,
)
from amct_pytorch.pruning.domains.cnn import CNNPruningDomain, CNNChannelTarget
from amct_pytorch.pruning.utils import RunningVariance

from amct_pytorch.pruning.prune_op import dense_low_variance as dlv
from amct_pytorch.pruning.prune_op import cnn_variance as cv
from amct_pytorch.pruning.prune_op import dense_reconstruct as dr
from amct_pytorch.pruning.prune_op import cnn_reconstruct as cr
from amct_pytorch.pruning.prune_op.dense_low_variance import (
    LowVarianceDensePruningMethod,
    _make_hook as dlv_make_hook,
    _register_two_layer_hook as dlv_register_two_layer_hook,
    _register_fused_hook as dlv_register_fused_hook,
    _register_gated_hooks as dlv_register_gated_hooks,
    _score_for_target as dlv_score_for_target,
    _prune_targets as dlv_prune_targets,
)
from amct_pytorch.pruning.prune_op.dense_reconstruct import (
    ReconstructDensePruningMethod,
    _fake_quant_amct as dr_fake_quant_amct,
    _reconstruct_one_target as dr_reconstruct_one_target,
)
from amct_pytorch.pruning.prune_op.cnn_variance import (
    VarianceChannelPruningMethod,
    _make_hook as cv_make_hook,
    _register_variance_hooks as cv_register_variance_hooks,
    _prune_targets_from_stats as cv_prune_targets_from_stats,
)
from amct_pytorch.pruning.prune_op.cnn_reconstruct import (
    ReconstructChannelPruningMethod,
    _channel_saliency as cv_recon_channel_saliency,
    _reconstruct_conv_consumer as cr_reconstruct_conv_consumer,
    _conv_bias_fold_delta as cr_conv_bias_fold_delta,
    _prune_and_reconstruct as cr_prune_and_reconstruct,
)

torch.set_num_threads(2)


# ---------------------------------------------------------------------------
# Toy models
# ---------------------------------------------------------------------------
class TwoLayerFFN(nn.Module):
    """Plain producer->consumer dense FFN (two-layer dense target)."""

    def __init__(self, hidden=16, inter=40):
        super().__init__()
        self.fc1 = nn.Linear(hidden, inter)
        self.fc2 = nn.Linear(inter, hidden)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class GatedFFN(nn.Module):
    """Llama-style gate_proj / up_proj / down_proj gated FFN (GatedDenseTarget)."""

    def __init__(self, hidden=16, inter=40):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class FusedGatedFFN(nn.Module):
    """Phi-3/GLM-4 style fused gate_up_proj (first half gate, second half up)."""

    def __init__(self, hidden=16, inter=40):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden, 2 * inter)
        self.down_proj = nn.Linear(inter, hidden)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class ConvBNConv(nn.Module):
    """producer conv -> bn -> consumer conv (sequential CNN channel target).

    cmid must exceed ``min_neurons`` (default 16) for the reconstruct path's guard
    (``total <= config.min_neurons``) not to short-circuit, so default cmid=24.
    """

    def __init__(self, cin=3, cmid=24, cout=8):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cmid, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cmid)
        self.conv2 = nn.Conv2d(cmid, cout, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        return self.conv2(F.relu(self.bn1(self.conv1(x))))


def _dense_cfg(name, **kwargs):
    return {
        "methods": {"dense": {"name": name, "kwargs": kwargs}},
        "missing_data_policy": "warn_skip",
    }


def _cnn_cfg(name, **kwargs):
    return {
        "methods": {"cnn": {"name": name, "kwargs": kwargs}},
        "missing_data_policy": "warn_skip",
    }


def _dense_calib(hidden=16, n=6):
    return [torch.randn(4, hidden) for _ in range(n)]


def _cnn_calib(cin=3, n=6):
    return [torch.randn(2, cin, 8, 8) for _ in range(n)]


# ===========================================================================
# dense_low_variance.py
# ===========================================================================
class TestDenseLowVariance(unittest.TestCase):
    def test_hook_tuple_output(self):
        # line 48: hook unwraps a tuple output (HF block returning (hidden, ...))
        rv = RunningVariance()
        hook = dlv_make_hook(rv)
        hook(nn.Identity(), (), (torch.randn(3, 5), torch.randn(3, 5)))
        self.assertGreater(rv.count, 0)

    def test_register_two_layer_hook_non_linear(self):
        # line 60: producer is not Linear -> early return, no stat registered
        m = nn.Sequential()
        m.add_module("p", nn.ReLU())
        target = TwoLayerDenseTarget(producer_path="p", consumer_path="p")
        stats, hooks = {}, []
        dlv_register_two_layer_hook(m, target, stats, hooks)
        self.assertEqual(stats, {})
        self.assertEqual(hooks, [])

    def test_register_fused_hook_non_linear(self):
        # line 69: fused gate_up is not Linear -> early return
        m = nn.Sequential()
        m.add_module("gu", nn.ReLU())

        class FakeTarget:
            gate_up_path = "gu"
            down_path = "gu"

        stats, hooks = {}, []
        dlv_register_fused_hook(m, FakeTarget(), stats, hooks)
        self.assertEqual((stats, hooks), ({}, []))

    def test_register_gated_hooks_non_linear(self):
        # line 79: gate/up not both Linear -> early return
        m = nn.Module()
        m.add_module("g", nn.ReLU())
        m.add_module("u", nn.ReLU())
        target = GatedDenseTarget(gate_path="g", up_path="u", down_path="d")
        stats, hooks = {}, []
        dlv_register_gated_hooks(m, target, stats, hooks)
        self.assertEqual((stats, hooks), ({}, []))

    def test_score_fused_count_zero(self):
        # line 115: fused gate_up RunningVariance with count==0 -> None
        rv = RunningVariance()  # never updated

        class FakeTarget:
            down_path = "blk.down"

        self.assertIsNone(dlv_score_for_target(FakeTarget(), {"blk.down": rv}))

    def test_score_fused_nonzero_returns_sum(self):
        # lines 116-119: fused path with data -> sum of the two interleaved halves
        rv = RunningVariance()
        rv.update(torch.randn(50, 8))  # inter*2 == 8

        class FakeTarget:
            down_path = "blk.down"

        scored = dlv_score_for_target(FakeTarget(), {"blk.down": rv})
        self.assertIsNotNone(scored)
        score, module = scored
        self.assertEqual(score.shape[0], 4)  # 8 // 2
        self.assertEqual(module, "blk")

    def test_score_not_tuple_returns_none(self):
        # line 121: gated stats present but neither RunningVariance nor tuple -> None
        class FakeTarget:
            down_path = "blk.down"

        self.assertIsNone(dlv_score_for_target(FakeTarget(), {"blk.down": 123}))

    def test_score_gated_count_zero(self):
        # line 124: gate_rv / up_rv tuple with count==0 -> None
        class FakeTarget:
            down_path = "blk.down"

        stats = {"blk.down": (RunningVariance(), RunningVariance())}
        self.assertIsNone(dlv_score_for_target(FakeTarget(), stats))

    def test_min_neurons_skips_target(self):
        # line 144: hidden <= min_neurons -> target skipped (no prune)
        model = TwoLayerFFN(hidden=16, inter=40).eval()
        cfg = PruneConfig(
            methods={"dense": MethodSpec("low_variance", {"prune_ratio": 0.5})},
            min_neurons=100,  # >= inter=40 -> every target skipped
        )
        p0 = sum(p.numel() for p in model.parameters())
        prune(model, cfg, data=_dense_calib())
        self.assertEqual(sum(p.numel() for p in model.parameters()), p0)

    def test_keep_all_no_prune(self):
        # line 152: topk keeps every neuron (len(keep_idx)==hidden) -> continue
        model = TwoLayerFFN(hidden=16, inter=40).eval()
        p0 = sum(p.numel() for p in model.parameters())
        # prune_ratio tiny: keep = max(min_neurons, round(40*0.998)) = 40 == hidden
        prune(model, _dense_cfg("low_variance", prune_ratio=0.002), data=_dense_calib())
        self.assertEqual(sum(p.numel() for p in model.parameters()), p0)

    def test_gated_ffn_pruned(self):
        # gated branch: _register_gated_hooks + tuple score + apply
        model = GatedFFN(hidden=16, inter=40).eval()
        p0 = sum(p.numel() for p in model.parameters())
        prune(model, _dense_cfg("low_variance", prune_ratio=0.4), data=_dense_calib())
        self.assertLess(sum(p.numel() for p in model.parameters()), p0)
        with torch.no_grad():
            out = model(torch.randn(2, 16))
        self.assertEqual(out.shape, (2, 16))

    def test_prune_ratio_zero_noop(self):
        # prune_ratio <= 0 -> apply() returns immediately (no hooks, no change)
        model = TwoLayerFFN().eval()
        p0 = sum(p.numel() for p in model.parameters())
        prune(model, _dense_cfg("low_variance", prune_ratio=0.0), data=_dense_calib())
        self.assertEqual(sum(p.numel() for p in model.parameters()), p0)

    def test_missing_data_raises(self):
        # lines 185-188: no data -> ValueError from apply()
        method = LowVarianceDensePruningMethod()
        with self.assertRaises(ValueError):
            method.apply(
                TwoLayerFFN(),
                DensePruningDomain(),
                [],
                PruneContext(data=None),
                PruneReport("pytorch", 0),
                PruneConfig(),
                MethodSpec("low_variance", {"prune_ratio": 0.5}),
            )

    def test_wrong_domain_raises(self):
        # lines 189-192: domain not a DensePruningDomain -> TypeError
        method = LowVarianceDensePruningMethod()
        with self.assertRaises(TypeError):
            method.apply(
                TwoLayerFFN(),
                CNNPruningDomain(),
                [],
                PruneContext(data=[torch.randn(2, 16)]),
                PruneReport("pytorch", 0),
                PruneConfig(),
                MethodSpec("low_variance", {"prune_ratio": 0.5}),
            )

    def test_empty_data_warn_skip(self):
        # lines 205-211 (+ _warn_skip 166-167): empty dataset -> warn + skip
        model = TwoLayerFFN().eval()
        p0 = sum(p.numel() for p in model.parameters())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            prune(model, _dense_cfg("low_variance", prune_ratio=0.5), data=[])
        self.assertEqual(sum(p.numel() for p in model.parameters()), p0)
        self.assertTrue(any("empty" in str(w.message).lower() for w in caught))

    def test_no_stats_collected_warn_skip(self):
        # lines 217-223: data present but no target collects stats -> warn + skip.
        # A model with no detectable dense FFN target yields any_collected=False.
        model = nn.Sequential(nn.Linear(16, 8))  # single Linear: no producer->consumer
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            method = LowVarianceDensePruningMethod()
            report = PruneReport("pytorch", 0)
            method.apply(
                model,
                DensePruningDomain(),
                [],  # no targets -> any_collected stays False
                PruneContext(data=[torch.randn(2, 16)]),
                report,
                PruneConfig(),
                MethodSpec("low_variance", {"prune_ratio": 0.5}),
            )
        self.assertTrue(
            any("could not collect" in str(w.message).lower() for w in caught)
        )
        self.assertTrue(any("could not collect" in m.lower() for m in report.warnings))


# ===========================================================================
# dense_reconstruct.py
# ===========================================================================
class TestDenseReconstruct(unittest.TestCase):
    def test_fake_quant_amct_fallback(self):
        # lines 95-99: AMCT quant helpers raise -> warn + per-channel int8 fallback
        w = torch.randn(8, 16)
        # quant_cfg shaped so the AMCT helper import/path raises -> fallback branch
        bad_cfg = {"weights_cfg": {"strategy": "not_a_strategy"}}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            wq = dr_fake_quant_amct(w, bad_cfg)
        self.assertEqual(wq.shape, w.shape)
        self.assertTrue(any("fallback" in str(c.message).lower() for c in caught))

    def test_recovery_ls(self):
        # lines 172 (hidden guard not tripped), 196-200 ls path on Linear consumer
        torch.manual_seed(0)
        model = TwoLayerFFN(hidden=16, inter=40).eval()
        calib = [torch.randn(8, 16) for _ in range(20)]  # rows >> kept neurons
        p0 = sum(p.numel() for p in model.parameters())
        prune(model, _dense_cfg("reconstruct", prune_ratio=0.4, recovery="ls"), data=calib)
        self.assertLess(sum(p.numel() for p in model.parameters()), p0)

    def test_recovery_bias_adds_bias(self):
        # lines 144-147 bias mean-fold prep + 203-212 bias apply (consumer already has bias)
        torch.manual_seed(0)
        model = TwoLayerFFN(hidden=16, inter=40).eval()
        before = model.fc2.bias.detach().clone()
        prune(
            model,
            _dense_cfg("reconstruct", prune_ratio=0.4, recovery="bias"),
            data=_dense_calib(),
        )
        # bias was folded -> changed
        self.assertFalse(torch.equal(before, model.fc2.bias.detach()))

    def test_recovery_bias_creates_bias_when_absent(self):
        # lines 204-211: consumer has no bias -> a new bias Parameter is created
        torch.manual_seed(0)
        model = GatedFFN(hidden=16, inter=40).eval()  # down_proj has bias=False
        self.assertIsNone(model.down_proj.bias)
        prune(
            model,
            _dense_cfg("reconstruct", prune_ratio=0.4, recovery="bias"),
            data=_dense_calib(),
        )
        self.assertIsNotNone(model.down_proj.bias)

    def test_recovery_ls_underdetermined_fallback(self):
        # lines 126-135: calib rows < kept neurons -> warn + demote ls to naive slice
        torch.manual_seed(0)
        model = TwoLayerFFN(hidden=16, inter=40).eval()
        # one tiny batch: 3 rows << kept (~24) neurons -> underdetermined
        calib = [torch.randn(3, 16)]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            prune(
                model,
                _dense_cfg("reconstruct", prune_ratio=0.4, recovery="ls"),
                data=calib,
            )
        self.assertTrue(any("ls" in str(c.message).lower() for c in caught))

    def test_quant_aware_int_path(self):
        # line 179: quant_aware (no quant_cfg) -> per-output-channel fake quant saliency
        torch.manual_seed(0)
        model = TwoLayerFFN(hidden=16, inter=40).eval()
        p0 = sum(p.numel() for p in model.parameters())
        prune(
            model,
            _dense_cfg("reconstruct", prune_ratio=0.4, quant_aware=True, quant_aware_bits=4),
            data=[torch.randn(8, 16) for _ in range(8)],
        )
        self.assertLess(sum(p.numel() for p in model.parameters()), p0)

    def test_keep_all_no_prune(self):
        # lines 184-185: topk keeps every neuron -> return before slicing
        model = TwoLayerFFN(hidden=16, inter=40).eval()
        p0 = sum(p.numel() for p in model.parameters())
        prune(model, _dense_cfg("reconstruct", prune_ratio=0.002), data=_dense_calib())
        self.assertEqual(sum(p.numel() for p in model.parameters()), p0)

    def test_missing_data_raises(self):
        # lines 291-294: no data -> ValueError
        method = ReconstructDensePruningMethod()
        with self.assertRaises(ValueError):
            method.apply(
                TwoLayerFFN(),
                DensePruningDomain(),
                [],
                PruneContext(data=None),
                PruneReport("pytorch", 0),
                PruneConfig(),
                MethodSpec("reconstruct", {"prune_ratio": 0.5}),
            )

    def test_wrong_domain_raises(self):
        # lines 295-298: wrong domain -> TypeError
        method = ReconstructDensePruningMethod()
        with self.assertRaises(TypeError):
            method.apply(
                TwoLayerFFN(),
                CNNPruningDomain(),
                [],
                PruneContext(data=[torch.randn(2, 16)]),
                PruneReport("pytorch", 0),
                PruneConfig(),
                MethodSpec("reconstruct", {"prune_ratio": 0.5}),
            )

    def test_prune_ratio_zero_noop(self):
        # lines 300-301: prune_ratio <= 0 -> return immediately
        model = TwoLayerFFN().eval()
        p0 = sum(p.numel() for p in model.parameters())
        prune(model, _dense_cfg("reconstruct", prune_ratio=0.0), data=_dense_calib())
        self.assertEqual(sum(p.numel() for p in model.parameters()), p0)

    def test_bad_recovery_raises(self):
        # lines 304-307: recovery not in RECOVERIES -> ValueError
        model = TwoLayerFFN().eval()
        with self.assertRaises(ValueError):
            prune(
                model,
                _dense_cfg("reconstruct", prune_ratio=0.5, recovery="nope"),
                data=_dense_calib(),
            )

    def test_empty_data_warn_skip(self):
        # lines 321-325: empty dataset -> warn + skip
        model = TwoLayerFFN().eval()
        p0 = sum(p.numel() for p in model.parameters())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            prune(model, _dense_cfg("reconstruct", prune_ratio=0.5), data=[])
        self.assertEqual(sum(p.numel() for p in model.parameters()), p0)
        self.assertTrue(any("empty" in str(c.message).lower() for c in caught))


# ===========================================================================
# cnn_variance.py
# ===========================================================================
class TestCNNVariance(unittest.TestCase):
    def test_hook_ignores_non_4d(self):
        # line 56: non-4D / non-tensor output -> hook returns without updating
        rv = RunningVariance()
        hook = cv_make_hook(rv)
        hook(nn.Identity(), None, torch.randn(3, 5))  # 2D -> ignored
        hook(nn.Identity(), None, "not a tensor")  # non-tensor -> ignored
        self.assertEqual(rv.count, 0)

    def test_register_hooks_skip_non_conv(self):
        # line 70: producer is not Conv2d -> continue (no stat / hook)
        m = nn.Module()
        m.add_module("p", nn.ReLU())
        # CNNChannelTarget pointing at a non-conv module
        t = CNNChannelTarget(producer_path="p", bn_path=None, consumer_path="p")
        stats, hooks = cv_register_variance_hooks(m, [t])
        self.assertEqual(stats, {})
        self.assertEqual(hooks, [])

    def test_min_channels_skips(self):
        # lines 93-94: n_out <= min_channels -> target skipped
        model = ConvBNConv(cin=3, cmid=12, cout=8).eval()
        cfg = PruneConfig(
            methods={"cnn": MethodSpec("variance_channel", {"prune_ratio": 0.5})},
            min_channels=100,  # >= 12 -> skip every target
        )
        p0 = sum(p.numel() for p in model.parameters())
        prune(model, cfg, data=_cnn_calib())
        self.assertEqual(sum(p.numel() for p in model.parameters()), p0)

    def test_keep_all_no_prune(self):
        # line 98: topk keeps every channel -> continue (no_collected stays False)
        model = ConvBNConv(cin=3, cmid=12, cout=8).eval()
        cfg = PruneConfig(
            methods={"cnn": MethodSpec("variance_channel", {"prune_ratio": 0.01})},
            min_channels=12,  # keep == n_out == 12
        )
        p0 = sum(p.numel() for p in model.parameters())
        prune(model, cfg, data=_cnn_calib())
        self.assertEqual(sum(p.numel() for p in model.parameters()), p0)

    def test_variance_prunes(self):
        # lines 145-160: full happy path through apply() + _prune_targets_from_stats
        model = ConvBNConv(cin=3, cmid=12, cout=8).eval()
        p0 = sum(p.numel() for p in model.parameters())
        prune(model, _cnn_cfg("variance_channel", prune_ratio=0.4), data=_cnn_calib())
        self.assertLess(sum(p.numel() for p in model.parameters()), p0)
        with torch.no_grad():
            out = model(torch.randn(2, 3, 8, 8))
        self.assertEqual(out.shape[1], 8)

    def test_wrong_domain_raises(self):
        # line 135: domain not CNNPruningDomain -> TypeError
        method = VarianceChannelPruningMethod()
        with self.assertRaises(TypeError):
            method.apply(
                ConvBNConv(),
                DensePruningDomain(),
                [],
                PruneContext(data=[torch.randn(2, 3, 8, 8)]),
                PruneReport("pytorch", 0),
                PruneConfig(),
                MethodSpec("variance_channel", {"prune_ratio": 0.5}),
            )

    def test_missing_data_raises(self):
        # lines 136-139: no data -> ValueError
        method = VarianceChannelPruningMethod()
        with self.assertRaises(ValueError):
            method.apply(
                ConvBNConv(),
                CNNPruningDomain(),
                [],
                PruneContext(data=None),
                PruneReport("pytorch", 0),
                PruneConfig(),
                MethodSpec("variance_channel", {"prune_ratio": 0.5}),
            )

    def test_prune_ratio_zero_noop(self):
        # lines 142-143: prune_ratio <= 0 -> return
        model = ConvBNConv().eval()
        p0 = sum(p.numel() for p in model.parameters())
        prune(model, _cnn_cfg("variance_channel", prune_ratio=0.0), data=_cnn_calib())
        self.assertEqual(sum(p.numel() for p in model.parameters()), p0)

    def test_empty_data_warn_skip(self):
        # lines 152-156: empty calibration -> warn + skip
        model = ConvBNConv().eval()
        p0 = sum(p.numel() for p in model.parameters())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            prune(model, _cnn_cfg("variance_channel", prune_ratio=0.5), data=[])
        self.assertEqual(sum(p.numel() for p in model.parameters()), p0)
        self.assertTrue(any("empty" in str(c.message).lower() for c in caught))

    def test_no_channels_pruned_records_event(self):
        # lines 161-167: any_collected False -> report records the "no channels" event
        method = VarianceChannelPruningMethod()
        report = PruneReport("pytorch", 0)
        model = ConvBNConv(cin=3, cmid=12, cout=8).eval()
        # min_channels == cmid so keep == n_out for every target -> nothing pruned
        method.apply(
            model,
            CNNPruningDomain(),
            CNNPruningDomain().find_targets(model, PruneConfig(min_channels=12)),
            PruneContext(data=_cnn_calib()),
            report,
            PruneConfig(min_channels=12),
            MethodSpec("variance_channel", {"prune_ratio": 0.01}),
        )
        self.assertTrue(any("No channels pruned" in e.detail for e in report.events))


# ===========================================================================
# cnn_reconstruct.py
# ===========================================================================
class TestCNNReconstruct(unittest.TestCase):
    def test_channel_saliency_linear_input(self):
        # lines 61-62: 2D (flattened linear) consumer input -> per-channel reshape rms
        acts = torch.randn(10, 8 * 4)  # n_channels=8, span=4
        sal = cv_recon_channel_saliency(acts, 8)
        self.assertEqual(sal.shape, (8,))

    def test_reconstruct_conv_consumer_underdetermined(self):
        # lines 82-83: rows < cols -> None (skip reconstruction)
        consumer = nn.Conv2d(2, 4, kernel_size=3, padding=1, bias=False)
        x_keep = torch.randn(1, 2, 3, 3)  # very few spatial rows
        y = torch.randn(1, 4, 3, 3)
        out = cr_reconstruct_conv_consumer(consumer, x_keep, y, ridge=1e-2)
        # rows (1*9=9) vs cols (2*3*3=18) -> underdetermined -> None
        self.assertIsNone(out)

    def test_conv_bias_fold_delta_non_conv(self):
        # lines 98-99: consumer not Conv2d -> None
        self.assertIsNone(
            cr_conv_bias_fold_delta(
                nn.Linear(4, 4), torch.randn(2, 4, 5, 5), [0, 1], 4
            )
        )

    def test_recovery_ls(self):
        # lines 170-175: ls reconstruct on conv consumer (ample calib rows).
        # cmid=24 > min_neurons(16) so the reconstruct guard does not short-circuit.
        torch.manual_seed(0)
        model = ConvBNConv(cin=3, cmid=24, cout=8).eval()
        calib = [torch.randn(4, 3, 8, 8) for _ in range(8)]
        p0 = sum(p.numel() for p in model.parameters())
        prune(model, _cnn_cfg("reconstruct", prune_ratio=0.4, recovery="ls"), data=calib)
        self.assertLess(sum(p.numel() for p in model.parameters()), p0)

    def test_recovery_bias(self):
        # lines 161-165 (bias prep) + 176-188 (bias apply on conv consumer with bias)
        torch.manual_seed(0)
        model = ConvBNConv(cin=3, cmid=24, cout=8).eval()  # conv2 has bias=True
        before = model.conv2.bias.detach().clone()
        prune(
            model,
            _cnn_cfg("reconstruct", prune_ratio=0.4, recovery="bias"),
            data=_cnn_calib(),
        )
        self.assertFalse(torch.equal(before, model.conv2.bias.detach()))

    def test_recovery_none(self):
        # line 169 (done=True via recovery=="none") + _finalize_report record path
        torch.manual_seed(0)
        model = ConvBNConv(cin=3, cmid=24, cout=8).eval()
        p0 = sum(p.numel() for p in model.parameters())
        prune(
            model, _cnn_cfg("reconstruct", prune_ratio=0.4, recovery="none"), data=_cnn_calib()
        )
        self.assertLess(sum(p.numel() for p in model.parameters()), p0)

    def test_keep_all_no_prune(self):
        # lines 157-158: topk keeps every channel -> return False (no prune)
        model = ConvBNConv(cin=3, cmid=24, cout=8).eval()
        cfg = PruneConfig(
            methods={"cnn": MethodSpec("reconstruct", {"prune_ratio": 0.01})},
            min_neurons=24,  # keep == total == 24
        )
        p0 = sum(p.numel() for p in model.parameters())
        prune(model, cfg, data=_cnn_calib())
        self.assertEqual(sum(p.numel() for p in model.parameters()), p0)

    def test_missing_data_raises(self):
        # line 239: no data -> ValueError
        method = ReconstructChannelPruningMethod()
        with self.assertRaises(ValueError):
            method.apply(
                ConvBNConv(),
                CNNPruningDomain(),
                [],
                PruneContext(data=None),
                PruneReport("pytorch", 0),
                PruneConfig(),
                MethodSpec("reconstruct", {"prune_ratio": 0.5}),
            )

    def test_wrong_domain_raises(self):
        # lines 240-243: wrong domain -> TypeError
        method = ReconstructChannelPruningMethod()
        with self.assertRaises(TypeError):
            method.apply(
                ConvBNConv(),
                DensePruningDomain(),
                [],
                PruneContext(data=[torch.randn(2, 3, 8, 8)]),
                PruneReport("pytorch", 0),
                PruneConfig(),
                MethodSpec("reconstruct", {"prune_ratio": 0.5}),
            )

    def test_prune_ratio_zero_noop(self):
        # lines 245-246: prune_ratio <= 0 -> return
        model = ConvBNConv().eval()
        p0 = sum(p.numel() for p in model.parameters())
        prune(model, _cnn_cfg("reconstruct", prune_ratio=0.0), data=_cnn_calib())
        self.assertEqual(sum(p.numel() for p in model.parameters()), p0)

    def test_bad_recovery_raises(self):
        # lines 249-252: recovery not in RECOVERIES -> ValueError
        model = ConvBNConv().eval()
        with self.assertRaises(ValueError):
            prune(
                model,
                _cnn_cfg("reconstruct", prune_ratio=0.5, recovery="nope"),
                data=_cnn_calib(),
            )

    def test_empty_data_warn_skip(self):
        # lines 264-267: empty calibration -> warn + skip
        model = ConvBNConv().eval()
        p0 = sum(p.numel() for p in model.parameters())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            prune(model, _cnn_cfg("reconstruct", prune_ratio=0.5), data=[])
        self.assertEqual(sum(p.numel() for p in model.parameters()), p0)
        self.assertTrue(any("empty" in str(c.message).lower() for c in caught))

    def test_underdetermined_ls_falls_back_with_warning(self):
        # lines 81/83 (None reconstruct) + _finalize_report warn (118-125): ls requested
        # but calib rows < keep*k*k -> naive slice + loud warning
        torch.manual_seed(0)
        model = ConvBNConv(cin=3, cmid=24, cout=8).eval()
        calib = [torch.randn(1, 3, 4, 4)]  # tiny: rows (16) << keep*3*3
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            prune(
                model,
                _cnn_cfg("reconstruct", prune_ratio=0.4, recovery="ls"),
                data=calib,
            )
        self.assertTrue(any("reconstruct" in str(c.message).lower() for c in caught))


# ===========================================================================
# Helper-level guard branches that the config-driven path filters out before
# the loop body (find_targets drops sub-min targets, so the inner ``continue``
# arms are only reachable by calling the helpers directly).
# ===========================================================================
class TestHelperGuardBranches(unittest.TestCase):
    # ---- cnn_variance._prune_targets_from_stats inner guards ----
    def test_cnn_var_stats_missing_or_empty(self):
        # line 91: rv None / count==0 -> continue
        model = ConvBNConv(cin=3, cmid=24, cout=8).eval()
        domain = CNNPruningDomain()
        target = CNNChannelTarget(
            producer_path="conv1", bn_path="bn1", consumer_path="conv2"
        )
        # stats with an empty RunningVariance (count==0) -> guard at 91
        collected = cv_prune_targets_from_stats(
            "variance_channel", model, domain, [target],
            {"conv1": RunningVariance()},
            PruneConfig(), PruneReport("pytorch", 0), 0.4,
        )
        self.assertFalse(collected)

    def test_cnn_var_n_out_le_min_channels(self):
        # line 94: n_out <= min_channels -> continue
        model = ConvBNConv(cin=3, cmid=24, cout=8).eval()
        domain = CNNPruningDomain()
        target = CNNChannelTarget(
            producer_path="conv1", bn_path="bn1", consumer_path="conv2"
        )
        rv = RunningVariance()
        rv.update(torch.randn(40, 24))  # 24 channels with data
        collected = cv_prune_targets_from_stats(
            "variance_channel", model, domain, [target], {"conv1": rv},
            PruneConfig(min_channels=24),  # n_out(24) <= min_channels(24)
            PruneReport("pytorch", 0), 0.4,
        )
        self.assertFalse(collected)

    def test_cnn_var_keep_all(self):
        # line 98: keep_idx >= n_out -> continue (prune_ratio ~ 0)
        model = ConvBNConv(cin=3, cmid=24, cout=8).eval()
        domain = CNNPruningDomain()
        target = CNNChannelTarget(
            producer_path="conv1", bn_path="bn1", consumer_path="conv2"
        )
        rv = RunningVariance()
        rv.update(torch.randn(40, 24))
        collected = cv_prune_targets_from_stats(
            "variance_channel", model, domain, [target], {"conv1": rv},
            # min_channels(4) < n_out(24) so guard 94 passes; prune_ratio ~0 keeps all
            # -> keep_idx(24) >= n_out(24) hits line 98.
            PruneConfig(min_channels=4), PruneReport("pytorch", 0), 0.0001,
        )
        self.assertFalse(collected)

    # ---- dense_low_variance._prune_targets inner guard ----
    def test_dense_lowvar_hidden_le_min_neurons(self):
        # line 144: hidden <= config.min_neurons -> continue
        model = TwoLayerFFN(hidden=16, inter=40).eval()
        domain = DensePruningDomain()
        target = TwoLayerDenseTarget(producer_path="fc1", consumer_path="fc2")
        rv = RunningVariance()
        rv.update(torch.randn(30, 40))
        last_hidden, any_collected = dlv_prune_targets(
            "low_variance", model, domain, [target], {"fc1": rv},
            PruneConfig(min_neurons=40),  # hidden(40) <= min_neurons(40)
            PruneReport("pytorch", 0), 0.5,
        )
        self.assertIsNone(last_hidden)

    # ---- dense_reconstruct._reconstruct_one_target hidden guard (line 172) ----
    def test_dense_recon_hidden_le_min_neurons(self):
        # line 172: hidden <= config.min_neurons -> return (no prune)
        model = TwoLayerFFN(hidden=16, inter=40).eval()
        domain = DensePruningDomain()
        target = TwoLayerDenseTarget(producer_path="fc1", consumer_path="fc2")
        chunks = [torch.randn(30, 40)]  # M=40
        p0 = sum(p.numel() for p in model.parameters())
        dr_reconstruct_one_target(
            model, domain, target, "fc2", chunks,
            PruneConfig(min_neurons=40),  # hidden(40) <= 40 -> return
            0.5, 1e-2, "ls", False, 8, None, PruneReport("pytorch", 0), "reconstruct",
        )
        self.assertEqual(sum(p.numel() for p in model.parameters()), p0)

    # ---- dense_reconstruct._collect_consumer_acts: cp is None (line 235) ----
    def test_dense_recon_consumer_path_none(self):
        model = TwoLayerFFN().eval()

        class _NoConsumer:  # neither down_path nor consumer_path
            pass

        acts, paths, n = getattr(dr, "_collect_consumer_acts")(
            model, [_NoConsumer()], PruneContext(data=[torch.randn(2, 16)]),
            torch.device("cpu"),
        )
        self.assertEqual(paths, {})  # cp None -> target skipped

    # ---- dense_reconstruct.apply per-target skips (lines 330, 333) ----
    def test_dense_recon_apply_skips_unpathed_and_empty(self):
        method = ReconstructDensePruningMethod()
        model = TwoLayerFFN().eval()

        class _NoConsumer:  # -> not in paths (line 330 continue)
            pass

        # a real two-layer target but feed zero calibration rows for it is hard;
        # instead drive both the unpathed-skip (330) and empty-chunks (333) arms.
        report = PruneReport("pytorch", 0)
        # monkeypatch _collect_consumer_acts so one target lands in paths with empty chunks
        real_target = TwoLayerDenseTarget(producer_path="fc1", consumer_path="fc2")

        def fake_collect(m, targets, ctx, dev):
            # real_target is pathed but has an empty chunk list (line 333),
            # _NoConsumer is unpathed (line 330)
            return ({id(real_target): []}, {id(real_target): "fc2"}, 1)

        orig = getattr(dr, "_collect_consumer_acts")
        setattr(dr, "_collect_consumer_acts", fake_collect)
        try:
            p0 = sum(p.numel() for p in model.parameters())
            method.apply(
                model, DensePruningDomain(), [_NoConsumer(), real_target],
                PruneContext(data=[torch.randn(2, 16)]), report,
                PruneConfig(), MethodSpec("reconstruct", {"prune_ratio": 0.5}),
            )
            self.assertEqual(sum(p.numel() for p in model.parameters()), p0)
        finally:
            setattr(dr, "_collect_consumer_acts", orig)

    # ---- cnn_reconstruct._prune_and_reconstruct keep-all (line 158) ----
    def test_cnn_recon_keep_all_returns_false(self):
        model = ConvBNConv(cin=3, cmid=24, cout=8).eval()
        domain = CNNPruningDomain()
        target = CNNChannelTarget(
            producer_path="conv1", bn_path="bn1", consumer_path="conv2"
        )
        x_full = torch.randn(4, 24, 8, 8)  # consumer input: 24 channels
        y = torch.randn(4, 8, 8, 8)
        pruned = cr_prune_and_reconstruct(
            model, domain, target, x_full, y,
            PruneConfig(min_neurons=4),  # below total, so guard 153 passes
            0.0001,  # keep == total -> line 158 return False
            1e-2, "none", PruneReport("pytorch", 0), "reconstruct",
        )
        self.assertFalse(pruned)

    # ---- cnn_reconstruct.apply: cache without 'x' (line 272) ----
    def test_cnn_recon_apply_skips_empty_cache(self):
        method = ReconstructChannelPruningMethod()
        model = ConvBNConv(cin=3, cmid=24, cout=8).eval()
        report = PruneReport("pytorch", 0)

        def fake_register(m, conv_targets):
            # caches with empty 'x' list -> line 272 continue
            return ({id(t): {"x": [], "y": []} for t in conv_targets}, [])

        orig = getattr(cr, "_register_consumer_hooks")
        setattr(cr, "_register_consumer_hooks", fake_register)
        try:
            p0 = sum(p.numel() for p in model.parameters())
            method.apply(
                model, CNNPruningDomain(),
                CNNPruningDomain().find_targets(model, PruneConfig()),
                PruneContext(data=_cnn_calib()), report,
                PruneConfig(), MethodSpec("reconstruct", {"prune_ratio": 0.5}),
            )
            self.assertEqual(sum(p.numel() for p in model.parameters()), p0)
        finally:
            setattr(cr, "_register_consumer_hooks", orig)


if __name__ == "__main__":
    unittest.main(verbosity=2)
