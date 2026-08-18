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
"""Tests for the parameterless-module transparency probe and the fused gate_up path (low_variance / reconstruct)."""

import unittest
import warnings

import torch
import torch.nn as nn

from amct_pytorch.pruning import prune
from amct_pytorch.pruning.config import PruneConfig
from amct_pytorch.pruning.domains.dense import DensePruningDomain


_HIDDEN = 32
_INTER = 48


class _GluHalver(nn.Module):
    """GLU-style parameterless halving module: halves last dim -> opaque to channel dim, must break the layer chain."""

    def forward(self, x):
        gate, up = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up


class _GluHalverMlp(nn.Module):
    """fc1 out-dim equals fc2 in-dim (48->48), but the intermediate GLU halving changes channel semantics."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(_HIDDEN, _INTER)
        self.glu = _GluHalver()
        self.fc2 = nn.Linear(_INTER, _HIDDEN)

    def forward(self, x):
        y = self.glu(self.fc1(x))
        return self.fc2(torch.cat([y, y], dim=-1))


class _TransparentAct(nn.Module):
    """Shape-preserving parameterless custom activation: the probe should deem it transparent, chain unbroken."""

    def forward(self, x):
        return torch.nn.functional.gelu(x)


class _TwoLayerActMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(_HIDDEN, _INTER)
        self.act = _TransparentAct()
        self.fc2 = nn.Linear(_INTER, _HIDDEN)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _FusedMlp(nn.Module):
    """Phi-3/GLM-4 style fused FFN: gate_up_proj is gate (first half) + up (second half)."""

    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Linear(_HIDDEN, 2 * _INTER)
        self.down_proj = nn.Linear(_INTER, _HIDDEN)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(torch.nn.functional.silu(gate) * up)


def _calib():
    return [torch.randn(8, _HIDDEN) for _ in range(20)]


def _cfg(method):
    return {
        "methods": {"dense": {"name": method, "kwargs": {"prune_ratio": 0.5}}},
        "missing_data_policy": "warn_skip",
    }


def _zero_dead_channels(model):
    """Zero dead channels (weights + bias) in both gate and up halves, so their
    output is a constant 0 with zero variance."""
    dead = list(range(_INTER // 2, _INTER))
    with torch.no_grad():
        for i in dead:
            for row in (i, _INTER + i):
                model.gate_up_proj.weight[row].zero_()
                model.gate_up_proj.bias[row].zero_()


class TestParamlessTransparencyProbe(unittest.TestCase):
    def test_glu_halver_breaks_chain_no_target(self):
        model = _GluHalverMlp().eval()
        targets = DensePruningDomain().find_targets(model, PruneConfig())
        self.assertEqual(targets, [])

    def test_glu_halver_model_not_pruned(self):
        torch.manual_seed(0)
        model = _GluHalverMlp().eval()
        p0 = sum(p.numel() for p in model.parameters())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prune(model, _cfg("low_variance"), data=_calib())
        self.assertEqual(sum(p.numel() for p in model.parameters()), p0)

    def test_shape_preserving_act_keeps_chain(self):
        model = _TwoLayerActMlp().eval()
        targets = DensePruningDomain().find_targets(model, PruneConfig())
        self.assertEqual(len(targets), 1)


class TestFusedGateUpCoverage(unittest.TestCase):
    def test_fused_low_variance_exact_on_zero_channels(self):
        ref, out = self._prune_zeroed_fused("low_variance")
        self.assertLess((out - ref).abs().max().item(), 1e-5)

    def test_fused_reconstruct_preserves_on_zero_channels(self):
        ref, out = self._prune_zeroed_fused("reconstruct")
        self.assertLess(((out - ref).norm() / ref.norm()).item(), 0.05)

    def _prune_zeroed_fused(self, method):
        torch.manual_seed(0)
        model = _FusedMlp().eval()
        _zero_dead_channels(model)
        x = torch.randn(4, _HIDDEN)
        with torch.no_grad():
            ref = model(x)
        prune(model, _cfg(method), data=_calib())
        self.assertEqual(model.down_proj.in_features, _INTER // 2)
        self.assertEqual(model.gate_up_proj.out_features, 2 * (_INTER // 2))
        with torch.no_grad():
            out = model(x)
        return ref, out


_EMBED = 24


class _OPTStyleFFNBlock(nn.Module):
    """A genuine two-layer FFN (should be detected and pruned)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(_HIDDEN, _INTER)
        self.fc2 = nn.Linear(_INTER, _HIDDEN)

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class _OPTStyleDecoder(nn.Module):
    """OPT-350m style word-embedding projections: project_out(hidden->embed) before project_in(embed->hidden).

    By registration order (project_out->project_in) they match the in->inter->in (hidden->embed->hidden) shape
    of a two-layer FFN, but in forward order they are separated by the whole decoder stack
    (project_in->layers->project_out) and are not a sequential FFN; the "intermediate dim" is really the
    vocab-bound embedding dim. Pruning them as an FFN would corrupt dimensions, so they must be excluded by name.
    """

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(40, _EMBED)
        self.project_out = nn.Linear(_HIDDEN, _EMBED, bias=False)
        self.project_in = nn.Linear(_EMBED, _HIDDEN, bias=False)
        self.layers = nn.ModuleList([_OPTStyleFFNBlock()])
        self.head = nn.Linear(_EMBED, 40, bias=False)

    def forward(self, ids):
        x = self.project_in(self.embed(ids))
        for blk in self.layers:
            x = blk(x)
        return self.head(self.project_out(x))


class TestEmbedProjectionExclusion(unittest.TestCase):
    def test_embed_projection_not_matched_as_ffn(self):
        model = _OPTStyleDecoder().eval()
        targets = DensePruningDomain().find_targets(model, PruneConfig())
        paths = set()
        for t in targets:
            paths |= {getattr(t, "producer_path", ""), getattr(t, "consumer_path", "")}
        self.assertNotIn("project_in", paths)
        self.assertNotIn("project_out", paths)
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].producer_path, "layers.0.fc1")
        self.assertEqual(targets[0].consumer_path, "layers.0.fc2")

    def test_embed_projection_model_prunes_and_runs(self):
        torch.manual_seed(0)
        model = _OPTStyleDecoder().eval()
        ids = torch.randint(0, 40, (4, 8))
        with torch.no_grad():
            ref_shape = model(ids).shape
        calib = [torch.randint(0, 40, (8, 8)) for _ in range(4)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prune(model, _cfg("reconstruct"), data=calib)
        self.assertEqual(model.project_in.in_features, _EMBED)
        self.assertEqual(model.project_in.out_features, _HIDDEN)
        self.assertEqual(model.project_out.in_features, _HIDDEN)
        self.assertEqual(model.project_out.out_features, _EMBED)
        with torch.no_grad():
            self.assertEqual(model(ids).shape, ref_shape)


class TestRouterSliceSquareDims(unittest.TestCase):
    def test_square_router_weight_slices_out_axis(self):
        from amct_pytorch.pruning.domains.moe import _slice_router_experts

        router = nn.Linear(8, 8, bias=False)
        _slice_router_experts(router, torch.tensor([0, 2, 4, 6]), orig_n=8)
        self.assertEqual(tuple(router.weight.shape), (4, 8))
        self.assertEqual(router.out_features, 4)

    def test_square_correction_bias_slices_last_axis(self):
        from amct_pytorch.pruning.domains.moe import _slice_router_experts

        mod = nn.Module()
        mod.register_buffer("e_score_correction_bias", torch.zeros(8, 8))
        _slice_router_experts(mod, torch.tensor([0, 1, 2, 3]), orig_n=8)
        self.assertEqual(tuple(mod.e_score_correction_bias.shape), (8, 4))


if __name__ == "__main__":
    unittest.main()
