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
"""Unit tests targeting uncovered branches of the pruning *domain* modules
(amct_pytorch/pruning/domains/{moe,cnn,base}.py).

These drive the structure-detection / topology-compat / co-resize / fused-slice
code paths directly with tiny toy modules (no HF downloads, no NPU, CPU only).
"""

import os
import sys
import warnings

import torch

torch.set_num_threads(2)

import torch.nn as nn
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from amct_pytorch.pruning.config import PruneConfig
from amct_pytorch.pruning.domains import moe as moe_mod
from amct_pytorch.pruning.domains import cnn as cnn_mod
from amct_pytorch.pruning.domains.base import BasePruningDomain
from amct_pytorch.pruning.domains.moe import (
    MoEPruningDomain,
    MoETarget,
    route_topk_from_output,
    _fused_expert_count,
    _slice_expert_dim,
    _slice_router_experts,
    _set_expert_count,
    _collapse_expert_groups,
    _sync_expert_count,
)
from amct_pytorch.pruning.domains.cnn import (
    CNNPruningDomain,
    CNNChannelTarget,
    _find_root_linear_consumer,
    _is_consumer_compat,
)


# ===========================================================================
# Toy MoE building blocks
# ===========================================================================
class _ToyExpert(nn.Module):
    def __init__(self, hidden=8, inter=12):
        super().__init__()
        self.fc1 = nn.Linear(hidden, inter)
        self.fc2 = nn.Linear(inter, hidden)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class _ToyMoEModuleList(nn.Module):
    """ModuleList experts + Linear router, with an optional shared_expert sibling."""

    def __init__(self, hidden=8, inter=12, num_experts=4, with_shared=True):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = 2
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [_ToyExpert(hidden, inter) for _ in range(num_experts)]
        )
        if with_shared:
            self.shared_expert = _ToyExpert(hidden, inter)

    def forward(self, x):
        logits = self.gate(x)
        probs = torch.softmax(logits, dim=-1)
        wt, idx = torch.topk(probs, self.top_k, dim=-1)
        out = torch.zeros_like(x)
        for e in range(len(self.experts)):
            out = out + self.experts[e](x) * 0.0  # keep cheap but shape-correct
        out = out + self.experts[0](x)
        if hasattr(self, "shared_expert"):
            out = out + self.shared_expert(x)
        return out


class _FusedExperts(nn.Module):
    """Modern fused (batched) experts: weights expert-first 3D tensors."""

    def __init__(self, num_experts=4, hidden=8, inter=12):
        super().__init__()
        self.num_experts = num_experts
        # expert-first batched weights (n, out, in) -- dim0 is expert count
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * inter, hidden))
        self.down_proj = nn.Parameter(torch.randn(num_experts, hidden, inter))
        # a per-expert buffer to exercise _slice_expert_dim buffer branch
        self.register_buffer("expert_scale", torch.ones(num_experts))


class _TopKRouter(nn.Module):
    """*TopKRouter style: not a Linear, has a (num_experts, hidden) weight."""

    def __init__(self, num_experts=4, hidden=8):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = 2
        self.weight = nn.Parameter(torch.randn(num_experts, hidden))


class _FusedMoEBlock(nn.Module):
    """Fused experts container + *TopKRouter + shared_expert + a block-level per-expert buffer."""

    def __init__(self, num_experts=4, hidden=8, inter=12):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = 2
        self.gate = _TopKRouter(num_experts, hidden)
        self.experts = _FusedExperts(num_experts, hidden, inter)
        self.shared_expert = _ToyExpert(hidden, inter)
        # block-level per-expert bias (in neither experts nor router)
        self.register_buffer("expert_bias", torch.zeros(num_experts))


class _SiblingFusedBlock(nn.Module):
    """Granite-style: experts split across sibling 3D batched submodules, no experts/* container."""

    def __init__(self, num_experts=4, hidden=8, inter=12):
        super().__init__()
        self.router = nn.Linear(hidden, num_experts, bias=False)
        self.input_linear = _ParallelExperts(num_experts, hidden, 2 * inter)
        self.output_linear = _ParallelExperts(num_experts, inter, hidden)
        self.shared_experts = _ToyExpert(hidden, inter)


class _ParallelExperts(nn.Module):
    def __init__(self, num_experts, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_experts, out_f, in_f))


class _NestedRouterBias(nn.Module):
    """Router subtree carrying a per-expert correction bias on the last dim (Ernie style)."""

    def __init__(self, num_experts=4, hidden=8, n_group=2):
        super().__init__()
        self.num_experts = num_experts
        self.weight = nn.Parameter(torch.randn(num_experts, hidden))
        # shaped (n_group, num_experts) -- expert dim is the LAST dim
        self.register_buffer(
            "e_score_correction_bias", torch.zeros(n_group, num_experts)
        )


# ===========================================================================
# moe.py : route_topk_from_output
# ===========================================================================
class TestRouteTopkFromOutput(unittest.TestCase):
    def test_tuple_with_int_indices_and_matching_weights(self):
        idx = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        wt = torch.tensor([[0.6, 0.4], [0.7, 0.3]])
        out_idx, out_wt = route_topk_from_output((idx, wt), num_experts=4, top_k=2)
        self.assertEqual(out_idx.shape, (2, 2))
        self.assertEqual(out_wt.shape, (2, 2))

    def test_tuple_with_int_indices_no_matching_weights(self):
        # int indices present but no float of same shape -> synthesised ones
        idx = torch.tensor([[0, 1]], dtype=torch.long)
        bad_wt = torch.tensor([1.0, 2.0, 3.0])  # different shape
        out_idx, out_wt = route_topk_from_output((idx, bad_wt), num_experts=4, top_k=2)
        self.assertTrue(torch.allclose(out_wt, torch.ones_like(out_wt)))

    def test_tuple_scores_matching_num_experts(self):
        # no int tensors -> float scores with last dim == num_experts (line 108)
        scores = torch.randn(3, 4)
        out_idx, out_wt = route_topk_from_output((scores,), num_experts=4, top_k=2)
        self.assertEqual(out_idx.shape[-1], 2)

    def test_tuple_scores_fallback_first_float(self):
        # float present but none with last dim == num_experts -> first float fallback (109-110)
        scores = torch.randn(3, 5)
        out_idx, out_wt = route_topk_from_output((scores,), num_experts=4, top_k=2)
        self.assertEqual(out_idx.shape[-1], 2)

    def test_tuple_no_tensors_returns_none(self):
        # tuple of non-tensors -> no floats -> (None, None) (111-112)
        out_idx, out_wt = route_topk_from_output(("a", 1), num_experts=4, top_k=2)
        self.assertIsNone(out_idx)
        self.assertIsNone(out_wt)

    def test_raw_tensor_output(self):
        scores = torch.randn(3, 4)
        out_idx, out_wt = route_topk_from_output(scores, num_experts=4, top_k=2)
        self.assertEqual(out_idx.shape[-1], 2)

    def test_non_tensor_non_tuple_returns_none(self):
        out_idx, out_wt = route_topk_from_output(None, num_experts=4, top_k=2)
        self.assertIsNone(out_idx)
        self.assertIsNone(out_wt)


# ===========================================================================
# moe.py : _fused_expert_count
# ===========================================================================
class TestFusedExpertCount(unittest.TestCase):
    def test_count_from_batched_param(self):
        # no num_experts attr -> inferred from dim0 of a >=2D param (123-125)
        m = nn.Module()
        m.register_parameter("w", nn.Parameter(torch.randn(5, 3)))
        self.assertEqual(_fused_expert_count(m), 5)

    def test_count_none_when_no_batched_param(self):
        # only 1D params -> None (126)
        m = nn.Module()
        m.register_parameter("b", nn.Parameter(torch.randn(3)))
        self.assertIsNone(_fused_expert_count(m))


# ===========================================================================
# moe.py : _slice_expert_dim buffer branch (136-137)
# ===========================================================================
class TestSliceExpertDim(unittest.TestCase):
    def test_slices_param_and_buffer(self):
        m = nn.Module()
        m.register_parameter("w", nn.Parameter(torch.randn(4, 3)))
        m.register_buffer("scale", torch.arange(4).float())
        keep = torch.tensor([0, 2], dtype=torch.long)
        _slice_expert_dim(m, keep, orig_n=4)
        self.assertEqual(m.w.shape[0], 2)
        self.assertTrue(torch.equal(m.scale, torch.tensor([0.0, 2.0])))


# ===========================================================================
# moe.py : _slice_router_experts ambiguity warn/skip + nested bias slice
# ===========================================================================
class TestSliceRouterExperts(unittest.TestCase):
    def test_ambiguous_param_axis_warns_and_skips(self):
        # square param (n, n) with a non-disambiguating name -> warn + skip (185-191)
        router = nn.Module()
        router.register_parameter("foo", nn.Parameter(torch.randn(4, 4)))
        keep = torch.tensor([0, 1], dtype=torch.long)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _slice_router_experts(router, keep, orig_n=4)
        self.assertTrue(any(issubclass(x.category, RuntimeWarning) for x in w))
        self.assertEqual(router.foo.shape, (4, 4))  # unchanged

    def test_ambiguous_buffer_axis_warns_and_skips(self):
        # square buffer (n, n), non-disambiguating name -> warn + skip (200-206)
        router = nn.Module()
        router.register_buffer("foo", torch.randn(4, 4))
        keep = torch.tensor([0, 1], dtype=torch.long)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _slice_router_experts(router, keep, orig_n=4)
        self.assertTrue(any(issubclass(x.category, RuntimeWarning) for x in w))
        self.assertEqual(router.foo.shape, (4, 4))

    def test_no_matching_dim_param_and_buffer_continue(self):
        # params/buffers with no dim == orig_n -> 'continue' branches (183, 199)
        router = nn.Module()
        router.register_parameter("p", nn.Parameter(torch.randn(2, 3)))
        router.register_buffer("b", torch.randn(2, 3))
        keep = torch.tensor([0], dtype=torch.long)
        _slice_router_experts(router, keep, orig_n=4)  # no-op, no error
        self.assertEqual(router.p.shape, (2, 3))
        self.assertEqual(router.b.shape, (2, 3))

    def test_square_linear_weight_axis0(self):
        # square (n, n) nn.Linear weight, multiple matching dims -> disambiguate to dim 0 (162)
        router = nn.Module()
        router.add_module("layer", nn.Linear(4, 4, bias=False))  # weight (4,4)
        keep = torch.tensor([0, 1], dtype=torch.long)
        _slice_router_experts(router, keep, orig_n=4)
        self.assertEqual(router.layer.weight.shape[0], 2)  # out_features axis sliced
        self.assertEqual(router.layer.out_features, 2)  # synced (208-209)

    def test_square_named_bias_buffer_last_axis(self):
        # square (n, n) buffer named with 'correction' -> last-dim expert axis (165)
        router = nn.Module()
        router.register_buffer("e_score_correction_bias", torch.randn(4, 4))
        keep = torch.tensor([0, 2], dtype=torch.long)
        _slice_router_experts(router, keep, orig_n=4)
        # only last dim sliced (dim0 left at 4); shape becomes (4, 2)
        self.assertEqual(router.e_score_correction_bias.shape, (4, 2))

    def test_nested_bias_sliced_on_last_dim(self):
        # e_score_correction_bias (n_group, n_experts): last-dim expert axis (164-165 path -> 207)
        router = _NestedRouterBias(num_experts=4, hidden=8, n_group=2)
        keep = torch.tensor([0, 2], dtype=torch.long)
        _slice_router_experts(router, keep, orig_n=4)
        self.assertEqual(router.weight.shape[0], 2)  # (n_experts, hidden) dim0
        self.assertEqual(router.e_score_correction_bias.shape, (2, 2))  # last dim sliced


# ===========================================================================
# moe.py : _set_expert_count / _collapse_expert_groups / _sync_expert_count
# TypeError/ValueError swallow branches (218-219, 230-231, 246-247)
# ===========================================================================
class _CountSetterRejects(nn.Module):
    """Module where the count attrs are read-only properties raising TypeError on set."""

    def __init__(self):
        super().__init__()
        # plain attr also present so hasattr/vars find them
        object.__setattr__(self, "_n", 4)

    @property
    def num_experts(self):
        return self._n

    @num_experts.setter
    def num_experts(self, v):
        raise TypeError("read only")

    @property
    def n_group(self):
        return 2

    @n_group.setter
    def n_group(self, v):
        raise ValueError("read only")


class TestExpertCountHelpersSwallow(unittest.TestCase):
    def test_set_expert_count_swallows(self):
        m = _CountSetterRejects()
        # should not raise (218-219)
        _set_expert_count(m, ("num_experts",), 2)

    def test_collapse_expert_groups_swallows(self):
        m = _CountSetterRejects()
        # should not raise (230-231)
        _collapse_expert_groups(m)

    def test_sync_expert_count_swallows(self):
        # _sync_expert_count iterates vars(); put an instance int attr == orig_n with
        # the word 'expert', backed by a setter that raises (246-247).
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                # store in __dict__ so vars() sees it
                self.__dict__["moe_num_experts"] = 4

            def __setattr__(self, name, value):
                if name == "moe_num_experts":
                    raise TypeError("nope")
                super().__setattr__(name, value)

        m = M()
        _sync_expert_count(m, orig_n=4, new_n=2)  # must not raise
        self.assertEqual(m.moe_num_experts, 4)  # unchanged due to swallow


# ===========================================================================
# moe.py : MoEPruningDomain method-level branches
# ===========================================================================
class TestMoEDomainMethods(unittest.TestCase):
    def setUp(self):
        self.dom = MoEPruningDomain()
        self.cfg = PruneConfig(min_experts=1)

    def test_num_experts_fused_none_raises(self):
        # fused target but expert count uninferable -> TypeError (296)
        model = nn.Module()
        bad = nn.Module()
        bad.register_parameter("b", nn.Parameter(torch.randn(3)))  # only 1D
        model.add_module("experts", bad)
        t = MoETarget(module_path="", router_path=None, experts_path="experts", fused=True)
        with self.assertRaises(TypeError):
            self.dom.num_experts(model, t)

    def test_num_experts_non_modulelist_raises(self):
        # non-fused target whose container is not a ModuleList -> TypeError (301)
        model = nn.Module()
        model.add_module("experts", nn.Linear(4, 4))
        t = MoETarget(module_path="", router_path=None, experts_path="experts", fused=False)
        with self.assertRaises(TypeError):
            self.dom.num_experts(model, t)

    def test_get_router_none_path(self):
        # router_path None -> None (308)
        t = MoETarget(module_path="", router_path=None, experts_path="e", fused=False)
        self.assertIsNone(self.dom.get_router(nn.Module(), t))

    def test_get_router_returns_submodule(self):
        # non-None router_path -> resolved submodule (309)
        model = nn.Module()
        gate = nn.Linear(8, 4, bias=False)
        model.add_module("gate", gate)
        t = MoETarget(module_path="", router_path="gate", experts_path="e", fused=False)
        self.assertIs(self.dom.get_router(model, t), gate)

    def test_get_experts_non_modulelist_raises(self):
        # 314
        model = nn.Module()
        model.add_module("experts", nn.Linear(4, 4))
        t = MoETarget(module_path="", router_path=None, experts_path="experts", fused=False)
        with self.assertRaises(TypeError):
            self.dom.get_experts(model, t)

    def test_prune_router_none_path(self):
        # router_path None -> early return (328)
        t = MoETarget(module_path="", router_path=None, experts_path="e", fused=False)
        self.dom.prune_router(nn.Module(), t, [0, 1])  # no error

    def test_update_moe_attributes_topk_swallow(self):
        # module.top_k setter raises -> swallowed (359-360)
        class Blk(nn.Module):
            @property
            def top_k(self):
                return 4

            @top_k.setter
            def top_k(self, v):
                raise ValueError("read only")

        model = nn.Module()
        model.add_module("blk", Blk())
        t = MoETarget(module_path="blk", router_path=None, experts_path="blk.e", fused=False)
        self.dom.update_moe_attributes(model, t, new_num_experts=2)  # no raise

    def test_is_shared_expert_name(self):
        self.assertTrue(self.dom._is_shared_expert_name("shared_expert"))
        self.assertTrue(self.dom._is_shared_expert_name("shared_experts"))
        self.assertFalse(self.dom._is_shared_expert_name("experts"))


# ===========================================================================
# moe.py : structure detection (find_targets / find_router / fallback) and
# the fused prune landing (shared-expert exclusion, sibling tensors, 399, 420-421)
# ===========================================================================
class TestMoEStructureDetection(unittest.TestCase):
    def setUp(self):
        self.dom = MoEPruningDomain()
        self.cfg = PruneConfig(min_experts=1)

    def test_modulelist_with_shared_expert_excluded(self):
        # shared_expert must not be detected as a routed container; experts is.
        model = nn.Module()
        model.add_module("blk", _ToyMoEModuleList(num_experts=4, with_shared=True))
        targets = self.dom.find_targets(model, self.cfg)
        self.assertEqual(len(targets), 1)
        t = targets[0]
        self.assertFalse(t.fused)
        self.assertTrue(t.experts_path.endswith(".experts"))
        self.assertTrue(t.router_path.endswith(".gate"))
        self.assertEqual(self.dom.num_experts(model, t), 4)
        # land a ModuleList prune (rebuild + router slice)
        self.dom.prune_experts(model, t, keep_idx=[0, 2], top_k=2)
        self.assertEqual(len(self.dom.get_experts(model, t)), 2)

    def test_min_experts_threshold_skips(self):
        # n_experts <= min_experts -> continue (270)
        model = nn.Module()
        model.add_module("blk", _ToyMoEModuleList(num_experts=2, with_shared=False))
        cfg = PruneConfig(min_experts=2)
        self.assertEqual(self.dom.find_targets(model, cfg), [])

    def test_fused_block_topkrouter_detection_and_prune(self):
        model = nn.Module()
        model.add_module("blk", _FusedMoEBlock(num_experts=4))
        targets = self.dom.find_targets(model, self.cfg)
        self.assertEqual(len(targets), 1)
        t = targets[0]
        self.assertTrue(t.fused)
        self.assertTrue(t.router_path.endswith(".gate"))
        self.assertEqual(self.dom.num_experts(model, t), 4)
        # land fused prune: slices experts + block buffer + router weight
        self.dom.prune_experts(model, t, keep_idx=[0, 1, 2], top_k=2)
        blk = model.blk
        self.assertEqual(blk.experts.gate_up_proj.shape[0], 3)
        self.assertEqual(blk.experts.down_proj.shape[0], 3)
        self.assertEqual(blk.experts.expert_scale.shape[0], 3)
        self.assertEqual(blk.expert_bias.shape[0], 3)  # block-level buffer co-resized
        self.assertEqual(blk.gate.weight.shape[0], 3)  # router weight sliced

    def test_sibling_fused_fallback_detection_and_prune(self):
        # Granite-style: experts not in experts/* -> _find_fused_experts_fallback (471, 475 region)
        model = nn.Module()
        model.add_module("blk", _SiblingFusedBlock(num_experts=4))
        targets = self.dom.find_targets(model, self.cfg)
        self.assertEqual(len(targets), 1)
        t = targets[0]
        self.assertTrue(t.fused)
        # the representative is input_linear or output_linear (a sibling fused tensor)
        self.assertTrue(
            t.experts_path.endswith(".input_linear")
            or t.experts_path.endswith(".output_linear")
        )
        self.dom.prune_experts(model, t, keep_idx=[0, 1, 2], top_k=2)
        # both sibling fused tensors sliced to 3
        self.assertEqual(model.blk.input_linear.weight.shape[0], 3)
        self.assertEqual(model.blk.output_linear.weight.shape[0], 3)
        # shared_experts NOT touched (not a routed sibling)
        self.assertEqual(model.blk.shared_experts.fc1.weight.shape[0], 12)

    def test_fallback_no_router_returns_none(self):
        # _find_fused_experts_fallback: no router sibling -> None (466-467)
        m = nn.Module()
        m.add_module("input_linear", _ParallelExperts(4, 8, 16))
        self.assertIsNone(self.dom._find_fused_experts_fallback(m))

    def test_fallback_router_but_no_3d_child_returns_none(self):
        # has a router sibling but no child carries a 3D batched expert weight -> None (475)
        m = nn.Module()
        m.add_module("router", nn.Linear(8, 4, bias=False))
        m.add_module("plain", nn.Linear(8, 8))  # only 2D weights
        self.assertIsNone(self.dom._find_fused_experts_fallback(m))

    def test_find_router_bare_linear_scan(self):
        # router not under a known name but a Linear(., num_experts) exists -> scan fallback (501-503)
        m = nn.Module()
        m.add_module("experts", nn.ModuleList([_ToyExpert() for _ in range(4)]))
        m.add_module("my_gate", nn.Linear(8, 4, bias=False))  # not in ROUTER_NAMES
        info = self.dom._find_router(m, num_experts=4)
        self.assertIsNotNone(info)
        self.assertEqual(info[0], "my_gate")

    def test_find_router_nested_gating_wrapper(self):
        # Granite TopKGating style: a child under a ROUTER_NAME is a wrapper whose inner
        # Linear has out_features == num_experts (496-500).
        class Gating(nn.Module):
            def __init__(self, hidden, n):
                super().__init__()
                self.layer = nn.Linear(hidden, n, bias=False)  # inner Linear

        m = nn.Module()
        m.add_module("experts", nn.ModuleList([_ToyExpert() for _ in range(4)]))
        m.add_module("gate", Gating(8, 4))  # wrapper, not a Linear itself, no .weight
        info = self.dom._find_router(m, num_experts=4)
        self.assertIsNotNone(info)
        self.assertEqual(info[0], "gate")  # the wrapper is returned

    def test_find_router_returns_none_when_absent(self):
        # no router anywhere -> None (504)
        m = nn.Module()
        m.add_module("experts", nn.ModuleList([_ToyExpert() for _ in range(4)]))
        self.assertIsNone(self.dom._find_router(m, num_experts=4))

    def test_prune_fused_router_none_early_return(self):
        # _prune_fused with router_path None -> early return at 399
        model = nn.Module()
        blk = nn.Module()
        blk.add_module("experts", _FusedExperts(num_experts=4))
        model.add_module("blk", blk)
        t = MoETarget(
            module_path="blk", router_path=None, experts_path="blk.experts", fused=True
        )
        self.dom.prune_experts(model, t, keep_idx=[0, 1], top_k=2)
        self.assertEqual(model.blk.experts.gate_up_proj.shape[0], 2)

    def test_prune_fused_router_topk_swallow(self):
        # _prune_fused_router: router.top_k setter raises -> swallowed (420-421)
        class Router(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_experts = 4
                self.weight = nn.Parameter(torch.randn(4, 8))

            @property
            def top_k(self):
                return 4

            @top_k.setter
            def top_k(self, v):
                raise TypeError("read only")

        model = nn.Module()
        blk = nn.Module()
        blk.add_module("experts", _FusedExperts(num_experts=4))
        blk.add_module("gate", Router())
        model.add_module("blk", blk)
        t = MoETarget(
            module_path="blk",
            router_path="blk.gate",
            experts_path="blk.experts",
            fused=True,
        )
        self.dom.prune_experts(model, t, keep_idx=[0, 1], top_k=2)  # no raise
        self.assertEqual(model.blk.gate.weight.shape[0], 2)


# ===========================================================================
# cnn.py : root-linear consumer, consumer-compat, target build, apply co-resize
# ===========================================================================
class _ToyCNNConvLinear(nn.Module):
    """Conv -> BN -> ReLU -> flatten -> root Linear consumer (the only Linear)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.act = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 5)  # in_features divisible by conv out_channels

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = torch.flatten(self.pool(x), 1)
        return self.fc(x)


class _ToyCNNConvConv(nn.Module):
    """Conv -> BN -> Conv (consumer)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 6, 3, padding=1, bias=True)

    def forward(self, x):
        return self.conv2(torch.relu(self.bn1(self.conv1(x))))


class TestCNNRootLinearConsumer(unittest.TestCase):
    def test_single_candidate_returned(self):
        # exactly one root Linear divisible by out_channels -> returned (86, 88)
        model = nn.Module()
        model.add_module("fc", nn.Linear(8, 5))
        name, lin = _find_root_linear_consumer(model, out_channels=8)
        self.assertEqual(name, "fc")
        self.assertIsInstance(lin, nn.Linear)

    def test_multiple_candidates_returns_none(self):
        # ambiguous: two compatible Linears -> (None, None) (89)
        model = nn.Module()
        model.add_module("fc1", nn.Linear(8, 5))
        model.add_module("fc2", nn.Linear(16, 5))  # also divisible
        name, lin = _find_root_linear_consumer(model, out_channels=8)
        self.assertIsNone(name)
        self.assertIsNone(lin)


class TestCNNConsumerCompat(unittest.TestCase):
    def setUp(self):
        self.conv = nn.Conv2d(3, 8, 3)

    def test_grouped_conv_consumer_rejected(self):
        # consumer Conv with groups != 1 -> False (96-97)
        cons = nn.Conv2d(8, 8, 3, groups=8)
        self.assertFalse(_is_consumer_compat(self.conv, None, cons))

    def test_in_channels_mismatch_rejected(self):
        # consumer Conv in_channels != producer out_channels -> False (98-99)
        cons = nn.Conv2d(4, 8, 3)
        self.assertFalse(_is_consumer_compat(self.conv, None, cons))

    def test_linear_not_divisible_rejected(self):
        # consumer Linear in_features not divisible -> False (101-102)
        cons = nn.Linear(7, 5)
        self.assertFalse(_is_consumer_compat(self.conv, None, cons))

    def test_bn_feature_mismatch_rejected(self):
        # bn.num_features != conv.out_channels -> False (103-107)
        cons = nn.Conv2d(8, 8, 3)
        bn = nn.BatchNorm2d(4)  # mismatched
        self.assertFalse(_is_consumer_compat(self.conv, bn, cons))

    def test_compatible_accept(self):
        cons = nn.Conv2d(8, 8, 3)
        bn = nn.BatchNorm2d(8)
        self.assertTrue(_is_consumer_compat(self.conv, bn, cons))


class _BlockOnlyRootLinear(nn.Module):
    """Conv lives inside a child block whose only forward neighbours are passthrough;
    the sole consumer is a root-level Linear -> exercises the root-linear fallback (153-156)."""

    def __init__(self):
        super().__init__()
        block = nn.Sequential()
        block.add_module("conv", nn.Conv2d(3, 8, 3, padding=1, bias=False))
        block.add_module("relu", nn.ReLU())  # no Conv/Linear after conv in this parent
        self.block = block
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 5)  # the only (root) Linear, divisible by 8

    def forward(self, x):
        x = self.block(x)
        x = torch.flatten(self.pool(x), 1)
        return self.fc(x)


class _IncompatBNModel(nn.Module):
    """conv1 -> BN(mismatched num_features) -> conv2: a consumer is found but compat fails (157-158)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(4)  # deliberately != conv1.out_channels (8)
        self.conv2 = nn.Conv2d(8, 6, 3, padding=1)

    def forward(self, x):
        return self.conv2(torch.relu(self.bn1(self.conv1(x))))


class TestCNNDomainApply(unittest.TestCase):
    def setUp(self):
        self.dom = CNNPruningDomain()
        self.cfg = PruneConfig(min_channels=2)

    def test_find_and_apply_conv_conv(self):
        model = _ToyCNNConvConv()
        targets = self.dom.find_targets(model, self.cfg)
        self.assertTrue(len(targets) >= 1)
        t = next(t for t in targets if t.producer_path == "conv1")
        self.assertEqual(self.dom.channel_count(model, t), 8)
        # keep 5 of 8 -> conv1.out, bn1, conv2.in all co-resize
        self.dom.apply_keep_indices(model, t, keep_idx=[0, 1, 2, 3, 4])
        self.assertEqual(model.conv1.out_channels, 5)
        self.assertEqual(model.bn1.num_features, 5)
        self.assertEqual(model.conv2.in_channels, 5)
        with torch.no_grad():
            out = model(torch.randn(2, 3, 8, 8))
        self.assertTrue(torch.isfinite(out).all())

    def test_find_and_apply_conv_linear_root_consumer(self):
        # root Linear consumer path (153-156) + Linear co-resize branch (210-220)
        model = _ToyCNNConvLinear()
        targets = self.dom.find_targets(model, self.cfg)
        t = next(t for t in targets if t.producer_path == "conv1")
        self.assertEqual(t.consumer_path, "fc")
        self.dom.apply_keep_indices(model, t, keep_idx=[0, 1, 2, 3])
        self.assertEqual(model.conv1.out_channels, 4)
        self.assertEqual(model.bn1.num_features, 4)
        self.assertEqual(model.fc.in_features, 4)
        with torch.no_grad():
            out = model(torch.randn(2, 3, 8, 8))
        self.assertTrue(torch.isfinite(out).all())

    def test_root_linear_fallback_when_no_local_consumer(self):
        # _find_consumer returns None -> _find_root_linear_consumer fallback (153-156)
        model = _BlockOnlyRootLinear()
        targets = self.dom.find_targets(model, self.cfg)
        t = next(t for t in targets if t.producer_path == "block.conv")
        self.assertEqual(t.consumer_path, "fc")
        self.dom.apply_keep_indices(model, t, keep_idx=[0, 1, 2, 3, 4])
        self.assertEqual(model.block.conv.out_channels, 5)
        self.assertEqual(model.fc.in_features, 5)
        with torch.no_grad():
            out = model(torch.randn(2, 3, 8, 8))
        self.assertTrue(torch.isfinite(out).all())

    def test_incompatible_bn_rejects_target(self):
        # consumer found but _is_consumer_compat fails on BN mismatch -> no target (157-158)
        model = _IncompatBNModel()
        targets = self.dom.find_targets(model, self.cfg)
        # conv1 must NOT yield a target (its bn is incompatible)
        self.assertFalse(any(t.producer_path == "conv1" for t in targets))

    def test_apply_producer_not_conv_raises(self):
        # 186-187
        model = nn.Module()
        model.add_module("p", nn.Linear(4, 4))
        model.add_module("c", nn.Conv2d(4, 4, 3))
        t = CNNChannelTarget(producer_path="p", bn_path=None, consumer_path="c")
        with self.assertRaises(TypeError):
            self.dom.apply_keep_indices(model, t, [0, 1])

    def test_apply_consumer_not_conv_or_linear_raises(self):
        # 188-191
        model = nn.Module()
        model.add_module("p", nn.Conv2d(3, 8, 3))
        model.add_module("c", nn.ReLU())
        t = CNNChannelTarget(producer_path="p", bn_path=None, consumer_path="c")
        with self.assertRaises(TypeError):
            self.dom.apply_keep_indices(model, t, [0, 1, 2, 3])

    def test_channel_count_producer_not_conv_raises(self):
        # 225
        model = nn.Module()
        model.add_module("p", nn.Linear(4, 4))
        t = CNNChannelTarget(producer_path="p", bn_path=None, consumer_path="p")
        with self.assertRaises(TypeError):
            self.dom.channel_count(model, t)


# ===========================================================================
# base.py : abstract find_targets NotImplementedError (line 32)
# ===========================================================================
class TestBaseDomainAbstract(unittest.TestCase):
    def test_find_targets_raises_not_implemented(self):
        class Concrete(BasePruningDomain):
            # call super().find_targets to actually execute the abstract body
            def find_targets(self, model, config):
                return super().find_targets(model, config)

        dom = Concrete()
        with self.assertRaises(NotImplementedError):
            dom.find_targets(nn.Module(), PruneConfig())


if __name__ == "__main__":
    unittest.main()
