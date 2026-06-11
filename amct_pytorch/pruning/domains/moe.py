# -*- coding: UTF-8 -*-
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
"""MoE expert-pruning domain, covering the following expert / router structures:

- **ModuleList experts** (legacy / custom): experts is an ``nn.ModuleList``, router is ``nn.Linear``.
- **Fused (batched) experts** (modern transformers, e.g. MixtralExperts / Qwen3MoeExperts):
  experts is a single module with weights as expert-first batched tensors (gate_up_proj/down_proj
  etc.); router is a ``*TopKRouter`` (not Linear, but has ``weight`` shaped (num_experts, hidden)).
- **Sibling two-tensor fused experts**: experts split across several parallel submodules of the
  parent MoE block (e.g. GraniteMoE ``input_linear`` / ``output_linear``, both expert-first 3D
  batched tensors, not inside an ``experts/*`` container); detected by
  :meth:`_find_fused_experts_fallback`, and all sibling fused tensors are scanned and sliced at landing.
- **Grouped routing** (DeepseekV3 / Glm4Moe style): the router builds group masks from
  ``n_routed_experts`` / ``n_group`` / ``topk_group``. After pruning the original grouping is invalid,
  so the expert count is synced (``_sync_expert_count``) and the group count collapses to 1
  (``_collapse_expert_groups``, = plain top-k), avoiding grouped ``view`` dim mismatches.
- **Nested router bias** (Ernie4_5 style): the router subtree has per-expert tensors like
  ``moe_statics.e_score_correction_bias`` (the expert dim is not necessarily dim 0);
  :meth:`_slice_router_experts` recursively slices the dim that is exactly the only one equal to the
  original expert count.

All structures share the same "which experts to keep (keep_idx)" logic (computed by the method from
routing statistics); they differ only in how :meth:`MoEPruningDomain.prune_experts` lands it: fused
structures slice batched tensors along dim 0 (including sibling tensors and router-side expert
tensors), ModuleList structures rebuild the sublist.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ..config import PruneConfig
from ..utils import (
    get_submodule,
    iter_direct_named_modules,
    prune_linear_out_features,
    replace_submodule,
)
from .base import BasePruningDomain


@dataclass(slots=True)
class MoETarget:
    module_path: str
    router_path: Optional[str]
    experts_path: str
    fused: bool = False


# ---------------------------------------------------------------------------
# Router-output parsing (reused by the methods' forward hooks, structure-agnostic)
# ---------------------------------------------------------------------------
def _topk_from_scores(
    scores: torch.Tensor, top_k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    # inferred from raw logits only: topk **indices** are invariant to the gating transform
    # (monotone -> softmax/sigmoid keep the same order, selection is correct); but the **weights**
    # here assume softmax gating -- sigmoid gating (e.g. DeepseekV3) has different real weights and
    # should come via the tuple path (real topk_wt, preferred). This fallback is only for routers
    # that return raw logits, so the variance scores are numerically approximate.
    flat = scores.detach().float().reshape(-1, scores.shape[-1]).cpu()
    probs = torch.softmax(flat, dim=-1)
    k = min(top_k, flat.shape[-1])
    wt, idx = torch.topk(probs, k, dim=-1)
    return idx, wt


def route_topk_from_output(output, num_experts: int, top_k: int):
    """Robustly parse ``(topk_idx[N,k], topk_wt[N,k])`` from a router's forward output.

    Handles: a raw logits tensor; a ``(logits, ...)`` tuple; the modern ``(logits, topk_wt,
    topk_idx)`` tuple (distinguished by dtype: integer = indices, float = weights). Returns
    ``(None, None)`` when it cannot be parsed.
    """
    if isinstance(output, (tuple, list)):
        tensors = [x for x in output if torch.is_tensor(x)]
        int_ts = [x for x in tensors if not x.is_floating_point()]
        flt_ts = [x for x in tensors if x.is_floating_point()]
        if int_ts:
            idx = int_ts[0]
            wt = next((f for f in flt_ts if f.shape == idx.shape), None)
            idx2 = idx.detach().reshape(-1, idx.shape[-1]).cpu()
            wt2 = (
                wt.detach().float().reshape(-1, wt.shape[-1]).cpu()
                if wt is not None
                else torch.ones_like(idx2, dtype=torch.float32)
            )
            return idx2, wt2
        scores = next((f for f in flt_ts if f.shape[-1] == num_experts), None)
        if scores is None and flt_ts:
            scores = flt_ts[0]
        if scores is None:
            return None, None
        return _topk_from_scores(scores, top_k)
    if torch.is_tensor(output):
        return _topk_from_scores(output, top_k)
    return None, None


def _fused_expert_count(experts: nn.Module) -> Optional[int]:
    n = getattr(experts, "num_experts", None)
    if isinstance(n, int) and n > 0:
        return n
    for p in experts.parameters(recurse=False):
        if p.dim() >= 2:  # batched tensor: dim 0 is the expert count
            return int(p.shape[0])
    return None


def _slice_expert_dim(module: nn.Module, keep: torch.Tensor, orig_n: int) -> None:
    """Slice all params/buffers in ``module`` whose dim 0 is ``orig_n`` along dim 0 by ``keep``."""
    for name, p in list(module.named_parameters(recurse=False)):
        if p.dim() >= 1 and p.shape[0] == orig_n:
            sliced = p.data.index_select(0, keep.to(p.device)).clone()
            setattr(module, name, nn.Parameter(sliced, requires_grad=p.requires_grad))
    for name, b in list(module.named_buffers(recurse=False)):
        if b is not None and b.dim() >= 1 and b.shape[0] == orig_n:
            module.register_buffer(name, b.index_select(0, keep.to(b.device)).clone())


# expert-count attrs (including grouped routing's n_routed_experts); all must be synced after
# pruning, else the router reads a stale count and the forward crashes.
_EXPERT_COUNT_ATTRS = (
    "num_experts",
    "n_experts",
    "num_local_experts",
    "n_routed_experts",
)
# expert-group attrs: after removing any expert the original grouping no longer holds, so it
# collapses to a single group (= plain top-k, matching the ModuleList router's
# group_limited_topk->simple-topk narrowing), avoiding view(-1, n_group, n_routed//n_group) mismatches.
_EXPERT_GROUP_ATTRS = ("n_group", "n_groups", "num_expert_group", "topk_group")


def _expert_slice_axis(module: nn.Module, name: str, dims: List[int]) -> Optional[int]:
    """Pick the expert axis when several dims equal orig_n (square ambiguity):
    Linear.weight -> dim 0 (out_features = expert count); tensors named bias/correction/score/stat ->
    last dim (e.g. Ernie ``e_score_correction_bias`` (n_group, n_experts)); else -> None (cannot disambiguate).
    """
    if len(dims) == 1:
        return dims[0]
    if isinstance(module, nn.Linear) and name == "weight":
        return 0
    low = name.lower()
    if any(k in low for k in ("bias", "correction", "score", "stat")):
        return dims[-1]
    return None


def _slice_router_experts(router: nn.Module, keep: torch.Tensor, orig_n: int) -> None:
    """Recursively slice per-expert tensors in the router subtree: slice the dim equal to orig_n by keep.

    Covers various layouts: ``*TopKRouter.weight`` (num_experts, hidden) sliced on dim 0; nested gating
    Linear.weight (Granite ``router.layer``) sliced on dim 0 with out_features synced;
    ``moe_statics.e_score_correction_bias`` shaped (num_experts_groups, num_experts) sliced on the last
    dim (Ernie). When several dims equal orig_n (e.g. square (n_experts, n_experts)),
    ``_expert_slice_axis`` disambiguates; only if it still cannot does it **warn and skip** (not silent,
    otherwise the router dims would silently mismatch the expert count and the forward goes out of bounds).
    """
    for m in router.modules():
        for name, p in list(m.named_parameters(recurse=False)):
            dims = [d for d in range(p.dim()) if p.shape[d] == orig_n]
            if not dims:
                continue
            ax = _expert_slice_axis(m, name, dims)
            if ax is None:
                warnings.warn(
                    f"[moe] router param '{name}' has multiple dims == {orig_n}, "
                    f"cannot determine the expert axis, not sliced "
                    f"(router may be inconsistent with the expert count).",
                    RuntimeWarning, stacklevel=2,
                )
                continue
            sliced = p.data.index_select(ax, keep.to(p.device)).clone()
            setattr(m, name, nn.Parameter(sliced, requires_grad=p.requires_grad))
        for name, b in list(m.named_buffers(recurse=False)):
            if b is None:
                continue
            dims = [d for d in range(b.dim()) if b.shape[d] == orig_n]
            if not dims:
                continue
            ax = _expert_slice_axis(m, name, dims)
            if ax is None:
                warnings.warn(
                    f"[moe] router buffer '{name}' has multiple dims == {orig_n}, "
                    f"cannot determine the expert axis, not sliced.",
                    RuntimeWarning, stacklevel=2,
                )
                continue
            m.register_buffer(name, b.index_select(ax, keep.to(b.device)).clone())
        if isinstance(m, nn.Linear) and m.weight.shape[0] != m.out_features:
            m.out_features = m.weight.shape[0]  # keep consistent with the already-sliced weight


def _set_expert_count(mod: nn.Module, attrs: tuple, n: int) -> None:
    """Set several "expert count" attrs on mod to n (tolerant). Reduces _prune_fused's nesting depth."""
    for attr in attrs:
        if hasattr(mod, attr):
            try:
                setattr(mod, attr, n)
            except (TypeError, ValueError):
                pass


def _collapse_expert_groups(mod: nn.Module) -> None:
    """Collapse grouped routing's group count to 1 (original grouping is invalid after pruning).
    Sets only existing attrs, tolerant.
    """
    for attr in _EXPERT_GROUP_ATTRS:
        if hasattr(mod, attr):
            try:
                setattr(mod, attr, 1)
            except (TypeError, ValueError):
                pass


def _sync_expert_count(mod: nn.Module, orig_n: int, new_n: int) -> None:
    """Change every integer attr on mod whose name contains "expert" and whose value is exactly orig_n to new_n.

    More robust than a fixed list -- covers naming variants across architectures (num_experts /
    n_experts / num_local_experts / n_routed_experts / moe_num_experts ...); the ``== orig_n`` guard
    avoids touching attrs like ``num_experts_per_tok`` (= top_k) that also contain "expert" but whose
    value is not the expert count.
    """
    for name, val in list(vars(mod).items()):
        if isinstance(val, int) and val == orig_n and "expert" in name.lower():
            try:
                setattr(mod, name, new_n)
            except (TypeError, ValueError):
                pass


class MoEPruningDomain(BasePruningDomain):
    name = "moe"

    ROUTER_NAMES = ("gate", "router", "gates")
    EXPERT_CONTAINER_NAMES = ("experts", "expert_layers", "moe_experts")
    # Shared experts (DeepSeek-MoE / Qwen2-MoE / Qwen3-MoE: ``shared_expert`` / ``shared_experts``,
    # config ``n_shared_experts`` / ``num_shared_experts`` / ``shared_expert_intermediate_size``) are
    # **always active** on every token -- they are NOT router-gated and carry no routing statistics.
    # Pruning them by routing mass/variance is meaningless and silently changes every token's output,
    # so any submodule whose name marks it a shared expert is excluded from routed-expert detection.
    SHARED_EXPERT_NAMES = ("shared_expert", "shared_experts")

    def find_targets(self, model: nn.Module, config: PruneConfig) -> List[MoETarget]:
        targets: List[MoETarget] = []
        for module_name, module in list(model.named_modules()):
            result = self._find_expert_container(module)
            if result is None:
                continue
            container_name, n_experts, fused = result
            if n_experts <= config.min_experts:
                continue
            router_info = self._find_router(module, n_experts)
            router_path = None
            if router_info is not None:
                router_name, _ = router_info
                router_path = (
                    f"{module_name}.{router_name}" if module_name else router_name
                )
            experts_path = (
                f"{module_name}.{container_name}" if module_name else container_name
            )
            targets.append(
                MoETarget(
                    module_path=module_name,
                    router_path=router_path,
                    experts_path=experts_path,
                    fused=fused,
                )
            )
        return targets

    def num_experts(self, model: nn.Module, target: MoETarget) -> int:
        experts = get_submodule(model, target.experts_path)
        if target.fused:
            n = _fused_expert_count(experts)
            if n is None:
                raise TypeError(
                    f"Cannot infer expert count for fused experts '{target.experts_path}'."
                )
            return n
        if not isinstance(experts, nn.ModuleList):
            raise TypeError(
                f"Experts container '{target.experts_path}' is not a ModuleList."
            )
        return len(experts)

    def get_router(self, model: nn.Module, target: MoETarget):
        if target.router_path is None:
            return None
        return get_submodule(model, target.router_path)

    def get_experts(self, model: nn.Module, target: MoETarget) -> nn.ModuleList:
        experts = get_submodule(model, target.experts_path)
        if not isinstance(experts, nn.ModuleList):
            raise TypeError(
                f"Experts container '{target.experts_path}' is not a ModuleList."
            )
        return experts

    def replace_experts(
        self, model: nn.Module, target: MoETarget, experts: nn.ModuleList
    ) -> None:
        replace_submodule(model, target.experts_path, experts)

    def prune_router(
        self, model: nn.Module, target: MoETarget, keep_idx: List[int]
    ) -> None:
        if target.router_path is None:
            return
        router = get_submodule(model, target.router_path)
        if isinstance(router, nn.Linear):
            replace_submodule(
                model, target.router_path, prune_linear_out_features(router, keep_idx)
            )

    def prune_experts(
        self, model: nn.Module, target: MoETarget, keep_idx: List[int], top_k: int
    ) -> None:
        """Land the prune: actually trim experts by keep_idx (ModuleList rebuild / fused tensor slice)."""
        if target.fused:
            self._prune_fused(model, target, keep_idx)
        else:
            experts = self.get_experts(model, target)
            new_experts = nn.ModuleList([experts[i] for i in keep_idx])
            self.replace_experts(model, target, new_experts)
            self.prune_router(model, target, keep_idx)
        self.update_moe_attributes(model, target, len(keep_idx))

    def update_moe_attributes(
        self, model: nn.Module, target: MoETarget, new_num_experts: int
    ) -> None:
        module = (
            get_submodule(model, target.module_path) if target.module_path else model
        )
        _set_expert_count(module, _EXPERT_COUNT_ATTRS, int(new_num_experts))
        _collapse_expert_groups(module)
        if hasattr(module, "top_k"):
            try:
                module.top_k = min(int(module.top_k), int(new_num_experts))
            except (TypeError, ValueError):
                pass

    def _prune_fused(
        self, model: nn.Module, target: MoETarget, keep_idx: List[int]
    ) -> None:
        experts = get_submodule(model, target.experts_path)
        orig_n = self.num_experts(model, target)
        keep = torch.as_tensor(keep_idx, dtype=torch.long)
        new_n = len(keep_idx)
        _slice_expert_dim(experts, keep, orig_n)
        _sync_expert_count(experts, orig_n, new_n)
        # sibling fused expert tensors: some structures split experts across several parallel modules
        # (e.g. GraniteMoeMoE input_linear + output_linear, both ParallelExperts, weights as expert-first
        # 3D batched tensors). Scan the parent MoE block for other submodules with a 3D batched expert
        # weight (dim0 == orig_n) and slice them too (excluding the router).
        block = (
            get_submodule(model, target.module_path) if target.module_path else model
        )
        # the block's own per-expert params/buffers (e.g. AfmoeSparseMoeBlock.expert_bias shaped
        # (num_experts,), in neither experts nor router): slice on the expert dim, else the forward
        # would mismatch the already-pruned expert count.
        if block is not experts:
            _slice_expert_dim(block, keep, orig_n)
            _sync_expert_count(block, orig_n, new_n)
        for cname, child in iter_direct_named_modules(block):
            # skip experts itself, the router, and any shared expert (never a routed sibling tensor)
            if (
                child is experts
                or cname in self.ROUTER_NAMES
                or self._is_shared_expert_name(cname)
            ):
                continue
            if any(
                p.dim() >= 3 and p.shape[0] == orig_n
                for p in child.parameters(recurse=False)
            ):
                _slice_expert_dim(child, keep, orig_n)
                _sync_expert_count(child, orig_n, new_n)
        if target.router_path is None:
            return
        self._prune_fused_router(model, target.router_path, keep, orig_n, new_n)

    def _prune_fused_router(
        self,
        model: nn.Module,
        router_path: str,
        keep: torch.Tensor,
        orig_n: int,
        new_n: int,
    ) -> None:
        """Constrain the fused router: recursively slice expert-dim weights/buffers (incl. nested
        gating Linear / moe_statics) + sync expert count/groups.
        """
        router = get_submodule(model, router_path)
        _slice_router_experts(router, keep, orig_n)
        _sync_expert_count(router, orig_n, new_n)
        _collapse_expert_groups(router)
        if hasattr(router, "top_k"):
            try:
                router.top_k = min(int(router.top_k), new_n)
            except (TypeError, ValueError):
                pass

    def _is_shared_expert_name(self, name: str) -> bool:
        """True if ``name`` marks a *shared* (always-on, non-router-gated) expert module.

        Shared experts run on every token and have no routing statistics, so they must never be
        treated as a routed-expert container nor scanned as a sibling fused-expert tensor. Matching
        is on the leaf attr name (e.g. ``shared_expert`` / ``shared_experts``); plain ``experts`` is
        intentionally not matched, so OLMoE/granite/Mixtral-style blocks are unaffected.
        """
        low = name.lower()
        return any(s in low for s in self.SHARED_EXPERT_NAMES)

    def _find_expert_container(
        self, module: nn.Module
    ) -> Optional[Tuple[str, int, bool]]:
        for name in self.EXPERT_CONTAINER_NAMES:
            # never treat a shared expert (always-active, non-gated) as a routed-expert container
            if self._is_shared_expert_name(name):
                continue
            child = getattr(module, name, None)
            if isinstance(child, nn.ModuleList):
                return name, len(child), False
            # fused experts: a non-ModuleList submodule, but with expert-first batched params
            if isinstance(child, nn.Module):
                n = _fused_expert_count(child)
                if n is not None and n > 1:
                    return name, n, True
        return self._find_fused_experts_fallback(module)

    def _find_fused_experts_fallback(
        self, module: nn.Module
    ) -> Optional[Tuple[str, int, bool]]:
        """When common names miss, detect fused structures where experts are not in an experts/*
        container (e.g. GraniteMoeMoE input_linear/output_linear): a direct submodule carrying an
        expert-first **3D batched expert weight** (dim >= 3).

        Constraints (to avoid false positives): (1) a router-style sibling submodule must exist;
        (2) the expert weight has dim >= 3, excluding the router's (num_experts, hidden) 2D weight.
        On multiple hits the first is taken as representative (_prune_fused scans and slices the
        remaining sibling fused tensors at landing).
        """
        has_router = any(
            getattr(module, rn, None) is not None for rn in self.ROUTER_NAMES
        )
        if not has_router:
            return None
        for name, child in iter_direct_named_modules(module):
            # skip the router and any shared expert (always-active, not router-gated)
            if name in self.ROUTER_NAMES or self._is_shared_expert_name(name):
                continue
            for p in child.parameters(recurse=False):
                if p.dim() >= 3 and p.shape[0] > 1:
                    return name, int(p.shape[0]), True
        return None

    def _find_router(
        self, module: nn.Module, num_experts: int
    ) -> Optional[Tuple[str, nn.Module]]:
        # prefer matching by common name: a Linear or a *TopKRouter with (num_experts, hidden) weight
        for name in self.ROUTER_NAMES:
            child = getattr(module, name, None)
            if child is None:
                continue
            if isinstance(child, nn.Linear) and child.out_features == num_experts:
                return name, child
            weight = getattr(child, "weight", None)
            if (
                isinstance(weight, torch.Tensor)
                and weight.dim() >= 1
                and weight.shape[0] == num_experts
            ):
                return name, child
            # nested gating: the router is a wrapper with an inner Linear(., num_experts)
            # (e.g. GraniteMoeTopKGating.layer); return the wrapper, slice its inner Linear at landing.
            if isinstance(child, nn.Module) and any(
                isinstance(inner, nn.Linear) and inner.out_features == num_experts
                for _, inner in iter_direct_named_modules(child)
            ):
                return name, child
        for name, child in iter_direct_named_modules(module):
            if isinstance(child, nn.Linear) and child.out_features == num_experts:
                return name, child
        return None
