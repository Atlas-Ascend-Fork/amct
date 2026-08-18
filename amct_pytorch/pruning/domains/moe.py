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
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ...common.utils.log import LOGGER

from .. import simulate
from ..config import PruneConfig
from ..utils import (
    get_submodule,
    iter_direct_named_modules,
    prune_linear_out_features,
    replace_submodule,
)
from .base import BasePruningDomain


@dataclass
class MoETarget:
    module_path: str
    router_path: Optional[str]
    experts_path: str
    fused: bool = False


def _topk_from_scores(
    scores: torch.Tensor, top_k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    flat = scores.detach().float().reshape(-1, scores.shape[-1]).cpu()
    probs = torch.softmax(flat, dim=-1)
    k = min(top_k, flat.shape[-1])
    wt, idx = torch.topk(probs, k, dim=-1)
    return idx, wt


def route_topk_from_output(output, num_experts: int, top_k: int):
    """Parse ``(topk_idx[N,k], topk_wt[N,k])`` from a router's forward output, or ``(None, None)``."""
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
        if p.dim() >= 2:
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


_EXPERT_COUNT_ATTRS = (
    "num_experts",
    "n_experts",
    "num_local_experts",
    "n_routed_experts",
)
_EXPERT_GROUP_ATTRS = ("n_group", "n_groups", "num_expert_group", "topk_group")
_TOPK_ATTRS = (
    "top_k",
    "topk",
    "num_experts_per_tok",
    "moe_top_k",
    "num_selected_experts",
)
DEFAULT_TOP_K = 8


def read_positive_int(obj: object, names: Tuple[str, ...]) -> Optional[int]:
    """First attribute in ``names`` that reads back as a positive int, else None."""
    if obj is None:
        return None
    for name in names:
        value = getattr(obj, name, None)
        if value is None or isinstance(value, bool):
            continue
        try:
            ivalue = int(value)
        except (TypeError, ValueError):
            continue
        if ivalue > 0:
            return ivalue
    return None


def _expert_slice_axis(module: nn.Module, name: str, dims: List[int]) -> Optional[int]:
    """Pick the expert axis when several dims equal orig_n; None if it cannot be disambiguated."""
    if len(dims) == 1:
        return dims[0]
    if isinstance(module, nn.Linear) and name == "weight":
        return 0
    low = name.lower()
    if any(k in low for k in ("bias", "correction", "score", "stat")):
        return dims[-1]
    return None


def _slice_router_experts(router: nn.Module, keep: torch.Tensor, orig_n: int) -> None:
    """Recursively slice per-expert tensors in the router subtree (the dim equal to orig_n) by keep."""
    for m in router.modules():
        for name, p in list(m.named_parameters(recurse=False)):
            dims = [d for d in range(p.dim()) if p.shape[d] == orig_n]
            if not dims:
                continue
            ax = _expert_slice_axis(m, name, dims)
            if ax is None:
                LOGGER.logw(
                    f"[moe] router param '{name}' has multiple dims == {orig_n}, "
                    f"cannot determine the expert axis, not sliced "
                    f"(router may be inconsistent with the expert count).",
                    "amct_prune",
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
                LOGGER.logw(
                    f"[moe] router buffer '{name}' has multiple dims == {orig_n}, "
                    f"cannot determine the expert axis, not sliced.",
                    "amct_prune",
                )
                continue
            m.register_buffer(name, b.index_select(ax, keep.to(b.device)).clone())
        if isinstance(m, nn.Linear) and m.weight.shape[0] != m.out_features:
            m.out_features = m.weight.shape[0]


def _set_expert_count(mod: nn.Module, attrs: tuple, n: int) -> None:
    """Set several "expert count" attrs on mod to n (tolerant)."""
    for attr in attrs:
        if hasattr(mod, attr):
            try:
                setattr(mod, attr, n)
            except (TypeError, ValueError):
                pass


def _collapse_expert_groups(mod: nn.Module) -> None:
    """Collapse grouped routing's group count to 1 (tolerant; existing attrs only)."""
    for attr in _EXPERT_GROUP_ATTRS:
        if hasattr(mod, attr):
            try:
                setattr(mod, attr, 1)
            except (TypeError, ValueError):
                pass


def _sync_expert_count(mod: nn.Module, orig_n: int, new_n: int) -> None:
    """Change every integer "expert"-named attr on mod whose value is exactly orig_n to new_n."""
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

    def resolve_top_k(
        self, model: nn.Module, target: MoETarget, override: Optional[int] = None
    ) -> int:
        """Experts-per-token as this model actually routes it.

        Looked up on the router, then the MoE block, then ``model.config``,
        under every spelling in use (``top_k`` / ``num_experts_per_tok`` / ...).
        A hardcoded value misestimates per-expert activation counts on models
        that do not route 8 per token -- Mixtral routes 2, others route 6 --
        which in turn skews which experts a mass/merge criterion decides to
        drop. ``override`` (the ``top_k`` kwarg) still wins when given.
        """
        if override is not None:
            return int(override)
        module = (
            get_submodule(model, target.module_path) if target.module_path else model
        )
        holders = (
            self.get_router(model, target),
            module,
            getattr(model, "config", None),
        )
        for holder in holders:
            found = read_positive_int(holder, _TOPK_ATTRS)
            if found is not None:
                return found
        LOGGER.logw(
            f"[amct.prune] could not read experts-per-token for "
            f"'{target.module_path or target.experts_path}'; falling back to "
            f"top_k={DEFAULT_TOP_K}. Pass top_k=... if the model routes a different number.",
            "amct_prune",
        )
        return DEFAULT_TOP_K

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
        self,
        model: nn.Module,
        target: MoETarget,
        keep_idx: List[int],
        orig_n: Optional[int] = None,
    ) -> None:
        if target.router_path is None:
            return
        router = get_submodule(model, target.router_path)
        if isinstance(router, nn.Linear):
            replace_submodule(
                model, target.router_path, prune_linear_out_features(router, keep_idx)
            )
            return
        # _find_router also accepts a bare weight holder and a wrapper around an inner
        # Linear; those must be sliced too, or the router keeps emitting logits for
        # experts that no longer exist and the next forward indexes out of range.
        if orig_n is None:
            return
        _slice_router_experts(
            router, torch.as_tensor(keep_idx, dtype=torch.long), orig_n
        )

    def prune_experts(
        self, model: nn.Module, target: MoETarget, keep_idx: List[int], top_k: int
    ) -> None:
        """Land the prune: actually trim experts by keep_idx (ModuleList rebuild / fused tensor slice)."""
        session = simulate.active()
        if session is not None:
            session.record_moe(model, target, self.num_experts(model, target), keep_idx)
            return
        if target.fused:
            self._prune_fused(model, target, keep_idx)
        else:
            experts = self.get_experts(model, target)
            orig_n = len(experts)
            new_experts = nn.ModuleList([experts[i] for i in keep_idx])
            self.replace_experts(model, target, new_experts)
            self.prune_router(model, target, keep_idx, orig_n)
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
        block = (
            get_submodule(model, target.module_path) if target.module_path else model
        )
        if block is not experts:
            _slice_expert_dim(block, keep, orig_n)
            _sync_expert_count(block, orig_n, new_n)
        for cname, child in iter_direct_named_modules(block):
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
        """Constrain the fused router: slice expert-dim weights/buffers + sync expert count/groups."""
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
        """True if ``name`` marks a shared (always-on, non-router-gated) expert module."""
        low = name.lower()
        return any(s in low for s in self.SHARED_EXPERT_NAMES)

    def _find_expert_container(
        self, module: nn.Module
    ) -> Optional[Tuple[str, int, bool]]:
        for name in self.EXPERT_CONTAINER_NAMES:
            if self._is_shared_expert_name(name):
                continue
            child = getattr(module, name, None)
            if isinstance(child, nn.ModuleList):
                return name, len(child), False
            if isinstance(child, nn.Module):
                n = _fused_expert_count(child)
                if n is not None and n > 1:
                    return name, n, True
        return self._find_fused_experts_fallback(module)

    def _find_fused_experts_fallback(
        self, module: nn.Module
    ) -> Optional[Tuple[str, int, bool]]:
        """Fallback: detect fused experts (3D batched weight) outside an experts/* container,
        requiring a router sibling."""
        has_router = any(
            getattr(module, rn, None) is not None for rn in self.ROUTER_NAMES
        )
        if not has_router:
            return None
        for name, child in iter_direct_named_modules(module):
            if name in self.ROUTER_NAMES or self._is_shared_expert_name(name):
                continue
            for p in child.parameters(recurse=False):
                if p.dim() >= 3 and p.shape[0] > 1:
                    return name, int(p.shape[0]), True
        return None

    def _find_router(
        self, module: nn.Module, num_experts: int
    ) -> Optional[Tuple[str, nn.Module]]:
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
            if isinstance(child, nn.Module) and any(
                isinstance(inner, nn.Linear) and inner.out_features == num_experts
                for _, inner in iter_direct_named_modules(child)
            ):
                return name, child
        for name, child in iter_direct_named_modules(module):
            if isinstance(child, nn.Linear) and child.out_features == num_experts:
                return name, child
        return None
