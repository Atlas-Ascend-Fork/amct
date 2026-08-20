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

from typing import Callable, Dict, List, Optional, Tuple, TypeVar

import torch
import torch.nn as nn

from .. import simulate
from ..domains.moe import (
    DEFAULT_TOP_K,
    MoEPruningDomain,
    MoETarget,
    read_positive_int,
)
from ..utils import get_submodule, record_prune_width

Accumulator = TypeVar("Accumulator")


def register_router_hooks(
    model: nn.Module,
    domain: MoEPruningDomain,
    targets: List[MoETarget],
    top_k: Optional[int],
    make_accumulator: Callable[
        [MoEPruningDomain, nn.Module, MoETarget, int], "Accumulator"
    ],
    make_hook: Callable[["Accumulator", int], Callable],
) -> Tuple[Dict[str, "Accumulator"], List["torch.utils.hooks.RemovableHandle"]]:
    """Build a per-router accumulator and register a forward hook for each routed target."""
    accumulators: Dict[str, "Accumulator"] = {}
    hooks: List["torch.utils.hooks.RemovableHandle"] = []
    for target in targets:
        if target.router_path is None:
            continue
        router = get_submodule(model, target.router_path)
        layer_k = domain.resolve_top_k(model, target, top_k)
        accumulator = make_accumulator(domain, model, target, layer_k)
        accumulators[target.router_path] = accumulator
        hooks.append(router.register_forward_hook(make_hook(accumulator, layer_k)))
    return accumulators, hooks


def patch_router_module(
    model: nn.Module,
    target: MoETarget,
    keep_idx: List[int],
    top_k: Optional[int],
) -> None:
    """Constrain the router to the kept experts (narrow topk + slice expert_bias to kept)."""
    moe_module = (
        get_submodule(model, target.module_path) if target.module_path else model
    )
    router = None
    for name in MoEPruningDomain.ROUTER_NAMES:
        candidate = getattr(moe_module, name, None)
        if candidate is not None:
            router = candidate
            break
    if router is None:
        return
    if hasattr(router, "group_limited_topk"):

        def _simple_topk(scores, _router=router, _k=top_k):
            router_k = read_positive_int(_router, ("top_k",))
            k = router_k if router_k is not None else (_k or DEFAULT_TOP_K)
            k = min(int(k), scores.shape[-1])
            return torch.topk(scores, k=k, dim=-1)

        router.group_limited_topk = _simple_topk
    if hasattr(router, "top_k"):
        try:
            router.top_k = min(int(router.top_k), len(keep_idx))
        except (TypeError, ValueError):
            pass
    if getattr(router, "expert_bias", None) is not None:
        # Keep the learned per-expert routing-correction bias for the surviving
        # experts. Slice it down only if the domain prune step has not already
        # done so (fused routers slice it; ModuleList routers leave it full).
        expert_bias = router.expert_bias
        if expert_bias.shape[0] != len(keep_idx):
            idx = torch.as_tensor(keep_idx, device=expert_bias.device, dtype=torch.long)
            kept_bias = expert_bias.data.index_select(0, idx)
            if isinstance(expert_bias, torch.nn.Parameter):
                router.expert_bias = torch.nn.Parameter(
                    kept_bias, requires_grad=expert_bias.requires_grad
                )
            else:
                router.expert_bias = kept_bias


def prune_one_layer(
    model: nn.Module,
    domain: MoEPruningDomain,
    target: MoETarget,
    keep_idx: List[int],
    top_k: Optional[int],
) -> None:
    """Land the prune (domain dispatches ModuleList rebuild / fused slice) + patch the router."""
    layer_k = domain.resolve_top_k(model, target, top_k)
    domain.prune_experts(model, target, keep_idx, layer_k)
    if simulate.active() is not None:
        # The session's router hook already routes the dropped experts away, and this
        # patch is exactly what must NOT land on the caller's model during a trial:
        # it lowers router.top_k, swaps group_limited_topk and reslices expert_bias,
        # none of which a later trial or the final model could recover from.
        return
    patch_router_module(model, target, keep_idx, layer_k)


def prune_moe_layers(
    method_name: str,
    model: nn.Module,
    domain: MoEPruningDomain,
    targets: List[MoETarget],
    report,
    top_k: Optional[int],
    select_fn: Callable,
) -> Optional[int]:
    """Shared per-layer MoE prune loop; select_fn(idx, target) -> (keep_idx, num_experts, message)."""
    last_num_experts: Optional[int] = None
    for idx, target in enumerate(targets):
        keep_idx, num_experts, message = select_fn(idx, target)
        if keep_idx is None:
            continue
        prune_one_layer(model, domain, target, keep_idx, top_k)
        path = target.module_path or target.experts_path
        report.add(domain.name, method_name, path, message)
        report.record_layer_sparsity(path, num_experts, len(keep_idx))
        last_num_experts = len(keep_idx)
        record_prune_width(model, "_moe_widths", len(keep_idx))
    return last_num_experts
