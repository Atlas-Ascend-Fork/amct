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
"""Shared landing helpers for the mass-family MoE methods (activation_count / mass_variance).

The two methods differ in the *signal* used to pick kept experts, but the landing step is
identical: rebuild experts + patch the router. Unified here instead of duplicated per file.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from ..domains.moe import MoEPruningDomain, MoETarget
from ..utils import get_submodule, infer_model_device


def patch_router_module(
    model: nn.Module,
    target: MoETarget,
    keep_idx: List[int],
    top_k: int,
) -> None:
    """Constrain the router to the kept experts (narrow topk + reset expert_bias)."""
    moe_module = (
        get_submodule(model, target.module_path) if target.module_path else model
    )
    router = getattr(moe_module, "gate", None) or getattr(moe_module, "router", None)
    if router is None:
        return
    if hasattr(router, "group_limited_topk"):

        def _simple_topk(scores, _router=router, _k=top_k):
            k = min(getattr(_router, "top_k", _k), scores.shape[-1])
            return torch.topk(scores, k=k, dim=-1)

        router.group_limited_topk = _simple_topk
    if hasattr(router, "expert_bias"):
        device = infer_model_device(model)
        router.expert_bias = torch.zeros(
            len(keep_idx), device=device, dtype=router.expert_bias.dtype
        )


def prune_one_layer(
    model: nn.Module,
    domain: MoEPruningDomain,
    target: MoETarget,
    keep_idx: List[int],
    top_k: int,
) -> None:
    """Land the prune (domain dispatches ModuleList rebuild / fused slice) + patch the router."""
    domain.prune_experts(model, target, keep_idx, top_k)
    patch_router_module(model, target, keep_idx, top_k)
