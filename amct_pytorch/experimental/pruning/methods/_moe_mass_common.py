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
"""moe mass 类方法（activation_count / mass_variance）的共享落地助手。

两种 mass 方法选保留专家的*信号*不同，但**落地**完全一致：重建专家 + 修补路由器。
原先两文件各复制一份（字节相同），现统一到此。
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
    """剪掉被移除专家后，把路由器收口到保留专家（topk 收窄 + expert_bias 重置）。"""
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
    """统一落地（ModuleList 重建 / 融合张量切片，由域按 target.fused 分派）+ 修补路由器。"""
    domain.prune_experts(model, target, keep_idx, top_k)
    patch_router_module(model, target, keep_idx, top_k)
