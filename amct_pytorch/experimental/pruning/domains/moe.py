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
"""MoE 专家剪枝域，同时支持两类专家结构：

- **ModuleList 专家**（旧式 / 自定义）：experts 为 ``nn.ModuleList``，router 为 ``nn.Linear``。
- **融合(批处理)专家**（现代 transformers，如 MixtralExperts / Qwen3MoeExperts）：
  experts 为单个模块，权重以专家为第 0 维的批张量（gate_up_proj/down_proj 等）；
  router 为 ``*TopKRouter``（非 Linear，但有 ``weight`` 形如 (num_experts, hidden)）。

两类共用同一“选择保留哪些专家(keep_idx)”逻辑（由方法基于路由统计计算），
仅在 :meth:`MoEPruningDomain.prune_experts` 的落地方式上不同：融合结构沿第 0 维
切片批张量，ModuleList 结构重建子列表。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ..config import PruneConfig
from ..utils import get_submodule, iter_direct_named_modules, prune_linear_out_features, replace_submodule
from .base import BasePruningDomain


@dataclass(slots=True)
class MoETarget:
    module_path: str
    router_path: Optional[str]
    experts_path: str
    fused: bool = False


# ---------------------------------------------------------------------------
# 路由输出解析（供方法的 forward hook 复用，结构无关）
# ---------------------------------------------------------------------------
def _topk_from_scores(scores: torch.Tensor, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    flat = scores.detach().float().reshape(-1, scores.shape[-1]).cpu()
    probs = torch.softmax(flat, dim=-1)
    k = min(top_k, flat.shape[-1])
    wt, idx = torch.topk(probs, k, dim=-1)
    return idx, wt


def route_topk_from_output(output, num_experts: int, top_k: int):
    """从 router 的 forward 输出中稳健地解析 ``(topk_idx[N,k], topk_wt[N,k])``。

    兼容：纯 logits 张量；``(logits, ...)`` 元组；现代 ``(logits, topk_wt, topk_idx)``
    元组（按 dtype 区分：整型=索引，浮点=权重）。无法解析时返回 ``(None, None)``。
    """
    if isinstance(output, (tuple, list)):
        tensors = [x for x in output if torch.is_tensor(x)]
        int_ts = [x for x in tensors if not x.is_floating_point()]
        flt_ts = [x for x in tensors if x.is_floating_point()]
        if int_ts:
            idx = int_ts[0]
            wt = next((f for f in flt_ts if f.shape == idx.shape), None)
            idx2 = idx.detach().reshape(-1, idx.shape[-1]).cpu()
            wt2 = (wt.detach().float().reshape(-1, wt.shape[-1]).cpu()
                   if wt is not None else torch.ones_like(idx2, dtype=torch.float32))
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
        if p.dim() >= 2:                      # 批张量：第 0 维为专家数
            return int(p.shape[0])
    return None


def _slice_expert_dim(module: nn.Module, keep: torch.Tensor, orig_n: int) -> None:
    """把 ``module`` 中所有第 0 维为 ``orig_n`` 的参数/缓冲沿第 0 维按 ``keep`` 切片。"""
    for name, p in list(module.named_parameters(recurse=False)):
        if p.dim() >= 1 and p.shape[0] == orig_n:
            sliced = p.data.index_select(0, keep.to(p.device)).clone()
            setattr(module, name, nn.Parameter(sliced, requires_grad=p.requires_grad))
    for name, b in list(module.named_buffers(recurse=False)):
        if b is not None and b.dim() >= 1 and b.shape[0] == orig_n:
            module.register_buffer(name, b.index_select(0, keep.to(b.device)).clone())


class MoEPruningDomain(BasePruningDomain):
    name = "moe"

    ROUTER_NAMES = ("gate", "router", "gates")
    EXPERT_CONTAINER_NAMES = ("experts", "expert_layers", "moe_experts")

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
                router_path = f"{module_name}.{router_name}" if module_name else router_name
            experts_path = f"{module_name}.{container_name}" if module_name else container_name
            targets.append(MoETarget(module_path=module_name, router_path=router_path,
                                     experts_path=experts_path, fused=fused))
        return targets

    def num_experts(self, model: nn.Module, target: MoETarget) -> int:
        experts = get_submodule(model, target.experts_path)
        if target.fused:
            n = _fused_expert_count(experts)
            if n is None:
                raise TypeError(f"Cannot infer expert count for fused experts '{target.experts_path}'.")
            return n
        if not isinstance(experts, nn.ModuleList):
            raise TypeError(f"Experts container '{target.experts_path}' is not a ModuleList.")
        return len(experts)

    def get_router(self, model: nn.Module, target: MoETarget):
        if target.router_path is None:
            return None
        return get_submodule(model, target.router_path)

    def get_experts(self, model: nn.Module, target: MoETarget) -> nn.ModuleList:
        experts = get_submodule(model, target.experts_path)
        if not isinstance(experts, nn.ModuleList):
            raise TypeError(f"Experts container '{target.experts_path}' is not a ModuleList.")
        return experts

    def replace_experts(self, model: nn.Module, target: MoETarget, experts: nn.ModuleList) -> None:
        replace_submodule(model, target.experts_path, experts)

    def prune_router(self, model: nn.Module, target: MoETarget, keep_idx: List[int]) -> None:
        if target.router_path is None:
            return
        router = get_submodule(model, target.router_path)
        if isinstance(router, nn.Linear):
            replace_submodule(model, target.router_path, prune_linear_out_features(router, keep_idx))

    def prune_experts(self, model: nn.Module, target: MoETarget, keep_idx: List[int], top_k: int) -> None:
        """统一落地：按 keep_idx 真正裁剪专家（ModuleList 重建 / 融合张量切片）。"""
        if target.fused:
            self._prune_fused(model, target, keep_idx)
        else:
            experts = self.get_experts(model, target)
            new_experts = nn.ModuleList([experts[i] for i in keep_idx])
            self.replace_experts(model, target, new_experts)
            self.prune_router(model, target, keep_idx)
        self.update_moe_attributes(model, target, len(keep_idx))

    def _prune_fused(self, model: nn.Module, target: MoETarget, keep_idx: List[int]) -> None:
        experts = get_submodule(model, target.experts_path)
        orig_n = self.num_experts(model, target)
        keep = torch.as_tensor(keep_idx, dtype=torch.long)
        _slice_expert_dim(experts, keep, orig_n)
        for attr in ("num_experts", "n_experts", "num_local_experts"):
            if hasattr(experts, attr):
                try:
                    setattr(experts, attr, len(keep_idx))
                except (TypeError, ValueError):
                    pass
        if target.router_path is not None:
            router = get_submodule(model, target.router_path)
            _slice_expert_dim(router, keep, orig_n)
            for attr in ("num_experts", "n_experts"):
                if hasattr(router, attr):
                    try:
                        setattr(router, attr, len(keep_idx))
                    except (TypeError, ValueError):
                        pass
            if hasattr(router, "top_k"):
                try:
                    router.top_k = min(int(router.top_k), len(keep_idx))
                except (TypeError, ValueError):
                    pass

    def update_moe_attributes(self, model: nn.Module, target: MoETarget, new_num_experts: int) -> None:
        module = get_submodule(model, target.module_path) if target.module_path else model
        for attr in ("num_experts", "n_experts", "num_local_experts"):
            if hasattr(module, attr):
                try:
                    setattr(module, attr, int(new_num_experts))
                except (TypeError, ValueError):
                    pass
        if hasattr(module, "top_k"):
            try:
                module.top_k = min(int(module.top_k), int(new_num_experts))
            except (TypeError, ValueError):
                pass

    def _find_expert_container(self, module: nn.Module) -> Optional[Tuple[str, int, bool]]:
        for name in self.EXPERT_CONTAINER_NAMES:
            child = getattr(module, name, None)
            if isinstance(child, nn.ModuleList):
                return name, len(child), False
            # 融合专家：非 ModuleList 的子模块，但带以专家为第 0 维的批参数
            if isinstance(child, nn.Module):
                n = _fused_expert_count(child)
                if n is not None and n > 1:
                    return name, n, True
        return None

    def _find_router(self, module: nn.Module, num_experts: int) -> Optional[Tuple[str, nn.Module]]:
        # 优先按惯用名匹配 Linear 或带 (num_experts, hidden) weight 的 *TopKRouter
        for name in self.ROUTER_NAMES:
            child = getattr(module, name, None)
            if child is None:
                continue
            if isinstance(child, nn.Linear) and child.out_features == num_experts:
                return name, child
            weight = getattr(child, "weight", None)
            if isinstance(weight, torch.Tensor) and weight.dim() >= 1 and weight.shape[0] == num_experts:
                return name, child
        for name, child in iter_direct_named_modules(module):
            if isinstance(child, nn.Linear) and child.out_features == num_experts:
                return name, child
        return None
