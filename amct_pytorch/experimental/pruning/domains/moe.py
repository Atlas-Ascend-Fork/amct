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
"""MoE 专家剪枝域，覆盖以下专家 / 路由结构：

- **ModuleList 专家**（旧式 / 自定义）：experts 为 ``nn.ModuleList``，router 为 ``nn.Linear``。
- **融合(批处理)专家**（现代 transformers，如 MixtralExperts / Qwen3MoeExperts）：
  experts 为单个模块，权重以专家为第 0 维的批张量（gate_up_proj/down_proj 等）；
  router 为 ``*TopKRouter``（非 Linear，但有 ``weight`` 形如 (num_experts, hidden)）。
- **兄弟双张量融合专家**：专家拆到父 MoE 块里多个并列子模块（如 GraniteMoE 的
  ``input_linear`` / ``output_linear``，均为以专家为第 0 维的 3D 批张量、不在 ``experts/*``
  容器内）；由 :meth:`_find_fused_experts_fallback` 识别、落地时扫描并切齐所有兄弟融合张量。
- **分组路由**（DeepseekV3 / Glm4Moe 风格）：router 依 ``n_routed_experts`` / ``n_group`` /
  ``topk_group`` 建分组掩码。剪枝后原分组失效，统一同步专家数(``_sync_expert_count``)并把
  分组数塌缩为 1(``_collapse_expert_groups``，= 普通 top-k)，避免分组 ``view`` 维度不匹配。
- **嵌套路由偏置**（Ernie4_5 风格）：router 子树内有 ``moe_statics.e_score_correction_bias``
  等逐专家张量（专家维不一定是第 0 维）；:meth:`_slice_router_experts` 递归切"恰好唯一一个维
  等于原专家数"的那一维。

各结构共用同一“选择保留哪些专家(keep_idx)”逻辑（由方法基于路由统计计算），
仅在 :meth:`MoEPruningDomain.prune_experts` 的落地方式上不同：融合结构沿第 0 维
切片批张量（含兄弟张量与路由侧专家张量），ModuleList 结构重建子列表。
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
# 路由输出解析（供方法的 forward hook 复用，结构无关）
# ---------------------------------------------------------------------------
def _topk_from_scores(
    scores: torch.Tensor, top_k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 仅从裸 logits 推断：topk **索引**与门控变换无关（单调 → softmax/sigmoid 同序，选择正确）；
    # 但 **权重** 在此假设 softmax 门控——sigmoid 门控（如 DeepseekV3）真实权重不同，应经元组路径
    # (real topk_wt) 传入（优先）。本回退仅供只返回裸 logits 的路由器，方差分数数值为近似。
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
        if p.dim() >= 2:  # 批张量：第 0 维为专家数
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


# 专家数属性（含分组路由的 n_routed_experts）；剪枝后都要同步，否则路由器读到陈旧数会前向崩溃。
_EXPERT_COUNT_ATTRS = (
    "num_experts",
    "n_experts",
    "num_local_experts",
    "n_routed_experts",
)
# 专家分组属性：剪掉任意专家后原分组不再成立，统一塌缩为单组（= 普通 top-k，与 ModuleList 路由的
# group_limited_topk→简单 topk 收口一致），避免 view(-1, n_group, n_routed//n_group) 维度不匹配。
_EXPERT_GROUP_ATTRS = ("n_group", "n_groups", "num_expert_group", "topk_group")


def _expert_slice_axis(module: nn.Module, name: str, dims: List[int]) -> Optional[int]:
    """多个维都等于 orig_n（方形歧义）时选出专家轴：
    Linear.weight → 第 0 维(out_features=专家数)；含 bias/correction/score/stat 的张量 → 最后一维
    （如 Ernie ``e_score_correction_bias`` (n_group, n_experts)）；都不匹配 → None(无法消歧)。"""
    if len(dims) == 1:
        return dims[0]
    if isinstance(module, nn.Linear) and name == "weight":
        return 0
    low = name.lower()
    if any(k in low for k in ("bias", "correction", "score", "stat")):
        return dims[-1]
    return None


def _slice_router_experts(router: nn.Module, keep: torch.Tensor, orig_n: int) -> None:
    """递归切片路由器子树里的逐专家张量：把维等于 orig_n 的那个维按 keep 切片。

    覆盖各种布局：``*TopKRouter.weight``(num_experts, hidden) 切第 0 维；嵌套门控 Linear.weight
    （Granite ``router.layer``）切第 0 维并同步 out_features；``moe_statics.e_score_correction_bias``
    形如 (num_experts_groups, num_experts) 切最后一维（Ernie）。多个维都等于 orig_n（如方形
    (n_experts, n_experts)）时按 ``_expert_slice_axis`` 消歧；仍无法消歧才**告警跳过**（不静默，
    否则路由器维度会与专家数静默不一致、前向越界）。
    """
    for m in router.modules():
        for name, p in list(m.named_parameters(recurse=False)):
            dims = [d for d in range(p.dim()) if p.shape[d] == orig_n]
            if not dims:
                continue
            ax = _expert_slice_axis(m, name, dims)
            if ax is None:
                warnings.warn(
                    f"[moe] router param '{name}' 有多个维 == {orig_n}，无法确定专家轴，"
                    f"未切片（路由器可能与专家数不一致）。",
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
                    f"[moe] router buffer '{name}' 有多个维 == {orig_n}，无法确定专家轴，未切片。",
                    RuntimeWarning, stacklevel=2,
                )
                continue
            m.register_buffer(name, b.index_select(ax, keep.to(b.device)).clone())
        if isinstance(m, nn.Linear) and m.weight.shape[0] != m.out_features:
            m.out_features = m.weight.shape[0]  # 与已切片的权重保持一致


def _set_expert_count(mod: nn.Module, attrs: tuple, n: int) -> None:
    """在 mod 上把若干"专家数"属性设为 n（容错）。降低 _prune_fused 的嵌套深度。"""
    for attr in attrs:
        if hasattr(mod, attr):
            try:
                setattr(mod, attr, n)
            except (TypeError, ValueError):
                pass


def _collapse_expert_groups(mod: nn.Module) -> None:
    """把分组路由的分组数塌缩为 1（剪枝后原分组失效）。仅设存在的属性，容错。"""
    for attr in _EXPERT_GROUP_ATTRS:
        if hasattr(mod, attr):
            try:
                setattr(mod, attr, 1)
            except (TypeError, ValueError):
                pass


def _sync_expert_count(mod: nn.Module, orig_n: int, new_n: int) -> None:
    """把 mod 上所有"名字含 expert、且值恰为原专家数 orig_n"的整型属性改为 new_n。

    比固定名单更稳健 —— 覆盖各架构的命名变体（num_experts / n_experts / num_local_experts /
    n_routed_experts / moe_num_experts ...）；``== orig_n`` 的守卫避免误改 ``num_experts_per_tok``
    (= top_k) 这类同样含 expert 但值不等于专家数的属性。
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
        """统一落地：按 keep_idx 真正裁剪专家（ModuleList 重建 / 融合张量切片）。"""
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
        # 兄弟融合专家张量：有些结构把专家拆到多个并列模块（如 GraniteMoeMoE 的
        # input_linear + output_linear，二者都是 ParallelExperts，权重以专家为第 0 维的 3D 批张量）。
        # 扫描父 MoE 块里"带 3D 批专家权重(dim0==orig_n)"的其它子模块并同样切片（路由器除外）。
        block = (
            get_submodule(model, target.module_path) if target.module_path else model
        )
        # 块自身的逐专家参数/缓冲（如 AfmoeSparseMoeBlock.expert_bias 形如 (num_experts,)，
        # 既不在 experts 也不在 router 里）：按专家维切片，否则前向会与已剪的专家数不匹配。
        if block is not experts:
            _slice_expert_dim(block, keep, orig_n)
            _sync_expert_count(block, orig_n, new_n)
        for cname, child in iter_direct_named_modules(block):
            if child is experts or cname in self.ROUTER_NAMES:
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
        """收口融合路由器：递归切专家维权重/缓冲（含嵌套门控 Linear / moe_statics）+ 同步专家数/分组。"""
        router = get_submodule(model, router_path)
        _slice_router_experts(router, keep, orig_n)
        _sync_expert_count(router, orig_n, new_n)
        _collapse_expert_groups(router)
        if hasattr(router, "top_k"):
            try:
                router.top_k = min(int(router.top_k), new_n)
            except (TypeError, ValueError):
                pass

    def _find_expert_container(
        self, module: nn.Module
    ) -> Optional[Tuple[str, int, bool]]:
        for name in self.EXPERT_CONTAINER_NAMES:
            child = getattr(module, name, None)
            if isinstance(child, nn.ModuleList):
                return name, len(child), False
            # 融合专家：非 ModuleList 的子模块，但带以专家为第 0 维的批参数
            if isinstance(child, nn.Module):
                n = _fused_expert_count(child)
                if n is not None and n > 1:
                    return name, n, True
        return self._find_fused_experts_fallback(module)

    def _find_fused_experts_fallback(
        self, module: nn.Module
    ) -> Optional[Tuple[str, int, bool]]:
        """惯用名未命中时，识别"专家不在 experts/* 容器里"的融合结构（如 GraniteMoeMoE 的
        input_linear/output_linear）：直接子模块带以专家为第 0 维的 **3D 批专家权重**(dim>=3)。

        约束（避免误判）：(1) 必须存在路由器样式的兄弟子模块；(2) 专家权重维度 dim>=3，从而把
        路由器的 (num_experts, hidden) 2D 权重排除在外。命中多个时取第一个为代表（落地阶段
        _prune_fused 会扫描并切其余兄弟融合张量）。
        """
        has_router = any(
            getattr(module, rn, None) is not None for rn in self.ROUTER_NAMES
        )
        if not has_router:
            return None
        for name, child in iter_direct_named_modules(module):
            if name in self.ROUTER_NAMES:
                continue
            for p in child.parameters(recurse=False):
                if p.dim() >= 3 and p.shape[0] > 1:
                    return name, int(p.shape[0]), True
        return None

    def _find_router(
        self, module: nn.Module, num_experts: int
    ) -> Optional[Tuple[str, nn.Module]]:
        # 优先按惯用名匹配 Linear 或带 (num_experts, hidden) weight 的 *TopKRouter
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
            # 嵌套门控：路由器是包装器、内部有 Linear(., num_experts)
            # （如 GraniteMoeTopKGating.layer）；返回包装器，落地阶段切其内部 Linear。
            if isinstance(child, nn.Module) and any(
                isinstance(inner, nn.Linear) and inner.out_features == num_experts
                for _, inner in iter_direct_named_modules(child)
            ):
                return name, child
        for name, child in iter_direct_named_modules(module):
            if isinstance(child, nn.Linear) and child.out_features == num_experts:
                return name, child
        return None
