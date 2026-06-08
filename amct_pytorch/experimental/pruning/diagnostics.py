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
"""剪枝功能自检（health-check）：在 *给定模型* 上实际验证剪枝是否真正可用，
避免“只识别到结构、实际剪不动”的情况被误当作支持。

回答两个问题：
 1. **固定剪枝率剪枝**在该模型上是否真的生效（参数下降且能前向）？
 2. **基于精度的二分搜索（accuracy_based_auto_prune）**在该模型上是否跑得通？

均在模型副本上 dry-run，不改动传入的模型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from torch import nn

from .config import PruneConfig
from .context import BatchAdapter
from .domains.cnn import CNNPruningDomain
from .domains.dense import DensePruningDomain
from .domains.moe import MoEPruningDomain
from .utils import count_parameters


@dataclass(slots=True)
class DiagnosisReport:
    targets: Dict[str, int]  # 每个域可剪枝目标数
    prune_works: bool  # 固定率剪枝是否真正减参
    prune_reduction: float  # 实测权重削减
    prune_forward_ok: Optional[bool]  # 剪枝后能否前向（无样本时为 None）
    search_works: bool  # 二分 acc 搜索是否跑通并选出剪枝率
    search_chosen_ratio: Optional[float]  # 搜索选中的剪枝率（None=未选出）
    notes: List[str] = field(default_factory=list)

    @property
    def any_domain_detected(self) -> bool:
        return any(v > 0 for v in self.targets.values())

    def summary(self) -> str:
        det = ", ".join(f"{k}={v}" for k, v in self.targets.items())
        lines = [
            f"[prune-diagnose] 可剪枝目标: {det}",
            f"  固定率剪枝: {'可用' if self.prune_works else '无效(0 削减)'}"
            f"（削减 {self.prune_reduction * 100:.1f}%"
            + (
                f", 前向 {'OK' if self.prune_forward_ok else '失败'}"
                if self.prune_forward_ok is not None
                else ""
            )
            + ")",
            f"  acc 二分搜索: {'可用' if self.search_works else '不可用'}"
            + (
                f"（选中剪枝率={self.search_chosen_ratio}）"
                if self.search_chosen_ratio is not None
                else ""
            ),
        ]
        for n in self.notes:
            lines.append(f"  - {n}")
        return "\n".join(lines)


def _forward_ok(
    model: nn.Module, batch: Any, batch_adapter: Optional[BatchAdapter]
) -> bool:
    import torch

    try:
        if batch_adapter is not None:
            args, kwargs = batch_adapter(batch)
        elif isinstance(batch, dict):
            args, kwargs = (), dict(batch)
        elif isinstance(batch, (tuple, list)):
            args, kwargs = tuple(batch), {}
        else:
            args, kwargs = (batch,), {}
        model.eval()
        with torch.no_grad():
            out = model(*args, **kwargs)
        logits = out.logits if hasattr(out, "logits") else out
        return bool(torch.isfinite(logits).all().item())
    except Exception:
        return False


def diagnose(
    model: nn.Module,
    *,
    data: Any = None,
    config=None,
    prune_ratio: float = 0.5,
    tolerance: float = 0.05,
    batch_adapter: Optional[BatchAdapter] = None,
) -> DiagnosisReport:
    """检查剪枝功能在 ``model`` 上是否真正可用（在副本上 dry-run，不改原模型）。

    Parameters
    ----------
    model : nn.Module
        待检查模型。
    data : Any
        校准/评估数据（方差类方法与 acc 搜索需要）。无则跳过对应检查。
    config : dict | PruneConfig | None
        剪枝配置；None 时用三域默认方法。
    prune_ratio : float
        固定率检查所用剪枝率。
    tolerance : float
        acc 搜索检查所用容差。

    Returns
    -------
    DiagnosisReport
    """
    cfg_obj = (
        config
        if isinstance(config, PruneConfig)
        else PruneConfig(**(dict(config) if isinstance(config, dict) else {}))
    )
    domains = {
        "cnn": CNNPruningDomain(),
        "dense": DensePruningDomain(),
        "moe": MoEPruningDomain(),
    }
    targets = {
        name: len(dom.find_targets(model, cfg_obj)) for name, dom in domains.items()
    }

    notes: List[str] = []
    if not any(v > 0 for v in targets.values()):
        notes.append("未在任何域识别到可剪枝目标：模型结构可能不被当前域检测支持。")

    # 1) 固定率剪枝 dry-run（副本）
    from .accuracy_based_auto_prune import (
        _config_to_dict,
        _merge_attention_skips,
        _with_ratio,
    )
    from .pruner import AutoPruner

    base_cfg = _merge_attention_skips(_config_to_dict(config))
    prune_works, prune_reduction, prune_forward_ok = False, 0.0, None
    try:
        trial_cfg = PruneConfig(**_with_ratio(base_cfg, prune_ratio))
        trial_cfg.copy_model = True
        trial_cfg.verbose = False
        pruned = AutoPruner(trial_cfg)(model, data=data, batch_adapter=batch_adapter)
        before, after = count_parameters(model), count_parameters(pruned)
        prune_reduction = 1.0 - after / before if before else 0.0
        prune_works = after < before
        if data:
            first = list(data)[0]
            prune_forward_ok = _forward_ok(pruned, first, batch_adapter)
    except Exception as exc:  # pragma: no cover - 防御
        notes.append(f"固定率剪枝 dry-run 异常: {type(exc).__name__}: {exc}")

    # 2) acc 二分搜索 dry-run（apply=False，副本内部完成）
    search_works, search_chosen = False, None
    if data is None:
        notes.append("未提供 data：跳过 acc 二分搜索检查。")
    else:
        try:
            from .accuracy_based_auto_prune import accuracy_based_auto_prune

            res = accuracy_based_auto_prune(
                model,
                config,
                data=data,
                tolerance=tolerance,
                batch_adapter=batch_adapter,
                apply=False,
            )
            search_chosen = res.chosen_ratio
            search_works = res.chosen_ratio is not None
            if not search_works:
                notes.append("acc 搜索在该容差下未选出任何剪枝率（可放宽容差）。")
        except Exception as exc:  # pragma: no cover - 防御
            notes.append(f"acc 搜索 dry-run 异常: {type(exc).__name__}: {exc}")

    return DiagnosisReport(
        targets=targets,
        prune_works=prune_works,
        prune_reduction=prune_reduction,
        prune_forward_ok=prune_forward_ok,
        search_works=search_works,
        search_chosen_ratio=search_chosen,
        notes=notes,
    )
