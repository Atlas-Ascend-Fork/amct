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
"""守卫式 recovery-HYBRID 的 FFN / CNN 剪枝入口。

与 :func:`guarded_variance_hybrid_prune`（MoE，搜*判据*）同构，但搜的是剪枝后**补偿方式**——
对 dense FFN / CNN 通道剪枝，决定精度的杠杆是 RECOVERY（最小二乘重构 / 均值折叠 / 不补偿），
而非选神经元的判据。本入口固定显著度判据、在 recovery MENU 上各剪一份副本、用小验证集择优：

    MENU = {none(朴素切片, fallback), bias(均值折叠, 无最小二乘), ls(最小二乘重构)}

守卫仅当某 recovery 在验证集上**严格优于** fallback(none) 时才切换 —— 故在验证集上**绝不劣于**
不补偿基线。RECOVERY 与校准规模相关：校准充分 → ls 最佳；校准 < 中间维 → ls 欠定过拟合，bias 更稳。

校准很小时的**安全旋钮**：ls/bias 都可能过拟合校准域的激活统计（在分布漂移的 test 上反而变差）。
若要 test 侧也稳健，传 ``recovery_menu=("none", "ls")`` 去掉 bias（实测：小校准下 bias 折叠最易过拟合）。

最简用法::

    from amct_pytorch.experimental.pruning import guarded_recovery_hybrid_prune
    res = guarded_recovery_hybrid_prune(model, data=calib, prune_ratio=0.5, eval_data=val)
    print(res.summary())   # 选中的 recovery + 相对 none 的验证增益 + 削减比例
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from torch import nn

from .accuracy_based_auto_prune import QualityFn, _resolve_quality
from .config import PruneConfig
from .context import BatchAdapter
from .pruner import AutoPruner
from .report import PruneReport
from .utils import count_parameters

#: 默认 recovery MENU；第 0 项 "none"(朴素切片) 为 fallback。
DEFAULT_RECOVERY_MENU: Tuple[str, ...] = ("none", "bias", "ls")

#: 支持的剪枝域 -> 其 reconstruct 方法名（dense FFN / CNN 通道，均带 recovery kwarg）。
_DOMAIN_METHOD = {"dense": "reconstruct", "cnn": "reconstruct"}


@dataclass(slots=True)
class RecoveryHybridResult:
    """守卫式 recovery-HYBRID 的结果。``applied`` 表示是否就地改了原模型。"""

    chosen_recovery: str
    fallback_recovery: str
    recovery_quality: Dict[str, float]
    chosen_quality: float
    fallback_quality: float
    gain_over_fallback: float
    prune_ratio: float
    domain: str
    params_before: int
    params_after: int
    weight_reduction: float
    applied: bool
    report: Optional[PruneReport]
    trials: List[Tuple[str, float]] = field(default_factory=list)

    def summary(self) -> str:
        verb = "已应用" if self.applied else "建议（未应用）"
        ranking = ", ".join(
            f"{name}={q:.4f}"
            for name, q in sorted(
                self.recovery_quality.items(), key=lambda kv: kv[1], reverse=True
            )
        )
        return (
            f"[recovery-hybrid:{self.domain}] {verb} recovery={self.chosen_recovery}"
            f"（fallback={self.fallback_recovery}，验证增益 {self.gain_over_fallback:+.4f}）："
            f"剪枝率={self.prune_ratio:.2f}，权重 {self.params_before:,} → {self.params_after:,}"
            f"（削减 {self.weight_reduction * 100:.1f}%）。 验证质量 [{ranking}]。"
        )


def _candidate_config(
    domain: str, recovery: str, prune_ratio: float, ridge: float, copy_model: bool
) -> PruneConfig:
    """只在目标域用 reconstruct(recovery=…)，其余可剪域置 prune_ratio=0（提前返回，绝不误剪）。"""
    base = {
        "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.0}},
        "cnn": {"name": "variance_channel", "kwargs": {"prune_ratio": 0.0}},
        "moe": {"name": "activation_count", "kwargs": {"prune_ratio": 0.0}},
    }
    base[domain] = {
        "name": _DOMAIN_METHOD[domain],
        "kwargs": {"prune_ratio": prune_ratio, "recovery": recovery, "ridge": ridge},
    }
    return PruneConfig(
        methods=base,
        copy_model=copy_model,
        verbose=False,
        missing_data_policy="warn_skip",
    )


def _prune_with(
    model, domain, recovery, prune_ratio, ridge, *, data, batch_adapter, copy_model
):
    cfg = _candidate_config(domain, recovery, prune_ratio, ridge, copy_model)
    pruner = AutoPruner(cfg)
    pruned = pruner(model, data=data, batch_adapter=batch_adapter)
    return pruned, pruner.last_report


def guarded_recovery_hybrid_prune(
    model: nn.Module,
    *,
    data: Any = None,
    prune_ratio: float = 0.50,
    domain: str = "dense",
    recovery_menu: Sequence[str] = DEFAULT_RECOVERY_MENU,
    ridge: float = 1e-2,
    eval_data: Optional[Sequence[Any]] = None,
    quality_fn: Optional[QualityFn] = None,
    evaluator: Optional[Any] = None,
    eval_iterations: int = 1,
    batch_adapter: Optional[BatchAdapter] = None,
    apply: bool = True,
) -> RecoveryHybridResult:
    """守卫式 recovery-HYBRID：一次性在 recovery MENU 上各剪一份，用验证集择优后就地应用最优补偿。

    Parameters
    ----------
    model : nn.Module
        含 dense FFN（或 CNN 顺序卷积）的模型（``apply=True`` 时就地修改）。
    data : Any
        校准数据（每个 recovery 各跑一次）；未单独给 ``eval_data`` 时也用作默认保真度验证集。
    prune_ratio : float
        中间神经元 / 通道剪枝比例（所有 recovery 用同一比例与同一显著度判据，只比"如何补偿"）。
    domain : str
        "dense"（FFN）或 "cnn"（顺序卷积通道）。
    recovery_menu : Sequence[str]
        候选补偿集合。默认 ``("none", "bias", "ls")``，**第 0 项为 fallback**。小校准且要 test 侧
        稳健时传 ``("none", "ls")`` 去掉 bias（实测 bias 折叠在小校准下最易过拟合校准域）。
    quality_fn / evaluator / eval_data
        验证质量来源（越大越好）。None 时用默认 top-1 保真度。优先级 quality_fn > evaluator > 保真度。
    apply : bool
        True 时把选中 recovery 就地应用到 ``model``。

    Returns
    -------
    RecoveryHybridResult
        含 ``chosen_recovery`` / ``recovery_quality`` / ``gain_over_fallback``（守卫保证 >= 0）与 ``summary()``。

    Notes
    -----
    守卫只用**验证集**择优，仅当某 recovery 严格优于 fallback(none) 时切换 —— 故验证集上绝不劣于不
    补偿基线。某 recovery 剪枝/前向失败时记 -inf 并告警，不影响其余，也不抛异常。
    """
    if not (0.0 <= prune_ratio < 1.0):
        raise ValueError(f"prune_ratio must be in [0.0, 1.0). Got {prune_ratio}.")
    if domain not in _DOMAIN_METHOD:
        raise ValueError(
            f"domain must be one of {tuple(_DOMAIN_METHOD)}, got {domain!r}."
        )
    menu = list(dict.fromkeys(recovery_menu))  # de-dup, keep order
    if not menu:
        raise ValueError("recovery_menu must contain at least the fallback recovery.")
    fallback = menu[0]

    eval_batches = (
        list(eval_data)
        if eval_data is not None
        else (list(data) if data is not None else [])
    )
    quality_fn, _ = _resolve_quality(
        model, quality_fn, evaluator, eval_iterations, eval_batches, batch_adapter
    )

    params_before = count_parameters(model)
    quality: Dict[str, float] = {}
    pa_by_rec: Dict[str, int] = {}
    for rec in menu:
        try:
            pruned, _ = _prune_with(
                model,
                domain,
                rec,
                prune_ratio,
                ridge,
                data=data,
                batch_adapter=batch_adapter,
                copy_model=True,
            )
            quality[rec] = float(quality_fn(pruned))
            pa_by_rec[rec] = count_parameters(pruned)
            del pruned
        except Exception as exc:
            warnings.warn(
                f"[recovery-hybrid] recovery {rec!r} 试探失败（非正常拒绝）："
                f"{type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            quality[rec] = float("-inf")
            pa_by_rec[rec] = params_before

    fb_q = quality[fallback]
    chosen = max(quality, key=lambda r: quality[r])
    if not (quality[chosen] > fb_q):
        chosen = fallback
    gain = 0.0 if chosen == fallback else quality[chosen] - fb_q

    report = None
    params_after = params_before
    if apply:
        _, report = _prune_with(
            model,
            domain,
            chosen,
            prune_ratio,
            ridge,
            data=data,
            batch_adapter=batch_adapter,
            copy_model=False,
        )
        params_after = count_parameters(model)
    else:
        params_after = pa_by_rec.get(chosen, params_before)

    return RecoveryHybridResult(
        chosen_recovery=chosen,
        fallback_recovery=fallback,
        recovery_quality=quality,
        chosen_quality=quality[chosen],
        fallback_quality=fb_q,
        gain_over_fallback=gain,
        prune_ratio=prune_ratio,
        domain=domain,
        params_before=params_before,
        params_after=params_after,
        weight_reduction=(1.0 - params_after / params_before) if params_before else 0.0,
        applied=apply,
        report=report,
        trials=sorted(quality.items(), key=lambda kv: kv[1], reverse=True),
    )
