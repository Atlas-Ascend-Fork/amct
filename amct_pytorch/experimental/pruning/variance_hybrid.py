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
"""守卫式 variance-HYBRID 的 MoE 专家剪枝入口。

动机
----
方差类路由打分（一次校准即可得到，无需梯度 / 最小二乘 / 专家前向）在 *某些* MoE 上
明显优于按激活频率（mass）的剪枝，但**没有**一个普适的"更好的方差"判据：

- ``cond``  = Var(w | 被选中)：稳健默认，但会误剪"始终强权重"的特化专家；
- ``peak``  = max(w)：在多专家 / 分组路由上更强，但在少专家 softmax 上不稳；
- ``cvxpeak`` = Var(w|sel) x max(w)：兼顾，但增益边际。

判据与路由结构相关（实测：在小型真实路由风格上单一"更好方差"不能泛化）。
因此本入口不赌某一判据，而是：**一次校准 → 在判据 MENU 上各剪一份副本 → 用一个很小的
验证集（quality_fn）实测择优 → 应用最优者**。fallback（mass = activation_count）始终在
MENU 中，且仅当某方差判据在验证集上**严格优于** fallback 时才切换 —— 因此

    守卫保证：HYBRID 在验证集上**绝不劣于** fallback（mass）。

诚实范围：方差判据相对 mass 的平均 +6pp 增益是在合成不均衡的 mini-suite（balanced-acc）上测得；真实
预训练 MoE 在 mass-biased 的保真度代理上守卫会回退到 mass（无方差增益，但绝不更差）。守卫是 **VAL 上**
的保证，不是 test 保证 —— `eval_data` 须代表部署分布。详见包 README 的"诚实范围"段。

与 :mod:`accuracy_based_auto_prune`（在*剪枝率*网格上做容差搜索）同构、互补：那里搜
ratio，这里搜 *判据*；二者复用同一套 "deepcopy 试探 + quality_fn 守卫 + 就地应用" 机制。

最简用法::

    from amct_pytorch.experimental.pruning import guarded_variance_hybrid_prune
    res = guarded_variance_hybrid_prune(model, data=calib, prune_ratio=0.5, eval_data=val)
    print(res.summary())   # 选中的判据 + 相对 fallback 的验证增益 + 削减比例
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

#: 默认判据 MENU：(展示名, MoE MethodSpec)。第 0 项为 fallback（mass）。
#: 方差三项用 boundary=-1 让**每一层**都用方差打分（而非默认的深度分界）。
DEFAULT_MENU: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("mass", {"name": "activation_count", "kwargs": {}}),
    (
        "cond_var",
        {"name": "mass_variance", "kwargs": {"boundary": -1, "variance_score": "cond"}},
    ),
    (
        "peak",
        {"name": "mass_variance", "kwargs": {"boundary": -1, "variance_score": "peak"}},
    ),
    (
        "cvxpeak",
        {
            "name": "mass_variance",
            "kwargs": {"boundary": -1, "variance_score": "cvxpeak"},
        },
    ),
)


@dataclass(slots=True)
class HybridResult:
    """守卫式 variance-HYBRID 的结果。``applied`` 表示是否就地改了原模型。"""

    chosen_criterion: str
    fallback_criterion: str
    #: 每个判据在验证集上的质量（越大越好）；剪枝/前向失败的判据记为 -inf。
    criterion_quality: Dict[str, float]
    chosen_quality: float
    fallback_quality: float
    #: 选中判据相对 fallback 的验证增益（守卫保证 >= 0）。
    gain_over_fallback: float
    prune_ratio: float
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
                self.criterion_quality.items(), key=lambda kv: kv[1], reverse=True
            )
        )
        return (
            f"[variance-hybrid] {verb}判据={self.chosen_criterion}"
            f"（fallback={self.fallback_criterion}，验证增益 {self.gain_over_fallback:+.4f}）："
            f"专家剪枝率={self.prune_ratio:.2f}，权重 {self.params_before:,} → "
            f"{self.params_after:,}（削减 {self.weight_reduction * 100:.1f}%）。"
            f" 验证质量 [{ranking}]。"
        )


def _candidate_config(
    spec: Dict[str, Any], prune_ratio: float, top_k: int, copy_model: bool
) -> PruneConfig:
    """构造只剪 MoE 的候选配置：cnn/dense 置 prune_ratio=0（提前返回，绝不误剪），
    仅 MoE 用本判据。这样判据间唯一变量就是 MoE 选保留专家的*信号*。"""
    kwargs = dict(spec.get("kwargs", {}))
    kwargs["prune_ratio"] = prune_ratio
    kwargs["top_k"] = top_k
    return PruneConfig(
        methods={
            "cnn": {"name": "variance_channel", "kwargs": {"prune_ratio": 0.0}},
            "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.0}},
            "moe": {"name": str(spec["name"]), "kwargs": kwargs},
        },
        copy_model=copy_model,
        verbose=False,
        missing_data_policy="warn_skip",
    )


def _prune_with(
    model: nn.Module,
    spec: Dict[str, Any],
    prune_ratio: float,
    top_k: int,
    *,
    data: Any,
    batch_adapter: Optional[BatchAdapter],
    copy_model: bool,
) -> Tuple[nn.Module, Optional[PruneReport]]:
    cfg = _candidate_config(spec, prune_ratio, top_k, copy_model)
    pruner = AutoPruner(cfg)
    pruned = pruner(model, data=data, batch_adapter=batch_adapter)
    return pruned, pruner.last_report


def guarded_variance_hybrid_prune(
    model: nn.Module,
    *,
    data: Any = None,
    prune_ratio: float = 0.50,
    top_k: int = 8,
    eval_data: Optional[Sequence[Any]] = None,
    quality_fn: Optional[QualityFn] = None,
    evaluator: Optional[Any] = None,
    eval_iterations: int = 1,
    batch_adapter: Optional[BatchAdapter] = None,
    menu: Optional[Sequence[Tuple[str, Dict[str, Any]]]] = None,
    apply: bool = True,
) -> HybridResult:
    """守卫式 variance-HYBRID：一次校准跑出判据 MENU，用验证集择优后就地应用最优判据。

    Parameters
    ----------
    model : nn.Module
        含 MoE 层的模型（``apply=True`` 时被就地修改）。
    data : Any
        校准数据（每个判据各跑一次校准统计路由）；未单独给 ``eval_data`` 时也用作默认
        保真度验证集。
    prune_ratio : float
        每层移除的专家比例（所有判据用同一剪枝率，只比"选哪些专家"）。默认 0.50。
    top_k : int
        每 token 选中的专家数。默认 8。
    eval_data : Sequence | None
        守卫用的**验证集**（强烈建议与 ``data`` 不同，避免选择偏置）。缺省回退到 ``data``。
    quality_fn : Callable[[nn.Module], float] | None
        验证质量函数（越大越好）。None 时用默认 top-1 保真度（剪枝后与原模型预测的一致率），
        无需标签。优先级：``quality_fn`` > ``evaluator`` > 默认保真度。
    evaluator : object | None
        与量化侧 ``accuracy_based_auto_calibration`` 共用的 ``evaluate(model[, iters]) -> float``
        精度抽象；给定时作为验证质量来源。
    menu : Sequence[(name, MethodSpec)] | None
        判据 MENU。None 时用 :data:`DEFAULT_MENU`（mass / cond_var / peak / cvxpeak）。
        **第 0 项被视为 fallback**，且必须始终在 MENU 中（守卫据此保证不劣于 fallback）。
    apply : bool
        True 时把选中判据就地应用到 ``model``；False 时只搜索不改原模型。

    Returns
    -------
    HybridResult
        含 ``chosen_criterion`` / ``criterion_quality`` / ``gain_over_fallback`` 与 ``summary()``。

    Notes
    -----
    守卫只用**验证集**择优：仅当某方差判据严格优于 fallback 时才切换，否则回退 mass，
    因此 HYBRID 在验证集上绝不劣于 fallback。某判据剪枝/前向失败时记为 -inf 并告警，
    不影响其余判据，也不会向外抛异常。
    """
    if not (0.0 <= prune_ratio < 1.0):
        raise ValueError(f"prune_ratio must be in [0.0, 1.0). Got {prune_ratio}.")
    menu = list(menu) if menu is not None else list(DEFAULT_MENU)
    if not menu:
        raise ValueError("menu must contain at least the fallback criterion.")
    fallback_name = menu[0][0]

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
    pa_by_name: Dict[str, int] = {}
    for name, spec in menu:
        try:
            pruned, _ = _prune_with(
                model,
                spec,
                prune_ratio,
                top_k,
                data=data,
                batch_adapter=batch_adapter,
                copy_model=True,
            )
            quality[name] = float(quality_fn(pruned))
            pa_by_name[name] = count_parameters(pruned)
            del pruned
        except Exception as exc:  # 不静默：失败判据记 -inf 并告警，不拖垮其余判据
            warnings.warn(
                f"[variance-hybrid] 判据 {name!r} 试探失败（非正常拒绝）："
                f"{type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            quality[name] = float("-inf")
            pa_by_name[name] = params_before

    if fallback_name not in quality:
        raise RuntimeError("fallback criterion missing from menu evaluation.")

    # 守卫择优：取验证质量最高者；仅当**严格优于** fallback 时才切换，否则回退 fallback。
    fb_q = quality[fallback_name]
    chosen = max(quality, key=lambda n: quality[n])
    if not (quality[chosen] > fb_q):
        chosen = fallback_name
    gain = 0.0 if chosen == fallback_name else quality[chosen] - fb_q

    report: Optional[PruneReport] = None
    params_after = params_before
    if apply:
        chosen_spec = dict(menu)[chosen]
        pruned, report = _prune_with(
            model,
            chosen_spec,
            prune_ratio,
            top_k,
            data=data,
            batch_adapter=batch_adapter,
            copy_model=False,
        )
        if pruned is not model:
            model.__dict__.update(pruned.__dict__)
        params_after = count_parameters(model)
    else:
        params_after = pa_by_name.get(chosen, params_before)

    return HybridResult(
        chosen_criterion=chosen,
        fallback_criterion=fallback_name,
        criterion_quality=quality,
        chosen_quality=quality[chosen],
        fallback_quality=fb_q,
        gain_over_fallback=gain,
        prune_ratio=prune_ratio,
        params_before=params_before,
        params_after=params_after,
        weight_reduction=(1.0 - params_after / params_before) if params_before else 0.0,
        applied=apply,
        report=report,
        trials=sorted(quality.items(), key=lambda kv: kv[1], reverse=True),
    )
