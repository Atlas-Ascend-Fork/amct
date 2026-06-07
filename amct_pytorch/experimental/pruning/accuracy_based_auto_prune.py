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
"""容差驱动的自动剪枝率搜索：用户给出可接受的精度损失（tolerance），
自动在候选剪枝率中挑选 *满足容差的最大剪枝率* 并就地应用。

核心思路
--------
- 把“精度”抽象成一个 ``quality_fn(model) -> float``（越大越好）。
  * 用户提供 ``quality_fn`` 时：baseline = quality_fn(原模型)，每个候选率的
    损失 = baseline - quality_fn(剪枝副本)。tolerance 与 quality_fn 同量纲
    （例如准确率返回 0~1，则 tolerance=0.02 表示“最多掉 2 个百分点”）。
  * 未提供时：使用默认*保真度*指标 —— 以原模型在校准数据上的 top-1 预测为
    参考，quality = 剪枝后与参考的 top-1 一致率（∈[0,1]，baseline=1.0）。
    此时 tolerance 表示“最多允许多少比例的预测发生翻转”。
- 在候选剪枝率网格上按“损失随剪枝率单调不降”的常见假设做二分查找，
  找到损失 ≤ tolerance 的最大剪枝率；找不到任何满足者则不剪（applied=False）。
- 每次试探都在 ``copy.deepcopy`` 的副本上进行（AutoPruner(copy_model=True)），
  原模型在最终应用前保持不变。
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, List, Mapping, Optional, Sequence

import torch
from torch import nn

from .config import MethodSpec, PruneConfig
from .context import BatchAdapter
from .pruner import AutoPruner
from .report import PruneReport
from .utils import count_parameters

QualityFn = Callable[[nn.Module], float]

DEFAULT_RATIO_GRID: tuple = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)

#: 默认容差 —— 对“不熟悉剪枝”的用户给一个保守起点（最多 2% 质量损失）。
DEFAULT_TOLERANCE: float = 0.02

#: 自动跳过的注意力子模块名片段。dense 域的“相邻 Linear=两层 FFN”启发式会把
#: 并列的 q/k/v 投影误判为可剪的 FFN，破坏注意力；默认把这些路径排除，
#: 让不了解内部结构的用户也能安全地对 LLM 做 FFN 剪枝。
ATTENTION_NAME_HINTS: tuple = ("self_attn", "attention", "attn")


@dataclass(slots=True)
class TrialResult:
    prune_ratio: float
    quality: float
    quality_drop: float
    params_after: int
    weight_reduction: float
    accepted: bool


@dataclass(slots=True)
class AutoTuneResult:
    """容差搜索结果。``applied`` 表示是否对原模型就地施加了剪枝。"""
    chosen_ratio: Optional[float]
    tolerance: float
    baseline_quality: float
    pruned_quality: Optional[float]
    quality_drop: Optional[float]
    params_before: int
    params_after: int
    weight_reduction: float
    applied: bool
    report: Optional[PruneReport]
    trials: List[TrialResult] = field(default_factory=list)

    def summary(self) -> str:
        """面向用户的一段式可读摘要。"""
        if self.chosen_ratio is None:
            return (
                f"[auto-prune] 在容差 {self.tolerance:.3f} 内找不到可接受的剪枝率"
                f"（已试 {len(self.trials)} 个候选），模型未改动。"
                f" 可放宽容差或在 ratio_grid 中加入更小的剪枝率。"
            )
        verb = "已应用" if self.applied else "建议（未应用）"
        return (
            f"[auto-prune] {verb}剪枝率={self.chosen_ratio:.2f}："
            f"权重 {self.params_before:,} → {self.params_after:,} "
            f"(削减 {self.weight_reduction * 100:.1f}%)，"
            f"质量损失 {self.quality_drop:.3f} ≤ 容差 {self.tolerance:.3f}。"
        )


# ---------------------------------------------------------------------------
# 配置工具
# ---------------------------------------------------------------------------
def _config_to_dict(config) -> dict:
    if config is None:
        # 默认覆盖三域，未命中的域会自然 no-op。
        return {"methods": {
            "cnn": {"name": "variance_channel", "kwargs": {}},
            "dense": {"name": "low_variance", "kwargs": {}},
            "moe": {"name": "mass_variance", "kwargs": {"boundary": 10}},
        }}
    if isinstance(config, PruneConfig):
        base = {f: getattr(config, f) for f in config.__slots__}
        return copy.deepcopy(base)
    if isinstance(config, Mapping):
        return copy.deepcopy(dict(config))
    raise TypeError(f"config must be dict / PruneConfig / None, got {type(config)!r}")


def _with_ratio(config_dict: dict, ratio: float) -> dict:
    """返回把所有方法的 prune_ratio 覆盖为 ``ratio`` 的配置副本。"""
    cfg = copy.deepcopy(config_dict)
    methods = cfg.get("methods", {})
    for name, spec in list(methods.items()):
        if isinstance(spec, MethodSpec):
            spec.kwargs["prune_ratio"] = ratio
        elif isinstance(spec, dict):
            kwargs = dict(spec.get("kwargs", {}))
            kwargs["prune_ratio"] = ratio
            spec["kwargs"] = kwargs
        elif isinstance(spec, str):
            # "name" 简写 → 展开为带 prune_ratio 的 dict
            methods[name] = {"name": spec, "kwargs": {"prune_ratio": ratio}}
    cfg["methods"] = methods
    return cfg


def _merge_attention_skips(config_dict: dict) -> dict:
    """把注意力子模块名片段并入 skip_layers，避免误剪并列 q/k/v 投影。"""
    cfg = copy.deepcopy(config_dict)
    skips = list(cfg.get("skip_layers", []) or [])
    for hint in ATTENTION_NAME_HINTS:
        if hint not in skips:
            skips.append(hint)
    cfg["skip_layers"] = skips
    return cfg


# ---------------------------------------------------------------------------
# 默认保真度指标
# ---------------------------------------------------------------------------
def _model_logits(model: nn.Module, batch: Any, batch_adapter: Optional[BatchAdapter]):
    if batch_adapter is not None:
        args, kwargs = batch_adapter(batch)
    elif isinstance(batch, Mapping):
        args, kwargs = (), dict(batch)
    elif isinstance(batch, (tuple, list)):
        args, kwargs = tuple(batch), {}
    else:
        args, kwargs = (batch,), {}
    out = model(*args, **kwargs)
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)) and out:
        return out[0]
    raise TypeError("Cannot extract logits from model output for fidelity metric; "
                    "pass an explicit quality_fn.")


def _make_fidelity_quality(
    model: nn.Module,
    eval_batches: Sequence[Any],
    batch_adapter: Optional[BatchAdapter],
) -> QualityFn:
    """以原模型的 top-1 预测为参考，构造 top-1 一致率质量函数。"""
    if not eval_batches:
        raise ValueError("Default fidelity metric needs eval data; pass data=/eval_data= "
                         "or provide a custom quality_fn.")
    refs = []
    model.eval()
    with torch.no_grad():
        for batch in eval_batches:
            refs.append(_model_logits(model, batch, batch_adapter).argmax(-1))

    def quality(candidate: nn.Module) -> float:
        candidate.eval()
        total = 0
        agree = 0
        with torch.no_grad():
            for batch, ref in zip(eval_batches, refs):
                pred = _model_logits(candidate, batch, batch_adapter).argmax(-1)
                agree += (pred == ref).sum().item()
                total += ref.numel()
        return agree / total if total else 1.0

    return quality


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def accuracy_based_auto_prune(
    model: nn.Module,
    config=None,
    *,
    data: Any = None,
    tolerance: float = DEFAULT_TOLERANCE,
    quality_fn: Optional[QualityFn] = None,
    evaluator: Optional[Any] = None,
    eval_iterations: int = 1,
    eval_data: Optional[Sequence[Any]] = None,
    ratio_grid: Sequence[float] = DEFAULT_RATIO_GRID,
    batch_adapter: Optional[BatchAdapter] = None,
    safe_skip_attention: bool = True,
    apply: bool = True,
) -> AutoTuneResult:
    """容差驱动的“一键”剪枝：给定可接受的精度损失，自动选出 *满足容差的
    最大剪枝率* 并就地应用 —— 用户无需了解剪枝率/层结构等细节。

    最简用法::

        from amct_pytorch.experimental.pruning import accuracy_based_auto_prune
        result = accuracy_based_auto_prune(model, data=calib, tolerance=0.02)
        print(result.summary())

    Parameters
    ----------
    model : nn.Module
        待剪枝模型（``apply=True`` 时被就地修改）。
    config : dict | PruneConfig | None
        剪枝配置；其各方法的 ``prune_ratio`` 会被搜索值覆盖。None 时覆盖三域默认方法。
    data : Any
        校准数据（方差类方法需要）；未单独给 ``eval_data`` 时也用作默认保真度评估集。
    tolerance : float
        可接受的精度损失上界，与 ``quality_fn`` 同量纲（默认保真度时为 top-1 翻转比例）。
        默认 ``0.02``（保守的 2%）。
    quality_fn : Callable[[nn.Module], float] | None
        精度评估函数（越大越好）。None 时用默认 top-1 保真度指标（剪枝后与原模型
        预测的一致率），无需标签。
    evaluator : object | None
        与量化侧 :func:`accuracy_based_auto_calibration` **共用同一精度抽象**：
        任意带 ``evaluate(model, iterations) -> float`` 方法的对象（如
        ``ModelEvaluator``）。给定时用它作为精度来源，从而同一个 evaluator 可同时
        驱动剪枝与量化的精度搜索。优先级：``quality_fn`` > ``evaluator`` > 默认保真度。
    eval_iterations : int
        传给 ``evaluator.evaluate`` 的迭代数。
    eval_data : Sequence | None
        默认保真度指标的评估批；缺省回退到 ``data``。
    ratio_grid : Sequence[float]
        候选剪枝率（升序）。
    safe_skip_attention : bool
        默认 True：自动把注意力子模块（self_attn/attention/attn）并入 skip_layers，
        防止 dense 域误把并列的 q/k/v 投影当作两层 FFN 剪掉而破坏模型。
        对纯 CNN/无注意力模型无副作用。
    apply : bool
        True 时把选中的剪枝率就地应用到 ``model``；False 时只搜索不改原模型。

    Returns
    -------
    AutoTuneResult
        含 ``chosen_ratio`` / ``weight_reduction`` / ``quality_drop`` 与 ``summary()``。

    Notes
    -----
    在网格上做二分查找，假设“质量损失随剪枝率单调不降”。若实测非单调，
    二分可能 *偏保守*（少剪一点）—— 对用户而言是安全方向。任何试探中
    剪枝或前向失败的剪枝率都会被判为不可接受，不会向外抛异常。
    """
    if not (0.0 <= tolerance):
        raise ValueError("tolerance must be >= 0.")
    grid = sorted(set(float(r) for r in ratio_grid if 0.0 <= float(r) < 1.0))
    if not grid:
        raise ValueError("ratio_grid must contain at least one ratio in [0.0, 1.0).")

    eval_batches = list(eval_data) if eval_data is not None else (list(data) if data is not None else [])
    base_cfg = _config_to_dict(config)
    if safe_skip_attention:
        base_cfg = _merge_attention_skips(base_cfg)
    params_before = count_parameters(model)

    # 精度来源优先级：quality_fn > evaluator > 默认 top-1 保真度
    if quality_fn is not None:
        baseline_quality = float(quality_fn(model))
    elif evaluator is not None:
        if not hasattr(evaluator, "evaluate"):
            raise TypeError("evaluator must expose .evaluate(model, iterations) -> float "
                            "(e.g. amct ModelEvaluator).")

        def quality_fn(m):
            return float(evaluator.evaluate(m, eval_iterations))

        baseline_quality = quality_fn(model)
    else:
        quality_fn = _make_fidelity_quality(model, eval_batches, batch_adapter)
        baseline_quality = 1.0  # 与自身参考完全一致

    def trial(ratio: float) -> TrialResult:
        cfg = PruneConfig(**_with_ratio(base_cfg, ratio))
        cfg.copy_model = True   # 在副本上试探，原模型不动
        cfg.verbose = False
        try:
            pruner = AutoPruner(cfg)
            pruned = pruner(model, data=data, batch_adapter=batch_adapter)
            pa = count_parameters(pruned)
            q = float(quality_fn(pruned))
        except Exception:
            # 该剪枝率导致剪枝/前向失败 → 判为不可接受，绝不向外抛异常。
            return TrialResult(prune_ratio=ratio, quality=float("-inf"),
                               quality_drop=float("inf"), params_after=params_before,
                               weight_reduction=0.0, accepted=False)
        drop = baseline_quality - q
        return TrialResult(
            prune_ratio=ratio, quality=q, quality_drop=drop,
            params_after=pa, weight_reduction=1.0 - pa / params_before if params_before else 0.0,
            accepted=drop <= tolerance,
        )

    # 在网格上二分：寻找“损失 ≤ tolerance”的最大剪枝率（单调假设）。
    trials: dict = {}

    def trial_cached(i: int) -> TrialResult:
        if i not in trials:
            trials[i] = trial(grid[i])
        return trials[i]

    lo, hi, best_idx = 0, len(grid) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        tr = trial_cached(mid)
        if tr.accepted:
            best_idx = mid
            lo = mid + 1   # 还能更激进
        else:
            hi = mid - 1   # 太激进，回退

    ordered_trials = [trials[i] for i in sorted(trials)]

    if best_idx is None:
        return AutoTuneResult(
            chosen_ratio=None, tolerance=tolerance, baseline_quality=baseline_quality,
            pruned_quality=None, quality_drop=None,
            params_before=params_before, params_after=params_before, weight_reduction=0.0,
            applied=False, report=None, trials=ordered_trials,
        )

    chosen = grid[best_idx]
    chosen_trial = trials[best_idx]
    report = None
    params_after = params_before
    if apply:
        cfg = PruneConfig(**_with_ratio(base_cfg, chosen))
        cfg.copy_model = False
        cfg.verbose = False
        pruner = AutoPruner(cfg)
        pruned = pruner(model, data=data, batch_adapter=batch_adapter)
        if pruned is not model:
            model.__dict__.update(pruned.__dict__)
        report = pruner.last_report
        params_after = count_parameters(model)

    return AutoTuneResult(
        chosen_ratio=chosen, tolerance=tolerance, baseline_quality=baseline_quality,
        pruned_quality=chosen_trial.quality, quality_drop=chosen_trial.quality_drop,
        params_before=params_before,
        params_after=params_after if apply else chosen_trial.params_after,
        weight_reduction=(1.0 - (params_after / params_before)) if apply and params_before
        else chosen_trial.weight_reduction,
        applied=apply, report=report, trials=ordered_trials,
    )
