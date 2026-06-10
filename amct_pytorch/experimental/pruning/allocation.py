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
"""按层敏感度分配剪枝率（sensitivity allocation）。

思路（已测量验证）：
- 敏感度 = 仅剪该层（其余层全部 skip）时的 top-1 保真度下降（无标签）。
- water-fill 分配 = 总削减量与 uniform 完全相同，但 robust（低敏感）层多剪、
  敏感层少剪：ratio = clip(cut * (1 + (mean - s)/mean), floor, ceil) 后归一。
- 守卫 = 在 deepcopy 候选上同口径构建 uniform/sensitivity 两个模型，比较校准
  NLL，取更低者 —— 防止个别模型（如 pythia）上分配反向，劣化时回退并告警。

应用方式与逐层隔离剪枝一致：对每层做一次 prune，其余层进 skip_layers。
"""

from __future__ import annotations

import copy
import math
import re
import warnings
from typing import Any, List, Mapping, Optional, Sequence

from torch import nn

from .accuracy_based_auto_prune import (
    _make_fidelity_quality,
    _target_module_paths,
    _with_ratio,
)
from .calib import calib_nll
from .compat import detect_backend
from .config import PruneConfig
from .pruner import AutoPruner
from .report import PruneReport
from .utils import count_parameters

# 通用层前缀：HF 系模型的 layers./blocks./h. 编号块。
_LAYER_RE = re.compile(r"(.*\b(?:layers|blocks|h)\.\d+)(?:\.|$)")
_IDX_RE = re.compile(r"(?:layers|blocks|h)\.(\d+)")


def layer_prefixes(model: nn.Module) -> List[str]:
    """模型内 '...layers.{i}' 形式的层前缀，按层号升序。"""
    keys = set()
    for name, _ in model.named_modules():
        match = _LAYER_RE.search(name)
        if match:
            keys.add(match.group(1))
    return sorted(keys, key=lambda s: int(_IDX_RE.search(s).group(1)))


def _base_config_dict(config: Optional[PruneConfig]) -> dict:
    """取逐层应用用的基础配置：复制字段、剥离 allocation、关闭克隆与冗余输出。"""
    cfg = config or PruneConfig()
    base = {f: copy.deepcopy(getattr(cfg, f)) for f in cfg.__slots__}
    base["allocation"] = None  # 防止逐层调用时递归触发分配路径
    base["copy_model"] = False
    base["verbose"] = False
    return base


def _prunable_target_paths(model: nn.Module, base_cfg: dict) -> List[str]:
    """三域（cnn/dense/moe）find_targets 命中的全部模块路径（与 diagnose 同口径）。

    某域识别失败时告警（不静默），隔离范围可能不完整。
    """
    from .domains.cnn import CNNPruningDomain
    from .domains.dense import DensePruningDomain
    from .domains.moe import MoEPruningDomain

    cfg = PruneConfig(**copy.deepcopy(base_cfg))
    paths: List[str] = []
    for domain in (CNNPruningDomain(), DensePruningDomain(), MoEPruningDomain()):
        try:
            targets = domain.find_targets(model, cfg)
        except Exception as exc:  # 失败≠静默：隔离可能不完整，必须告警
            warnings.warn(
                f"[allocation] {domain.name} find_targets failed; per-layer "
                f"isolation may be incomplete ({type(exc).__name__}: {exc}).",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        for target in targets:
            paths.extend(_target_module_paths(target))
    return sorted(set(paths))


def _isolation_skips(
    keys: Sequence[str], keep: str, extra: Sequence[str], target_paths: Sequence[str]
) -> List[str]:
    """skip keep 前缀之外的一切：其余层前缀（加 '.' 防止 layers.1 误匹配 layers.10）
    + 不在 keep 前缀下的全部可剪目标路径（覆盖 head/pooler/顶层 FFN），并保留用户 skip。"""
    skips = list(extra) + [k + "." for k in keys if k != keep]
    prefix = keep + "."
    skips += [p for p in target_paths if p != keep and not p.startswith(prefix)]
    return skips


def _prune_single_layer(model, base_cfg, keys, key, ratio, data, batch_adapter):
    """仅对 key 层按 ratio 剪枝（其余层与层外全部可剪目标隔离），就地修改 model。"""
    cfg_dict = _with_ratio(base_cfg, float(ratio))
    cfg_dict["skip_layers"] = _isolation_skips(
        keys, key, base_cfg.get("skip_layers") or [], _prunable_target_paths(model, base_cfg)
    )
    pruner = AutoPruner(PruneConfig(**cfg_dict))
    pruned = pruner(model, data=data, batch_adapter=batch_adapter)
    if pruned is not model:
        model.__dict__.update(pruned.__dict__)


def _apply_layer_ratios(model, base_cfg, keys, ratios, data, batch_adapter):
    """按 ratios 逐层顺序剪枝（ratio<=0 的层跳过），就地修改并返回 model。"""
    for key, ratio in zip(keys, ratios):
        if ratio <= 0:
            continue
        _prune_single_layer(model, base_cfg, keys, key, ratio, data, batch_adapter)
    return model


def measure_layer_sensitivity(
    model: nn.Module,
    data: Sequence[Any],
    config: Optional[PruneConfig] = None,
    ref_ratio: float = 0.5,
    skip_layers: Optional[Sequence[str]] = None,
) -> "dict[str, float]":
    """逐层单独以 ref_ratio 剪枝，测 top-1 保真度下降 → {层前缀: 下降}（越大越敏感）。

    每次试探都在 deepcopy 副本上做，原模型不动；某层剪枝/前向失败时记最大敏感度
    并告警（不静默），该层后续会被分到最小剪枝率。
    """
    batches = list(data)
    if not batches:
        raise ValueError("measure_layer_sensitivity needs calibration data batches.")
    keys = layer_prefixes(model)
    if not keys:
        raise ValueError(
            "no '(layers|blocks|h).<i>' prefixed layers found; "
            "sensitivity allocation needs a layered model."
        )
    base_cfg = _base_config_dict(config)
    if skip_layers:
        base_cfg["skip_layers"] = list(base_cfg.get("skip_layers") or []) + list(
            skip_layers
        )
    fid = _make_fidelity_quality(model, batches, None)
    sens: dict = {}
    for key in keys:
        cand = copy.deepcopy(model)
        try:
            _prune_single_layer(cand, base_cfg, keys, key, ref_ratio, batches, None)
            sens[key] = 1.0 - float(fid(cand))
        except Exception as exc:  # 失败≠静默：记满敏感度并明确告警
            warnings.warn(
                f"[allocation] layer '{key}' sensitivity probe failed "
                f"({type(exc).__name__}: {exc}); treating as maximally sensitive.",
                RuntimeWarning,
                stacklevel=2,
            )
            sens[key] = 1.0
    return sens


def _redistribute_residual(ratios: List[float], total: float, floor: float, ceil: float):
    """把预算残差迭代摊给未被夹紧的层（夹紧后继续迭代），直到总量收敛或全部夹紧。"""
    for _ in range(len(ratios) + 1):
        residual = total - sum(ratios)
        if abs(residual) <= 1e-9:
            return
        if residual > 0:
            free = [i for i, r in enumerate(ratios) if r < ceil - 1e-12]
        else:
            free = [i for i, r in enumerate(ratios) if r > floor + 1e-12]
        if not free:
            return
        share = residual / len(free)
        for i in free:
            ratios[i] = min(ceil, max(floor, ratios[i] + share))


def water_fill_ratios(
    sens, total_cut: float, floor: float = 0.05, ceil: float = 0.9
) -> List[float]:
    """water-fill 分配：总削减量 = total_cut*N 固定，robust（低敏感）层分到更大比例。

    ratio_i = clip(total_cut * (1 + (mean - s_i)/mean), floor, ceil)，clip 造成的预算
    残差迭代摊回未夹紧层 —— 保证 sum == total_cut*N（1e-6 内）且 floor<=r<=ceil。
    """
    vals = list(sens.values()) if isinstance(sens, Mapping) else list(sens)
    if not vals:
        raise ValueError("sens must be non-empty.")
    n = len(vals)
    total = total_cut * n
    if not (floor * n - 1e-6 <= total <= ceil * n + 1e-6):
        raise ValueError(
            f"total budget {total:.4f} infeasible under floor={floor}, ceil={ceil}, n={n}."
        )
    mean = sum(vals) / n if sum(vals) else 1.0
    ratios = [
        min(ceil, max(floor, total_cut * (1.0 + (mean - s) / (mean + 1e-9))))
        for s in vals
    ]
    _redistribute_residual(ratios, total, floor, ceil)
    return ratios


def _uniform_cut(config: PruneConfig) -> float:
    """uniform 口径的单层剪枝率 = dense 方法的 prune_ratio（分配保持总量与之相同）。"""
    return float(config.resolved_methods()["dense"].kwargs.get("prune_ratio", 0.5))


def _guard_pick(model, base_cfg, keys, ratios, cut, batches, batch_adapter):
    """deepcopy 上构建 sensitivity / uniform 双候选，按校准 NLL 取低；回退时告警。"""
    cand_sens = _apply_layer_ratios(
        copy.deepcopy(model), base_cfg, keys, ratios, batches, batch_adapter
    )
    cand_uni = _apply_layer_ratios(
        copy.deepcopy(model), base_cfg, keys, [cut] * len(keys), batches, batch_adapter
    )
    nll_sens = calib_nll(cand_sens, batches)
    nll_uni = calib_nll(cand_uni, batches)
    if math.isnan(nll_sens) or math.isnan(nll_uni):
        warnings.warn(
            f"[allocation] NaN calib NLL detected (sens={nll_sens}, uniform={nll_uni}); "
            f"guard cannot trust sensitivity candidate.",
            RuntimeWarning,
            stacklevel=3,
        )
    # 注意取 not(<)：NaN 比较恒 False → 兜底回退 uniform，不会静默选 NaN 候选
    if not nll_sens < nll_uni:
        warnings.warn(
            f"[allocation] sensitivity allocation does not beat uniform on calib NLL "
            f"(sens={nll_sens:.4f} vs uniform={nll_uni:.4f}); reverting to uniform.",
            RuntimeWarning,
            stacklevel=3,
        )
        return cand_uni, "uniform"
    return cand_sens, "sensitivity"


def _uniform_whole_model(model, base_cfg, keys, cut, batches, batch_adapter, report):
    """层数不足（keys<2）时真实执行整模型 uniform 剪枝（绝不 warn 后 no-op），就地修改。"""
    warnings.warn(
        "[allocation] fewer than 2 layered blocks found; "
        "applying uniform pruning to the whole model.",
        RuntimeWarning,
        stacklevel=3,
    )
    pruner = AutoPruner(PruneConfig(**copy.deepcopy(base_cfg)))
    pruned = pruner(model, data=batches, batch_adapter=batch_adapter)
    if pruned is not model:
        model.__dict__.update(pruned.__dict__)
    for key in keys:
        report.add("allocation", "uniform", key, f"ratio={cut:.3f}")
    report.params_after = count_parameters(model)
    report.allocation_choice = "uniform"
    return report


def prune_with_allocation(
    model: nn.Module,
    config: PruneConfig,
    data: Any = None,
    batch_adapter=None,
) -> PruneReport:
    """sensitivity 分配剪枝主入口：测层敏感度 → water-fill 分配 → 守卫 → 就地应用。

    总削减量与 uniform 相同；guard='calib_nll'（默认）时与 uniform 候选比较校准
    NLL，劣化则回退 uniform 并告警。结果写入 report.allocation_choice。
    """
    if data is None:
        raise ValueError(
            "allocation strategy 'sensitivity' requires calibration data; "
            "pass prune(..., data=...)."
        )
    alloc = dict(config.allocation or {})
    batches = list(data)
    backend = detect_backend(model)
    params_before = count_parameters(model)
    base_cfg = _base_config_dict(config)
    keys = layer_prefixes(model)
    cut = _uniform_cut(config)
    report = PruneReport(backend=backend.name, params_before=params_before)
    if len(keys) < 2:
        return _uniform_whole_model(model, base_cfg, keys, cut, batches, batch_adapter, report)
    sens = measure_layer_sensitivity(
        model, batches, config, float(alloc.get("ref_ratio", cut)),
        config.skip_layers,
    )
    ratios = water_fill_ratios(
        sens, cut, float(alloc.get("min_ratio", 0.05)),
        float(alloc.get("max_ratio", 0.9)),
    )
    if alloc.get("guard", "calib_nll") == "calib_nll":
        chosen, choice = _guard_pick(
            model, base_cfg, keys, ratios, cut, batches, batch_adapter
        )
    else:
        chosen = _apply_layer_ratios(
            copy.deepcopy(model), base_cfg, keys, ratios, batches, batch_adapter
        )
        choice = "sensitivity"
    model.__dict__.update(chosen.__dict__)
    for key, ratio in zip(keys, ratios if choice == "sensitivity" else [cut] * len(keys)):
        report.add("allocation", choice, key, f"ratio={ratio:.3f}")
    report.params_after = count_parameters(model)
    report.allocation_choice = choice
    return report
