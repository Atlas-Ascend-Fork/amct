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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from torch import nn

from ..common.utils.log import LOGGER
from .config import PruneConfig
from .context import BatchAdapter
from .domains.cnn import CNNPruningDomain
from .domains.dense import DensePruningDomain
from .domains.moe import MoEPruningDomain
from .utils import count_parameters


@dataclass
class DiagnosisReport:
    targets: Dict[str, int]
    prune_works: bool
    prune_reduction: float
    prune_forward_ok: Optional[bool]
    search_works: bool
    search_chosen_ratio: Optional[float]
    notes: List[str] = field(default_factory=list)

    @property
    def any_domain_detected(self) -> bool:
        return any(v > 0 for v in self.targets.values())

    def summary(self) -> str:
        det = ", ".join(f"{k}={v}" for k, v in self.targets.items())
        lines = [
            f"[prune-diagnose] prunable targets: {det}",
            f"  fixed-ratio prune: {'available' if self.prune_works else 'ineffective (0 cut)'}"
            f" (cut {self.prune_reduction * 100:.1f}%"
            + (
                f", forward {'OK' if self.prune_forward_ok else 'failed'}"
                if self.prune_forward_ok is not None
                else ""
            )
            + ")",
            f"  acc binary search: {'available' if self.search_works else 'unavailable'}"
            + (
                f" (chosen prune_ratio={self.search_chosen_ratio})"
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
    except Exception as exc:
        LOGGER.logw(
            f"[diagnose] forward check did not complete: {type(exc).__name__}: {exc}",
            "amct_prune",
        )
        return False


def _diag_fixed_ratio(model, base_cfg, prune_ratio, *, data, batch_adapter, notes):
    """Fixed-ratio prune dry-run (on a copy). Returns (works, reduction, forward_ok); errors go into notes."""
    import torch

    from .accuracy_based_auto_prune import _with_ratio
    from .pruner import AutoPruner

    try:
        trial_cfg = PruneConfig(**_with_ratio(base_cfg, prune_ratio))
        trial_cfg.copy_model = True
        pruned = AutoPruner(trial_cfg)(model, data=data, batch_adapter=batch_adapter)
        before, after = count_parameters(model), count_parameters(pruned)
        reduction = 1.0 - after / before if before else 0.0
        sample = None
        if data is not None:
            sample = data if torch.is_tensor(data) else next(iter(data), None)
        forward_ok = (
            _forward_ok(pruned, sample, batch_adapter) if sample is not None else None
        )
        return after < before, reduction, forward_ok
    except Exception as exc:
        LOGGER.logw(
            f"[diagnose] fixed-ratio prune dry-run did not complete: {type(exc).__name__}: {exc}",
            "amct_prune",
        )
        notes.append(f"fixed-ratio prune dry-run error: {type(exc).__name__}: {exc}")
        return False, 0.0, None


def _diag_acc_search(model, config, *, data, tolerance, batch_adapter, notes):
    """Acc binary-search dry-run (apply=False). Returns (works, chosen); errors/no-solution go into notes."""
    if data is None:
        notes.append("no data provided: skipping acc binary-search check.")
        return False, None
    try:
        from .accuracy_based_auto_prune import _accuracy_based_auto_prune

        res = _accuracy_based_auto_prune(
            model,
            config,
            data=data,
            tolerance=tolerance,
            batch_adapter=batch_adapter,
            apply=False,
        )
        if res.chosen_ratio is None:
            notes.append(
                "acc search found no acceptable prune ratio at this tolerance (relax the tolerance)."
            )
        return res.chosen_ratio is not None, res.chosen_ratio
    except Exception as exc:
        LOGGER.logw(
            f"[diagnose] acc search dry-run did not complete: {type(exc).__name__}: {exc}",
            "amct_prune",
        )
        notes.append(f"acc search dry-run error: {type(exc).__name__}: {exc}")
        return False, None


def prune_diagnose(
    model: nn.Module,
    data: Any = None,
    config=None,
    prune_ratio: float = 0.5,
    tolerance: float = 0.05,
) -> DiagnosisReport:
    """Check whether pruning works on the model (dry-run on a copy; original untouched)."""
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
        notes.append(
            "no prunable targets identified in any domain: the model structure may not be "
            "supported by the current domain detection."
        )

    from .accuracy_based_auto_prune import _config_to_dict, _merge_attention_skips

    base_cfg = _merge_attention_skips(_config_to_dict(config))
    prune_works, prune_reduction, prune_forward_ok = _diag_fixed_ratio(
        model,
        base_cfg,
        prune_ratio,
        data=data,
        batch_adapter=None,
        notes=notes,
    )
    search_works, search_chosen = _diag_acc_search(
        model,
        config,
        data=data,
        tolerance=tolerance,
        batch_adapter=None,
        notes=notes,
    )

    return DiagnosisReport(
        targets=targets,
        prune_works=prune_works,
        prune_reduction=prune_reduction,
        prune_forward_ok=prune_forward_ok,
        search_works=search_works,
        search_chosen_ratio=search_chosen,
        notes=notes,
    )
