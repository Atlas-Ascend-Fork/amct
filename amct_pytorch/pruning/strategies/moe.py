from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from ..report import PruneReport
from ..utils import prune_linear_out_features, replace_submodule, topk_keep_indices
from .base import BasePruningStage


class MoEPruningStage(BasePruningStage):
    stage_name = "moe"

    ROUTER_NAMES = ("gate", "router", "gates")
    EXPERT_CONTAINER_NAMES = ("experts", "expert_layers", "moe_experts")

    def _expert_score(self, expert: nn.Module) -> float:
        total = 0.0
        count = 0
        for param in expert.parameters():
            total += float(param.detach().abs().mean().cpu())
            count += 1
        return total / max(count, 1)

    def _find_expert_container(self, module: nn.Module) -> Optional[Tuple[str, nn.ModuleList]]:
        for name in self.EXPERT_CONTAINER_NAMES:
            child = getattr(module, name, None)
            if isinstance(child, nn.ModuleList):
                return name, child
        return None

    def _find_router(self, module: nn.Module, num_experts: int) -> Optional[Tuple[str, nn.Linear]]:
        for name in self.ROUTER_NAMES:
            child = getattr(module, name, None)
            if isinstance(child, nn.Linear) and child.out_features == num_experts:
                return name, child
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and child.out_features == num_experts:
                return name, child
        return None

    def apply(self, model: nn.Module, report: PruneReport) -> None:
        ratio = self.config.moe_ratio
        if ratio <= 0:
            return

        last_num_experts: Optional[int] = None
        for module_name, module in list(model.named_modules()):
            result = self._find_expert_container(module)
            if result is None:
                continue
            container_name, experts = result
            num_experts = len(experts)
            if num_experts <= self.config.min_experts:
                continue

            scores = torch.tensor([self._expert_score(expert) for expert in experts], dtype=torch.float32)
            keep_idx = topk_keep_indices(scores, ratio, self.config.min_experts)
            if len(keep_idx) == num_experts:
                continue

            kept_experts = nn.ModuleList([experts[i] for i in keep_idx])
            container_path = f"{module_name}.{container_name}" if module_name else container_name
            replace_submodule(model, container_path, kept_experts)

            router_info = self._find_router(module, num_experts)
            if router_info is not None:
                router_name, router = router_info
                router_path = f"{module_name}.{router_name}" if module_name else router_name
                replace_submodule(model, router_path, prune_linear_out_features(router, keep_idx))

            for attr in ("num_experts", "n_experts", "num_local_experts"):
                if hasattr(module, attr):
                    try:
                        setattr(module, attr, len(keep_idx))
                    except Exception:
                        pass
            if hasattr(module, "top_k"):
                try:
                    module.top_k = min(int(module.top_k), len(keep_idx))
                except Exception:
                    pass

            report.add(
                self.stage_name,
                module_name or "<root>",
                f"Experts {num_experts} -> {len(keep_idx)}",
            )
            last_num_experts = len(keep_idx)

        if last_num_experts is not None:
            meta = getattr(model, "_ascend_prune_meta", {})
            meta["moe_num_experts"] = int(last_num_experts)
            setattr(model, "_ascend_prune_meta", meta)
