from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch.nn as nn

from ..config import PruneConfig
from ..utils import get_submodule, iter_direct_named_modules, prune_linear_out_features, replace_submodule
from .base import BasePruningDomain


@dataclass(slots=True)
class MoETarget:
    module_path: str
    router_path: Optional[str]
    experts_path: str


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
            container_name, experts = result
            if len(experts) <= config.min_experts:
                continue
            router_info = self._find_router(module, len(experts))
            router_path = None
            if router_info is not None:
                router_name, _ = router_info
                router_path = f"{module_name}.{router_name}" if module_name else router_name
            experts_path = f"{module_name}.{container_name}" if module_name else container_name
            targets.append(MoETarget(module_path=module_name, router_path=router_path, experts_path=experts_path))
        return targets

    def num_experts(self, model: nn.Module, target: MoETarget) -> int:
        experts = get_submodule(model, target.experts_path)
        if not isinstance(experts, nn.ModuleList):
            raise TypeError(f"Experts container '{target.experts_path}' is not a ModuleList.")
        return len(experts)

    def get_router(self, model: nn.Module, target: MoETarget) -> Optional[nn.Linear]:
        if target.router_path is None:
            return None
        router = get_submodule(model, target.router_path)
        if not isinstance(router, nn.Linear):
            raise TypeError(f"Router '{target.router_path}' is not Linear.")
        return router

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
        for name, child in iter_direct_named_modules(module):
            if isinstance(child, nn.Linear) and child.out_features == num_experts:
                return name, child
        return None
