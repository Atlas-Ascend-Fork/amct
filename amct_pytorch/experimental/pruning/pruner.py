from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch.nn as nn

from .compat import detect_backend, patch_common_config, save_model
from .config import MethodSpec, PruneConfig
from .context import BatchAdapter, PruneContext
from .domains.base import BasePruningDomain
from .methods.base import BasePruningMethod
from .registry import MethodRegistry, create_default_registry
from .report import PruneReport
from .utils import clone_model_if_needed, count_parameters


def _target_paths(target) -> list[str]:
    paths: list[str] = []
    for attr in ("producer_path", "consumer_path", "gate_path", "up_path",
                  "down_path", "module_path", "router_path", "experts_path"):
        val = getattr(target, attr, None)
        if isinstance(val, str):
            paths.append(val)
    return paths


def _is_skipped(target, skip_layers: list[str]) -> bool:
    if not skip_layers:
        return False
    for path in _target_paths(target):
        for pattern in skip_layers:
            if pattern in path:
                return True
    return False


def _warn(report: PruneReport, message: str) -> None:
    warnings.warn(message, RuntimeWarning, stacklevel=3)
    report.add_warning(message)


@dataclass(slots=True)
class PruningStage:
    domain_name: str
    domain: BasePruningDomain
    method: BasePruningMethod
    spec: MethodSpec

    def apply(self, model: nn.Module, context: PruneContext, report: PruneReport, config: PruneConfig) -> None:
        targets = self.domain.find_targets(model, config)
        if config.skip_layers:
            targets = [t for t in targets if not _is_skipped(t, config.skip_layers)]
        if not targets:
            report.add(self.domain_name, self.method.name, "<none>", "No compatible pruning targets were found")
            return
        if self.method.requires_data and context.data is None:
            message = (
                f"Skipping pruning method '{self.method.name}' for domain '{self.domain_name}'"
                f" because pruning data was not provided. Pass prune(..., data=...) to enable this stage."
            )
            if config.missing_data_policy == "raise":
                raise ValueError(message)
            _warn(report, message)
            return
        try:
            self.method.apply(model, self.domain, targets, context, report, config, self.spec)
        except Exception as exc:
            if config.stage_error_policy == "warn_skip":
                _warn(
                    report,
                    f"Skipping pruning method '{self.method.name}' for domain '{self.domain_name}' after error: {exc}",
                )
                return
            raise


class AutoPruner:
    def __init__(self, config: PruneConfig | None = None, registry: MethodRegistry | None = None) -> None:
        self.config = config or PruneConfig()
        self.config.validate()
        self.registry = (registry or create_default_registry()).clone()
        self.stages = self._build_stages()
        self.last_report: Optional[PruneReport] = None

    def __call__(
        self,
        model: nn.Module,
        data=None,
        batch_adapter: BatchAdapter | None = None,
    ) -> nn.Module:
        return self.prune(model, data=data, batch_adapter=batch_adapter)

    def prune(
        self,
        model: nn.Module,
        data=None,
        batch_adapter: BatchAdapter | None = None,
    ) -> nn.Module:
        backend = detect_backend(model)
        working = clone_model_if_needed(model, self.config.copy_model)
        before = count_parameters(working)
        report = PruneReport(backend=backend.name, params_before=before)
        context = PruneContext(data=data, batch_adapter=batch_adapter)

        meta = getattr(working, "_amct_prune_meta", {})
        meta["backend"] = backend.name
        meta["methods"] = {
            domain_name: {"name": spec.name, "kwargs": dict(spec.kwargs)}
            for domain_name, spec in self.config.resolved_methods().items()
        }
        setattr(working, "_amct_prune_meta", meta)

        for stage in self.stages:
            stage.apply(working, context, report, self.config)

        patch_common_config(working)
        report.params_after = count_parameters(working)
        self.last_report = report
        return working

    def save(self, model: nn.Module, save_dir: str | Path) -> Path:
        if self.last_report is None:
            before = count_parameters(model)
            backend = detect_backend(model)
            self.last_report = PruneReport(backend=backend.name, params_before=before, params_after=before)
        return save_model(model, save_dir, self.last_report)

    def register_method(self, method: BasePruningMethod) -> None:
        self.registry.register_method(method)
        self.stages = self._build_stages()

    def list_methods(self) -> dict[str, list[str]]:
        return self.registry.list_methods()

    def _build_stages(self) -> list[PruningStage]:
        stages: list[PruningStage] = []
        for domain_name, spec in self.config.resolved_methods().items():
            binding = self.registry.get_binding(domain_name, spec.name)
            stages.append(
                PruningStage(
                    domain_name=domain_name,
                    domain=binding.domain,
                    method=binding.method,
                    spec=spec,
                )
            )
        return stages
