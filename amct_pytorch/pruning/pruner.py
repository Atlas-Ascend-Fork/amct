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

import copy
from dataclasses import dataclass
from typing import Optional

import torch.nn as nn

from ..common.utils.log import LOGGER

from . import simulate
from .compat import detect_backend, patch_common_config
from .config import MethodSpec, PruneConfig
from .context import BatchAdapter, PruneContext
from .domains.base import BasePruningDomain
from .prune_op.base import BasePruningMethod
from .registry import RegistryMapping, create_default_registry, get_binding
from .report import PruneReport
from .utils import clone_model_if_needed, count_parameters


def _target_paths(target) -> list[str]:
    paths: list[str] = []
    for attr in (
        "producer_path",
        "consumer_path",
        "gate_path",
        "gate_up_path",
        "up_path",
        "down_path",
        "module_path",
        "router_path",
        "experts_path",
    ):
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
    LOGGER.logw(message, "amct_prune")
    report.add_warning(message)


@dataclass
class PruningStage:
    domain_name: str
    domain: BasePruningDomain
    method: BasePruningMethod
    spec: MethodSpec

    def apply(
        self,
        model: nn.Module,
        context: PruneContext,
        report: PruneReport,
        config: PruneConfig,
    ) -> nn.Module:
        """Apply this stage, returning the (possibly rolled-back) model to continue with."""
        targets = self.domain.find_targets(model, config)
        if config.skip_layers:
            targets = [t for t in targets if not _is_skipped(t, config.skip_layers)]
        if not targets and getattr(self.method, "requires_targets", True):
            report.add(
                self.domain_name,
                self.method.name,
                "<none>",
                "No compatible pruning targets were found",
            )
            return model
        if self.method.requires_data and context.data is None:
            message = (
                f"Skipping pruning method '{self.method.name}' for domain '{self.domain_name}'"
                f" because pruning data was not provided. Pass prune(..., data=...) to enable this stage."
            )
            if config.missing_data_policy == "raise":
                raise ValueError(message)
            _warn(report, message)
            return model
        warn_skip = config.stage_error_policy == "warn_skip"
        target_model = copy.deepcopy(model) if warn_skip else model
        if warn_skip and target_model is not model:
            targets = self.domain.find_targets(target_model, config)
            if config.skip_layers:
                targets = [t for t in targets if not _is_skipped(t, config.skip_layers)]
        try:
            self.method.apply(
                target_model, self.domain, targets, context, report, config, self.spec
            )
        except Exception as exc:
            if warn_skip:
                _warn(
                    report,
                    f"Skipping pruning method '{self.method.name}' for domain '{self.domain_name}' after error: {exc}."
                    f" The partially-applied changes from this stage were rolled back; earlier stages remain applied.",
                )
                return model
            raise
        return target_model


class AutoPruner:
    def __init__(
        self, config: PruneConfig | None = None, registry: RegistryMapping | None = None
    ) -> None:
        self.config = config or PruneConfig()
        self.config.validate()
        self.registry = (
            dict(registry) if registry is not None else create_default_registry()
        )
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
            working = stage.apply(working, context, report, self.config)

        if simulate.active() is None:
            # A trial must not rewrite model.config: _clamp_int_attrs only ever lowers
            # num_experts_per_tok, so an aggressive candidate would leave a wrong value
            # behind that no later trial or the final prune could raise back.
            patch_common_config(working)
        report.params_after = count_parameters(working)
        self.last_report = report
        return working

    def _build_stages(self) -> list[PruningStage]:
        stages: list[PruningStage] = []
        for domain_name, spec in self.config.resolved_methods().items():
            binding = get_binding(self.registry, domain_name, spec.name)
            stages.append(
                PruningStage(
                    domain_name=domain_name,
                    domain=binding.domain,
                    method=binding.method,
                    spec=spec,
                )
            )
        return stages
