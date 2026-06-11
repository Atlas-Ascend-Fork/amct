# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
from typing import Any, Dict, List, Mapping, Optional


@dataclass(slots=True)
class MethodSpec:
    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)


def _normalize_method_spec(spec: MethodSpec | Dict[str, Any] | str) -> MethodSpec:
    if isinstance(spec, MethodSpec):
        return MethodSpec(name=spec.name, kwargs=dict(spec.kwargs))
    if isinstance(spec, str):
        return MethodSpec(name=spec, kwargs={})
    if isinstance(spec, Mapping):
        if "name" not in spec:
            raise ValueError("Method spec dict must include a 'name' field.")
        kwargs = dict(spec.get("kwargs", {}))
        extra = {k: v for k, v in spec.items() if k not in {"name", "kwargs"}}
        kwargs.update(extra)
        return MethodSpec(name=str(spec["name"]), kwargs=kwargs)
    raise TypeError(f"Unsupported method spec type: {type(spec)!r}")


def _validate_method_kwargs(spec: MethodSpec) -> None:
    if not spec.name:
        raise ValueError("Method name must be a non-empty string.")
    if "prune_ratio" in spec.kwargs:
        try:
            prune_ratio = float(spec.kwargs["prune_ratio"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"prune_ratio for method '{spec.name}' must be numeric."
            ) from exc
        if not (0.0 <= prune_ratio < 1.0):
            raise ValueError(
                f"prune_ratio for method '{spec.name}' must be in [0.0, 1.0). Got {prune_ratio}."
            )
    if "seed" in spec.kwargs:
        try:
            int(spec.kwargs["seed"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"seed for method '{spec.name}' must be an integer-like value."
            ) from exc


def _validate_allocation(alloc: Optional[Mapping[str, Any]]) -> None:
    if alloc is None:
        return
    if not isinstance(alloc, Mapping):
        raise TypeError(f"allocation must be a dict or None, got {type(alloc)!r}")
    if alloc.get("strategy", "uniform") not in {"uniform", "sensitivity"}:
        raise ValueError("allocation.strategy must be 'uniform' or 'sensitivity'")
    if alloc.get("guard", "calib_nll") not in {"calib_nll", "none"}:
        raise ValueError("allocation.guard must be 'calib_nll' or 'none'")
    for key in ("ref_ratio", "min_ratio", "max_ratio"):
        if key in alloc:
            val = float(alloc[key])
            if not (0.0 <= val < 1.0):
                raise ValueError(f"allocation.{key} must be in [0.0, 1.0). Got {val}.")
    lo = float(alloc.get("min_ratio", 0.05))
    hi = float(alloc.get("max_ratio", 0.9))
    if lo > hi:
        raise ValueError(f"allocation.min_ratio ({lo}) must be <= max_ratio ({hi}).")


@dataclass(slots=True)
class PruneConfig:
    methods: Dict[str, MethodSpec | Dict[str, Any] | str] = field(default_factory=dict)
    min_channels: int = 4
    min_neurons: int = 16
    min_experts: int = 1
    skip_layers: List[str] = field(default_factory=list)
    missing_data_policy: str = "warn_skip"
    stage_error_policy: str = "raise"
    allocation: Optional[Dict[str, Any]] = None
    copy_model: bool = field(default=True, repr=False)

    def validate(self) -> None:
        if self.min_channels < 1:
            raise ValueError("min_channels must be >= 1")
        if self.min_neurons < 1:
            raise ValueError("min_neurons must be >= 1")
        if self.min_experts < 1:
            raise ValueError("min_experts must be >= 1")
        if self.missing_data_policy not in {"warn_skip", "raise"}:
            raise ValueError(
                "missing_data_policy must be one of {'warn_skip', 'raise'}"
            )
        if self.stage_error_policy not in {"raise", "warn_skip"}:
            raise ValueError("stage_error_policy must be one of {'raise', 'warn_skip'}")
        _validate_allocation(self.allocation)

        for spec in self.methods.values():
            normalized = _normalize_method_spec(spec)
            _validate_method_kwargs(normalized)

    def resolved_methods(self) -> Dict[str, MethodSpec]:
        resolved: Dict[str, MethodSpec] = {
            "cnn": MethodSpec("variance_channel", {"prune_ratio": 0.30}),
            "dense": MethodSpec("low_variance", {"prune_ratio": 0.50}),
            "moe": MethodSpec("activation_count", {"prune_ratio": 0.50}),
        }
        for domain, spec in self.methods.items():
            resolved[domain] = _normalize_method_spec(spec)
        for spec in resolved.values():
            _validate_method_kwargs(spec)
        return resolved
