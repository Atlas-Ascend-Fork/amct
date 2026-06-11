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

from dataclasses import InitVar, dataclass, field
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
    verbose: bool = True
    missing_data_policy: str = "warn_skip"
    stage_error_policy: str = "raise"
    #: Per-layer prune-ratio allocation; None = uniform (default). Shaped like
    #: {"strategy": "sensitivity", "ref_ratio": 0.5, "min_ratio": 0.05,
    #:  "max_ratio": 0.9, "guard": "calib_nll"}.
    allocation: Optional[Dict[str, Any]] = None
    #: Internal clone-control flag -- NOT a user-facing config field. Whether the pruner
    #: deep-copies the model before pruning (direct prune clones by default; auto-prune
    #: and the optimizer Pass force it off for in-place semantics). It is force-managed by
    #: api.py / the search drivers, never set by the end user, so it is kept as a private
    #: attribute (``_copy_model``) rather than a public field. The public ``copy_model``
    #: name is preserved as a constructor alias + property for internal call sites only.
    _copy_model: bool = field(default=True, repr=False)
    #: Constructor alias for the private ``_copy_model`` flag (internal use). ``None`` means
    #: "leave the default / round-tripped value untouched". An init-only var, so ``copy_model``
    #: does not appear among the public dataclass fields.
    copy_model: InitVar[Optional[bool]] = None

    def __post_init__(self, copy_model: Optional[bool]) -> None:
        if copy_model is not None:
            self._copy_model = copy_model

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


# ``copy_model`` is an internal clone-control flag, not a user-facing field; it is exposed as
# a read/write property backed by the private ``_copy_model`` slot. Assigned after the class is
# built so it overrides the InitVar's leftover class attribute (the InitVar shares the name to
# accept ``PruneConfig(copy_model=...)`` from internal call sites).
def _get_copy_model(self: PruneConfig) -> bool:
    return self._copy_model


def _set_copy_model(self: PruneConfig, value: bool) -> None:
    self._copy_model = value


PruneConfig.copy_model = property(_get_copy_model, _set_copy_model)
