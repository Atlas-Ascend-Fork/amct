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
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Tuple


@dataclass
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


#: Spec-level keys that ``_normalize_method_spec`` folds in alongside the method kwargs.
_SPEC_LEVEL_KEYS = frozenset({"menu"})


@lru_cache(maxsize=1)
def _builtin_accepted_kwargs() -> Dict[Tuple[str, str], Optional[frozenset]]:
    """(domain, method) -> declared kwargs, for the methods shipped in the default registry."""
    from .registry import create_default_registry  # local: registry imports this module

    return {
        key: getattr(binding.method, "accepted_kwargs", None)
        for key, binding in create_default_registry().items()
    }


def _menu_variants(spec: MethodSpec) -> List[MethodSpec]:
    """Normalised specs of a method ``menu``, so a typo inside a variant is caught too."""
    menu = spec.kwargs.get("menu")
    if not isinstance(menu, (list, tuple)):
        return []
    variants = []
    for entry in menu:
        if not (isinstance(entry, (list, tuple)) and len(entry) == 2):
            continue
        try:
            variants.append(_normalize_method_spec(entry[1]))
        except (TypeError, ValueError):
            continue
    return variants


def _validate_method_kwargs(spec: MethodSpec, domain_name: str = "") -> None:
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
    # An unknown kwarg used to be dropped in silence, which let a typo such as
    # 'prune_rate' fall through to the default ratio and cut far more than asked.
    accepted = _builtin_accepted_kwargs().get((domain_name, spec.name))
    if accepted is not None:
        unknown = sorted(set(spec.kwargs) - set(accepted) - _SPEC_LEVEL_KEYS)
        if unknown:
            raise ValueError(
                f"Unknown kwarg(s) {unknown} for method '{spec.name}' in domain "
                f"'{domain_name}'. Accepted: {sorted(accepted)}."
            )


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


@dataclass
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

        for domain_name, spec in self.methods.items():
            normalized = _normalize_method_spec(spec)
            _validate_method_kwargs(normalized, domain_name)
            for variant in _menu_variants(normalized):
                _validate_method_kwargs(variant, domain_name)

    def resolved_methods(self) -> Dict[str, MethodSpec]:
        """Per-domain method specs. A domain ``methods`` does not name is left untouched.

        Naming no domain at all means "prune everything at the defaults"; naming even one
        means the caller chose their targets, so the rest stay at prune_ratio 0.0 rather
        than being cut at a default they never asked for.
        """
        defaults = {
            "cnn": MethodSpec("variance_channel", {"prune_ratio": 0.30}),
            "dense": MethodSpec("low_variance", {"prune_ratio": 0.50}),
            "moe": MethodSpec("activation_count", {"prune_ratio": 0.50}),
        }
        if not self.methods:
            resolved = defaults
        else:
            resolved = {
                domain: MethodSpec(spec.name, {**spec.kwargs, "prune_ratio": 0.0})
                for domain, spec in defaults.items()
            }
            for domain, spec in self.methods.items():
                resolved[domain] = _normalize_method_spec(spec)
        for spec in resolved.values():
            _validate_method_kwargs(spec)
        return resolved
