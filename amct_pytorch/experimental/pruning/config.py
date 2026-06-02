from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping


@dataclass(slots=True)
class MethodSpec:
    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PruneConfig:
    methods: Dict[str, MethodSpec | Dict[str, Any] | str] = field(default_factory=dict)
    min_channels: int = 4
    min_neurons: int = 16
    min_experts: int = 1
    skip_layers: List[str] = field(default_factory=list)
    copy_model: bool = True
    verbose: bool = True
    save_dtype: str = "state_dict"
    backend_overrides: Dict[str, Any] = field(default_factory=dict)
    missing_data_policy: str = "warn_skip"
    stage_error_policy: str = "raise"

    @staticmethod
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

    @staticmethod
    def _validate_method_kwargs(spec: MethodSpec) -> None:
        if not spec.name:
            raise ValueError("Method name must be a non-empty string.")
        if "prune_ratio" in spec.kwargs:
            try:
                prune_ratio = float(spec.kwargs["prune_ratio"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"prune_ratio for method '{spec.name}' must be numeric.") from exc
            if not (0.0 <= prune_ratio < 1.0):
                raise ValueError(f"prune_ratio for method '{spec.name}' must be in [0.0, 1.0). Got {prune_ratio}.")
        if "seed" in spec.kwargs:
            try:
                int(spec.kwargs["seed"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"seed for method '{spec.name}' must be an integer-like value.") from exc

    def resolved_methods(self) -> Dict[str, MethodSpec]:
        resolved: Dict[str, MethodSpec] = {
            "cnn": MethodSpec("variance_channel", {"prune_ratio": 0.30}),
            "dense": MethodSpec("low_variance", {"prune_ratio": 0.50}),
            "moe": MethodSpec("activation_count", {"prune_ratio": 0.50}),
        }
        for domain, spec in self.methods.items():
            resolved[domain] = self._normalize_method_spec(spec)
        for spec in resolved.values():
            self._validate_method_kwargs(spec)
        return resolved

    def validate(self) -> None:
        if self.min_channels < 1:
            raise ValueError("min_channels must be >= 1")
        if self.min_neurons < 1:
            raise ValueError("min_neurons must be >= 1")
        if self.min_experts < 1:
            raise ValueError("min_experts must be >= 1")
        if self.missing_data_policy not in {"warn_skip", "raise"}:
            raise ValueError("missing_data_policy must be one of {'warn_skip', 'raise'}")
        if self.stage_error_policy not in {"raise", "warn_skip"}:
            raise ValueError("stage_error_policy must be one of {'raise', 'warn_skip'}")

        for spec in self.methods.values():
            normalized = self._normalize_method_spec(spec)
            self._validate_method_kwargs(normalized)
