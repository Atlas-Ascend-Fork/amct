# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

"""Contracts shared by tensor deployment producers and validators."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantSchemeKey:
    """Complete module quantization identity used by a deployment artifact."""

    quant_dtype: str
    weight_bits: int
    activation_bits: int


@dataclass(frozen=True)
class QuantScheme:
    """VLLM metadata and tensor contract for one supported scheme."""

    key: QuantSchemeKey
    model_quant_type: str
    group_size: int
    tensor_suffixes: tuple[str, ...]
    weight_dtype: str
    scale_dtype: str
    global_fields: tuple[tuple[str, object], ...]

    @property
    def quant_dtype(self):
        return self.key.quant_dtype

    @property
    def weight_bits(self):
        return self.key.weight_bits

    @property
    def activation_bits(self):
        return self.key.activation_bits

    def globals_dict(self):
        return dict(self.global_fields)


_INT_W8A8 = QuantSchemeKey("int", 8, 8)
_MXFP_W8A8 = QuantSchemeKey("mxfp", 8, 8)
_MXFP_W4A8 = QuantSchemeKey("mxfp", 4, 8)

QUANT_SCHEMES = {
    _INT_W8A8: QuantScheme(
        key=_INT_W8A8,
        model_quant_type="W8A8_DYNAMIC",
        group_size=0,
        tensor_suffixes=(".weight", ".weight_scale", ".weight_offset"),
        weight_dtype="I8",
        scale_dtype="BF16",
        global_fields=(
            ("version", "1.0.0"),
            ("model_quant_type", "W8A8_DYNAMIC"),
            ("metadata", {}),
            ("group_size", 0),
            ("is_rot_used", False),
        ),
    ),
    _MXFP_W8A8: QuantScheme(
        key=_MXFP_W8A8,
        model_quant_type="W8A8_MXFP8",
        group_size=32,
        tensor_suffixes=(".weight", ".weight_scale"),
        weight_dtype="F8_E4M3",
        scale_dtype="U8",
        global_fields=(
            ("version", "1.0.0"),
            ("model_quant_type", "W8A8_MXFP8"),
            ("metadata", {}),
            ("group_size", 32),
            ("optional", {}),
        ),
    ),
    _MXFP_W4A8: QuantScheme(
        key=_MXFP_W4A8,
        model_quant_type="W4A8_MXFP",
        group_size=32,
        tensor_suffixes=(".weight", ".weight_scale"),
        weight_dtype="U8",
        scale_dtype="U8",
        global_fields=(
            ("version", "1.0.0"),
            ("model_quant_type", "W4A8_MXFP"),
            ("metadata", {}),
            ("group_size", 32),
            ("optional", {}),
        ),
    ),
}

PLATFORM_QUANT_DTYPES = {
    "A3": frozenset({"int"}),
    "ascend950": frozenset({"int", "mxfp"}),
}
DEFAULT_DEPLOY_PLATFORM = "A3"


def resolve_deploy_platform(deploy_platform: str | None) -> str:
    """Resolve the common deployment platform default."""
    return deploy_platform or DEFAULT_DEPLOY_PLATFORM


def get_quant_scheme(key: QuantSchemeKey) -> QuantScheme:
    """Resolve a complete supported scheme or raise a clear configuration error."""
    try:
        return QUANT_SCHEMES[key]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Unsupported deployment quantization scheme: "
            f"({getattr(key, 'quant_dtype', None)!r}, "
            f"{getattr(key, 'weight_bits', None)!r}, "
            f"{getattr(key, 'activation_bits', None)!r}); "
            "supported schemes are int/W8A8, mxfp/W8A8, mxfp/W4A8"
        ) from exc


def _selection_schemes(selection):
    rules = getattr(selection, "quant_layers", ())
    keys = set()
    for rule in rules:
        spec = getattr(rule, "quant_spec", None)
        if spec is None:
            raise ValueError("Deployment JSON requires structured quant_layers rules")
        keys.add(
            QuantSchemeKey(spec.quant_dtype, spec.weight_bits, spec.activation_bits)
        )
    if not keys:
        raise ValueError("Deployment requires a nonempty quantization scheme")
    return keys


def validate_selection_quant_scheme(selection) -> QuantScheme:
    """Validate that a JSON selection describes one supported model scheme."""
    keys = _selection_schemes(selection)
    if len(keys) != 1:
        raise ValueError(
            "Deployment requires one quantization scheme per model, "
            f"got {sorted(keys, key=lambda key: (key.quant_dtype, key.weight_bits, key.activation_bits))}"
        )
    return get_quant_scheme(next(iter(keys)))


def validate_platform_quant_scheme(selection, deploy_platform: str) -> QuantScheme:
    """Validate the JSON scheme against the requested Deployment platform."""
    try:
        allowed = PLATFORM_QUANT_DTYPES[deploy_platform]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported Deployment platform: {deploy_platform!r}"
        ) from exc
    scheme = validate_selection_quant_scheme(selection)
    if scheme.quant_dtype not in allowed:
        supported_platforms = sorted(
            platform
            for platform, dtypes in PLATFORM_QUANT_DTYPES.items()
            if platform != deploy_platform and scheme.quant_dtype in dtypes
        )
        hint = (
            f" Use --deploy_platform {' or '.join(supported_platforms)}."
            if supported_platforms
            else ""
        )
        raise ValueError(
            f"Deployment platform {deploy_platform} does not support quant_dtype "
            f"{scheme.quant_dtype!r}; supported values: {sorted(allowed)}.{hint}"
        )
    return scheme
