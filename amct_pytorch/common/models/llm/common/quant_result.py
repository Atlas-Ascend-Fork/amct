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

"""V2 quant-result adapter for the blockwise LLM PTQ workflow."""

from __future__ import annotations

import copy
import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from amct_pytorch.classic.quant_result import (
    QuantResultWriter,
    digest_of,
    load_quant_manifest,
    read_quant_unit,
    sha256_of_file,
    validate_quant_result_dir,
    validate_result_directory,
)
from amct_pytorch.quantization.bit_policy import BitPolicy


LLM_ADAPTER_FORMAT = "amct-llm-ptq-unit"
LLM_ADAPTER_SCHEMA_VERSION = 1
LLM_PAYLOAD_FORMAT = "amct-llm-ptq-payload"
LLM_PAYLOAD_SCHEMA_VERSION = 1
LLM_QUANT_MODULE_TYPE = "LlmPtqUnit"
LLM_ARTIFACT_KIND_DIRECT_CONFIG = "direct_config"
LLM_ARTIFACT_KIND_PTQ_PARAMS = "ptq_params"
_LLM_ARTIFACT_KINDS = {
    LLM_ARTIFACT_KIND_DIRECT_CONFIG,
    LLM_ARTIFACT_KIND_PTQ_PARAMS,
}

_BASE_CONFIG_FIELDS = {
    "adapter_format",
    "adapter_schema_version",
    "model_name",
    "quant_target",
    "quant_dtype",
    "bit_policy",
    "algos",
    "artifact_kind",
    "is_per_tensor",
    "k_size",
}
_UNIT_CONFIG_FIELDS = _BASE_CONFIG_FIELDS | {"unit"}
_UNIT_ID_FIELDS = {"layer_idx", "kind", "name", "metadata"}
_PAYLOAD_FIELDS = {"format", "schema_version", "structure"}


@dataclass(frozen=True)
class LlmPtqUnitSpec:
    """Stable identity of one :class:`PtqUnit` inside its source model."""

    layer_name: str
    layer_idx: int
    kind: str
    name: str
    module_type: str
    metadata: dict[str, Any]

    @property
    def key(self):
        return self.layer_idx, self.kind, self.name


def normalize_quant_result_dirs(value) -> tuple[str, ...]:
    """Normalize the CLI/API form while keeping ``None`` as the legacy path."""

    if value is None or value == "":
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, (list, tuple)):
        raise TypeError(
            "quant_result_dir must be a path, a list of paths, or None, got "
            f"{type(value).__name__}."
        )
    paths = tuple(value)
    if not paths or any(not isinstance(path, str) or not path for path in paths):
        raise ValueError("quant_result_dir paths must be non-empty strings.")
    if len(set(paths)) != len(paths):
        raise ValueError("quant_result_dir contains duplicate paths.")
    return paths


def _load_json_object(path: Path, what: str) -> dict:
    with path.open(encoding="utf-8") as file_handle:
        value = json.load(file_handle)
    if not isinstance(value, dict):
        raise ValueError(f"{what} '{path}' must contain a JSON object.")
    return value


def _resolve_checkpoint_file(model_dir: Path, file_name: str) -> Path:
    if not isinstance(file_name, str) or not file_name:
        raise ValueError(
            f"checkpoint file name must be a non-empty string: {file_name!r}"
        )
    relative = Path(file_name)
    if relative.is_absolute() or len(relative.parts) != 1 or relative.name != file_name:
        raise ValueError(f"unsafe checkpoint file name in weight index: {file_name!r}")
    path = (model_dir / file_name).resolve()
    if path.parent != model_dir.resolve() or not path.is_file():
        raise FileNotFoundError(f"checkpoint weight file is missing: {path}")
    return path


def compute_checkpoint_fingerprint(model_path: str) -> dict[str, str]:
    """Fingerprint a sharded checkpoint without materializing its state dict.

    The state digest binds the canonical weight map to every referenced shard's
    byte size and SHA-256. It therefore describes the exact immutable source
    checkpoint while using constant memory for multi-gigabyte models.
    """

    model_dir = Path(model_path).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"source model directory does not exist: {model_dir}")
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"source model config is missing: {config_path}")
    config = _load_json_object(config_path, "source model config")
    model_type = config.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        raise ValueError("source model config.model_type must be a non-empty string.")

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = _load_json_object(index_path, "source model weight index")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("source model weight index.weight_map must be non-empty.")
        for tensor_name, file_name in weight_map.items():
            if not isinstance(tensor_name, str) or not tensor_name:
                raise ValueError(
                    "source model weight-map keys must be non-empty strings."
                )
            if not isinstance(file_name, str) or not file_name:
                raise ValueError(
                    f"source model weight-map value for '{tensor_name}' is invalid."
                )
        file_names = sorted(set(weight_map.values()))
    else:
        single_file = model_dir / "model.safetensors"
        if not single_file.is_file():
            raise FileNotFoundError(
                "source model has neither model.safetensors.index.json nor "
                "model.safetensors."
            )
        weight_map = {"__single_checkpoint__": single_file.name}
        file_names = [single_file.name]

    files = []
    for file_name in file_names:
        file_path = _resolve_checkpoint_file(model_dir, file_name)
        files.append(
            {
                "file": file_name,
                "file_size": file_path.stat().st_size,
                "sha256": sha256_of_file(str(file_path)),
            }
        )
    state_descriptor = {"weight_map": weight_map, "files": files}
    return {
        "model_type": model_type,
        "config_sha256": digest_of(config),
        "state_dict_sha256": digest_of(state_descriptor),
    }


def validate_checkpoint_fingerprint(model_path: str, expected: dict[str, str]):
    actual = compute_checkpoint_fingerprint(model_path)
    mismatches = [
        field
        for field in ("model_type", "config_sha256", "state_dict_sha256")
        if actual.get(field) != expected.get(field)
    ]
    if mismatches:
        details = "; ".join(
            f"{field}: artifact has {expected.get(field)!r}, source has {actual.get(field)!r}"
            for field in mismatches
        )
        raise ValueError(f"source checkpoint does not match quant result: {details}")
    return actual


def _module_path(block, module) -> str:
    matches = [name for name, candidate in block.named_modules() if candidate is module]
    if len(matches) != 1:
        raise ValueError(
            "each LLM PTQ unit must reference exactly one module inside its block; "
            f"found {len(matches)} path(s)."
        )
    return matches[0]


def _unit_spec(pipeline, layer_idx: int, block, unit) -> LlmPtqUnitSpec:
    if unit.layer_idx != layer_idx:
        raise ValueError(
            f"PTQ unit '{unit.name}' reports layer {unit.layer_idx}, expected {layer_idx}."
        )
    relative_name = _module_path(block, unit.module)
    prefix = pipeline.get_layer_weight_prefix(layer_idx)
    layer_name = f"{prefix}{relative_name}" if relative_name else prefix.rstrip(".")
    return LlmPtqUnitSpec(
        layer_name=layer_name,
        layer_idx=layer_idx,
        kind=unit.kind,
        name=unit.name,
        module_type=type(unit.module).__name__,
        metadata=dict(unit.metadata),
    )


def discover_llm_ptq_units(pipeline, start_layer: int, end_layer: int):
    """Discover the complete unit set before the artifact directory is created."""

    specs = []
    seen_keys = set()
    seen_paths = set()
    for layer_idx in range(start_layer, end_layer):
        block = pipeline.build_quant_block(layer_idx)
        try:
            units = list(pipeline.iter_ptq_units(layer_idx, block))
            if not units:
                raise ValueError(f"no PTQ units found for layer {layer_idx}.")
            for unit in units:
                spec = _unit_spec(pipeline, layer_idx, block, unit)
                if spec.key in seen_keys:
                    raise ValueError(f"duplicate LLM PTQ unit identity: {spec.key!r}")
                if spec.layer_name in seen_paths:
                    raise ValueError(
                        f"duplicate LLM PTQ module path: {spec.layer_name}"
                    )
                seen_keys.add(spec.key)
                seen_paths.add(spec.layer_name)
                specs.append(spec)
        finally:
            del block
            gc.collect()
    if not specs:
        raise ValueError("no LLM PTQ units were discovered for quant-result export.")
    return tuple(specs)


def _base_quant_config(args, quant_target: str) -> dict[str, Any]:
    bit_policy = getattr(args, "bit_policy", None)
    bit_policy_cfg = getattr(bit_policy, "cfg", {})
    algos = list(getattr(args, "algos", []))
    if any(not isinstance(algo, str) or not algo for algo in algos):
        raise ValueError("args.algos must contain only non-empty strings.")
    config = {
        "adapter_format": LLM_ADAPTER_FORMAT,
        "adapter_schema_version": LLM_ADAPTER_SCHEMA_VERSION,
        "model_name": str(args.model_name),
        "quant_target": quant_target,
        "quant_dtype": str(getattr(args, "quant_dtype", "")),
        "bit_policy": bit_policy_cfg,
        "algos": algos,
        "artifact_kind": (
            LLM_ARTIFACT_KIND_PTQ_PARAMS if algos else LLM_ARTIFACT_KIND_DIRECT_CONFIG
        ),
        "is_per_tensor": bool(getattr(args, "is_per_tensor", False)),
        "k_size": int(getattr(args, "k_size", 128)),
    }
    for field in ("model_name", "quant_target", "quant_dtype"):
        if not config[field]:
            raise ValueError(f"LLM artifact {field} must be a non-empty string.")
    return config


def _encode_payload(value, tensors: dict[str, torch.Tensor], path: str):
    if isinstance(value, torch.Tensor):
        key = f"tensor_{len(tensors):06d}"
        tensors[key] = value.detach().cpu().contiguous()
        return {"type": "tensor", "key": key}
    if value is None:
        return {"type": "none"}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": value}
    if isinstance(value, float):
        if not torch.isfinite(torch.tensor(value)):
            raise ValueError(f"{path} contains a non-finite float {value!r}.")
        return {"type": "float", "value": value}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, torch.dtype):
        return {"type": "dtype", "value": str(value).removeprefix("torch.")}
    if isinstance(value, dict):
        items = []
        for key in value:
            if not isinstance(key, str) or not key:
                raise TypeError(f"{path} has a non-string or empty key: {key!r}.")
        for key in sorted(value):
            items.append([key, _encode_payload(value[key], tensors, f"{path}.{key}")])
        return {"type": "dict", "items": items}
    if isinstance(value, list):
        return {
            "type": "list",
            "items": [
                _encode_payload(item, tensors, f"{path}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "items": [
                _encode_payload(item, tensors, f"{path}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    raise TypeError(
        f"{path} contains unsupported {type(value).__name__}; LLM PTQ payloads may "
        "contain tensors, JSON scalars, torch.dtype, dict, list, and tuple only."
    )


def flatten_ptq_payload(value):
    tensors: dict[str, torch.Tensor] = {}
    structure = _encode_payload(value, tensors, "ptq_payload")
    return tensors, structure


def _decode_payload(node, tensors, used, path):
    if not isinstance(node, dict) or not isinstance(node.get("type"), str):
        raise ValueError(f"{path} is not a valid LLM PTQ payload node.")
    node_type = node["type"]
    if node_type == "tensor":
        if set(node) != {"type", "key"} or not isinstance(node["key"], str):
            raise ValueError(f"{path} has an invalid tensor reference.")
        key = node["key"]
        if key not in tensors:
            raise ValueError(f"{path} references missing tensor '{key}'.")
        if key in used:
            raise ValueError(f"{path} references tensor '{key}' more than once.")
        used.add(key)
        return tensors[key]
    if node_type == "none":
        if set(node) != {"type"}:
            raise ValueError(f"{path} has invalid none metadata.")
        return None
    if node_type in {"bool", "int", "float", "str"}:
        if set(node) != {"type", "value"}:
            raise ValueError(f"{path} has invalid scalar metadata.")
        value = node["value"]
        expected_type = {"bool": bool, "int": int, "float": float, "str": str}[
            node_type
        ]
        if type(value) is not expected_type:
            raise ValueError(f"{path} scalar type does not match '{node_type}'.")
        return value
    if node_type == "dtype":
        if set(node) != {"type", "value"} or not isinstance(node["value"], str):
            raise ValueError(f"{path} has invalid dtype metadata.")
        dtype = getattr(torch, node["value"], None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"{path} names unsupported torch dtype '{node['value']}'.")
        return dtype
    if node_type in {"list", "tuple"}:
        if set(node) != {"type", "items"} or not isinstance(node["items"], list):
            raise ValueError(f"{path} has invalid {node_type} metadata.")
        items = [
            _decode_payload(item, tensors, used, f"{path}[{index}]")
            for index, item in enumerate(node["items"])
        ]
        return items if node_type == "list" else tuple(items)
    if node_type == "dict":
        if set(node) != {"type", "items"} or not isinstance(node["items"], list):
            raise ValueError(f"{path} has invalid dict metadata.")
        result = {}
        for index, item in enumerate(node["items"]):
            if (
                not isinstance(item, list)
                or len(item) != 2
                or not isinstance(item[0], str)
                or not item[0]
            ):
                raise ValueError(f"{path}.items[{index}] is invalid.")
            key = item[0]
            if key in result:
                raise ValueError(f"{path} contains duplicate key '{key}'.")
            result[key] = _decode_payload(item[1], tensors, used, f"{path}.{key}")
        return result
    raise ValueError(f"{path} has unknown payload node type '{node_type}'.")


def unflatten_ptq_payload(tensors: dict[str, torch.Tensor], structure):
    used = set()
    value = _decode_payload(structure, tensors, used, "ptq_payload")
    unexpected = sorted(set(tensors) - used)
    if unexpected:
        raise ValueError(f"LLM PTQ payload has unreferenced tensor(s): {unexpected}.")
    return value


class LlmQuantResultWriter:
    """Map finalized LLM PTQ units onto the frozen V2 writer."""

    def __init__(self, writer, specs, args, quant_target):
        self._writer = writer
        self._specs = {spec.key: spec for spec in specs}
        self._base_config = _base_quant_config(args, quant_target)

    @property
    def artifact_kind(self):
        return self._base_config["artifact_kind"]

    @classmethod
    def create(cls, result_dir, specs, source_model, args, quant_target):
        specs = tuple(specs)
        writer = QuantResultWriter.create(
            result_dir, [spec.layer_name for spec in specs], source_model
        )
        return cls(writer, specs, args, quant_target)

    @property
    def manifest_path(self):
        return self._writer.manifest_path

    @property
    def pending_units(self):
        return self._writer.pending_units

    def write_unit(self, unit, result):
        key = (unit.layer_idx, unit.kind, unit.name)
        spec = self._specs.get(key)
        if spec is None:
            raise ValueError(f"unexpected LLM PTQ unit during artifact write: {key!r}")
        if type(unit.module).__name__ != spec.module_type:
            raise ValueError(
                f"LLM PTQ unit {key!r} changed module type from '{spec.module_type}' "
                f"to '{type(unit.module).__name__}'."
            )
        if dict(unit.metadata) != spec.metadata:
            raise ValueError(f"LLM PTQ unit {key!r} changed metadata before write.")
        _validate_payload_for_artifact_kind(self._base_config, result)
        tensors, structure = flatten_ptq_payload(result)
        unit_config = dict(self._base_config)
        unit_config["unit"] = {
            "layer_idx": spec.layer_idx,
            "kind": spec.kind,
            "name": spec.name,
            "metadata": spec.metadata,
        }
        return self._writer.write_unit(
            layer_name=spec.layer_name,
            tensors=tensors,
            non_tensor_params={
                "format": LLM_PAYLOAD_FORMAT,
                "schema_version": LLM_PAYLOAD_SCHEMA_VERSION,
                "structure": structure,
            },
            quant_config=unit_config,
            quant_module_type=LLM_QUANT_MODULE_TYPE,
            ori_module_type=spec.module_type,
        )

    def publish(self):
        if not self._writer.publish_if_complete():
            raise RuntimeError(
                "LLM quant result is incomplete; missing unit(s): "
                + ", ".join(self._writer.pending_units)
            )
        return self._writer.manifest_path


def _require_exact_fields(mapping, fields, what):
    if not isinstance(mapping, dict) or set(mapping) != fields:
        raise ValueError(
            f"{what} must contain exactly {sorted(fields)}, got "
            f"{sorted(mapping) if isinstance(mapping, dict) else type(mapping).__name__}."
        )


def _validate_adapter_entry(entry):
    if entry["quant_module_type"] != LLM_QUANT_MODULE_TYPE:
        raise ValueError(
            f"unit '{entry['layer_name']}' is not an LLM PTQ artifact unit."
        )
    config = entry["quant_config"]
    _require_exact_fields(config, _UNIT_CONFIG_FIELDS, "LLM unit quant_config")
    if config["adapter_format"] != LLM_ADAPTER_FORMAT:
        raise ValueError("unsupported LLM quant-result adapter format.")
    if (
        type(config["adapter_schema_version"]) is not int
        or config["adapter_schema_version"] != LLM_ADAPTER_SCHEMA_VERSION
    ):
        raise ValueError("unsupported LLM quant-result adapter schema version.")
    for field in ("model_name", "quant_target", "quant_dtype"):
        if not isinstance(config[field], str) or not config[field]:
            raise ValueError(f"LLM unit quant_config.{field} must be non-empty.")
    if not isinstance(config["bit_policy"], dict):
        raise ValueError("LLM unit quant_config.bit_policy must be an object.")
    if not isinstance(config["algos"], list) or any(
        not isinstance(item, str) or not item for item in config["algos"]
    ):
        raise ValueError("LLM unit quant_config.algos must be a string list.")
    if (
        not isinstance(config["artifact_kind"], str)
        or config["artifact_kind"] not in _LLM_ARTIFACT_KINDS
    ):
        raise ValueError("unsupported LLM artifact kind.")
    if config["artifact_kind"] == LLM_ARTIFACT_KIND_DIRECT_CONFIG:
        if config["algos"]:
            raise ValueError("direct_config artifacts must not declare PTQ algorithms.")
    elif not config["algos"]:
        raise ValueError("ptq_params artifacts must declare at least one algorithm.")
    if type(config["is_per_tensor"]) is not bool:
        raise ValueError("LLM unit quant_config.is_per_tensor must be bool.")
    if type(config["k_size"]) is not int or config["k_size"] <= 0:
        raise ValueError("LLM unit quant_config.k_size must be a positive int.")
    unit = config["unit"]
    _require_exact_fields(unit, _UNIT_ID_FIELDS, "LLM unit identity")
    if type(unit["layer_idx"]) is not int or unit["layer_idx"] < 0:
        raise ValueError("LLM unit layer_idx must be a non-negative int.")
    for field in ("kind", "name"):
        if not isinstance(unit[field], str) or not unit[field]:
            raise ValueError(f"LLM unit {field} must be non-empty.")
    if not isinstance(unit["metadata"], dict):
        raise ValueError("LLM unit metadata must be an object.")
    payload = entry["non_tensor_params"]
    _require_exact_fields(payload, _PAYLOAD_FIELDS, "LLM payload metadata")
    if payload["format"] != LLM_PAYLOAD_FORMAT:
        raise ValueError("unsupported LLM PTQ payload format.")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != LLM_PAYLOAD_SCHEMA_VERSION
    ):
        raise ValueError("unsupported LLM PTQ payload schema version.")
    return config, unit, payload


def _validate_payload_for_artifact_kind(config, payload):
    if not isinstance(payload, dict):
        raise TypeError("LLM artifact unit payload must be a dict.")
    artifact_kind = config["artifact_kind"]
    if artifact_kind == LLM_ARTIFACT_KIND_DIRECT_CONFIG:
        if payload:
            raise ValueError(
                "direct_config artifact units must have an empty payload; "
                "direct quantization has no calibrated parameters."
            )
        return
    if artifact_kind == LLM_ARTIFACT_KIND_PTQ_PARAMS:
        if not payload:
            raise ValueError(
                "ptq_params artifact units must contain calibrated parameters."
            )
        return
    raise ValueError(f"unsupported LLM artifact kind '{artifact_kind}'.")


class LlmQuantResultReader:
    """Fully verified, in-memory index of one target-specific LLM artifact."""

    def __init__(self, result_dir, manifest):
        self.result_dir = os.path.abspath(os.path.expanduser(result_dir))
        self.manifest = manifest
        self.payloads = {}
        self.entries = {}
        base_config = None
        for entry in manifest["units"]:
            config, unit, payload_meta = _validate_adapter_entry(entry)
            current_base = {key: config[key] for key in _BASE_CONFIG_FIELDS}
            if base_config is None:
                base_config = current_base
            elif current_base != base_config:
                raise ValueError(
                    "LLM quant-result units disagree on construction config."
                )
            key = (unit["layer_idx"], unit["kind"], unit["name"])
            if key in self.entries:
                raise ValueError(f"LLM quant result contains duplicate unit {key!r}.")
            tensors, _, _ = read_quant_unit(
                self.result_dir, entry["layer_name"], manifest
            )
            self.payloads[key] = unflatten_ptq_payload(
                tensors, payload_meta["structure"]
            )
            _validate_payload_for_artifact_kind(config, self.payloads[key])
            self.entries[key] = entry
        self.base_config = base_config
        self.quant_target = base_config["quant_target"]
        self.artifact_kind = base_config["artifact_kind"]

    @classmethod
    def from_validated_manifest(cls, result_dir, manifest):
        return cls(result_dir, manifest)

    @property
    def layer_indices(self):
        return {key[0] for key in self.entries}

    def validate_full_coverage(self, num_layers: int):
        expected = set(range(num_layers))
        if self.layer_indices != expected:
            missing = sorted(expected - self.layer_indices)
            extra = sorted(self.layer_indices - expected)
            raise ValueError(
                f"LLM artifact for target '{self.quant_target}' does not cover the "
                f"whole model; missing layers={missing}, extra layers={extra}."
            )


class LlmQuantResultSet:
    """One or more target artifacts loaded as one transactional deploy source."""

    def __init__(self, readers):
        self.readers = tuple(readers)
        targets = [reader.quant_target for reader in self.readers]
        if len(set(targets)) != len(targets):
            raise ValueError("multiple LLM artifacts describe the same quant_target.")

    @classmethod
    def from_directories(cls, result_dirs, model_path):
        paths = normalize_quant_result_dirs(result_dirs)
        if not paths:
            raise ValueError("at least one quant_result_dir is required.")
        validated = []
        source_model = None
        for path in paths:
            manifest = load_quant_manifest(path)
            validate_result_directory(path, manifest)
            if source_model is None:
                source_model = manifest["source_model"]
            elif manifest["source_model"] != source_model:
                raise ValueError(
                    "LLM artifacts were produced from different source models."
                )
            for entry in manifest["units"]:
                _validate_adapter_entry(entry)
            validated.append((path, manifest))
        validate_checkpoint_fingerprint(model_path, source_model)
        return cls(
            LlmQuantResultReader.from_validated_manifest(path, manifest)
            for path, manifest in validated
        )

    @property
    def quant_targets(self):
        return tuple(reader.quant_target for reader in self.readers)

    def apply_to_args(self, args):
        configs = [reader.base_config for reader in self.readers]
        reference = configs[0]
        shared_fields = _BASE_CONFIG_FIELDS - {"quant_target"}
        for config in configs[1:]:
            mismatches = [
                field for field in shared_fields if config[field] != reference[field]
            ]
            if mismatches:
                raise ValueError(
                    "LLM artifacts disagree on shared construction field(s): "
                    + ", ".join(sorted(mismatches))
                )
        args.model_name = reference["model_name"]
        args.quant_target = list(self.quant_targets)
        args.quant_dtype = reference["quant_dtype"]
        args.bit_policy = BitPolicy(copy.deepcopy(reference["bit_policy"]))
        args.algos = list(reference["algos"])
        args.is_per_tensor = reference["is_per_tensor"]
        args.k_size = reference["k_size"]

    def validate_for_pipeline(self, pipeline):
        targets = tuple(
            pipeline.quant_target
            if isinstance(pipeline.quant_target, (list, tuple))
            else [pipeline.quant_target]
        )
        if set(targets) != set(self.quant_targets):
            raise ValueError(
                f"pipeline targets {targets!r} do not match artifact targets "
                f"{self.quant_targets!r}."
            )
        for reader in self.readers:
            reader.validate_full_coverage(pipeline.num_layers)

    def load_block(self, pipeline, layer_idx: int, block):
        """Load all units into a deep copy; the input block is never mutated."""

        staged = copy.deepcopy(block)
        original_target = pipeline.quant_target
        try:
            for reader in self.readers:
                pipeline.quant_target = [reader.quant_target]
                units = list(pipeline.iter_ptq_units(layer_idx, staged))
                actual = {
                    (unit.layer_idx, unit.kind, unit.name): unit for unit in units
                }
                expected = {
                    key: payload
                    for key, payload in reader.payloads.items()
                    if key[0] == layer_idx
                }
                if set(actual) != set(expected):
                    raise ValueError(
                        f"layer {layer_idx} target '{reader.quant_target}' unit set "
                        f"does not match artifact: model={sorted(actual)}, "
                        f"artifact={sorted(expected)}."
                    )
                for key, unit in actual.items():
                    entry = reader.entries[key]
                    spec = _unit_spec(pipeline, layer_idx, staged, unit)
                    if spec.layer_name != entry["layer_name"]:
                        raise ValueError(
                            f"LLM unit {key!r} path changed from "
                            f"'{entry['layer_name']}' to '{spec.layer_name}'."
                        )
                    if spec.module_type != entry["ori_module_type"]:
                        raise ValueError(
                            f"LLM unit {key!r} module type changed from "
                            f"'{entry['ori_module_type']}' to '{spec.module_type}'."
                        )
                    expected_metadata = entry["quant_config"]["unit"]["metadata"]
                    if spec.metadata != expected_metadata:
                        raise ValueError(
                            f"LLM unit {key!r} metadata does not match artifact."
                        )
                    if reader.artifact_kind == LLM_ARTIFACT_KIND_PTQ_PARAMS:
                        pipeline.ptq_param_handler.load_unit(unit, expected[key])
        finally:
            pipeline.quant_target = original_target
        return staged


__all__ = [
    "LLM_ARTIFACT_KIND_DIRECT_CONFIG",
    "LLM_ARTIFACT_KIND_PTQ_PARAMS",
    "LlmPtqUnitSpec",
    "LlmQuantResultReader",
    "LlmQuantResultSet",
    "LlmQuantResultWriter",
    "compute_checkpoint_fingerprint",
    "discover_llm_ptq_units",
    "flatten_ptq_payload",
    "normalize_quant_result_dirs",
    "unflatten_ptq_payload",
    "validate_checkpoint_fingerprint",
    "validate_quant_result_dir",
]
