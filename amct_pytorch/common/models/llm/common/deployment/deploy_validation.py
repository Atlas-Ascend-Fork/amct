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

"""Validate Deployment artifacts using headers and one auxiliary tensor at a time."""

import json
import math
from pathlib import Path

import torch
from safetensors import SafetensorError, safe_open

from amct_pytorch.common.models.llm.common.deployment.deploy_selection import (
    ResolvedDeployPlan,
)

from .schemes import QuantSchemeKey, get_quant_scheme

_FLOAT_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8}
_DTYPE_BYTES = {
    **_FLOAT_BYTES,
    "F8_E4M3": 1,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
}


def _artifact_contract(
    quant_dtype: str, weight_bits: int = 8, activation_bits: int = 8
) -> dict:
    scheme = get_quant_scheme(QuantSchemeKey(quant_dtype, weight_bits, activation_bits))
    return {
        "quant_dtype": scheme.quant_dtype,
        "quant_type": scheme.model_quant_type,
        "suffixes": scheme.tensor_suffixes,
        "weight_dtype": scheme.weight_dtype,
        "weight_bits": scheme.weight_bits,
        "scale_dtype": scheme.scale_dtype,
        "group_size": scheme.group_size,
        "globals": scheme.globals_dict(),
    }


def _resolve_plan_contracts(plan: ResolvedDeployPlan):
    specs = getattr(plan, "module_quant_specs", None)
    if not isinstance(specs, dict) or not specs:
        raise ValueError("Deployment artifact validation requires module_quant_specs")
    module_contracts = {}
    scheme_keys = set()
    for module, spec in specs.items():
        bits = (spec.weight_bits, spec.activation_bits)
        key = QuantSchemeKey(spec.quant_dtype, *bits)
        get_quant_scheme(key)
        scheme_keys.add(key)
        module_contracts[module] = _artifact_contract(
            key.quant_dtype, key.weight_bits, key.activation_bits
        )
    if len(scheme_keys) != 1:
        raise ValueError(
            "Deployment artifact validation requires one Deployment quantization scheme "
            f"per model, got {sorted(scheme_keys, key=lambda key: (key.quant_dtype, key.weight_bits, key.activation_bits))}"
        )
    key = next(iter(scheme_keys))
    return key.quant_dtype, (key.weight_bits, key.activation_bits), module_contracts


def validate_artifacts(output_dir: Path, plan: ResolvedDeployPlan) -> dict:
    """Return a passed/failed report; callers must reject a failed report."""
    root = Path(output_dir)
    counts = dict.fromkeys(
        (
            "missing",
            "extra",
            "duplicate",
            "mapping",
            "dtype",
            "shape",
            "values",
            "description",
            "metadata",
            "files",
        ),
        0,
    )
    errors = []

    def error(kind, message):
        counts[kind] += 1
        # Keep diagnostics bounded even for a corrupt full-model checkpoint.
        if len(errors) < 100:
            errors.append(message)

    report = {
        "status": "failed",
        "selected_modules": len(plan.selected_modules),
        "tensor_count": 0,
        "dynamic_tensor_count": 0,
        "float_tensor_count": 0,
        "total_tensor_bytes": 0,
        "error_counts": counts,
        "errors": errors,
        "contract": {},
        "artifacts": {
            "config": "config.json",
            "description": "quant_model_description.json",
            "index": "model.safetensors.index.json",
            "shards": [],
        },
    }
    try:
        quant_dtype, bits, module_contracts = _resolve_plan_contracts(plan)
    except (AttributeError, TypeError, ValueError) as exc:
        error("metadata", str(exc))
        return report
    contract = _artifact_contract(quant_dtype, *bits)
    report["contract"] = {
        "quant_dtype": quant_dtype,
        "weight_bits": bits[0],
        "activation_bits": bits[1],
        "model_quant_type": contract["quant_type"],
        "group_size": contract["globals"]["group_size"],
    }
    try:
        index = json.loads(
            (root / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
        description = json.loads(
            (root / "quant_model_description.json").read_text(encoding="utf-8")
        )
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
        if not all(
            isinstance(data, dict) for data in (index, description, config, weight_map)
        ):
            raise ValueError("Expected JSON objects")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        error("files", f"Cannot read artifact metadata: {exc}")
        return report

    globals_ = contract["globals"]
    quant_type = contract["quant_type"]
    for key, value in globals_.items():
        if (
            type(description.get(key)) is not type(value)
            or description.get(key) != value
        ):
            error("metadata", f"Invalid description field: {key}")
    if quant_dtype == "int" and "optional" in description:
        error("metadata", "Rotation optional metadata must be absent")
    quantization_config = config.get("quantization_config")
    if quantization_config is not None:
        if not isinstance(quantization_config, dict):
            error("metadata", "config.json quantization_config must be an object")
        else:
            declared_type = quantization_config.get("model_quant_type")
            if declared_type is not None and declared_type != quant_type:
                error(
                    "metadata",
                    "config.json model_quant_type conflicts with deployment plan: "
                    f"{declared_type!r} != {quant_type!r}",
                )
    if config.get("is_rot_used") not in (None, False):
        error("metadata", "config.json must not enable rotation")
    dynamic_contracts = {
        module + suffix: module_contract
        for module, module_contract in module_contracts.items()
        for suffix in module_contract["suffixes"]
    }
    dynamic = set(dynamic_contracts)
    headers = {}
    locations = {}
    shard_paths = sorted(root.glob("*.safetensors"))
    report["artifacts"]["shards"] = [path.name for path in shard_paths]
    actual_shards = set(report["artifacts"]["shards"])
    indexed_shards = set()
    for name, file_name in weight_map.items():
        if (
            not isinstance(file_name, str)
            or Path(file_name).name != file_name
            or not file_name.endswith(".safetensors")
        ):
            error(
                "files",
                f"Index entry {name} must reference a local shard filename: "
                f"{file_name!r}",
            )
            continue
        indexed_shards.add(file_name)
    for file_name in sorted(indexed_shards - actual_shards):
        error("files", f"Index references missing shard: {file_name}")
    for file_name in sorted(actual_shards - indexed_shards):
        error("files", f"Unreferenced shard in output directory: {file_name}")
    for path in shard_paths:
        if path.is_symlink():
            error("files", f"Symlink shard is not allowed: {path.name}")
            continue
        try:
            with safe_open(str(path), framework="pt", device="cpu") as shard:
                for name in shard.keys():  # noqa: SIM118
                    if name in headers:
                        error("duplicate", f"Duplicate tensor: {name}")
                    view = shard.get_slice(name)
                    dtype, shape = view.get_dtype(), view.get_shape()
                    headers[name] = (dtype, shape)
                    locations[name] = path.name
                    if dtype not in _DTYPE_BYTES:
                        error("dtype", f"Unsupported dtype {dtype}: {name}")
                    else:
                        report["total_tensor_bytes"] += (
                            math.prod(shape) * _DTYPE_BYTES[dtype]
                        )
                    if (
                        name in dynamic
                        and dynamic_contracts[name]["quant_dtype"] in {"int", "hifp"}
                        and name.endswith((".weight_scale", ".weight_offset"))
                    ):
                        tensor = shard.get_tensor(name)
                        if name.endswith(".weight_scale"):
                            valid = torch.isfinite(tensor).all() and (tensor > 0).all()
                        else:
                            valid = (tensor == 0).all()
                        if not valid:
                            error("values", f"Invalid auxiliary values: {name}")
                        del tensor
        except (OSError, ValueError, SafetensorError) as exc:
            error("files", f"Cannot inspect {path.name}: {exc}")

    names = set(headers)
    if not names:
        error("missing", "No output tensors")
    unexpected_offsets = {
        module + ".weight_offset"
        for module, module_contract in module_contracts.items()
        if ".weight_offset" not in module_contract["suffixes"]
    } & names
    for name in sorted(unexpected_offsets):
        error("description", f"{quant_type} must not contain weight_offset: {name}")
    described = set(description) - set(globals_)
    for label, expected in (("index", set(weight_map)), ("description", described)):
        for name in sorted(expected - names):
            error("missing", f"{label} references missing tensor: {name}")
        for name in sorted(names - expected):
            error("extra", f"Tensor absent from {label}: {name}")
    for name in sorted(dynamic - names):
        error("missing", f"Missing quantized tensor: {name}")
    for name in names:
        if weight_map.get(name) != locations[name]:
            error("mapping", f"Wrong shard mapping: {name}")
        expected_type = (
            dynamic_contracts[name]["quant_type"] if name in dynamic else "FLOAT"
        )
        if description.get(name) != expected_type:
            error("description", f"Wrong quantization type: {name}")
        dtype, shape = headers[name]
        if name in dynamic:
            module_contract = dynamic_contracts[name]
            expected_dtype = (
                module_contract["weight_dtype"]
                if name.endswith(".weight")
                else module_contract["scale_dtype"]
            )
            if dtype != expected_dtype:
                error("dtype", f"Wrong dynamic dtype: {name}")
        elif dtype not in _FLOAT_BYTES:
            error("dtype", f"FLOAT tensor is not floating point: {name}")
        if name == "rot.weight" or name.endswith(
            (
                ".rot.weight",
                ".input_scale",
                ".input_offset",
                ".quant_bias",
                ".deq_scale",
            )
        ):
            error("description", f"Unexpected rotation/static parameter: {name}")
    for module, module_contract in module_contracts.items():
        weight = headers.get(module + ".weight")
        if weight is None:
            continue
        shape = weight[1]
        if len(shape) != 2 or any(d <= 0 for d in shape):
            error("shape", f"Weight must be nonempty 2-D: {module}")
            continue
        # Stored MXFP4 columns contain two logical weights per UINT8 element.
        input_size = shape[1] * (8 // module_contract["weight_bits"])
        for suffix in module_contract["suffixes"][1:]:
            auxiliary = headers.get(module + suffix)
            expected_shape = [shape[0], 1]
            if module_contract["group_size"]:
                expected_shape = [shape[0], input_size // module_contract["group_size"]]
            if auxiliary is not None and auxiliary[1] != expected_shape:
                error(
                    "shape",
                    f"Auxiliary must have shape {expected_shape}: {module + suffix}",
                )
        if (
            module_contract["group_size"]
            and input_size % module_contract["group_size"] != 0
        ):
            error(
                "shape",
                f"{module_contract['quant_type']} logical input size {input_size} "
                f"must be divisible by {module_contract['group_size']}: {module}",
            )
    try:
        main_layers = int(config.get("num_hidden_layers", 0))
        mtp_layers = int(config.get("num_nextn_predict_layers", 0))
        mtp_prefixes = tuple(
            f"model.layers.{i}." for i in range(main_layers, main_layers + mtp_layers)
        )
        for name in names:
            if name.startswith(mtp_prefixes) and (
                name in dynamic or description.get(name) != "FLOAT"
            ):
                error("description", f"MTP must remain FLOAT: {name}")
        if index.get("metadata", {}).get("total_size") != report["total_tensor_bytes"]:
            error("metadata", "Index total_size differs from actual tensor bytes")
    except (ValueError, TypeError, AttributeError) as exc:
        error("metadata", f"Invalid config/index metadata: {exc}")
    report.update(
        tensor_count=len(names),
        dynamic_tensor_count=len(names & dynamic),
        float_tensor_count=len(names - dynamic),
    )
    if not any(counts.values()):
        report["status"] = "passed"
    return report
