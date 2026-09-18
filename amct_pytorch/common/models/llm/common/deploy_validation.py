# coding=utf-8
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

"""Validate Ascend artifacts using headers and one auxiliary tensor at a time."""

import json
import math
from pathlib import Path

import torch
from safetensors import SafetensorError, safe_open

from amct_pytorch.common.models.llm.common.deploy_selection import ResolvedDeployPlan


_FLOAT_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8}
_DTYPE_BYTES = {
    **_FLOAT_BYTES,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
}
_GLOBALS = {
    "version": "1.0.0",
    "model_quant_type": "W8A8_DYNAMIC",
    "metadata": {},
    "group_size": 0,
    "is_rot_used": False,
}


def validate_ascend_artifacts(output_dir: Path, plan: ResolvedDeployPlan) -> dict:
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
    }
    try:
        index = json.loads((root / "model.safetensors.index.json").read_text())
        description = json.loads((root / "quant_model_description.json").read_text())
        config = json.loads((root / "config.json").read_text())
        weight_map = index["weight_map"]
        if not all(
            isinstance(data, dict) for data in (index, description, config, weight_map)
        ):
            raise ValueError("Expected JSON objects")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        error("files", f"Cannot read artifact metadata: {exc}")
        return report

    for key, value in _GLOBALS.items():
        if (
            type(description.get(key)) is not type(value)
            or description.get(key) != value
        ):
            error("metadata", f"Invalid description field: {key}")
    if "optional" in description:
        error("metadata", "Rotation optional metadata must be absent")
    dynamic = {
        m + suffix
        for m in plan.selected_modules
        for suffix in (".weight", ".weight_scale", ".weight_offset")
    }
    headers = {}
    locations = {}
    for path in sorted(root.glob("*.safetensors")):
        if path.is_symlink():
            error("files", f"Symlink shard is not allowed: {path.name}")
            continue
        try:
            with safe_open(str(path), framework="pt", device="cpu") as shard:
                for name in shard.keys():
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
                    if name in dynamic and name.endswith(
                        (".weight_scale", ".weight_offset")
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
    described = set(description) - set(_GLOBALS)
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
        expected_type = "W8A8_DYNAMIC" if name in dynamic else "FLOAT"
        if description.get(name) != expected_type:
            error("description", f"Wrong quantization type: {name}")
        dtype, shape = headers[name]
        if name in dynamic:
            if dtype != ("I8" if name.endswith(".weight") else "BF16"):
                error("dtype", f"Wrong dynamic dtype: {name}")
        elif dtype not in _FLOAT_BYTES:
            error("dtype", f"FLOAT tensor is not floating point: {name}")
        if (
            name == "rot.weight"
            or name.endswith(".rot.weight")
            or name.endswith(
                (".input_scale", ".input_offset", ".quant_bias", ".deq_scale")
            )
        ):
            error("description", f"Unexpected rotation/static parameter: {name}")
    for module in plan.selected_modules:
        weight = headers.get(module + ".weight")
        if weight is None:
            continue
        shape = weight[1]
        if len(shape) != 2 or any(d <= 0 for d in shape):
            error("shape", f"Weight must be nonempty 2-D: {module}")
            continue
        for suffix in (".weight_scale", ".weight_offset"):
            auxiliary = headers.get(module + suffix)
            if auxiliary is not None and auxiliary[1] != [shape[0], 1]:
                error("shape", f"Auxiliary must have shape [out, 1]: {module + suffix}")
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
