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

"""Preparation boundary for the Classic layerwise quantization pipeline."""

from __future__ import annotations

import json
import os
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from amct_pytorch.algorithms import AlgorithmRegistry
from amct_pytorch.common.config import set_default_config
from amct_pytorch.common.config.fields import QuantConfig
from amct_pytorch.common.datasets.ptq_provider import LlmPtqDataProvider
from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm import available_llm_adapters, register_llm_models
from amct_pytorch.common.optimization import SOLVER_REGISTRY, register_solvers
from amct_pytorch.classic.quant_result import QuantResultWriter, dtype_name
from amct_pytorch.quantization.bit_policy import BitPolicy
from amct_pytorch.quantization.dtypes import register_dtype
from amct_pytorch.algorithms.quant import register_algorithms


_SUPPORTED_ALGORITHM = "minmax"
_DEFAULT_QUANT_RESULT_DIR = "./quant_result"
_SUPPORTED_WEIGHT_TYPES = {"int4": 4, "int8": 8}
_SUPPORTED_ACTIVATION_TYPES = {"int8": 8}
_LAYERWISE_CONFIG_FIELDS = {
    "algorithm",
    "batch_num",
    "batch_size",
    "dataset",
    "device",
    "max_seq_length",
    "num_calibration_samples",
    "pipeline",
    "quant_cfg",
    "skip_layers",
    "trust_remote_code",
    "work_dir",
}
_QUANT_CONFIG_FIELDS = {"inputs", "kvcache", "weights"}
_WEIGHT_CONFIG_FIELDS = {"group_size", "strategy", "symmetric", "type"}
_INPUT_CONFIG_FIELDS = {
    "dynamic",
    "enable_quant",
    "strategy",
    "symmetric",
    "type",
}
_KVCACHE_CONFIG_FIELDS = {"enable_quant", "strategy", "symmetric", "type"}
_MOE_CONFIG_FIELDS = (
    "n_routed_experts",
    "num_experts",
    "num_local_experts",
    "num_experts_per_tok",
)


@dataclass
class LayerwiseArgs:
    """Typed subset of CLI-shaped fields consumed by current LLM adapters."""

    model: str
    model_name: str
    quant_result_dir: str
    quant_target: list[str]
    quant_dtype: str
    w_bits: int
    a_bits: int
    bit_policy: BitPolicy
    algos: list[str]
    batch_num: int
    cali_bsz: int
    nsamples: int
    seq_len: int
    device: str
    work_dir: str
    data_dir: str
    output_dir: str
    skip_layers: set[str]
    dataset: str | None
    trust_remote_code: bool
    weight_strategy: str
    weight_symmetric: bool
    activation_strategy: str | None
    activation_symmetric: bool | None
    activation_dynamic: bool
    group_size: int | None
    granularity: str = "block"
    solver: str = "calibrate"
    start_block_idx: int = 0
    end_block_idx: int = 2**31 - 1
    attn_linear_param_dir: str = ""
    attn_cache_param_dir: str = ""
    moe_mlp_param_dir: str = ""


@dataclass
class LayerwiseQuantizationPlan:
    """Validated hand-off consumed by the later Extract/PTQ stages."""

    args: LayerwiseArgs
    adapter: Any
    model_config: dict[str, Any]


@dataclass(frozen=True)
class PlannedLinearUnit:
    """One reloadable Linear unit in a layer's persisted plan."""

    layer_idx: int
    relative_name: str
    layer_name: str
    skip: bool


def load_model_config(model_path: str) -> tuple[str, dict[str, Any]]:
    """Load a local checkpoint config without importing remote model code."""
    if not isinstance(model_path, str):
        raise TypeError(
            "pipeline='layerwise' requires model to be a local directory path "
            "str, got {}.".format(type(model_path).__name__)
        )
    if not model_path.strip():
        raise ValueError("layerwise model path must be a non-empty str.")

    normalized_path = os.path.abspath(os.path.expanduser(model_path))
    if not os.path.exists(normalized_path):
        raise ValueError(
            "layerwise model path '{}' does not exist.".format(normalized_path)
        )
    if not os.path.isdir(normalized_path):
        raise ValueError(
            "layerwise model path '{}' must be a directory.".format(normalized_path)
        )

    config_path = Path(normalized_path) / "config.json"
    if not config_path.is_file():
        raise ValueError(
            "layerwise model directory '{}' must contain a config.json file.".format(
                normalized_path
            )
        )

    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            model_config = json.load(
                config_file, object_pairs_hook=_mapping_without_duplicate_keys
            )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            "model config '{}' is not valid UTF-8 JSON: {}.".format(config_path, error)
        ) from None

    if not isinstance(model_config, dict):
        raise ValueError(
            "model config '{}' must contain a JSON object.".format(config_path)
        )
    model_type = model_config.get("model_type")
    if not isinstance(model_type, str) or not model_type.strip():
        raise ValueError(
            "model config '{}' must define a non-empty string 'model_type'.".format(
                config_path
            )
        )
    return normalized_path, model_config


def resolve_model_name(model_path: str, registry=None) -> str:
    """Resolve local ``config.json:model_type`` to one registered Adapter key."""
    _, model_config = load_model_config(model_path)
    model_type = model_config["model_type"]
    registry = _registered_model_registry(model_type, registry)
    return _resolve_model_name_from_type(model_path, model_type, registry)


def _resolve_model_name_from_type(model_path: str, model_type: str, registry) -> str:
    matches = [
        name
        for name, item in registry.items().items()
        if item.metadata.get("model_type") == model_type
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            "model_type '{}' matches multiple registered adapters: {}.".format(
                model_type, ", ".join(sorted(matches))
            )
        )

    available_by_name = {
        name: item.metadata.get("model_type", "missing metadata")
        for name, item in registry.items().items()
    }
    if registry is MODEL_REGISTRY:
        available_by_name.update(available_llm_adapters())
    available = [
        "{} ({})".format(name, model_type)
        for name, model_type in sorted(available_by_name.items())
    ]
    raise ValueError(
        "model_type '{}' from '{}' has no registered layerwise adapter. "
        "Available adapters: {}.".format(
            model_type,
            os.path.join(
                os.path.abspath(os.path.expanduser(model_path)), "config.json"
            ),
            ", ".join(available) if available else "<empty>",
        )
    )


def build_args_from_config(
    config: dict | None,
    model_path: str,
    model_name: str,
    model_config: dict[str, Any],
    quant_result_dir: str,
) -> LayerwiseArgs:
    """Translate a Classic config into the explicit Adapter argument contract."""
    calibration_shape_config = {} if config is None else config
    if config is None:
        config = set_default_config()
    if not isinstance(config, dict):
        raise TypeError(
            "layerwise config must be a dict or None, got {}.".format(
                type(config).__name__
            )
        )

    _check_layerwise_config_structure(config)
    parsed = QuantConfig(config, AlgorithmRegistry)
    _check_layerwise_p0_config(parsed)

    weights = parsed.quant_cfg.weights_cfg
    inputs = parsed.quant_cfg.inputs_cfg
    w_bits = _SUPPORTED_WEIGHT_TYPES[weights.quant_type]
    if inputs.quant_input:
        a_bits = _SUPPORTED_ACTIVATION_TYPES[inputs.quant_type]
        activation_dynamic = inputs.dynamic is True
        activation_strategy = inputs.strategy
        activation_symmetric = inputs.symmetric
    else:
        a_bits = 16
        activation_dynamic = False
        activation_strategy = None
        activation_symmetric = None

    batch_num, batch_size, num_samples = _resolve_calibration_shape(
        calibration_shape_config
    )
    model_target = "moe" if _is_moe_model(model_config) else "mlp"
    _reject_moe_target(model_target, model_config)
    bit_policy = BitPolicy({"w_bits": w_bits, "a_bits": a_bits})
    bit_policy.a_dynamic = activation_dynamic

    device = _non_empty_string(config.get("device", "npu:0"), "config.device")
    dataset = config.get("dataset")
    if dataset is not None:
        dataset = _non_empty_string(dataset, "config.dataset")
        dataset = os.path.abspath(os.path.expanduser(dataset))

    work_dir_value = config.get("work_dir")
    if work_dir_value is None:
        work_dir = quant_result_dir + ".work"
    else:
        work_dir = os.path.abspath(
            os.path.expanduser(_non_empty_string(work_dir_value, "config.work_dir"))
        )
    if os.path.exists(work_dir) and not os.path.isdir(work_dir):
        raise ValueError("config.work_dir '{}' must not be a file.".format(work_dir))
    _check_work_dir_separation(work_dir, quant_result_dir)

    trust_remote_code = config.get("trust_remote_code", False)
    if type(trust_remote_code) is not bool:
        raise TypeError(
            "config.trust_remote_code must be bool, got {}.".format(
                type(trust_remote_code).__name__
            )
        )

    seq_len = _positive_int(config.get("max_seq_length", 4096), "max_seq_length")
    target_data_dir = os.path.join(work_dir, "ptq_data", model_target)
    return LayerwiseArgs(
        model=model_path,
        model_name=model_name,
        quant_result_dir=quant_result_dir,
        quant_target=[model_target],
        quant_dtype="int",
        w_bits=w_bits,
        a_bits=a_bits,
        bit_policy=bit_policy,
        algos=list(parsed.algorithm.names),
        batch_num=batch_num,
        cali_bsz=batch_size,
        nsamples=num_samples,
        seq_len=seq_len,
        device=device,
        work_dir=work_dir,
        data_dir=target_data_dir,
        output_dir=work_dir,
        skip_layers=set(parsed.skip_layers.skip_layers),
        dataset=dataset,
        trust_remote_code=trust_remote_code,
        weight_strategy=weights.strategy,
        weight_symmetric=weights.symmetric,
        activation_strategy=activation_strategy,
        activation_symmetric=activation_symmetric,
        activation_dynamic=activation_dynamic,
        group_size=weights.group_size,
    )


def prepare_layerwise_quantization(
    model_path: str,
    config: dict | None,
    quant_result_dir: str | None,
    registry=None,
) -> LayerwiseQuantizationPlan:
    """Validate the Week 5 contract and construct exactly one model Adapter."""
    normalized_path, model_config = load_model_config(model_path)
    normalized_result_dir = _resolve_layerwise_quant_result_dir(quant_result_dir)

    # Config errors fail before model registration or Adapter construction.
    provisional_name = model_config["model_type"]
    args = build_args_from_config(
        config,
        normalized_path,
        provisional_name,
        model_config,
        normalized_result_dir,
    )
    resolved_registry = _registered_model_registry(model_config["model_type"], registry)
    model_name = _resolve_model_name_from_type(
        normalized_path, model_config["model_type"], resolved_registry
    )
    args.model_name = model_name
    adapter_cls = resolved_registry.get(model_name)
    adapter = adapter_cls(args)
    return LayerwiseQuantizationPlan(
        args=args, adapter=adapter, model_config=model_config
    )


def run_layerwise_quantization(plan: LayerwiseQuantizationPlan) -> str:
    """Run Extract and PTQ for one prepared layerwise target."""
    args = plan.args
    adapter = plan.adapter
    target = _single_target(args.quant_target)
    _register_layerwise_components()
    _prepare_result_directory(args.quant_result_dir)

    unit_plan = _plan_linear_units(adapter, args)
    expected_units = tuple(unit.layer_name for unit in unit_plan if not unit.skip)
    source_model = _source_model_fingerprint(adapter)
    writer = QuantResultWriter.create(
        args.quant_result_dir, expected_units, source_model
    )
    data_provider = LlmPtqDataProvider(args, adapter)

    samples = None
    cache_valid = _has_valid_activation_cache(adapter, args, target)
    if _needs_extract(args) and not cache_valid:
        samples = load_calibration_data(adapter)
        try:
            _run_extract(adapter, args, samples, target)
            _publish_activation_cache_manifest(adapter, args, target)
        finally:
            _release_device(samples)

    _run_ptq(adapter, args, data_provider, writer)
    if not writer.publish_if_complete():
        raise RuntimeError(
            "layerwise quantization finished with pending units: {}.".format(
                ", ".join(writer.pending_units)
            )
        )
    return args.quant_result_dir


def _register_layerwise_components() -> None:
    register_algorithms()
    register_dtype()
    register_solvers()


def _single_target(targets: list[str]) -> str:
    if not isinstance(targets, (list, tuple)) or len(targets) != 1:
        raise ValueError("pipeline='layerwise' supports exactly one quant target.")
    target = targets[0]
    if target not in {"mlp", "moe"}:
        raise ValueError(
            "pipeline='layerwise' supports target 'mlp' or 'moe', got {!r}.".format(
                target
            )
        )
    return target


def _prepare_result_directory(result_dir: str) -> None:
    os.makedirs(result_dir, exist_ok=True)
    if not os.path.isdir(result_dir):
        raise ValueError(
            "quant_result_dir '{}' must be a directory.".format(result_dir)
        )
    if not os.access(result_dir, os.R_OK | os.W_OK | os.X_OK):
        raise ValueError(
            "quant_result_dir '{}' must be readable, writable, and searchable.".format(
                result_dir
            )
        )
    if os.path.lexists(os.path.join(result_dir, "quant_manifest.json")):
        raise FileExistsError(
            "quant result manifest already exists in '{}'; use a new result directory.".format(
                result_dir
            )
        )


def _plan_linear_units(adapter, args) -> tuple[PlannedLinearUnit, ...]:
    """Build one stable plan shared by writer validation, cache and PTQ."""
    units = []
    for layer_idx in range(_num_layers(adapter)):
        block = adapter.build_quant_block(layer_idx)
        try:
            mlp = _planned_mlp(adapter, block)
            for name, _module in mlp.named_children():
                if name not in {"gate_proj", "up_proj", "down_proj"}:
                    continue
                _linear_child(mlp, name)
                layer_name = _linear_layer_name(adapter, layer_idx, name)
                units.append(
                    PlannedLinearUnit(
                        layer_idx=layer_idx,
                        relative_name=name,
                        layer_name=layer_name,
                        skip=_should_skip_names(
                            name, layer_name, skip_layers=args.skip_layers
                        ),
                    )
                )
        finally:
            _release_device(block)
    if not units:
        raise ValueError("the adapter produced no layerwise Linear units.")
    if len(set(unit.layer_name for unit in units)) != len(units):
        raise ValueError("layerwise Linear unit names are not globally unique.")
    return tuple(units)


def _planned_mlp(adapter, block) -> torch.nn.Module:
    mlp = getattr(block, "mlp", None)
    if mlp is None:
        raise ValueError("layerwise P0 expects each decoder block to expose 'mlp'.")
    if hasattr(mlp, "experts"):
        raise ValueError(
            "layerwise P0 does not support routed MoE experts; use a registered "
            "dense model."
        )
    return mlp


def _linear_child(mlp: torch.nn.Module, name: str) -> torch.nn.Linear:
    """Resolve the original Linear in either a plain or quantized MLP."""
    module = getattr(mlp, name)
    linear = getattr(module, "linear", None)
    if isinstance(linear, torch.nn.Linear):
        return linear
    if isinstance(module, torch.nn.Linear):
        return module
    raise TypeError(
        "layerwise P0 expects MLP child '{}' to be nn.Linear or a wrapper with "
        "a .linear nn.Linear, got {}.".format(name, type(module).__name__)
    )


def _linear_layer_name(adapter, layer_idx: int, relative_name: str) -> str:
    prefix = getattr(adapter, "get_layer_weight_prefix", None)
    weight_prefix = (
        prefix(layer_idx) if callable(prefix) else "model.layers.{}.".format(layer_idx)
    )
    return "{}mlp.{}".format(weight_prefix, relative_name)


def _should_skip_names(*names, skip_layers) -> bool:
    return any(name in skip_layers for name in names)


def _num_layers(adapter) -> int:
    num_layers = getattr(adapter, "num_layers", None)
    if type(num_layers) is not int or num_layers <= 0:
        raise ValueError("adapter must expose a positive integer num_layers.")
    return num_layers


def _source_model_fingerprint(adapter):
    """Build the artifact identity expected by QuantResultWriter/reload."""
    config = getattr(adapter, "config", None)
    return {
        "model_type": getattr(config, "model_type", adapter.args.model_name),
        "torch_dtype": _source_model_dtype(adapter),
    }


def _activation_cache_model_fingerprint(adapter):
    """Build the stronger identity used to validate activation caches."""
    config = getattr(adapter, "config", None)
    config_dict = (
        config.to_dict() if hasattr(config, "to_dict") else plan_model_config(adapter)
    )
    config_text = _canonical_json(_jsonable_lossy(config_dict))
    return {
        "model_type": getattr(config, "model_type", adapter.args.model_name),
        "config_sha256": "sha256:" + hashlib.sha256(config_text.encode()).hexdigest(),
        "state_dict_sha256": _checkpoint_state_dict_sha256(adapter),
    }


def _source_model_dtype(adapter):
    config = getattr(adapter, "config", None)
    configured = getattr(config, "torch_dtype", None)
    if configured is None:
        to_dict = getattr(config, "to_dict", None)
        if callable(to_dict):
            config_dict = to_dict()
            if isinstance(config_dict, dict):
                configured = config_dict.get("torch_dtype")
    if isinstance(configured, torch.dtype):
        return dtype_name(configured)
    if isinstance(configured, str) and configured:
        return dtype_name(configured)
    for _name, tensor in _checkpoint_tensors(adapter):
        return dtype_name(tensor.dtype)
    return "unknown"


def plan_model_config(adapter):
    return (
        getattr(adapter, "args", None).model_config
        if hasattr(adapter.args, "model_config")
        else {}
    )


def _checkpoint_state_dict_sha256(adapter):
    """Match the tensor-byte digest computed by compute_model_fingerprint()."""
    hasher = hashlib.sha256()
    for name, tensor in _checkpoint_tensors(adapter):
        hasher.update(
            _canonical_json(
                {
                    "key": name,
                    "dtype": str(tensor.dtype).removeprefix("torch."),
                    "shape": list(tensor.shape),
                }
            ).encode()
        )
        _hash_tensor_bytes(hasher, tensor)
    return "sha256:" + hasher.hexdigest()


def _checkpoint_tensors(adapter):
    for name in sorted(_weight_map(adapter)):
        file_name = _weight_map(adapter)[name]
        if file_name != os.path.basename(file_name):
            raise ValueError(
                "checkpoint weight mapping contains an unsafe path: {}.".format(
                    file_name
                )
            )
        file_path = os.path.join(adapter.args.model, file_name)
        with safe_open(file_path, framework="pt", device="cpu") as file_handle:
            if name not in file_handle.keys():
                raise KeyError(
                    "checkpoint file '{}' is missing tensor '{}'.".format(
                        file_path, name
                    )
                )
            yield name, file_handle.get_tensor(name)


def _hash_tensor_bytes(hasher, tensor) -> None:
    byte_view = tensor.detach().reshape(-1).contiguous().view(torch.uint8)
    for start in range(0, byte_view.numel(), 1024 * 1024):
        hasher.update(byte_view[start : start + 1024 * 1024].numpy().tobytes())


def _weight_map(adapter):
    cached = getattr(adapter, "_weight_map", None)
    if cached is not None:
        return cached
    model_path = adapter.args.model
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    single_path = os.path.join(model_path, "model.safetensors")
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as handle:
            mapping = json.load(handle)["weight_map"]
    elif os.path.isfile(single_path):
        with safe_open(single_path, framework="pt") as handle:
            mapping = {name: "model.safetensors" for name in handle.keys()}
    else:
        raise FileNotFoundError(
            "model directory '{}' has no safetensors index or model.safetensors.".format(
                model_path
            )
        )
    setattr(adapter, "_weight_map", mapping)
    return mapping


def _canonical_json(data) -> str:
    return json.dumps(
        data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _jsonable_lossy(value):
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, torch.Tensor):
        return {"dtype": str(value.dtype), "shape": list(value.shape)}
    if isinstance(value, dict):
        return {str(key): _jsonable_lossy(item) for key, item in value.items()}
    if callable(value):
        return "{}.{}".format(type(value).__module__, type(value).__qualname__)
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable_lossy(item) for item in value]
    return str(value)


def load_calibration_data(adapter):
    """Load tokenized calibration samples from the configured local dataset."""
    args = adapter.args
    if args.dataset is None:
        raise ValueError(
            "pipeline='layerwise' requires config.dataset to be a local dataset path."
        )
    if not isinstance(args.dataset, str):
        raise TypeError("config.dataset must be a str path.")
    dataset_path = os.path.abspath(os.path.expanduser(args.dataset))
    if not os.path.isdir(dataset_path):
        raise ValueError(
            "config.dataset '{}' must be an existing directory.".format(dataset_path)
        )

    from datasets import load_from_disk

    dataset = load_from_disk(dataset_path)
    if hasattr(dataset, "keys") and callable(dataset.keys):
        splits = list(dataset.keys())
        if len(splits) != 1:
            raise ValueError(
                "dataset '{}' must contain exactly one split, got {}.".format(
                    dataset_path, splits
                )
            )
        dataset = dataset[splits[0]]
    if "text" not in dataset.column_names:
        raise ValueError(
            "dataset '{}' must contain a 'text' string column, got columns {}.".format(
                dataset_path, list(dataset.column_names)
            )
        )

    tokens = []
    for record in dataset:
        text = record["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "dataset '{}' contains an empty or non-string 'text' item.".format(
                    dataset_path
                )
            )
        encoded = list(adapter.tokenizer.encode(text))
        if not encoded:
            raise ValueError(
                "dataset '{}' contains a 'text' item that tokenizes to zero tokens.".format(
                    dataset_path
                )
            )
        tokens.extend(encoded)
        if len(tokens) >= args.nsamples * args.seq_len:
            break
    required_tokens = args.nsamples * args.seq_len
    if len(tokens) < required_tokens:
        raise ValueError(
            "dataset '{}' has only {} usable token(s); {} are required for {} sample(s) "
            "of length {}.".format(
                dataset_path, len(tokens), required_tokens, args.nsamples, args.seq_len
            )
        )
    return [
        torch.tensor(
            tokens[index * args.seq_len : (index + 1) * args.seq_len], dtype=torch.long
        ).unsqueeze(0)
        for index in range(args.nsamples)
    ]


def _needs_extract(args) -> bool:
    return args.a_bits < 16 and not args.activation_dynamic and "minmax" in args.algos


def _run_extract(adapter, args, samples, target: str) -> None:
    hook_name = getattr(adapter, "ffn_norm_name", "post_attention_layernorm")
    _reset_activation_cache(args.data_dir)
    inter_io = adapter.do_embedding_forward(samples, hook_name=hook_name)
    try:
        for layer_idx in range(_num_layers(adapter)):
            inter_io = adapter.do_block_forward(
                layer_idx, inter_io, hook_name=hook_name
            )
    finally:
        _release_device(inter_io)


def _reset_activation_cache(data_dir: str) -> None:
    """Remove only an incomplete cache; a valid cache is never re-extracted."""
    os.makedirs(data_dir, exist_ok=True)
    for name in os.listdir(data_dir):
        path = os.path.join(data_dir, name)
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
        else:
            shutil.rmtree(path)


def _activation_cache_manifest(args, adapter, target: str) -> dict:
    files = [
        "block_{}_{}_in.pkl".format(index, target)
        for index in range(_num_layers(adapter))
    ]
    payload = {
        "format": "amct-layerwise-activation-cache",
        "schema_version": 1,
        "model_path": os.path.realpath(args.model),
        "model_fingerprint": _activation_cache_model_fingerprint(adapter),
        "model_type": getattr(adapter.config, "model_type", None),
        "dataset": os.path.realpath(args.dataset),
        "tokenizer": {
            "path": os.path.realpath(args.model),
            "sha256": _optional_file_sha256(os.path.join(args.model, "tokenizer.json")),
        },
        "num_samples": args.nsamples,
        "batch_size": args.cali_bsz,
        "seq_len": args.seq_len,
        "target": target,
        "hook": getattr(adapter, "ffn_norm_name", "post_attention_layernorm"),
        "sampling": {"order": "dataset", "concatenate": True, "seed": 0},
    }
    payload["cache_key"] = (
        "sha256:" + hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
    )
    payload["files"] = [
        {
            "name": name,
            "size": os.path.getsize(os.path.join(args.data_dir, name)),
            "sha256": _file_sha256(os.path.join(args.data_dir, name)),
        }
        for name in files
    ]
    return payload


def _publish_activation_cache_manifest(adapter, args, target: str) -> None:
    manifest = _activation_cache_manifest(args, adapter, target)
    path = os.path.join(args.data_dir, "cache_manifest.json")
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(_canonical_json(manifest) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _has_valid_activation_cache(adapter, args, target: str) -> bool:
    path = os.path.join(args.data_dir, "cache_manifest.json")
    if not os.path.isfile(path):
        return False
    try:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        expected_manifest = _activation_cache_manifest(args, adapter, target)
        if manifest.get("cache_key") != expected_manifest["cache_key"]:
            return False
        if manifest.get("files") != expected_manifest["files"]:
            return False
        return True
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _select_solver_cls(algorithms: list[str]):
    for name in algorithms:
        if name == "minmax":
            continue
        item = _algorithm_registry_item(name)
        if item.metadata.get("solver", "block") != "calibrate":
            return SOLVER_REGISTRY.get("block")
    return SOLVER_REGISTRY.get("calibrate")


def _algorithm_registry_item(name):
    from amct_pytorch.algorithms import ALGO_REGISTRY

    return ALGO_REGISTRY.get_item(name)


def _run_ptq(adapter, args, data_provider, writer: QuantResultWriter) -> None:
    solver_cls = _select_solver_cls(args.algos)
    grouped_units = _group_planned_units(_plan_linear_units(adapter, args))
    for layer_idx in range(_num_layers(adapter)):
        block = adapter.build_quant_block(layer_idx)
        try:
            mlp = _planned_mlp(adapter, block)
            block_inputs = None
            block_kwargs = {}
            if _needs_extract(args):
                block_inputs, block_kwargs = _load_block_inputs(
                    data_provider, layer_idx, target=args.quant_target[0]
                )
                block_inputs = _to_device(block_inputs, args.device)
                captured = _capture_linear_inputs(mlp, block_inputs)
            else:
                captured = None
            for planned in grouped_units.get(layer_idx, []):
                if planned.skip:
                    continue
                _quantize_linear(
                    adapter,
                    args,
                    data_provider,
                    solver_cls,
                    mlp,
                    planned,
                    captured,
                    writer,
                )
        finally:
            _release_device(block)


def _load_block_inputs(data_provider, layer_idx: int, target: str):
    from amct_pytorch.common.models.llm.common.ptq_units import make_ptq_unit

    unit = make_ptq_unit(target, "mlp", layer_idx, None)
    return data_provider.load_unit_inputs(unit)


def _group_planned_units(unit_plan):
    grouped = {}
    for unit in unit_plan:
        grouped.setdefault(unit.layer_idx, []).append(unit)
    return grouped


def _capture_linear_inputs(mlp: torch.nn.Module, inputs: torch.Tensor):
    """Capture each MLP Linear's real input without relying on target naming."""
    captured = {}
    hooks = []

    def capture(name):
        def hook(_module, args, _output=None):
            captured[name] = args[0].detach()

        return hook

    try:
        for name, _module in mlp.named_children():
            if name not in {"gate_proj", "up_proj", "down_proj"}:
                continue
            hooks.append(mlp.get_submodule(name).register_forward_hook(capture(name)))
        if len(hooks) != 3:
            raise ValueError(
                "layerwise P0 expects gate_proj, up_proj, and down_proj Linear modules, "
                "got {}.".format(len(hooks))
            )
        with torch.no_grad():
            _forward_mlp_for_capture(mlp, inputs)
        expected = {"gate_proj", "up_proj", "down_proj"}
        if set(captured) != expected:
            missing = sorted(expected - set(captured))
            raise RuntimeError(
                "failed to capture MLP Linear input(s): {}.".format(", ".join(missing))
            )
    finally:
        for hook in hooks:
            hook.remove()
    return captured


def _forward_mlp_for_capture(mlp: torch.nn.Module, inputs: torch.Tensor) -> None:
    """Run the gated MLP while preserving each wrapped Linear's actual input."""
    if type(mlp).__name__ != "QuantGatedMLP":
        mlp(inputs)
        return
    gate = mlp.gate_proj(inputs)
    up = mlp.up_proj(inputs)
    mlp.down_proj(mlp.act_fn(gate) * up)


def _quantize_linear(
    adapter, args, data_provider, solver_cls, mlp, planned, captured, writer
) -> None:
    module = _linear_child(mlp, planned.relative_name)
    if _needs_extract(args):
        inputs = captured.get(planned.relative_name)
        if inputs is None:
            raise FileNotFoundError(
                "activation cache is missing input for layerwise unit '{}'.".format(
                    planned.layer_name
                )
            )
    else:
        inputs = torch.zeros((1, module.in_features), device=args.device)
    if inputs.dtype != module.weight.dtype:
        inputs = inputs.to(module.weight.dtype)
    unit = _make_linear_unit(args, planned, module)
    batch = data_provider.build_unit_batch(unit, inputs, {}, None)
    solver = solver_cls(args, planned.layer_idx, module)
    if type(solver.model).__name__ != "MinMaxQuant":
        quant_module = _build_classic_minmax(module, planned.layer_name, args)
        for data_batch in batch.data_loader:
            quant_module(_unpack_single(data_batch))
    else:
        quant_module = solver.model
    tensors, metadata = _export_quant_params(quant_module.export_quant_params())
    writer.write_unit(
        layer_name=planned.layer_name,
        tensors=tensors,
        non_tensor_params=metadata,
        quant_config=_unit_quant_config(args),
        quant_module_type="MinMaxQuant",
        ori_module_type=type(module).__name__,
    )
    _release_device(quant_module)
    module = None


def _original_module(module):
    return getattr(module, "ori_module", module)


def _unit_quant_config(args) -> dict:
    config = {
        "batch_num": args.batch_num,
        "weights_cfg": {
            "quant_type": "int8" if args.w_bits == 8 else "int4",
            "symmetric": args.weight_symmetric,
            "strategy": args.weight_strategy,
        },
        "inputs_cfg": {"enable_quant": False},
        "algorithm": {"minmax": {}},
    }
    if args.group_size is not None:
        config["weights_cfg"]["group_size"] = args.group_size
    if args.a_bits < 16:
        config["inputs_cfg"] = {
            "enable_quant": True,
            "quant_type": "int8",
            "symmetric": args.activation_symmetric,
            "strategy": args.activation_strategy,
            "dynamic": False,
        }
    return config


def _export_quant_params(result):
    if isinstance(result, tuple) and len(result) == 2:
        tensors, metadata = result
    else:
        tensors, metadata = result, {}
    tensors = tensors or {}
    metadata = metadata or {}
    checked_tensors = {}
    for key, value in tensors.items():
        if isinstance(value, torch.Tensor):
            checked_tensors[key] = value.detach().cpu().contiguous()
        else:
            metadata[key] = value
    return checked_tensors, metadata


def _make_linear_unit(args, planned, module):
    from amct_pytorch.common.models.llm.common.ptq_units import make_ptq_unit

    return make_ptq_unit(
        "mlp",
        planned.relative_name,
        planned.layer_idx,
        module,
        {"layer_name": planned.layer_name},
    )


def _build_classic_minmax(module, layer_name, args):
    from amct_pytorch.algorithms.quant import register_algorithms
    from amct_pytorch.classic.quantize_op.minmax_module import MinMaxQuant

    register_algorithms()
    return MinMaxQuant(module, layer_name, _unit_quant_config(args))


def _unpack_single(data_batch):
    if isinstance(data_batch, (tuple, list)):
        if len(data_batch) != 1:
            raise ValueError(
                "layerwise calibration expects one input tensor per batch."
            )
        return data_batch[0]
    return data_batch


def _to_device(value, device: str):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _release_device(value) -> None:
    if isinstance(value, torch.nn.Module):
        value.to("cpu")
    elif isinstance(value, list):
        for item in value:
            _release_device(item)
    elif isinstance(value, tuple):
        for item in value:
            _release_device(item)
    del value
    gc_collect()


def gc_collect() -> None:
    import gc

    gc.collect()
    npu = getattr(torch, "npu", None)
    if callable(getattr(npu, "is_available", None)) and npu.is_available():
        torch.npu.empty_cache()


def _file_sha256(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return "sha256:" + hasher.hexdigest()


def _optional_file_sha256(path: str):
    return _file_sha256(path) if os.path.isfile(path) else None


def _check_layerwise_p0_config(parsed: QuantConfig) -> None:
    algorithms = parsed.algorithm.names
    if algorithms not in ([], [_SUPPORTED_ALGORITHM]):
        raise ValueError(
            "pipeline='layerwise' currently supports only no algorithm or "
            "algorithm 'minmax', got {}.".format(algorithms)
        )

    quant_cfg = parsed.quant_cfg
    fuzzy = {
        name: entries for name, entries in quant_cfg.fuzzy_configs.items() if entries
    }
    if fuzzy:
        raise ValueError(
            "pipeline='layerwise' does not support fuzzy quant_cfg entries; "
            "got {}.".format(sorted(fuzzy))
        )
    if quant_cfg.kvcache_cfg.quant_kv:
        raise ValueError(
            "pipeline='layerwise' currently supports one FFN target only; kvcache "
            "quantization is not supported."
        )

    weights = quant_cfg.weights_cfg
    if weights.quant_type not in _SUPPORTED_WEIGHT_TYPES:
        raise ValueError(
            "pipeline='layerwise' weights.type must be one of {}, got {!r}.".format(
                sorted(_SUPPORTED_WEIGHT_TYPES), weights.quant_type
            )
        )
    inputs = quant_cfg.inputs_cfg
    if not inputs.quant_input:
        return
    if not algorithms:
        raise ValueError(
            "pipeline='layerwise' static activation quantization requires "
            "algorithm 'minmax'."
        )
    if inputs.quant_type not in _SUPPORTED_ACTIVATION_TYPES:
        raise ValueError(
            "pipeline='layerwise' static MinMax inputs.type must be one of {}, "
            "got {!r}.".format(sorted(_SUPPORTED_ACTIVATION_TYPES), inputs.quant_type)
        )
    if inputs.dynamic is True:
        raise ValueError(
            "pipeline='layerwise' currently supports static MinMax activation only; "
            "config.quant_cfg.inputs.dynamic must be False or omitted."
        )


def _reject_moe_target(model_target: str, model_config: dict[str, Any]) -> None:
    if model_target != "moe":
        return
    raise ValueError(
        "pipeline='layerwise' P0 supports registered dense FFN models only; "
        "model '{}' declares routed MoE fields {}.".format(
            getattr(model_config, "model_type", "unknown"),
            ", ".join(
                field
                for field in _MOE_CONFIG_FIELDS
                if type(model_config.get(field)) is int and model_config[field] > 0
            ),
        )
    )


def _check_layerwise_config_structure(config: dict) -> None:
    _reject_unknown_fields(config, _LAYERWISE_CONFIG_FIELDS, "config")

    quant_cfg = config.get("quant_cfg", {})
    if not isinstance(quant_cfg, dict):
        raise TypeError(
            "config.quant_cfg must be a dict, got {}.".format(type(quant_cfg).__name__)
        )
    _reject_unknown_fields(
        quant_cfg,
        _QUANT_CONFIG_FIELDS,
        "config.quant_cfg (fuzzy entries are outside Week 5 P0)",
    )
    for field, allowed in (
        ("weights", _WEIGHT_CONFIG_FIELDS),
        ("inputs", _INPUT_CONFIG_FIELDS),
        ("kvcache", _KVCACHE_CONFIG_FIELDS),
    ):
        section = quant_cfg.get(field)
        if section is None:
            continue
        if not isinstance(section, dict):
            raise TypeError(
                "config.quant_cfg.{} must be a dict, got {}.".format(
                    field, type(section).__name__
                )
            )
        _reject_unknown_fields(section, allowed, "config.quant_cfg.{}".format(field))

    skip_layers = config.get("skip_layers", [])
    if not isinstance(skip_layers, (list, tuple, set)):
        raise TypeError(
            "config.skip_layers must be a list, tuple, or set of layer names, "
            "got {}.".format(type(skip_layers).__name__)
        )

    algorithm = config.get("algorithm")
    if isinstance(algorithm, dict) and set(algorithm) == {_SUPPORTED_ALGORITHM}:
        attributes = algorithm[_SUPPORTED_ALGORITHM]
        if attributes not in (None, {}):
            raise ValueError(
                "pipeline='layerwise' MinMax does not accept algorithm options, "
                "got {!r}.".format(attributes)
            )


def _reject_unknown_fields(config: dict, allowed: set[str], location: str) -> None:
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(
            "pipeline='layerwise' does not support {} fields: {}.".format(
                location, ", ".join(sorted(map(repr, unknown)))
            )
        )


def _resolve_calibration_shape(config: dict) -> tuple[int, int, int]:
    names = ("batch_num", "batch_size", "num_calibration_samples")
    supplied = {name: config[name] for name in names if name in config}
    values = {name: _positive_int(value, name) for name, value in supplied.items()}
    if not values:
        return 128, 1, 128

    if len(values) == 1:
        if "batch_size" in values:
            batch_size = values["batch_size"]
            num_samples = 128
            if num_samples % batch_size != 0:
                raise ValueError(
                    "default num_calibration_samples 128 must be divisible by "
                    "batch_size."
                )
            return num_samples // batch_size, batch_size, num_samples
        samples_or_batches = next(iter(values.values()))
        return samples_or_batches, 1, samples_or_batches

    if len(values) == 2:
        if "batch_num" not in values:
            batch_size = values["batch_size"]
            num_samples = values["num_calibration_samples"]
            if num_samples % batch_size != 0:
                raise ValueError(
                    "num_calibration_samples must be divisible by batch_size."
                )
            return num_samples // batch_size, batch_size, num_samples
        if "batch_size" not in values:
            batch_num = values["batch_num"]
            num_samples = values["num_calibration_samples"]
            if num_samples % batch_num != 0:
                raise ValueError(
                    "num_calibration_samples must be divisible by batch_num."
                )
            return batch_num, num_samples // batch_num, num_samples
        batch_num = values["batch_num"]
        batch_size = values["batch_size"]
        return batch_num, batch_size, batch_num * batch_size

    batch_num = values["batch_num"]
    batch_size = values["batch_size"]
    num_samples = values["num_calibration_samples"]
    if num_samples != batch_num * batch_size:
        raise ValueError(
            "num_calibration_samples must equal batch_size * batch_num, got "
            "{} != {} * {}.".format(num_samples, batch_size, batch_num)
        )
    return batch_num, batch_size, num_samples


def _check_work_dir_separation(work_dir: str, quant_result_dir: str) -> None:
    work_path = os.path.realpath(work_dir)
    result_path = os.path.realpath(quant_result_dir)
    common_path = os.path.commonpath((work_path, result_path))
    if common_path in (work_path, result_path):
        raise ValueError(
            "config.work_dir '{}' and quant_result_dir '{}' must be separate, "
            "non-overlapping directory trees.".format(work_dir, quant_result_dir)
        )


def _resolve_layerwise_quant_result_dir(quant_result_dir: str | None) -> str:
    if quant_result_dir is None:
        quant_result_dir = _DEFAULT_QUANT_RESULT_DIR
    if not isinstance(quant_result_dir, str):
        raise TypeError(
            "quant_result_dir must be a str path or None, got {}.".format(
                type(quant_result_dir).__name__
            )
        )
    if not quant_result_dir.strip():
        raise ValueError("quant_result_dir must be a non-empty path.")

    target = os.path.abspath(os.path.expanduser(quant_result_dir))
    if os.path.lexists(target) and not os.path.isdir(target):
        raise ValueError(
            "quant_result_dir '{}' must be a directory path, not a file or "
            "broken symlink.".format(target)
        )
    if os.path.isdir(target) and not os.access(target, os.R_OK | os.W_OK | os.X_OK):
        raise ValueError(
            "quant_result_dir '{}' must be readable, writable, and searchable.".format(
                target
            )
        )
    return target


def _registered_model_registry(model_type: str, registry):
    if registry is not None:
        return registry
    try:
        register_llm_models(model_type=model_type)
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "cannot register the adapter for model_type '{}': missing dependency "
            "'{}'. Install the repository-pinned model dependencies before using "
            "pipeline='layerwise'.".format(model_type, error.name)
        ) from error
    return MODEL_REGISTRY


def _mapping_without_duplicate_keys(pairs):
    mapping = {}
    for key, value in pairs:
        if key in mapping:
            raise ValueError("duplicate JSON key {!r}".format(key))
        mapping[key] = value
    return mapping


def _is_moe_model(model_config: dict[str, Any]) -> bool:
    for field in _MOE_CONFIG_FIELDS:
        value = model_config.get(field)
        if type(value) is int and value > 0:
            return True
    return False


def _positive_int(value, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(
            "config.{} must be a positive int, got {!r}.".format(field, value)
        )
    return value


def _non_empty_string(value, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError("{} must be a str, got {}.".format(field, type(value).__name__))
    if not value.strip():
        raise ValueError("{} must be a non-empty str.".format(field))
    return value
