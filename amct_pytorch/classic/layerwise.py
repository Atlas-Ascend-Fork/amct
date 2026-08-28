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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from amct_pytorch.algorithms import AlgorithmRegistry
from amct_pytorch.common.config import set_default_config
from amct_pytorch.common.config.fields import QuantConfig
from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm import available_llm_adapters, register_llm_models
from amct_pytorch.quantization.bit_policy import BitPolicy


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
