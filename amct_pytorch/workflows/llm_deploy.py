# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

from amct_pytorch.algorithms.quant import register_algorithms
from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm import register_llm_models
from amct_pytorch.common.models.llm.common.deploy_export import (
    adapt_ascend_payload,
    convert_state_dict,
    export_block_deploy,
    generate_quant_config,
    generate_tensor_quant_config,
    make_ascend_description,
    quant_payload,
)
from amct_pytorch.common.models.llm.common.deploy_selection import (
    build_tensor_deploy_plan,
    log_mtp_selection,
    requires_tensor_deploy_config,
    validate_ascend_deploy_args,
    validate_ascend_deploy_plan,
    validate_tensor_deploy_args,
)
from amct_pytorch.common.models.llm.common.deploy_validation import (
    validate_ascend_artifacts,
)
from amct_pytorch.common.models.llm.common.packed_expert_export import (
    detect_packed_gated_expert_layout,
)
from amct_pytorch.common.models.llm.common.weight_path_validation import (
    collect_safetensors_files,
    resolve_safetensors_path,
    validate_weight_map,
)
from amct_pytorch.common.utils.run_logging import ensure_log_dir, setup_run_logging
from amct_pytorch.common.utils.seed_utils import seed_everything
from amct_pytorch.quantization.dtypes import DTYPE_REGISTRY, register_dtype


class LlmDeployWorkflow:
    """Export deploy-ready quantized artifacts."""

    def __init__(self, args):
        self.args = args
        self.granularity = args.granularity
        if self.granularity not in ("block", "tensor"):
            raise ValueError(
                f"deploy only supports granularity 'block' or 'tensor', "
                f"got '{self.granularity}'."
            )
        if self.granularity == "block":
            if not args.quant_target:
                raise ValueError(
                    "deploy with --granularity block requires: --quant_target."
                )
            if not args.quant_dtype:
                raise ValueError(
                    "deploy with --granularity block requires: --quant_dtype."
                )
        self.pipeline = None
        self.model_name = args.model_name
        self.model_path = args.model
        self.safetensors_files = None
        self.quant_dtype = args.quant_dtype
        self.output_dir = self.args.output_dir
        self.is_mx = self.quant_dtype.startswith("mx")
        self.is_int = self.quant_dtype.startswith("int")
        self.is_hif = self.quant_dtype.startswith("hif")
        self.seed = args.seed
        seed_everything(self.seed)
        self.ascend_checkpoint_layout = None

    @staticmethod
    def _is_weight_file(path: Path) -> bool:
        return (
            path.name == "model.safetensors.index.json" or path.suffix == ".safetensors"
        )

    @staticmethod
    def _register_components():
        register_llm_models()
        register_dtype()
        register_algorithms()

    @staticmethod
    def _collect_replaced_original_weights(
        layer_tensors: dict[str, object],
        tensor_routes: dict[str, str],
        original_weight_map: dict[str, str],
    ):
        replaced = set()
        for weight_name in layer_tensors:
            base_weight_name = tensor_routes.get(weight_name, weight_name)
            if base_weight_name in original_weight_map:
                replaced.add(base_weight_name)
        return replaced

    def run(self):
        if self.granularity == "tensor":
            validate_tensor_deploy_args(self.args)
        validate_ascend_deploy_args(self.args)
        if getattr(self.args, "deploy_format", "legacy") == "ascend":
            self._validate_ascend_output()
            self._validate_ascend_source()
        sink_id = self.setup()
        try:
            if self.granularity == "block":
                return self._run_blockwise()
            if self.granularity == "tensor":
                return self._run_tensorwise()
            raise ValueError(
                f"Unsupported granularity '{self.granularity}' for deploy."
            )
        finally:
            logger.remove(sink_id)

    def _uses_tensor_deploy_plan(self):
        return getattr(
            self.args, "deploy_format", "legacy"
        ) == "ascend" or requires_tensor_deploy_config(self.args)

    def setup(self):
        if self.granularity != "tensor":
            os.makedirs(self.output_dir, exist_ok=True)
            ensure_log_dir(self.args)
        self._register_components()
        self.pipeline = self._build_pipeline()
        if self.granularity == "tensor" and self._uses_tensor_deploy_plan():
            self._prepare_tensor_deploy_plan()
        if self.granularity == "tensor":
            os.makedirs(self.output_dir, exist_ok=True)
            ensure_log_dir(self.args)
        sink_id, _ = setup_run_logging(self.args, "deploy")
        return sink_id

    def _build_pipeline(self):
        model_cls = MODEL_REGISTRY.get(self.model_name)
        return model_cls(self.args)

    def get_safetensors_files(self):
        if getattr(self, "safetensors_files", None) is None:
            self.safetensors_files = collect_safetensors_files(self.model_path)
        return self.safetensors_files

    def _convert_tensor(self, weight_name: str, tensor: torch.Tensor) -> torch.Tensor:
        if self.quant_dtype == "bf16":
            return tensor.to(torch.bfloat16)
        raise NotImplementedError(
            f"tensor granularity does not support quant_dtype '{self.quant_dtype}' yet"
        )

    def _validate_ascend_output(self):
        source = Path(self.model_path).resolve()
        output = Path(self.output_dir).resolve()
        if source == output or source in output.parents or output in source.parents:
            raise ValueError("Ascend output and source directories must not overlap")
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError("Ascend output directory must be new or empty")

    def _validate_ascend_source(self):
        if self.model_name in MODEL_REGISTRY:
            source_validator = getattr(
                MODEL_REGISTRY.get(self.model_name), "validate_ascend_source", None
            )
            if callable(source_validator) and source_validator(self.args):
                return
        source = Path(self.model_path)
        with open(source / "config.json", encoding="utf-8") as f:
            config = json.load(f)
        markers = (
            "quant_model_description.json",
            "quant_model_weights.safetensors.index.json",
            "rot.safetensors",
        )
        if (
            not isinstance(config, dict)
            or config.get("quantization_config")
            or config.get("is_rot_used")
            or any((source / name).exists() for name in markers)
        ):
            raise ValueError(
                "Ascend source must be an unquantized, unrotated BF16 checkpoint"
            )

    @staticmethod
    def _write_json_file(path: Path, data: dict):
        temporary = path.with_name(f".{path.name}.tmp")
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(temporary, path)

    def _copy_support_files(self):
        src_dir = Path(self.model_path)
        dst_dir = Path(self.output_dir)
        for src_path in src_dir.iterdir():
            if src_path.name.startswith("."):
                continue
            if self._is_weight_file(src_path):
                continue
            if getattr(self.args, "deploy_format", "legacy") == "ascend" and (
                src_path.name.endswith(".safetensors.index.json")
                or src_path.name == "quant_model_description.json"
            ):
                continue
            dst_path = dst_dir / src_path.name
            if dst_path.exists():
                continue
            if src_path.is_dir():
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

    def _load_weight_index(self):
        # Align with HF/vLLM checkpoint loading: read index.json when present;
        # otherwise this is a single-shard model -- map every tensor to the lone
        # safetensors file and synthesize an equivalent index. Filenames reuse the
        # transformers ecosystem constants to avoid hardcoded drift.
        index_path = Path(self.model_path) / SAFE_WEIGHTS_INDEX_NAME
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            validate_weight_map(
                self.model_path,
                index.get("weight_map"),
                self.get_safetensors_files(),
            )
            return index
        single_path = Path(self.model_path) / SAFE_WEIGHTS_NAME
        if not single_path.exists():
            raise FileNotFoundError(
                f"Neither {SAFE_WEIGHTS_INDEX_NAME} nor {SAFE_WEIGHTS_NAME} "
                f"found in {self.model_path}"
            )
        with safe_open(str(single_path), framework="pt") as f:
            weight_map = {key: SAFE_WEIGHTS_NAME for key in f.keys()}
        validate_weight_map(self.model_path, weight_map, self.get_safetensors_files())
        # total_size is a placeholder; _refresh_weight_index() recomputes and
        # overwrites it from the actual output shard sizes.
        return {
            "metadata": {"total_size": single_path.stat().st_size},
            "weight_map": weight_map,
        }

    def _refresh_config(self, quant_ignore_layers, *, tensor_plan=None):
        config_file = os.path.join(self.output_dir, 'config.json')
        with open(config_file, "r") as f:
            config = json.load(f)
        if self.quant_dtype is not None:
            cache_scheme_fn = getattr(self.pipeline, "cache_scheme", None)
            cache_scheme = cache_scheme_fn() if callable(cache_scheme_fn) else None
            if tensor_plan is not None:
                quantization_config = generate_tensor_quant_config(
                    tensor_plan, cache_scheme
                )
            else:
                bits_scheme_fn = getattr(self.pipeline, "bits_scheme", None)
                bits_scheme = bits_scheme_fn() if callable(bits_scheme_fn) else None
                quantization_config = generate_quant_config(
                    cache_scheme,
                    quant_ignore_layers,
                    is_mx=self.is_mx,
                    bits_scheme=bits_scheme,
                )
            config['quantization_config'] = quantization_config
        else:
            config.pop('quantization_config', None)

        new_config_file = os.path.join(self.output_dir, "config.json")
        with open(new_config_file, "w") as f:
            json.dump(config, f, indent=2)

    def _refresh_config_tensor(self):
        config_file = os.path.join(self.output_dir, 'config.json')
        with open(config_file, "r") as f:
            config = json.load(f)
        if self.quant_dtype == "bf16":
            config["torch_dtype"] = "bfloat16"
            config.pop('quantization_config', None)
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

    def _refresh_weight_index(
        self, original_index, updated_weight_map, total_size=None
    ):
        metadata = dict(original_index.get("metadata", {}))
        if total_size is None:
            total_size = sum(
                os.path.getsize(os.path.join(self.output_dir, file_name))
                for file_name in set(updated_weight_map.values())
            )
        metadata["total_size"] = total_size

        output_index = {
            "metadata": metadata,
            "weight_map": updated_weight_map,
        }
        index_path = os.path.join(self.output_dir, "model.safetensors.index.json")
        self._write_json_file(Path(index_path), output_index)
        return index_path

    def _run_blockwise(self):
        self._copy_support_files()
        quant_ignore_layers = []
        original_index = self._load_weight_index()
        original_weight_map = dict(original_index.get("weight_map", {}))
        updated_weight_map = {}
        replaced_original_weights = set()
        for layer_idx in tqdm(
            range(self.pipeline.num_layers), desc="Block Processing..."
        ):
            layer_tensors, tensor_routes = export_block_deploy(
                self.pipeline,
                layer_idx,
                quant_ignore_layers,
            )
            if not layer_tensors:
                continue
            updated_weight_map.update(self._write_block_file(layer_idx, layer_tensors))
            replaced_original_weights.update(
                self._collect_replaced_original_weights(
                    layer_tensors,
                    tensor_routes,
                    original_weight_map,
                )
            )

        updated_weight_map.update(
            self._write_remaining_original_weights(
                original_weight_map,
                replaced_original_weights,
            )
        )
        index_path = self._refresh_weight_index(original_index, updated_weight_map)
        self._refresh_config(quant_ignore_layers)
        logger.info("Exported deploy model to {}", self.output_dir)
        logger.info("Refreshed weight index at {}", index_path)
        return {
            "index_path": index_path,
            "num_output_files": len(set(updated_weight_map.values())),
        }

    def _run_tensorwise(self):
        if not self._uses_tensor_deploy_plan():
            return self._run_legacy_tensorwise()

        is_ascend = getattr(self.args, "deploy_format", "legacy") == "ascend"
        original_index = self._load_weight_index()
        if not hasattr(self, "deploy_plan"):
            validate_tensor_deploy_args(self.args)
            self._prepare_tensor_deploy_plan(original_index)
        plan = self.deploy_plan
        quant_layers = plan.quant_layers
        self._output_tensor_bytes = 0
        self._quantized_modules = set()
        log_mtp_selection(plan.mtp, module_bits=plan.module_bits)
        if plan.unmatched_ignore_patterns:
            logger.warning(
                "ignore_layers patterns matched no modules: {}",
                sorted(plan.unmatched_ignore_patterns),
            )
        self._copy_support_files()
        original_weight_map = dict(original_index.get("weight_map", {}))
        weights_by_file = defaultdict(list)
        for weight_name, file_name in original_weight_map.items():
            weights_by_file[file_name].append(weight_name)

        updated_weight_map = {}
        model_dir = Path(self.model_path)
        loaded_files = {}
        convert_shard = getattr(self.pipeline, "convert_tensorwise_shard", None)
        if not callable(convert_shard):
            convert_shard = self._convert_tensorwise_shard
        for source_file in tqdm(sorted(weights_by_file), desc="Tensor convert..."):
            updated_weight_map.update(
                convert_shard(
                    source_file,
                    model_dir,
                    original_weight_map,
                    quant_layers,
                    loaded_files,
                )
            )
        validate_result = getattr(self.pipeline, "validate_tensorwise_result", None)
        if callable(validate_result):
            self._quantized_modules = set(validate_result(plan))
        if self._quantized_modules != plan.selected_modules:
            raise ValueError(
                "Actual quantized modules differ from JSON deployment plan"
            )
        extra_results = {}
        if is_ascend:
            describe = getattr(self.pipeline, "make_ascend_description", None)
            if not callable(describe):
                describe = make_ascend_description
            description = describe(set(updated_weight_map), plan.selected_modules)
            description_path = Path(self.output_dir) / "quant_model_description.json"
            self._write_json_file(description_path, description)
            output_tensor_bytes = getattr(
                self.pipeline, "output_tensor_bytes", self._output_tensor_bytes
            )
            index_path = self._refresh_weight_index(
                original_index, updated_weight_map, total_size=output_tensor_bytes
            )
            refresh = getattr(self.pipeline, "refresh_config", None)
            if callable(refresh):
                refresh(sorted(plan.ignored_modules), tensor_plan=plan)
            report = validate_ascend_artifacts(Path(self.output_dir), plan)
            validation_path = Path(self.output_dir) / "deployment_validation.json"
            self._write_json_file(validation_path, report)
            if report["status"] != "passed":
                raise ValueError(
                    f"Ascend artifact validation failed; see {validation_path}"
                )
            extra_results["description_path"] = str(description_path)
            extra_results["validation_path"] = str(validation_path)
        else:
            index_path = self._refresh_weight_index(original_index, updated_weight_map)
            refresh = getattr(self.pipeline, "refresh_config", None)
            if not callable(refresh):
                refresh = self._refresh_config
            refresh(sorted(plan.ignored_modules), tensor_plan=plan)
        log_mtp_selection(
            plan.mtp,
            phase="exported",
            quantized_modules=self._quantized_modules,
            module_bits=plan.module_bits,
        )
        logger.info("Exported tensor-converted model to {}", self.output_dir)
        logger.info("Refreshed weight index at {}", index_path)
        return {
            "index_path": index_path,
            "num_output_files": len(set(updated_weight_map.values())),
            **extra_results,
        }

    def _run_legacy_tensorwise(self):
        self._copy_support_files()
        original_index = self._load_weight_index()
        original_weight_map = dict(original_index.get("weight_map", {}))
        quant_layers = self.pipeline.generate_tensorwise_quant_layers()
        quant_ignore_layers = self.pipeline.generate_tensorwise_ignore_layers()
        weights_by_file = defaultdict(list)
        for weight_name, file_name in original_weight_map.items():
            weights_by_file[file_name].append(weight_name)

        updated_weight_map = {}
        model_dir = Path(self.model_path)
        loaded_files = {}
        for source_file in tqdm(sorted(weights_by_file), desc="Tensor convert..."):
            updated_weight_map.update(
                self._convert_tensorwise_shard(
                    source_file,
                    model_dir,
                    original_weight_map,
                    quant_layers,
                    loaded_files,
                )
            )
        index_path = self._refresh_weight_index(original_index, updated_weight_map)
        self._refresh_config(quant_ignore_layers)
        logger.info("Exported tensor-converted model to {}", self.output_dir)
        logger.info("Refreshed weight index at {}", index_path)
        return {
            "index_path": index_path,
            "num_output_files": len(set(updated_weight_map.values())),
        }

    def _prepare_tensor_deploy_plan(self, original_index=None):
        if original_index is None:
            original_index = self._load_weight_index()
        prepare = getattr(self.pipeline, "prepare_tensor_deploy_plan", None)
        if callable(prepare):
            self.deploy_plan = prepare(original_index["weight_map"])
        elif getattr(self.args, "deploy_format", "legacy") == "ascend":
            self._prepare_ascend_deploy_plan(original_index)
        else:
            self.deploy_plan = build_tensor_deploy_plan(
                original_index["weight_map"],
                self.args.deploy_selection,
                self.args.bit_policy,
                config=self.pipeline.config,
            )
        logger.info(
            "Tensor selection: {} modules, pattern counts: {}",
            len(self.deploy_plan.selected_modules),
            self.deploy_plan.match_counts,
        )
        return self.deploy_plan

    def _prepare_ascend_deploy_plan(self, original_index=None):
        if original_index is None:
            original_index = self._load_weight_index()
        if any(
            not isinstance(name, str) or Path(name).name != name
            for name in original_index["weight_map"].values()
        ):
            raise ValueError("Ascend shard names must be local filenames")
        if any(
            name == "rot.weight" or name.endswith(".rot.weight")
            for name in original_index["weight_map"]
        ):
            raise ValueError("Ascend source contains rotation weights")
        self.ascend_checkpoint_layout = detect_packed_gated_expert_layout(
            original_index["weight_map"], getattr(self.pipeline, "config", None)
        )
        plan_weight_map = original_index["weight_map"]
        if self.ascend_checkpoint_layout is not None:
            plan_weight_map = self.ascend_checkpoint_layout.expand_weight_map(
                plan_weight_map
            )
        self.deploy_plan = build_tensor_deploy_plan(
            plan_weight_map,
            self.args.deploy_selection,
            self.args.bit_policy,
            config=self.pipeline.config,
        )
        logger.info(
            "Ascend selection: {} quantized modules planned, {} ignored modules, "
            "MTP modules={} (ignored={}, selected={}), pattern counts={}, "
            "selected sample (up to 20)={}",
            len(self.deploy_plan.selected_modules),
            len(self.deploy_plan.ignored_modules),
            len(self.deploy_plan.mtp.modules),
            len(self.deploy_plan.mtp.ignored),
            len(self.deploy_plan.mtp.selected),
            self.deploy_plan.match_counts,
            sorted(self.deploy_plan.selected_modules)[:20],
        )
        self.ascend_deploy_plan = validate_ascend_deploy_plan(
            self.deploy_plan, plan_weight_map, self.pipeline
        )
        return self.ascend_deploy_plan

    def _convert_tensorwise_shard(
        self, source_file, model_dir, original_weight_map, quant_layers, loaded_files
    ):
        source_path = resolve_safetensors_path(
            model_dir, source_file, self.get_safetensors_files()
        )
        current_state_dict = load_file(str(source_path), device="cpu")
        is_ascend = getattr(self.args, "deploy_format", "legacy") == "ascend"
        checkpoint_layout = getattr(self, "ascend_checkpoint_layout", None)
        if is_ascend and checkpoint_layout is not None:
            selected_weight_keys = getattr(
                getattr(self, "ascend_deploy_plan", None), "selected_weight_keys", None
            )
            current_state_dict = checkpoint_layout.expand_tensors(
                current_state_dict, selected_weight_keys
            )
        if not is_ascend:
            loaded_files[source_file] = current_state_dict

        new_state_dict = {}
        for weight_name, weight in current_state_dict.items():
            # Ascend accepts original BF16 weights and preserves auxiliary tensors.
            # Source quantized-format decoding remains specific to legacy export.
            if not is_ascend:
                scale_prefix, scale_inv_name = self.pipeline.get_scale_name(weight_name)
                if weight_name.endswith(scale_prefix):
                    continue
                # NVFP4: scale_inv_name 为 (scale_name, scale_2_name) 元组，
                # `<name>_scale_2` 不满足 endswith("_scale")，需单独跳过。
                if isinstance(scale_inv_name, tuple) and weight_name in scale_inv_name:
                    continue
                block_size = self.pipeline.block_size(weight)
                weight = convert_state_dict(
                    weight,
                    weight_name,
                    scale_inv_name,
                    original_weight_map,
                    model_dir,
                    loaded_files,
                    block_size,
                    self.get_safetensors_files(),
                )
            new_state_dict[weight_name] = weight
            if self.quant_dtype in ["int", "mxfp"]:
                new_weight_name = weight_name.rsplit(".", 1)[0]
                is_weight = weight_name.endswith(".weight")
                if is_weight and new_weight_name in quant_layers:
                    bit = quant_layers[new_weight_name]
                    state_dict = self._export_tensor_payload(weight_name, weight, bit)
                    new_state_dict.update(state_dict)
                    if hasattr(self, "_quantized_modules"):
                        self._quantized_modules.add(new_weight_name)
        self._write_safetensor_file(source_file, new_state_dict)
        if is_ascend:
            self._output_tensor_bytes = getattr(self, "_output_tensor_bytes", 0) + sum(
                tensor.numel() * tensor.element_size()
                for tensor in new_state_dict.values()
            )
        return {weight_name: source_file for weight_name in new_state_dict}

    def _export_tensor_payload(
        self, weight_name: str, weight: torch.Tensor, bit: int
    ) -> dict[str, torch.Tensor]:
        is_ascend = getattr(self.args, "deploy_format", "legacy") == "ascend"
        if is_ascend:
            if (
                not weight_name.endswith(".weight")
                or weight.dtype != torch.bfloat16
                or weight.ndim != 2
                or weight.numel() == 0
            ):
                raise ValueError(
                    f"Ascend selected weight {weight_name} must be nonempty 2-D BF16"
                )
            if not torch.isfinite(weight).all():
                raise ValueError(f"Ascend selected weight {weight_name} must be finite")
        if weight.ndim != 2 or weight.numel() == 0 or not weight.is_floating_point():
            raise ValueError(
                f"Selected weight {weight_name} must be a nonempty 2-D floating tensor after source decoding"
            )
        quant_cls = DTYPE_REGISTRY.get(self.quant_dtype)
        payload = quant_payload(
            quant_cls,
            weight_name,
            weight,
            bit,
            block_size_col=getattr(self.args, "block_size_col", 128),
            scale_dtype=getattr(self.args, "scale_dtype", "fp32"),
        )
        if is_ascend:
            payload = adapt_ascend_payload(weight_name, payload)
        return payload

    def _write_block_file(self, layer_idx: int, layer_tensors: dict[str, object]):
        width = max(3, len(str(max(self.pipeline.num_layers - 1, 0))))
        file_name = f"layer_{layer_idx:0{width}d}.safetensors"
        self._write_safetensor_file(file_name, layer_tensors)
        return {weight_name: file_name for weight_name in layer_tensors}

    def _write_remaining_original_weights(
        self,
        original_weight_map: dict[str, str],
        replaced_original_weights: set[str],
    ):
        remaining_by_file = defaultdict(list)
        for weight_name, file_name in original_weight_map.items():
            if weight_name in replaced_original_weights:
                continue
            remaining_by_file[file_name].append(weight_name)

        max_shard_size = 8 * 1024**3
        rest_idx = 0
        current_tensors = {}
        current_size = 0
        updated_entries = {}

        def flush_current():
            nonlocal rest_idx, current_tensors, current_size
            if not current_tensors:
                return
            file_name = f"rest_{rest_idx:05d}.safetensors"
            self._write_safetensor_file(file_name, current_tensors)
            for weight_name in current_tensors:
                updated_entries[weight_name] = file_name
            rest_idx += 1
            current_tensors = {}
            current_size = 0

        model_dir = Path(self.model_path)
        for source_file in sorted(remaining_by_file):
            source_path = resolve_safetensors_path(
                model_dir, source_file, self.get_safetensors_files()
            )
            with safe_open(str(source_path), framework="pt", device="cpu") as f:
                for weight_name in remaining_by_file[source_file]:
                    tensor = f.get_tensor(weight_name)
                    tensor_size = tensor.numel() * tensor.element_size()
                    if current_tensors and current_size + tensor_size > max_shard_size:
                        flush_current()
                    current_tensors[weight_name] = tensor
                    current_size += tensor_size

        flush_current()
        return updated_entries

    def _write_safetensor_file(self, file_name: str, tensors: dict[str, object]):
        if not tensors:
            return
        output_path = Path(self.output_dir) / file_name
        tmp_path = output_path.parent / f".{output_path.name}.tmp"
        save_file(tensors, str(tmp_path))
        os.replace(tmp_path, output_path)
