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

from amct_pytorch.algorithms.quant import register_algorithms
from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm import register_llm_models
from amct_pytorch.common.models.llm.common.deploy_export import export_block_deploy, generate_quant_config
from amct_pytorch.common.utils.run_logging import ensure_log_dir, setup_run_logging
from amct_pytorch.quantization.dtypes import DTYPE_REGISTRY, register_dtype
from amct_pytorch.quantization.dtypes.mxfp_impl import weight_dequant


def _group_weights_by_file(weight_map):
    weights_by_file = defaultdict(list)
    for weight_name, file_name in weight_map.items():
        weights_by_file[file_name].append(weight_name)
    return weights_by_file


class LlmDeployWorkflow:
    """Export deploy-ready quantized artifacts."""

    def __init__(self, args):
        self.args = args
        self.granularity = args.granularity
        self.pipeline = None
        self.model_name = args.model_name
        self.model_path = args.model
        self.quant_dtype = args.quant_dtype
        self.output_dir = self.args.output_dir
        self.is_mx = self.quant_dtype.startswith("mx")
        self.is_int = self.quant_dtype.startswith("int")
        self.is_hif = self.quant_dtype.startswith("hif")
        self._tensorwise_quant_layers = None

    @staticmethod
    def _is_weight_file(path: Path) -> bool:
        return path.name == "model.safetensors.index.json" or path.suffix == ".safetensors"

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
        sink_id = self.setup()
        if self.granularity == "block":
            results = self._run_blockwise()
        elif self.granularity == "tensor":
            results = self._run_tensorwise()
        else:
            raise ValueError(
                f"Unsupported granularity '{self.granularity}' for deploy."
            )
        logger.remove(sink_id)
        return results

    def setup(self):
        os.makedirs(self.output_dir, exist_ok=True)
        ensure_log_dir(self.args)
        self._register_components()
        self.pipeline = self._build_pipeline()
        sink_id, _ = setup_run_logging(self.args, "deploy")
        return sink_id

    def _build_pipeline(self):
        model_cls = MODEL_REGISTRY.get(self.model_name)
        return model_cls(self.args)

    def _convert_tensor(self, weight_name: str, tensor: torch.Tensor) -> torch.Tensor:
        if self.quant_dtype == "bf16":
            return tensor.to(torch.bfloat16)
        raise NotImplementedError(
            f"tensor granularity does not support quant_dtype '{self.quant_dtype}' yet"
        )

    def _copy_support_files(self):
        src_dir = Path(self.model_path)
        dst_dir = Path(self.output_dir)
        for src_path in src_dir.iterdir():
            if src_path.name.startswith("."):
                continue
            if self._is_weight_file(src_path):
                continue
            dst_path = dst_dir / src_path.name
            if dst_path.exists():
                continue
            if src_path.is_dir():
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

    def _load_weight_index(self):
        index_path = Path(self.model_path) / "model.safetensors.index.json"
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_model_config(self):
        config_path = Path(self.model_path) / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _refresh_config(self, quant_ignore_layers):
        config_file = os.path.join(self.output_dir, 'config.json')
        with open(config_file, "r") as f:
            config = json.load(f)

        cache_scheme = {"kv_cache_scheme": {"num_bits": 8, "type": "float"},
                        "li_cache_scheme": {
                            "type": "float" if self.is_mx else "int",
                            "num_bits": 8,
                        }}
        quantization_config = generate_quant_config(
            cache_scheme, quant_ignore_layers, w4a8=False, w4a4=False, is_mx=self.is_mx)
        config['quantization_config'] = quantization_config

        new_config_file = os.path.join(self.output_dir, "config.json")
        with open(new_config_file, "w") as f:
            json.dump(config, f, indent=2)

    def _refresh_config_tensor(self):
        config_file = os.path.join(self.output_dir, "config.json")
        with open(config_file, "r") as f:
            config = json.load(f)
        if self.quant_dtype == "bf16":
            config["torch_dtype"] = "bfloat16"
            config.pop("quantization_config", None)
        else:
            quant_config_fn = getattr(self.pipeline, "generate_tensorwise_quant_config", None)
            if callable(quant_config_fn):
                quantization_config = quant_config_fn(
                    config, getattr(self, "_tensorwise_quant_layers", None))
                if quantization_config is not None:
                    config["quantization_config"] = quantization_config
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

    def _refresh_weight_index(self, original_index, updated_weight_map):
        metadata = dict(original_index.get("metadata", {}))
        total_size = 0
        for file_name in sorted(set(updated_weight_map.values())):
            total_size += os.path.getsize(os.path.join(self.output_dir, file_name))
        metadata["total_size"] = total_size

        output_index = {
            "metadata": metadata,
            "weight_map": updated_weight_map,
        }
        index_path = os.path.join(self.output_dir, "model.safetensors.index.json")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(output_index, f, ensure_ascii=False, indent=2, sort_keys=True)
        return index_path

    def _run_blockwise(self):
        self._copy_support_files()
        quant_ignore_layers = []
        original_index = self._load_weight_index()
        original_weight_map = dict(original_index.get("weight_map", {}))
        updated_weight_map = {}
        replaced_original_weights = set()
        for layer_idx in tqdm(range(self.pipeline.num_layers), desc="Block Processing..."):
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
        self._copy_support_files()
        model_config = self._load_model_config()
        original_index = self._load_weight_index()
        original_weight_map = dict(original_index.get("weight_map", {}))
        quant_layers = self._prepare_tensorwise_quant_layers(model_config, original_weight_map)
        weights_by_file = _group_weights_by_file(original_weight_map)
        updated_weight_map = {}
        model_dir = Path(self.model_path)
        loaded_files = {}

        for source_file in tqdm(sorted(weights_by_file), desc="Tensor convert..."):
            source_path = model_dir / source_file
            current_state_dict = load_file(source_path, device="cpu")
            loaded_files[source_file] = current_state_dict
            new_state_dict = self._convert_tensorwise_state_dict(
                current_state_dict,
                quant_layers,
                original_weight_map,
                model_dir,
                loaded_files,
            )
            self._write_safetensor_file(source_file, new_state_dict)
            for weight_name in new_state_dict:
                updated_weight_map[weight_name] = source_file

        index_path = self._refresh_weight_index(original_index, updated_weight_map)
        self._refresh_config_tensor()
        logger.info("Exported tensor-converted model to {}", self.output_dir)
        logger.info("Refreshed weight index at {}", index_path)
        return {
            "index_path": index_path,
            "num_output_files": len(set(updated_weight_map.values())),
        }

    def _prepare_tensorwise_quant_layers(self, model_config, original_weight_map):
        quant_layers = None
        if self.quant_dtype != "bf16":
            quant_layers = self._generate_tensorwise_quant_layers(model_config, original_weight_map)
        self._tensorwise_quant_layers = quant_layers
        return quant_layers

    def _convert_tensorwise_state_dict(
        self,
        state_dict,
        quant_layers,
        original_weight_map,
        model_dir,
        loaded_files,
    ):
        new_state_dict = {}
        for weight_name, weight in state_dict.items():
            converted = self._convert_tensorwise_weight(
                weight_name,
                weight,
                quant_layers,
                original_weight_map,
                model_dir,
                loaded_files,
            )
            if converted is not None:
                new_state_dict.update(converted)
        return new_state_dict

    def _convert_tensorwise_weight(
        self,
        weight_name,
        weight,
        quant_layers,
        original_weight_map,
        model_dir,
        loaded_files,
    ):
        deploy_info = self._get_tensorwise_deploy_info(weight_name, quant_layers)
        if deploy_info.get("skip", False):
            return None

        can_export_quant = "bit" in deploy_info
        if weight.element_size() == 1 and deploy_info.get("scale_name"):
            weight, can_export_quant = self._dequant_tensorwise_weight(
                weight_name,
                weight,
                deploy_info,
                original_weight_map,
                model_dir,
                loaded_files,
                can_export_quant,
            )

        if can_export_quant:
            return self._format_tensorwise_quant_payload(weight_name, weight, deploy_info["bit"])
        if self.quant_dtype == "bf16" and weight.is_floating_point():
            weight = weight.to(torch.bfloat16)
        return {weight_name: weight}

    @staticmethod
    def _dequant_tensorwise_weight(
        weight_name,
        weight,
        deploy_info,
        original_weight_map,
        model_dir,
        loaded_files,
        can_export_quant,
    ):
        scale_inv_name = deploy_info["scale_name"]
        try:
            file_name = original_weight_map[scale_inv_name]
            if file_name not in loaded_files:
                file_path = model_dir / file_name
                loaded_files[file_name] = load_file(file_path, device="cpu")
            scale_inv = loaded_files[file_name][scale_inv_name]
        except KeyError:
            logger.warning(
                "Missing scale tensor '{}' for '{}', skipping conversion.",
                scale_inv_name,
                weight_name,
            )
            return weight, False

        block_size = deploy_info.get("block_size", 32 if weight.dtype == torch.int8 else 128)
        is_mx = deploy_info.get("is_mx", weight.dtype == torch.int8)
        is_packed = deploy_info.get("is_packed", weight.dtype == torch.int8)
        weight = weight_dequant(weight, scale_inv, block_size=block_size, is_mx=is_mx, is_packed=is_packed)
        return weight, can_export_quant

    def _format_tensorwise_quant_payload(self, weight_name, weight, bit):
        payload = self._export_tensorwise_quant_weight(weight, bit)
        tensors = {weight_name: payload["qweight"]}
        for extra_name, extra_tensor in payload.items():
            if extra_name == "qweight" or extra_tensor is None:
                continue
            extra_key = weight_name.replace(".weight", f".{extra_name}")
            tensors[extra_key] = extra_tensor
        return tensors

    def _generate_tensorwise_quant_layers(self, config: dict, weight_map: dict[str, str]):
        quant_layers_fn = getattr(self.pipeline, "generate_tensorwise_quant_layers", None)
        if not callable(quant_layers_fn):
            return None
        return quant_layers_fn(config, weight_map)

    def _export_tensorwise_quant_weight(self, weight: torch.Tensor, bit: int) -> dict[str, torch.Tensor]:
        quant_cls = DTYPE_REGISTRY.get(self.quant_dtype)
        quant_obj = quant_cls(bits=int(bit))
        return quant_obj.export_deploy(weight)

    def _get_tensorwise_deploy_info(self, weight_name: str, quant_layers=None) -> dict:
        deploy_info_fn = getattr(self.pipeline, "get_tensorwise_deploy_info", None)
        if callable(deploy_info_fn):
            deploy_info = deploy_info_fn(weight_name)
            if deploy_info is not None:
                return dict(deploy_info)

        scale_name_fn = getattr(self.pipeline, "get_scale_name", None)
        if callable(scale_name_fn):
            scale_prefix, scale_inv_name = scale_name_fn(weight_name)
            if weight_name.endswith(scale_prefix):
                return {"skip": True}
            if quant_layers is not None:
                layer_name = weight_name.rsplit(".", 1)[0]
                if layer_name in quant_layers:
                    return {"scale_name": scale_inv_name, "bit": quant_layers[layer_name]}
                return {}
            return {"scale_name": scale_inv_name}
        return {}

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

        max_shard_size = 8 * 1024 ** 3
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
            source_path = model_dir / source_file
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
