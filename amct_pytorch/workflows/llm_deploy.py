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
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME
from tqdm import tqdm

from amct_pytorch.algorithms.quant import register_algorithms
from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm import register_llm_models
from amct_pytorch.common.models.llm.common.deploy_export import (
    export_block_deploy,
    generate_quant_config,
    convert_state_dict,
    quant_payload,
)
from amct_pytorch.common.utils.run_logging import ensure_log_dir, setup_run_logging
from amct_pytorch.quantization.dtypes import DTYPE_REGISTRY, register_dtype


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
        # Align with HF/vLLM checkpoint loading: read index.json when present;
        # otherwise this is a single-shard model -- map every tensor to the lone
        # safetensors file and synthesize an equivalent index. Filenames reuse the
        # transformers ecosystem constants to avoid hardcoded drift.
        index_path = Path(self.model_path) / SAFE_WEIGHTS_INDEX_NAME
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        single_path = Path(self.model_path) / SAFE_WEIGHTS_NAME
        if not single_path.exists():
            raise FileNotFoundError(
                f"Neither {SAFE_WEIGHTS_INDEX_NAME} nor {SAFE_WEIGHTS_NAME} "
                f"found in {self.model_path}"
            )
        with safe_open(str(single_path), framework="pt") as f:
            weight_map = {key: SAFE_WEIGHTS_NAME for key in f.keys()}
        # total_size is a placeholder; _refresh_weight_index() recomputes and
        # overwrites it from the actual output shard sizes.
        return {
            "metadata": {"total_size": single_path.stat().st_size},
            "weight_map": weight_map,
        }

    def _refresh_config(self, quant_ignore_layers):
        config_file = os.path.join(self.output_dir, 'config.json')
        with open(config_file, "r") as f:
            config = json.load(f)
        if self.quant_dtype is not None:
            cache_scheme_fn = getattr(self.pipeline, "cache_scheme", None)
            cache_scheme = cache_scheme_fn() if callable(cache_scheme_fn) else None
            bits_scheme_fn = getattr(self.pipeline, "bits_scheme", None)
            bits_scheme = bits_scheme_fn() if callable(bits_scheme_fn) else None
            quantization_config = generate_quant_config(
                cache_scheme, quant_ignore_layers, is_mx=self.is_mx, bits_scheme=bits_scheme)
            config['quantization_config'] = quantization_config
        else:
            config.pop('quantization_config', None)

        new_config_file = os.path.join(self.output_dir, "config.json")
        with open(new_config_file, "w") as f:
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
            source_path = model_dir / source_file
            current_state_dict = load_file(source_path, device="cpu")
            loaded_files[source_file] = current_state_dict

            new_state_dict = {}
            for weight_name, weight in current_state_dict.items():
                scale_prefix, scale_inv_name = self.pipeline.get_scale_name(weight_name)
                if weight_name.endswith(scale_prefix):
                    continue
                # FP8 -> bf16
                block_size = self.pipeline.block_size(weight)
                weight = convert_state_dict(weight, weight_name, scale_inv_name, original_weight_map,
                                            model_dir, loaded_files, block_size)
                new_state_dict[weight_name] = weight
                if self.quant_dtype in ["int", "mxfp"]:
                    quant_cls = DTYPE_REGISTRY.get(self.quant_dtype)
                    new_weight_name = weight_name.rsplit(".", 1)[0]
                    if new_weight_name in quant_layers:
                        bit = quant_layers[new_weight_name]
                        state_dict = quant_payload(quant_cls, weight_name, weight, bit)
                        new_state_dict.update(state_dict)
            self._write_safetensor_file(source_file, new_state_dict)
            for weight_name in new_state_dict:
                updated_weight_map[weight_name] = source_file
        index_path = self._refresh_weight_index(original_index, updated_weight_map)
        self._refresh_config(quant_ignore_layers)
        logger.info("Exported tensor-converted model to {}", self.output_dir)
        logger.info("Refreshed weight index at {}", index_path)
        return {
            "index_path": index_path,
            "num_output_files": len(set(updated_weight_map.values())),
        }

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
