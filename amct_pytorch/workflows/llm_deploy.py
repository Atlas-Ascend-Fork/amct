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
from amct_pytorch.quantization.dtypes import DTYPE_REGISTRY
from amct_pytorch.common.models.llm import register_llm_models
from amct_pytorch.common.models.llm.common.deploy_export import export_block_deploy, generate_quant_config, get_quant_ignore_linear_names
from amct_pytorch.common.utils.run_logging import ensure_log_dir, setup_run_logging
from amct_pytorch.quantization.dtypes import register_dtype
from amct_pytorch.quantization.dtypes.mxfp_impl import weight_dequant
from amct_pytorch.quantization.dtypes import int_impl


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
        if self.args.only_config:
            results = self._run_only_config()
        elif self.granularity == "block":
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

    def _is_w4a8(self):
        """Infer W4A8: moe.routed.w_bits == 4 in bit_policy."""
        try:
            w_bits, _ = self.args.bit_policy.linear_bits(group="moe.routed")
            return w_bits == 4
        except (AttributeError, KeyError):
            return False

    def _collect_extra_ignore_names(self):
        """适配器按需补充 get_quant_ignore_linear_names 遍历 block 拿不到的 ignore 条目。

        典型场景：非 nn.Linear 的模块（如 MoE router mlp.gate）、
        MTP 层（不在 num_layers 遍历范围内）、lm_head 等模型级模块。
        未实现时返回空列表，对 dense 模型无影响。
        """
        method = getattr(self.pipeline, "get_extra_ignore_names", None)
        if method is None:
            return []
        return method()

    def _collect_ignored_layers(self):
        """调用适配器的 get_ignored_layers，返回整层忽略的模块名列表。

        适配器未实现时返回 None，config.json 不出现 ignored_layers 字段。
        """
        method = getattr(self.pipeline, "get_ignored_layers", None)
        if method is None:
            return None
        return method()

    def _refresh_config(self, quant_ignore_layers):
        config_file = os.path.join(self.output_dir, 'config.json')
        with open(config_file, "r") as f:
            config = json.load(f)

        cache_scheme = {"kv_cache_scheme": self.pipeline.get_kv_cache_scheme(),
                        "li_cache_scheme": {
                            "type": "float" if self.is_mx else "int",
                            "num_bits": 8,
                        }}
        w4a8 = self._is_w4a8()
        ignored_layers = self._collect_ignored_layers()
        quantization_config = generate_quant_config(
            cache_scheme, quant_ignore_layers,
            ignored_layers=ignored_layers,
            w4a8=w4a8, w4a4=False, is_mx=self.is_mx)
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

    def _run_only_config(self):
        """Generate only config.json, bypassing all weight loading and export.

        Builds quant blocks with empty weights (init_empty_weights) to
        determine the ignore list via get_quant_ignore_linear_names, then
        writes config.json with the quantization_config.
        """
        self._copy_support_files()
        quant_ignore_layers = []
        for layer_idx in tqdm(range(self.pipeline.num_layers), desc="Config-only..."):
            block = self.pipeline.build_empty_quant_block(layer_idx)
            weight_prefix = self.pipeline.get_layer_weight_prefix(layer_idx)
            quant_ignore_layers.extend(
                get_quant_ignore_linear_names(block, weight_prefix)
            )
        quant_ignore_layers.extend(self._collect_extra_ignore_names())
        self._refresh_config(quant_ignore_layers)
        config_path = os.path.join(self.output_dir, "config.json")
        logger.info("Generated config-only deploy at {}", config_path)
        return {"index_path": None, "num_output_files": 0}

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
        # 处理block拿不到的ignore条目
        quant_ignore_layers.extend(self._collect_extra_ignore_names())
        self._refresh_config(quant_ignore_layers)
        logger.info("Exported deploy model to {}", self.output_dir)
        logger.info("Refreshed weight index at {}", index_path)
        return {
            "index_path": index_path,
            "num_output_files": len(set(updated_weight_map.values())),
        }

    def _build_ignore_layers(self):
        """通过空 block 遍历收集 ignore 列表，和 block 模式产出一致。"""
        quant_ignore_layers = []
        for layer_idx in range(self.pipeline.num_layers):
            block = self.pipeline.build_empty_quant_block(layer_idx)
            weight_prefix = self.pipeline.get_layer_weight_prefix(layer_idx)
            quant_ignore_layers.extend(get_quant_ignore_linear_names(block, weight_prefix))
        return quant_ignore_layers

    def _run_tensorwise(self):
        if self.quant_dtype in ("int", "mxfp"):
            return self._run_tensorwise_quant()

        self._copy_support_files()
        original_index = self._load_weight_index()
        original_weight_map = dict(original_index.get("weight_map", {}))

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
                elif weight.element_size() == 1:
                    # FP8 weight
                    try:
                        # Get scale_inv from the correct file
                        file_name = original_weight_map[scale_inv_name]
                        if file_name not in loaded_files:
                            file_path = model_dir / file_name
                            loaded_files[file_name] = load_file(file_path, device="cpu")
                        scale_inv = loaded_files[file_name][scale_inv_name]
                        if weight.dtype == torch.int8:
                            weight = weight_dequant(weight, scale_inv, block_size=32, is_mx=True, is_packed=True)
                        else:
                            weight = weight_dequant(weight, scale_inv)
                    except KeyError:
                        print(
                            f"Warning: Missing scale_inv tensor for {weight_name}, skipping conversion")
                    new_state_dict[weight_name] = weight
                    updated_weight_map[weight_name] = source_file
                else:
                    new_state_dict[weight_name] = weight
                    updated_weight_map[weight_name] = source_file

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

    def _run_tensorwise_quant(self):
        """tensor 粒度 int/mxfp 权重量化（不走前向传播，不构建 block 网络）。

        与 _run_tensorwise（FP8→BF16 反量化）不同，本函数直接对 BF16 权重
        做 per-channel min-max 量化，产出 int8 或 packed int4 + scale + bias。

        ===== 老逻辑（与 _run_tensorwise 相同的文件遍历框架）=====
        1. 复制非权重辅助文件（tokenizer、config 等）
        2. 读取 model.safetensors.index.json，按文件分组
        3. 逐文件加载 safetensors，逐权重处理，写回新 safetensors
        4. 缓存最近 2 个文件控制内存
        5. 刷新 index.json

        ===== 新逻辑（int/mxfp 量化独有）=====
        A. get_quant_weight_specs: 适配器返回 {权重名: w_bits}，
           告诉框架哪些权重量化、用几位。bits 来自 bit_policy，
           和 quant_module.py 里 QuantLinear 的解析方式一致。
        B. 量化权重: 按 quant_dtype 分流
           - int: weight_quant(real_quant=True) -> (qweight, scale, bias)
             bits=8: qweight 为 int8, scale 为 float32, bias 为 None
             bits=4: qweight 为 packed int4, scale 为 uint64, bias 为 float32
           - mxfp: QuantDequantMx.deploy() -> (qweight, scale)
             bits=8: qweight 为 float8_e4m3, scale 为 uint8(e8m0)
             bits=4: qweight 为 packed f4, scale 为 uint8(e8m0)
        C. _build_ignore_layers + _refresh_config: 生成 quantization_config，
           和 block 模式产出一致的 config.json。
        """
        # --- 老逻辑：复制辅助文件、读取权重索引 ---
        self._copy_support_files()
        original_index = self._load_weight_index()
        original_weight_map = dict(original_index.get("weight_map", {}))

        # --- 老逻辑：按文件分组权重 ---
        weights_by_file = defaultdict(list)
        for weight_name, file_name in original_weight_map.items():
            weights_by_file[file_name].append(weight_name)

        # --- 新逻辑 A: 适配器返回需要量化的权重及对应位宽 ---
        quant_specs = self.pipeline.get_quant_weight_specs()

        # --- 新逻辑 B: 按 quant_dtype 创建量化器（和 block 模式同一个 export_deploy）---
        # int: QuantDequantInt(bits).export_deploy -> {qweight, weight_scale, weight_bias}
        # mxfp: QuantDequantMx(bits).export_deploy -> {qweight, weight_scale}
        quant_objs = {}

        def quantize_fn(weight, bits):
            if bits not in quant_objs:
                quant_objs[bits] = DTYPE_REGISTRY.get(self.quant_dtype)(bits=bits)
            payload = quant_objs[bits].export_deploy(weight)
            return payload["qweight"], payload["weight_scale"], payload.get("weight_bias")

        # --- 老逻辑：逐文件加载、处理、写回 ---
        updated_weight_map = {}
        model_dir = Path(self.model_path)
        loaded_files = {}
        for source_file in tqdm(sorted(weights_by_file), desc="Tensor convert..."):
            source_path = model_dir / source_file
            current_state_dict = load_file(source_path, device="cpu")
            loaded_files[source_file] = current_state_dict

            new_state_dict = {}
            for weight_name, weight in current_state_dict.items():
                if weight_name in quant_specs:
                    # --- 新逻辑 B: 对权重做量化（int 或 mxfp）---
                    qx, scale, bias = quantize_fn(weight, quant_specs[weight_name])
                    new_state_dict[weight_name] = qx
                    # scale 命名: xxx.weight -> xxx.weight_scale
                    scale_name = weight_name.replace(".weight", ".weight_scale")
                    new_state_dict[scale_name] = scale
                    updated_weight_map[weight_name] = source_file
                    updated_weight_map[scale_name] = source_file
                    if bias is not None:
                        # int W4 路径产出 assistance bias，供 infer 仓 MoEGMM 算子使用
                        bias_name = weight_name.replace(".weight", ".weight_bias")
                        new_state_dict[bias_name] = bias
                        updated_weight_map[bias_name] = source_file
                else:
                    # 不在 quant_specs 中的权重原样保留
                    new_state_dict[weight_name] = weight
                    updated_weight_map[weight_name] = source_file

            self._write_safetensor_file(source_file, new_state_dict)
            for weight_name in new_state_dict:
                updated_weight_map[weight_name] = source_file

            # --- 老逻辑：缓存控制，只保留最近 2 个文件 ---
            if len(loaded_files) > 2:
                oldest_file = next(iter(loaded_files))
                if oldest_file != source_file:
                    del loaded_files[oldest_file]

        # --- 老逻辑：刷新 index.json ---
        index_path = self._refresh_weight_index(original_index, updated_weight_map)

        # --- 新逻辑 C: 生成 quantization_config（与 block 模式一致）---
        quant_ignore_layers = self._build_ignore_layers()
        quant_ignore_layers.extend(self._collect_extra_ignore_names())
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
