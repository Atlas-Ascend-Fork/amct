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
from collections import defaultdict
from pathlib import Path

import torch

from loguru import logger
from safetensors.torch import load_file
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
from amct_pytorch.common.utils.checkpoint_io import CheckpointShardWriter
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
    def _register_components():
        register_llm_models()
        register_dtype()
        register_algorithms()

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
        self._maybe_decode_source()
        self.pipeline = self._build_pipeline()
        sink_id, _ = setup_run_logging(self.args, "deploy")
        return sink_id

    def _maybe_decode_source(self):
        """Auto-detect a foreign FP4/FP8 source and decode it to BF16 before deploy.

        Deploy loads a float model, then packs weights to HiF4 (``QuantDequantHiF4.export_deploy``).
        For nvfp4/mxfp4/mxfp8 input, decode to a temp BF16 checkpoint first (``fpx_decode``) and
        point the pipeline at it; fp16/bf16 pass through unchanged.
        """
        from amct_pytorch.common.utils.fpx_decode.detect import (
            detect_source_format, decode_to_bf16, FOREIGN_FORMATS,
        )
        fmt = detect_source_format(self.model_path)
        logger.info("Deploy source format detected: {}", fmt)
        if fmt in FOREIGN_FORMATS:
            decoded = os.path.join(self.output_dir, "_decoded_bf16")
            logger.info("Decoding {} -> BF16 at {} before HiF4 pack", fmt, decoded)
            decode_to_bf16(self.model_path, decoded, fmt)
            self.model_path = decoded
            self.args.model = decoded

    def _build_pipeline(self):
        model_cls = MODEL_REGISTRY.get(self.model_name)
        return model_cls(self.args)

    def _convert_tensor(self, weight_name: str, tensor: torch.Tensor) -> torch.Tensor:
        if self.quant_dtype == "bf16":
            return tensor.to(torch.bfloat16)
        raise NotImplementedError(
            f"tensor granularity does not support quant_dtype '{self.quant_dtype}' yet"
        )

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

    def _refresh_config_tensor(self):
        config_file = os.path.join(self.output_dir, 'config.json')
        with open(config_file, "r") as f:
            config = json.load(f)
        if self.quant_dtype == "bf16":
            config["torch_dtype"] = "bfloat16"
            config.pop('quantization_config', None)
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

    def _export_block(self, layer_idx, quant_ignore_layers):
        """Per-layer export hook. Default: real HiF4/int/mxfp pack (``export_block_deploy``).
        Subclasses (e.g. ptq convert) override this to emit fake-quantized float weights."""
        return export_block_deploy(self.pipeline, layer_idx, quant_ignore_layers)

    def _run_blockwise(self):
        # Local writer bound to the *current* model_path (post foreign-format decode) / output_dir.
        writer = CheckpointShardWriter(self.model_path, self.output_dir)
        writer.copy_support_files()
        quant_ignore_layers = []
        original_index = writer.load_weight_index()
        original_weight_map = dict(original_index.get("weight_map", {}))
        updated_weight_map = {}
        replaced_original_weights = set()
        for layer_idx in tqdm(range(self.pipeline.num_layers), desc="Block Processing..."):
            layer_tensors, tensor_routes = self._export_block(layer_idx, quant_ignore_layers)
            if not layer_tensors:
                continue
            updated_weight_map.update(
                writer.write_block_file(layer_idx, self.pipeline.num_layers, layer_tensors)
            )
            replaced_original_weights.update(
                writer.collect_replaced_original_weights(
                    layer_tensors,
                    tensor_routes,
                    original_weight_map,
                )
            )

        updated_weight_map.update(
            writer.write_remaining_original_weights(
                original_weight_map,
                replaced_original_weights,
            )
        )
        index_path = writer.refresh_weight_index(original_index, updated_weight_map)
        self._refresh_config(quant_ignore_layers)
        logger.info("Exported deploy model to {}", self.output_dir)
        logger.info("Refreshed weight index at {}", index_path)
        return {
            "index_path": index_path,
            "num_output_files": len(set(updated_weight_map.values())),
        }

    def _run_tensorwise(self):
        writer = CheckpointShardWriter(self.model_path, self.output_dir)
        writer.copy_support_files()
        original_index = writer.load_weight_index()
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
            writer.write_safetensor_file(source_file, new_state_dict)
            for weight_name in new_state_dict:
                updated_weight_map[weight_name] = source_file
        index_path = writer.refresh_weight_index(original_index, updated_weight_map)
        self._refresh_config(quant_ignore_layers)
        logger.info("Exported tensor-converted model to {}", self.output_dir)
        logger.info("Refreshed weight index at {}", index_path)
        return {
            "index_path": index_path,
            "num_output_files": len(set(updated_weight_map.values())),
        }
