#!/usr/bin/env python3
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

"""Bake HiGPTQ-calibrated weights into a plain BF16 checkpoint.

ptq_result/ only stores the Hessian-error-compensated *floating-point* weight
(QuantLinear.export_ptq_params() just returns self.linear.weight) -- it is not
yet quantized. amct_pytorch.eval fake-quantizes it on the fly at forward time
via WeightQuantizer -> QuantDequantHifp.forward() (hifloat4_fake_quant, bits=4),
which is a pure quant/dequant simulation and needs no bit-packing. With
--quant_dtype hifp, amct_pytorch.deploy can also call
QuantDequantHifp.export_deploy() for real HiFloat4 4-bit packing (see
amct_pytorch/quantization/dtypes/hifp.py).

This script instead targets deployment on engines that don't have a HiFloat4
kernel (e.g. vLLM): it runs each calibrated weight through the same fake-quant
simulation eval already exercises, bakes the result into BF16, and writes a
normal HF/vLLM-loadable checkpoint (safetensors + copied config/tokenizer
files) -- no packing required, at the cost of storing weights at BF16 size
rather than HiFloat4 size.

Mirrors amct_pytorch/common/models/llm/common/deploy_export.py's
export_block_deploy() and amct_pytorch/workflows/llm_deploy.py's safetensors
shard read/write helpers, but is kept as a standalone script (does not modify
those files or amct_pytorch's public deploy CLI) since its output is a
different kind of artifact -- a dense BF16 model, not a packed quant format --
and doesn't need llm_deploy.py's compressed-tensors quantization_config export.

Usage mirrors scripts/run_eval_higptq.sh: pass every --quant_target you
calibrated with HiGPTQ, together with the matching --*_param_dir. Any
QuantLinear in scope whose target wasn't loaded (missing/omitted param dir)
still gets fake-quantized, just without HiGPTQ's error compensation -- this
matches amct_pytorch.deploy's existing behavior, not a new gap.

    python scripts/export_higptq_bf16.py \
        --model /path/to/Qwen3.5-9B \
        --model_name qwen3_5 \
        --quant_dtype hifp \
        --bit_config amct_pytorch/configs/hifloat4.yaml \
        --quant_target attn-linear mlp \
        --attn_linear_param_dir ptq_result/ptq_params/qwen3_5/attn-linear \
        --moe_mlp_param_dir ptq_result/ptq_params/qwen3_5/mlp \
        --output_dir ptq_result/bf16_export/qwen3_5
"""

import json
import os
import shutil
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME
from tqdm import tqdm

from amct_pytorch.algorithms.quant import register_algorithms
from amct_pytorch.cli.llm.args import parser_gen
from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm import register_llm_models
from amct_pytorch.common.models.llm.common.quant_apply import set_model_weight_quant_state
from amct_pytorch.quantization.dtypes import register_dtype

_MAX_SHARD_SIZE = 8 * 1024 ** 3


def _is_weight_file(path: Path) -> bool:
    return path.name == SAFE_WEIGHTS_INDEX_NAME or path.suffix == ".safetensors"


def _copy_support_files(model_path, output_dir):
    src_dir, dst_dir = Path(model_path), Path(output_dir)
    for src_path in src_dir.iterdir():
        if src_path.name.startswith(".") or _is_weight_file(src_path):
            continue
        dst_path = dst_dir / src_path.name
        if dst_path.exists():
            continue
        if src_path.is_dir():
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)


def _load_weight_index(model_path):
    index_path = Path(model_path) / SAFE_WEIGHTS_INDEX_NAME
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)
    single_path = Path(model_path) / SAFE_WEIGHTS_NAME
    if not single_path.exists():
        raise FileNotFoundError(
            f"Neither {SAFE_WEIGHTS_INDEX_NAME} nor {SAFE_WEIGHTS_NAME} found in {model_path}"
        )
    with safe_open(str(single_path), framework="pt") as f:
        weight_map = {key: SAFE_WEIGHTS_NAME for key in f.keys()}
    return {"metadata": {"total_size": single_path.stat().st_size}, "weight_map": weight_map}


def _write_safetensor_file(output_dir, file_name, tensors):
    if not tensors:
        return
    output_path = Path(output_dir) / file_name
    tmp_path = output_path.parent / f".{output_path.name}.tmp"
    save_file(tensors, str(tmp_path))
    os.replace(tmp_path, output_path)


def _write_remaining_original_weights(model_path, output_dir, original_weight_map, replaced):
    remaining_by_file = {}
    for weight_name, file_name in original_weight_map.items():
        if weight_name in replaced:
            continue
        remaining_by_file.setdefault(file_name, []).append(weight_name)

    rest_idx = 0
    current_tensors, current_size = {}, 0
    updated_entries = {}

    def flush():
        nonlocal rest_idx, current_tensors, current_size
        if not current_tensors:
            return
        file_name = f"rest_{rest_idx:05d}.safetensors"
        _write_safetensor_file(output_dir, file_name, current_tensors)
        for weight_name in current_tensors:
            updated_entries[weight_name] = file_name
        rest_idx += 1
        current_tensors, current_size = {}, 0

    model_dir = Path(model_path)
    for source_file in sorted(remaining_by_file):
        with safe_open(str(model_dir / source_file), framework="pt", device="cpu") as f:
            for weight_name in remaining_by_file[source_file]:
                tensor = f.get_tensor(weight_name)
                tensor_size = tensor.numel() * tensor.element_size()
                if current_tensors and current_size + tensor_size > _MAX_SHARD_SIZE:
                    flush()
                current_tensors[weight_name] = tensor
                current_size += tensor_size
    flush()
    return updated_entries


def _refresh_weight_index(output_dir, original_index, updated_weight_map):
    metadata = dict(original_index.get("metadata", {}))
    total_size = 0
    for file_name in sorted(set(updated_weight_map.values())):
        total_size += os.path.getsize(os.path.join(output_dir, file_name))
    metadata["total_size"] = total_size
    output_index = {"metadata": metadata, "weight_map": updated_weight_map}
    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(output_index, f, ensure_ascii=False, indent=2, sort_keys=True)
    return index_path


def _export_block_fakequant_bf16(pipeline, layer_idx):
    """Load this layer's HiGPTQ ptq_result and fake-quantize each QuantLinear's
    calibrated weight, returning {original_weight_name: bf16_tensor}.

    Uses the WeightQuantizer's fake-quant *simulation* path
    (module.weight_quantizer(weight), same as amct_pytorch.eval) rather than
    its export_deploy() real-packing path -- this script's whole point is a
    dense BF16 checkpoint for engines without a HiFloat4 kernel, not a packed
    one (use amct_pytorch.deploy for that).
    """
    block = pipeline.build_quant_block(layer_idx).to(pipeline.args.device).eval()
    pipeline.load_selected_layer_ptq_params(layer_idx, block, strict=False)
    set_model_weight_quant_state(block, True)

    tensors = {}
    with torch.no_grad():
        for weight_key, module in pipeline.iter_deploy_bindings(layer_idx, block):
            fq_weight = module.weight_quantizer(module.linear.weight)
            tensors[weight_key] = fq_weight.detach().to(torch.bfloat16).cpu()
    return tensors


def export_bf16(args):
    register_llm_models()
    register_dtype()
    register_algorithms()

    model_cls = MODEL_REGISTRY.get(args.model_name)
    pipeline = model_cls(args)

    os.makedirs(args.output_dir, exist_ok=True)
    _copy_support_files(args.model, args.output_dir)

    original_index = _load_weight_index(args.model)
    original_weight_map = dict(original_index.get("weight_map", {}))
    updated_weight_map = {}
    replaced = set()

    width = max(3, len(str(max(pipeline.num_layers - 1, 0))))
    for layer_idx in tqdm(range(pipeline.num_layers), desc="Block processing..."):
        layer_tensors = _export_block_fakequant_bf16(pipeline, layer_idx)
        if not layer_tensors:
            continue
        file_name = f"layer_{layer_idx:0{width}d}.safetensors"
        _write_safetensor_file(args.output_dir, file_name, layer_tensors)
        for weight_key in layer_tensors:
            updated_weight_map[weight_key] = file_name
            if weight_key in original_weight_map:
                replaced.add(weight_key)

    updated_weight_map.update(
        _write_remaining_original_weights(args.model, args.output_dir, original_weight_map, replaced)
    )
    index_path = _refresh_weight_index(args.output_dir, original_index, updated_weight_map)
    logger.info("Exported BF16 checkpoint to {}", args.output_dir)
    logger.info("Refreshed weight index at {}", index_path)


def main():
    args = parser_gen(command="export_higptq_bf16")
    export_bf16(args)


if __name__ == "__main__":
    main()
