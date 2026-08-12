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

"""Bake plain (uncalibrated) HiFloat4 fake-quant weights into a BF16 checkpoint.

Unlike scripts/export_higptq_bf16.py -- which needs a prior HiGPTQ calibration
run and its ptq_result/ptq_params/ -- this script needs no calibration data at
all: it loads the model as-is and round-trips every QuantLinear's raw weight
through the same weight-only HiFloat4 fake-quant simulation amct_pytorch.eval
exercises with `--eval_mode quant --quant_dtype hifp
--bit_config amct_pytorch/configs/hifloat4.yaml` (see scripts/run_eval_hif4.sh).
That's plain RTN (round-to-nearest): each weight is independently
quantize-dequantized in HiFloat4's 64-element block format, with no Hessian
error compensation. The result is written as a normal HF/vLLM-loadable BF16
checkpoint (safetensors + copied config/tokenizer files) -- no bit-packing, at
the cost of storing weights at BF16 size rather than HiFloat4 size.

Reuses scripts/export_higptq_bf16.py's safetensors shard read/write helpers
(_copy_support_files, _load_weight_index, _write_safetensor_file,
_write_remaining_original_weights, _refresh_weight_index) rather than
duplicating them -- only the per-block quantization step differs (no
ptq_result to load here).

Usage mirrors scripts/run_eval_hif4.sh:

    python scripts/export_hif4_bf16.py \
        --model /path/to/Qwen3.5-9B \
        --model_name qwen3_5 \
        --quant_dtype hifp \
        --bit_config amct_pytorch/configs/hifloat4.yaml \
        --quant_target attn-linear mlp \
        --output_dir ptq_result/hif4_bf16_export/qwen3_5

Do not pass --algos gptq/higptq here -- that would pull in calibration
machinery this script deliberately skips; use export_higptq_bf16.py instead
for calibrated weights.
"""

import os

import torch
from loguru import logger
from tqdm import tqdm

from amct_pytorch.algorithms.quant import register_algorithms
from amct_pytorch.cli.llm.args import parser_gen
from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm import register_llm_models
from amct_pytorch.common.models.llm.common.quant_apply import set_model_weight_quant_state
from amct_pytorch.quantization.dtypes import register_dtype

from export_higptq_bf16 import (
    _copy_support_files,
    _load_weight_index,
    _refresh_weight_index,
    _write_remaining_original_weights,
    _write_safetensor_file,
)


def _export_block_rtn_fakequant_bf16(pipeline, layer_idx):
    """Fake-quantize each QuantLinear's raw weight in this layer with no prior
    calibration, returning {original_weight_name: bf16_tensor}."""
    block = pipeline.build_quant_block(layer_idx).to(pipeline.args.device).eval()
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
        layer_tensors = _export_block_rtn_fakequant_bf16(pipeline, layer_idx)
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
    logger.info("Exported HiFloat4 RTN BF16 checkpoint to {}", args.output_dir)
    logger.info("Refreshed weight index at {}", index_path)


def main():
    args = parser_gen(command="export_hif4_bf16")
    export_bf16(args)


if __name__ == "__main__":
    main()
