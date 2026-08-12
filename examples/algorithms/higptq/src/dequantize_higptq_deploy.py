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

"""Dequantize a real HiFloat4-packed `amct_pytorch.deploy` checkpoint back to BF16.

`amct_pytorch.deploy` (--quant_dtype hifp --algos higptq ...) already produces a
deployable checkpoint by packing each calibrated weight for real, through
QuantDequantHifp.export_deploy() -> hif4_pack(): every quantized nn.Linear weight
ends up stored as a uint8 "<name>.weight" (packed 4-bit codes) plus a uint8
"<name>.scale" sibling tensor (see
amct_pytorch/common/models/llm/common/deploy_export.py:export_block_deploy). That
packed format needs a HiFloat4 unpacking kernel to run (e.g. on Ascend NPU via
amct_ops); engines without one (e.g. vLLM) can't load it as-is.

hif4_unpack() is the exact inverse of hif4_pack() -- same encode/decode grid, so
unpacking loses no precision relative to a plain fake-quant simulation (see
amct_pytorch/quantization/dtypes/hifp_impl.py). This script takes an already
-deployed packed checkpoint and, for every "<name>.weight" + "<name>.scale" pair,
writes back a plain BF16 "<name>.weight" tensor, dropping the ".scale" siblings
and the `quantization_config` block from config.json. Tensors with no ".scale"
sibling (embeddings, norms, lm_head, biases, ...) are copied through unchanged.

Numerically this reproduces the same weights as scripts/export_higptq_bf16.py,
which instead fake-quantizes ptq_result/'s calibrated float weights directly.
Use this script when you already have (or want) a real packed HiFloat4 deploy
artifact and only need a second, engine-compatible BF16 copy derived from it;
use export_higptq_bf16.py to go straight from ptq_result/ without a deploy step.

Usage:

    python -m amct_pytorch.deploy \\
        --model /path/to/Qwen3.5-9B \\
        --model_name qwen3_5 \\
        --granularity block \\
        --quant_target attn-linear mlp \\
        --quant_dtype hifp \\
        --bit_config amct_pytorch/configs/hifloat4.yaml \\
        --attn_linear_param_dir ptq_result/ptq_params/qwen3_5/attn-linear \\
        --moe_mlp_param_dir ptq_result/ptq_params/qwen3_5/mlp \\
        --output_dir ptq_result/deploy/qwen3_5

    python examples/algorithms/higptq/src/dequantize_higptq_deploy.py \\
        --input_dir ptq_result/deploy/qwen3_5 \\
        --output_dir ptq_result/bf16_export/qwen3_5
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME
from tqdm import tqdm

from amct_pytorch.quantization.dtypes.hifp_impl import hif4_unpack


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir",
        required=True,
        help="amct_pytorch.deploy output directory (packed HiFloat4 checkpoint).",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Where to write the plain BF16 checkpoint."
    )
    return parser.parse_args()


def _copy_support_files(input_dir, output_dir):
    src_dir, dst_dir = Path(input_dir), Path(output_dir)
    for src_path in src_dir.iterdir():
        if src_path.name.startswith(".") or src_path.suffix == ".safetensors":
            continue
        if src_path.name == SAFE_WEIGHTS_INDEX_NAME:
            continue
        dst_path = dst_dir / src_path.name
        if src_path.is_dir():
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)


def _strip_quantization_config(output_dir):
    config_path = Path(output_dir) / "config.json"
    if not config_path.exists():
        return
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if config.pop("quantization_config", None) is None:
        return
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def _load_weight_index(input_dir):
    index_path = Path(input_dir) / SAFE_WEIGHTS_INDEX_NAME
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _group_by_shard(weight_map):
    by_shard = {}
    for weight_name, file_name in weight_map.items():
        by_shard.setdefault(file_name, []).append(weight_name)
    return by_shard


def _dequantize_shard(input_dir, file_name, weight_names):
    """Read one shard and unpack every HiF4 "<name>.weight"/"<name>.scale" pair.

    Returns {weight_name: tensor}, already dropping consumed ".scale" tensors.
    """
    weight_name_set = set(weight_names)
    with safe_open(str(Path(input_dir) / file_name), framework="pt", device="cpu") as f:
        tensors = {name: f.get_tensor(name) for name in weight_names}

    output = {}
    for name in weight_names:
        if not name.endswith(".weight"):
            continue
        packed = tensors[name]
        scale_name = name[: -len(".weight")] + ".scale"
        if scale_name not in weight_name_set or packed.dtype != torch.uint8:
            # Not a HiGPTQ-packed weight (embeddings/norms/lm_head/...); keep as-is.
            output[name] = packed
            continue
        dequant = torch.from_numpy(hif4_unpack(tensors[scale_name], packed))
        output[name] = dequant.to(torch.bfloat16)

    for name in weight_names:
        if name in output:
            continue
        if name.endswith(".scale") and (name[: -len(".scale")] + ".weight") in weight_name_set:
            continue  # consumed above as the sibling of its packed weight
        output[name] = tensors[name]
    return output


def _write_safetensor_file(output_dir, file_name, tensors):
    if not tensors:
        return
    output_path = Path(output_dir) / file_name
    tmp_path = output_path.parent / f".{output_path.name}.tmp"
    save_file(tensors, str(tmp_path))
    os.replace(tmp_path, output_path)


def _refresh_weight_index(output_dir, original_index, updated_weight_map):
    metadata = dict(original_index.get("metadata", {}))
    total_size = 0
    for file_name in sorted(set(updated_weight_map.values())):
        total_size += os.path.getsize(os.path.join(output_dir, file_name))
    metadata["total_size"] = total_size
    output_index = {"metadata": metadata, "weight_map": updated_weight_map}
    index_path = os.path.join(output_dir, SAFE_WEIGHTS_INDEX_NAME)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(output_index, f, ensure_ascii=False, indent=2, sort_keys=True)
    return index_path


def dequantize(args):
    os.makedirs(args.output_dir, exist_ok=True)
    _copy_support_files(args.input_dir, args.output_dir)
    _strip_quantization_config(args.output_dir)

    original_index = _load_weight_index(args.input_dir)
    by_shard = _group_by_shard(dict(original_index.get("weight_map", {})))

    updated_weight_map = {}
    for file_name in tqdm(sorted(by_shard), desc="Dequantizing shards..."):
        tensors = _dequantize_shard(args.input_dir, file_name, by_shard[file_name])
        _write_safetensor_file(args.output_dir, file_name, tensors)
        for weight_name in tensors:
            updated_weight_map[weight_name] = file_name

    index_path = _refresh_weight_index(args.output_dir, original_index, updated_weight_map)
    logger.info("Dequantized BF16 checkpoint written to {}", args.output_dir)
    logger.info("Refreshed weight index at {}", index_path)


def main():
    args = _parse_args()
    dequantize(args)


if __name__ == "__main__":
    main()
