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
"""HiF4 checkpoint unpack -- reconstruct a packed HiFloat4 checkpoint back to FP16/BF16 float.

This is the reverse of deploy's HiF4 *pack* (``QuantDequantHifp.export_deploy`` -> ``hif4_cast``).
It is **model-agnostic**: the model pipeline cannot load packed HiF4 weights, so unpack walks the
safetensors shards by their index (via the shared ``checkpoint_io`` layer) and reconstructs each
packed weight tensor with ``hif4_bit_reconstruct``. Exposed as a plain function -- the ``ptq`` CLI
routes HiF4 input here; there is no standalone command / arg parsing.

Pack layout consumed (deploy's canonical layout): the packed *value* bytes live under the weight
key (uint8) and the *scale* bytes under ``<weight>_scale`` (uint8). Scale keys are consumed, not
re-emitted. Non-packed tensors (norms/embeddings/biases) are copied through unchanged.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import torch

from amct_pytorch.quantization.dtypes.hifp_impl import hif4_bit_reconstruct
from ..checkpoint_io import resolve_index, invert_weight_map, write_index, copy_side_files, save_shard

# unpack --dtype name -> the torch float dtype the reconstructed weight is emitted as.
_DTYPE_BY_NAME = {"fp16": torch.float16, "float16": torch.float16,
                  "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}


def unpack_hif4_checkpoint(input_dir: str, output_dir: str,
                           out_dtype: Optional[str] = "fp16", quiet: bool = False) -> None:
    """Reconstruct a packed HiF4 checkpoint at *input_dir* to float weights at *output_dir*.

    Every ``<weight>`` that is uint8 and has a matching ``<weight>_scale`` is rebuilt via
    ``hif4_bit_reconstruct`` and cast to ``out_dtype`` (fp16/bf16); the scale key is dropped.
    All other tensors are copied verbatim. The output index and side files are written via the
    shared ``checkpoint_io`` helpers.
    """
    from safetensors.torch import load_file

    os.makedirs(output_dir, exist_ok=True)
    out_dtype_t = _DTYPE_BY_NAME.get(out_dtype or "fp16", torch.float16)

    index = resolve_index(input_dir)
    weight_map = index["weight_map"]
    s2k = invert_weight_map(weight_map)
    all_keys = set(weight_map.keys())
    # scale keys (the ``<weight>_scale`` half of a packed weight) are consumed, not emitted.
    scale_keys = {k + "_scale" for k in all_keys if (k + "_scale") in all_keys}

    _shard_cache: Dict[str, dict] = {}

    def _load(shard: str) -> dict:
        if shard not in _shard_cache:
            _shard_cache[shard] = load_file(os.path.join(input_dir, shard))
        return _shard_cache[shard]

    out_weight_map: Dict[str, str] = {}
    total_bytes = 0
    n_deq = n_copy = 0

    for shard, keys in s2k.items():
        tensors = _load(shard)
        out_tensors: Dict[str, torch.Tensor] = {}
        for name in keys:
            if name in scale_keys:
                continue
            t = tensors[name]
            scale_key = name + "_scale"
            if scale_key in all_keys and getattr(t, "dtype", None) == torch.uint8:
                scale = _load(weight_map[scale_key])[scale_key]  # scale may live in another shard
                recon = hif4_bit_reconstruct(scale, t)           # fp32 [rows, N]
                out_tensors[name] = torch.as_tensor(recon).to(out_dtype_t)
                n_deq += 1
            else:
                out_tensors[name] = t
                n_copy += 1
            out_weight_map[name] = shard
        for v in out_tensors.values():
            total_bytes += v.numel() * v.element_size()
        save_shard(output_dir, shard, out_tensors)
        if not quiet:
            print(f"[shard] {shard}: {len(out_tensors)} tensors written", flush=True)

    write_index(output_dir, out_weight_map, total_size=total_bytes)
    copy_side_files(input_dir, output_dir, torch_dtype=out_dtype or "fp16")

    print(f"[unpack] hif4 -> float | weights={n_deq} copied={n_copy} -> {output_dir}", flush=True)
