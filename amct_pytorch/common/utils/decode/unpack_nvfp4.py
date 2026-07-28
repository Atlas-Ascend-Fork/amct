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
"""NVFP4 → BF16 whole-checkpoint decoder (strategy + CLI).

Pipeline::

    NVFP4 packed  ──nvfp4_to_bf16──►  BF16
                        (fp4_formats)

Decode only: dequantize NVFP4 → BF16 safetensors. HiF4 (pack via deploy, fake-quant via
ptq) is applied downstream by ``hifloat4_quant`` on the decoded BF16 -- not here.

Minimal CLI (same ``--model`` / ``--output_dir`` flags as ptq/deploy); the output plan is
derived from the NVFP4 checkpoint's own index (``resolve_index``), so nothing else is needed::

    python -m amct_pytorch.common.utils.decode.unpack_nvfp4 \\
        --model /path/to/nvfp4_ckpt \\
        --output_dir /path/to/output

In practice this is invoked automatically by ``ptq``/``deploy --quant_dtype hifp`` (auto-detect
+ decode), so the user only ever passes ptq/deploy's own ``--model`` / ``--output_dir``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Optional

import torch

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.realpath(os.path.join(_CUR_DIR, "../../../"))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from amct_pytorch.common.utils.decode.fp4_formats import (  # noqa: E402
    unpack_nibbles, decode_fp4_e2m1,
)
from amct_pytorch.common.utils.checkpoint_io import (  # noqa: E402
    resolve_index, invert_weight_map, save_shard, write_index,
)
from safetensors.torch import load_file  # noqa: E402


# ══════════════════════════════════════════════════════════════════════
# NVFP4 format math (E2M1 codec × E4M3 block scale × global scale → BF16)
# ══════════════════════════════════════════════════════════════════════


def decode_sf_to_fp32(sf: torch.Tensor) -> torch.Tensor:
    """Decode NVFP4 block-scale (E4M3 or uint8 view of E4M3) to float32."""
    if sf.dtype == torch.float8_e4m3fn:
        return sf.to(torch.float32)
    if sf.dtype == torch.uint8:
        return sf.view(torch.float8_e4m3fn).to(torch.float32)
    return sf.to(torch.float32)


def nvfp4_to_bf16(
    weight_fp4_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    out_features: int,
    in_features: int,
    sf_vec_size: int = 16,
) -> torch.Tensor:
    """Decode NVFP4 (TRT-LLM) packed weight to BF16.

    NVFP4 format: E2M1 packed weight × block-scale (E4M3, one per *sf_vec_size* elements) ×
    global per-channel scale. Uses the shared E2M1 codec from ``fp4_formats``.
    """
    device = weight_fp4_packed.device

    w_e2m1 = decode_fp4_e2m1(
        unpack_nibbles(weight_fp4_packed, low_first=True)
    ).to(torch.float32)

    def _pad_up(x: int, m: int) -> int:
        return ((x + m - 1) // m) * m

    nrows = _pad_up(out_features, 128)
    ncols = _pad_up(in_features // sf_vec_size, 4)

    sf = weight_scale
    if sf.dim() == 1:
        sf = sf.view(nrows, ncols)
    elif sf.dim() != 2:
        raise ValueError(f"Unexpected weight_scale dim={sf.dim()}, expected 1 or 2")

    sf = sf[:out_features, : in_features // sf_vec_size].contiguous()
    sf = decode_sf_to_fp32(sf).to(device=device)

    sf_full = sf.repeat_interleave(sf_vec_size, dim=1)
    global_scale = weight_scale_2.to(torch.float32).to(device)

    w_fp32 = w_e2m1[:out_features, :in_features] * sf_full * global_scale
    return w_fp32.to(torch.bfloat16).contiguous()


# ══════════════════════════════════════════════════════════════════════
# NVFP4 key filtering
# ══════════════════════════════════════════════════════════════════════

_DROP_SUFFIXES = (
    ".input_scale", ".inv_input_scale", ".weight_scale", ".weight_scale_2",
    ".alpha", ".kv_scales", ".inv_kv_scales", ".pre_quant_scale",
)

_DROP_CONTAINS = (
    "input_scale", "inv_input_scale", "weight_scale_2", "weight_scale",
    "scalar_alpha", "alpha", "kv_scales", "inv_kv_scales",
)


def should_drop_key(key: str) -> bool:
    """Return True if *key* is an NVFP4 quantization intermediate."""
    for suf in _DROP_SUFFIXES:
        if key.endswith(suf):
            return True
    for s in _DROP_CONTAINS:
        if key.endswith("." + s) or key.endswith("_" + s) or key.endswith(s):
            return True
    return False


# ══════════════════════════════════════════════════════════════════════
# Per-tensor transform callback
# ══════════════════════════════════════════════════════════════════════


def transform_fn(key: str, tensor: torch.Tensor, lookup: Any) -> Optional[torch.Tensor]:
    """NVFP4 per-tensor strategy.

    1. Drop quantization intermediates → None.
    2. Decode .weight keys with paired weight_scale: nvfp4_to_bf16 → BF16 (decode only;
       HiF4 pack/fake-quant applied downstream by deploy/ptq).
    3. Copy everything else as BF16.
    """
    if should_drop_key(key):
        return None

    if key.endswith(".weight"):
        prefix = key[: -len(".weight")]
        w_sf, _ = lookup(prefix + ".weight_scale")
        if w_sf is not None:
            w_sfg, _ = lookup(prefix + ".weight_scale_2")
            out_features = tensor.shape[0]
            in_features = tensor.shape[1] * 2
            # NVFP4 → BF16 (decode only). HiF4 is applied downstream by the caller
            # (deploy → pack, ptq → fake-quant) via hifloat4_quant, not here.
            return nvfp4_to_bf16(
                tensor, w_sf, w_sfg, out_features, in_features,
            ).contiguous()

    if tensor.dtype in (torch.float16, torch.float32, torch.bfloat16):
        return tensor.to(torch.bfloat16).contiguous()
    return tensor


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="NVFP4 -> BF16 checkpoint decoder")
    ap.add_argument("--model", type=str, required=True,
                    help="NVFP4 checkpoint directory with *.safetensors shards.")
    ap.add_argument("--output_dir", type=str, required=True,
                    help="Output directory for decoded BF16 shards + index.")
    return ap


def main(argv: Optional[list] = None) -> None:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Small per-shard decode loop (output layout mirrors the source index; a weight's
    # weight_scale/weight_scale_2 may live in another shard -> resolved via _lookup).
    index = resolve_index(args.model)
    weight_map = index["weight_map"]
    by_file = invert_weight_map(weight_map)

    _cache: dict = {}

    def _load(shard: str) -> dict:
        if shard not in _cache:
            _cache[shard] = load_file(os.path.join(args.model, shard))
        return _cache[shard]

    def _lookup(key: str):
        shard = weight_map.get(key)
        if shard is None:
            return None, None
        return _load(shard).get(key), shard

    stats = {"converted": 0, "copied": 0, "dropped": 0, "missing": 0}
    out_weight_map: dict = {}
    for shard, keys in by_file.items():
        src = _load(shard)
        out_tensors = {}
        for key in keys:
            tensor = src.get(key)
            if tensor is None:
                stats["missing"] += 1
                continue
            result = transform_fn(key, tensor, _lookup)
            if result is None:
                stats["dropped"] += 1
                continue
            stats["converted" if result is not tensor else "copied"] += 1
            out_tensors[key] = result.contiguous()
        if out_tensors:
            save_shard(str(out_dir), shard, out_tensors)
            for key in out_tensors:
                out_weight_map[key] = shard

    write_index(str(out_dir), out_weight_map, metadata=index.get("metadata", {}))
    print(
        f"[decode] nvfp4 -> bf16: converted={stats['converted']}, "
        f"copied={stats['copied']}, dropped={stats['dropped']}, missing={stats['missing']}"
    )


if __name__ == "__main__":
    main()
