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

The traversal/output plan is derived from the NVFP4 checkpoint's own index
(``resolve_index``), so no separate bf16 reference is required; ``--bf16-index`` is an
optional override.

Usage::

    python -m amct_pytorch.common.utils.fpx_decode.convert_nvfp4_to_hif4 \\
        --nvfp4-dir /path/to/nvfp4_ckpt \\
        --out-dir /path/to/output          # plan derived from the nvfp4 checkpoint itself
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

from amct_pytorch.common.utils.fpx_decode.fp4_formats import nvfp4_to_bf16  # noqa: E402
from amct_pytorch.common.utils.fpx_decode.ckpt_io import walk_checkpoint  # noqa: E402


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


def transform_fn(
    key: str,
    tensor: torch.Tensor,
    lookup: Any,
    sf_vec_size: int = 16,
    dry_run: bool = False,
    verbose: bool = False,
) -> Optional[torch.Tensor]:
    """NVFP4 per-tensor strategy.

    1. Drop quantization intermediates → None.
    2. Decode .weight keys with paired weight_scale: nvfp4_to_bf16 → BF16 (decode only;
       HiF4 pack/fake-quant applied downstream by deploy/ptq).
    3. Copy everything else as BF16.
    """
    if should_drop_key(key):
        if verbose:
            print(f"  [DROP] {key}")
        return None

    if key.endswith(".weight"):
        prefix = key[: -len(".weight")]
        sf_key = prefix + ".weight_scale"
        sfg_key = prefix + ".weight_scale_2"
        w_sf, _ = lookup(sf_key)
        if w_sf is not None:
            w_sfg, _ = lookup(sfg_key)
            out_features = tensor.shape[0]
            in_features = tensor.shape[1] * 2
            if verbose:
                print(f"  [CONVERT] {key}  shape=({out_features},{in_features})")
            if dry_run:
                return tensor
            # NVFP4 → BF16 (decode only). HiF4 is applied downstream by the caller
            # (deploy → pack, ptq → fake-quant) via hifloat4_quant, not here.
            w_bf16 = nvfp4_to_bf16(
                tensor, w_sf, w_sfg, out_features, in_features,
                sf_vec_size=sf_vec_size,
            )
            return w_bf16.contiguous()

    if verbose:
        print(f"  [COPY]  {key}")
    if tensor.dtype in (torch.float16, torch.float32, torch.bfloat16):
        return tensor.to(torch.bfloat16).contiguous()
    return tensor


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="NVFP4 → BF16 → HiFloat4 checkpoint converter"
    )
    ap.add_argument("--nvfp4-dir", type=str, required=True,
                    help="NVFP4 checkpoint directory with *.safetensors shards.")
    ap.add_argument("--bf16-index", type=str, default=None,
                    help="Optional target sharding plan (model.safetensors.index.json). "
                         "If omitted, the plan is derived from the NVFP4 checkpoint's own "
                         "index (self-sufficient; no bf16 reference required).")
    ap.add_argument("--out-dir", type=str, required=True,
                    help="Output directory for converted BF16 shards + index.")
    ap.add_argument("--sf-vec-size", type=int, default=16,
                    help="NVFP4 block-scale vector size (default 16).")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu"])
    ap.add_argument("--dry-run", action="store_true",
                    help="Print decisions without writing output.")
    ap.add_argument("--verbose", action="store_true",
                    help="Print per-key decisions.")
    return ap


def main(argv: Optional[list] = None) -> None:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("[DRY RUN] NVFP4 → BF16 (decode only)")

    def _transform(key: str, tensor: torch.Tensor, lookup: Any) -> Optional[torch.Tensor]:
        return transform_fn(
            key, tensor, lookup,
            sf_vec_size=args.sf_vec_size,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )

    if args.dry_run:
        from amct_pytorch.common.utils.fpx_decode.ckpt_io import (
            resolve_index, ShardCache, invert_weight_map,
        )
        import glob as glob_mod

        plan_index = resolve_index(args.nvfp4_dir, args.bf16_index)
        by_file = invert_weight_map(plan_index["weight_map"])
        source_shards = sorted(
            glob_mod.glob(os.path.join(args.nvfp4_dir, "*.safetensors"))
        )
        cache = ShardCache(source_shards)
        for shard_fname, keys in by_file.items():
            print(f"\n[Shard] {shard_fname}: {len(keys)} tensors")
            for k in keys:
                tensor, _ = cache.find_key(k)
                if tensor is None:
                    print(f"  [MISS] {k}")
                    continue
                def _lookup(lk: str):
                    return cache.find_key(lk)
                _transform(k, tensor, _lookup)
        return

    stats = walk_checkpoint(
        src_dir=args.nvfp4_dir,
        index_path=args.bf16_index,
        out_dir=str(out_dir),
        transform_fn=_transform,
        verbose=not args.verbose,
    )

    if args.verbose:
        print(
            f"\nFinal: converted={stats['converted']}, copied={stats['copied']}, "
            f"dropped={stats['dropped']}, missing={stats['missing']}"
        )


if __name__ == "__main__":
    main()
