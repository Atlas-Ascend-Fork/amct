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
"""NVFP4 → BF16 → HiFloat4 whole-checkpoint converter (strategy + CLI).

Converts NVIDIA TRT-LLM NVFP4 checkpoints to HiFloat4 via BF16 intermediate.

Pipeline::

    NVFP4 packed weight  ──nvfp4_to_bf16──►  BF16  ──bf16_to_hif4──►  HiFloat4
         │                    (fp4_formats)              │
         │                                   ┌─ e6m2: official encode_to_hifloat4 (default)
         │                                   └─ e6m0: bf16_to_hif4_e6m0 (optional)

Usage::

    python -m amct_pytorch.experimental.fp4_convert.convert_nvfp4_to_hif4 \\
        --nvfp4-dir /path/to/nvfp4_ckpt \\
        --bf16-index /path/to/model.safetensors.index.json \\
        --out-dir /path/to/output \\
        --hif4-format e6m2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

# Allow importing amct_ops / sibling fp4_convert modules from the repo root.
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.realpath(os.path.join(_CUR_DIR, "../../../"))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from amct_pytorch.experimental.fp4_convert.fp4_formats import (  # noqa: E402
    bf16_to_hif4,
    hif4_to_bf16,
    nvfp4_to_bf16,
)
from amct_pytorch.experimental.fp4_convert.ckpt_io import (  # noqa: E402
    walk_checkpoint,
)
from amct_pytorch.experimental.fp4_convert.metrics import make_stats  # noqa: E402


# ══════════════════════════════════════════════════════════════════════
# NVFP4-specific key filtering
# ══════════════════════════════════════════════════════════════════════

_DROP_SUFFIXES = (
    ".input_scale",
    ".inv_input_scale",
    ".weight_scale",
    ".weight_scale_2",
    ".alpha",
    ".kv_scales",
    ".inv_kv_scales",
    ".pre_quant_scale",
)

_DROP_CONTAINS = (
    "input_scale",
    "inv_input_scale",
    "weight_scale_2",
    "weight_scale",
    "scalar_alpha",
    "alpha",
    "kv_scales",
    "inv_kv_scales",
)


def should_drop_key(key: str) -> bool:
    """Return True if *key* is an NVFP4 quantization intermediate to discard."""
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
    hif4_format: str = "e6m2",
    sf_vec_size: int = 16,
    dry_run: bool = False,
    verbose: bool = False,
) -> Optional[torch.Tensor]:
    """NVFP4 per-tensor strategy — the "foreman" for each weight.

    Decision logic:
      1. **Drop** — quantization intermediate keys → return ``None``.
      2. **Convert** — ``.weight`` keys that have a paired ``weight_scale``:
         ``nvfp4_to_bf16`` → ``bf16_to_hif4`` (e6m2 or e6m0).
      3. **Copy** — everything else (norms, embeddings, biases, etc.).

    Args:
        key: Tensor key name.
        tensor: The tensor value loaded from the source checkpoint.
        lookup: ``lookup(sibling_key) → (tensor, path)`` closure.
        hif4_format: ``"e6m2"`` (official, default) or ``"e6m0"``.
        sf_vec_size: NVFP4 block-size for ``weight_scale`` (default 16).
        dry_run: If True, print conversion decisions but return *tensor*
            unchanged (for inspection).
        verbose: Print per-key decisions.

    Returns:
        Converted tensor, copied tensor, or ``None`` (dropped).
    """
    # 1) Drop quantization intermediates
    if should_drop_key(key):
        if verbose:
            print(f"  [DROP] {key}")
        return None

    # 2) Try NVFP4 dequant for .weight keys
    if key.endswith(".weight"):
        prefix = key[: -len(".weight")]
        sf_key = prefix + ".weight_scale"
        sfg_key = prefix + ".weight_scale_2"

        w_sf, _ = lookup(sf_key)
        if w_sf is not None:
            w_sfg, _ = lookup(sfg_key)

            out_features = tensor.shape[0]
            in_features = tensor.shape[1] * 2  # packed: 2 nibbles per byte

            if verbose:
                print(
                    f"  [CONVERT] {key}  shape=({out_features},{in_features})"
                )

            if dry_run:
                return tensor

            # NVFP4 → BF16 (first half of the chain)
            w_bf16 = nvfp4_to_bf16(
                tensor, w_sf, w_sfg, out_features, in_features,
                sf_vec_size=sf_vec_size,
            )

            # BF16 → HiF4 (second half, shared with mxfp4)
            hif4_result, orig_len = bf16_to_hif4(
                w_bf16, hif4_format=hif4_format
            )

            if hif4_format == "e6m2":
                # For e6m2, the result is packed bytes → store as uint8 tensor
                return hif4_result
            else:
                # For e6m0, store the packed S1P2 bytes as the "weight"
                return hif4_result["s1p2_packed"]

        # No scale found → this is a normal weight (embedding, etc.)
        # Fall through to copy.

    # 3) Copy (non-quantized weights, norms, embeddings, biases)
    if verbose:
        print(f"  [COPY]  {key}")
    if tensor.dtype in (torch.float16, torch.float32, torch.bfloat16):
        return tensor.to(torch.bfloat16).contiguous()
    return tensor


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    """Build the NVFP4→HiF4 argument parser."""
    ap = argparse.ArgumentParser(
        description="NVFP4 → BF16 → HiFloat4 checkpoint converter"
    )
    ap.add_argument(
        "--nvfp4-dir", type=str, required=True,
        help="Directory containing NVFP4 *.safetensors shards.",
    )
    ap.add_argument(
        "--bf16-index", type=str, required=True,
        help="Path to BF16 model.safetensors.index.json (target sharding plan).",
    )
    ap.add_argument(
        "--out-dir", type=str, required=True,
        help="Output directory for HiFloat4 safetensors shards + index.json.",
    )
    ap.add_argument(
        "--hif4-format", type=str, default="e6m2",
        choices=["e6m2", "e6m0"],
        help="HiFloat4 target format: e6m2 (official E6M2, default) | "
             "e6m0 (simplified variant — NOT official HiF4).",
    )
    ap.add_argument(
        "--sf-vec-size", type=int, default=16,
        help="NVFP4 block-scale vector size (default: 16).",
    )
    ap.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu"],
        help="Device to run on (CPU only, avoids NPU float8 limitations).",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print conversion decisions without writing output.",
    )
    ap.add_argument(
        "--verbose", action="store_true",
        help="Print per-key conversion decisions.",
    )
    return ap


def main(argv: Optional[list] = None) -> None:
    """Entry point for NVFP4 → HiF4 checkpoint conversion."""
    args = build_parser().parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(
            f"[DRY RUN] NVFP4 → HiF4 ({args.hif4_format}) — "
            f"inspecting keys only"
        )

    # Build a closure that captures CLI args for the transform callback.
    def _transform(
        key: str,
        tensor: torch.Tensor,
        lookup: Any,
    ) -> Optional[torch.Tensor]:
        return transform_fn(
            key,
            tensor,
            lookup,
            hif4_format=args.hif4_format,
            sf_vec_size=args.sf_vec_size,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )

    if args.dry_run:
        # In dry-run mode we still walk but don't write.
        from amct_pytorch.experimental.fp4_convert.ckpt_io import (
            load_index,
            ShardCache,
            invert_weight_map,
        )
        import glob as glob_mod

        bf16_index = load_index(args.bf16_index)
        weight_map = bf16_index["weight_map"]
        by_file = invert_weight_map(weight_map)
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

    # Normal mode: full walk + write.
    stats = walk_checkpoint(
        src_dir=args.nvfp4_dir,
        index_path=args.bf16_index,
        out_dir=str(out_dir),
        transform_fn=_transform,
        verbose=not args.verbose,  # walk_checkpoint prints shard-level by default
    )

    if args.verbose:
        print(
            f"\nFinal: converted={stats['converted']}, "
            f"copied={stats['copied']}, dropped={stats['dropped']}, "
            f"missing={stats['missing']}"
        )


if __name__ == "__main__":
    main()
