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
"""MXFP4 → BF16 → HiFloat4 converter (strategy + CLI).

Converts OCP MXFP4 checkpoints to HiFloat4 via BF16 intermediate.

Pipeline::

    MXFP4 packed  ──mxfp4_to_bf16──►  BF16  ──bf16_to_hif4──►  HiFloat4
         │              (fp4_formats)              │
         │                             ┌─ e6m2: official encode_to_hifloat4 (default)
         │                             └─ e6m0: bf16_to_hif4_e6m0 (optional)

Subcommands::

    selftest        Random-data round-trip validation (no model needed).
    list-keys       List tensor keys in a safetensors file.
    convert-pair    Convert a single (packed_fp4, e8m0_scale) pair.
    convert-dir     Whole-checkpoint conversion (new in integrated version).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

# Allow importing amct_ops / sibling fp4_convert modules from the repo root.
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.realpath(os.path.join(_CUR_DIR, "../../../"))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from amct_pytorch.experimental.fp4_convert.fp4_formats import (  # noqa: E402
    _as_uint8,
    bf16_to_hif4,
    bf16_to_hif4_e6m0,
    compute_e6_exp_from_e8_pair,
    decode_fp4_e2m1,
    e2m1_lut,
    e8m0_to_unbiased,
    exp2_int,
    hif4_e6m0_to_fp32,
    hif4_to_bf16,
    mxfp4_to_bf16,
    pack_nibbles,
    unpack_nibbles,
)
from amct_pytorch.experimental.fp4_convert.ckpt_io import (  # noqa: E402
    load_index,
    load_tensor_from_spec,
    parse_spec,
    save_convert_output,
    ShardCache,
    walk_checkpoint,
)
from amct_pytorch.experimental.fp4_convert.metrics import (  # noqa: E402
    Hif4Stats,
    make_stats,
)


# ══════════════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════════════


@dataclass
class ConvertMeta:
    """Record of a conversion run's configuration (for reproducibility)."""

    low_first: bool
    e8_bias: int
    e6_rule: str
    e6_bias: int
    hif4_format: str
    note: str


# ══════════════════════════════════════════════════════════════════════
# MXFP4-specific key filtering
# ══════════════════════════════════════════════════════════════════════

_MXFP4_DROP_SUFFIXES = (
    ".weight_scale",
    ".input_scale",
    ".inv_input_scale",
    ".alpha",
    ".kv_scales",
    ".inv_kv_scales",
)

_MXFP4_DROP_CONTAINS = (
    "weight_scale",
    "input_scale",
    "inv_input_scale",
    "alpha",
    "kv_scales",
    "inv_kv_scales",
)


def should_drop_key(key: str) -> bool:
    """Return True if *key* is an MXFP4 quantization intermediate to discard."""
    for suf in _MXFP4_DROP_SUFFIXES:
        if key.endswith(suf):
            return True
    for s in _MXFP4_DROP_CONTAINS:
        if key.endswith("." + s) or key.endswith("_" + s):
            return True
    return False


# ══════════════════════════════════════════════════════════════════════
# Per-tensor transform callback (for whole-checkpoint conversion)
# ══════════════════════════════════════════════════════════════════════


def transform_fn(
    key: str,
    tensor: torch.Tensor,
    lookup: Any,
    hif4_format: str = "e6m2",
    low_first: bool = True,
    e8_bias: int = 127,
    e6_rule: str = "unbiased",
    e6_bias: int = 0,
    dry_run: bool = False,
    verbose: bool = False,
) -> Optional[torch.Tensor]:
    """MXFP4 per-tensor strategy — the "foreman" for each weight.

    Decision logic:
      1. **Drop** — quantization intermediate keys → return ``None``.
      2. **Convert** — ``.weight`` keys that have a paired ``weight_scale``:
         ``mxfp4_to_bf16`` → ``bf16_to_hif4`` (e6m2 or e6m0).
      3. **Copy** — everything else (norms, embeddings, biases).

    Args:
        key: Tensor key name from the weight map.
        tensor: The tensor value loaded from the source checkpoint.
        lookup: ``lookup(sibling_key) → (tensor, path)`` closure.
        hif4_format: ``"e6m2"`` (official, default) or ``"e6m0"``.
        low_first: Nibble order in packed bytes.
        e8_bias: E8M0 exponent bias (default 127).
        e6_rule: E6 exponent derivation rule (``"unbiased"`` / ``"raw"``).
        e6_bias: Offset applied to produce ``e6_code``.
        dry_run: Print decisions, return *tensor* unchanged.
        verbose: Print per-key decisions.

    Returns:
        Converted tensor, copied tensor, or ``None`` (dropped).
    """
    # 1) Drop quantization intermediates
    if should_drop_key(key):
        if verbose:
            print(f"  [DROP] {key}")
        return None

    # 2) Try MXFP4 dequant for .weight keys
    if key.endswith(".weight"):
        prefix = key[: -len(".weight")]
        sf_key = prefix + ".weight_scale"

        w_sf, _ = lookup(sf_key)
        if w_sf is not None:
            out_features = tensor.shape[0]
            in_features = tensor.shape[1] * 2  # packed: 2 nibbles per byte

            if verbose:
                print(
                    f"  [CONVERT] {key}  shape=({out_features},{in_features})"
                )

            if dry_run:
                return tensor

            # MXFP4 → BF16 (first half of the chain)
            w_bf16 = mxfp4_to_bf16(
                tensor, w_sf,
                low_first=low_first,
                e8_bias=e8_bias,
            )

            # BF16 → HiF4 (second half, shared with nvfp4)
            result, orig_len = bf16_to_hif4(
                w_bf16, hif4_format=hif4_format,
                e8m0_scale=w_sf if hif4_format == "e6m0" else None,
                e8_bias=e8_bias, e6_rule=e6_rule, e6_bias=e6_bias,
                pack_low_first=low_first,
            )

            if hif4_format == "e6m2":
                return result  # packed uint8 bytes
            else:
                return result["s1p2_packed"]  # type: ignore[index]

        # No scale → normal weight, fall through to copy.

    # 3) Copy
    if verbose:
        print(f"  [COPY]  {key}")
    if tensor.dtype in (torch.float16, torch.float32, torch.bfloat16):
        return tensor.to(torch.bfloat16).contiguous()
    return tensor


# ══════════════════════════════════════════════════════════════════════
# Subcommand: selftest
# ══════════════════════════════════════════════════════════════════════


def cmd_selftest(args: argparse.Namespace) -> None:
    """Random-data round-trip validation: MXFP4 → BF16 → HiF4 → BF16."""
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # ── known-value decode test ──
    known_nibbles = torch.arange(16, dtype=torch.uint8, device=device).reshape(1, 16)
    known_nibbles_32 = known_nibbles.repeat(1, 2)
    known_packed_32 = pack_nibbles(known_nibbles_32, low_first=True)
    known_scale_32 = torch.full((1, 1), 127, dtype=torch.uint8, device=device)
    decoded = mxfp4_to_bf16(known_packed_32, known_scale_32, low_first=True)
    expected = e2m1_lut(device).repeat(2).reshape(1, 32).to(torch.bfloat16)
    if not torch.equal(decoded, expected):
        raise AssertionError(f"known decode failed: {decoded} vs {expected}")
    print("[OK] known E2M1 decode test passed")

    # ── random round-trip test ──
    rows = args.rows
    values = args.groups * 64
    nibbles = torch.randint(0, 16, (rows, values), dtype=torch.uint8, device=device)
    packed = pack_nibbles(nibbles, low_first=args.low_first)
    e8_exp = torch.randint(
        args.exp_min, args.exp_max + 1,
        (rows, values // 32), dtype=torch.int16, device=device,
    )
    e8_code = torch.clamp(e8_exp + args.e8_bias, 0, 254).to(torch.uint8)

    x_bf16 = mxfp4_to_bf16(
        packed, e8_code, low_first=args.low_first, e8_bias=args.e8_bias,
    )

    if args.hif4_format == "e6m2":
        # Official E6M2 path
        packed_hif4, orig_len = bf16_to_hif4(x_bf16, hif4_format="e6m2")
        recon = hif4_to_bf16(
            packed_hif4, hif4_format="e6m2", orig_len=orig_len,
        )
        stats = make_stats(x_bf16, recon)
        print("[OK] random MXFP4 → BF16 → HiF4(E6M2) → BF16 reconstruction completed")
        print(json.dumps(asdict(stats), indent=2, ensure_ascii=False))
        print(f"  packed_fp4:   {tuple(packed.shape)} {packed.dtype}")
        print(f"  e8m0_scale:   {tuple(e8_code.shape)} {e8_code.dtype}")
        print(f"  bf16:         {tuple(x_bf16.shape)} {x_bf16.dtype}")
        print(f"  packed_hif4:  {tuple(packed_hif4.shape)} {packed_hif4.dtype}")
    else:
        # E6M0 path (returns full component dict)
        hif = bf16_to_hif4_e6m0(
            x_bf16, e8_code,
            e8_bias=args.e8_bias,
            e6_rule=args.e6_rule,
            e6_bias=args.e6_bias,
            round_mode=args.round_mode,
            pack_low_first=args.low_first,
        )
        recon = hif4_e6m0_to_fp32(
            hif["e6_exp"], hif["e2"], hif["e3"], hif["s1p2"],
        )
        stats = make_stats(x_bf16, recon, hif["qmag"], hif["y_scaled"])
        print("[OK] random MXFP4 → BF16 → HiF4_E6M0 → FP32 reconstruction completed")
        print(json.dumps(asdict(stats), indent=2, ensure_ascii=False))
        print("shapes:")
        print(f"  packed_fp4:   {tuple(packed.shape)} {packed.dtype}")
        print(f"  e8m0_scale:   {tuple(e8_code.shape)} {e8_code.dtype}")
        print(f"  bf16:         {tuple(x_bf16.shape)} {x_bf16.dtype}")
        print(f"  e6_exp:       {tuple(hif['e6_exp'].shape)} {hif['e6_exp'].dtype}")
        print(f"  e6_code:      {tuple(hif['e6_code'].shape)} {hif['e6_code'].dtype}")
        print(f"  e2:           {tuple(hif['e2'].shape)} {hif['e2'].dtype}")
        print(f"  e3:           {tuple(hif['e3'].shape)} {hif['e3'].dtype}")
        print(f"  s1p2:         {tuple(hif['s1p2'].shape)} {hif['s1p2'].dtype}")
        print(f"  s1p2_packed:  {tuple(hif['s1p2_packed'].shape)} {hif['s1p2_packed'].dtype}")


# ══════════════════════════════════════════════════════════════════════
# Subcommand: convert-pair
# ══════════════════════════════════════════════════════════════════════


def cmd_convert_pair(args: argparse.Namespace) -> None:
    """Convert a single (packed_fp4, e8m0_scale) tensor pair."""
    packed = load_tensor_from_spec(args.packed_spec, map_location=args.device)
    scale = load_tensor_from_spec(args.scale_spec, map_location=args.device)

    x_bf16 = mxfp4_to_bf16(
        packed, scale, low_first=args.low_first, e8_bias=args.e8_bias,
    )

    meta = ConvertMeta(
        low_first=args.low_first,
        e8_bias=args.e8_bias,
        e6_rule=args.e6_rule,
        e6_bias=args.e6_bias,
        hif4_format=args.hif4_format,
        note=(
            "e6_exp is the actual integer exponent used for reconstruction. "
            "e6_code = clamp(e6_exp + e6_bias, 0, 63). "
            "If your hardware expects raw E8M0-code-domain E6M0, use --e6-rule raw."
        ),
    )

    if args.hif4_format == "e6m2":
        packed_hif4, orig_len = bf16_to_hif4(x_bf16, hif4_format="e6m2")
        recon = hif4_to_bf16(packed_hif4, hif4_format="e6m2", orig_len=orig_len)
        stats = make_stats(x_bf16, recon)
        payload: Dict[str, Any] = {
            "bf16": x_bf16.cpu(),
            "hif4_packed": packed_hif4.cpu(),
            "orig_len": orig_len,
            "stats": asdict(stats),
            "meta": asdict(meta),
        }
    else:
        hif = bf16_to_hif4_e6m0(
            x_bf16, scale,
            e8_bias=args.e8_bias,
            e6_rule=args.e6_rule,
            e6_bias=args.e6_bias,
            round_mode=args.round_mode,
            pack_low_first=args.low_first,
        )
        recon = hif4_e6m0_to_fp32(
            hif["e6_exp"], hif["e2"], hif["e3"], hif["s1p2"],
        )
        stats = make_stats(x_bf16, recon, hif["qmag"], hif["y_scaled"])
        payload = {
            "bf16": x_bf16.cpu(),
            "hif4": {k: v.cpu() for k, v in hif.items() if k not in ("y_scaled",)},
            "stats": asdict(stats),
            "meta": asdict(meta),
        }

    save_convert_output(args.out, payload)
    print(f"[OK] saved: {args.out}")
    print(json.dumps(payload["stats"], indent=2, ensure_ascii=False))


# ══════════════════════════════════════════════════════════════════════
# Subcommand: list-keys
# ══════════════════════════════════════════════════════════════════════


def cmd_list_keys(args: argparse.Namespace) -> None:
    """List tensor keys in a safetensors file."""
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError(
            "pip install safetensors is required for list-keys"
        ) from exc

    filt = args.filter
    with safe_open(args.file, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        if filt:
            keys = [k for k in keys if filt in k]
        for k in keys:
            try:
                t = f.get_tensor(k)
                print(f"{k}\tshape={tuple(t.shape)}\tdtype={t.dtype}")
            except Exception:
                print(k)


# ══════════════════════════════════════════════════════════════════════
# Subcommand: convert-dir (whole-checkpoint conversion)
# ══════════════════════════════════════════════════════════════════════


def cmd_convert_dir(args: argparse.Namespace) -> None:
    """Whole-checkpoint MXFP4 → BF16 → HiF4 conversion."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(
            f"[DRY RUN] MXFP4 → HiF4 ({args.hif4_format}) — "
            f"inspecting keys only"
        )

    def _transform(
        key: str,
        tensor: torch.Tensor,
        lookup: Any,
    ) -> Optional[torch.Tensor]:
        return transform_fn(
            key, tensor, lookup,
            hif4_format=args.hif4_format,
            low_first=args.low_first,
            e8_bias=args.e8_bias,
            e6_rule=args.e6_rule,
            e6_bias=args.e6_bias,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )

    if args.dry_run:
        import glob as glob_mod
        bf16_index = load_index(args.bf16_index)
        weight_map = bf16_index["weight_map"]
        from amct_pytorch.experimental.fp4_convert.ckpt_io import invert_weight_map
        by_file = invert_weight_map(weight_map)
        source_shards = sorted(
            glob_mod.glob(os.path.join(args.mxfp4_dir, "*.safetensors"))
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
        src_dir=args.mxfp4_dir,
        index_path=args.bf16_index,
        out_dir=str(out_dir),
        transform_fn=_transform,
        verbose=not args.verbose,
    )

    if args.verbose:
        print(
            f"\nFinal: converted={stats['converted']}, "
            f"copied={stats['copied']}, dropped={stats['dropped']}, "
            f"missing={stats['missing']}"
        )


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    """Build the MXFP4→HiF4 argument parser with subcommands."""
    p = argparse.ArgumentParser(
        description="MXFP4 → BF16 → HiFloat4 converter"
    )
    sub = p.add_subparsers(dest="cmd")

    # ── shared flags (used by selftest, convert-pair, convert-dir) ──
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--low-first", action="store_true",
        help="Unpack/pack low nibble first; default if neither flag is set.",
    )
    common.add_argument(
        "--high-first", action="store_true",
        help="Unpack/pack high nibble first.",
    )
    common.add_argument("--e8-bias", type=int, default=127)
    common.add_argument(
        "--e6-rule", choices=["unbiased", "raw"], default="unbiased",
    )
    common.add_argument(
        "--e6-bias", type=int, default=0,
        help="Offset: e6_code = clamp(e6_exp + e6_bias, 0, 63).",
    )
    common.add_argument(
        "--round-mode", choices=["nearest_even", "half_up"],
        default="nearest_even",
    )
    common.add_argument("--device", default="cpu")
    common.add_argument(
        "--hif4-format", type=str, default="e6m2",
        choices=["e6m2", "e6m0"],
        help="HiFloat4 target format: e6m2 (official E6M2, default) | "
             "e6m0 (simplified variant — NOT official HiF4).",
    )

    # ── selftest ──
    sp = sub.add_parser(
        "selftest", parents=[common],
        help="Run random and known-value round-trip tests.",
    )
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--rows", type=int, default=4)
    sp.add_argument(
        "--groups", type=int, default=8,
        help="Number of 64-value HiF4 groups per row.",
    )
    sp.add_argument("--exp-min", type=int, default=-4)
    sp.add_argument("--exp-max", type=int, default=5)
    sp.set_defaults(func=cmd_selftest)

    # ── convert-pair ──
    sp = sub.add_parser(
        "convert-pair", parents=[common],
        help="Convert one (packed_fp4, e8m0_scale) tensor pair.",
    )
    sp.add_argument(
        "--packed-spec", required=True,
        help="path[:key], supports .pt/.pth/.npy/.safetensors:key",
    )
    sp.add_argument(
        "--scale-spec", required=True,
        help="path[:key], supports .pt/.pth/.npy/.safetensors:key",
    )
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_convert_pair)

    # ── list-keys ──
    sp = sub.add_parser(
        "list-keys",
        help="List tensor keys in a safetensors file.",
    )
    sp.add_argument("--file", required=True)
    sp.add_argument("--filter", default=None)
    sp.set_defaults(func=cmd_list_keys)

    # ── convert-dir (whole checkpoint) ──
    sp = sub.add_parser(
        "convert-dir", parents=[common],
        help="Convert an entire MXFP4 checkpoint directory to HiF4.",
    )
    sp.add_argument(
        "--mxfp4-dir", type=str, required=True,
        help="Directory containing MXFP4 *.safetensors shards.",
    )
    sp.add_argument(
        "--bf16-index", type=str, required=True,
        help="Path to BF16 model.safetensors.index.json (target sharding plan).",
    )
    sp.add_argument(
        "--out-dir", type=str, required=True,
        help="Output directory for HiF4 safetensors shards + index.json.",
    )
    sp.add_argument(
        "--dry-run", action="store_true",
        help="Print conversion decisions without writing output.",
    )
    sp.add_argument(
        "--verbose", action="store_true",
        help="Print per-key conversion decisions.",
    )
    sp.set_defaults(func=cmd_convert_dir)

    return p


def main(argv: Optional[list] = None) -> None:
    """Entry point for MXFP4 → HiF4 conversion CLI."""
    args = build_parser().parse_args(argv)
    if not hasattr(args, "func"):
        build_parser().print_help()
        return

    # Resolve nibble order default: low-first if --high-first is not given.
    if hasattr(args, "low_first"):
        if args.low_first and args.high_first:
            raise ValueError(
                "choose only one of --low-first and --high-first"
            )
        args.low_first = True if not args.high_first else False

    args.func(args)


if __name__ == "__main__":
    main()
