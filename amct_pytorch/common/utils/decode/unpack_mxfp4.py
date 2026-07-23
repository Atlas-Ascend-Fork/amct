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
"""MXFP4 → BF16 decoder + HiF4 diagnostics (strategy + CLI).

``convert-dir`` is decode-only (the deploy/ptq path)::

    MXFP4 packed  ──mxfp4_to_bf16──►  BF16
                       (fp4_formats)

HiF4 (pack via deploy, fake-quant via ptq) is applied downstream by ``hifloat4_quant``.
The ``selftest`` / ``convert-pair`` subcommands keep the HiF4 e6m2/e6m0 round-trips as
standalone validation utilities (they still exercise ``quant_dequant_float`` /
``bf16_to_hif4_e6m0``).

Subcommands: selftest / list-keys / convert-pair / convert-dir.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.realpath(os.path.join(_CUR_DIR, "../../../"))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from amct_pytorch.common.utils.hifloat4_quant import QType, quant_dequant_float  # noqa: E402
from amct_pytorch.common.utils.decode.fp4_formats import (  # noqa: E402
    bf16_to_hif4_e6m0,
    e2m1_lut,
    hif4_e6m0_to_fp32,
    mxfp4_to_bf16,
    pack_nibbles,
)
from amct_pytorch.common.utils.checkpoint_io import (  # noqa: E402
    load_index,
    resolve_index,
    load_tensor_from_spec,
    save_convert_output,
    ShardCache,
    walk_checkpoint,
)
from amct_pytorch.common.utils.decode.metrics import make_stats  # noqa: E402

_HIF4_QTYPE = QType("hifx4").dim(-1)


# ══════════════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════════════


@dataclass
class ConvertMeta:
    low_first: bool
    e8_bias: int
    e6_rule: str
    e6_bias: int
    hif4_format: str
    note: str


# ══════════════════════════════════════════════════════════════════════
# MXFP4 key filtering
# ══════════════════════════════════════════════════════════════════════

_MXFP4_DROP_SUFFIXES = (
    ".weight_scale", ".input_scale", ".inv_input_scale",
    ".alpha", ".kv_scales", ".inv_kv_scales",
)

_MXFP4_DROP_CONTAINS = (
    "weight_scale", "input_scale", "inv_input_scale",
    "alpha", "kv_scales", "inv_kv_scales",
)


def should_drop_key(key: str) -> bool:
    for suf in _MXFP4_DROP_SUFFIXES:
        if key.endswith(suf):
            return True
    for s in _MXFP4_DROP_CONTAINS:
        if key.endswith("." + s) or key.endswith("_" + s):
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
    low_first: bool = True,
    e8_bias: int = 127,
    e6_rule: str = "unbiased",
    e6_bias: int = 0,
    dry_run: bool = False,
    verbose: bool = False,
) -> Optional[torch.Tensor]:
    """MXFP4 per-tensor strategy.

    1. Drop quantization intermediates → None.
    2. Decode .weight keys with paired weight_scale: mxfp4_to_bf16 → BF16 (decode only;
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
        w_sf, _ = lookup(sf_key)
        if w_sf is not None:
            out_features = tensor.shape[0]
            in_features = tensor.shape[1] * 2
            if verbose:
                print(f"  [CONVERT] {key}  shape=({out_features},{in_features})")
            if dry_run:
                return tensor
            # MXFP4 → BF16 (decode only). HiF4 is applied downstream by the caller
            # (deploy → pack, ptq → fake-quant) via hifloat4_quant, not here.
            return mxfp4_to_bf16(
                tensor, w_sf, low_first=low_first, e8_bias=e8_bias,
            ).contiguous()

    if verbose:
        print(f"  [COPY]  {key}")
    if tensor.dtype in (torch.float16, torch.float32, torch.bfloat16):
        return tensor.to(torch.bfloat16).contiguous()
    return tensor


# ══════════════════════════════════════════════════════════════════════
# Subcommand: selftest
# ══════════════════════════════════════════════════════════════════════


def cmd_selftest(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # known-value test
    known_nibbles = torch.arange(16, dtype=torch.uint8, device=device).reshape(1, 16)
    known_nibbles_32 = known_nibbles.repeat(1, 2)
    known_packed_32 = pack_nibbles(known_nibbles_32, low_first=True)
    known_scale_32 = torch.full((1, 1), 127, dtype=torch.uint8, device=device)
    decoded = mxfp4_to_bf16(known_packed_32, known_scale_32, low_first=True)
    expected = e2m1_lut(device).repeat(2).reshape(1, 32).to(torch.bfloat16)
    if not torch.equal(decoded, expected):
        raise AssertionError(f"known decode failed: {decoded} vs {expected}")
    print("[OK] known E2M1 decode test passed")

    # random round-trip
    rows, values = args.rows, args.groups * 64
    nibbles = torch.randint(0, 16, (rows, values), dtype=torch.uint8, device=device)
    packed = pack_nibbles(nibbles, low_first=args.low_first)
    e8_exp = torch.randint(
        args.exp_min, args.exp_max + 1,
        (rows, values // 32), dtype=torch.int16, device=device,
    )
    e8_code = torch.clamp(e8_exp + args.e8_bias, 0, 254).to(torch.uint8)

    x_bf16 = mxfp4_to_bf16(packed, e8_code, low_first=args.low_first, e8_bias=args.e8_bias)

    if args.hif4_format == "e6m2":
        # quant_dequant_float round-trip
        y_bf16 = quant_dequant_float(x_bf16, _HIF4_QTYPE, force_py=True, force_fp32=True)
        stats = make_stats(x_bf16, y_bf16)
        print("[OK] MXFP4 -> BF16 -> HiF4(E6M2) fake-quant -> BF16")
        print(json.dumps(asdict(stats), indent=2, ensure_ascii=False))
        print(f"  bf16 in:  {tuple(x_bf16.shape)}")
        print(f"  bf16 out: {tuple(y_bf16.shape)}")
    else:
        hif = bf16_to_hif4_e6m0(
            x_bf16, e8_code, e8_bias=args.e8_bias, e6_rule=args.e6_rule,
            e6_bias=args.e6_bias, round_mode=args.round_mode,
            pack_low_first=args.low_first,
        )
        recon = hif4_e6m0_to_fp32(hif["e6_exp"], hif["e2"], hif["e3"], hif["s1p2"])
        stats = make_stats(x_bf16, recon, hif["qmag"], hif["y_scaled"])
        print("[OK] MXFP4 -> BF16 -> HiF4_E6M0 -> FP32 reconstruction")
        print(json.dumps(asdict(stats), indent=2, ensure_ascii=False))
        for k in ("e6_exp", "e6_code", "e2", "e3", "s1p2", "s1p2_packed"):
            print(f"  {k:14s} {tuple(hif[k].shape)} {hif[k].dtype}")


# ══════════════════════════════════════════════════════════════════════
# Subcommand: convert-pair
# ══════════════════════════════════════════════════════════════════════


def cmd_convert_pair(args: argparse.Namespace) -> None:
    packed = load_tensor_from_spec(args.packed_spec, map_location=args.device)
    scale = load_tensor_from_spec(args.scale_spec, map_location=args.device)
    x_bf16 = mxfp4_to_bf16(packed, scale, low_first=args.low_first, e8_bias=args.e8_bias)

    meta = ConvertMeta(
        low_first=args.low_first, e8_bias=args.e8_bias,
        e6_rule=args.e6_rule, e6_bias=args.e6_bias,
        hif4_format=args.hif4_format,
        note="e6m2: quant_dequant_float. e6m0: bf16_to_hif4_e6m0.",
    )

    if args.hif4_format == "e6m2":
        y_bf16 = quant_dequant_float(x_bf16, _HIF4_QTYPE, force_py=True, force_fp32=True)
        stats = make_stats(x_bf16, y_bf16)
        payload: Dict[str, Any] = {
            "bf16_in": x_bf16.cpu(), "bf16_out": y_bf16.cpu(),
            "stats": asdict(stats), "meta": asdict(meta),
        }
    else:
        hif = bf16_to_hif4_e6m0(
            x_bf16, scale, e8_bias=args.e8_bias, e6_rule=args.e6_rule,
            e6_bias=args.e6_bias, round_mode=args.round_mode,
            pack_low_first=args.low_first,
        )
        recon = hif4_e6m0_to_fp32(hif["e6_exp"], hif["e2"], hif["e3"], hif["s1p2"])
        stats = make_stats(x_bf16, recon, hif["qmag"], hif["y_scaled"])
        payload = {
            "bf16": x_bf16.cpu(),
            "hif4": {k: v.cpu() for k, v in hif.items() if k != "y_scaled"},
            "stats": asdict(stats), "meta": asdict(meta),
        }

    save_convert_output(args.out, payload)
    print(f"[OK] saved: {args.out}")
    print(json.dumps(payload["stats"], indent=2, ensure_ascii=False))


# ══════════════════════════════════════════════════════════════════════
# Subcommand: list-keys
# ══════════════════════════════════════════════════════════════════════


def cmd_list_keys(args: argparse.Namespace) -> None:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError("pip install safetensors is required") from exc
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
# Subcommand: convert-dir
# ══════════════════════════════════════════════════════════════════════


def cmd_convert_dir(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("[DRY RUN] MXFP4 → BF16 (decode only)")

    def _transform(key: str, tensor: torch.Tensor, lookup: Any) -> Optional[torch.Tensor]:
        return transform_fn(
            key, tensor, lookup, hif4_format=args.hif4_format,
            low_first=args.low_first, e8_bias=args.e8_bias,
            e6_rule=args.e6_rule, e6_bias=args.e6_bias,
            dry_run=args.dry_run, verbose=args.verbose,
        )

    if args.dry_run:
        import glob as glob_mod
        plan_index = resolve_index(args.mxfp4_dir, args.bf16_index)
        from amct_pytorch.common.utils.checkpoint_io import invert_weight_map
        by_file = invert_weight_map(plan_index["weight_map"])
        source_shards = sorted(glob_mod.glob(os.path.join(args.mxfp4_dir, "*.safetensors")))
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
        src_dir=args.mxfp4_dir, index_path=args.bf16_index,
        out_dir=str(out_dir), transform_fn=_transform,
        verbose=not args.verbose,
    )
    if args.verbose:
        print(
            f"\nFinal: converted={stats['converted']}, copied={stats['copied']}, "
            f"dropped={stats['dropped']}, missing={stats['missing']}"
        )


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MXFP4 → BF16 → HiFloat4 converter")
    sub = p.add_subparsers(dest="cmd")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--low-first", action="store_true")
    common.add_argument("--high-first", action="store_true")
    common.add_argument("--e8-bias", type=int, default=127)
    common.add_argument("--e6-rule", choices=["unbiased", "raw"], default="unbiased")
    common.add_argument("--e6-bias", type=int, default=0)
    common.add_argument("--round-mode", choices=["nearest_even", "half_up"], default="nearest_even")
    common.add_argument("--device", default="cpu")
    common.add_argument("--hif4-format", type=str, default="e6m2", choices=["e6m2", "e6m0"],
                        help="e6m2: quant_dequant_float (default). e6m0: bf16_to_hif4_e6m0.")

    sp = sub.add_parser("selftest", parents=[common], help="Random round-trip validation.")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--rows", type=int, default=4)
    sp.add_argument("--groups", type=int, default=8)
    sp.add_argument("--exp-min", type=int, default=-4)
    sp.add_argument("--exp-max", type=int, default=5)
    sp.set_defaults(func=cmd_selftest)

    sp = sub.add_parser("convert-pair", parents=[common], help="Convert single tensor pair.")
    sp.add_argument("--packed-spec", required=True)
    sp.add_argument("--scale-spec", required=True)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_convert_pair)

    sp = sub.add_parser("list-keys", help="List safetensors keys.")
    sp.add_argument("--file", required=True)
    sp.add_argument("--filter", default=None)
    sp.set_defaults(func=cmd_list_keys)

    sp = sub.add_parser("convert-dir", parents=[common], help="Whole-checkpoint conversion.")
    sp.add_argument("--mxfp4-dir", type=str, required=True)
    sp.add_argument("--bf16-index", type=str, default=None,
                    help="Optional target sharding plan; if omitted, derived from the "
                         "MXFP4 checkpoint's own index (self-sufficient).")
    sp.add_argument("--out-dir", type=str, required=True)
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--verbose", action="store_true")
    sp.set_defaults(func=cmd_convert_dir)

    return p


def main(argv: Optional[list] = None) -> None:
    args = build_parser().parse_args(argv)
    if not hasattr(args, "func"):
        build_parser().print_help()
        return
    if hasattr(args, "low_first"):
        if args.low_first and args.high_first:
            raise ValueError("choose only one of --low-first and --high-first")
        args.low_first = True if not args.high_first else False
    args.func(args)


if __name__ == "__main__":
    main()
