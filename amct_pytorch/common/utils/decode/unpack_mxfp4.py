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
"""MXFP4 → BF16 decoder + HiF4_E6M0 diagnostics (strategy + CLI).

``convert-dir`` is decode-only (the deploy/ptq path)::

    MXFP4 packed  ──mxfp4_to_bf16──►  BF16

HiF4 (pack via deploy, fake-quant via ptq) is applied downstream by ``hifloat4_quant``.
The ``selftest`` / ``convert-pair`` subcommands exercise the HiF4 e6m2/e6m0 round-trips as
standalone validation utilities (``quant_dequant_float`` / ``bf16_to_hif4_e6m0``); they
report only basic shape/error info (no quality-metrics module).

Subcommands: selftest / convert-pair / convert-dir.
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
    resolve_index, invert_weight_map, save_shard, write_index,
)
from safetensors.torch import load_file  # noqa: E402

_HIF4_QTYPE = QType("hifx4").dim(-1)


# ══════════════════════════════════════════════════════════════════════
# Data classes / small helpers
# ══════════════════════════════════════════════════════════════════════


@dataclass
class ConvertMeta:
    low_first: bool
    e8_bias: int
    e6_rule: str
    e6_bias: int
    hif4_format: str
    note: str


def _basic_err(ref: torch.Tensor, out: torch.Tensor) -> Dict[str, float]:
    """Basic reconstruction error (max / mean abs); replaces the removed metrics module."""
    diff = (ref.float() - out.float()).abs()
    return {"max_abs": diff.max().item(), "mean_abs": diff.mean().item()}


def _parse_spec(spec: str):
    """Parse a ``path[:key]`` spec string."""
    if ":" in spec:
        path, key = spec.split(":", 1)
        return path, key
    return spec, None


def _load_tensor_from_spec(spec: str, map_location: str = "cpu") -> torch.Tensor:
    """Load a single tensor from ``.pt`` / ``.pth`` / ``.npy`` / ``.safetensors:key``."""
    path, key = _parse_spec(spec)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    lower = path.lower()
    if lower.endswith(".pt") or lower.endswith(".pth"):
        obj = torch.load(path, map_location=map_location)
        if key is not None:
            if not isinstance(obj, dict):
                raise TypeError(f"{path} is not a dict, cannot get key {key}")
            obj = obj[key]
        if not isinstance(obj, torch.Tensor):
            raise TypeError(f"loaded object is not a tensor: {type(obj)}")
        return obj
    if lower.endswith(".npy"):
        import numpy as np  # local import; only needed for .npy specs
        if key is not None:
            raise ValueError(".npy spec should not include a key")
        return torch.from_numpy(np.load(path))
    if lower.endswith(".safetensors"):
        if key is None:
            raise ValueError("safetensors spec must be path.safetensors:key")
        return load_file(path, device=map_location)[key]
    raise ValueError(f"unsupported input file type: {path}")


def _save_convert_output(path: str, payload: Dict[str, Any]) -> None:
    """Save conversion output as a ``.pt`` file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save(payload, path)


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
    """Return True if *key* is an MXFP4 quantization intermediate."""
    for suf in _MXFP4_DROP_SUFFIXES:
        if key.endswith(suf):
            return True
    for s in _MXFP4_DROP_CONTAINS:
        if key.endswith("." + s) or key.endswith("_" + s):
            return True
    return False


# ══════════════════════════════════════════════════════════════════════
# Per-tensor transform callback (decode-only)
# ══════════════════════════════════════════════════════════════════════


def transform_fn(
    key: str,
    tensor: torch.Tensor,
    lookup: Any,
    low_first: bool = True,
    e8_bias: int = 127,
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
        w_sf, _ = lookup(prefix + ".weight_scale")
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
# Subcommand: selftest (HiF4 e6m2/e6m0 round-trip validation)
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
        print("[OK] MXFP4 -> BF16 -> HiF4(E6M2) fake-quant -> BF16")
        print(json.dumps(_basic_err(x_bf16, y_bf16), indent=2, ensure_ascii=False))
        print(f"  bf16 in:  {tuple(x_bf16.shape)}")
        print(f"  bf16 out: {tuple(y_bf16.shape)}")
    else:
        hif = bf16_to_hif4_e6m0(
            x_bf16, e8_code, e8_bias=args.e8_bias, e6_rule=args.e6_rule,
            e6_bias=args.e6_bias, round_mode=args.round_mode,
            pack_low_first=args.low_first,
        )
        recon = hif4_e6m0_to_fp32(hif["e6_exp"], hif["e2"], hif["e3"], hif["s1p2"])
        print("[OK] MXFP4 -> BF16 -> HiF4_E6M0 -> FP32 reconstruction")
        print(json.dumps(_basic_err(x_bf16, recon), indent=2, ensure_ascii=False))
        for k in ("e6_exp", "e6_code", "e2", "e3", "s1p2", "s1p2_packed"):
            print(f"  {k:14s} {tuple(hif[k].shape)} {hif[k].dtype}")


# ══════════════════════════════════════════════════════════════════════
# Subcommand: convert-pair (single tensor pair -> HiF4 e6m2/e6m0)
# ══════════════════════════════════════════════════════════════════════


def cmd_convert_pair(args: argparse.Namespace) -> None:
    packed = _load_tensor_from_spec(args.packed_spec, map_location=args.device)
    scale = _load_tensor_from_spec(args.scale_spec, map_location=args.device)
    x_bf16 = mxfp4_to_bf16(packed, scale, low_first=args.low_first, e8_bias=args.e8_bias)

    meta = ConvertMeta(
        low_first=args.low_first, e8_bias=args.e8_bias,
        e6_rule=args.e6_rule, e6_bias=args.e6_bias,
        hif4_format=args.hif4_format,
        note="e6m2: quant_dequant_float. e6m0: bf16_to_hif4_e6m0.",
    )

    if args.hif4_format == "e6m2":
        y_bf16 = quant_dequant_float(x_bf16, _HIF4_QTYPE, force_py=True, force_fp32=True)
        payload: Dict[str, Any] = {
            "bf16_in": x_bf16.cpu(), "bf16_out": y_bf16.cpu(),
            "stats": _basic_err(x_bf16, y_bf16), "meta": asdict(meta),
        }
    else:
        hif = bf16_to_hif4_e6m0(
            x_bf16, scale, e8_bias=args.e8_bias, e6_rule=args.e6_rule,
            e6_bias=args.e6_bias, round_mode=args.round_mode,
            pack_low_first=args.low_first,
        )
        recon = hif4_e6m0_to_fp32(hif["e6_exp"], hif["e2"], hif["e3"], hif["s1p2"])
        payload = {
            "bf16": x_bf16.cpu(),
            "hif4": {k: v.cpu() for k, v in hif.items() if k != "y_scaled"},
            "stats": _basic_err(x_bf16, recon), "meta": asdict(meta),
        }

    _save_convert_output(args.out, payload)
    print(f"[OK] saved: {args.out}")
    print(json.dumps(payload["stats"], indent=2, ensure_ascii=False))


# ══════════════════════════════════════════════════════════════════════
# Subcommand: convert-dir (whole-checkpoint decode)
# ══════════════════════════════════════════════════════════════════════


def cmd_convert_dir(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("[DRY RUN] MXFP4 → BF16 (decode only)")

    def _transform(key: str, tensor: torch.Tensor, lookup: Any) -> Optional[torch.Tensor]:
        return transform_fn(
            key, tensor, lookup, low_first=args.low_first, e8_bias=args.e8_bias,
            dry_run=args.dry_run, verbose=args.verbose,
        )

    # Small per-shard decode loop (output layout mirrors the source index; a weight's
    # weight_scale may live in another shard -> resolved via _lookup).
    index = resolve_index(args.mxfp4_dir, args.bf16_index)
    weight_map = index["weight_map"]
    by_file = invert_weight_map(weight_map)

    _cache: dict = {}

    def _load(shard: str) -> dict:
        if shard not in _cache:
            _cache[shard] = load_file(os.path.join(args.mxfp4_dir, shard))
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
                if args.verbose:
                    print(f"  [MISS] {key}")
                continue
            result = _transform(key, tensor, _lookup)
            if result is None:
                stats["dropped"] += 1
                continue
            stats["converted" if result is not tensor else "copied"] += 1
            if not args.dry_run:
                out_tensors[key] = result.contiguous()
        if not args.dry_run and out_tensors:
            save_shard(str(out_dir), shard, out_tensors)
            for key in out_tensors:
                out_weight_map[key] = shard

    if not args.dry_run:
        write_index(str(out_dir), out_weight_map, metadata=index.get("metadata", {}))

    print(
        f"Final: converted={stats['converted']}, copied={stats['copied']}, "
        f"dropped={stats['dropped']}, missing={stats['missing']}"
    )


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MXFP4 → BF16 decoder + HiF4_E6M0 diagnostics")
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

    sp = sub.add_parser("convert-dir", parents=[common], help="Whole-checkpoint decode.")
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
