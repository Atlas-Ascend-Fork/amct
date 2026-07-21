#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
"""MXFP8 → BF16 → HiFloat4 checkpoint converter.

Pipeline::

    MXFP8 packed  ──dequant_mxfp8──►  BF16  ──quant_dequant_float──►  BF16
                         (fp4_formats)              (hifloat4_quant)

Conversion is self-sufficient — the output plan is derived from the MXFP8 checkpoint's own
index, so no bf16 reference is needed.

Typical usage::

    python -m amct_pytorch.cli.llm.convert_to_hif4 --source-format mxfp8 \\
        --quant_model_dir /path/to/MXFP8 \\
        --output_dir /path/to/output
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import shutil
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from safetensors.torch import load_file, save_file

from amct_pytorch.common.utils.hifloat4_quant import QType, quant_dequant_float

_HIF4_QTYPE = QType("hifx4").dim(-1)


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def get_fp8_dtypes() -> set[torch.dtype]:
    names = [
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
    ]
    out = set()
    for name in names:
        if hasattr(torch, name):
            out.add(getattr(torch, name))
    return out


FP8_DTYPES = get_fp8_dtypes()


def is_fp8_tensor(t: torch.Tensor) -> bool:
    return t.dtype in FP8_DTYPES


def prod(xs: Sequence[int]) -> int:
    p = 1
    for x in xs:
        p *= int(x)
    return p


def human_bytes(n: int | float | None) -> str:
    if n is None:
        return "unknown"
    n = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024.0:
            return f"{n:.2f}{unit}"
        n /= 1024.0
    return f"{n:.2f}PB"


# -----------------------------------------------------------------------------
# HF index helpers
# -----------------------------------------------------------------------------

def find_model_index_or_single_file(model_dir: Path) -> tuple[str, Path]:
    index_candidates = [
        model_dir / "model.safetensors.index.json",
        model_dir / "pytorch_model.bin.index.json",
    ]
    for p in index_candidates:
        if p.exists():
            return "index_json", p

    single_candidates = [
        model_dir / "model.safetensors",
    ]
    for p in single_candidates:
        if p.exists():
            return "single_safetensors", p

    raise FileNotFoundError(
        f"Cannot find model.safetensors.index.json or model.safetensors in {model_dir}"
    )


def load_index(model_dir: Path) -> dict:
    kind, path = find_model_index_or_single_file(model_dir)

    if kind == "index_json":
        obj = load_json(path)
        if "weight_map" not in obj:
            raise ValueError(f"Invalid index file without weight_map: {path}")
        return obj

    tensors = load_file(str(path))
    return {
        "metadata": {"total_size": path.stat().st_size},
        "weight_map": {k: path.name for k in tensors.keys()},
    }


def build_shard_to_keys(index_obj: dict) -> Dict[str, List[str]]:
    shard_to_keys: Dict[str, List[str]] = defaultdict(list)
    for key, shard_name in index_obj["weight_map"].items():
        shard_to_keys[shard_name].append(key)
    return dict(shard_to_keys)


class ShardCache:
    """Small LRU cache for safetensors shards."""

    def __init__(self, model_dir: Path, max_cached: int = 3):
        self.model_dir = model_dir
        self.max_cached = max(1, int(max_cached))
        self.cache: OrderedDict[str, Dict[str, torch.Tensor]] = OrderedDict()

    def get_shard(self, shard_name: str) -> Dict[str, torch.Tensor]:
        if shard_name in self.cache:
            self.cache.move_to_end(shard_name)
            return self.cache[shard_name]

        path = self.model_dir / shard_name
        if not path.exists():
            raise FileNotFoundError(f"Shard does not exist: {path}")

        tensors = load_file(str(path))
        self.cache[shard_name] = tensors
        self.cache.move_to_end(shard_name)

        while len(self.cache) > self.max_cached:
            self.cache.popitem(last=False)
            gc.collect()

        return tensors

    def get_tensor_by_name(self, index_obj: dict, tensor_name: str) -> torch.Tensor:
        shard_name = index_obj["weight_map"][tensor_name]
        return self.get_shard(shard_name)[tensor_name]

    def clear(self) -> None:
        self.cache.clear()
        gc.collect()


# -----------------------------------------------------------------------------
# Quant config parsing
# -----------------------------------------------------------------------------

def maybe_load_json(model_dir: Path, filename: str) -> Optional[dict]:
    p = model_dir / filename
    if p.exists():
        return load_json(p)
    return None


def infer_quant_scope(model_dir: Path) -> dict:
    scope = {
        "source": None,
        "quant_method": None,
        "format": None,
        "quant_algo": None,
        "group_size": 32,
        "targets": [],
        "ignore": [],
        "config_groups": {},
        "raw": None,
    }

    cfg = maybe_load_json(model_dir, "config.json")
    if cfg and isinstance(cfg.get("quantization_config"), dict):
        qcfg = cfg["quantization_config"]
        scope["source"] = "config.json:quantization_config"
        scope["quant_method"] = qcfg.get("quant_method")
        scope["format"] = qcfg.get("format")
        scope["ignore"] = qcfg.get("ignore", []) or []
        scope["config_groups"] = qcfg.get("config_groups", {}) or {}
        scope["raw"] = qcfg

        fmt = str(qcfg.get("format", "")).lower()
        if "mxfp8" in fmt:
            scope["quant_algo"] = "MXFP8"
        elif "mxfp4" in fmt:
            scope["quant_algo"] = "MXFP4"

        for _, group in scope["config_groups"].items():
            if not isinstance(group, dict):
                continue
            weights = group.get("weights", {}) or {}
            if isinstance(weights, dict) and weights.get("group_size") is not None:
                scope["group_size"] = int(weights["group_size"])
            targets = group.get("targets", []) or []
            if targets:
                scope["targets"].extend(targets)
            break
        return scope

    hqc = maybe_load_json(model_dir, "hf_quant_config.json")
    if hqc:
        q = hqc.get("quantization", {}) or {}
        scope["source"] = "hf_quant_config.json"
        scope["quant_algo"] = str(q.get("quant_algo", "")).upper() or None
        scope["group_size"] = int(q.get("group_size") or 32)
        scope["ignore"] = q.get("exclude_modules", []) or []
        scope["raw"] = hqc
        return scope

    return scope


# -----------------------------------------------------------------------------
# MXFP8 dequant
# -----------------------------------------------------------------------------

def scale_to_mxfp8_factor(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype != torch.uint8:
        if scale.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64}:
            raise TypeError(
                f"MXFP8 scale is expected to be uint8/integer E8M0, got dtype={scale.dtype}"
            )

    exponent = scale.to(torch.float32) - 127.0
    return torch.exp2(exponent)


def normalize_scale_shape(
    qweight: torch.Tensor,
    scale: torch.Tensor,
    group_size: int,
    tensor_name: str,
    allow_reshape_scale: bool = True,
) -> torch.Tensor:
    if qweight.ndim < 1:
        raise ValueError(f"qweight must have at least 1 dim: {tensor_name}")

    last = int(qweight.shape[-1])
    if last % group_size != 0:
        raise ValueError(
            f"Last dim of {tensor_name} is {last}, not divisible by group_size={group_size}"
        )

    expected_shape = tuple(qweight.shape[:-1]) + (last // group_size,)
    if tuple(scale.shape) == expected_shape:
        return scale

    if allow_reshape_scale and scale.numel() == prod(expected_shape):
        return scale.reshape(expected_shape)

    raise ValueError(
        f"Scale shape mismatch for {tensor_name}: "
        f"qweight_shape={tuple(qweight.shape)}, scale_shape={tuple(scale.shape)}, "
        f"expected_scale_shape={expected_shape}, group_size={group_size}"
    )


def dequant_mxfp8_last_dim(
    qweight: torch.Tensor,
    scale: torch.Tensor,
    group_size: int = 32,
    out_dtype: torch.dtype = torch.bfloat16,
    tensor_name: str = "<unknown>",
    allow_reshape_scale: bool = True,
) -> torch.Tensor:
    if not is_fp8_tensor(qweight):
        raise TypeError(f"qweight must be FP8, got {qweight.dtype} for {tensor_name}")

    scale = normalize_scale_shape(
        qweight=qweight, scale=scale, group_size=group_size,
        tensor_name=tensor_name, allow_reshape_scale=allow_reshape_scale,
    )

    num_blocks = qweight.shape[-1] // group_size
    x = qweight.to(torch.float32)
    x_blk = x.reshape(*x.shape[:-1], num_blocks, group_size)

    factor = scale_to_mxfp8_factor(scale).to(torch.float32)
    y = x_blk * factor.unsqueeze(-1)
    return y.reshape_as(x).to(out_dtype)


# -----------------------------------------------------------------------------
# Conversion
# -----------------------------------------------------------------------------

def build_global_scale_key(weight_name: str) -> str:
    return weight_name + "_scale"


def is_potential_weight_name(name: str) -> bool:
    return name.endswith(".weight")


def copy_model_side_files(quant_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    common_files = [
        "config.json", "generation_config.json", "tokenizer_config.json",
        "special_tokens_map.json", "tokenizer.json", "vocab.json",
        "merges.txt", "tokenizer.model", "added_tokens.json", "chat_template.jinja",
    ]

    for filename in common_files:
        src = quant_dir / filename
        if src.exists():
            dst = output_dir / filename
            if filename == "config.json":
                cfg = load_json(src)
                cfg.pop("quantization_config", None)
                save_json(cfg, dst)
            else:
                shutil.copy2(src, dst)


def convert_mxfp8_checkpoint_to_hif4(
    quant_model_dir: Path,
    output_dir: Path,
    group_size_override: Optional[int] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    max_cached_shards: int = 3,
    allow_reshape_scale: bool = True,
    strict_no_fp8_copy: bool = True,
    debug_regex: Optional[str] = None,
    quiet: bool = False,
) -> None:
    """MXFP8 → BF16 → HiFloat4 full pipeline (self-sufficient; no bf16 reference)."""
    quant_index = load_index(quant_model_dir)
    # Output plan derived from the quant checkpoint's own index, dropping the consumed
    # ``*_scale`` tensors (read during dequant, not emitted).
    scale_keys = {
        build_global_scale_key(w) for w in quant_index["weight_map"]
        if build_global_scale_key(w) in quant_index["weight_map"]
    }
    plan_index = {
        "metadata": quant_index.get("metadata", {}),
        "weight_map": {
            k: v for k, v in quant_index["weight_map"].items() if k not in scale_keys
        },
    }

    quant_scope = infer_quant_scope(quant_model_dir)
    group_size = int(group_size_override or quant_scope.get("group_size") or 32)

    if not quiet:
        print("========== quant metadata ==========")
        printable = dict(quant_scope)
        printable.pop("raw", None)
        print(json.dumps(printable, indent=2, ensure_ascii=False))
        print(f"FP8 dtypes recognized: {[dtype_name(x) for x in FP8_DTYPES]}")
        print(f"Using group_size={group_size}, out_dtype={out_dtype}")

    output_dir.mkdir(parents=True, exist_ok=True)
    plan_shard_to_keys = build_shard_to_keys(plan_index)

    quant_cache = ShardCache(quant_model_dir, max_cached=max_cached_shards)

    debug_pat = re.compile(debug_regex) if debug_regex else None

    conversion_stats = {
        "dequantized_fp8_weights": 0,
        "copied_non_fp8_tensors": 0,
        "reshaped_scales": 0,
        "written_tensors": 0,
        "written_shards": 0,
        "dequant_examples": [],
    }

    for out_shard_name, plan_tensor_names in plan_shard_to_keys.items():
        out_tensors: Dict[str, torch.Tensor] = {}

        if not quiet:
            print(f"\n[Output shard] {out_shard_name}: {len(plan_tensor_names)} tensors")

        for tensor_name in plan_tensor_names:
            q_shard_name = quant_index["weight_map"][tensor_name]
            q_shard = quant_cache.get_shard(q_shard_name)
            if tensor_name not in q_shard:
                raise KeyError(f"{tensor_name} missing in its quant shard {q_shard_name}")

            qt = q_shard[tensor_name]

            if is_fp8_tensor(qt):
                if not is_potential_weight_name(tensor_name):
                    raise RuntimeError(
                        f"FP8 non-weight tensor encountered: {tensor_name}, "
                        f"shape={tuple(qt.shape)}, dtype={qt.dtype}"
                    )

                scale_key = build_global_scale_key(tensor_name)
                if scale_key not in quant_index["weight_map"]:
                    raise KeyError(
                        f"FP8 weight has no global scale tensor: {tensor_name}"
                    )

                scale_shard_name = quant_index["weight_map"][scale_key]
                scale_shard = quant_cache.get_shard(scale_shard_name)
                if scale_key not in scale_shard:
                    raise KeyError(f"{scale_key} missing in its quant shard {scale_shard_name}")
                scale = scale_shard[scale_key]

                expected_scale_shape = tuple(qt.shape[:-1]) + (qt.shape[-1] // group_size,)
                scale_shape_before = tuple(scale.shape)

                # step 1: MXFP8 → BF16
                w_bf16 = dequant_mxfp8_last_dim(
                    qweight=qt, scale=scale, group_size=group_size,
                    out_dtype=out_dtype, tensor_name=tensor_name,
                    allow_reshape_scale=allow_reshape_scale,
                ).contiguous()

                if scale_shape_before != expected_scale_shape:
                    conversion_stats["reshaped_scales"] += 1

                # step 2: BF16 → HiF4 fake-quant → BF16
                out = quant_dequant_float(w_bf16, _HIF4_QTYPE, force_py=True, force_fp32=True)

                out_tensors[tensor_name] = out
                conversion_stats["dequantized_fp8_weights"] += 1
                conversion_stats["written_tensors"] += 1

                if len(conversion_stats["dequant_examples"]) < 50:
                    conversion_stats["dequant_examples"].append(
                        {
                            "name": tensor_name,
                            "weight_shard": q_shard_name,
                            "scale_shard": scale_shard_name,
                            "q_shape": list(qt.shape),
                            "scale_shape": list(scale.shape),
                            "out_absmax": float(out.to(torch.float32).abs().max().item()),
                        }
                    )

                if not quiet:
                    msg = (
                        f"  [dequant MXFP8→HiF4] {tensor_name} "
                        f"weight_shard={q_shard_name}, scale_shard={scale_shard_name}"
                    )
                    if scale_shard_name != q_shard_name:
                        msg += "  <scale in different shard>"
                    print(msg)

                if debug_pat and debug_pat.search(tensor_name):
                    q_absmax = qt.to(torch.float32).abs().max().item()
                    s_min = int(scale.min().item())
                    s_max = int(scale.max().item())
                    out_absmax = out.to(torch.float32).abs().max().item()
                    print(f"    [DEBUG] q_absmax={q_absmax:.6g}, "
                          f"scale_minmax=({s_min},{s_max}), out_absmax={out_absmax:.6g}")

            else:
                if strict_no_fp8_copy and is_fp8_tensor(qt):
                    raise RuntimeError(f"FP8 tensor would be copied without dequant: {tensor_name}")
                out_tensors[tensor_name] = qt.to(out_dtype).contiguous()
                conversion_stats["copied_non_fp8_tensors"] += 1
                conversion_stats["written_tensors"] += 1
                if not quiet:
                    print(f"  [copy non-FP8] {tensor_name} dtype={qt.dtype}")

        out_path = output_dir / out_shard_name
        save_file(out_tensors, str(out_path))
        conversion_stats["written_shards"] += 1

        if not quiet:
            size = out_path.stat().st_size if out_path.exists() else None
            print(f"  [saved] {out_path.name}, size={human_bytes(size)}")

        del out_tensors
        gc.collect()

    out_index = {
        "metadata": plan_index.get("metadata", {}),
        "weight_map": dict(plan_index["weight_map"]),
    }
    save_json(out_index, output_dir / "model.safetensors.index.json")

    copy_model_side_files(quant_model_dir, output_dir)

    save_json(conversion_stats, output_dir / "mxfp8_to_hif4_conversion_stats.json")

    quant_cache.clear()

    print("\n========== conversion done ==========")
    print(f"Output dir: {output_dir}")
    summary = {k: v for k, v in conversion_stats.items() if k != "dequant_examples"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Stats file: {output_dir / 'mxfp8_to_hif4_conversion_stats.json'}")


# -----------------------------------------------------------------------------
# Scan / dump helpers
# -----------------------------------------------------------------------------

def scan_quant_checkpoint(
    quant_model_dir: Path,
    quant_index: dict,
    max_cached_shards: int = 2,
    max_print: int = 200,
) -> dict:
    shard_to_keys = build_shard_to_keys(quant_index)
    cache = ShardCache(quant_model_dir, max_cached=max_cached_shards)

    stats = {
        "num_tensors": 0,
        "dtype_counts": defaultdict(int),
        "num_fp8_tensors": 0,
        "num_fp8_weight_tensors": 0,
        "num_fp8_weight_with_scale_global": 0,
        "num_fp8_weight_missing_scale": 0,
        "fp8_missing_scale": [],
        "fp8_non_weight": [],
        "fp8_weight_examples": [],
    }

    for shard_name, keys in shard_to_keys.items():
        shard = cache.get_shard(shard_name)
        for name in keys:
            if name not in shard:
                continue
            t = shard[name]
            stats["num_tensors"] += 1
            stats["dtype_counts"][dtype_name(t.dtype)] += 1

            if is_fp8_tensor(t):
                stats["num_fp8_tensors"] += 1
                if is_potential_weight_name(name):
                    stats["num_fp8_weight_tensors"] += 1
                    scale_key = build_global_scale_key(name)
                    has_global_scale = scale_key in quant_index["weight_map"]

                    if has_global_scale:
                        stats["num_fp8_weight_with_scale_global"] += 1
                    else:
                        stats["num_fp8_weight_missing_scale"] += 1
                        if len(stats["fp8_missing_scale"]) < max_print:
                            stats["fp8_missing_scale"].append(name)

                    if len(stats["fp8_weight_examples"]) < max_print:
                        stats["fp8_weight_examples"].append(
                            {
                                "name": name,
                                "shape": list(t.shape),
                                "dtype": dtype_name(t.dtype),
                                "scale_key": scale_key,
                            }
                        )
                else:
                    if len(stats["fp8_non_weight"]) < max_print:
                        stats["fp8_non_weight"].append(name)

    stats["dtype_counts"] = dict(stats["dtype_counts"])
    cache.clear()
    return stats


def dump_quant_keys(
    quant_model_dir: Path,
    quant_index: dict,
    max_keys: int = 500,
    max_cached_shards: int = 2,
    regex: Optional[str] = None,
) -> None:
    cache = ShardCache(quant_model_dir, max_cached=max_cached_shards)
    pat = re.compile(regex) if regex else None

    printed = 0
    for shard_name, keys in build_shard_to_keys(quant_index).items():
        shard = cache.get_shard(shard_name)
        print(f"\n[{shard_name}]")
        for key in keys:
            if pat and not pat.search(key):
                continue
            t = shard[key]
            print(f"  {key:90s} shape={tuple(t.shape)!s:24s} dtype={t.dtype}")
            printed += 1
            if printed >= max_keys:
                print("\n... truncated ...")
                cache.clear()
                return
    cache.clear()


def parse_dtype(name: str) -> torch.dtype:
    normalized = name.lower().replace("torch.", "")
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported out dtype: {name}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MXFP8 → BF16 → HiFloat4 converter")
    parser.add_argument("--quant_model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--group_size", type=int, default=None,
                        help="Override MXFP8 group size. Default: read from config, fallback 32.")
    parser.add_argument("--out_dtype", type=str, default="bf16",
                        choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--max_cached_shards", type=int, default=3)
    parser.add_argument("--no_reshape_scale", action="store_true")
    parser.add_argument("--debug_regex", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--scan_quant_only", action="store_true")
    parser.add_argument("--dump_quant_keys", action="store_true")
    parser.add_argument("--dump_max_keys", type=int, default=300)
    parser.add_argument("--dump_regex", type=str, default=None)

    return parser


def main(argv: Optional[list] = None) -> None:
    args = build_parser().parse_args(argv)

    quant_model_dir = Path(args.quant_model_dir)
    output_dir = Path(args.output_dir)
    out_dtype = parse_dtype(args.out_dtype)

    quant_index = load_index(quant_model_dir)

    if args.dump_quant_keys:
        dump_quant_keys(
            quant_model_dir=quant_model_dir, quant_index=quant_index,
            max_keys=args.dump_max_keys, max_cached_shards=args.max_cached_shards,
            regex=args.dump_regex,
        )
        return

    if args.scan_quant_only:
        scope = infer_quant_scope(quant_model_dir)
        scan = scan_quant_checkpoint(
            quant_model_dir=quant_model_dir, quant_index=quant_index,
            max_cached_shards=args.max_cached_shards,
        )
        print("========== quant metadata ==========")
        printable = dict(scope)
        printable.pop("raw", None)
        print(json.dumps(printable, indent=2, ensure_ascii=False))
        print("\n========== quant scan ==========")
        print(json.dumps(scan, indent=2, ensure_ascii=False))
        return

    convert_mxfp8_checkpoint_to_hif4(
        quant_model_dir=quant_model_dir,
        output_dir=output_dir, group_size_override=args.group_size,
        out_dtype=out_dtype, max_cached_shards=args.max_cached_shards,
        allow_reshape_scale=not args.no_reshape_scale,
        strict_no_fp8_copy=True, debug_regex=args.debug_regex,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
