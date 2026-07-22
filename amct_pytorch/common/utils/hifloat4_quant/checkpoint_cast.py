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
"""HiFloat4 checkpoint fake-quant / unpack (implementation).

Two verb subcommands. Real HiF4 *packing* is NOT here -- deploy packs via
``QuantDequantHiF4.export_deploy`` -> ``hif4_cast``; this module only fake-quants and unpacks.

    fake-quant   FP16/BF16 --HiF4 fake-quant--> FP16/BF16   (inject HiF4 error into float
                 --dtype {fp16,bf16}                         weights; NPU kernel if available,
                                                             else pure-torch reference)
    unpack       HiF4 packed --reconstruct----> FP16/BF16    (deploy layout: weight=value uint8
                 --dtype {fp16,bf16}                          + weight_scale; hif4_bit_reconstruct)

Only 2-D+ ``*.weight`` tensors are quantized/reconstructed; everything else is copied. Used
programmatically (e.g. by ``cli/llm/ptq.py``); the standalone ``cast_hif4`` CLI was removed.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import torch

from .hif4_pack_unpack import hif4_bit_reconstruct
from .hif4_fake_quant import hifloat4_fake_quant


# --------------------------------------------------------------------------- I/O
def _require_safetensors():
    try:
        from safetensors.torch import load_file, save_file  # noqa: F401
        from safetensors import safe_open  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError("cast_hif4 needs safetensors: pip install safetensors") from exc


def load_index(model_dir: str) -> Dict:
    """Return {'weight_map': {tensor -> shard file}} for a sharded or single-file model."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        if "weight_map" not in index:
            raise ValueError(f"{index_path} has no weight_map")
        return index
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        from safetensors import safe_open
        with safe_open(single, framework="pt") as f:
            keys = list(f.keys())
        return {"weight_map": {k: "model.safetensors" for k in keys}}
    raise FileNotFoundError(
        f"no model.safetensors.index.json or model.safetensors under {model_dir}")


def shard_to_keys(weight_map: Dict[str, str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for key, shard in weight_map.items():
        out.setdefault(shard, []).append(key)
    return out


def write_index(weight_map: Dict[str, str], total_bytes: int, out_dir: str) -> None:
    index = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


# ------------------------------------------------------------------- cast (fwd)
def _is_weight(name: str, t: torch.Tensor) -> bool:
    """Quantize 2-D+ ``*.weight`` tensors; copy everything else (norms, biases, 1-D)."""
    return name.endswith(".weight") and t.dim() >= 2


# fake-quant --dtype choices -> the torch dtype the fake-quantized weight is emitted as.
_FLOAT_FMT = {"fp16": torch.float16, "bf16": torch.bfloat16}
# name (as stored in meta / unpack --dtype) -> torch float dtype.
_DTYPE_BY_NAME = {"fp16": torch.float16, "float16": torch.float16,
                  "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}


def _npu_available() -> bool:
    """True if torch_npu is importable and an Ascend NPU is visible."""
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return getattr(torch, "npu", None) is not None and torch.npu.is_available()


def cast_weight_float(t: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Fake-quant: float -> HiF4 -> float (dequantized, output ``out_dtype``, no packing).

    Routed through the AscendC NPU kernel when an NPU is available (the weight is moved to
    NPU); off-NPU it falls back to the pure-torch reference. ``hifloat4_fake_quant`` makes
    that choice -- kernel only for an npu-resident fp16/bf16 tensor, else pure torch. The two
    paths are numerically equivalent (kernel validated against the reference to <1e-6)."""
    x = t.to(out_dtype)                                  # kernel needs fp16/bf16
    if _npu_available():
        x = x.npu()
    return hifloat4_fake_quant(x, qdim=-1).cpu().to(out_dtype)


def _cast(input_dir: str, output_dir: str, out_format: str, quiet: bool) -> None:
    """Fake-quant a whole checkpoint: FP16/BF16 -> HiF4 -> FP16/BF16 (weights only).

    Real HiF4 packing is not done here -- deploy packs via ``QuantDequantHiF4.export_deploy``
    (``hif4_cast``); this module only provides the fake-quant (and unpack) checkpoint passes."""
    _require_safetensors()
    from safetensors.torch import load_file, save_file

    if out_format not in _FLOAT_FMT:
        raise ValueError(f"output format must be fp16 or bf16, got {out_format!r}")
    os.makedirs(output_dir, exist_ok=True)
    copy_dtype = _FLOAT_FMT[out_format]

    index = load_index(input_dir)
    weight_map = index["weight_map"]
    s2k = shard_to_keys(weight_map)

    out_weight_map: Dict[str, str] = {}
    total_bytes = 0
    n_quant = n_copy = 0

    for shard, keys in s2k.items():
        tensors = load_file(os.path.join(input_dir, shard))
        out_tensors: Dict[str, torch.Tensor] = {}
        for name in keys:
            t = tensors[name]
            if _is_weight(name, t):
                out_tensors[name] = cast_weight_float(t, copy_dtype)
                n_quant += 1
            else:
                out_tensors[name] = t.to(copy_dtype) if t.is_floating_point() else t
                n_copy += 1
            out_weight_map[name] = shard
        for v in out_tensors.values():
            total_bytes += v.numel() * v.element_size()
        save_file(out_tensors, os.path.join(output_dir, shard),
                  metadata={"format": "pt", "hif4_output": out_format})
        if not quiet:
            print(f"[shard] {shard}: {len(out_tensors)} tensors written", flush=True)

    write_index(out_weight_map, total_bytes, output_dir)
    _copy_side_files(input_dir, output_dir)
    print(f"[fake-quant({out_format})] weights={n_quant} copied={n_copy} -> {output_dir}",
          flush=True)


def run_fake_quant(args) -> None:
    """``fake-quant`` subcommand: FP16/BF16 -> HiF4 -> FP16/BF16 (weights only)."""
    _cast(args.input, args.output, out_format=args.dtype, quiet=args.quiet)


# --------------------------------------------------------------- unpack (rev)
def _unpack(input_dir: str, output_dir: str, out_dtype: Optional[str], quiet: bool) -> None:
    """Reverse pass: HiF4 packed bytes -> FP16/BF16 float.

    Reads the canonical pack layout produced by deploy (``QuantDequantHiF4.export_deploy`` ->
    ``hif4_cast``): the packed *value* bytes live under the weight key (uint8) and the *scale*
    bytes under ``<weight>_scale``. Reconstruction uses ``hif4_bit_reconstruct``
    (hif4_pack_unpack). No separate metadata file is required."""
    _require_safetensors()
    from safetensors.torch import load_file, save_file

    os.makedirs(output_dir, exist_ok=True)
    out_dtype_t = _DTYPE_BY_NAME.get(out_dtype or "fp16", torch.float16)

    index = load_index(input_dir)
    weight_map = index["weight_map"]
    s2k = shard_to_keys(weight_map)
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
                scale = _load(weight_map[scale_key])[scale_key]  # scale may be in another shard
                recon = hif4_bit_reconstruct(scale, t)           # fp32 [rows, N]
                out_tensors[name] = torch.as_tensor(recon).to(out_dtype_t)
                n_deq += 1
            else:
                out_tensors[name] = t
                n_copy += 1
            out_weight_map[name] = shard
        for v in out_tensors.values():
            total_bytes += v.numel() * v.element_size()
        save_file(out_tensors, os.path.join(output_dir, shard), metadata={"format": "pt"})
        if not quiet:
            print(f"[shard] {shard}: {len(out_tensors)} tensors written", flush=True)

    write_index(out_weight_map, total_bytes, output_dir)
    _copy_side_files(input_dir, output_dir)
    print(f"[unpack] hif4 -> float | weights={n_deq} copied={n_copy} -> {output_dir}",
          flush=True)


def run_unpack(args) -> None:
    """``unpack`` subcommand: real HiF4 packed bytes -> FP16/BF16 float."""
    _unpack(args.input, args.output, out_dtype=args.dtype, quiet=args.quiet)


# -------------------------------------------------------------------- helpers
_SIDE_FILES = ("config.json", "generation_config.json", "tokenizer_config.json",
               "special_tokens_map.json", "tokenizer.json", "tokenizer.model",
               "vocab.json", "merges.txt", "added_tokens.json", "chat_template.jinja")


def _copy_side_files(src: str, dst: str, skip=()) -> None:
    import shutil
    os.makedirs(dst, exist_ok=True)
    for fn in _SIDE_FILES:
        if fn in skip:
            continue
        p = os.path.join(src, fn)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(dst, fn))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="HiFloat4 checkpoint fake-quant / unpack (pack is done by deploy)")
    sub = ap.add_subparsers(dest="command", required=True)

    fq = sub.add_parser(
        "fake-quant",
        help="FP16/BF16 -> HiF4 -> FP16/BF16 fake-quant (weights only). "
             "NPU kernel if available, else pure-torch reference")
    fq.add_argument("--input", required=True, help="FP16/BF16 model dir (safetensors)")
    fq.add_argument("--output", required=True, help="output dir")
    fq.add_argument("--dtype", default="bf16", choices=["fp16", "bf16"],
                    help="output float dtype (default bf16)")
    fq.add_argument("--quiet", action="store_true")
    fq.set_defaults(func=run_fake_quant)

    up = sub.add_parser(
        "unpack",
        help="HiF4 packed bytes (deploy layout: weight=value uint8 + weight_scale) -> FP16/BF16")
    up.add_argument("--input", required=True, help="packed HiF4 model dir")
    up.add_argument("--output", required=True, help="output float dir")
    up.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"],
                    help="output float dtype (default fp16)")
    up.add_argument("--quiet", action="store_true")
    up.set_defaults(func=run_unpack)
    return ap


def main(argv: Optional[list] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
