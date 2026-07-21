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
"""FP16/BF16 <-> HiFloat4 checkpoint conversion (implementation).

The thin CLI front door ``amct_pytorch.cli.llm.cast_hif4`` just calls ``main`` here. Three verb
subcommands:

    fake-quant   FP16/BF16 --HiF4 fake-quant--> FP16/BF16   (eval: inject HiF4 error into
                 --dtype {fp16,bf16}                         float weights; NPU kernel if
                                                             available, else pure torch)
    pack         FP16/BF16 --real HiF4 pack---> scale+value uint8 safetensors  (deploy;
                                                             pure PyTorch)
    unpack       real HiF4 --reconstruct------> FP16/BF16    (reverse of pack; --dtype
                 --dtype {fp16,bf16}                          overrides, default = pack-time dtype)

Routing: ``fake-quant`` (the eval path) prefers the AscendC kernel and falls back to the
pure-torch reference off-NPU; ``pack`` (the deploy path) always uses the pure-torch packer.
Kernel and torch reference are numerically equivalent (<1e-6); ``pack`` and ``unpack`` are
bit-identical. Only 2-D+ ``*.weight`` tensors are quantized; everything else is copied.

Typical usage::

    python -m amct_pytorch.cli.llm.cast_hif4 fake-quant --input <model> --output <out> --dtype bf16
    python -m amct_pytorch.cli.llm.cast_hif4 pack       --input <model> --output <out>
    python -m amct_pytorch.cli.llm.cast_hif4 unpack     --input <packed> --output <out>
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import torch

from .hif4_pack_unpack import hif4_cast, hif4_bit_reconstruct
from .hif4_fake_quant import hifloat4_fake_quant

# HiF4 quantizes along the last dim in blocks of 64 (see hif4_pack_unpack / _hif4_encode_decode).
HIF4_BLOCK = 64
# Marks a directory produced by ``pack``; read back by ``unpack``.
HIF4_META_NAME = "hif4_cast_meta.json"


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


def _pad_last_to_block(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    rem = n % HIF4_BLOCK
    if rem == 0:
        return x
    return torch.nn.functional.pad(x, (0, HIF4_BLOCK - rem))


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


def cast_weight_hif4(t: torch.Tensor):
    """Real pack: BF16 -> HiF4 (scale, value) uint8 buffers + reconstruction metadata.

    Leading dims are flattened to a matrix and the last dim is padded to a multiple of 64;
    the original shape and pre-pad width are recorded so ``unpack`` restores exactly."""
    orig_shape = list(t.shape)
    mat = t.reshape(-1, orig_shape[-1])
    n_orig = mat.shape[-1]
    mat = _pad_last_to_block(mat)
    scale, value = hif4_cast(mat.to(torch.float32), output="packed")
    meta = {"orig_shape": orig_shape, "n_orig": n_orig, "n_padded": mat.shape[-1],
            "orig_dtype": str(t.dtype).replace("torch.", "")}  # e.g. "bfloat16" / "float16"
    return scale.contiguous(), value.contiguous(), meta


def _cast(input_dir: str, output_dir: str, out_format: str, quiet: bool) -> None:
    """Core forward pass shared by ``fake-quant`` (out_format fp16/bf16) and ``pack`` (hif4)."""
    _require_safetensors()
    from safetensors.torch import load_file, save_file

    if out_format not in ("fp16", "bf16", "hif4"):
        raise ValueError(f"output format must be fp16, bf16 or hif4, got {out_format!r}")
    os.makedirs(output_dir, exist_ok=True)
    # tensors copied verbatim (norms/biases) and the packed dir's non-weights use this float dtype.
    copy_dtype = _FLOAT_FMT.get(out_format, torch.bfloat16)

    index = load_index(input_dir)
    weight_map = index["weight_map"]
    s2k = shard_to_keys(weight_map)

    out_weight_map: Dict[str, str] = {}
    hif4_meta: Dict[str, Dict] = {}
    total_bytes = 0
    n_quant = n_copy = 0

    for shard, keys in s2k.items():
        tensors = load_file(os.path.join(input_dir, shard))
        out_tensors: Dict[str, torch.Tensor] = {}
        for name in keys:
            t = tensors[name]
            if _is_weight(name, t):
                if out_format in _FLOAT_FMT:  # fp16 / bf16 fake-quant
                    out_tensors[name] = cast_weight_float(t, _FLOAT_FMT[out_format])
                    out_weight_map[name] = shard
                else:  # hif4 real pack
                    scale, value, meta = cast_weight_hif4(t)
                    sk, vk = name + ".hif4_scale", name + ".hif4_value"
                    out_tensors[sk] = scale
                    out_tensors[vk] = value
                    out_weight_map[sk] = shard
                    out_weight_map[vk] = shard
                    hif4_meta[name] = {**meta, "scale_key": sk, "value_key": vk}
                n_quant += 1
            else:
                out_tensors[name] = t.to(copy_dtype) if t.is_floating_point() else t
                out_weight_map[name] = shard
                n_copy += 1
        for v in out_tensors.values():
            total_bytes += v.numel() * v.element_size()
        save_file(out_tensors, os.path.join(output_dir, shard),
                  metadata={"format": "pt", "hif4_output": out_format})
        if not quiet:
            print(f"[shard] {shard}: {len(out_tensors)} tensors written", flush=True)

    write_index(out_weight_map, total_bytes, output_dir)
    if out_format == "hif4":
        payload = {"format": "hif4_packed", "block": HIF4_BLOCK, "weights": hif4_meta}
        with open(os.path.join(output_dir, HIF4_META_NAME), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    _copy_side_files(input_dir, output_dir)
    mode = "pack" if out_format == "hif4" else f"fake-quant({out_format})"
    print(f"[{mode}] weights={n_quant} copied={n_copy} -> {output_dir}", flush=True)


def run_fake_quant(args) -> None:
    """``fake-quant`` subcommand: FP16/BF16 -> HiF4 -> FP16/BF16 (weights only)."""
    _cast(args.input, args.output, out_format=args.dtype, quiet=args.quiet)


def run_pack(args) -> None:
    """``pack`` subcommand: FP16/BF16 -> real HiF4 packed bytes."""
    _cast(args.input, args.output, out_format="hif4", quiet=args.quiet)


# --------------------------------------------------------------- unpack (rev)
def _unpack(input_dir: str, output_dir: str, out_dtype: Optional[str], quiet: bool) -> None:
    """Core reverse pass for ``unpack``: real HiF4 bytes -> FP16/BF16 float."""
    _require_safetensors()
    from safetensors.torch import load_file, save_file

    os.makedirs(output_dir, exist_ok=True)
    meta_path = os.path.join(input_dir, HIF4_META_NAME)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"{meta_path} not found — unpack expects a directory produced by "
            f"'cast_hif4 pack'")
    with open(meta_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    hif4_meta = payload["weights"]

    index = load_index(input_dir)
    weight_map = index["weight_map"]
    s2k = shard_to_keys(weight_map)
    # keys that are the two halves of a packed weight -> the weight they belong to
    packed_child = {}
    for wname, m in hif4_meta.items():
        packed_child[m["scale_key"]] = wname
        packed_child[m["value_key"]] = wname

    out_weight_map: Dict[str, str] = {}
    total_bytes = 0
    n_deq = n_copy = 0

    for shard, keys in s2k.items():
        tensors = load_file(os.path.join(input_dir, shard))
        out_tensors: Dict[str, torch.Tensor] = {}
        done_weights = set()
        for name in keys:
            if name in packed_child:
                wname = packed_child[name]
                if wname in done_weights:
                    continue
                m = hif4_meta[wname]
                scale = tensors[m["scale_key"]]
                value = tensors[m["value_key"]]
                # output dtype: --dtype override, else the source dtype recorded at pack time.
                out_dtype_t = (_DTYPE_BY_NAME[out_dtype] if out_dtype
                               else _DTYPE_BY_NAME.get(m.get("orig_dtype", "bf16"), torch.bfloat16))
                recon = hif4_bit_reconstruct(scale, value)          # fp32 [rows, n_padded]
                recon = torch.as_tensor(recon)[:, : m["n_orig"]]     # unpad
                recon = recon.reshape(*m["orig_shape"]).to(out_dtype_t)
                out_tensors[wname] = recon
                out_weight_map[wname] = shard
                done_weights.add(wname)
                n_deq += 1
            else:
                out_tensors[name] = tensors[name]
                out_weight_map[name] = shard
                n_copy += 1
        for v in out_tensors.values():
            total_bytes += v.numel() * v.element_size()
        save_file(out_tensors, os.path.join(output_dir, shard), metadata={"format": "pt"})
        if not quiet:
            print(f"[shard] {shard}: {len(out_tensors)} tensors written", flush=True)

    write_index(out_weight_map, total_bytes, output_dir)
    _copy_side_files(input_dir, output_dir, skip=(HIF4_META_NAME,))
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
        description="FP16/BF16 <-> HiFloat4 checkpoint conversion (fake-quant / pack / unpack)")
    sub = ap.add_subparsers(dest="command", required=True)

    fq = sub.add_parser(
        "fake-quant",
        help="FP16/BF16 -> HiF4 -> FP16/BF16 fake-quant (eval; weights only). "
             "NPU kernel if available, else pure-torch reference")
    fq.add_argument("--input", required=True, help="FP16/BF16 model dir (safetensors)")
    fq.add_argument("--output", required=True, help="output dir")
    fq.add_argument("--dtype", default="bf16", choices=["fp16", "bf16"],
                    help="output float dtype (default bf16)")
    fq.add_argument("--quiet", action="store_true")
    fq.set_defaults(func=run_fake_quant)

    pk = sub.add_parser(
        "pack", help="FP16/BF16 -> real HiF4 packed bytes (scale+value uint8 safetensors; deploy)")
    pk.add_argument("--input", required=True, help="FP16/BF16 model dir (safetensors)")
    pk.add_argument("--output", required=True, help="output dir")
    pk.add_argument("--quiet", action="store_true")
    pk.set_defaults(func=run_pack)

    up = sub.add_parser("unpack", help="real HiF4 packed bytes -> FP16/BF16 float (reverse of pack)")
    up.add_argument("--input", required=True, help="packed HiF4 dir (from 'cast_hif4 pack')")
    up.add_argument("--output", required=True, help="output float dir")
    up.add_argument("--dtype", default=None, choices=["fp16", "bf16"],
                    help="output dtype (default: the source dtype recorded at pack time)")
    up.add_argument("--quiet", action="store_true")
    up.set_defaults(func=run_unpack)
    return ap


def main(argv: Optional[list] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
