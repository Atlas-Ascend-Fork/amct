# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""CLI entry point for FP16/BF16 <-> HiFloat4 checkpoint cast.

The user chooses what a cast produces via ``--output-format``:

    cast --output-format fp16   FP16/BF16 --quant_dequant_float--> FP16  (HiF4 fake-quant,
    cast --output-format bf16   FP16/BF16 --quant_dequant_float--> BF16   dequantized float;
                                runs on any hardware, e.g. Atlas A2/A3/A5)
    cast --output-format hif4   FP16/BF16 --hif4_cast(packed)----> real HiF4 packed bytes
                                (scale + value uint8 safetensors; for HiF4-capable
                                hardware, e.g. Atlas A6)

    dequant                     real HiF4 safetensors --hif4_bit_reconstruct--> FP16/BF16

Both the fake-quant (float) and the real-pack paths share the same HiF4 quantization
(``amct_pytorch.common.utils.hifx4_quant.bit_dump``), so a fake-quant float tensor and
the dequant of the matching packed tensor are bit-identical.

Typical usage::

    python -m amct_pytorch.cli.cast_hif4 cast \\
        --input /path/to/model --output /path/to/out --output-format hif4
    python -m amct_pytorch.cli.cast_hif4 dequant \\
        --input /path/to/hif4-packed --output /path/to/float-out
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import torch

from amct_pytorch.common.utils.hifx4_quant import (
    QType,
    quant_dequant_float,
    hif4_cast,
    hif4_bit_reconstruct,
)

# HiF4 quantizes along the last dim in blocks of 64 (see bit_dump / quant_hifx).
HIF4_BLOCK = 64
# Marks a directory produced by ``cast --output-format hif4``; read back by ``dequant``.
HIF4_META_NAME = "hif4_cast_meta.json"
_HIF4_QTYPE = QType("hifx4").dim(-1)


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


# --output-format float choices -> the torch dtype the dequantized weight is emitted as.
_FLOAT_FMT = {"fp16": torch.float16, "bf16": torch.bfloat16}
# name (as stored in meta / --out-dtype) -> torch float dtype, for dequant.
_DTYPE_BY_NAME = {"fp16": torch.float16, "float16": torch.float16,
                  "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}


def cast_weight_float(t: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Fake-quant: float -> HiF4 -> float (dequantized, output ``out_dtype``, no packing)."""
    return quant_dequant_float(t, _HIF4_QTYPE, force_py=True, force_fp32=True).to(out_dtype)


def cast_weight_hif4(t: torch.Tensor):
    """Real pack: BF16 -> HiF4 (scale, value) uint8 buffers + reconstruction metadata.

    Leading dims are flattened to a matrix and the last dim is padded to a multiple of 64;
    the original shape and pre-pad width are recorded so ``dequant`` restores exactly."""
    orig_shape = list(t.shape)
    mat = t.reshape(-1, orig_shape[-1])
    n_orig = mat.shape[-1]
    mat = _pad_last_to_block(mat)
    scale, value = hif4_cast(mat.to(torch.float32), output="packed")
    meta = {"orig_shape": orig_shape, "n_orig": n_orig, "n_padded": mat.shape[-1],
            "orig_dtype": str(t.dtype).replace("torch.", "")}  # e.g. "bfloat16" / "float16"
    return scale.contiguous(), value.contiguous(), meta


def run_cast(args) -> None:
    _require_safetensors()
    from safetensors.torch import load_file, save_file

    if args.output_format not in ("fp16", "bf16", "hif4"):
        raise ValueError(f"--output-format must be fp16, bf16 or hif4, got {args.output_format!r}")
    os.makedirs(args.output, exist_ok=True)
    # tensors copied verbatim (norms/biases) and the packed dir's non-weights use this float dtype.
    copy_dtype = _FLOAT_FMT.get(args.output_format, torch.bfloat16)

    index = load_index(args.input)
    weight_map = index["weight_map"]
    s2k = shard_to_keys(weight_map)

    out_weight_map: Dict[str, str] = {}
    hif4_meta: Dict[str, Dict] = {}
    total_bytes = 0
    n_quant = n_copy = 0

    for shard, keys in s2k.items():
        tensors = load_file(os.path.join(args.input, shard))
        out_tensors: Dict[str, torch.Tensor] = {}
        for name in keys:
            t = tensors[name]
            if _is_weight(name, t):
                if args.output_format in _FLOAT_FMT:  # fp16 / bf16 fake-quant
                    out_tensors[name] = cast_weight_float(t, _FLOAT_FMT[args.output_format])
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
        save_file(out_tensors, os.path.join(args.output, shard),
                  metadata={"format": "pt", "hif4_output": args.output_format})
        if not args.quiet:
            print(f"[shard] {shard}: {len(out_tensors)} tensors written", flush=True)

    write_index(out_weight_map, total_bytes, args.output)
    if args.output_format == "hif4":
        payload = {"format": "hif4_packed", "block": HIF4_BLOCK, "weights": hif4_meta}
        with open(os.path.join(args.output, HIF4_META_NAME), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    _copy_side_files(args.input, args.output)
    print(f"[cast] output-format={args.output_format} | quantized={n_quant} copied={n_copy} "
          f"-> {args.output}", flush=True)


# --------------------------------------------------------------- dequant (rev)
def run_dequant(args) -> None:
    _require_safetensors()
    from safetensors.torch import load_file, save_file

    os.makedirs(args.output, exist_ok=True)
    meta_path = os.path.join(args.input, HIF4_META_NAME)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"{meta_path} not found — dequant expects a directory produced by "
            f"'cast --output-format hif4'")
    with open(meta_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    hif4_meta = payload["weights"]

    index = load_index(args.input)
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
        tensors = load_file(os.path.join(args.input, shard))
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
                # output dtype: --out-dtype override, else the source dtype recorded at pack time.
                out_dtype = (_DTYPE_BY_NAME[args.out_dtype] if args.out_dtype
                             else _DTYPE_BY_NAME.get(m.get("orig_dtype", "bf16"), torch.bfloat16))
                recon = hif4_bit_reconstruct(scale, value)          # fp32 [rows, n_padded]
                recon = torch.as_tensor(recon)[:, : m["n_orig"]]     # unpad
                recon = recon.reshape(*m["orig_shape"]).to(out_dtype)
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
        save_file(out_tensors, os.path.join(args.output, shard), metadata={"format": "pt"})
        if not args.quiet:
            print(f"[shard] {shard}: {len(out_tensors)} tensors written", flush=True)

    write_index(out_weight_map, total_bytes, args.output)
    _copy_side_files(args.input, args.output, skip=(HIF4_META_NAME,))
    print(f"[dequant] hif4 -> float | dequantized={n_deq} copied={n_copy} -> {args.output}",
          flush=True)


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
    ap = argparse.ArgumentParser(description="BF16 <-> HiFloat4 checkpoint cast")
    sub = ap.add_subparsers(dest="command", required=True)

    c = sub.add_parser("cast", help="cast an FP16/BF16 checkpoint to HiF4 (float fake-quant or real packed)")
    c.add_argument("--input", required=True, help="FP16/BF16 model dir (safetensors)")
    c.add_argument("--output", required=True, help="output dir")
    c.add_argument("--output-format", default="bf16", choices=["fp16", "bf16", "hif4"],
                   help="fp16/bf16 = HiF4 fake-quant, output that float dtype; "
                        "hif4 = real packed HiF4 safetensors (scale+value bytes)")
    c.add_argument("--quiet", action="store_true")
    c.set_defaults(func=run_cast)

    d = sub.add_parser("dequant", help="unpack a real HiF4 checkpoint back to FP16/BF16")
    d.add_argument("--input", required=True, help="packed HiF4 dir (from 'cast --output-format hif4')")
    d.add_argument("--output", required=True, help="output float dir")
    d.add_argument("--out-dtype", default=None, choices=["fp16", "bf16"],
                   help="dequant output dtype (default: the source dtype recorded at pack time)")
    d.add_argument("--quiet", action="store_true")
    d.set_defaults(func=run_dequant)
    return ap


def main(argv: Optional[list] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
