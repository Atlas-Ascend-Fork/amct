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
from torch import Tensor

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.realpath(os.path.join(_CUR_DIR, "../../../"))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from amct_pytorch.common.utils.hifloat4_quant import QType, quant_dequant_float  # noqa: E402
from amct_pytorch.common.utils.decode.fp4_formats import (  # noqa: E402
    unpack_nibbles, decode_fp4_e2m1, e2m1_lut, _as_uint8,
)
from amct_pytorch.common.utils.checkpoint_io import (  # noqa: E402
    resolve_index, invert_weight_map, save_shard, write_index,
)
from safetensors.torch import load_file  # noqa: E402

_HIF4_QTYPE = QType("hifx4").dim(-1)


# ============================================================
# MXFP4 format math (moved from fp4_formats: dequant + E8M0 + HiF4_E6M0 codec)
# ============================================================


def pack_nibbles(nibbles_u8: Tensor, low_first: bool = True) -> Tensor:
    """Pack 4-bit nibble codes along the last dimension back into uint8 bytes."""
    nibbles_u8 = _as_uint8(nibbles_u8, "nibbles_u8")
    if nibbles_u8.shape[-1] % 2 != 0:
        raise ValueError(
            f"last dim must be even to pack nibbles, got {nibbles_u8.shape[-1]}"
        )
    a = nibbles_u8[..., 0::2] & 0x0F
    b = nibbles_u8[..., 1::2] & 0x0F
    if low_first:
        return (a | (b << 4)).contiguous()
    return ((a << 4) | b).contiguous()


def exp2_int(exp: Tensor) -> Tensor:
    """Return ``2**exp`` for an integer exponent tensor."""
    return torch.ldexp(
        torch.ones_like(exp, dtype=torch.float32), exp.to(torch.int32)
    )


def e8m0_to_unbiased(e8m0_code: Tensor, e8_bias: int = 127) -> Tensor:
    """Convert E8M0 scale codes to signed unbiased integer exponents."""
    e8m0_code = _as_uint8(e8m0_code, "e8m0_code")
    if bool((e8m0_code == 255).any()):
        raise ValueError("E8M0 code 255 is reserved/NaN; found in scale tensor")
    return e8m0_code.to(torch.int16) - int(e8_bias)


def _round_to_int(x: Tensor, mode: str) -> Tensor:
    """Round float tensor to integer by *mode* (``nearest_even`` / ``half_up``)."""
    if mode == "nearest_even":
        return torch.round(x)
    if mode == "half_up":
        return torch.floor(x + 0.5)
    raise ValueError(f"unsupported round mode: {mode}")


def mxfp4_to_bf16(
    packed_fp4: Tensor,
    e8m0_scale: Tensor,
    *,
    low_first: bool = True,
    e8_bias: int = 127,
    block_size: int = 32,
) -> Tensor:
    """Decode MXFP4 packed tensor to BF16.

    One E8M0 scale code is shared by every ``block_size`` (=32) MXFP4 elements.
    Real value = E2M1(nibble) × 2^(E8M0_code − e8_bias).
    """
    if block_size != 32:
        raise ValueError("MXFP4 block_size is expected to be 32")
    packed_fp4 = _as_uint8(packed_fp4, "packed_fp4")
    e8m0_scale = _as_uint8(e8m0_scale, "e8m0_scale")

    nibbles = unpack_nibbles(packed_fp4, low_first=low_first)
    num_values = nibbles.shape[-1]
    if num_values % block_size != 0:
        raise ValueError(
            f"decoded value dim must be multiple of {block_size}, got {num_values}"
        )
    if e8m0_scale.shape[-1] != num_values // block_size:
        raise ValueError(
            f"scale last dim mismatch: scale={e8m0_scale.shape[-1]}, "
            f"expected={num_values // block_size}"
        )
    if packed_fp4.shape[:-1] != e8m0_scale.shape[:-1]:
        raise ValueError(
            f"prefix shape mismatch: packed={packed_fp4.shape[:-1]}, "
            f"scale={e8m0_scale.shape[:-1]}"
        )

    vals = decode_fp4_e2m1(nibbles)
    vals = vals.reshape(*vals.shape[:-1], e8m0_scale.shape[-1], block_size)

    e8_exp = e8m0_to_unbiased(e8m0_scale, e8_bias=e8_bias)
    scale = exp2_int(e8_exp).unsqueeze(-1)
    x = vals * scale
    return x.reshape(*nibbles.shape[:-1], num_values).to(torch.bfloat16).contiguous()


def compute_e6_exp_from_e8_pair(
    e8m0_scale: Tensor,
    *,
    e8_bias: int = 127,
    e6_rule: str = "unbiased",
) -> Tensor:
    """Convert pairs of E8M0 scales into one level-1 E6 exponent per 64 values.

    Args:
        e8m0_scale: ``[..., G*2]`` uint8 E8M0 codes.
        e8_bias: E8M0 exponent bias.
        e6_rule: ``"unbiased"`` — ``e6_exp = max(e8_code − bias) − 2``;
                 ``"raw"``      — ``e6_exp = max(e8_code) − 2``.

    Returns:
        ``[..., G]`` int16 exponent tensor.
    """
    e8m0_scale = _as_uint8(e8m0_scale, "e8m0_scale")
    if e8m0_scale.shape[-1] % 2 != 0:
        raise ValueError(
            f"scale last dim must be even, got {e8m0_scale.shape[-1]}"
        )
    pair = e8m0_scale.reshape(
        *e8m0_scale.shape[:-1], e8m0_scale.shape[-1] // 2, 2
    )
    if e6_rule == "unbiased":
        pair_exp = e8m0_to_unbiased(pair, e8_bias=e8_bias)
        return pair_exp.max(dim=-1).values.to(torch.int16) - 2
    if e6_rule == "raw":
        return pair.to(torch.int16).max(dim=-1).values - 2
    raise ValueError("e6_rule must be 'unbiased' or 'raw'")


def bf16_to_hif4_e6m0(
    x_bf16: Tensor,
    e8m0_scale: Tensor,
    *,
    e8_bias: int = 127,
    e6_rule: str = "unbiased",
    e6_bias: int = 0,
    round_mode: str = "nearest_even",
    pack_low_first: bool = True,
) -> Dict[str, Tensor]:
    """Encode BF16 tensor into HiFloat4_E6M0 components (**NOT official HiF4**).

    64 values per group; Level-1 E6 exponent from two E8M0 scales;
    Level-2 8×1-bit + Level-3 16×1-bit + 64×S1P2.

    Returns dict: e6_exp, e6_code, e2, e3, s1p2, s1p2_packed, qmag, y_scaled.
    """
    if x_bf16.shape[-1] % 64 != 0:
        raise ValueError(f"x last dim must be multiple of 64, got {x_bf16.shape[-1]}")
    if e8m0_scale.shape[-1] * 32 != x_bf16.shape[-1]:
        raise ValueError(
            f"scale last dim mismatch: {e8m0_scale.shape[-1] * 32} vs {x_bf16.shape[-1]}"
        )
    if x_bf16.shape[:-1] != e8m0_scale.shape[:-1]:
        raise ValueError("prefix shape mismatch")

    x = x_bf16.to(torch.float32).reshape(
        *x_bf16.shape[:-1], x_bf16.shape[-1] // 64, 64
    )
    e6_exp = compute_e6_exp_from_e8_pair(
        e8m0_scale, e8_bias=e8_bias, e6_rule=e6_rule
    ).to(x.device)
    e6_code = torch.clamp(e6_exp.to(torch.int16) + int(e6_bias), 0, 63).to(torch.uint8)

    base = exp2_int(e6_exp).unsqueeze(-1).to(x.device)
    abs_x = x.abs()

    m4 = abs_x.reshape(*abs_x.shape[:-1], 16, 4).amax(dim=-1)
    m8 = m4.reshape(*m4.shape[:-1], 8, 2).amax(dim=-1)

    e2 = (m8 > (3.5 * base)).to(torch.uint8)
    e2_for_e3 = e2.repeat_interleave(2, dim=-1)
    e3_threshold = 1.75 * base * torch.pow(2.0, e2_for_e3.float())
    e3 = (m4 > e3_threshold).to(torch.uint8)

    e2_elem = e2.repeat_interleave(8, dim=-1)
    e3_elem = e3.repeat_interleave(4, dim=-1)
    total_scale = base * torch.pow(2.0, e2_elem.float()) * torch.pow(2.0, e3_elem.float())

    y = x / total_scale
    qmag = _round_to_int(y.abs() * 4.0, round_mode).clamp(0, 7).to(torch.uint8)
    sign = (y < 0).to(torch.uint8)
    sign = torch.where(qmag == 0, torch.zeros_like(sign), sign)
    s1p2 = ((sign << 3) | qmag).to(torch.uint8)
    s1p2_packed = pack_nibbles(s1p2, low_first=pack_low_first)

    return {
        "e6_exp": e6_exp.to(torch.int16).contiguous(),
        "e6_code": e6_code.contiguous(),
        "e2": e2.contiguous(),
        "e3": e3.contiguous(),
        "s1p2": s1p2.contiguous(),
        "s1p2_packed": s1p2_packed.contiguous(),
        "qmag": qmag.contiguous(),
        "y_scaled": y.contiguous(),
    }


def hif4_e6m0_to_fp32(
    e6_exp: Tensor,
    e2: Tensor,
    e3: Tensor,
    s1p2: Tensor,
) -> Tensor:
    """Reconstruct FP32 values from HiF4_E6M0 components (validation only)."""
    if s1p2.shape[-1] != 64:
        raise ValueError(f"s1p2 last dim should be 64, got {s1p2.shape[-1]}")
    sign = torch.where(((s1p2 >> 3) & 1) != 0, -1.0, 1.0).to(torch.float32)
    mag = (s1p2 & 0x07).to(torch.float32) / 4.0
    base = exp2_int(e6_exp.to(torch.int16)).unsqueeze(-1).to(mag.device)
    e2_elem = e2.repeat_interleave(8, dim=-1).to(torch.float32)
    e3_elem = e3.repeat_interleave(4, dim=-1).to(torch.float32)
    x = sign * mag * base * torch.pow(2.0, e2_elem) * torch.pow(2.0, e3_elem)
    return x.reshape(*s1p2.shape[:-2], s1p2.shape[-2] * 64).contiguous()


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
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _transform(key: str, tensor: torch.Tensor, lookup: Any) -> Optional[torch.Tensor]:
        return transform_fn(key, tensor, lookup, low_first=args.low_first, e8_bias=args.e8_bias)

    # Small per-shard decode loop (output layout mirrors the source index; a weight's
    # weight_scale may live in another shard -> resolved via _lookup).
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
            result = _transform(key, tensor, _lookup)
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
        f"[decode] mxfp4 -> bf16: converted={stats['converted']}, "
        f"copied={stats['copied']}, dropped={stats['dropped']}, missing={stats['missing']}"
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
    sp.add_argument("--model", type=str, required=True,
                    help="MXFP4 checkpoint directory (same flag as ptq/deploy).")
    sp.add_argument("--output_dir", type=str, required=True,
                    help="Output directory for decoded BF16 shards + index.")
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
