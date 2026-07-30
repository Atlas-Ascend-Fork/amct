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
"""Detect a checkpoint's quant source format and (on downstream branches) decode it to BF16.

Used by the ptq/deploy entry points to route an input checkpoint. On this baseline branch only
two source formats exist: our own HiF4 packed checkpoints (``ptq`` reconstructs them to float via
``decode.unpack_hif4``) and plain float. ``FOREIGN_FORMATS`` is therefore empty here; downstream
branches register foreign FP4/FP8 formats (nvfp4/mxfp4/mxfp8) by adding them to ``FOREIGN_FORMATS``
and a matching branch in ``decode_to_bf16``.

NOTE: ``detect_source_format`` is a best-effort heuristic (config + weight markers) and should be
validated against real checkpoints of each format.
"""
import json
import os
import shutil
from typing import Set

# Foreign (non-float, non-HiF4) formats that must be decoded to BF16 before the HiF4 path. Empty on
# this baseline branch; downstream branches append their format (e.g. ("nvfp4",)).
FOREIGN_FORMATS = ()

_SIDE_FILES = (
    "config.json", "generation_config.json", "tokenizer_config.json",
    "special_tokens_map.json", "tokenizer.json", "tokenizer.model",
    "vocab.json", "merges.txt", "added_tokens.json", "chat_template.jinja",
)


def _read_config(model_dir: str) -> dict:
    path = os.path.join(model_dir, "config.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _tensor_keys(model_dir: str) -> Set[str]:
    """Tensor-key set from the index.json, else the single-file safetensors."""
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        try:
            with open(idx, "r", encoding="utf-8") as f:
                return set(json.load(f).get("weight_map", {}).keys())
        except (OSError, ValueError):
            return set()
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        try:
            from safetensors import safe_open
            with safe_open(single, framework="pt") as f:
                return set(f.keys())
        except Exception:  # pragma: no cover
            return set()
    return set()


def _peek_tensor(model_dir: str, key: str):
    """``(shape tuple, dtype string)`` of one tensor via its index shard / single file.

    Returns ``(None, None)`` on any failure. dtype is the safetensors string, e.g. 'U8', 'F8_E4M3'.
    """
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        try:
            with open(idx, "r", encoding="utf-8") as f:
                shard = json.load(f)["weight_map"].get(key)
        except (OSError, ValueError, KeyError):
            shard = None
        path = os.path.join(model_dir, shard) if shard else None
    else:
        path = os.path.join(model_dir, "model.safetensors")
    if not path or not os.path.exists(path):
        return None, None
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt") as f:
            sl = f.get_slice(key)
            return tuple(sl.get_shape()), sl.get_dtype()
    except Exception:  # pragma: no cover
        return None, None


def detect_source_format(model_dir: str) -> str:
    """Best-effort detection. Returns 'hif4'|'fp16'|'bf16' on this baseline branch.

    Order: HiF4 pack marker (weight_scale shape) -> unsupported-quantized guard -> float dtype
    fallback. A checkpoint that carries quant markers but is not HiF4 raises ``ValueError`` here
    (foreign FP4/FP8 decode is only available on downstream branches); it is never silently treated
    as float, which would corrupt packed bytes.
    """
    keys = _tensor_keys(model_dir)
    # 1. HiF4 packed (our deploy pack): every quantized weight stores <w>.weight (uint8 codes)
    #    plus <w>.weight_scale as a uint8 record of shape [M, N//64, 4] (E6M2 + 3 exponent bytes
    #    per 64-group). A 3-D uint8 weight_scale with a trailing 4 uniquely marks HiF4.
    if (any(k.endswith(".weight_scale") for k in keys)
            and not any(k.endswith(".weight_scale_2") for k in keys)):
        s_key = next(k for k in sorted(keys) if k.endswith(".weight_scale"))
        shape, dtype = _peek_tensor(model_dir, s_key)
        if dtype and dtype.upper() == "U8" and shape and len(shape) == 3 and shape[-1] == 4:
            return "hif4"

    # 2. Unsupported quantized format on this branch: it carries quant markers (a `.weight_scale` /
    #    `.weight_scale_2` tensor or a `quantization_config`) but is not HiF4. Report clearly rather
    #    than silently loading packed bytes as float. Downstream branches decode foreign FP4/FP8.
    cfg = _read_config(model_dir)
    qc = cfg.get("quantization_config", {}) or {}
    if (any(k.endswith(".weight_scale_2") for k in keys)
            or any(k.endswith(".weight_scale") for k in keys) or qc):
        raise ValueError(
            f"Unsupported quantized checkpoint at {model_dir!r}: this branch supports HiF4 and "
            f"plain float only. Foreign FP4/FP8 (nvfp4/mxfp4/mxfp8) decode is added on downstream "
            f"branches."
        )

    # 3. plain float (check bfloat16 first -- "float16" is a substring of "bfloat16")
    dt = str(cfg.get("torch_dtype", "")).lower()
    if "bfloat16" in dt or dt == "bf16":
        return "bf16"
    return "fp16" if ("float16" in dt or dt == "fp16") else "bf16"


def copy_side_files(src: str, dst: str) -> None:
    """Copy config/tokenizer side files so the decoded dir loads as a plain BF16 model.

    ``config.json`` has its ``quantization_config`` stripped (the decoded weights are plain BF16,
    not quantized) and ``torch_dtype`` set to bfloat16, so the model loader does not treat the
    decoded checkpoint as still-quantized.
    """
    os.makedirs(dst, exist_ok=True)
    for fn in _SIDE_FILES:
        p = os.path.join(src, fn)
        if not os.path.exists(p) or os.path.exists(os.path.join(dst, fn)):
            continue
        if fn == "config.json":
            try:
                with open(p, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                cfg.pop("quantization_config", None)
                cfg["torch_dtype"] = "bfloat16"
                with open(os.path.join(dst, fn), "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2)
                continue
            except (OSError, ValueError):
                pass  # fall back to a verbatim copy
        shutil.copy2(p, os.path.join(dst, fn))


def decode_to_bf16(model_dir: str, out_dir: str, fmt: str) -> str:
    """Decode a foreign checkpoint to a BF16 checkpoint at ``out_dir``; returns out_dir.

    No foreign formats are registered on this baseline branch (``FOREIGN_FORMATS`` is empty), so
    any ``fmt`` reaching here is an error. Downstream branches add an ``elif fmt == "<format>"``
    branch that dispatches to the format's ``decode`` unpacker.
    """
    os.makedirs(out_dir, exist_ok=True)
    raise ValueError(f"decode_to_bf16: {fmt!r} is not a foreign format {FOREIGN_FORMATS}")
