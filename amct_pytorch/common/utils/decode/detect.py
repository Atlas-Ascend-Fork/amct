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
"""Detect a checkpoint's quant source format and decode foreign formats to BF16.

Used by the ptq/deploy entry points to route an input checkpoint: foreign FP4/FP8 formats
(nvfp4/mxfp4/mxfp8) are decoded to BF16 first (via the decode unpackers), after which
HiF4 fake-quant (ptq) or pack (deploy) is applied by ``hifloat4_quant``.

NOTE: ``detect_source_format`` is a best-effort heuristic (config + weight markers) and should
be validated against real checkpoints of each format.
"""
import json
import os
import shutil
from typing import Optional, Set

FOREIGN_FORMATS = ("nvfp4", "mxfp4", "mxfp8")

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


def _peek_weight_dtype(model_dir: str, keys: Set[str]) -> Optional[str]:
    """safetensors dtype string of the first ``*.weight`` (e.g. 'F8_E4M3', 'U8', 'BF16')."""
    weight_keys = sorted(k for k in keys if k.endswith(".weight"))
    if not weight_keys:
        return None
    return _peek_tensor(model_dir, weight_keys[0])[1]


def detect_source_format(model_dir: str) -> str:
    """Best-effort detection. Returns 'hif4'|'nvfp4'|'mxfp8'|'mxfp4'|'fp16'|'bf16'.

    Order: HiF4 pack marker (weight_scale shape) -> config.json quantization_config hint ->
    weight-key markers (with a dtype peek to split mxfp4 vs mxfp8) -> float dtype fallback.
    """
    keys = _tensor_keys(model_dir)
    # 1. HiF4 packed (our deploy pack): every quantized weight stores <w>.weight (uint8 codes)
    #    plus <w>.weight_scale as a uint8 record of shape [M, N//64, 4] (E6M2 + 3 exponent bytes
    #    per 64-group). mxfp4/mxfp8 also carry .weight_scale but it is 2-D; nvfp4 additionally has
    #    .weight_scale_2. So a 3-D uint8 weight_scale with a trailing 4 uniquely marks HiF4.
    if (any(k.endswith(".weight_scale") for k in keys)
            and not any(k.endswith(".weight_scale_2") for k in keys)):
        s_key = next(k for k in sorted(keys) if k.endswith(".weight_scale"))
        shape, dtype = _peek_tensor(model_dir, s_key)
        if dtype and dtype.upper() == "U8" and shape and len(shape) == 3 and shape[-1] == 4:
            return "hif4"

    # 2. config.json quantization_config hints
    cfg = _read_config(model_dir)
    qc = cfg.get("quantization_config", {}) or {}
    hint = " ".join(
        str(qc.get(k, "")) for k in
        ("quant_algo", "quant_method", "weight_format", "activation_scheme", "fmt")
    ).lower()
    if "nvfp4" in hint:
        return "nvfp4"
    if "mxfp8" in hint:
        return "mxfp8"
    if "mxfp4" in hint:
        return "mxfp4"

    # 3. weight-key markers
    if any(k.endswith(".weight_scale_2") for k in keys):
        return "nvfp4"  # NVFP4 carries both weight_scale (block) and weight_scale_2 (global)
    if any(k.endswith(".weight_scale") for k in keys):
        # mxfp4 (E2M1 packed -> uint8 weight) vs mxfp8 (E4M3 -> fp8 weight): split by dtype.
        dt = (_peek_weight_dtype(model_dir, keys) or "").upper()
        if dt.startswith("F8") or "E4M3" in dt or "E5M2" in dt:
            return "mxfp8"
        return "mxfp4"

    # 4. plain float (check bfloat16 first -- "float16" is a substring of "bfloat16")
    dt = str(cfg.get("torch_dtype", "")).lower()
    if "bfloat16" in dt or dt == "bf16":
        return "bf16"
    return "fp16" if ("float16" in dt or dt == "fp16") else "bf16"


def copy_side_files(src: str, dst: str) -> None:
    """Copy config/tokenizer side files so the decoded dir loads as a plain BF16 model.

    ``config.json`` has its ``quantization_config`` stripped (the decoded weights are plain
    BF16, not quantized) and ``torch_dtype`` set to bfloat16, so the model loader does not treat
    the decoded checkpoint as still-quantized.
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
    """Decode a foreign FP4/FP8 checkpoint to a BF16 checkpoint at ``out_dir``; returns out_dir.

    Dispatches to the format's ``decode`` unpacker (decode-only; no HiF4). Imports are lazy so
    this module loads on branches that only carry a subset of the unpackers.
    """
    os.makedirs(out_dir, exist_ok=True)
    if fmt == "nvfp4":
        from amct_pytorch.common.utils.decode.unpack_nvfp4 import main as run
        run(["--model", model_dir, "--output_dir", out_dir])
    elif fmt == "mxfp4":
        from amct_pytorch.common.utils.decode.unpack_mxfp4 import main as run
        run(["convert-dir", "--mxfp4-dir", model_dir, "--out-dir", out_dir])
    elif fmt == "mxfp8":
        from amct_pytorch.common.utils.decode.unpack_mxfp8 import main as run
        run(["--quant_model_dir", model_dir, "--output_dir", out_dir])
    else:
        raise ValueError(f"decode_to_bf16: {fmt!r} is not a foreign format {FOREIGN_FORMATS}")
    copy_side_files(model_dir, out_dir)
    return out_dir
