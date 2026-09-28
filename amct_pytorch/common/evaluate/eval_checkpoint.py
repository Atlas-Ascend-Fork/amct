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

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import shutil
from pathlib import Path

import torch
from loguru import logger

from amct_pytorch.common.utils.safe_load import safe_torch_load
from amct_pytorch.quantization.bit_policy import ensure_bit_policy

_LINEAR_TARGETS = ("attn-linear", "mlp", "moe")
_CACHE_KEYS = ("q", "k", "p", "v")
_PTQ_PARAM_DIRS = {
    "attn-linear": "attn_linear_param_dir",
    "attn-cache": "attn_cache_param_dir",
    "mlp": "moe_mlp_param_dir",
    "moe": "moe_mlp_param_dir",
}


def _normalized_model_path(model: str) -> str:
    path = Path(model).expanduser()
    if path.exists():
        return str(path.resolve())
    return model


def _directory_fingerprint(raw_path: str):
    if not raw_path:
        return None

    path = Path(raw_path).expanduser()
    normalized_path = str(path.resolve())
    if not path.is_dir():
        return {"path": normalized_path, "exists": False}

    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative_path = file_path.relative_to(path).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as file_obj:
            while chunk := file_obj.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return {
        "path": normalized_path,
        "exists": True,
        "sha256": digest.hexdigest(),
    }


def _effective_bit_config(args) -> dict:
    if args.eval_mode == "bf16":
        return {"mode": "bf16"}

    policy = ensure_bit_policy(args)
    targets = sorted(set(args.quant_target))
    effective = {}
    if set(targets).intersection(_LINEAR_TARGETS):
        effective["w_bits"] = policy.w_bits
        effective["a_bits"] = policy.a_bits
        for target in _LINEAR_TARGETS:
            if target not in targets:
                continue
            group_config = policy.cfg.get(target)
            if isinstance(group_config, dict):
                effective[target] = group_config
    if "attn-cache" in targets:
        effective["attn-cache"] = {key: policy.cache_bits(key) for key in _CACHE_KEYS}
    return effective


def _bit_config_name(args) -> str:
    if args.eval_mode == "bf16":
        return "bf16"
    bit_config = getattr(args, "bit_config", None)
    return Path(bit_config).stem if bit_config else "default"


def build_eval_signature(args, num_layers: int) -> dict:
    """Build the normalized identity of a blockwise eval run."""
    targets = sorted(set(args.quant_target))
    is_bf16 = args.eval_mode == "bf16"
    ptq_param_dirs = {}
    if not is_bf16:
        for target in targets:
            arg_name = _PTQ_PARAM_DIRS[target]
            ptq_param_dirs[target] = _directory_fingerprint(getattr(args, arg_name, ""))

    return {
        "model": _normalized_model_path(args.model),
        "model_name": args.model_name,
        "eval_mode": args.eval_mode,
        "quant_dtype": "bf16" if is_bf16 else (args.quant_dtype or "none"),
        "quant_target": [] if is_bf16 else targets,
        "algos": [] if is_bf16 else list(getattr(args, "algos", [])),
        "seq_len": args.seq_len,
        "num_layers": num_layers,
        "dataset": "Salesforce/wikitext:wikitext-2-raw-v1:test",
        "bit_config_name": _bit_config_name(args),
        "effective_bit_config": _effective_bit_config(args),
        "ptq_param_dirs": ptq_param_dirs,
    }


def _signature_hash(signature: dict) -> str:
    serialized = json.dumps(
        signature,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:8]


def _slug(value: str, max_length: int = 48) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return (normalized or "none")[:max_length]


def _run_name(signature: dict) -> str:
    targets = "-".join(signature["quant_target"]) or "none"
    if signature["eval_mode"] == "bf16":
        algorithm = "none"
    else:
        algorithm = "-".join(signature["algos"]) or "direct"
    fields = (
        signature["model_name"],
        signature["eval_mode"],
        signature["quant_dtype"],
        targets,
        algorithm,
        signature["bit_config_name"],
        _signature_hash(signature),
    )
    return "__".join(_slug(str(field)) for field in fields)


class EvalCheckpointManager:
    """Save and restore the latest completed block output for eval."""

    def __init__(self, args, num_layers: int):
        self.num_layers = num_layers
        self.signature = build_eval_signature(args, num_layers)
        self.run_dir = (
            Path(args.output_dir) / "temp" / "eval" / _run_name(self.signature)
        )
        self.checkpoint_path = self.run_dir / "checkpoint.pt"
        self.temporary_path = self.run_dir / "checkpoint.pt.tmp"

    def save(self, completed_block_idx: int, inter_io):
        if not 0 <= completed_block_idx < self.num_layers:
            raise ValueError(
                f"completed block index {completed_block_idx} is outside "
                f"[0, {self.num_layers})."
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.temporary_path.exists():
            self.temporary_path.unlink()
        payload = {
            "signature": self.signature,
            "completed_block_idx": completed_block_idx,
            "num_layers": self.num_layers,
            "inter_io": inter_io,
        }
        with self.temporary_path.open("wb") as file_obj:
            torch.save(payload, file_obj)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(self.temporary_path, self.checkpoint_path)

    def load(self):
        if self.temporary_path.exists():
            self.temporary_path.unlink()
        if not self.checkpoint_path.exists():
            return None

        try:
            payload = safe_torch_load(self.checkpoint_path, map_location="cpu")
        except (EOFError, pickle.UnpicklingError, RuntimeError) as error:
            logger.warning(
                "Discard corrupt eval checkpoint '{}': {}",
                self.checkpoint_path,
                error,
            )
            self.cleanup()
            return None

        if not isinstance(payload, dict) or payload.get("signature") is None:
            logger.warning(
                "Discard invalid eval checkpoint '{}'.", self.checkpoint_path
            )
            self.cleanup()
            return None
        if payload["signature"] != self.signature:
            raise ValueError(
                f"Eval checkpoint signature mismatch at '{self.checkpoint_path}'."
            )

        completed_block_idx = payload.get("completed_block_idx")
        inter_io = payload.get("inter_io")
        if (
            type(completed_block_idx) is not int
            or not 0 <= completed_block_idx < self.num_layers
            or payload.get("num_layers") != self.num_layers
            or not isinstance(inter_io, (list, tuple))
            or not all(isinstance(item, torch.Tensor) for item in inter_io)
        ):
            logger.warning(
                "Discard invalid eval checkpoint '{}'.", self.checkpoint_path
            )
            self.cleanup()
            return None
        return completed_block_idx, inter_io

    def cleanup(self):
        if self.run_dir.exists():
            shutil.rmtree(self.run_dir)
