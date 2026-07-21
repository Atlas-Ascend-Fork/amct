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
"""Generic checkpoint I/O layer — format-agnostic shard traversal + file helpers.

This layer has **no knowledge** of FP4 formats or dequant strategies.
All format-specific logic lives in the ``convert_*.py`` strategy callbacks.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file, save_file

try:
    import numpy as np  # noqa: F401
except ImportError:  # pragma: no cover
    np = None


# ══════════════════════════════════════════════════════════════════════
# 1. Index helpers
# ══════════════════════════════════════════════════════════════════════


def load_index(index_path: str) -> dict:
    """Read a ``model.safetensors.index.json`` file."""
    with open(index_path, "r") as f:
        return json.load(f)


def invert_weight_map(weight_map: Dict[str, str]) -> Dict[str, List[str]]:
    """Invert ``{tensor_key → shard_filename}`` to ``{shard_filename → [tensor_keys]}``."""
    by_file: Dict[str, List[str]] = defaultdict(list)
    for k, fname in weight_map.items():
        by_file[fname].append(k)
    return dict(by_file)


def write_index(
    out_dir: str,
    weight_map: Dict[str, str],
    metadata: Optional[dict] = None,
    index_name: str = "model.safetensors.index.json",
) -> None:
    """Write a safetensors index.json to *out_dir*."""
    index: Dict[str, Any] = {"weight_map": weight_map}
    if metadata is not None:
        index["metadata"] = metadata

    if metadata is None or "total_size" not in (metadata or {}):
        total_size = 0
        for fname in set(weight_map.values()):
            path = os.path.join(out_dir, fname)
            if os.path.exists(path):
                sd = load_file(path, device="cpu")
                total_size += sum(
                    int(t.numel() * t.element_size())
                    for t in sd.values()
                    if isinstance(t, torch.Tensor)
                )
        index.setdefault("metadata", {})["total_size"] = total_size

    out_path = os.path.join(out_dir, index_name)
    with open(out_path, "w") as f:
        json.dump(index, f)


# ══════════════════════════════════════════════════════════════════════
# 2. Shard cache
# ══════════════════════════════════════════════════════════════════════


class ShardCache:
    """LRU-like cache for safetensors shards with cross-shard key lookup."""

    def __init__(self, shard_paths: List[str]) -> None:
        self._paths = list(shard_paths)
        self._cache: Dict[str, dict] = {}

    def get(self, path: str) -> dict:
        if path not in self._cache:
            self._cache[path] = load_file(path, device="cpu")
        return self._cache[path]

    def find_key(self, key: str) -> Tuple[Optional[torch.Tensor], Optional[str]]:
        for p in self._paths:
            d = self.get(p)
            if key in d:
                return d[key], p
        return None, None


# ══════════════════════════════════════════════════════════════════════
# 3. Generic tensor loader
# ══════════════════════════════════════════════════════════════════════


def parse_spec(spec: str) -> Tuple[str, Optional[str]]:
    """Parse a ``path[:key]`` spec string."""
    if ":" in spec:
        path, key = spec.split(":", 1)
        return path, key
    return spec, None


def load_tensor_from_spec(spec: str, map_location: str = "cpu") -> torch.Tensor:
    """Load a single tensor from .pt / .pth / .npy / .safetensors:key."""
    path, key = parse_spec(spec)
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
        if np is None:
            raise ImportError("numpy is required to load .npy files")
        if key is not None:
            raise ValueError(".npy spec should not include a key")
        return torch.from_numpy(np.load(path))

    if lower.endswith(".safetensors"):
        if key is None:
            raise ValueError("safetensors spec must be path.safetensors:key")
        return load_file(path, device=map_location)[key]

    raise ValueError(f"unsupported input file type: {path}")


# ══════════════════════════════════════════════════════════════════════
# 4. Shard write helpers
# ══════════════════════════════════════════════════════════════════════


def save_shard(out_dir: str, fname: str, tensors: Dict[str, torch.Tensor]) -> str:
    """Save a dict of tensors as a safetensors shard."""
    out_path = os.path.join(out_dir, fname)
    os.makedirs(os.path.dirname(out_path) or out_dir, exist_ok=True)
    save_file(tensors, out_path)
    return out_path


def save_convert_output(path: str, payload: Dict[str, Any]) -> None:
    """Save conversion output as a .pt file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save(payload, path)


# ══════════════════════════════════════════════════════════════════════
# 5. Main traversal loop
# ══════════════════════════════════════════════════════════════════════


def walk_checkpoint(
    src_dir: str,
    index_path: str,
    out_dir: str,
    transform_fn: Callable[
        [str, torch.Tensor, Callable[[str], Tuple[Optional[torch.Tensor], Optional[str]]]],
        Optional[torch.Tensor],
    ],
    *,
    extra_shard_globs: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, int]:
    """Walk every tensor in a safetensors checkpoint, transform, and save.

    Args:
        src_dir: Source safetensors directory.
        index_path: Path to BF16 ``model.safetensors.index.json`` (target sharding plan).
        out_dir: Output directory.
        transform_fn: ``fn(key, tensor, lookup) -> Optional[Tensor]``.
            Return ``None`` to drop the key.
        extra_shard_globs: Additional source globs for ShardCache.
        verbose: Print per-shard progress.

    Returns:
        Summary dict: ``{converted, copied, dropped, missing}``.
    """
    import glob as glob_mod

    out_path_obj = Path(out_dir)
    out_path_obj.mkdir(parents=True, exist_ok=True)

    bf16_index = load_index(index_path)
    if "weight_map" not in bf16_index:
        raise ValueError("index.json must contain 'weight_map'")
    weight_map: Dict[str, str] = bf16_index["weight_map"]
    by_file = invert_weight_map(weight_map)

    source_shards = sorted(glob_mod.glob(os.path.join(src_dir, "*.safetensors")))
    if extra_shard_globs:
        for g in extra_shard_globs:
            source_shards.extend(sorted(glob_mod.glob(g)))
    if not source_shards:
        raise FileNotFoundError(f"No *.safetensors found in {src_dir}")
    cache = ShardCache(source_shards)

    stats: Dict[str, int] = {"converted": 0, "copied": 0, "dropped": 0, "missing": 0}
    actual_weight_map: Dict[str, str] = {}

    for shard_fname, keys in by_file.items():
        shard_out: Dict[str, torch.Tensor] = {}
        if verbose:
            print(f"\n[Shard] {shard_fname}: {len(keys)} tensors")

        for k in keys:
            tensor, src_path = cache.find_key(k)
            if tensor is None:
                stats["missing"] += 1
                if verbose:
                    print(f"  [MISS] {k}")
                continue

            def _lookup(lk: str) -> Tuple[Optional[torch.Tensor], Optional[str]]:
                return cache.find_key(lk)

            result = transform_fn(k, tensor, _lookup)
            if result is None:
                stats["dropped"] += 1
                continue

            shard_out[k] = result.contiguous()
            if result is not tensor:
                stats["converted"] += 1
            else:
                stats["copied"] += 1

        out_path = save_shard(str(out_path_obj), shard_fname, shard_out)
        if verbose:
            print(f"  [Saved] {out_path}  tensors={len(shard_out)}")

        for written_key in shard_out:
            actual_weight_map[written_key] = shard_fname

    new_metadata = dict(bf16_index.get("metadata", {}))
    write_index(str(out_path_obj), actual_weight_map, new_metadata)
    if verbose:
        print(f"\n[OK] wrote index: {out_path_obj / 'model.safetensors.index.json'}")

    if verbose:
        print(
            f"\nSummary: converted={stats['converted']}, copied={stats['copied']}, "
            f"dropped={stats['dropped']}, missing={stats['missing']}"
        )
        if stats["missing"] > 0:
            print("NOTE: missing>0 — bf16 index keys may not match source checkpoint keys.")

    return stats
