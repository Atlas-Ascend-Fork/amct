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

Provides the "warehouse" for both NVFP4 and MXFP4 pipelines:
  - Read / write safetensors index.json.
  - Cached shard loading with cross-shard key lookup (``ShardCache``).
  - Tensor loading from various file formats (.pt/.npy/.safetensors).
  - ``walk_checkpoint`` — the main loop that iterates every tensor in a
    checkpoint and delegates per-tensor decisions to a callback.

This layer has **no knowledge** of FP4 formats, dequant strategies, or
which keys to drop. All format-specific logic lives in the ``convert_*.py``
strategy callbacks.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file, save_file

# ──────────────────────────────────────────────────────────────────────
# optional numpy import (only needed for .npy loading)
# ──────────────────────────────────────────────────────────────────────
try:
    import numpy as np  # noqa: F401
except ImportError:  # pragma: no cover
    np = None


# ══════════════════════════════════════════════════════════════════════
# 1. Index helpers
# ══════════════════════════════════════════════════════════════════════


def load_index(index_path: str) -> dict:
    """Read a ``model.safetensors.index.json`` file.

    Args:
        index_path: Path to the index JSON file.

    Returns:
        Parsed dict with ``weight_map`` and optional ``metadata``.
    """
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
    """Write a safetensors index.json to *out_dir*.

    Args:
        out_dir: Output directory.
        weight_map: ``{tensor_key → shard_filename}`` mapping.
        metadata: Optional metadata dict (e.g. with ``total_size``).
        index_name: Filename for the index (default: standard safetensors name).
    """
    index: Dict[str, Any] = {"weight_map": weight_map}
    if metadata is not None:
        index["metadata"] = metadata

    # Compute total_size from actual shards if not already provided.
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
    """LRU-like cache for safetensors shards with cross-shard key lookup.

    Usage::

        cache = ShardCache(shard_paths)
        tensor, src_path = cache.find_key("model.layers.0.weight")
    """

    def __init__(self, shard_paths: List[str]) -> None:
        self._paths = list(shard_paths)
        self._cache: Dict[str, dict] = {}

    def get(self, path: str) -> dict:
        """Load (and cache) a single shard file."""
        if path not in self._cache:
            self._cache[path] = load_file(path, device="cpu")
        return self._cache[path]

    def find_key(self, key: str) -> Tuple[Optional[torch.Tensor], Optional[str]]:
        """Search all shards for *key*; return ``(tensor, shard_path)`` or ``(None, None)``."""
        for p in self._paths:
            d = self.get(p)
            if key in d:
                return d[key], p
        return None, None


# ══════════════════════════════════════════════════════════════════════
# 3. Generic tensor loader (multiple file formats)
# ══════════════════════════════════════════════════════════════════════


def parse_spec(spec: str) -> Tuple[str, Optional[str]]:
    """Parse a ``path[:key]`` spec string.

    Returns:
        ``(path, key_or_None)``.
    """
    if ":" in spec:
        path, key = spec.split(":", 1)
        return path, key
    return spec, None


def load_tensor_from_spec(
    spec: str, map_location: str = "cpu"
) -> torch.Tensor:
    """Load a single tensor from a spec string.

    Supported formats:
      - ``.pt`` / ``.pth`` — ``torch.load``; if *key* is given the loaded
        object must be a dict.
      - ``.npy`` — ``numpy.load`` (requires numpy).
      - ``.safetensors`` — ``safetensors.torch.load_file`` (requires *key*).

    Args:
        spec: ``path[:key]`` string.
        map_location: Device string for :func:`torch.load`.

    Returns:
        Loaded tensor.

    Raises:
        FileNotFoundError, ImportError, TypeError, ValueError.
    """
    path, key = parse_spec(spec)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    lower = path.lower()
    if lower.endswith(".pt") or lower.endswith(".pth"):
        obj = torch.load(path, map_location=map_location)
        if key is not None:
            if not isinstance(obj, dict):
                raise TypeError(
                    f"{path} is not a dict, cannot get key {key}"
                )
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
            raise ValueError(
                "safetensors spec must be path.safetensors:key"
            )
        return load_file(path, device=map_location)[key]

    raise ValueError(f"unsupported input file type: {path}")


# ══════════════════════════════════════════════════════════════════════
# 4. Shard write helpers
# ══════════════════════════════════════════════════════════════════════


def save_shard(
    out_dir: str, fname: str, tensors: Dict[str, torch.Tensor]
) -> str:
    """Save a dict of tensors as a safetensors shard.

    Args:
        out_dir: Output directory.
        fname: Shard filename (e.g. ``model-00001-of-00003.safetensors``).
        tensors: ``{key: tensor}`` dict.

    Returns:
        Full path to the saved file.
    """
    out_path = os.path.join(out_dir, fname)
    os.makedirs(os.path.dirname(out_path) or out_dir, exist_ok=True)
    save_file(tensors, out_path)
    return out_path


def save_convert_output(path: str, payload: Dict[str, Any]) -> None:
    """Save conversion output as a .pt file (for single-pair conversions)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save(payload, path)


# ══════════════════════════════════════════════════════════════════════
# 5. Main traversal loop
# ══════════════════════════════════════════════════════════════════════

# Callback signature:
#   transform_fn(key, tensor, lookup) → Optional[torch.Tensor]
#
#   - Return a tensor → write it to the output.
#   - Return None     → drop the key (e.g. quantization intermediates).
#
#   ``lookup(key)`` returns ``(tensor, shard_path)`` from any shard in the
#   source checkpoint.


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

    This is the **single shared traversal loop** for whole-checkpoint
    conversion. It reads the BF16 index.json as the target sharding plan,
    iterates through every tensor, calls *transform_fn* once per tensor,
    collects the results into output shards, and writes the new index.json.

    Args:
        src_dir: Directory containing source safetensors shards.
        index_path: Path to the BF16 ``model.safetensors.index.json``
            that defines the target sharding plan.
        out_dir: Output directory for converted shards and index.
        transform_fn: Per-tensor callback with signature
            ``fn(key, tensor, lookup) -> Optional[Tensor]``.
            *tensor* is loaded from *src_dir*; *lookup(key)* fetches
            any sibling tensor from any shard in *src_dir*.
            Return ``None`` to drop the key.
        extra_shard_globs: Additional source directories to include in
            the ``ShardCache`` (e.g. when scales live in a separate dir).
        verbose: Print per-shard progress.

    Returns:
        Summary dict with keys ``converted``, ``copied``, ``dropped``, ``missing``.
    """
    import glob as glob_mod

    out_path_obj = Path(out_dir)
    out_path_obj.mkdir(parents=True, exist_ok=True)

    # 1) read target index
    bf16_index = load_index(index_path)
    if "weight_map" not in bf16_index:
        raise ValueError(
            "index.json must contain 'weight_map'"
        )
    weight_map: Dict[str, str] = bf16_index["weight_map"]
    by_file = invert_weight_map(weight_map)

    # 2) collect source shards
    source_shards = sorted(
        glob_mod.glob(os.path.join(src_dir, "*.safetensors"))
    )
    if extra_shard_globs:
        for g in extra_shard_globs:
            source_shards.extend(sorted(glob_mod.glob(g)))
    if not source_shards:
        raise FileNotFoundError(
            f"No *.safetensors found in {src_dir}"
        )
    cache = ShardCache(source_shards)

    # 3) walk every shard → every tensor
    stats: Dict[str, int] = {
        "converted": 0,
        "copied": 0,
        "dropped": 0,
        "missing": 0,
    }
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

            # Call the strategy callback.
            # The lookup closure gives the callback access to sibling keys.
            def _lookup(lookup_key: str) -> Tuple[Optional[torch.Tensor], Optional[str]]:
                return cache.find_key(lookup_key)

            result = transform_fn(k, tensor, _lookup)

            if result is None:
                stats["dropped"] += 1
                continue

            shard_out[k] = result.contiguous()
            if result is not tensor:
                stats["converted"] += 1
            else:
                stats["copied"] += 1

        # 4) save shard
        out_path = save_shard(str(out_path_obj), shard_fname, shard_out)
        if verbose:
            print(f"  [Saved] {out_path}  tensors={len(shard_out)}")

        # track written keys
        for written_key in shard_out:
            actual_weight_map[written_key] = shard_fname

    # 5) write output index
    new_metadata = dict(bf16_index.get("metadata", {}))
    write_index(str(out_path_obj), actual_weight_map, new_metadata)
    if verbose:
        print(f"\n[OK] wrote index: {out_path_obj / 'model.safetensors.index.json'}")

    # 6) summary
    if verbose:
        print(
            f"\nSummary: converted={stats['converted']}, "
            f"copied={stats['copied']}, dropped={stats['dropped']}, "
            f"missing={stats['missing']}"
        )
        if stats["missing"] > 0:
            print(
                "NOTE: missing>0 means your bf16 index keys don't match "
                "source checkpoint key names."
            )

    return stats
