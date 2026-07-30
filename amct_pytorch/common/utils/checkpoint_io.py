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
"""Generic checkpoint shard I/O -- format-agnostic safetensors traversal + write helpers.

Single home for reading/writing sharded safetensors checkpoints (``index.json`` + shards).
Shared by:
  - deploy (``LlmDeployWorkflow``, real pack) and ptq convert (fake-quant) via
    :class:`CheckpointShardWriter`;
  - the HiF4 ptq unpack (``decode.unpack_hif4``) via the free functions;
  - the ``decode`` foreign-format converters (folded in downstream).

No knowledge of any quant format lives here -- only index parsing, shard read/write, and
support-file copying. Deliberately depends on nothing beyond ``torch`` + ``safetensors`` so
lightweight callers (e.g. ``decode.unpack_hif4``) do not pull in ``transformers``.
"""
from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

# Stable safetensors checkpoint filenames (mirror ``transformers.utils`` constants; hardcoded
# so this module -- and its lightweight callers -- need no transformers dependency).
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
SAFE_WEIGHTS_NAME = "model.safetensors"

# Known text/side files copied verbatim next to converted weights (config + tokenizer assets).
SIDE_FILES = (
    "config.json", "generation_config.json", "tokenizer_config.json",
    "special_tokens_map.json", "tokenizer.json", "tokenizer.model",
    "vocab.json", "merges.txt", "added_tokens.json", "chat_template.jinja",
)

_MAX_SHARD_SIZE = 8 * 1024 ** 3       # 8 GiB per "rest_*.safetensors" shard


# --------------------------------------------------------------------------- index
def load_index(index_path: str) -> dict:
    """Read a ``model.safetensors.index.json`` file."""
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_index(model_dir: str, explicit_index_path: Optional[str] = None) -> dict:
    """Return a ``{'metadata', 'weight_map'}`` index for a checkpoint directory.

    Precedence: (1) an explicit index path; (2) the checkpoint's own sharded
    ``model.safetensors.index.json``; (3) a synthesized single-file index from
    ``model.safetensors``. This lets any caller derive its traversal/output plan from the
    *source* checkpoint itself, so no separate reference index is required. The synthesized
    ``total_size`` is a placeholder -- writers recompute it from the actual output shards.
    """
    if explicit_index_path:
        return load_index(explicit_index_path)
    own = os.path.join(model_dir, SAFE_WEIGHTS_INDEX_NAME)
    if os.path.exists(own):
        return load_index(own)
    single = os.path.join(model_dir, SAFE_WEIGHTS_NAME)
    if os.path.exists(single):
        with safe_open(single, framework="pt") as f:
            weight_map = {key: SAFE_WEIGHTS_NAME for key in f.keys()}
        return {"metadata": {"total_size": os.path.getsize(single)}, "weight_map": weight_map}
    raise FileNotFoundError(
        f"Neither {SAFE_WEIGHTS_INDEX_NAME} nor {SAFE_WEIGHTS_NAME} found in {model_dir}"
    )


def invert_weight_map(weight_map: Dict[str, str]) -> Dict[str, List[str]]:
    """Invert ``{tensor_key -> shard_file}`` to ``{shard_file -> [tensor_keys]}``."""
    by_file: Dict[str, List[str]] = defaultdict(list)
    for key, shard in weight_map.items():
        by_file[shard].append(key)
    return dict(by_file)


def write_index(
    out_dir: str,
    weight_map: Dict[str, str],
    *,
    metadata: Optional[dict] = None,
    total_size: Optional[int] = None,
    sort_keys: bool = False,
) -> str:
    """Write a safetensors ``index.json`` to *out_dir*, returning its path.

    ``total_size`` is used as-is when given; otherwise it is recomputed from the on-disk sizes
    of the referenced output shards (fast, no tensor load).
    """
    meta = dict(metadata or {})
    if total_size is None:
        total_size = sum(
            os.path.getsize(os.path.join(out_dir, fname))
            for fname in sorted(set(weight_map.values()))
        )
    meta["total_size"] = total_size
    index = {"metadata": meta, "weight_map": weight_map}
    index_path = os.path.join(out_dir, SAFE_WEIGHTS_INDEX_NAME)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2, sort_keys=sort_keys)
    return index_path


# ----------------------------------------------------------------------- side files
def copy_side_files(src_dir: str, out_dir: str, skip=()) -> None:
    """Copy the known text/config :data:`SIDE_FILES` (whitelist) from *src_dir* to *out_dir*."""
    os.makedirs(out_dir, exist_ok=True)
    for fn in SIDE_FILES:
        if fn in skip:
            continue
        src = os.path.join(src_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, fn))


def is_weight_file(path: Path) -> bool:
    """True for a safetensors shard or its index (i.e. a file a writer will regenerate)."""
    return path.name == SAFE_WEIGHTS_INDEX_NAME or path.suffix == ".safetensors"


def copy_support_files(src_dir: str, out_dir: str) -> None:
    """Copy every non-weight, non-hidden entry (files *and* dirs) from *src_dir* to *out_dir*.

    Weight shards / index are skipped (the writer regenerates them); existing destinations are
    left untouched. Use this when the output must preserve arbitrary sidecars, not just the
    known :data:`SIDE_FILES`.
    """
    src = Path(src_dir)
    dst = Path(out_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for src_path in src.iterdir():
        if src_path.name.startswith("."):
            continue
        if is_weight_file(src_path):
            continue
        dst_path = dst / src_path.name
        if dst_path.exists():
            continue
        if src_path.is_dir():
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)


# --------------------------------------------------------------------------- shards
def save_shard(out_dir: str, file_name: str, tensors: Dict[str, object], atomic: bool = True) -> Optional[str]:
    """Save a dict of tensors as one safetensors shard (skips empty). Atomic by default."""
    if not tensors:
        return None
    out_path = Path(out_dir) / file_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if atomic:
        tmp_path = out_path.parent / f".{out_path.name}.tmp"
        save_file(tensors, str(tmp_path))
        os.replace(tmp_path, out_path)
    else:
        save_file(tensors, str(out_path))
    return str(out_path)


class CheckpointShardWriter:
    """Stateful writer for block/tensor checkpoint conversion flows.

    Holds the ``(model_path, output_dir)`` pair and encapsulates the shard-level write
    machinery shared by deploy (real pack) and ptq convert (fake-quant): copy support files,
    resolve the source index, write per-layer shards, copy through the remaining (un-converted)
    original weights, and refresh the output index. The per-layer *content* (pack vs fake-quant)
    is decided by the caller; this class only writes what it is handed.
    """

    def __init__(self, model_path: str, output_dir: str):
        self.model_path = model_path
        self.output_dir = output_dir

    @staticmethod
    def collect_replaced_original_weights(
        layer_tensors: Dict[str, object],
        tensor_routes: Dict[str, str],
        original_weight_map: Dict[str, str],
    ) -> set:
        """Original weight names that a layer's output tensors replace (so they aren't copied)."""
        replaced = set()
        for weight_name in layer_tensors:
            base = tensor_routes.get(weight_name, weight_name)
            if base in original_weight_map:
                replaced.add(base)
        return replaced

    def copy_support_files(self) -> None:
        copy_support_files(self.model_path, self.output_dir)

    def load_weight_index(self) -> dict:
        """Load the source checkpoint's index (sharded or synthesized single-file)."""
        return resolve_index(self.model_path)

    def refresh_weight_index(self, original_index: dict, updated_weight_map: Dict[str, str]) -> str:
        """Write the output index, carrying original metadata and recomputing ``total_size``."""
        return write_index(
            self.output_dir,
            updated_weight_map,
            metadata=original_index.get("metadata", {}),
            sort_keys=True,
        )

    def write_safetensor_file(self, file_name: str, tensors: Dict[str, object]) -> None:
        save_shard(self.output_dir, file_name, tensors, atomic=True)

    def write_block_file(self, layer_idx: int, num_layers: int, layer_tensors: Dict[str, object]) -> Dict[str, str]:
        """Write one layer's tensors to ``layer_<idx>.safetensors``; return their weight-map entries."""
        width = max(3, len(str(max(num_layers - 1, 0))))
        file_name = f"layer_{layer_idx:0{width}d}.safetensors"
        self.write_safetensor_file(file_name, layer_tensors)
        return {weight_name: file_name for weight_name in layer_tensors}

    def write_remaining_original_weights(
        self,
        original_weight_map: Dict[str, str],
        replaced_original_weights: set,
    ) -> Dict[str, str]:
        """Copy through every original weight not replaced by a converted one, re-sharded to 8 GiB."""
        remaining_by_file = defaultdict(list)
        for weight_name, file_name in original_weight_map.items():
            if weight_name in replaced_original_weights:
                continue
            remaining_by_file[file_name].append(weight_name)

        rest_idx = 0
        current_tensors: Dict[str, object] = {}
        current_size = 0
        updated_entries: Dict[str, str] = {}

        def flush_current():
            nonlocal rest_idx, current_tensors, current_size
            if not current_tensors:
                return
            file_name = f"rest_{rest_idx:05d}.safetensors"
            self.write_safetensor_file(file_name, current_tensors)
            for weight_name in current_tensors:
                updated_entries[weight_name] = file_name
            rest_idx += 1
            current_tensors = {}
            current_size = 0

        model_dir = Path(self.model_path)
        for source_file in sorted(remaining_by_file):
            source_path = model_dir / source_file
            with safe_open(str(source_path), framework="pt", device="cpu") as f:
                for weight_name in remaining_by_file[source_file]:
                    tensor = f.get_tensor(weight_name)
                    tensor_size = tensor.numel() * tensor.element_size()
                    if current_tensors and current_size + tensor_size > _MAX_SHARD_SIZE:
                        flush_current()
                    current_tensors[weight_name] = tensor
                    current_size += tensor_size

        flush_current()
        return updated_entries


def export_blockwise(model_path, output_dir, num_blocks, export_block_fn, refresh_config_fn):
    """Assemble an output checkpoint block-by-block, preserving un-replaced original weights.

    Format/model-agnostic checkpoint assembly: iterate ``num_blocks`` blocks, get each block's
    tensors from ``export_block_fn(block_idx, quant_ignore_layers)`` -> ``(tensors, routes)``, write
    them via :class:`CheckpointShardWriter`, then write the remaining original weights and refresh
    the index; ``refresh_config_fn(quant_ignore_layers)`` rewrites config.json. Callers (deploy pack
    / ptq convert) supply the two callbacks. ``tqdm``/``loguru`` are imported lazily so importers
    that only need the shard primitives (e.g. ``decode.unpack_hif4``) stay lightweight.
    """
    from loguru import logger
    from tqdm import tqdm

    writer = CheckpointShardWriter(model_path, output_dir)
    writer.copy_support_files()
    quant_ignore_layers = []
    original_index = writer.load_weight_index()
    original_weight_map = dict(original_index.get("weight_map", {}))
    updated_weight_map = {}
    replaced_original_weights = set()
    for block_idx in tqdm(range(num_blocks), desc="Block Processing..."):
        layer_tensors, tensor_routes = export_block_fn(block_idx, quant_ignore_layers)
        if not layer_tensors:
            continue
        updated_weight_map.update(
            writer.write_block_file(block_idx, num_blocks, layer_tensors)
        )
        replaced_original_weights.update(
            writer.collect_replaced_original_weights(
                layer_tensors, tensor_routes, original_weight_map,
            )
        )
    updated_weight_map.update(
        writer.write_remaining_original_weights(original_weight_map, replaced_original_weights)
    )
    index_path = writer.refresh_weight_index(original_index, updated_weight_map)
    refresh_config_fn(quant_ignore_layers)
    logger.info("Exported model to {}", output_dir)
    logger.info("Refreshed weight index at {}", index_path)
    return {
        "index_path": index_path,
        "num_output_files": len(set(updated_weight_map.values())),
    }
