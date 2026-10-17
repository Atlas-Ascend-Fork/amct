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
"""Shared config for the quant-benchmark-eval skill.

Portability: this skill ships inside the amct repo and targets any Ascend+amct
machine, so nothing here hardcodes one box's paths. Locations are derived from
this file, or taken from env vars when the caller wants to override them.
amct / torch_npu / lm-eval remain hard dependencies.

Env vars (all optional):
  AMCT_HOME     amct source root (for default bit_config).
                default: derived by walking up from this file to the directory
                that holds both `amct_pytorch/` and `setup.py`.
  BENCH_ASSETS  local dataset cache root.
                default: $XDG_CACHE_HOME/amct-bench-assets, else
                ~/.cache/amct-bench-assets.
  HF_ENDPOINT   HF mirror for prefetch.  default: https://hf-mirror.com
"""

import json
import os
from glob import glob

# Characters that make a registry entry a glob candidate instead of a literal path.
_GLOB_MAGIC = "*?["


def _default_amct_home():
    """Locate the amct repo root from this file; no machine-specific defaults.

    Walks up from `scripts/` until a directory holds both `amct_pytorch/` and
    `setup.py`. Returns "" when not found so callers fail fast with a clear
    message instead of silently using an unrelated path.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    cand = here
    while True:
        if os.path.isdir(os.path.join(cand, "amct_pytorch")) and os.path.isfile(
            os.path.join(cand, "setup.py")
        ):
            return cand
        parent = os.path.dirname(cand)
        if parent == cand:
            return ""
        cand = parent


def _default_bench_assets():
    """Per-user cache dir for benchmark datasets (XDG-friendly)."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return os.path.join(base, "amct-bench-assets")


AMCT_HOME = os.environ.get("AMCT_HOME") or _default_amct_home()
BENCH_ASSETS = os.environ.get("BENCH_ASSETS") or _default_bench_assets()
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

DEFAULT_BIT_CONFIG = os.path.join(AMCT_HOME, "amct_pytorch", "configs", "w8a8.yaml")


def require_amct_home():
    """Return AMCT_HOME, or fail fast with an actionable message.

    Derivation only fails when the skill is used outside an amct checkout; in
    that case a relative `amct_pytorch/configs/w8a8.yaml` would surface later as
    a confusing file-not-found, so report it here instead.
    """
    if not AMCT_HOME:
        raise RuntimeError(
            "cannot locate the amct repo root (no directory with both "
            "amct_pytorch/ and setup.py above this skill). Set AMCT_HOME to "
            "your amct checkout, or pass --bit_config explicitly."
        )
    return AMCT_HOME


_DATASETS_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "datasets.json"
)


def load_dataset_registry():
    """Return {task: {repo_id, subdir, splits{split: relpath}}} from datasets.json."""
    with open(_DATASETS_JSON, "r", encoding="utf-8") as f:
        reg = json.load(f)
    return {k: v for k, v in reg.items() if not k.startswith("_")}


def split_declaration(rels):
    """Normalize one split's declaration to `(mode, [relpath, ...])`.

    Accepted forms:

    - `"test.jsonl"` / `["a.jsonl", "b.jsonl"]` -> `("all_of", [...])`, the
      historical form: every entry is a required shard (or a glob whose matches
      are all shards of one layout), all merged.
    - `{"any_of": [...]}` -> alternative layouts; first hit wins.
    - `{"all_of": [...]}` -> same as the bare list, spelled out.
    """
    if isinstance(rels, str):
        return "all_of", [rels]
    if isinstance(rels, dict):
        modes = [mode for mode in ("any_of", "all_of") if mode in rels]
        if len(modes) != 1:
            raise ValueError(
                "split declaration must carry exactly one of 'any_of' / 'all_of', "
                f"got keys {sorted(rels)}"
            )
        value = rels[modes[0]]
        return modes[0], [value] if isinstance(value, str) else list(value)
    return "all_of", list(rels)


def _expand(base, rel):
    """Expand one declared path/glob under `base` (deduped, sorted)."""
    if any(ch in rel for ch in _GLOB_MAGIC):
        return list(
            dict.fromkeys(sorted(glob(os.path.join(base, rel), recursive=True)))
        )
    return [os.path.join(base, rel)]


def resolve_split_files(base, rels):
    """Resolve one split's declared paths/globs under `base`.

    Two declaration modes, because a split's paths can mean opposite things:

    - ``any_of`` — alternative layouts of the *same* data (`test.jsonl` **or**
      ``data/test-*.parquet``); the first candidate that exists wins, so a cache
      holding both layouts is not read twice (samples would silently double).
    - ``all_of`` — shards that must all be merged (LiveCodeBench's cumulative
      `test.jsonl` … `test6.jsonl`).

    A bare string / list keeps the historical ``all_of`` meaning. Globs are
    expanded; when nothing exists the declared paths come back verbatim so the
    caller's error can name them.
    """
    mode, rel_list = split_declaration(rels)
    if mode == "any_of":
        declared = []
        for rel in rel_list:
            matches = _expand(base, rel)
            declared.extend(matches)
            existing = [path for path in matches if os.path.exists(path)]
            if existing:
                return existing
        return list(dict.fromkeys(declared))
    deduped = []
    for rel in rel_list:
        deduped.extend(_expand(base, rel))
    deduped = list(dict.fromkeys(deduped))
    existing = [path for path in deduped if os.path.exists(path)]
    return existing or deduped


def local_dataset_map():
    """Return {(repo_id, dataset_name): {split: [absolute paths]}} for offline shim.

    The key is what lm-eval hands to `datasets.load_dataset(path=..., name=...)`,
    so it is *not* per task: several tasks may legitimately share one repo
    (datasets.json already has livecodebench / livecodebench_release_v6 on
    livecodebench/code_generation_lite, and arc_challenge / arc_easy / boolq /
    copa on two repos) as long as their `dataset_name` tells them apart. Two
    tasks that share the key but declare different local data are
    contradictory -- one of them would silently win the dict slot and the shim
    would serve the wrong files to the other, scoring data that is not the one
    the registry claims (constraints.md #2). Raise instead of overwriting;
    identical declarations are a harmless duplicate and stay allowed.
    """
    reg = load_dataset_registry()
    out = {}
    owner = {}
    for task, spec in reg.items():
        base = os.path.join(BENCH_ASSETS, spec["subdir"])
        key = (spec["repo_id"], spec.get("dataset_name"))
        entry = {
            split: resolve_split_files(base, rel)
            for split, rel in spec["splits"].items()
        }
        if key in out:
            prev_task = owner[key]
            prev_base = os.path.join(BENCH_ASSETS, reg[prev_task]["subdir"])
            if prev_base != base or out[key] != entry:
                raise ValueError(
                    f"datasets.json tasks '{prev_task}' and '{task}' both map to "
                    f"(repo_id, dataset_name)={key}, but declare different local "
                    f"data ({reg[prev_task]['subdir']!r} vs {spec['subdir']!r}). "
                    "The offline shim keys by that pair, so one task would "
                    "silently read the other's files. Give the second task a "
                    "distinct dataset_name (the value lm-eval passes as "
                    "`name=`), or point both at the same subdir."
                )
            continue
        out[key] = entry
        owner[key] = task
    return out
