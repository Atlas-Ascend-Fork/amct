# -*- coding: UTF-8 -*-
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


def _resolve_split_files(base, rels):
    if isinstance(rels, str):
        rels = [rels]
    out = []
    for rel in rels:
        matches = sorted(glob(os.path.join(base, rel), recursive=True))
        if matches:
            out.extend(matches)
        else:
            out.append(os.path.join(base, rel))
    return out


def local_dataset_map():
    """Return {(repo_id, dataset_name): {split: [absolute paths]}} for offline shim."""
    reg = load_dataset_registry()
    out = {}
    for task, spec in reg.items():
        base = os.path.join(BENCH_ASSETS, spec["subdir"])
        key = (spec["repo_id"], spec.get("dataset_name"))
        out[key] = {
            split: _resolve_split_files(base, rel)
            for split, rel in spec["splits"].items()
        }
    return out


def local_parquet_map():
    """Backward-compatible alias for local_dataset_map()."""
    return local_dataset_map()
