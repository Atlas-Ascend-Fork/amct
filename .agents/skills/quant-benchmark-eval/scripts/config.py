# -*- coding: UTF-8 -*-
"""Shared config for the quant-benchmark-eval skill.

Portability: this skill targets other Ascend+amct machines. Machine-specific
locations are env vars with sensible defaults for the original box; override them
on a new machine. amct / torch_npu / lm-eval remain hard dependencies.

Env vars:
  AMCT_HOME     amct source root (for default bit_config).  default: /mnt/workspace/gitCode/yaoguangxiu/amct
  BENCH_ASSETS  local dataset cache root.                    default: /home/developer/bench_assets
  HF_ENDPOINT   HF mirror for prefetch.                      default: https://hf-mirror.com
"""
import json
import os
from glob import glob

AMCT_HOME = os.environ.get("AMCT_HOME", "/mnt/workspace/gitCode/yaoguangxiu/amct")
BENCH_ASSETS = os.environ.get("BENCH_ASSETS", "/home/developer/bench_assets")
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

DEFAULT_BIT_CONFIG = os.path.join(AMCT_HOME, "amct_pytorch", "configs", "w8a8.yaml")

_DATASETS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets.json")


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
