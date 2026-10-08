# -*- coding: UTF-8 -*-
"""Prefetch benchmark datasets to local cache ($BENCH_ASSETS) via snapshot_download.

snapshot_download is robust (retries, per-file) where datasets' remote glob times
out on this box's mirror. Run once per benchmark before offline eval. The dataset
registry lives in datasets.json (shared with bench_offline.py) via config.py.
"""

import argparse
import os
import sys
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BENCH_ASSETS, HF_ENDPOINT, load_dataset_registry

os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def main():
    reg = load_dataset_registry()
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(reg))
    a = ap.parse_args()
    from huggingface_hub import snapshot_download

    spec = reg[a.task]
    dst = os.path.join(BENCH_ASSETS, spec["subdir"])
    allow_patterns = spec.get(
        "allow_patterns", ["*.parquet", "*.json", "*.jsonl", "*.md"]
    )
    p = snapshot_download(
        repo_id=spec["repo_id"],
        repo_type="dataset",
        local_dir=dst,
        allow_patterns=allow_patterns,
    )
    print(f"[prefetch] {spec['repo_id']} -> {p}")
    for split, rels in spec["splits"].items():
        if isinstance(rels, str):
            rels = [rels]
        found = []
        for rel in rels:
            found.extend(glob(os.path.join(dst, rel), recursive=True))
        status = f"OK ({len(found)} files)" if found else "MISSING"
        print(f"   {split}: {rels} {status}")


if __name__ == "__main__":
    main()
