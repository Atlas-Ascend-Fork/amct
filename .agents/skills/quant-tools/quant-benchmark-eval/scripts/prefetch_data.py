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
from config import BENCH_ASSETS, HF_ENDPOINT, load_dataset_registry, split_declaration

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
    unresolved = []
    for split, rels in spec["splits"].items():
        mode, rel_list = split_declaration(rels)
        per_rel = []
        for rel in rel_list:
            per_rel.append((rel, sorted(glob(os.path.join(dst, rel), recursive=True))))
        found = sorted({fp for _, hits in per_rel for fp in hits})
        if not found:
            unresolved.append(split)
            print(f"   {split}: MISSING, looked for {rel_list} under {dst}")
            continue
        if mode == "any_of":
            # Alternatives of one layout: only the first hit is used at eval
            # time, so report exactly that one (a second layout on disk is
            # ignored, not merged).
            winner = next(rel for rel, hits in per_rel if hits)
            print(f"   {split}: OK (any_of -> {winner})")
            continue
        partial = [rel for rel, hits in per_rel if not hits]
        if len(rel_list) > 1 and partial:
            # all_of means every declared shard is required (LiveCodeBench's
            # cumulative set); a partial merge would silently shrink the
            # benchmark and skew the metric.
            unresolved.append(split)
            print(f"   {split}: PARTIAL, all_of missing {partial} under {dst}")
            continue
        print(f"   {split}: OK ({len(found)} files) -> {found[:3]}")
    if unresolved:
        # Fail loud: a MISSING split means eval will die later with a confusing
        # FileNotFoundError, or silently fall back to a network read.
        sys.exit(
            f"[prefetch] FAILED for task '{a.task}': split(s) {unresolved} matched no "
            f"downloaded file. Either the download was blocked (check network / "
            f"allow_patterns) or datasets.json globs do not match the repo layout "
            f"(list it with: huggingface_hub.HfApi().list_repo_files("
            f"'{spec['repo_id']}', repo_type='dataset'))."
        )
    print(f"[prefetch] {a.task}: all declared splits resolved")


if __name__ == "__main__":
    main()
