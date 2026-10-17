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
"""Offline data shim for lm-eval-harness on Ascend NPU boxes.

Intercepts datasets.load_dataset for pre-downloaded benchmarks and reads local
parquet via fastparquet or jsonl via pandas (pyarrow crashes "Index not in
dictionary bounds"; the HF remote glob times out with FSTimeoutError). Import +
install() BEFORE lm_eval.

Dataset->local-path mapping comes from datasets.json (shared with prefetch_data.py)
via config.py, resolved under $BENCH_ASSETS. Add a benchmark = add a datasets.json
entry, no code change here.
"""

import os
import sys
import inspect
import pandas as pd
import datasets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import UnresolvedSplit, local_dataset_map  # noqa: E402
from utils import read_dataset_file  # noqa: E402

# (repo_id, dataset_name) -> {split: [local paths]}, built from datasets.json + $BENCH_ASSETS
_LOCAL = local_dataset_map()

_ORIG = datasets.load_dataset

# Signature of the real `datasets.load_dataset`, used to normalize whatever
# positional/keyword mix a caller passes into named arguments. `split` sits at
# index 4 (path, name, data_dir, data_files, split), so reading it out of
# `*args` by position would be wrong the moment a caller skips `data_dir`.
_ORIG_PARAMS = tuple(inspect.signature(_ORIG).parameters)


def _bind(path, args, kwargs):
    """`(name, split)` a caller asked for, positionals and keywords both handled."""
    bound = {}
    for key, value in zip(_ORIG_PARAMS[1:], args):
        bound[key] = value
    bound.update(kwargs)
    return bound.get("name"), bound.get("split")


def _patched(path, *args, **kwargs):
    name, wanted_split = _bind(path, args, kwargs)
    key = (path, name)
    if key not in _LOCAL:
        key = (path, None)
    if key in _LOCAL:
        declared = _LOCAL[key]
        # `datasets.load_dataset(..., split="test")` returns one Dataset, not a
        # DatasetDict; lm-eval forwards a task's `dataset_kwargs` verbatim, so a
        # task that declares `split` would otherwise get the wrong container
        # type and fail somewhere inside the harness instead of here.
        #
        # Validate and read *only* what was asked for: loading every declared
        # split makes an unprefetched `validation` fail a run that asked for
        # `test` (MMLU-Pro / boolq / copa all declare a split the caller never
        # scores) and parses data nobody reads.
        if wanted_split is not None:
            if wanted_split not in declared:
                raise KeyError(
                    f"{path} has no split '{wanted_split}' in the prefetched cache; "
                    f"datasets.json declares {sorted(declared)}. Add the split to "
                    f"the registry entry (and prefetch it) or drop `split` from "
                    f"the task's dataset_kwargs."
                )
            wanted = [wanted_split]
        else:
            wanted = sorted(declared)
        splits = {}
        for split in wanted:
            fps = declared[split]
            if isinstance(fps, UnresolvedSplit):
                # Deferred from local_dataset_map(): a partial all_of shard set.
                fps.raise_it()
            if not fps:
                raise FileNotFoundError(
                    f"{path} split '{split}' resolved to no local file under "
                    f"$BENCH_ASSETS: every datasets.json glob for it matched "
                    f"nothing. Prefetch it (prefetch_data.py) or fix the split "
                    f"globs to the layout the dataset repo actually ships."
                )
            missing = [fp for fp in fps if not os.path.exists(fp)]
            if missing:
                raise FileNotFoundError(
                    f"{path} split '{split}' not prefetched at {missing}; "
                    f"run prefetch_data.py first"
                )
            frames = [read_dataset_file(fp) for fp in fps]
            df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
            splits[split] = datasets.Dataset.from_pandas(df, preserve_index=False)
        if wanted_split is not None:
            return splits[wanted_split]
        return datasets.DatasetDict(splits)
    try:
        return _ORIG(path, *args, **kwargs)
    except Exception as exc:
        # Not a registry key: the request goes to the real loader, i.e. the
        # network / HF cache. When that fails the original message (typically
        # FSTimeoutError on these boxes) says nothing about *why* the shim did
        # not serve it, so chain the context that makes it actionable.
        raise RuntimeError(
            f"datasets.load_dataset(path={path!r}, name={name!r}) has no entry in "
            f"datasets.json, so the offline shim passed it through to the real "
            f"loader, which failed with {type(exc).__name__}: {exc}. Registered "
            f"(repo_id, dataset_name): {sorted(_LOCAL)}. Prefetch the dataset and "
            f"add a registry entry (prefetch_data.py), or use a registered task."
        ) from exc


def install():
    """Redirect dataset loads to $BENCH_ASSETS instead of the network / HF cache.

    lm-eval 0.4.12 resolves the loader through the module attribute at call time
    (`api/task.py`: `datasets.load_dataset(path=..., name=...)`), so the patch on
    `datasets.load_dataset` is the one that actually intercepts it, and it is
    applied unconditionally. A layout that instead binds `load_dataset` into
    `lm_eval.api.task` at import time keeps its own reference and would go
    straight to the network; patch that name too when it exists -- assigning it
    blindly only creates an attribute nothing reads, and swallowing the import
    error hides which of the two happened (constraints.md #2).
    """
    datasets.load_dataset = _patched
    patched = ["datasets.load_dataset"]
    try:
        import lm_eval.api.task as _t
    except ImportError as exc:
        print(f"[bench_offline] lm_eval.api.task not importable ({exc})")
    else:
        if hasattr(_t, "load_dataset"):
            _t.load_dataset = _patched
            patched.append("lm_eval.api.task.load_dataset")
    print(f"[bench_offline] patched {' + '.join(patched)} for {list(_LOCAL)}")
