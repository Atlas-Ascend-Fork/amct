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
"""Shared helpers for quant-benchmark-eval scripts."""

import importlib.metadata
import importlib.util
import ast
import os
import re

import pandas as pd

ATTN_QUANT_TARGETS = {"attn-linear", "attn-cache"}
REQUIRED_LCB_FIELDS = ("question_id", "question_content")
# transformers 4.56 renamed from_pretrained's `torch_dtype` to `dtype`; this repo
# still pins 4.51 / 4.52 for some models (examples/algorithms/flatquant,
# experimental DeepSeekV3.2), which reject the new name outright.
_DTYPE_KWARG_MIN_VERSION = (4, 56)

# Task names this skill accepts that are not lm-eval task names.
TASK_ALIASES = {"math500": "hendrycks_math500"}
# Scored by the official lcb_runner instead of lm-eval, so they have no
# "<metric>,<filter>" result key to cross-check.
NON_LM_EVAL_TASKS = {"livecodebench", "livecodebench_release_v6"}


def read_dataset_file(path):
    """Read one registered dataset file into a pandas DataFrame."""
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    df = pd.read_parquet(path, engine="fastparquet")
    return _restore_dotted_columns(df)


def _version_prefix(version):
    """Turn "4.52.4" / "5.12.1rc1" into a comparable (4, 52) / (5, 12) tuple."""
    parts = []
    for chunk in re.split(r"[.\-+]", str(version)):
        if not chunk.isdigit():
            break
        parts.append(int(chunk))
    return tuple(parts[:2]) if len(parts) >= 2 else (0, 0)


def dtype_kwarg_name():
    """Weight-dtype kwarg name the installed transformers actually accepts.

    Passing `dtype=` to a 4.52 install does not fail with a clear "unexpected
    keyword": the value lands in `**kwargs`, gets written into the config and
    blows up later as `TypeError: Object of type dtype is not JSON serializable`.
    Resolve the name per version instead of forcing every user onto >=4.56.
    """
    try:
        import transformers

        version = transformers.__version__
    except Exception:  # pragma: no cover - transformers is a hard dep elsewhere
        return "torch_dtype"
    if _version_prefix(version) >= _DTYPE_KWARG_MIN_VERSION:
        return "dtype"
    return "torch_dtype"


def dtype_kwargs(dtype):
    """`{<kwarg>: dtype}` for from_pretrained, version-correct (see constraints.md #13)."""
    return {dtype_kwarg_name(): dtype}


def _restore_dotted_columns(df):
    """Restore flattened parquet struct columns like choices.text -> choices."""
    dotted = [col for col in df.columns if "." in col]
    if not dotted:
        return df
    groups = {}
    for col in dotted:
        prefix, suffix = col.split(".", 1)
        if prefix in df.columns:
            continue
        groups.setdefault(prefix, []).append((suffix, col))
    for prefix, items in groups.items():
        df[prefix] = [
            {suffix: row[col] for suffix, col in items} for _, row in df.iterrows()
        ]
        df = df.drop(columns=[col for _, col in items])
    return df


def local_dataset_files(task, registry, mapping, split="test", existing_only=True):
    """Resolve local dataset files for a task using its exact registry key."""
    spec = registry[task]
    key = (spec["repo_id"], spec.get("dataset_name"))
    fps = mapping.get(key, {}).get(split, [])
    if existing_only:
        fps = [fp for fp in fps if os.path.exists(fp)]
    if fps:
        return fps
    raise FileNotFoundError(
        f"{spec['repo_id']} split '{split}' is not prefetched for task '{task}'. "
        f"Run prefetch_data.py --task {task} first."
    )


def validate_lcb_docs(docs):
    """Fail before generation if the LiveCodeBench cache misses required fields."""
    for idx, doc in enumerate(docs, 1):
        missing = [field for field in REQUIRED_LCB_FIELDS if not doc.get(field)]
        if missing:
            raise ValueError(
                f"LiveCodeBench doc #{idx} missing required fields: {missing}"
            )


def split_quant_targets(quant_target):
    if not quant_target:
        return []
    if isinstance(quant_target, str):
        return [item.strip() for item in quant_target.split(",") if item.strip()]
    return list(quant_target)


def split_list_arg(value, default=None):
    """Split a comma-separated CLI value into a list; empty → `default`.

    Accepts an already-split list so callers can forward either form.
    """
    if value is None or value == "":
        return list(default) if default else []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


def has_attention_quant(quant_target):
    return bool(ATTN_QUANT_TARGETS.intersection(split_quant_targets(quant_target)))


def ensure_supported_quant_target(quant_target, allow_attn_quant=False):
    targets = split_quant_targets(quant_target)
    unsupported = sorted(ATTN_QUANT_TARGETS.intersection(targets))
    if unsupported and not allow_attn_quant:
        raise RuntimeError(
            "attention quant target requires explicit confirmation: "
            f"{unsupported}. Quantized attention disables KV cache and is much "
            "slower for generation. Re-run with --allow_attn_quant only after "
            "confirming this is the intended benchmark object."
        )
    return targets


def check_environment(require_lm_eval=False, require_lcb_runner=False):
    from config import BENCH_ASSETS, require_amct_home

    require_amct_home()
    requirements = [
        ("torch", "torch", None),
        ("torch_npu", "torch_npu", None),
        ("transformers", "transformers", None),
        ("datasets", "datasets", None),
        ("pandas", "pandas", None),
        ("fastparquet", "fastparquet", None),
        ("amct_pytorch", "amct_pytorch", None),
    ]
    if require_lm_eval:
        requirements.append(("lm_eval", "lm-eval", "0.4.12"))
    if require_lcb_runner:
        requirements.append(("lcb_runner", "livecodebench", None))

    missing = []
    version_mismatch = []
    for module_name, package_name, expected_version in requirements:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
            continue
        if expected_version is None:
            continue
        try:
            installed = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            installed = "unknown"
        if installed != expected_version:
            version_mismatch.append(
                f"{package_name}=={expected_version} required, found {installed}"
            )

    if missing or version_mismatch:
        parts = []
        if missing:
            parts.append("missing modules: " + ", ".join(missing))
        if version_mismatch:
            parts.append("version mismatch: " + "; ".join(version_mismatch))
        raise RuntimeError("benchmark environment check failed: " + " | ".join(parts))
    print(f"[env] BENCH_ASSETS={BENCH_ASSETS}")
    print(
        f"[env] transformers={_installed_version('transformers')} "
        f"(weight dtype kwarg: {dtype_kwarg_name()})"
    )


def _installed_version(package_name):
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def warn_trust_remote_code(model_path, enabled):
    """Mirror amct's CLI convention for `trust_remote_code` (see constraints.md #12).

    `amct_pytorch/cli/llm/args.py` defaults the flag to False and
    `common/models/llm/common/base.py` warns when it is off. These benchmark
    scripts used to hardcode True, which silently executes whatever custom
    modeling code ships in a user-supplied directory; the flag is now explicit
    and both directions are announced.
    """
    if enabled:
        print(
            f"[warn] trust_remote_code enabled: custom modeling code under "
            f"{model_path} WILL be executed. Only pass --trust_remote_code for "
            f"model directories you trust."
        )
    else:
        print(
            "[warn] running without --trust_remote_code: some models need it to "
            "load custom model code. If loading fails with a "
            "'requires you to execute the modeling file' error, re-run with "
            "--trust_remote_code (only for a trusted model directory)."
        )
    return enabled


def parse_result_log(path):
    """Parse [RESULT] lines from benchmark logs into a compact metrics dict."""
    pattern = re.compile(r"^\[RESULT\]\s+(\S+)\s+([^:]+):\s+(.*)$")
    results = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.match(line.strip())
            if not match:
                continue
            tag, task, payload = match.groups()
            try:
                metrics = ast.literal_eval(payload)
            except (SyntaxError, ValueError):
                metrics = {"raw": payload}
            results.setdefault(tag, {})[task] = metrics
    return results


def resolve_task_metric_keys(task_name, task_manager=None):
    """Return lm-eval's result key names ("<metric>,<filter>") for a task/group.

    lm-eval names every result key `f"{metric},{filter_key}"`, where filter_key
    is the `name` of an entry in the task's `filter_list` ("none" when the task
    declares no filter_list). Reading it back from the installed task index keeps
    SKILL.md's metric column from drifting away from what lm-eval emits.

    Only the task index / YAML configs are read -- no dataset is downloaded, so
    this is safe offline. Requires lm-eval to be importable.
    """
    from lm_eval.tasks import TaskManager

    tm = task_manager or TaskManager()
    entry = tm.task_index.get(task_name)
    if entry is None:
        raise KeyError(f"task '{task_name}' is not in the lm-eval task index")
    cfg = getattr(entry, "cfg", None) or {}
    metrics = [
        m.get("metric") for m in (cfg.get("metric_list") or []) if m.get("metric")
    ]
    filters = [
        f.get("name") for f in (cfg.get("filter_list") or []) if f.get("name")
    ] or ["none"]
    keys = {f"{metric},{f}" for metric in metrics for f in filters}
    for agg in cfg.get("aggregate_metric_list") or []:
        metric = agg.get("metric")
        if not metric:
            continue
        agg_filter = agg.get("filter_list")
        agg_filters = (
            agg_filter
            if isinstance(agg_filter, list)
            else ([agg_filter] if agg_filter else ["none"])
        )
        keys |= {f"{metric},{f}" for f in agg_filters}
    if not keys:
        raise KeyError(f"task '{task_name}' declares no metrics")
    return sorted(keys)


def resolve_task_dataset_keys(task_name, task_manager=None):
    """`{(dataset_path, dataset_name)}` lm-eval will request for a task or group.

    Groups (`mmlu_pro`) declare no dataset of their own, so walk their subtask
    list. A *leaf* task's cfg also carries a `task` key -- its own name as a
    string -- so only a list means "group"; recursing on the string form loops
    forever. Only the installed task index / YAML configs are read, no dataset
    download, so this stays safe offline. Requires lm-eval to be importable.
    """
    from lm_eval.tasks import TaskManager

    tm = task_manager or TaskManager()
    wanted = set()
    pending = [task_name]
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        entry = tm.task_index.get(name)
        if entry is None:
            raise KeyError(f"task '{name}' is not in the lm-eval task index")
        cfg = getattr(entry, "cfg", None) or {}
        subtasks = cfg.get("task")
        if isinstance(subtasks, list):
            pending.extend(subtasks)
            continue
        path = cfg.get("dataset_path")
        if not path:
            raise KeyError(f"task '{name}' declares no dataset_path")
        wanted.add((path, cfg.get("dataset_name")))
    return wanted


def registered_dataset_keys(registry=None):
    """`(repo_id, dataset_name)` pairs datasets.json can serve offline."""
    from config import load_dataset_registry

    reg = load_dataset_registry() if registry is None else registry
    return {(spec["repo_id"], spec.get("dataset_name")) for spec in reg.values()}


def check_task_offline_ready(task_name, registry=None, task_manager=None):
    """Fail before a model is loaded if `--task` cannot be served offline.

    Two different mistakes look identical once lm-eval is running: a typo
    (`arc-challenge`) and a task whose dataset was never registered
    (`gsm8k`). Both make `bench_offline._patched` fall through to the real
    `datasets.load_dataset` -- a network read that on these boxes dies as
    `FSTimeoutError` minutes in, *after* the model was loaded and every layer
    assembled, and with nothing in the traceback pointing at `--task`.
    Resolving the task through lm-eval's index here turns that into an
    immediate, actionable error (constraints.md #2).

    A requested `(path, name)` counts as registered when `(path, None)` is too:
    that is the shim's own fallback, and MATH-500 needs it (the task declares
    `dataset_name: default`, datasets.json keys the repo with no name).

    Returns the resolved pairs so callers can log what will be read.
    """
    try:
        wanted = resolve_task_dataset_keys(task_name, task_manager)
    except KeyError as exc:
        known = sorted({repo for repo, _ in registered_dataset_keys(registry)})
        raise RuntimeError(
            f"--task {task_name!r} cannot be checked against datasets.json: {exc}. "
            f"Registered tasks' repos: {known}. Check the spelling (lm-eval names "
            f"are what --task takes, plus this skill's aliases "
            f"{sorted(TASK_ALIASES)}), or add a datasets.json entry + prefetch it."
        ) from exc
    known = registered_dataset_keys(registry)
    missing = sorted(
        pair for pair in wanted if pair not in known and (pair[0], None) not in known
    )
    if missing:
        raise RuntimeError(
            f"--task {task_name!r} reads {missing} from the network: datasets.json "
            f"has no entry for it, so the offline shim would pass the request "
            f"through to `datasets.load_dataset` (constraints.md #2) and the run "
            f"would die as FSTimeoutError only after the model was assembled. "
            f"Registered (repo_id, dataset_name): {sorted(known)}. Add an entry "
            f"with matching globs, run prefetch_data.py for it, then re-run."
        )
    return wanted
