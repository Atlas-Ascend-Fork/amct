# -*- coding: UTF-8 -*-
"""Shared helpers for quant-benchmark-eval scripts."""

import importlib.metadata
import importlib.util
import ast
import os
import re
from glob import glob

import pandas as pd

ATTN_QUANT_TARGETS = {"attn-linear", "attn-cache"}
REQUIRED_LCB_FIELDS = ("question_id", "question_content")


def read_dataset_file(path):
    """Read one registered dataset file into a pandas DataFrame."""
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    df = pd.read_parquet(path, engine="fastparquet")
    return _restore_dotted_columns(df)


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


def expand_existing(paths):
    out = []
    for path in paths:
        matches = sorted(glob(path))
        out.extend(matches or [path])
    return out
