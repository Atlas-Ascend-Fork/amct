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
import json
import os
import re
import sys

import pandas as pd

ATTN_QUANT_TARGETS = {"attn-linear", "attn-cache"}
REQUIRED_LCB_FIELDS = ("question_id", "question_content")
# LiveCodeBench filename conventions, one source for the writer
# (livecodebench_eval.py) and the reader (compare_results.py). The official
# `custom_evaluator` derives its score file from `--custom_output_file`, so the
# generation side's run parameters stay reachable from the score file's own
# name -- no extra flag for the caller to forget.
LCB_RUN_META_SUFFIX = "_run_meta.json"
LCB_EVAL_SUFFIX = "_codegeneration_output_eval.json"


def strip_json_suffix(path):
    """`path` without a trailing `.json`."""
    return path[:-5] if path.endswith(".json") else path


def lcb_run_meta_path(output_json):
    """Where the generation run's parameters go, next to `--output_json`."""
    return strip_json_suffix(output_json) + LCB_RUN_META_SUFFIX


def lcb_run_meta_from_eval_path(eval_path):
    """The run-parameter sidecar an official `*_eval.json` came from, or ''.

    Empty means `eval_path` is not an official evaluator output, so there is no
    sidecar to look for.
    """
    name = os.path.basename(eval_path)
    if not name.endswith(LCB_EVAL_SUFFIX):
        return ""
    stem = name[: -len(LCB_EVAL_SUFFIX)]
    return os.path.join(os.path.dirname(eval_path), stem + LCB_RUN_META_SUFFIX)


# transformers 4.56 renamed from_pretrained's `torch_dtype` to `dtype`; this repo
# still pins 4.51 / 4.52 for some models (examples/algorithms/flatquant,
# experimental DeepSeekV3.2), which reject the new name outright.
_DTYPE_KWARG_MIN_VERSION = (4, 56)


class TaskIndexMiss(KeyError):
    """`--task` names something the installed lm-eval does not know at all.

    A typo, or an alias this skill maps but lm-eval does not ship. Fixable by
    the user alone -- the registry is not involved.
    """


class TaskHasNoDataset(KeyError):
    """The lm-eval task exists but declares no `dataset_path`.

    Nothing to prefetch and nothing to match against datasets.json, so the
    local read shim could never serve it. Distinct from TaskIndexMiss: adding a
    datasets.json entry would not help, the task name itself is unusable here.
    """


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
    """`{<kwarg>: dtype}` for from_pretrained, version-correct (env-and-portability.md dtype section)."""
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
    from config import UnresolvedSplit

    spec = registry[task]
    key = (spec["repo_id"], spec.get("dataset_name"))
    fps = mapping.get(key, {}).get(split, [])
    if isinstance(fps, UnresolvedSplit):
        # Deferred from local_dataset_map(): a partial all_of shard set. This
        # task is the one being scored, so the error is raised here rather than
        # silently merging the shards that happen to be on disk.
        fps.raise_it()
    if existing_only:
        fps = [fp for fp in fps if os.path.exists(fp)]
    if fps:
        return fps
    raise FileNotFoundError(
        f"{spec['repo_id']} split '{split}' has no local file for task '{task}'. "
        f"Either the download never landed it (endpoint / proxy unreachable, "
        f"allow_patterns too narrow) or the datasets.json declaration no longer "
        f"matches where the repo ships the file. Run prefetch_data.py --task "
        f"{task} and read its per-split OK / MISSING / PARTIAL lines; MISSING / "
        f"PARTIAL means fix the entry (datasets-registry.md)."
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


def ensure_single_attention_target(quant_target):
    """Reject both attention targets in one run.

    amct trains exactly one target per PTQ round (`workflows/llm_ptq.py` raises
    on `len(args.quant_target) != 1`) and names the attention unit from the
    layer type alone, so an attn-linear round and an attn-cache round both save
    `layer_<idx>_self_attn.pt` -- into different directories, one per target.
    This skill takes a single `--param_dir`, so it can only ever feed one of
    them. Asking for both used to map the same directory to
    `attn_linear_param_dir` and `attn_cache_param_dir`, load the same file into
    the same `self_attn` unit twice, and report every unit as loaded while one
    target's quantizers stayed on construction defaults: a silent half-load
    with green stats (constraints.md #1).
    """
    targets = split_quant_targets(quant_target)
    both = sorted(ATTN_QUANT_TARGETS.intersection(targets))
    if len(both) > 1:
        raise RuntimeError(
            f"--quant_target {quant_target!r} names both attention targets "
            f"({', '.join(both)}), which one run cannot evaluate: amct trains one "
            f"target per round and both save the same filename "
            f"(layer_<idx>_self_attn.pt), so a single --param_dir holds only one "
            f"of them. Run once per target -- --param_dir and --quant_target for "
            f"attn-linear, then again for attn-cache -- and report the two "
            f"scores separately."
        )
    return targets


def ensure_direct_mode_args(direct, bf16, param_dir, quant_target, algos):
    """Reject `--direct` combined with a flag direct conversion cannot honour.

    Both entry points take the same flags, so the four messages live here once
    instead of twice. Each combination is a request the script cannot honour
    without scoring a different object than the one asked for: `--bf16` builds
    no quant block at all, `--param_dir` / `--algos` describe a PTQ checkpoint,
    and an omitted `--quant_target` has no directory to be derived from.
    """
    if not direct:
        return
    if bf16:
        sys.exit(
            "--direct and --bf16 are mutually exclusive: --direct quantizes "
            "every layer by direct conversion, --bf16 assembles no quant "
            "block at all. Pick one; the comparison needs one run of each."
        )
    if param_dir:
        sys.exit(
            f"--direct reads no ptq params, so --param_dir {param_dir} would be "
            "ignored. Drop --param_dir to score direct conversion, or drop "
            "--direct to score the checkpoint that directory holds."
        )
    if not quant_target:
        sys.exit(
            "--direct requires an explicit --quant_target: the target is "
            "otherwise derived from --param_dir and there is no directory "
            "here. Pass the amct flat target(s) the direct conversion covers: "
            "mlp / moe / attn-linear / attn-cache."
        )
    if algos is not None:
        sys.exit(
            f"--direct builds no algorithm module, so --algos {algos!r} cannot "
            "be used. Drop --algos for direct conversion, or drop --direct to "
            "score a PTQ checkpoint trained with those algorithms."
        )


def resolve_sample_limit(limit):
    """The sample cap to run with: None (= no cap) or a positive int.

    A non-positive cap caps nothing on either entry -- lm-eval does
    `int(limit) if limit else None` (`api/task.py`), `load_lcb_docs` heads the
    frame by the cap -- yet it is still the value both payloads record, and
    compare_results.py gates on that record: a full run written as `limit: 0`
    is not equal to a full run written as `limit: None`, so the two refuse to
    be compared and the message sends somebody to re-run a benchmark that
    already ran in full. None is the only spelling of "no cap" (both entries
    default to it), so refuse the rest before anything is loaded.
    """
    if limit is not None and limit <= 0:
        sys.exit(
            f"--limit {limit} caps nothing: both entries score the full "
            "benchmark on a non-positive cap while recording the value, and "
            "the recorded run then fails the comparability gate against one "
            "that omitted --limit. Drop --limit for a full run, or pass the "
            "number of samples to score."
        )
    return limit


def sample_limit_fields(limit):
    """The two payload keys that say how many samples were scored.

    One expression for both entries: they wrote `a.limit is not None` and
    `bool(a.limit)`, which agreed only while a non-positive cap was
    unreachable, and `limited` is what a report quotes when it says whether
    the score came from a subset.
    """
    return {"limited": limit is not None, "limit": limit}


def ensure_supported_quant_target(quant_target, allow_attn_quant=False):
    targets = ensure_single_attention_target(quant_target)
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
            # The install line is part of the message: the import name and the
            # package name differ (lm_eval / lm-eval), so "missing lm_eval"
            # alone leaves the reader to guess what to install.
            missing.append(
                f"{module_name} (pip install --user {package_name})"
                if expected_version is None
                else f"{module_name} (pip install --user "
                f"'{package_name}=={expected_version}')"
            )
            continue
        if expected_version is None:
            continue
        try:
            installed = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            installed = "unknown"
        if installed != expected_version:
            version_mismatch.append(
                f"{package_name}=={expected_version} required, found {installed} "
                f"(pip install --user '{package_name}=={expected_version}')"
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
    """Mirror amct's CLI convention for `trust_remote_code` (see constraints.md #5).

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


# Weight bytes one card is assumed to hold, leaving room for activations, the
# KV-cache-less full-context recompute and amct's quant intermediates. Kept well
# under the 64GB HBM of an Ascend 910 card. `cards_for_weights` divides a
# checkpoint's weight size by it, so the card count is a property of the
# checkpoint, and a machine whose cards hold less scales that count up. Lives
# here, not in `assemble_and_eval.py`, so `plan_device.py` can answer the card
# question without importing torch (~5s) just to read a config file.
SINGLE_CARD_WEIGHT_BUDGET = 40 * 1024**3


def estimate_weight_bytes(model_path):
    """Rough bf16 weight size from `config.json`, or None when it cannot tell.

    Reads the config only -- no weights, no NPU. Enough to separate the 8B class
    (~16GB) from a 35B MoE (hundreds of GB of expert weights), which is the whole
    decision `--device_map` has to make.
    """
    try:
        with open(os.path.join(model_path, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return None
    # VL-nested checkpoints keep the language config one level down.
    text_cfg = cfg.get("text_config") or cfg
    layers = text_cfg.get("num_hidden_layers")
    hidden = text_cfg.get("hidden_size")
    if not layers or not hidden:
        return None
    intermediate = text_cfg.get("intermediate_size") or 0
    experts = (
        text_cfg.get("num_experts")
        or text_cfg.get("n_routed_experts")
        or text_cfg.get("num_local_experts")
        or 0
    )
    per_layer = 4 * hidden * hidden  # q / k / v / o
    if experts:
        moe_intermediate = text_cfg.get("moe_intermediate_size") or intermediate
        per_layer += experts * 3 * hidden * moe_intermediate
        shared = text_cfg.get("shared_expert_intermediate_size") or 0
        per_layer += 3 * hidden * shared
    elif intermediate:
        per_layer += 3 * hidden * intermediate  # SwiGLU gate / up / down
    else:
        return None
    vocab = text_cfg.get("vocab_size") or 0
    return (layers * per_layer + 2 * vocab * hidden) * 2  # bf16


def cards_for_weights(weight_bytes):
    """How many cards hold `weight_bytes` at `SINGLE_CARD_WEIGHT_BUDGET` each."""
    return max(1, -(-weight_bytes // SINGLE_CARD_WEIGHT_BUDGET))  # ceil division


def plan_devices(model_path):
    """``(device_map, ASCEND_RT_VISIBLE_DEVICES)`` read off `config.json` alone.

    `device_map` is ``'auto'`` when the checkpoint cannot fit one card, ``''``
    when it can. The old ``''`` default put a 35B MoE on a single npu:0 and
    OOM'd inside `from_pretrained` with nothing pointing back here
    (env-and-portability.md #2). When `config.json` declares neither layer count
    nor hidden size the size is unknowable, so require an explicit value instead
    of guessing.

    The card list is the same division the verdict already makes: the budget is
    per card, so a checkpoint N budgets wide needs N of them. Both launch-time
    knobs come back from one read of the config.
    """
    weight_bytes = estimate_weight_bytes(model_path)
    if weight_bytes is None:
        raise RuntimeError(
            f"cannot infer --device_map: {model_path}/config.json is unreadable or "
            f"declares no num_hidden_layers/hidden_size, so the checkpoint size is "
            f"unknown. Pass --device_map explicitly: '' for single npu:0 (8B "
            f"class), 'auto' for multi-card (35B MoE)."
        )
    gib = weight_bytes / 1024**3
    budget_gib = SINGLE_CARD_WEIGHT_BUDGET / 1024**3
    cards = cards_for_weights(weight_bytes)
    devices = ",".join(str(i) for i in range(cards))
    if weight_bytes > SINGLE_CARD_WEIGHT_BUDGET:
        print(
            f"[device] config.json implies ~{gib:.0f}GiB of bf16 weights, over the "
            f"{budget_gib:.0f}GiB one card is budgeted for -> using device_map=auto "
            f"across {cards} cards (set ASCEND_RT_VISIBLE_DEVICES={devices}); a card "
            f"holding less than {budget_gib:.0f}GiB of weights takes more of them"
        )
        return "auto", devices
    print(
        f"[device] config.json implies ~{gib:.0f}GiB of bf16 weights, within the "
        f"{budget_gib:.0f}GiB one card is budgeted for -> single card npu:0 "
        f"(set ASCEND_RT_VISIBLE_DEVICES={devices})"
    )
    return "", devices


def infer_device_map(model_path):
    """The `--device_map` half of `plan_devices`, for the evaluation entries."""
    return plan_devices(model_path)[0]


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
            raise TaskIndexMiss(f"task '{name}' is not in the lm-eval task index")
        cfg = getattr(entry, "cfg", None) or {}
        subtasks = cfg.get("task")
        if isinstance(subtasks, list):
            pending.extend(subtasks)
            continue
        path = cfg.get("dataset_path")
        if not path:
            raise TaskHasNoDataset(f"task '{name}' declares no dataset_path")
        wanted.add((path, cfg.get("dataset_name")))
    return wanted


def registered_dataset_keys(registry=None):
    """`(repo_id, dataset_name)` pairs datasets.json can serve from local files."""
    from config import load_dataset_registry

    reg = load_dataset_registry() if registry is None else registry
    return {(spec["repo_id"], spec.get("dataset_name")) for spec in reg.values()}


def check_task_offline_ready(task_name, registry=None, task_manager=None):
    """Fail before a model is loaded if `--task` cannot be served from local files.

    Two different mistakes look identical once lm-eval is running: a typo
    (`arc-challenge`) and a task whose dataset was never registered
    (`gsm8k`). Both make `bench_offline._patched` fall through to the real
    `datasets.load_dataset` -- a hub request that fails minutes in with whatever
    error the box produces (timeout, unreachable), *after* the model was loaded
    and every layer assembled, and with nothing in the traceback pointing at
    `--task`.
    Resolving the task through lm-eval's index here turns that into an
    immediate, actionable error (constraints.md #2).

    A requested `(path, name)` counts as registered when `(path, None)` is too:
    that is the shim's own fallback, and MATH-500 needs it (the task declares
    `dataset_name: default`, datasets.json keys the repo with no name).

    Returns the resolved pairs so callers can log what will be read.
    """
    known = registered_dataset_keys(registry)
    try:
        wanted = resolve_task_dataset_keys(task_name, task_manager)
    except TaskIndexMiss as exc:
        # lm-eval never heard of it: a spelling / alias problem, and no
        # datasets.json entry could fix it.
        lcb_hint = (
            " It is scored by the official lcb_runner, not lm-eval: generate "
            "with livecodebench_eval.py."
            if task_name in NON_LM_EVAL_TASKS
            else ""
        )
        raise RuntimeError(
            f"--task {task_name!r} cannot be checked against datasets.json: {exc}. "
            f"--task takes lm-eval task names plus this skill's aliases "
            f"{sorted(TASK_ALIASES)}.{lcb_hint} Registered tasks' repos: "
            f"{sorted({repo for repo, _ in known})}."
        ) from exc
    except TaskHasNoDataset as exc:
        # The task exists but has no dataset of its own, so there is nothing to
        # prefetch and nothing for the local read shim to serve.
        raise RuntimeError(
            f"--task {task_name!r} cannot be served offline: {exc}, so there is "
            f"no (dataset_path, dataset_name) to match against datasets.json and "
            f"the local read shim would pass the request to the network "
            f"(constraints.md #2). Pass a leaf task that declares a dataset "
            f"(a group without one expands to its subtasks), or register the "
            f"dataset it reads. Registered (repo_id, dataset_name): "
            f"{sorted(known)}."
        ) from exc
    missing = sorted(
        pair for pair in wanted if pair not in known and (pair[0], None) not in known
    )
    if missing:
        raise RuntimeError(
            f"--task {task_name!r} reads {missing} from the network: datasets.json "
            f"has no entry for it, so the local read shim would pass the request "
            f"through to `datasets.load_dataset` (constraints.md #2) and the run "
            f"would die on the hub request only after the model was assembled. "
            f"Registered (repo_id, dataset_name): {sorted(known)}. Two ways "
            f"forward: register it (add a datasets.json entry with "
            f"matching globs, then run prefetch_data.py for it), or pull the data "
            f"yourself with huggingface_hub.snapshot_download into $BENCH_ASSETS "
            f"and add the entry afterwards -- what must stay fixed is the read "
            f"path, not where the bytes came from."
        )
    return wanted
