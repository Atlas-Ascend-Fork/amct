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
"""Compare bf16 and quant benchmark result files."""

import argparse
import json
import os
import sys

SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SKILL_DIR)

from utils import parse_result_log  # noqa: E402

PREFERRED_METRICS = (
    "exact_match",
    "acc_norm",
    "acc",
    "pass@1",
    "pass_at_1",
    "math_verify",
)

# Bookkeeping keys, not scores. `livecodebench_eval.py` reports the sample count
# and the output paths it wrote; if `sample_len` stayed eligible it would be the
# only numeric key in a generation-only log, get auto-picked, and print
# `delta_quant_minus_bf16: 0.0` -- indistinguishable from "quantization is
# lossless". LiveCodeBench pass@1 comes from the official `lcb_runner`
# evaluator output (`*_eval.json`), never from the generation log.
NON_METRIC_KEYS = frozenset(
    {
        "sample_len",
        "samples",
        "num_samples",
        "n_samples",
        "total",
        "count",
    }
)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _unwrap_lcb_eval(obj):
    """Return the metrics dict from an official lcb_runner `*_eval.json`.

    `lcb_runner.runner.custom_evaluator` dumps `get_metrics()`' return value,
    which for the codegeneration scenario is a 3-element list
    `[pass_at_k, results, final_metadata]`; `pass_at_k` is the dict holding
    `pass@1` / `pass@5` / ... (plus a per-task `detail` map whose values are
    lists, so `_flatten` drops them). Accept the list form and the bare dict.
    """
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict) and any(str(k).startswith("pass@") for k in item):
                return item
    return obj


def _load_result(path):
    """Load a result JSON, or a run log's `[RESULT]` lines.

    Every failure mode here is reported with an actionable message: a truncated
    or interrupted run leaves no `[RESULT]` line, and the old behaviour surfaced
    that much later as "no numeric metric found. Available: []".
    """
    if not os.path.isfile(path):
        raise SystemExit(f"result file not found: {path}")
    try:
        return _unwrap_lcb_eval(_load_json(path))
    except json.JSONDecodeError:
        pass
    parsed = parse_result_log(path)
    if not parsed:
        hint = _generation_log_hint(path)
        raise SystemExit(
            f"{path} is not JSON and contains no "
            f"'[RESULT] <tag> <task>: <metrics>' line: the benchmark did not "
            f"finish (interrupted / OOM / crashed before scoring) or its results "
            f"went to another file. Re-run it and pass the log that ends with a "
            f"[RESULT] line.{hint}"
        )
    return parsed


def _generation_log_hint(path):
    """Extra hint when the file is a LiveCodeBench *generation* log (no score).

    `livecodebench_eval.py` prints `[lcb-gen]` instead of `[RESULT]` on purpose:
    that log proves generation finished but carries no pass@1. Naming it here
    turns a generic "did not finish" into the one action that helps -- score it
    with the official evaluator and feed the resulting `*_eval.json`.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            if not any(line.lstrip().startswith("[lcb-gen]") for line in f):
                return ""
    except OSError:
        return ""
    return (
        " This file is a LiveCodeBench *generation* log ([lcb-gen]): it holds no "
        "score by design. Score it with `python -m "
        "lcb_runner.runner.custom_evaluator ...`, then pass the "
        "`*_codegeneration_output_eval.json` it writes (see "
        "references/livecodebench.md §2)."
    )


def _flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(value, next_prefix))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = float(obj)
    return out


def _split_key(key):
    """Split a flattened key into (depth, metric_field, metric_name).

    "quant.mmlu_pro.exact_match,custom-extract" -> (3, "exact_match,custom-extract",
    "exact_match"). lm-eval suffixes every metric with its filter name, so match
    on the metric name and keep the full field only for display / exact requests.
    """
    parts = key.split(".")
    metric_field = parts[-1]
    return len(parts), metric_field, metric_field.split(",")[0]


def _candidates(flat):
    """Numeric score candidates: key -> (depth, key, metric_field, metric_name, value).

    `_stderr` keys and bookkeeping keys (NON_METRIC_KEYS) are not scores, so they
    are dropped here and can never be auto-picked -- or matched by name -- and
    reported as if they were an accuracy delta.
    """
    out = {}
    for key, value in flat.items():
        depth, metric_field, metric_name = _split_key(key)
        if metric_name.endswith("_stderr"):
            continue
        if metric_name in NON_METRIC_KEYS:
            continue
        out[key] = (depth, key, metric_field, metric_name, value)
    return out


def _select(cands):
    """Prefer the shallowest key so a group aggregate wins over its subtasks.

    Ties are broken by key length then lexicographically, so "mmlu_pro" wins over
    "mmlu_pro_biology" deterministically instead of by ASCII luck.
    """
    best = min(cands.values(), key=lambda c: (c[0], len(c[1]), c[1]))
    return best[1], best[4]


def _pick_metric(flat, metric):
    cands = _candidates(flat)
    if not cands:
        raise SystemExit(
            "no numeric score found in this result file. Expected lm-eval "
            "[RESULT] lines, a JSON metrics object, or the official lcb_runner "
            "`*_eval.json` (which holds `pass@k`). A LiveCodeBench *generation* "
            "log carries no score -- feed the evaluator output instead (see "
            "references/livecodebench.md §2). Parsed keys: "
            f"{sorted(flat) or '<empty>'}"
        )
    available = sorted({c[2] for c in cands.values()})

    if metric:
        matches = {
            k: c
            for k, c in cands.items()
            if c[2] == metric
            or c[3] == metric
            or k == metric
            or k.endswith("." + metric)
        }
        if not matches:
            hint = (
                ""
                if available
                else (
                    " (this file has no score key at all: it looks like a "
                    "generation-only log, not an evaluator result)"
                )
            )
            raise SystemExit(
                f"metric '{metric}' not found. Available: {available}{hint}"
            )
        return _select(matches)

    for preferred in PREFERRED_METRICS:
        matches = {k: c for k, c in cands.items() if c[3] == preferred}
        if matches:
            return _select(matches)
    if len(cands) == 1:
        only = next(iter(cands.values()))
        return only[1], only[4]
    raise SystemExit(
        f"cannot auto-pick a metric; pass --metric. Available: {available}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True, help="bf16 result log or JSON")
    ap.add_argument("--quant", required=True, help="quant result log or JSON")
    ap.add_argument(
        "--metric",
        default="",
        help=(
            "metric to compare: either the metric name (exact_match, acc_norm, "
            "pass@1) or lm-eval's full result key (<metric>,<filter>, e.g. "
            "exact_match,custom-extract). Omit to auto-pick."
        ),
    )
    ap.add_argument("--sample_len", type=int, default=0)
    ap.add_argument(
        "--limited",
        action="store_true",
        help="mark the comparison as a smoke/limited sample",
    )
    args = ap.parse_args()

    bf16_flat = _flatten(_load_result(args.bf16))
    quant_flat = _flatten(_load_result(args.quant))
    bf16_key, bf16_value = _pick_metric(bf16_flat, args.metric)
    metric_field = bf16_key.split(".")[-1]
    quant_key, quant_value = _pick_metric(quant_flat, metric_field)
    result = {
        "metric": metric_field,
        "bf16_key": bf16_key,
        "quant_key": quant_key,
        "bf16": bf16_value,
        "quant": quant_value,
        "delta_quant_minus_bf16": round(quant_value - bf16_value, 12),
        "sample_len": args.sample_len or None,
        "limited": args.limited,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
