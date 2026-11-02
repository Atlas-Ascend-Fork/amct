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

from utils import lcb_run_meta_from_eval_path, parse_result_log  # noqa: E402

PREFERRED_METRICS = (
    "exact_match",
    "acc_norm",
    "acc",
    "pass@1",
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


# Keys whose difference makes the subtraction describe something other than
# quantization. The two tuples below are the single source for which side a key
# belongs to and for what `_run_metadata` reads back out of a result payload.
# Neither half is counted in the prose: SKILL.md #2 routes the object side to
# the shared-parameter table in SKILL.md #4.2, whose rows the self-check pairs
# with `_OBJECT_SIDE_KEYS`, and states the evaluation side as alignment points
# with no key count -- so adding a key here leaves no number behind to go stale.

# Evaluation side: these decide the accuracy itself, so both sides carry them
# and both must agree. None is a value here, not an absence -- `limit: None`
# says the run had no sample cap -- so it is compared like any other value.
# `release_version` selects the LiveCodeBench question set, so it decides the
# accuracy exactly like `task` does; only the LiveCodeBench sidecar writes it,
# and a key absent from a side is skipped rather than treated as a mismatch.
_EVAL_SIDE_KEYS = ("task", "limit", "max_gen_toks", "release_version")

# Object side: these name what was evaluated. A bf16 baseline assembles no quant
# block and writes None for them (see assemble_and_eval.py's payload): None
# means "not applicable", not "different", so they only bite when both sides
# describe a quantized object. Every one of them has a CLI flag on the entry
# points and a row in SKILL.md #4.2's shared-parameter table; self_check.py
# enforces that pairing, and that `_run_metadata` reads the key back, so a key
# added here without telling the agent to align it fails the run instead of
# shipping.
_OBJECT_SIDE_KEYS = (
    "model_name",
    "quant_target",
    "bit_config",
    "quant_dtype",
    "algos",
    "is_per_tensor",
)

_COMPARABILITY_KEYS = _EVAL_SIDE_KEYS + _OBJECT_SIDE_KEYS

# Bookkeeping keys a comparison prints but does not gate on. `direct` and
# `quant_scope` say which object the quant side scored -- amct direct
# conversion or a PTQ checkpoint, and which decoder layers were quantized.
# Both entries already write them. Printed, never gated: the baseline
# quantizes nothing and writes False / None, so gating on them would fail
# every pair.
_META_BOOKKEEPING_KEYS = (
    "tag",
    "sample_len",
    "limited",
    "bootstrap_iters",
    "direct",
    "quant_scope",
)


def _extract_metadata(obj):
    """The comparability + bookkeeping keys `obj` carries."""
    return {
        key: obj[key]
        for key in _META_BOOKKEEPING_KEYS + _COMPARABILITY_KEYS
        if key in obj
    }


def _run_metadata(obj):
    """Run parameters that `assemble_and_eval.py --output_path` wrote, or {}.

    ``--sample_len`` and ``--limited`` used to be hand-typed here: nothing
    checked them, so a full run reported as a smoke test (or the reverse) went
    out in the JSON as fact. Both are already in the result payload the run
    produced, along with every parameter that decides whether two runs are
    comparable at all, so read them back.

    The key set is derived from ``_COMPARABILITY_KEYS``, never hand-listed: a
    key this function does not extract is a key ``_check_comparable`` cannot
    see, so a declared-but-unread comparability key stops gating silently and a
    delta across differing values prints as fact. ``self_check.py`` re-derives
    the coverage from this function.

    An lm-eval `[RESULT]` log, a bare metrics JSON and the official lcb_runner
    `*_eval.json` carry none of these keys, and return {}; for the last of those
    `_sidecar_metadata` supplies them from the file `livecodebench_eval.py`
    wrote next to its `--output_json`.
    """
    if not isinstance(obj, dict) or "results" not in obj or "tag" not in obj:
        return {}
    return _extract_metadata(obj)


def _sidecar_metadata(path):
    """Run parameters beside an official `*_eval.json`, or {}.

    `lcb_runner.runner.custom_evaluator` writes only scores, so the LiveCodeBench
    path would otherwise reach `_check_comparable` with nothing on either side
    and print a delta across two runs nobody compared. `livecodebench_eval.py`
    writes these next to its `--output_json`; the evaluator's filename
    convention points back at them.
    """
    meta_path = lcb_run_meta_from_eval_path(path)
    if not meta_path or not os.path.isfile(meta_path):
        return {}
    try:
        obj = _load_json(meta_path)
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(obj, dict) or "tag" not in obj:
        return {}
    return _extract_metadata(obj)


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
    """Load (metrics, run metadata) from a result JSON or a log's `[RESULT]` lines.

    Every failure mode here is reported with an actionable message: a truncated
    or interrupted run leaves no `[RESULT]` line, and the old behaviour surfaced
    that much later as "no numeric metric found. Available: []".
    """
    if not os.path.isfile(path):
        raise SystemExit(f"result file not found: {path}")
    try:
        obj = _load_json(path)
    except json.JSONDecodeError:
        pass
    else:
        return _unwrap_lcb_eval(obj), _run_metadata(obj) or _sidecar_metadata(path)
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
    return parsed, {}


def _key_mismatch(key, bf16_meta, quant_meta):
    """True when the two sides disagree on `key`, under that key's None rule.

    A key missing from either side is not comparable at all and is skipped
    (`release_version` is written only by the LiveCodeBench sidecar). For an
    object-side key, None means that side has no quantized object -- a bf16
    baseline assembles no quant block and writes None for all of them -- so it
    is skipped too. An evaluation-side key has no such reading: `limit: None`
    is the value "no sample cap", and a full run must not be subtracted from a
    capped one.
    """
    if key not in bf16_meta or key not in quant_meta:
        return False
    left, right = bf16_meta[key], quant_meta[key]
    if key in _OBJECT_SIDE_KEYS and (left is None or right is None):
        return False
    return left != right


def _check_side_tags(bf16_meta, quant_meta, bf16_path, quant_path):
    """Refuse to subtract two runs that are not baseline-minus-quantized.

    `tag` is the only thing in a payload that says which side a run was (each
    entry writes `"bf16" if a.bf16 else "quant"`), and the object-side None
    exemption in `_check_comparable` cannot tell a baseline from a
    direct-conversion run: `--direct` builds no algorithm module either, so it
    writes `algos: None` exactly like a baseline. Passed as `--bf16` against a
    PTQ run, the pair gates clean and the difference goes out as
    `delta_quant_minus_bf16` -- an algorithm delta wearing a quantization field
    name, with `compared_from: "run metadata (both sides)"` beside it.
    """
    if bf16_meta.get("tag") != "bf16":
        raise SystemExit(
            f"{bf16_path} is tagged {bf16_meta.get('tag')!r}, not 'bf16': "
            f"--bf16 takes the baseline run of the same float model, assembled "
            f"with no quant block. A --direct run scores an object too, so "
            f"subtracting a PTQ run from it would measure the algorithm rather "
            f"than quantization, and this script does not do that subtraction. "
            f"Pass the --bf16 run here."
        )
    if quant_meta.get("tag") != "quant":
        raise SystemExit(
            f"{quant_path} is tagged {quant_meta.get('tag')!r}, not 'quant': "
            f"--quant takes the quantized run. Two baselines subtract to "
            f"run-to-run noise and would print as delta_quant_minus_bf16."
        )


def _check_comparable(bf16_meta, quant_meta, bf16_path, quant_path):
    """Refuse to subtract two accuracies that were not measured the same way.

    Only runs when *both* sides carry metadata (i.e. both came from
    `--output_path`); a log or an lcb `*_eval.json` has nothing to compare
    against and is left alone. `_COMPARABILITY_KEYS` holds the evaluation-side
    parameters that decide the accuracy plus the parameters that name the
    evaluated object, so a delta across differing values is not a quantization
    effect; `_key_mismatch` applies the None rule each half of that split needs.
    Failing here is cheap and loud; printing the number and letting a report
    call it a delta is neither.
    """
    if not bf16_meta or not quant_meta:
        return
    _check_side_tags(bf16_meta, quant_meta, bf16_path, quant_path)
    differing = {
        key: (bf16_meta[key], quant_meta[key])
        for key in _COMPARABILITY_KEYS
        if _key_mismatch(key, bf16_meta, quant_meta)
    }
    if differing:
        detail = ", ".join(
            f"{k}: {v[0]!r} vs {v[1]!r}" for k, v in sorted(differing.items())
        )
        raise SystemExit(
            f"the two sides were not measured the same way ({detail}), so "
            f"delta_quant_minus_bf16 would not describe quantization. "
            f"{bf16_path} and {quant_path} must agree on every comparability "
            f"key (SKILL.md #2) -- re-run whichever side used the wrong value."
        )


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
        "references/livecodebench.md #2)."
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
            "references/livecodebench.md #2). Parsed keys: "
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
        help=(
            "mark the comparison as a smoke/limited sample. Taken from the "
            "result JSON when both sides came from --output_path."
        ),
    )
    args = ap.parse_args()

    bf16_obj, bf16_meta = _load_result(args.bf16)
    quant_obj, quant_meta = _load_result(args.quant)
    _check_comparable(bf16_meta, quant_meta, args.bf16, args.quant)
    bf16_flat = _flatten(bf16_obj)
    quant_flat = _flatten(quant_obj)
    bf16_key, bf16_value = _pick_metric(bf16_flat, args.metric)
    metric_field = bf16_key.split(".")[-1]
    quant_key, quant_value = _pick_metric(quant_flat, metric_field)
    # Explicit flags win; otherwise fall back to what the runs recorded. A
    # sample count the run itself reported beats one typed in by hand.
    sample_len = (
        args.sample_len or quant_meta.get("sample_len") or bf16_meta.get("sample_len")
    )
    limited = args.limited or bool(
        quant_meta.get("limited") or bf16_meta.get("limited")
    )
    result = {
        "metric": metric_field,
        "bf16_key": bf16_key,
        "quant_key": quant_key,
        "bf16": bf16_value,
        "quant": quant_value,
        "delta_quant_minus_bf16": round(quant_value - bf16_value, 12),
        "sample_len": sample_len or None,
        "limited": limited,
        # Which object the quant side scored. SKILL.md #4.3 requires the
        # report to state it, and this output is what a reviewer quotes; the
        # log line is not always at hand. Read off the quant side only -- the
        # baseline writes False / None for both. Null means this side carried
        # no run metadata, which `compared_from` below already says.
        "direct": quant_meta.get("direct"),
        "quant_scope": quant_meta.get("quant_scope"),
    }
    if bf16_meta and quant_meta:
        result["compared_from"] = "run metadata (both sides)"
    else:
        # Say so in the output rather than staying quiet: `_check_comparable`
        # skips a side with no metadata, so the delta above is ungated and the
        # reader has to align the two `[run]` lines by hand.
        sides = ((args.bf16, bf16_meta), (args.quant, quant_meta))
        bare = [p for p, m in sides if not m]
        result["compared_from"] = "none"
        result["comparability_checked"] = False
        result["comparability_gap"] = (
            f"no run metadata in {' and '.join(bare)}: the comparability keys "
            f"were not compared. For a LiveCodeBench score file this means "
            f"livecodebench_eval.py's *_run_meta.json sidecar is not next to "
            f"it (moved, renamed, or generated before the sidecar existed); "
            f"align the two [run] lines by hand before reporting the delta."
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
