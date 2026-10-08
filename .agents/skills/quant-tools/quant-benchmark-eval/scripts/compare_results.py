# -*- coding: UTF-8 -*-
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


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_result(path):
    """Load a result JSON, or a run log's `[RESULT]` lines.

    Every failure mode here is reported with an actionable message: a truncated
    or interrupted run leaves no `[RESULT]` line, and the old behaviour surfaced
    that much later as "no numeric metric found. Available: []".
    """
    if not os.path.isfile(path):
        raise SystemExit(f"result file not found: {path}")
    try:
        return _load_json(path)
    except json.JSONDecodeError:
        pass
    parsed = parse_result_log(path)
    if not parsed:
        raise SystemExit(
            f"{path} is not JSON and contains no "
            f"'[RESULT] <tag> <task>: <metrics>' line: the benchmark did not "
            f"finish (interrupted / OOM / crashed before scoring) or its results "
            f"went to another file. Re-run it and pass the log that ends with a "
            f"[RESULT] line."
        )
    return parsed


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
    """Numeric, non-stderr candidates: key -> (depth, key, metric_field, metric_name, value)."""
    out = {}
    for key, value in flat.items():
        depth, metric_field, metric_name = _split_key(key)
        if metric_name.endswith("_stderr"):
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
            "no numeric metric found in this result file. Expected lm-eval "
            "[RESULT] lines or a JSON metrics object; parsed keys: "
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
            raise SystemExit(f"metric '{metric}' not found. Available: {available}")
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
