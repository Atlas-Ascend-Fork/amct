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
    "exact_match,none",
    "exact_match",
    "acc,none",
    "acc",
    "pass@1",
    "pass_at_1",
)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_result(path):
    try:
        return _load_json(path)
    except json.JSONDecodeError:
        return parse_result_log(path)


def _flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(value, next_prefix))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = float(obj)
    return out


def _pick_metric(flat, metric):
    if metric:
        matches = {k: v for k, v in flat.items() if k.endswith(metric)}
        if not matches:
            raise KeyError(f"metric '{metric}' not found. Available: {sorted(flat)}")
        key = sorted(matches)[0]
        return key, matches[key]

    for preferred in PREFERRED_METRICS:
        matches = {k: v for k, v in flat.items() if k.endswith(preferred)}
        if matches:
            key = sorted(matches)[0]
            return key, matches[key]
    if len(flat) == 1:
        key = next(iter(flat))
        return key, flat[key]
    raise KeyError(f"cannot auto-pick metric. Available: {sorted(flat)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True, help="bf16 result log or JSON")
    ap.add_argument("--quant", required=True, help="quant result log or JSON")
    ap.add_argument(
        "--metric",
        default="",
        help="metric suffix to compare, e.g. exact_match,none or pass@1",
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
    quant_key, quant_value = _pick_metric(
        quant_flat, args.metric or bf16_key.split(".")[-1]
    )
    result = {
        "metric": args.metric or bf16_key.split(".")[-1],
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
