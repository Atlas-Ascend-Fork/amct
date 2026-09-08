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
"""Qwen3.6-35B-A3B MoE expert pruning sample.

Task focus:
- method: activation_count
- mode: tolerance
- calibration set: Pileval
- evaluation set: WikiText2 test split
- eval sequence length: 4096

The script prints the full runtime arguments, runs prune_diagnose() first, then
applies tolerance-driven pruning and saves the pruned model/report if requested.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import shlex
import sys
import time
from pathlib import Path

import torch_npu

from utils import (
    NegativePplEvaluator,
    build_pileval_batches,
    build_wikitext2_batches,
    count_params,
    _disable_torchaudio_for_transformers,
    dump_json,
    eval_wikitext2_ppl,
    load_qwen36_moe,
)

_disable_torchaudio_for_transformers()

import amct_pytorch as amct  # noqa: E402  (must import after the torchaudio patch)
from amct_pytorch.pruning import PruneReport, prune_diagnose  # noqa: E402


def _parse_ratio(text):
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("ratio grid must not be empty")
    try:
        return [float(item) for item in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "ratio grid must be a comma-separated list of floats"
        ) from exc


def _runtime_args(args):
    return {
        "model_path": args.model_path,
        "save_dir": args.save_dir,
        "save_model": args.save_model,
        "device_map": args.device_map,
        "device": args.device,
        "tolerance": args.tolerance,
        "preview_prune_ratio": args.preview_prune_ratio,
        "calib_samples": args.calib_samples,
        "calib_seq_len": args.calib_seq_len,
        "eval_batches": args.eval_batches if args.eval_batches is not None else "all",
        "seq_len": args.seq_len,
        "ratio_grid": args.ratio_grid
        if args.ratio_grid is not None
        else "builtin_default",
    }


def _prune_cfg():
    return {
        "methods": {
            "moe": {
                "name": "activation_count",
                "kwargs": {},
            },
        },
        "missing_data_policy": "warn_skip",
    }


def dump_repro_metadata(save_dir):
    """Persist the exact invocation and accelerator-related environment."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "run_command.txt").write_text(
        shlex.join([sys.executable, *sys.argv]) + "\n"
    )
    env_keys = (
        "PYTHONPATH",
        "PATH",
        "ASCEND_VISIBLE_DEVICES",
        "NPU_VISIBLE_DEVICES",
        "NPU-VISIBLE-DEVICES",
        "ASCEND_TOOLKIT_HOME",
        "ASCEND_HOME_PATH",
        "ASCEND_OPP_PATH",
        "ASCEND_RUNTIME_OPTIONS",
        "LD_LIBRARY_PATH",
        "HF_DATASETS_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    )
    env = {key: os.environ[key] for key in env_keys if key in os.environ}
    (save_dir / "run_environment.txt").write_text(
        json.dumps(env, ensure_ascii=False, indent=2) + "\n"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3.6-35B-A3B MoE expert pruning sample"
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Relative or absolute path to the local Qwen3.6-35B-A3B weights",
    )
    parser.add_argument(
        "--save-dir",
        default="./outputs/qwen3.6_activation_count_tolerance",
        help="Relative output directory for the pruned model and reports",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the pruned model/tokenizer shards in save_dir",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        choices=("auto", "single"),
        help="Model placement mode; auto shards across visible NPU cards, single keeps one device",
    )
    parser.add_argument(
        "--device",
        default="npu:0",
        help="Single-device fallback, or the first input device when device_map=auto",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        required=True,
        help="Allowed PPL increase for tolerance search",
    )
    parser.add_argument(
        "--preview-prune-ratio",
        type=float,
        default=0.1,
        help="Dry-run prune ratio used by prune_diagnose()",
    )
    parser.add_argument(
        "--calib-samples",
        type=int,
        default=1,
        help="Number of Pileval calibration samples",
    )
    parser.add_argument(
        "--calib-seq-len",
        type=int,
        default=512,
        help="Calibration sequence length for Pileval",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=4096,
        help="WikiText2 evaluation sequence length",
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=None,
        help="Optional cap on WikiText2 4096-token batches; omit to use the full test split",
    )
    parser.add_argument(
        "--ratio-grid",
        type=_parse_ratio,
        default=None,
        help="Optional comma-separated ratio grid; default is the built-in 0.1..0.8 grid",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("[qwen3.6.activation_count.tolerance] runtime args:")
    print(json.dumps(_runtime_args(args), ensure_ascii=False, indent=2))
    save_dir = Path(args.save_dir)
    dump_repro_metadata(save_dir)

    model, tokenizer = load_qwen36_moe(
        args.model_path, device_map="auto" if args.device_map == "auto" else None
    )
    runtime_device = str(next(model.parameters()).device)
    if args.device_map == "single":
        model = model.to(args.device)
        runtime_device = args.device
    calib_batches = build_pileval_batches(
        tokenizer, n_samples=args.calib_samples, seq_len=args.calib_seq_len
    )
    eval_batches = build_wikitext2_batches(tokenizer, seq_len=args.seq_len)
    eval_batches = eval_batches[: args.eval_batches]

    print(
        f"[qwen3.6.activation_count.tolerance] parameters before: {count_params(model):,}"
    )
    baseline_ppl = eval_wikitext2_ppl(
        model, eval_batches, seq_len=args.seq_len, device=runtime_device
    )
    print(f"[qwen3.6.activation_count.tolerance] baseline ppl: {baseline_ppl:.6f}")

    cfg = _prune_cfg()
    diagnose = prune_diagnose(
        model,
        data=calib_batches,
        config=copy.deepcopy(cfg),
        prune_ratio=args.preview_prune_ratio,
        tolerance=args.tolerance,
    )
    print("[qwen3.6.activation_count.tolerance] prune_diagnose():")
    diagnose_summary = diagnose.summary()
    print(diagnose_summary)

    # Release leftovers of the diagnose dry-run (a failed full-model copy can
    # leave cyclic garbage alive on a 2-card 35B deployment) before searching.
    gc.collect()
    if torch_npu.npu.is_available():
        torch_npu.npu.empty_cache()

    report = PruneReport()
    eval_metric = NegativePplEvaluator(
        eval_batches, seq_len=args.seq_len, device=runtime_device
    )
    prune_kwargs = {
        "data": calib_batches,
        "tolerance": args.tolerance,
        "evaluator": eval_metric,
        "report": report,
    }
    if args.ratio_grid is not None:
        prune_kwargs["ratio_grid"] = args.ratio_grid
    prune_t0 = time.perf_counter()
    amct.prune(model, cfg, **prune_kwargs)
    prune_seconds = time.perf_counter() - prune_t0

    pruned_ppl = eval_wikitext2_ppl(
        model, eval_batches, seq_len=args.seq_len, device=runtime_device
    )
    print("[qwen3.6.activation_count.tolerance] PruneReport:")
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    print(
        f"[qwen3.6.activation_count.tolerance] params {report.params_before:,} -> "
        f"{report.params_after:,}, cut {100 * (1 - report.params_after / report.params_before):.2f}%"
    )
    print(
        f"[qwen3.6.activation_count.tolerance] ppl {baseline_ppl:.6f} -> "
        f"{pruned_ppl:.6f}"
    )
    print(
        f"[qwen3.6.activation_count.tolerance] prune time: {prune_seconds / 60:.2f} min"
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    if args.save_model:
        try:
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
        except OSError as exc:
            print(
                f"[qwen3.6.activation_count.tolerance] save_pretrained skipped: "
                f"{type(exc).__name__}: {exc}"
            )
    dump_json(save_dir / "prune_report.json", report.as_dict())
    dump_json(save_dir / "run_config.json", _runtime_args(args))
    (save_dir / "diagnose_summary.txt").write_text(diagnose_summary + "\n")
    dump_json(
        save_dir / "run_metrics.json",
        {
            "baseline_ppl": baseline_ppl,
            "pruned_ppl": pruned_ppl,
            "params_before": report.params_before,
            "params_after": report.params_after,
            "param_cut_ratio": 1 - report.params_after / report.params_before,
            "prune_time_seconds": prune_seconds,
            "prune_time_minutes": prune_seconds / 60,
            "device_map": args.device_map,
            "device": runtime_device,
            "tolerance": args.tolerance,
            "preview_prune_ratio": args.preview_prune_ratio,
            "calib_samples": args.calib_samples,
            "calib_seq_len": args.calib_seq_len,
            "eval_batches": args.eval_batches
            if args.eval_batches is not None
            else "all",
            "seq_len": args.seq_len,
            "ratio_grid": args.ratio_grid
            if args.ratio_grid is not None
            else "builtin_default",
        },
    )
    print(f"[qwen3.6.activation_count.tolerance] saved to {save_dir}")


if __name__ == "__main__":
    main()
