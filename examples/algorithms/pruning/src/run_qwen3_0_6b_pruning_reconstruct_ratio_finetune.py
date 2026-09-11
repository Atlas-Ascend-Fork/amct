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
"""Qwen3-0.6B Dense FFN reconstruct + finetune sample.

Community issue 183 / task 17:
- domain: dense FFN intermediate dim
- method: reconstruct (recovery defaults to ls)
- mode: ratio only, prune_ratio=0.5
- do NOT pass tolerance / size_budget / evaluator
- calibration: Pileval, default 8 samples, seq_len=512
- evaluation: WikiText2 test split, seq_len=4096
- recovery train: prune_finetune(steps=200, lr=1e-4) on Pileval batches

The script prints the full runtime arguments, then:
baseline PPL -> prune_diagnose() -> amct.prune(..., report=PruneReport())
-> pruned PPL -> prune_finetune() -> post-process PPL.
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
    as_causal_lm_batches,
    build_pileval_batches,
    build_wikitext2_batches,
    count_params,
    _disable_torchaudio_for_transformers,
    dump_json,
    eval_wikitext2_ppl,
    load_qwen3_dense,
)

_disable_torchaudio_for_transformers()

import amct_pytorch as amct  # noqa: E402  (must import after the torchaudio patch)
from amct_pytorch.pruning import (  # noqa: E402
    PruneReport,
    prune_diagnose,
    prune_finetune,
)


def _runtime_args(args):
    return {
        "model_path": args.model_path,
        "save_dir": args.save_dir,
        "save_model": args.save_model,
        "device": args.device,
        "prune_ratio": args.prune_ratio,
        "calib_samples": args.calib_samples,
        "calib_seq_len": args.calib_seq_len,
        "eval_batches": args.eval_batches if args.eval_batches is not None else "all",
        "seq_len": args.seq_len,
        "finetune_steps": args.finetune_steps,
        "finetune_lr": args.finetune_lr,
        "method": "reconstruct",
        "mode": "ratio",
        "recovery": "ls (library default)",
    }


def _prune_cfg(prune_ratio):
    return {
        "methods": {
            "dense": {
                "name": "reconstruct",
                "kwargs": {"prune_ratio": prune_ratio},
            },
        },
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
        "HF_ENDPOINT",
        "HF_HUB_DISABLE_XET",
    )
    env = {key: os.environ[key] for key in env_keys if key in os.environ}
    (save_dir / "run_environment.txt").write_text(
        json.dumps(env, ensure_ascii=False, indent=2) + "\n"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3-0.6B reconstruct ratio + finetune pruning sample"
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Relative or absolute path to the local Qwen3-0.6B weights",
    )
    parser.add_argument(
        "--save-dir",
        default="./outputs/qwen3_0_6b_reconstruct_ratio_finetune",
        help="Relative output directory for reports and optional pruned weights",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the pruned model/tokenizer in save_dir",
    )
    parser.add_argument(
        "--device",
        default="npu:0",
        help="Single-card device for load, prune, eval, and finetune",
    )
    parser.add_argument(
        "--prune-ratio",
        type=float,
        default=0.5,
        help="Dense-domain prune ratio for ratio mode (task 17 uses 0.5)",
    )
    parser.add_argument(
        "--calib-samples",
        type=int,
        default=8,
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
        "--finetune-steps",
        type=int,
        default=200,
        help="prune_finetune steps on Pileval batches",
    )
    parser.add_argument(
        "--finetune-lr",
        type=float,
        default=1e-4,
        help="prune_finetune learning rate",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("[qwen3.0.6b.reconstruct.ratio.finetune] runtime args:")
    print(json.dumps(_runtime_args(args), ensure_ascii=False, indent=2))
    save_dir = Path(args.save_dir)
    dump_repro_metadata(save_dir)

    model, tokenizer = load_qwen3_dense(args.model_path, device_map=None)
    model = model.to(args.device)
    runtime_device = args.device

    # Prune-time clock: calibration collection through prune() only.
    prune_t0 = time.perf_counter()
    calib_batches = build_pileval_batches(
        tokenizer, n_samples=args.calib_samples, seq_len=args.calib_seq_len
    )
    eval_batches = build_wikitext2_batches(tokenizer, seq_len=args.seq_len)
    eval_batches = eval_batches[: args.eval_batches]

    print(
        f"[qwen3.0.6b.reconstruct.ratio.finetune] parameters before: {count_params(model):,}"
    )
    baseline_ppl = eval_wikitext2_ppl(
        model, eval_batches, seq_len=args.seq_len, device=runtime_device
    )
    print(f"[qwen3.0.6b.reconstruct.ratio.finetune] baseline ppl: {baseline_ppl:.6f}")

    cfg = _prune_cfg(args.prune_ratio)
    diagnose = prune_diagnose(
        model,
        data=calib_batches,
        config=copy.deepcopy(cfg),
        prune_ratio=args.prune_ratio,
    )
    print("[qwen3.0.6b.reconstruct.ratio.finetune] prune_diagnose():")
    diagnose_summary = diagnose.summary()
    print(diagnose_summary)

    gc.collect()
    if torch_npu.npu.is_available():
        torch_npu.npu.empty_cache()

    report = PruneReport()
    amct.prune(model, cfg, data=calib_batches, report=report)
    prune_seconds = time.perf_counter() - prune_t0

    params_before = report.params_before
    params_after = report.params_after
    cut_ratio = 1 - params_after / params_before if params_before else 0.0
    if params_after >= params_before:
        raise RuntimeError(
            "Task 17 requires an actual 0.5-ratio prune; params_after "
            f"{params_after} is not smaller than params_before {params_before}."
        )

    pruned_ppl = eval_wikitext2_ppl(
        model, eval_batches, seq_len=args.seq_len, device=runtime_device
    )
    print("[qwen3.0.6b.reconstruct.ratio.finetune] PruneReport:")
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    print(
        f"[qwen3.0.6b.reconstruct.ratio.finetune] params {params_before:,} -> "
        f"{params_after:,}, cut {100 * cut_ratio:.2f}%"
    )
    print(
        f"[qwen3.0.6b.reconstruct.ratio.finetune] ppl {baseline_ppl:.6f} -> "
        f"{pruned_ppl:.6f}"
    )
    print(
        f"[qwen3.0.6b.reconstruct.ratio.finetune] prune time: {prune_seconds / 60:.2f} min"
    )

    finetune_t0 = time.perf_counter()
    ft_stats = prune_finetune(
        model,
        data=as_causal_lm_batches(calib_batches),
        steps=args.finetune_steps,
        lr=args.finetune_lr,
        device=runtime_device,
    )
    finetune_seconds = time.perf_counter() - finetune_t0
    model.eval()
    post_ppl = eval_wikitext2_ppl(
        model, eval_batches, seq_len=args.seq_len, device=runtime_device
    )
    print("[qwen3.0.6b.reconstruct.ratio.finetune] prune_finetune():")
    print(
        json.dumps(
            {
                "steps": ft_stats.get("steps"),
                "initial_loss": ft_stats.get("initial_loss"),
                "final_loss": ft_stats.get("final_loss"),
                "lr": args.finetune_lr,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"[qwen3.0.6b.reconstruct.ratio.finetune] post-process ppl: {post_ppl:.6f}")
    print(
        f"[qwen3.0.6b.reconstruct.ratio.finetune] finetune time: {finetune_seconds / 60:.2f} min"
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    if args.save_model:
        try:
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
        except OSError as exc:
            print(
                f"[qwen3.0.6b.reconstruct.ratio.finetune] save_pretrained skipped: "
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
            "post_ppl": post_ppl,
            "params_before": params_before,
            "params_after": params_after,
            "param_cut_ratio": cut_ratio,
            "prune_time_seconds": prune_seconds,
            "prune_time_minutes": prune_seconds / 60,
            "finetune_time_seconds": finetune_seconds,
            "finetune_time_minutes": finetune_seconds / 60,
            "finetune_steps": ft_stats.get("steps"),
            "finetune_lr": args.finetune_lr,
            "finetune_initial_loss": ft_stats.get("initial_loss"),
            "finetune_final_loss": ft_stats.get("final_loss"),
            "device": runtime_device,
            "prune_ratio": args.prune_ratio,
            "calib_samples": args.calib_samples,
            "calib_seq_len": args.calib_seq_len,
            "eval_batches": args.eval_batches
            if args.eval_batches is not None
            else "all",
            "seq_len": args.seq_len,
        },
    )
    print(f"[qwen3.0.6b.reconstruct.ratio.finetune] saved to {save_dir}")


if __name__ == "__main__":
    main()
