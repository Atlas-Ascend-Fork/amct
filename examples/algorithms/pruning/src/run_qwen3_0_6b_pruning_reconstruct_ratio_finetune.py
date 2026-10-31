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
"""Qwen3-0.6B dense FFN structured-pruning sample (reconstruct + finetune / ratio).

Task focus (issue 183, task 17):
- model: Qwen3-0.6B (dense)
- domain: dense FFN intermediate dim
- method: reconstruct (least-squares recovery) + finetune (post-processing)
- mode: fixed prune_ratio = 0.5
- calibration set: Pileval (sample count / length configurable, see README)
- evaluation set: WikiText2 test split, seq_len=4096
- finetune set: WikiText2 train split, non-overlapping 512-token blocks

The script prints the full runtime arguments, runs prune_diagnose() first, applies a
fixed-ratio reconstruct pruning, fine-tunes the pruned model for recovery, and saves
the pruned/finetuned model plus reports if requested.

Run:
    python3 src/run_qwen3_0_6b_pruning_reconstruct_ratio_finetune.py \
      --model-path ./path/to/Qwen3-0.6B \
      --save-dir ./outputs/qwen3_0_6b_reconstruct_ratio_finetune/r0_5 \
      --prune-ratio 0.5
"""

from __future__ import annotations

import logging

import argparse
import copy
import gc
import json
import os
import shlex
import sys
import time
from pathlib import Path

import torch
import torch_npu

from utils import (
    build_pileval_batches,
    build_wikitext2_batches,
    count_params,
    disable_torchaudio,
    dump_json,
    eval_wikitext2_ppl,
    load_model,
    to_device_batches,
)

disable_torchaudio()  # must run before transformers imports torchaudio

import amct_pytorch as amct  # noqa: E402  (must import after the torchaudio patch)
from amct_pytorch.pruning import PruneReport, prune_diagnose, prune_finetune  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logging.getLogger("Log").propagate = False


def build_wikitext2_train_batches(tokenizer, n_batches=300, seq_len=512):
    """Non-overlapping contiguous token blocks from the WikiText2 train split.

    Used only for finetune (recovery); the WikiText2 test split is never used
    for training.
    """
    from datasets import load_dataset

    train = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train')
    ids = tokenizer('\n\n'.join(train['text']), return_tensors='pt').input_ids
    n_chunks = ids.numel() // seq_len
    n = min(n_batches, n_chunks)
    batches = []
    for i in range(n):
        chunk = ids[:, i * seq_len : (i + 1) * seq_len].reshape(1, seq_len)
        batches.append({'input_ids': chunk})
    return batches


def _runtime_args(args):
    return {
        "model_path": args.model_path,
        "trust_remote_code": args.trust_remote_code,
        "save_dir": args.save_dir,
        "save_model": args.save_model,
        "device": args.device,
        "prune_ratio": args.prune_ratio,
        "recovery": args.recovery,
        "finetune": args.finetune,
        "finetune_steps": args.finetune_steps if args.finetune else "disabled",
        "finetune_lr": args.finetune_lr if args.finetune else "disabled",
        "finetune_warmup": args.finetune_warmup if args.finetune else "disabled",
        "finetune_batches": args.finetune_batches if args.finetune else "disabled",
        "finetune_seq_len": args.finetune_seq_len if args.finetune else "disabled",
        "preview_prune_ratio": args.preview_prune_ratio,
        "calib_samples": args.calib_samples,
        "calib_seq_len": args.calib_seq_len,
        "eval_batches": args.eval_batches if args.eval_batches is not None else "all",
        "seq_len": args.seq_len,
    }


def _prune_cfg(args):
    """Fixed-ratio reconstruct config for the dense FFN domain.

    Only the dense domain is listed, so cnn/moe stay untouched. Attention
    projections and lm_head are excluded via skip_layers for the fixed-ratio mode.
    """
    return {
        "methods": {
            "dense": {
                "name": "reconstruct",
                "kwargs": {
                    "prune_ratio": args.prune_ratio,
                    "recovery": args.recovery,
                },
            },
        },
        "skip_layers": ["self_attn", "lm_head"],
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
        description="Qwen3-0.6B dense FFN reconstruct ratio + finetune pruning sample"
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Relative or absolute path to the local Qwen3-0.6B weights",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help=(
            "Allow transformers to execute custom Python shipped with the "
            "model. Off by default; Qwen3-0.6B is natively supported by "
            "transformers and does not need this flag. Only enable for a "
            "local checkpoint you trust."
        ),
    )
    parser.add_argument(
        "--save-dir",
        default="./outputs/qwen3_0_6b_reconstruct_ratio_finetune/r0_5",
        help="Relative output directory for the pruned model and reports",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the finetuned model/tokenizer in save_dir",
    )
    parser.add_argument(
        "--device",
        default="npu:0",
        help="Single NPU device used for calibration/evaluation/finetune",
    )
    parser.add_argument(
        "--prune-ratio",
        type=float,
        default=0.5,
        help="Fixed prune ratio for dense FFN intermediate-dim pruning",
    )
    parser.add_argument(
        "--recovery",
        default="ls",
        choices=("ls", "bias", "none"),
        help="reconstruct recovery mode: ls (least squares) / bias / none",
    )
    parser.add_argument(
        "--finetune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run recovery finetune after pruning (default: enabled)",
    )
    parser.add_argument(
        "--finetune-steps",
        type=int,
        default=300,
        help="Finetune steps on WikiText2 train blocks",
    )
    parser.add_argument(
        "--finetune-lr",
        type=float,
        default=1e-2,
        help="Finetune learning rate (momentum-free SGD)",
    )
    parser.add_argument(
        "--finetune-warmup",
        type=int,
        default=20,
        help="Linear warmup steps for finetune",
    )
    parser.add_argument(
        "--finetune-batches",
        type=int,
        default=300,
        help="Number of non-overlapping WikiText2 train blocks for finetune",
    )
    parser.add_argument(
        "--finetune-seq-len",
        type=int,
        default=512,
        help="Finetune block sequence length",
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
        default=8,
        help="Number of Pileval calibration samples "
        "(dense reconstruct LS recovery needs calib rows >= kept neurons)",
    )
    parser.add_argument(
        "--calib-seq-len",
        type=int,
        default=2048,
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
    args = parser.parse_args()
    if args.finetune_steps <= 0:
        parser.error("--finetune-steps must be a positive integer")
    if args.finetune_batches <= 0:
        parser.error("--finetune-batches must be a positive integer")
    if args.calib_samples <= 0:
        parser.error("--calib-samples must be a positive integer")
    if args.finetune_seq_len <= 0:
        parser.error("--finetune-seq-len must be a positive integer")
    if not 0.0 <= args.prune_ratio < 1.0:
        parser.error("--prune-ratio must be in [0.0, 1.0)")
    if not 0.0 <= args.preview_prune_ratio < 1.0:
        parser.error("--preview-prune-ratio must be in [0.0, 1.0)")
    if args.eval_batches is not None and args.eval_batches <= 0:
        parser.error("--eval-batches must be a positive integer")
    return args


def main():
    args = parse_args()
    logger.info("[qwen3.0.6b.reconstruct.ratio.finetune] runtime args:")
    logger.info(json.dumps(_runtime_args(args), ensure_ascii=False, indent=2))
    save_dir = Path(args.save_dir)
    dump_repro_metadata(save_dir)

    if args.trust_remote_code:
        logger.info(
            "WARNING: --trust-remote-code is enabled. Custom Python from the "
            "model directory may execute under the current user. Use only "
            "with checkpoints you trust.",
        )
    model, tokenizer = load_model(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model = model.to(args.device)
    runtime_device = args.device

    collect_t0 = time.perf_counter()
    calib_batches = build_pileval_batches(
        tokenizer, n_samples=args.calib_samples, seq_len=args.calib_seq_len
    )
    calib_data_seconds = time.perf_counter() - collect_t0
    # prune_diagnose()'s dry-run forwards calibration batches without moving
    # them, so align them with the model device here (the real prune's
    # run_calibration_epoch moves batches via move_batch_to_device, making
    # this a no-op there).
    calib_batches = to_device_batches(calib_batches, args.device)
    eval_batches = build_wikitext2_batches(tokenizer, seq_len=args.seq_len)
    eval_batches = eval_batches[: args.eval_batches]

    logger.info(
        f"[qwen3.0.6b.reconstruct.ratio.finetune] parameters before: "
        f"{count_params(model):,}"
    )
    baseline_ppl = eval_wikitext2_ppl(
        model, eval_batches, seq_len=args.seq_len, device=runtime_device
    )
    logger.info(
        f"[qwen3.0.6b.reconstruct.ratio.finetune] baseline ppl: {baseline_ppl:.6f}"
    )

    cfg = _prune_cfg(args)
    diagnose_t0 = time.perf_counter()
    diagnose = prune_diagnose(
        model,
        data=calib_batches,
        config=copy.deepcopy(cfg),
        prune_ratio=args.preview_prune_ratio,
    )
    diagnose_seconds = time.perf_counter() - diagnose_t0
    logger.info(
        f"[qwen3.0.6b.reconstruct.ratio.finetune] diagnose time: {diagnose_seconds / 60:.2f} min"
    )
    logger.info("[qwen3.0.6b.reconstruct.ratio.finetune] prune_diagnose():")
    diagnose_summary = diagnose.summary()
    logger.info(diagnose_summary)

    # Release leftovers of the diagnose dry-run before the real pruning.
    gc.collect()
    if torch_npu.npu.is_available():
        torch_npu.npu.empty_cache()

    report = PruneReport()
    prune_t0 = time.perf_counter()
    amct.prune(model, cfg, data=calib_batches, report=report)
    prune_seconds = time.perf_counter() - prune_t0
    collect_to_pruned_seconds = time.perf_counter() - collect_t0

    pruned_ppl = eval_wikitext2_ppl(
        model, eval_batches, seq_len=args.seq_len, device=runtime_device
    )
    logger.info("[qwen3.0.6b.reconstruct.ratio.finetune] PruneReport:")
    logger.info(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    logger.info(
        f"[qwen3.0.6b.reconstruct.ratio.finetune] params {report.params_before:,} -> "
        f"{report.params_after:,}, cut {100 * (1 - report.params_after / report.params_before):.2f}%"
    )
    logger.info(
        f"[qwen3.0.6b.reconstruct.ratio.finetune] ppl {baseline_ppl:.6f} -> "
        f"{pruned_ppl:.6f}"
    )
    logger.info(
        f"[qwen3.0.6b.reconstruct.ratio.finetune] prune time: {prune_seconds / 60:.2f} min"
    )

    # Post-processing: recovery finetune on WikiText2 train blocks.
    finetuned_ppl = None
    finetune_seconds = None
    finetune_stats = None
    if args.finetune:
        train_batches = build_wikitext2_train_batches(
            tokenizer, n_batches=args.finetune_batches, seq_len=args.finetune_seq_len
        )
        model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=args.finetune_lr, momentum=0.0)
        ft_t0 = time.perf_counter()
        finetune_stats = prune_finetune(
            model,
            train_batches,
            steps=args.finetune_steps,
            lr=args.finetune_lr,
            warmup=args.finetune_warmup,
            optimizer=optimizer,
            device=runtime_device,
            grad_clip=1.0,
            log_every=50,
        )
        finetune_seconds = time.perf_counter() - ft_t0
        model.config.use_cache = True
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        finetuned_ppl = eval_wikitext2_ppl(
            model, eval_batches, seq_len=args.seq_len, device=runtime_device
        )
        logger.info(
            f"[qwen3.0.6b.reconstruct.ratio.finetune] finetune loss "
            f"{finetune_stats['initial_loss']:.6f} -> {finetune_stats['final_loss']:.6f}, "
            f"{finetune_seconds / 60:.2f} min"
        )
        logger.info(
            f"[qwen3.0.6b.reconstruct.ratio.finetune] finetuned ppl: {finetuned_ppl:.6f}"
        )

    save_model_status = "not_requested"
    if args.save_model:
        try:
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            save_model_status = "saved"
        except Exception as exc:  # NPU OOM raises RuntimeError, not OSError
            save_model_status = f"failed: {type(exc).__name__}: {exc}"
            logger.info(
                f"[qwen3.0.6b.reconstruct.ratio.finetune] save_pretrained skipped: {save_model_status}"
            )
    dump_json(save_dir / "prune_report.json", report.as_dict())
    dump_json(save_dir / "run_config.json", _runtime_args(args))
    (save_dir / "diagnose_summary.txt").write_text(diagnose_summary + "\n")
    dump_json(
        save_dir / "run_metrics.json",
        {
            "baseline_ppl": baseline_ppl,
            "pruned_ppl": pruned_ppl,
            "finetuned_ppl": finetuned_ppl,
            "finetune_seconds": finetune_seconds,
            "finetune_stats": finetune_stats,
            "params_before": report.params_before,
            "params_after": report.params_after,
            "param_cut_ratio": 1 - report.params_after / report.params_before,
            "prune_time_seconds": prune_seconds,
            "prune_time_minutes": prune_seconds / 60,
            "calib_data_seconds": calib_data_seconds,
            "diagnose_time_seconds": diagnose_seconds,
            "diagnose_time_minutes": diagnose_seconds / 60,
            "collect_to_pruned_seconds": collect_to_pruned_seconds,
            "collect_to_pruned_minutes": collect_to_pruned_seconds / 60,
            "save_model_status": save_model_status,
            "device": runtime_device,
            "prune_ratio": args.prune_ratio,
            "recovery": args.recovery,
            "preview_prune_ratio": args.preview_prune_ratio,
            "calib_samples": args.calib_samples,
            "calib_seq_len": args.calib_seq_len,
            "eval_batches": args.eval_batches
            if args.eval_batches is not None
            else "all",
            "seq_len": args.seq_len,
        },
    )
    if save_model_status == "saved":
        logger.info(f"[qwen3.0.6b.reconstruct.ratio.finetune] saved to {save_dir}")
    else:
        logger.info(
            f"[qwen3.0.6b.reconstruct.ratio.finetune] model weights not saved "
            f"({save_model_status}); metrics still written to {save_dir}"
        )


if __name__ == "__main__":
    main()
