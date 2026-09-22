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
"""Qwen3-0.6B dense FFN structured-pruning sample (reconstruct + size_budget).

Issue #183 task 10: prune the dense FFN intermediate dimension with the
``reconstruct`` method under a *parameter-retention budget* (``size_budget =
0.9 / 0.8 / 0.5``) instead of a fixed pruning ratio. amct searches the
candidate ratios in ``--ratio-grid`` and keeps the largest one that still
honours the requested budget.

Search knobs (``size_budget``, ``ratio_grid``) are passed to ``amct.prune()``
and stay out of the method kwargs, matching the merged budget sample
``run_qwen3_6_35b_a3b_pruning_activation_count_budget.py``.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import shlex
import sys
import time
from pathlib import Path

import torch
import torch_npu
from utils import (
    _disable_torchaudio_for_transformers,
    count_params,
    dump_json,
    eval_wikitext2_ppl,
)

_disable_torchaudio_for_transformers()

import amct_pytorch as amct
from amct_pytorch.pruning import PruneReport, prune_diagnose

TAG = "[qwen3.0.6b.reconstruct.budget]"


def build_pileval(tokenizer, n_samples, seq_len, source, seed=42):
    """Calibration batches from the official Pileval mirror or Hugging Face.

    ``modelscope`` mirrors the same ``pile-val-backup`` validation split and is
    the route used when the Hugging Face Xet CDN is unreachable; both feed the
    repo's own ``pileval_awq`` packing helper, so samples stay identical.
    """
    from amct_pytorch.common.datasets.preproc import pileval_awq

    if source == "modelscope":
        from modelscope import MsDataset

        raw = MsDataset.load("swift/pile-val-backup", "default", split="validation")
    else:
        from datasets import load_dataset

        raw = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    return pileval_awq(raw, tokenizer, n_samples, seq_len, seed=seed)


def build_wiki_batches(tokenizer, seq_len, test_file=None):
    """WikiText2 test batches.

    Defaults to ``preproc.get_wiki_inputs`` (the repo-wide official route). With
    ``--wiki-test-file`` the same three steps are replayed on a local parquet
    copy of ``Salesforce/wikitext`` ``wikitext-2-raw-v1`` test split: join rows
    with ``\\n\\n``, tokenize once, then cut non-overlapping ``seq_len`` blocks.
    """
    from amct_pytorch.common.datasets.preproc import get_wiki_inputs

    if not test_file:
        return get_wiki_inputs(tokenizer, seq_len=seq_len)

    import torch as _torch
    from datasets import load_dataset
    from tqdm import tqdm

    ds = load_dataset("parquet", data_files={"test": test_file}, split="test")
    text = "\n\n".join(ds["text"])
    testenc = tokenizer(text, return_tensors="pt").input_ids
    nsamples = testenc.numel() // seq_len
    batches = []
    one = _torch.Size([1, seq_len])
    for i in tqdm(range(nsamples), desc="Chunking wikitext2 (local)"):
        chunk = testenc[:, i * seq_len : (i + 1) * seq_len]
        batches.append(chunk.reshape(one))
    print(f"{TAG} local wikitext2 batches: {len(batches)} (seq_len={seq_len})")
    return batches


def load_qwen3_dense(model_path, torch_dtype=torch.bfloat16, trust_remote_code=False):
    """Load a dense Qwen3 checkpoint strictly offline.

    Qwen3 is natively supported by transformers, so ``trust_remote_code``
    defaults to False and ``local_files_only=True`` keeps the loader from
    executing anything inside the model directory.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if trust_remote_code:
        print(
            f"{TAG} WARNING: trust_remote_code=True executes custom code "
            f"shipped with {model_path}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )
    model.config.use_cache = True
    model.eval()
    return model, tokenizer


def parse_ratio_grid(raw):
    """Parse a comma-separated list of finite pruning ratios inside (0, 1)."""
    ratios = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise argparse.ArgumentTypeError(
                f"ratio {value} is not a finite number between 0 and 1"
            )
        ratios.append(value)
    if not ratios:
        raise argparse.ArgumentTypeError("--ratio-grid must not be empty")
    if len(set(ratios)) != len(ratios):
        raise argparse.ArgumentTypeError("--ratio-grid contains duplicates")
    return tuple(ratios)


def _runtime_args(args):
    return {
        "model_path": args.model_path,
        "trust_remote_code": args.trust_remote_code,
        "save_dir": args.save_dir,
        "save_model": args.save_model,
        "device": args.device,
        "size_budget": args.size_budget,
        "ratio_grid": list(args.ratio_grid),
        "recovery": args.recovery,
        "pileval_source": args.pileval_source,
        "wiki_test_file": args.wiki_test_file if args.wiki_test_file else "huggingface",
        "preview_prune_ratio": args.preview_prune_ratio,
        "calib_samples": args.calib_samples,
        "calib_seq_len": args.calib_seq_len,
        "eval_batches": args.eval_batches if args.eval_batches is not None else "all",
        "seq_len": args.seq_len,
    }


def _prune_cfg(args):
    """Dense-domain reconstruct config; the budget search owns the ratio.

    Only the dense domain is listed, so cnn/moe stay untouched. Attention
    projections and lm_head are excluded through ``skip_layers``.
    """
    return {
        "methods": {
            "dense": {
                "name": "reconstruct",
                "kwargs": {"recovery": args.recovery},
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
        "HF_ENDPOINT",
        "HF_HUB_DISABLE_XET",
        "MODELSCOPE_CACHE",
    )
    env = {key: os.environ[key] for key in env_keys if key in os.environ}
    (save_dir / "run_environment.txt").write_text(
        json.dumps(env, ensure_ascii=False, indent=2) + "\n"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Qwen3-0.6B dense FFN reconstruct pruning sample driven by a "
            "size_budget target (issue #183 task 10)"
        )
    )
    parser.add_argument(
        "--model-path", required=True, help="Local Qwen3-0.6B directory"
    )
    parser.add_argument(
        "--save-dir", required=True, help="Output directory for reports"
    )
    parser.add_argument(
        "--size-budget",
        type=float,
        required=True,
        choices=(0.9, 0.8, 0.5),
        help="Target parameter-retention ratio; issue #183 task 10 uses 0.9/0.8/0.5",
    )
    parser.add_argument(
        "--ratio-grid",
        type=parse_ratio_grid,
        default=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        help="Comma-separated candidate pruning ratios for the budget search",
    )
    parser.add_argument(
        "--recovery",
        choices=("ls", "naive"),
        default="ls",
        help="reconstruct recovery mode (task book keeps the default 'ls')",
    )
    parser.add_argument(
        "--device", default="npu:0", help="Torch device for eval and prune"
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
        help="Pileval calibration sequence length",
    )
    parser.add_argument(
        "--seq-len", type=int, default=4096, help="WikiText2 evaluation sequence length"
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=None,
        help="Optional cap on WikiText2 4096-token batches; omit to use the full test split",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow executing modeling code shipped inside --model-path",
    )
    parser.add_argument(
        "--pileval_source",
        choices=("huggingface", "modelscope"),
        default="huggingface",
        help="Official Pileval source or its ModelScope mirror.",
    )
    parser.add_argument(
        "--wiki-test-file",
        default=None,
        help=(
            "Optional local parquet copy of the Salesforce/wikitext wikitext-2-raw-v1 test "
            "split; omit to use the repo-wide Hugging Face route (preproc.get_wiki_inputs)."
        ),
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Also save the pruned weights and tokenizer",
    )
    args = parser.parse_args()

    if not math.isfinite(args.size_budget) or not 0.0 < args.size_budget <= 1.0:
        parser.error("--size-budget must be finite and between 0 and 1")
    if args.calib_samples <= 0:
        parser.error("--calib-samples must be a positive integer")
    if args.calib_seq_len <= 0:
        parser.error("--calib-seq-len must be a positive integer")
    if args.seq_len <= 0:
        parser.error("--seq-len must be a positive integer")
    if args.eval_batches is not None and args.eval_batches <= 0:
        parser.error("--eval-batches must be a positive integer")
    if (
        not math.isfinite(args.preview_prune_ratio)
        or not 0.0 < args.preview_prune_ratio < 1.0
    ):
        parser.error("--preview-prune-ratio must be a finite number between 0 and 1")
    if not Path(args.model_path).is_dir():
        parser.error(
            f"--model-path does not exist or is not a directory: {args.model_path}"
        )
    return args


def main():
    args = parse_args()
    print(f"{TAG} runtime args:")
    print(json.dumps(_runtime_args(args), ensure_ascii=False, indent=2))
    save_dir = Path(args.save_dir)
    dump_repro_metadata(save_dir)

    model, tokenizer = load_qwen3_dense(
        args.model_path, trust_remote_code=args.trust_remote_code
    )
    model = model.to(args.device)
    runtime_device = args.device

    calib_batches = build_pileval(
        tokenizer,
        n_samples=args.calib_samples,
        seq_len=args.calib_seq_len,
        source=args.pileval_source,
    )
    eval_batches = build_wiki_batches(
        tokenizer, seq_len=args.seq_len, test_file=args.wiki_test_file
    )
    eval_batches = eval_batches[: args.eval_batches]

    # The size-budget search replays the calibration batches through the model, so
    # they must live on the same device as the weights; pileval_awq returns CPU
    # tensors and only the fixed-ratio path moves them internally.
    calib_batches = [batch.to(runtime_device) for batch in calib_batches]

    print(f"{TAG} parameters before: {count_params(model):,}")
    baseline_ppl = eval_wikitext2_ppl(
        model, eval_batches, seq_len=args.seq_len, device=runtime_device
    )
    print(f"{TAG} baseline ppl: {baseline_ppl:.6f}")

    cfg = _prune_cfg(args)
    diagnose = prune_diagnose(
        model,
        data=calib_batches,
        config=copy.deepcopy(cfg),
        prune_ratio=args.preview_prune_ratio,
    )
    print(f"{TAG} prune_diagnose():")
    diagnose_summary = diagnose.summary()
    print(diagnose_summary)

    # Release leftovers of the diagnose dry-run before the real pruning.
    gc.collect()
    if torch_npu.npu.is_available():
        torch_npu.npu.empty_cache()

    report = PruneReport()
    prune_t0 = time.perf_counter()
    amct.prune(
        model,
        cfg,
        data=calib_batches,
        size_budget=args.size_budget,
        ratio_grid=args.ratio_grid,
        report=report,
    )
    prune_seconds = time.perf_counter() - prune_t0

    pruned_ppl = eval_wikitext2_ppl(
        model, eval_batches, seq_len=args.seq_len, device=runtime_device
    )
    report_dict = report.as_dict()
    print(f"{TAG} PruneReport:")
    print(json.dumps(report_dict, ensure_ascii=False, indent=2, default=str))
    cut = 1 - report.params_after / report.params_before
    print(
        f"{TAG} params {report.params_before:,} -> {report.params_after:,}, cut "
        f"{100 * cut:.2f}%, size_budget target {args.size_budget:.2f}"
    )
    print(
        f"{TAG} achieved retention: {report.params_after / report.params_before:.4f} "
        f"(budget_unreachable={report_dict.get('budget_unreachable')}, "
        f"prunable_fraction={report_dict.get('prunable_fraction')})"
    )
    print(f"{TAG} ppl {baseline_ppl:.6f} -> {pruned_ppl:.6f}")
    print(f"{TAG} prune time: {prune_seconds / 60:.2f} min")

    if args.save_model:
        try:
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
        except OSError as exc:
            print(f"{TAG} save_pretrained skipped: {type(exc).__name__}: {exc}")

    dump_json(save_dir / "prune_report.json", report_dict)
    dump_json(save_dir / "run_config.json", _runtime_args(args))
    (save_dir / "diagnose_summary.txt").write_text(diagnose_summary + "\n")
    dump_json(
        save_dir / "run_metrics.json",
        {
            "baseline_ppl": baseline_ppl,
            "pruned_ppl": pruned_ppl,
            "ppl_delta": pruned_ppl - baseline_ppl,
            "params_before": report.params_before,
            "params_after": report.params_after,
            "param_cut_ratio": cut,
            "retained_fraction": report.params_after / report.params_before,
            "budget_unreachable": report_dict.get("budget_unreachable"),
            "prunable_fraction": report_dict.get("prunable_fraction"),
            "warnings": report_dict.get("warnings"),
            "allocation_choice": report_dict.get("allocation_choice"),
            "prune_time_seconds": prune_seconds,
            "prune_time_minutes": prune_seconds / 60,
            "device": runtime_device,
            "size_budget": args.size_budget,
            "ratio_grid": list(args.ratio_grid),
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
    print(f"{TAG} saved to {save_dir}")


if __name__ == "__main__":
    main()
