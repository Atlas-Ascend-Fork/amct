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
"""Qwen3-0.6B dense FFN intermediate-dim pruning sample (issue 183, task 5).

Task focus:
- method: low_variance (dense FFN intermediate dim)
- mode: ratio (fixed prune_ratio, no search)
- calibration set: Pileval (mit-han-lab/pile-val-backup)
- evaluation set: WikiText2 (wikitext-2-raw-v1, test split)
- eval sequence length: 4096

The script prints the full runtime arguments, runs prune_diagnose() first, then
applies fixed-ratio pruning and reports baseline/pruned PPL and PruneReport.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import sys
import time
from pathlib import Path

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

import amct_pytorch as amct  # noqa: E402
from amct_pytorch.pruning import PruneReport, prune_diagnose  # noqa: E402

import torch  # noqa: E402


def _runtime_args(args):
    return {
        "model_path": args.model_path,
        "save_dir": args.save_dir,
        "save_model": args.save_model,
        "device": args.device,
        "prune_ratio": args.prune_ratio,
        "preview_prune_ratio": args.preview_prune_ratio,
        "skip_diagnose": args.skip_diagnose,
        "calib_samples": args.calib_samples,
        "calib_seq_len": args.calib_seq_len,
        "eval_batches": args.eval_batches if args.eval_batches is not None else "all",
        "seq_len": args.seq_len,
    }


def _positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer (>= 1)")
    return value


def _prune_cfg(prune_ratio):
    return {
        "methods": {
            "dense": {
                "name": "low_variance",
                "kwargs": {"prune_ratio": prune_ratio},
            },
        },
        "missing_data_policy": "warn_skip",
    }


def _file_head(path):
    """Value of the first "Version=..."-style line of a CANN info file, or None."""
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return None
    keys = ("Version=", "version=")
    line = next((line for line in lines if line.startswith(keys)), None)
    return line.split("=", 1)[1].strip() if line else None


def _cann_versions():
    """Best-effort CANN toolkit / driver versions (None when not installed)."""
    toolkit_home = os.environ.get("ASCEND_TOOLKIT_HOME", "")
    toolkit = (
        _file_head(Path(toolkit_home) / "opp" / "version.info")
        if toolkit_home
        else None
    )
    driver = _file_head(Path("/usr/local/Ascend/driver/version.info"))
    return {"cann_toolkit": toolkit, "npu_driver": driver}


def _package_versions():
    """Versions of every package the PPL results depend on (None when unknown)."""
    import importlib.metadata

    def _dist(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    def _amct_version():
        # Prefer the version of the package actually imported: installed wheels
        # carry their own .version next to __init__.py, which may differ from
        # the source checkout the sample script lives in.
        try:
            import amct_pytorch

            content = (
                (Path(amct_pytorch.__file__).parent / ".version").read_text().strip()
            )
            if content:
                return content
        except (ImportError, OSError):
            pass
        return _dist("amct_pytorch")

    versions = {
        "python": sys.version.split()[0],
        "amct_pytorch": _amct_version(),
        **{name: _dist(name) for name in ("torch", "torch_npu", "transformers")},
        **_cann_versions(),
    }
    return versions


def dump_repro_metadata(save_dir):
    """Persist the exact invocation, versions, and accelerator environment."""
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
        "HF_ENDPOINT",
        "HF_DATASETS_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    )
    env = {key: os.environ[key] for key in env_keys if key in os.environ}
    env["versions"] = _package_versions()
    (save_dir / "run_environment.txt").write_text(
        json.dumps(env, ensure_ascii=False, indent=2) + "\n"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3-0.6B dense FFN low_variance pruning sample (ratio mode)"
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Relative or absolute path to the local Qwen3-0.6B weights",
    )
    parser.add_argument(
        "--save-dir",
        default="./outputs/qwen3_0_6b_low_variance_ratio",
        help="Relative output directory for the pruned model and reports",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the pruned model/tokenizer shards in save_dir",
    )
    parser.add_argument(
        "--device",
        default="npu:0",
        help="Device for model placement and evaluation",
    )
    parser.add_argument(
        "--prune-ratio",
        type=float,
        required=True,
        help="Fixed FFN intermediate-dim prune ratio, e.g. 0.1 / 0.2 / 0.5",
    )
    parser.add_argument(
        "--preview-prune-ratio",
        type=float,
        default=0.1,
        help="Dry-run prune ratio used by prune_diagnose()",
    )
    parser.add_argument(
        "--skip-diagnose",
        action="store_true",
        help="Skip the prune_diagnose() dry-run (saves its runtime)",
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
        type=_positive_int,
        default=None,
        help="Optional cap on WikiText2 4096-token batches; omit to use the full test split",
    )
    args = parser.parse_args()
    # Reject invalid values up front: an empty calibration set silently skips
    # pruning (only a warning in the report) and an out-of-range ratio would
    # otherwise surface after the full baseline evaluation has already run.
    if not 0.0 <= args.prune_ratio < 1.0:
        parser.error("--prune-ratio must be in [0.0, 1.0)")
    if not 0.0 <= args.preview_prune_ratio < 1.0:
        parser.error("--preview-prune-ratio must be in [0.0, 1.0)")
    if args.calib_samples < 1:
        parser.error("--calib-samples must be a positive integer (>= 1)")
    return args


def main():
    args = parse_args()
    print("[qwen3_0_6b.low_variance.ratio] runtime args:")
    print(json.dumps(_runtime_args(args), ensure_ascii=False, indent=2))
    save_dir = Path(args.save_dir)
    dump_repro_metadata(save_dir)

    model, tokenizer = load_model(args.model_path, torch_dtype=torch.bfloat16)
    model = model.to(args.device)
    calib_batches = build_pileval_batches(
        tokenizer, n_samples=args.calib_samples, seq_len=args.calib_seq_len
    )
    # prune_diagnose()'s dry-run forwards calibration batches without moving
    # them, so align them with the model device here (the real prune's
    # run_calibration_epoch moves batches via move_batch_to_device, making
    # this a no-op there).
    calib_batches = to_device_batches(calib_batches, args.device)
    eval_batches = build_wikitext2_batches(tokenizer, seq_len=args.seq_len)
    eval_batches = eval_batches[: args.eval_batches]

    print(f"[qwen3_0_6b.low_variance.ratio] parameters before: {count_params(model):,}")
    baseline_ppl = eval_wikitext2_ppl(
        model, eval_batches, seq_len=args.seq_len, device=args.device
    )
    print(f"[qwen3_0_6b.low_variance.ratio] baseline ppl: {baseline_ppl:.6f}")

    cfg = _prune_cfg(args.prune_ratio)
    diagnose_summary = None
    diagnose_seconds = None
    if args.skip_diagnose:
        print(
            "[qwen3_0_6b.low_variance.ratio] prune_diagnose(): skipped (--skip-diagnose)"
        )
    else:
        diagnose_t0 = time.perf_counter()
        diagnose = prune_diagnose(
            model,
            data=calib_batches,
            config=copy.deepcopy(cfg),
            prune_ratio=args.preview_prune_ratio,
        )
        diagnose_seconds = time.perf_counter() - diagnose_t0
        print("[qwen3_0_6b.low_variance.ratio] prune_diagnose():")
        diagnose_summary = diagnose.summary()
        print(diagnose_summary)
        print(
            f"[qwen3_0_6b.low_variance.ratio] diagnose time: {diagnose_seconds / 60:.2f} min"
        )

    report = PruneReport()
    prune_t0 = time.perf_counter()
    amct.prune(model, cfg, data=calib_batches, report=report)
    prune_seconds = time.perf_counter() - prune_t0

    pruned_ppl = eval_wikitext2_ppl(
        model, eval_batches, seq_len=args.seq_len, device=args.device
    )
    print("[qwen3_0_6b.low_variance.ratio] PruneReport:")
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    print(
        f"[qwen3_0_6b.low_variance.ratio] params {report.params_before:,} -> "
        f"{report.params_after:,}, cut {100 * (1 - report.params_after / report.params_before):.2f}%"
    )
    print(f"[qwen3_0_6b.low_variance.ratio] ppl {baseline_ppl:.6f} -> {pruned_ppl:.6f}")
    print(f"[qwen3_0_6b.low_variance.ratio] prune time: {prune_seconds / 60:.2f} min")

    save_model_status = "not_requested"
    if args.save_model:
        try:
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            save_model_status = "saved"
        except Exception as exc:  # NPU OOM raises RuntimeError, not OSError
            save_model_status = f"failed: {type(exc).__name__}: {exc}"
            print(
                f"[qwen3_0_6b.low_variance.ratio] save_pretrained skipped: {save_model_status}"
            )
    dump_json(save_dir / "prune_report.json", report.as_dict())
    dump_json(save_dir / "run_config.json", _runtime_args(args))
    if diagnose_summary is not None:
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
            "diagnose_time_seconds": diagnose_seconds,
            "diagnose_time_minutes": diagnose_seconds / 60
            if diagnose_seconds is not None
            else None,
            "diagnose_skipped": args.skip_diagnose,
            "save_model_status": save_model_status,
            "device": args.device,
            "prune_ratio": args.prune_ratio,
            "preview_prune_ratio": args.preview_prune_ratio,
            "calib_samples": args.calib_samples,
            "calib_seq_len": args.calib_seq_len,
            "eval_batches": args.eval_batches
            if args.eval_batches is not None
            else "all",
            "seq_len": args.seq_len,
        },
    )
    print(f"[qwen3_0_6b.low_variance.ratio] saved to {save_dir}")


if __name__ == "__main__":
    main()
