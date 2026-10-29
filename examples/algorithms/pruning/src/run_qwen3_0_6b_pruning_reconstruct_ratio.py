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
"""Qwen3-0.6B reconstruct/ls fixed-ratio pruning (issue 183, task 6).

Reuse the pruning samples' model, Pileval, WikiText2 and PPL helpers. Each
invocation loads the original weights and uses a fresh output directory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import shlex
import subprocess
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

disable_torchaudio()

import amct_pytorch as amct  # noqa: E402
from amct_pytorch.pruning import PruneReport, prune_diagnose  # noqa: E402

import torch  # noqa: E402


def _positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer (>= 1)")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--save-dir", default="./outputs/qwen3_0_6b_reconstruct_ratio")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--prune-ratio", type=float, required=True, choices=(0.1, 0.2, 0.5)
    )
    parser.add_argument("--calib-samples", type=_positive_int, default=32)
    parser.add_argument("--calib-seq-len", type=_positive_int, default=512)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--eval-batches", type=_positive_int, default=None)
    parser.add_argument("--skip-diagnose", action="store_true")
    args = parser.parse_args()
    if args.seq_len != 4096:
        parser.error("task 6 requires --seq-len 4096")
    if args.calib_seq_len < 2:
        parser.error("--calib-seq-len must be at least 2")
    return args


def _sync(device):
    if device.startswith("npu"):
        torch.npu.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def _stage(save_dir, stage, status="running", error=None):
    dump_json(
        save_dir / "run_status.json", {"stage": stage, "status": status, "error": error}
    )
    print(f"[{status}] {stage}", flush=True)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _batch_hash(batches):
    digest = hashlib.sha256()
    for batch in batches:
        digest.update(batch.cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _version_from_file(path):
    """Read an optional installation version without making it a prerequisite."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return None
    for line in lines:
        key, separator, value = line.partition("=")
        if separator and key.strip().lower() == "version" and value.strip():
            return value.strip()
    return None


def _cann_versions():
    roots = [
        Path(value)
        for key in ("ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH")
        if (value := os.environ.get(key))
    ]
    roots.append(Path("/usr/local/Ascend/ascend-toolkit/latest"))
    toolkit = None
    for root in roots:
        for relative in ("opp/version.info", "version.info", "version.cfg"):
            toolkit = _version_from_file(root / relative)
            if toolkit is not None:
                break
        if toolkit is not None:
            break
    return {
        "cann_toolkit": toolkit,
        "npu_driver": _version_from_file(Path("/usr/local/Ascend/driver/version.info")),
    }


def _repro_metadata(args, save_dir):
    versions = {}
    for name in (
        "torch",
        "torch-npu",
        "transformers",
        "datasets",
        "huggingface-hub",
        "amct-pytorch",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    source_hashes = {}
    for name in (
        "amct_pytorch.pruning.prune_op.dense_reconstruct",
        "amct_pytorch.pruning.context",
        "amct_pytorch.pruning.utils",
        "amct_pytorch.common.datasets.preproc",
        "amct_pytorch.common.evaluate.eval_ppl",
    ):
        try:
            source = Path(importlib.import_module(name).__file__)
            source_hashes[name] = {"path": str(source), "sha256": _sha256(source)}
        except (ImportError, AttributeError, TypeError, OSError):
            source_hashes[name] = None
    checkout_commit = None
    try:
        repository = Path(__file__).resolve().parents[4]
        commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if commit.returncode == 0:
            checkout_commit = commit.stdout.strip() or None
    except (IndexError, OSError):
        pass
    hardware = {}
    if args.device.startswith("npu"):
        index = torch.device(args.device).index or 0
        free, total = torch.npu.mem_get_info(index)
        hardware = {
            "name": torch.npu.get_device_name(index),
            "visible_count": torch.npu.device_count(),
            "free_bytes": free,
            "total_bytes": total,
        }
    dump_json(
        save_dir / "run_environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "versions": versions,
            **_cann_versions(),
            "hardware": hardware,
            "amct_import_path": amct.__file__,
            "source_hashes": source_hashes,
            "sample_sha256": _sha256(__file__),
            "helper_sha256": _sha256(Path(__file__).with_name("utils.py")),
            "checkout_commit": checkout_commit,
            "environment": {
                key: os.environ[key]
                for key in (
                    "ASCEND_HOME_PATH",
                    "ASCEND_OPP_PATH",
                    "ASCEND_VISIBLE_DEVICES",
                    "NPU_VISIBLE_DEVICES",
                )
                if key in os.environ
            },
        },
    )
    model_dir = Path(args.model_path)
    model_files = sorted(
        {
            *model_dir.glob("*.safetensors"),
            *model_dir.glob("*.bin"),
            *model_dir.glob("*.json"),
        }
    )
    dump_json(
        save_dir / "model_input_sha256.json",
        {path.name: _sha256(path) for path in model_files},
    )


def _widths(model):
    widths = []
    for layer in model.model.layers:
        mlp = layer.mlp
        width = mlp.down_proj.in_features
        if mlp.gate_proj.out_features != width or mlp.up_proj.out_features != width:
            raise RuntimeError("inconsistent gate/up/down projection dimensions")
        widths.append(width)
    if len(widths) != 28 or len(set(widths)) != 1:
        raise RuntimeError(f"unexpected Qwen3-0.6B FFN widths: {widths}")
    if model.config.intermediate_size != widths[0]:
        raise RuntimeError("saved model config does not match pruned FFN width")
    return widths


def _save_and_validate(model, tokenizer, args, save_dir, eval_batches):
    widths = _widths(model)
    reference_ppl = eval_wikitext2_ppl(
        model, eval_batches[:1], seq_len=args.seq_len, device=args.device
    )
    expected_params = count_params(model)
    model.to("cpu")
    model_dir = save_dir / "model"
    model.save_pretrained(model_dir, safe_serialization=True)
    tokenizer.save_pretrained(model_dir)
    reloaded, reloaded_tokenizer = load_model(
        str(model_dir), torch_dtype=torch.bfloat16
    )
    original_state, restored_state = model.state_dict(), reloaded.state_dict()
    if original_state.keys() != restored_state.keys():
        raise RuntimeError("reloaded state-dict keys differ")
    mismatches = [
        key
        for key in original_state
        if not torch.equal(original_state[key], restored_state[key])
    ]
    if mismatches:
        raise RuntimeError(f"reloaded weights differ: {mismatches[:5]}")
    if count_params(reloaded) != expected_params or _widths(reloaded) != widths:
        raise RuntimeError("reloaded parameter count or FFN dimensions differ")
    text = "The model is ready for evaluation."
    if tokenizer.encode(text) != reloaded_tokenizer.encode(text):
        raise RuntimeError("reloaded tokenizer differs")
    reloaded.to(args.device)
    restored_ppl = eval_wikitext2_ppl(
        reloaded, eval_batches[:1], seq_len=args.seq_len, device=args.device
    )
    if not math.isfinite(restored_ppl) or not math.isclose(
        reference_ppl, restored_ppl, rel_tol=1e-4, abs_tol=1e-4
    ):
        raise RuntimeError(f"reloaded PPL differs: {reference_ppl} -> {restored_ppl}")
    dump_json(
        save_dir / "reload_validation.json",
        {
            "status": "passed",
            "state_dict_tensors_checked": len(original_state),
            "all_weights_equal": True,
            "params": expected_params,
            "ffn_widths": widths,
            "tokenizer_check": "passed",
            "evaluation_blocks": 1,
            "seq_len": args.seq_len,
            "reference_ppl": reference_ppl,
            "reloaded_ppl": restored_ppl,
        },
    )
    print("save/reload validation: PASS", flush=True)


def _run(args, save_dir):
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            raise RuntimeError("NPU is unavailable")
    torch.manual_seed(42)
    _repro_metadata(args, save_dir)
    _stage(save_dir, "load_model")
    model, tokenizer = load_model(args.model_path, torch_dtype=torch.bfloat16)
    model.to(args.device)
    _stage(save_dir, "calibration")
    _sync(args.device)
    prepare_start = time.perf_counter()
    calib_cpu = build_pileval_batches(
        tokenizer, n_samples=args.calib_samples, seq_len=args.calib_seq_len
    )
    if len(calib_cpu) != args.calib_samples:
        raise RuntimeError("calibration sample count differs from request")
    calib = to_device_batches(calib_cpu, args.device)
    _sync(args.device)
    prepare_seconds = time.perf_counter() - prepare_start
    _stage(save_dir, "evaluation_data")
    eval_batches = build_wikitext2_batches(tokenizer, seq_len=args.seq_len)
    eval_batches = eval_batches[: args.eval_batches]
    if not eval_batches:
        raise RuntimeError("no evaluation batches")
    dump_json(
        save_dir / "data_protocol.json",
        {
            "calibration_source": "mit-han-lab/pile-val-backup",
            "calibration_split": "validation",
            "calibration_seed": 42,
            "calibration_samples": len(calib),
            "calibration_seq_len": args.calib_seq_len,
            "calibration_preprocessing": "shared get_pileval/pileval_awq: shuffle seed 42; strip; encode; skip rows longer than seq_len; concatenate; first N full blocks",
            "calibration_token_sha256": _batch_hash(calib_cpu),
            "evaluation_source": "Salesforce/wikitext",
            "evaluation_config": "wikitext-2-raw-v1",
            "evaluation_split": "test",
            "evaluation_preprocessing": "shared get_wiki_inputs: join all rows with double newline; tokenize once; nonoverlapping full blocks; drop remainder",
            "evaluation_blocks": len(eval_batches),
            "evaluation_seq_len": args.seq_len,
            "evaluation_token_sha256": _batch_hash(eval_batches),
            "ppl_helper": "utils.eval_wikitext2_ppl",
        },
    )
    _stage(save_dir, "baseline")
    before = count_params(model)
    baseline_ppl = eval_wikitext2_ppl(
        model, eval_batches, seq_len=args.seq_len, device=args.device
    )
    if not math.isfinite(baseline_ppl):
        raise RuntimeError("non-finite baseline PPL")
    dump_json(
        save_dir / "baseline_metrics.json",
        {"params_before": before, "baseline_ppl": baseline_ppl},
    )
    config = {
        "methods": {
            "dense": {
                "name": "reconstruct",
                "kwargs": {"prune_ratio": args.prune_ratio, "recovery": "ls"},
            }
        }
    }
    dump_json(save_dir / "prune_config.json", config)
    diagnose_seconds = None
    if not args.skip_diagnose:
        _stage(save_dir, "diagnose")
        _sync(args.device)
        start = time.perf_counter()
        diagnose = prune_diagnose(
            model,
            data=calib,
            config=copy.deepcopy(config),
            prune_ratio=args.prune_ratio,
        )
        _sync(args.device)
        diagnose_seconds = time.perf_counter() - start
        summary = diagnose.summary()
        (save_dir / "diagnose_summary.txt").write_text(summary + "\n")
        print(summary, flush=True)
    _stage(save_dir, "prune")
    report = PruneReport()
    _sync(args.device)
    start = time.perf_counter()
    amct.prune(model, config, data=calib, report=report)
    _sync(args.device)
    execute_seconds = time.perf_counter() - start
    report_dict = report.as_dict()
    dump_json(save_dir / "prune_report.json", report_dict)
    print(json.dumps(report_dict, ensure_ascii=False, indent=2), flush=True)
    after = count_params(model)
    events = [
        event
        for event in report_dict["events"]
        if event["stage"] == "dense" and event["method"] == "reconstruct"
    ]
    if len(events) != 28 or any("(ls)" not in event["detail"] for event in events):
        raise RuntimeError("not all 28 FFN layers used LS; inspect prune_report.json")
    widths = _widths(model)
    if after >= before:
        raise RuntimeError("parameter count did not decrease")
    _stage(save_dir, "pruned_evaluation")
    pruned_ppl = eval_wikitext2_ppl(
        model, eval_batches, seq_len=args.seq_len, device=args.device
    )
    if not math.isfinite(pruned_ppl):
        raise RuntimeError("non-finite pruned PPL")
    metrics = {
        "model": "Qwen3-0.6B",
        "method": "reconstruct",
        "recovery": "ls",
        "mode": "ratio",
        "prune_ratio": args.prune_ratio,
        "baseline_ppl": baseline_ppl,
        "pruned_ppl": pruned_ppl,
        "post_process_ppl": "N/A",
        "params_before": before,
        "params_after": after,
        "param_cut_ratio": 1 - after / before,
        "prune_time_minutes": (prepare_seconds + execute_seconds) / 60,
        "calib_prepare_seconds": prepare_seconds,
        "prune_execute_seconds": execute_seconds,
        "diagnose_time_seconds": diagnose_seconds,
        "calib_samples": len(calib),
        "calib_seq_len": args.calib_seq_len,
        "eval_batches": len(eval_batches),
        "seq_len": args.seq_len,
        "ffn_width": widths[0],
        "run_kind": "smoke" if args.eval_batches is not None else "full_evaluation",
        "save_model_status": "pending_validation"
        if args.save_model
        else "not_requested",
        "dtype": "bfloat16",
        "device": args.device,
    }
    dump_json(save_dir / "run_metrics.json", metrics)
    if args.save_model:
        _stage(save_dir, "save_and_reload")
        _save_and_validate(model, tokenizer, args, save_dir, eval_batches)
        metrics["save_model_status"] = "saved_and_validated"
        dump_json(save_dir / "run_metrics.json", metrics)
    print(
        f"PPL {baseline_ppl:.6f} -> {pruned_ppl:.6f}; params {before} -> {after}",
        flush=True,
    )


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    if save_dir.exists() and any(save_dir.iterdir()):
        raise RuntimeError(
            f"output directory is not empty: {save_dir}; choose a fresh --save-dir"
        )
    save_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(vars(args), ensure_ascii=False, indent=2), flush=True)
    dump_json(save_dir / "run_config.json", vars(args))
    (save_dir / "run_command.txt").write_text(
        shlex.join([sys.executable, *sys.argv]) + "\n"
    )
    _stage(save_dir, "startup")
    try:
        _run(args, save_dir)
    except Exception as exc:
        stage = json.loads((save_dir / "run_status.json").read_text())["stage"]
        _stage(save_dir, stage, "failed", f"{type(exc).__name__}: {exc}")
        raise
    _stage(save_dir, "complete", "complete")


if __name__ == "__main__":
    main()
