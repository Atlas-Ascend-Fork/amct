#!/usr/bin/env python3
"""Qwen3.6-35B-A3B MoE pruning with activation_count and size_budget."""

import argparse
import importlib.machinery
import json
import sys
import time
import types
from dataclasses import asdict
from pathlib import Path

# Qwen3.6 is text-only. Some Transformers versions import torchaudio while
# loading model classes; hide an incompatible optional installation.
torchaudio_stub = types.ModuleType("torchaudio")
torchaudio_stub.__version__ = "0.0.0"
torchaudio_stub.__spec__ = importlib.machinery.ModuleSpec(
    "torchaudio", loader=None
)
torchaudio_stub.functional = types.SimpleNamespace()
sys.modules.setdefault("torchaudio", torchaudio_stub)

import torch
import torch_npu
import amct_pytorch as amct
from datasets import load_dataset
from modelscope import MsDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from amct_pytorch.pruning import PruneReport


DEFAULT_MODEL = "/mnt/workspace/models/Qwen3.6-35B-A3B"
DEFAULT_REPORT_DIR = "/mnt/workspace/qwen3_6_pruning_results"


def load_pileval(tokenizer, seq_len, sample_count, source):
    if source == "huggingface":
        dataset = load_dataset(
            "mit-han-lab/pile-val-backup", split="validation"
        )
    else:
        dataset = MsDataset.load(
            "swift/pile-val-backup", "default", split="validation"
        )
    samples = []
    for row in dataset:
        text = row.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        encoded = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=seq_len
        )
        if encoded["input_ids"].shape[1] != seq_len:
            continue
        samples.append(encoded["input_ids"])
        if len(samples) == sample_count:
            break
    if len(samples) != sample_count:
        raise RuntimeError(
            f"only collected {len(samples)} samples, expected {sample_count}"
        )
    return samples


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def count_experts(model):
    return [
        int(module.num_experts)
        for name, module in model.named_modules()
        if name.endswith(".mlp.experts") and hasattr(module, "num_experts")
    ]


def save_pruned_model(model, tokenizer, model_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # AMCT's Qwen3.6 evaluator expects the original VL wrapper config.
    # AutoModelForCausalLM.save_pretrained writes a flat text-only config.
    original_config_path = Path(model_path) / "config.json"
    with original_config_path.open("r", encoding="utf-8") as file:
        original_config = json.load(file)
    text_config = original_config.get("text_config")
    experts = count_experts(model)
    if isinstance(text_config, dict) and experts:
        text_config["num_experts"] = experts[0]
        with (output_dir / "config.json").open("w", encoding="utf-8") as file:
            json.dump(original_config, file, ensure_ascii=False, indent=2)
            file.write("\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument(
        "--size_budget", type=float, required=True, choices=(0.9, 0.8, 0.5)
    )
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--calibration_samples", type=int, default=4)
    parser.add_argument(
        "--pileval_source",
        choices=("huggingface", "modelscope"),
        default="huggingface",
        help="Official Pileval source or its ModelScope mirror.",
    )
    parser.add_argument("--ratio_grid", default="0.1,0.2,0.3")
    parser.add_argument("--report_dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--save_model_dir",
        default="",
        help="Optional output directory for the pruned model.",
    )
    return parser.parse_args()


def parse_ratio_grid(value):
    ratios = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not ratios or any(ratio <= 0.0 or ratio >= 1.0 for ratio in ratios):
        raise ValueError("ratio_grid must contain values between 0 and 1")
    return ratios


def main():
    args = parse_args()
    ratios = parse_ratio_grid(args.ratio_grid)
    workflow_start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    calibration = load_pileval(
        tokenizer,
        args.seq_len,
        args.calibration_samples,
        args.pileval_source,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()

    before = count_parameters(model)
    experts_before = count_experts(model)
    config = {
        "methods": {
            "moe": {"name": "activation_count", "kwargs": {}}
        }
    }
    report = PruneReport()

    amct.prune(
        model,
        config,
        data=calibration,
        size_budget=args.size_budget,
        ratio_grid=ratios,
        report=report,
    )
    prune_finish = time.time()

    after = count_parameters(model)
    with torch.no_grad():
        logits = model(calibration[0]).logits

    print(f"size_budget={args.size_budget}")
    print(f"experts: {experts_before[0]} -> {count_experts(model)[0]}")
    print(f"parameters: {before} -> {after}")
    print(f"parameter_reduction: {1.0 - after / before:.6f}")
    print(f"budget_unreachable: {report.budget_unreachable}")
    print(f"forward: shape={tuple(logits.shape)}, finite={bool(torch.isfinite(logits).all())}")
    print(
        "calibration_to_prune_seconds: "
        f"{prune_finish - workflow_start:.2f}"
    )

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / (
        "qwen3_6_35b_a3b_activation_count_budget_"
        f"{args.size_budget:.1f}.json"
    )
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "model_path": args.model_path,
                "method": "activation_count",
                "target": "moe",
                "size_budget": args.size_budget,
                "ratio_grid": list(ratios),
                "seq_len": args.seq_len,
                "calibration_samples": args.calibration_samples,
                "pileval_source": args.pileval_source,
                "calibration_to_prune_seconds": prune_finish - workflow_start,
                "report": asdict(report),
            },
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    print(f"report: {report_path}")

    if args.save_model_dir:
        save_pruned_model(
            model,
            tokenizer,
            args.model_path,
            args.save_model_dir,
        )
        print(f"model: {args.save_model_dir}")


if __name__ == "__main__":
    main()
