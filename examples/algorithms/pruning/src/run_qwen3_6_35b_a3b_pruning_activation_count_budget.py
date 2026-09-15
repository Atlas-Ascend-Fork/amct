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
import argparse
import importlib.machinery
import json
import math
import sys
import time
import types
from dataclasses import asdict
from pathlib import Path

# 某些 Transformers 版本在加载模型类时会导入 torchaudio；此处隐藏不兼容的可选安装
torchaudio_stub = types.ModuleType("torchaudio")
torchaudio_stub.__version__ = "0.0.0"
torchaudio_stub.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)
torchaudio_stub.functional = types.SimpleNamespace()
sys.modules.setdefault("torchaudio", torchaudio_stub)

import torch  # noqa: E402
import amct_pytorch as amct  # noqa: E402
from datasets import load_dataset  # noqa: E402
from modelscope import MsDataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from amct_pytorch.pruning import PruneReport, prune_diagnose  # noqa: E402


def load_pileval(tokenizer, seq_len, sample_count, source):
    if source == "huggingface":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    else:
        dataset = MsDataset.load("swift/pile-val-backup", "default", split="validation")
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

    # AMCT 的 Qwen3.6 评估器期望原始的 VL wrapper 配置，但 AutoModelForCausalLM.save_pretrained 会写出扁平化的纯文本配置。
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


def model_load_kwargs(trust_remote_code):
    if trust_remote_code:
        print(
            "WARNING: --trust_remote_code is enabled. Custom Python from the "
            "model directory may execute under the current user. Use only with "
            "checkpoints you trust.",
            flush=True,
        )
        return {"trust_remote_code": True}
    return {"trust_remote_code": False}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the local Qwen3.6-35B-A3B model directory.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=(
            "Allow transformers to execute custom Python shipped with the model. "
            "Off by default. Only enable for a local checkpoint you trust; "
            "the Qwen3.6 weights require this flag."
        ),
    )

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
    parser.add_argument(
        "--ratio_grid",
        type=parse_ratio_grid,
        default=(0.1, 0.2, 0.3),
        help="Comma-separated finite pruning ratios between 0 and 1.",
    )
    parser.add_argument(
        "--report_dir",
        required=True,
        help="Directory for JSON pruning reports.",
    )
    parser.add_argument(
        "--save_model_dir",
        default="",
        help="Optional output directory for the pruned model.",
    )

    args = parser.parse_args()
    if not math.isfinite(args.size_budget) or not 0.0 < args.size_budget <= 1.0:
        parser.error("--size_budget must be finite and between 0 and 1")

    if args.seq_len <= 0:
        parser.error("--seq_len must be a positive integer")

    if args.calibration_samples <= 0:
        parser.error("--calibration_samples must be a positive integer")
    if not Path(args.model_path).is_dir():
        parser.error(
            f"--model_path does not exist or is not a directory: {args.model_path}"
        )
    return args


def parse_ratio_grid(value):
    try:
        ratios = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "ratio_grid must contain comma-separated numbers"
        ) from error

    if not ratios or any(
        not math.isfinite(ratio) or ratio <= 0.0 or ratio >= 1.0 for ratio in ratios
    ):
        raise argparse.ArgumentTypeError(
            "ratio_grid values must be finite and between 0 and 1"
        )

    return ratios


def main():
    args = parse_args()
    ratios = args.ratio_grid
    load_kwargs = model_load_kwargs(args.trust_remote_code)
    run_parameters = {
        "model_path": args.model_path,
        "trust_remote_code": args.trust_remote_code,
        "size_budget": args.size_budget,
        "seq_len": args.seq_len,
        "calibration_samples": args.calibration_samples,
        "pileval_source": args.pileval_source,
        "ratio_grid": list(ratios),
        "report_dir": args.report_dir,
        "save_model_dir": args.save_model_dir,
    }

    print("=== run parameters ===")
    print(
        json.dumps(
            run_parameters,
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, **load_kwargs
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        **load_kwargs,
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()

    calibration = load_pileval(
        tokenizer,
        args.seq_len,
        args.calibration_samples,
        args.pileval_source,
    )

    before = count_parameters(model)
    experts_before = count_experts(model)
    config = {"methods": {"moe": {"name": "activation_count", "kwargs": {}}}}

    diagnosis = prune_diagnose(
        model,
        data=calibration,
        config=config,
        prune_ratio=ratios[0],
        tolerance=0.1,
    )

    print("[prune-diagnose]")
    print(
        json.dumps(
            asdict(diagnosis),
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        flush=True,
    )

    report = PruneReport()
    workflow_start = time.time()
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
    print(
        f"forward: shape={tuple(logits.shape)}, finite={bool(torch.isfinite(logits).all())}"
    )
    print(f"calibration_to_prune_seconds: {prune_finish - workflow_start:.2f}")

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / (
        f"qwen3_6_35b_a3b_activation_count_budget_{args.size_budget:.1f}.json"
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
                "run_parameters": run_parameters,
                "diagnosis": asdict(diagnosis),
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
