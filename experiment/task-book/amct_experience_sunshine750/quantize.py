# -*- coding: UTF-8 -*-
"""Run reproducible AMCT HiFloat8 evaluation on Qwen3 and Wikitext-2."""

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer

import amct_pytorch as amct


ALGORITHM_CONFIGS = {
    "cast": amct.HIFP8_CAST_CFG,
    "quantile": amct.HIFP8_QUANTILE_CFG,
    "ofmr": amct.HIFP8_OFMR_CFG,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate BF16 or AMCT HiFloat8 fake quantization on NPU"
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Local wikitext-2-raw-v1 directory containing TXT or parquet splits",
    )
    parser.add_argument(
        "--algorithm",
        choices=("baseline", "cast", "quantile", "ofmr"),
        required=True,
    )
    parser.add_argument(
        "--profile",
        choices=("official", "controlled"),
        default="official",
        help="controlled uses tensor weight scale and skips lm_head for all algorithms",
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--max-eval-segments", type=int, default=0)
    parser.add_argument("--calibration-batches", type=int, default=1)
    parser.add_argument("--calibration-seq-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--try-convert", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validate_args(args):
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"model directory not found: {args.model_path}")
    if not args.dataset_path.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {args.dataset_path}")
    for filename in ("config.json", "model.safetensors", "tokenizer.json"):
        path = args.model_path / filename
        if not path.is_file():
            raise FileNotFoundError(f"required model file not found: {path}")
    for split in ("train", "test"):
        text_path = args.dataset_path / f"{split}.txt"
        parquet_path = args.dataset_path / f"{split}-00000-of-00001.parquet"
        if not text_path.is_file() and not parquet_path.is_file():
            raise FileNotFoundError(
                f"required dataset split not found: {text_path} or {parquet_path}"
            )
    if not args.device.startswith("npu:"):
        raise ValueError(f"only an NPU device is supported, got: {args.device}")
    if args.seq_len < 2:
        raise ValueError("seq-len must be at least 2")
    if args.max_eval_segments < 0:
        raise ValueError("max-eval-segments cannot be negative")
    if args.calibration_batches <= 0:
        raise ValueError("calibration-batches must be positive")
    if args.calibration_seq_len < 2:
        raise ValueError("calibration-seq-len must be at least 2")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_report(device_id):
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "transformers": package_version("transformers"),
        "datasets": package_version("datasets"),
        "amct_pytorch": package_version("amct-pytorch"),
        "amct_ops": package_version("amct-ops"),
        "device_id": device_id,
        "device_name": torch.npu.get_device_name(device_id),
        "device_count": torch.npu.device_count(),
    }


def asset_report(args):
    model_files = ("config.json", "model.safetensors", "tokenizer.json")
    dataset_files = (
        "train.txt",
        "validation.txt",
        "test.txt",
        "train-00000-of-00001.parquet",
        "validation-00000-of-00001.parquet",
        "test-00000-of-00001.parquet",
    )
    return {
        "model": {
            "path": str(args.model_path.resolve()),
            "files": {
                name: {
                    "bytes": (args.model_path / name).stat().st_size,
                    "sha256": sha256_file(args.model_path / name),
                }
                for name in model_files
            },
        },
        "dataset": {
            "path": str(args.dataset_path.resolve()),
            "files": {
                name: {
                    "bytes": (args.dataset_path / name).stat().st_size,
                    "sha256": sha256_file(args.dataset_path / name),
                }
                for name in dataset_files
                if (args.dataset_path / name).is_file()
            },
        },
    }


def load_text_split(dataset_path, split):
    text_path = dataset_path / f"{split}.txt"
    if text_path.is_file():
        return text_path.read_text(encoding="utf-8")

    parquet_path = dataset_path / f"{split}-00000-of-00001.parquet"
    try:
        import pandas as pd

        values = pd.read_parquet(parquet_path, engine="fastparquet")["text"].tolist()
    except (ImportError, ValueError) as error:
        raise RuntimeError(
            f"cannot read {parquet_path}; provide {text_path} or install fastparquet"
        ) from error
    return "\n\n".join(values)


def tokenize_split(tokenizer, dataset_path, split):
    text = load_text_split(dataset_path, split)
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids


def make_calibration_batches(train_ids, batch_count, seq_len, seed):
    complete_segments = train_ids.numel() // seq_len
    if complete_segments < batch_count:
        raise ValueError(
            f"calibration data has {complete_segments} complete segments, "
            f"but {batch_count} were requested"
        )
    rng = random.Random(seed)
    indices = rng.sample(range(complete_segments), batch_count)
    batches = [
        train_ids[:, index * seq_len:(index + 1) * seq_len]
        for index in indices
    ]
    return batches, indices


def load_model(model_path, device):
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="eager",
    ).eval()
    model.config.use_cache = False
    model = model.to(device)
    torch.npu.synchronize()
    return model, time.perf_counter() - started


def build_quant_config(algorithm, profile, calibration_batches):
    config = copy.deepcopy(ALGORITHM_CONFIGS[algorithm])
    if algorithm in {"quantile", "ofmr"}:
        config["batch_num"] = calibration_batches
    if profile == "controlled":
        config["quant_cfg"]["weights"]["strategy"] = "tensor"
        config["quant_cfg"]["inputs"]["strategy"] = "tensor"
        config["skip_layers"] = {"lm_head"}
    return config


def count_module_types(model):
    counts = {}
    for module in model.modules():
        name = type(module).__name__
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def linear_weight_report(model, skip_layers=None):
    skip_layers = skip_layers or set()
    rows = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        skipped = any(
            name == skip_name or name.endswith(f".{skip_name}")
            for skip_name in skip_layers
        )
        rows.append(
            {
                "name": name,
                "parameters": module.weight.numel(),
                "skipped": skipped,
            }
        )
    selected = [row for row in rows if not row["skipped"]]
    selected_parameters = sum(row["parameters"] for row in selected)
    return {
        "linear_modules": len(rows),
        "linear_weight_parameters": sum(row["parameters"] for row in rows),
        "selected_modules": len(selected),
        "selected_weight_parameters": selected_parameters,
        "theoretical_bf16_payload_bytes": selected_parameters * 2,
        "theoretical_hifloat8_payload_bytes": selected_parameters,
        "theoretical_payload_reduction_percent": 50.0,
        "scale_and_metadata_overhead_included": False,
    }


def quantize_model(model, algorithm, profile, calibration_batches):
    config = build_quant_config(algorithm, profile, calibration_batches)
    weight_scope = linear_weight_report(model, config.get("skip_layers"))
    started = time.perf_counter()
    amct.quantize(model, config)
    torch.npu.synchronize()
    return config, time.perf_counter() - started, weight_scope


def calibrate_model(model, batches, device):
    started = time.perf_counter()
    with torch.inference_mode():
        for index, batch in enumerate(batches, start=1):
            model(batch.to(device), use_cache=False)
            torch.npu.synchronize()
            print(f"calibration {index}/{len(batches)}", flush=True)
    return time.perf_counter() - started


def evaluate_ppl(model, test_ids, device, seq_len, max_segments):
    available_segments = test_ids.numel() // seq_len
    segment_count = (
        min(available_segments, max_segments) if max_segments else available_segments
    )
    if segment_count == 0:
        raise ValueError(
            f"test data has {test_ids.numel()} tokens, fewer than seq-len={seq_len}"
        )

    torch.npu.reset_peak_memory_stats()
    total_nll = 0.0
    predicted_tokens = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for index in range(segment_count):
            batch = test_ids[:, index * seq_len:(index + 1) * seq_len].to(device)
            logits = model(batch, use_cache=False).logits
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"non-finite logits in segment {index}")
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = batch[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )
            total_nll += float(loss.cpu())
            predicted_tokens += shift_labels.numel()
            del batch, logits, shift_logits, shift_labels, loss
            if (index + 1) % 10 == 0 or index + 1 == segment_count:
                print(f"evaluation {index + 1}/{segment_count}", flush=True)
    torch.npu.synchronize()
    elapsed = time.perf_counter() - started
    mean_nll = total_nll / predicted_tokens
    return {
        "ppl": math.exp(mean_nll),
        "mean_nll": mean_nll,
        "total_nll": total_nll,
        "predicted_tokens": predicted_tokens,
        "source_tokens": segment_count * seq_len,
        "segments": segment_count,
        "available_segments": available_segments,
        "seq_len": seq_len,
        "elapsed_seconds": elapsed,
        "tokens_per_second": predicted_tokens / elapsed,
        "peak_memory_bytes": torch.npu.max_memory_allocated(),
    }


def try_convert(model):
    started = time.perf_counter()
    try:
        amct.convert(model)
        torch.npu.synchronize()
        return {
            "requested": True,
            "verified": True,
            "elapsed_seconds": time.perf_counter() - started,
        }
    except Exception as error:  # deployment capability is part of the report
        return {
            "requested": True,
            "verified": False,
            "elapsed_seconds": time.perf_counter() - started,
            "error_type": type(error).__name__,
            "error": str(error),
        }


def write_report(path, report):
    def json_default(value):
        if isinstance(value, set):
            return sorted(value)
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"cannot serialize {type(value).__name__} to JSON")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


def configure_runtime(args):
    validate_args(args)
    device_id = int(args.device.split(":", maxsplit=1)[1])
    if device_id < 0 or device_id >= torch.npu.device_count():
        raise ValueError(
            f"device id {device_id} is outside available range "
            f"[0, {torch.npu.device_count() - 1}]"
        )
    torch.npu.set_device(device_id)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    return device_id


def initialize_report(args, device_id):
    return {
        "status": "running",
        "environment": environment_report(device_id),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "assets": asset_report(args),
        "result_kind": "bf16_baseline"
        if args.algorithm == "baseline"
        else "hifloat8_fake_quant",
    }


def prepare_tokenized_data(args, report):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        local_files_only=True,
    )
    tokenizer.model_max_length = 10**12
    test_ids = tokenize_split(tokenizer, args.dataset_path, "test")
    report["dataset_tokens"] = {"test": test_ids.numel()}

    if args.algorithm not in {"quantile", "ofmr"}:
        return test_ids, []

    train_ids = tokenize_split(tokenizer, args.dataset_path, "train")
    calibration_batches, calibration_indices = make_calibration_batches(
        train_ids,
        args.calibration_batches,
        args.calibration_seq_len,
        args.seed,
    )
    report["dataset_tokens"]["train"] = train_ids.numel()
    report["calibration"] = {
        "batches": args.calibration_batches,
        "seq_len": args.calibration_seq_len,
        "tokens": args.calibration_batches * args.calibration_seq_len,
        "segment_indices": calibration_indices,
        "sampling": "seeded sample without replacement over complete train segments",
    }
    return test_ids, calibration_batches


def record_model_details(report, model, load_seconds):
    report["model"] = {
        "class": type(model).__name__,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "load_seconds": load_seconds,
        "dtype": str(next(model.parameters()).dtype),
        "module_types_before": count_module_types(model),
        "linear_weights": linear_weight_report(model),
    }


def apply_quantization(args, report, model, calibration_batches):
    config, quantize_seconds, weight_scope = quantize_model(
        model,
        args.algorithm,
        args.profile,
        args.calibration_batches,
    )
    report["quantization"] = {
        "algorithm": args.algorithm,
        "profile": args.profile,
        "config": config,
        "quantize_seconds": quantize_seconds,
        "module_types_after": count_module_types(model),
        "weight_scope": weight_scope,
    }
    if calibration_batches:
        report["calibration"]["elapsed_seconds"] = calibrate_model(
            model, calibration_batches, args.device
        )


def run(args):
    device_id = configure_runtime(args)
    report = initialize_report(args, device_id)
    write_report(args.output, report)
    test_ids, calibration_batches = prepare_tokenized_data(args, report)
    model, load_seconds = load_model(args.model_path, args.device)
    record_model_details(report, model, load_seconds)
    if args.algorithm != "baseline":
        apply_quantization(args, report, model, calibration_batches)
    report["evaluation"] = evaluate_ppl(
        model,
        test_ids,
        args.device,
        args.seq_len,
        args.max_eval_segments,
    )
    report["deployment"] = (
        try_convert(model) if args.try_convert else {"requested": False}
    )
    report["status"] = "success"
    write_report(args.output, report)
    print(json.dumps(report["evaluation"], indent=2), flush=True)

    del model, test_ids, calibration_batches
    gc.collect()
    torch.npu.empty_cache()


def main():
    args = parse_args()
    try:
        run(args)
    except Exception as error:
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        if args.output.exists():
            try:
                existing = json.loads(args.output.read_text(encoding="utf-8"))
                existing.update(failure)
                failure = existing
            except (OSError, json.JSONDecodeError):
                pass
        write_report(args.output, failure)
        raise


if __name__ == "__main__":
    main()
