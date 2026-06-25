#!/usr/bin/env python3

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate causal LM perplexity with a sliding window.")
    parser.add_argument("--model-path", required=True, help="HF model path to evaluate.")
    parser.add_argument(
        "--dataset",
        default="wikitext",
        help="Dataset path or HuggingFace dataset id. Defaults to wikitext.",
    )
    parser.add_argument(
        "--dataset-name",
        default="wikitext-2-raw-v1",
        help="Dataset config/name for HuggingFace datasets.",
    )
    parser.add_argument("--dataset-split", default="test", help="Dataset split. Defaults to test.")
    parser.add_argument("--text-column", default="text", help="Text column name. Defaults to text.")
    parser.add_argument("--workspace", default="./workspace/ppl", help="Workspace directory.")
    parser.add_argument("--output-name", default="ppl_results.json", help="Output JSON filename.")
    parser.add_argument("--max-length", type=int, default=2048, help="Sliding window max length.")
    parser.add_argument("--stride", type=int, default=2048, help="Sliding window stride.")
    parser.add_argument("--batch-size", type=int, default=1, help="Evaluation batch size.")
    parser.add_argument("--num-samples", type=int, default=0, help="Limit samples; 0 means all.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "npu"],
        help="Execution device. Defaults to auto.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model dtype. Defaults to auto.",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code to HF loaders.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if hasattr(torch, "npu") and torch.npu.is_available():
            return torch.device("npu:0")
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    if device_arg == "npu":
        return torch.device("npu:0")
    if device_arg == "cuda":
        return torch.device("cuda:0")
    return torch.device("cpu")


def resolve_torch_dtype(dtype_arg: str):
    if dtype_arg == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_arg]


def load_texts(dataset_arg: str, dataset_name: str, dataset_split: str, text_column: str, num_samples: int):
    dataset_path = Path(dataset_arg)
    if dataset_path.exists() and dataset_path.is_file():
        suffix = dataset_path.suffix.lower()
        if suffix in {".json", ".jsonl"}:
            ds = load_dataset("json", data_files=str(dataset_path), split="train")
        elif suffix == ".txt":
            ds = load_dataset("text", data_files=str(dataset_path), split="train")
        elif suffix == ".parquet":
            ds = load_dataset("parquet", data_files=str(dataset_path), split="train")
        else:
            raise ValueError(f"Unsupported dataset file suffix: {suffix}")
    else:
        # 目录或远程 HuggingFace dataset id
        ds = load_dataset(dataset_arg, dataset_name, split=dataset_split)

    if num_samples > 0:
        ds = ds.select(range(min(num_samples, len(ds))))

    if text_column not in ds.column_names:
        raise ValueError(f"Text column '{text_column}' not found. Available columns: {ds.column_names}")

    texts = [x for x in ds[text_column] if isinstance(x, str) and x.strip()]
    if not texts:
        raise ValueError("Dataset produced no non-empty text samples.")
    return texts


def build_windows(input_ids: torch.Tensor, max_length: int, stride: int):
    windows = []
    seq_len = input_ids.shape[1]
    prev_end = 0
    for begin in range(0, max(seq_len - 1, 1), stride):
        end = min(begin + max_length, seq_len)
        target_len = end - prev_end
        chunk = input_ids[:, begin:end]
        labels = chunk.clone()
        labels[:, :-target_len] = -100
        windows.append((chunk.squeeze(0), labels.squeeze(0)))
        prev_end = end
        if end >= seq_len:
            break
    return windows


def collate_windows(batch_windows, pad_token_id: int):
    max_len = max(input_ids.shape[0] for input_ids, _ in batch_windows)
    batch_size = len(batch_windows)
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

    for row, (row_input_ids, row_labels) in enumerate(batch_windows):
        length = row_input_ids.shape[0]
        input_ids[row, :length] = row_input_ids
        attention_mask[row, :length] = 1
        labels[row, :length] = row_labels
    return input_ids, attention_mask, labels


def batched(iterable, batch_size: int):
    for start in range(0, len(iterable), batch_size):
        yield iterable[start : start + batch_size]


def load_model(model_path: str, device: torch.device, torch_dtype, trust_remote_code: bool):
    import glob
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    is_quip_model = hasattr(config, "quip_params")
    if is_quip_model and device.type in ("cuda", "npu"):
        repo_root = Path(__file__).resolve().parents[1]
        for extra_path in (str(repo_root / "create_init_ckpt"), str(repo_root)):
            if extra_path not in sys.path:
                sys.path.insert(0, extra_path)
        from model.general_model import format_quantize_model

        quant_torch_dtype = torch.float16 if torch_dtype == "auto" else torch_dtype
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=trust_remote_code,
            torch_dtype=quant_torch_dtype,
        )
        format_quantize_model(model=model, config=config)
        if config.tie_word_embeddings:
            model.tie_weights()

        state_dict = {}
        sf_files = sorted(glob.glob(f"{model_path}/*.safetensors"))
        if sf_files:
            from safetensors.torch import load_file as sf_load
            for sf in sf_files:
                state_dict.update(sf_load(sf, device="cpu"))
        else:
            for bf in sorted(glob.glob(f"{model_path}/pytorch_model*.bin")):
                state_dict.update(torch.load(bf, map_location="cpu"))

        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        return model, True

    model_kwargs = {"trust_remote_code": trust_remote_code}
    if torch_dtype != "auto":
        model_kwargs["torch_dtype"] = torch_dtype
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.to(device)
    return model, False


def main():
    args = parse_args()
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    output_path = workspace / args.output_name

    device = resolve_device(args.device)
    torch_dtype = resolve_torch_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model, is_quip_model = load_model(
        model_path=args.model_path,
        device=device,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    texts = load_texts(
        dataset_arg=args.dataset,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        text_column=args.text_column,
        num_samples=args.num_samples,
    )
    encoded = tokenizer("\n\n".join(texts), return_tensors="pt")
    input_ids = encoded.input_ids

    windows = build_windows(input_ids, max_length=args.max_length, stride=args.stride)
    if not windows:
        raise ValueError("No evaluation windows were generated from the tokenized dataset.")

    total_nll = 0.0
    total_tokens = 0
    start_time = time.time()
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    with torch.no_grad():
        for batch_windows in batched(windows, args.batch_size):
            batch_input_ids, batch_attention_mask, batch_labels = collate_windows(batch_windows, pad_token_id)
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                labels=batch_labels,
            )
            valid_tokens = int((batch_labels[:, 1:] != -100).sum().item())
            total_nll += float(outputs.loss.item()) * valid_tokens
            total_tokens += valid_tokens

    elapsed_sec = time.time() - start_time
    perplexity = math.exp(total_nll / total_tokens)
    result = {
        "model_path": args.model_path,
        "dataset": args.dataset,
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "text_column": args.text_column,
        "num_text_samples": len(texts),
        "num_windows": len(windows),
        "num_loss_tokens": total_tokens,
        "max_length": args.max_length,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "device": str(device),
        "dtype": args.dtype,
        "is_quip_model": is_quip_model,
        "perplexity": perplexity,
        "avg_nll": total_nll / total_tokens,
        "elapsed_sec": elapsed_sec,
        "tokens_per_sec": total_tokens / elapsed_sec if elapsed_sec > 0 else None,
    }

    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
