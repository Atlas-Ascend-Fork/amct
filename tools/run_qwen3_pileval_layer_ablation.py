#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
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

"""Run W8A8 fake-quant layer ablation for Qwen3 on local Pileval."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from amct_pytorch.quantization.bit_policy import BitPolicy
from amct_pytorch.quantization.dtypes import register_dtype

from run_qwen3_wikitext2_layer_ablation import (
    _maybe_empty_cache,
    build_bit_policy,
    evaluate_nll_metrics,
    patch_layer,
    parse_exclude_cases,
    parse_layer_ids,
    quant_targets,
    restore_layer,
    run_exclude_ablation,
    run_single_layer_ablation,
)


DEFAULT_MODEL = "/mnt/workspace/gitCode/yaoguangxiu/models/Qwen3-0.6B"
DEFAULT_PILEVAL = "/mnt/workspace/gitCode/yaoguangxiu/models/pile-val-backup/val.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Local Qwen3 model directory.")
    parser.add_argument("--pileval", default=DEFAULT_PILEVAL, help="Local Pileval jsonl file.")
    parser.add_argument("--output-dir", default="outputs/diagnostics", help="Result directory.")
    parser.add_argument("--device", default="npu:0", help="Evaluation device.")
    parser.add_argument("--seq-len", type=int, default=128, help="PPL segment length.")
    parser.add_argument("--max-samples", type=int, default=4, help="0 means full Pileval token stream.")
    parser.add_argument("--bits", type=int, default=8, choices=(4, 8, 16), help="W/A fake quant bits.")
    parser.add_argument("--target", choices=("both", "attn-linear", "mlp"), default="both")
    parser.add_argument(
        "--metric",
        choices=("ppl", "nll"),
        default="nll",
        help="Ranking metric. nll ranks by whole-model gold-token mean NLL delta.",
    )
    parser.add_argument(
        "--mode",
        choices=("single-layer", "exclude"),
        default="single-layer",
        help="single-layer: quantize one layer each time; exclude: quantize all layers except each case list.",
    )
    parser.add_argument("--layers", default="", help="Comma-separated layer ids. Empty means all layers.")
    parser.add_argument(
        "--exclude-cases",
        default="",
        help=(
            "Semicolon-separated exclude cases for --mode exclude, e.g. "
            "'all:;top1:9;top3:9,27,2'. Empty means one all-quant case."
        ),
    )
    parser.add_argument("--quant-dtype", default="int", help="Registered quant dtype.")
    return parser.parse_args()


def load_pileval_tokens(path: Path, tokenizer, seq_len: int, max_samples: int) -> torch.Tensor:
    samples = []
    token_buffer: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            encoded = tokenizer.encode(text)
            if not encoded or len(encoded) > seq_len:
                continue
            token_buffer.extend(encoded)
            while len(token_buffer) >= seq_len and (max_samples <= 0 or len(samples) < max_samples):
                chunk = token_buffer[:seq_len]
                token_buffer = token_buffer[seq_len:]
                samples.append(torch.tensor([chunk], dtype=torch.long))
            if max_samples > 0 and len(samples) >= max_samples:
                break
    if not samples:
        raise ValueError(f"Not enough local Pileval tokens for seq_len={seq_len}.")
    return torch.cat(samples, dim=1)


def write_single_layer_markdown(path: Path, rows: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    lines = [
        "# Qwen3-0.6B Pileval 逐层量化消融报告",
        "",
        "## 实验口径",
        "",
        f"- 模型：`{meta['model']}`",
        f"- 数据集：`{meta['pileval']}`",
        f"- 设备：`{meta['device']}`",
        f"- 量化：W{meta['bits']}A{meta['bits']} direct fake quant",
        f"- 目标范围：`{meta['target']}`",
        f"- seq_len：`{meta['seq_len']}`",
        f"- 样本数：`{meta['nsamples']}`",
        f"- Baseline PPL：`{meta['baseline_ppl']:.6f}`",
        f"- Baseline mean NLL：`{meta['baseline_nll']:.6f}`",
        f"- 排序指标：`{meta['metric']}`",
        "",
        "说明：每次只量化一个 decoder layer，其它 layer 保持原精度。"
        "`delta_ppl` 越大表示 PPL 影响越大；`delta_nll` 越大表示 gold-token mean NLL 恶化越明显。",
        "",
        "## Top 敏感层",
        "",
        "| Rank | Layer | Mean NLL | Delta NLL | Gold LogProb Delta | PPL | Delta PPL | Relative Delta |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows[: min(12, len(rows))]:
        lines.append(
            f"| {row['rank']} | {row['layer']} | {row['mean_nll']:.6f} | "
            f"{row['delta_nll']:.6f} | {row['gold_logprob_delta']:.6f} | {row['ppl']:.6f} | "
            f"{row['delta_ppl']:.6f} | {row['relative_delta']:.6%} |"
        )
    lines.extend(
        [
            "",
            "## 全量结果",
            "",
            "| Rank | Layer | Mean NLL | Delta NLL | Gold LogProb Delta | PPL | Delta PPL | Relative Delta |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['rank']} | {row['layer']} | {row['mean_nll']:.6f} | "
            f"{row['delta_nll']:.6f} | {row['gold_logprob_delta']:.6f} | {row['ppl']:.6f} | "
            f"{row['delta_ppl']:.6f} | {row['relative_delta']:.6%} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_exclude_markdown(path: Path, rows: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    lines = [
        "# Qwen3-0.6B Pileval 敏感层保留精度消融报告",
        "",
        "## 实验口径",
        "",
        f"- 模型：`{meta['model']}`",
        f"- 数据集：`{meta['pileval']}`",
        f"- 设备：`{meta['device']}`",
        f"- 量化：W{meta['bits']}A{meta['bits']} direct fake quant",
        f"- 目标范围：`{meta['target']}`",
        f"- seq_len：`{meta['seq_len']}`",
        f"- 样本数：`{meta['nsamples']}`",
        f"- Baseline PPL：`{meta['baseline_ppl']:.6f}`",
        f"- Baseline mean NLL：`{meta['baseline_nll']:.6f}`",
        f"- 排序指标：`{meta['metric']}`",
        "",
        "说明：每个 case 都量化除 `excluded_layers` 外的全部 decoder layer。"
        "`delta_ppl` / `delta_nll` 越小，说明保留这些层为原精度后整体量化误差越小。",
        "",
        "## 结果",
        "",
        "| Case | Excluded Layers | Quantized Layers | Mean NLL | Delta NLL | PPL | Delta PPL | Relative Delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        excluded = ",".join(str(item) for item in row["excluded_layers"]) or "-"
        lines.append(
            f"| {row['case']} | {excluded} | {row['quantized_layers']} | {row['mean_nll']:.6f} | "
            f"{row['delta_nll']:.6f} | {row['ppl']:.6f} | "
            f"{row['delta_ppl']:.6f} | {row['relative_delta']:.6%} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> None:
    register_dtype()
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.config.use_cache = False
    tokens = load_pileval_tokens(Path(args.pileval), tokenizer, args.seq_len, args.max_samples)

    total_layers = len(model.model.layers)
    layer_ids = parse_layer_ids(args.layers) if args.layers else list(range(total_layers))

    nsamples = tokens.numel() // args.seq_len
    target = args.target
    qargs = SimpleNamespace(
        model=args.model,
        quant_target=quant_targets(target),
        quant_dtype=args.quant_dtype,
        device=args.device,
        w_bits=args.bits,
        a_bits=args.bits,
        q_bits=args.bits,
        k_bits=args.bits,
        p_bits=args.bits,
        v_bits=args.bits,
        algos=[],
        output_dir="/tmp/qwen3_pileval_layer_ablation",
        data_dir="/tmp/qwen3_pileval_layer_ablation",
        attn_linear_param_dir="",
        attn_cache_param_dir="",
        moe_mlp_param_dir="",
        bit_policy=build_bit_policy(args.bits),
    )

    print(f"[baseline] evaluating {nsamples} Pileval segments")
    baseline_metrics = evaluate_nll_metrics(model, tokens, args.seq_len, args.max_samples, device)
    baseline_ppl = baseline_metrics["ppl"]
    print(f"[baseline] ppl={baseline_ppl:.6f}")

    sample_tag = "full" if args.max_samples == 0 else f"s{args.max_samples}"
    mode_tag = "layer_ablation" if args.mode == "single-layer" else "exclude_ablation"
    metric_tag = "nll" if args.metric == "nll" else "ppl"
    stem = f"qwen3_0_6b_pileval_{mode_tag}_{metric_tag}_w{args.bits}a{args.bits}_{target}_{sample_tag}_t{args.seq_len}"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    meta = {
        "model": args.model,
        "pileval": args.pileval,
        "device": args.device,
        "bits": args.bits,
        "target": target,
        "seq_len": args.seq_len,
        "nsamples": nsamples,
        "baseline_ppl": baseline_ppl,
        "baseline_nll": baseline_metrics["mean_nll"],
        "token_count": int(baseline_metrics["token_count"]),
        "layers": layer_ids,
        "mode": args.mode,
        "metric": args.metric,
        "exclude_cases": args.exclude_cases,
    }

    if args.mode == "single-layer":
        rows = run_single_layer_ablation(model, tokens, args, baseline_metrics, layer_ids, qargs, device)
        json_path.write_text(json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
        write_csv(
            csv_path,
            rows,
            [
                "layer",
                "target",
                "mean_nll",
                "baseline_nll",
                "delta_nll",
                "gold_logprob_delta",
                "ppl",
                "baseline_ppl",
                "delta_ppl",
                "relative_delta",
                "rank",
            ],
        )
        write_single_layer_markdown(md_path, rows, meta)
    else:
        rows = run_exclude_ablation(model, tokens, args, baseline_metrics, total_layers, qargs, device)
        json_path.write_text(json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
        write_csv(
            csv_path,
            rows,
            [
                "case",
                "excluded_layers",
                "quantized_layers",
                "target",
                "mean_nll",
                "baseline_nll",
                "delta_nll",
                "gold_logprob_delta",
                "ppl",
                "baseline_ppl",
                "delta_ppl",
                "relative_delta",
            ],
        )
        write_exclude_markdown(md_path, rows, meta)

    print(f"json={json_path}")
    print(f"csv={csv_path}")
    print(f"md={md_path}")


if __name__ == "__main__":
    main()
