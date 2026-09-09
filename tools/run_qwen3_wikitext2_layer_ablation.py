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

"""Run W8A8 fake-quant layer ablation for Qwen3 on WikiText2."""

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

from amct_pytorch.common.models.llm.common.quant_apply import (
    apply_quant_to_attn,
    apply_quant_to_moe_mlp,
)
from amct_pytorch.common.models.llm.qwen.qwen3.quant_module import (
    QuantQwen3Attn,
    QuantQwen3MLP,
)
from amct_pytorch.quantization.bit_policy import BitPolicy
from amct_pytorch.quantization.dtypes import register_dtype
from amct_pytorch.quantization.modules.quant_base import set_quantizer_state


DEFAULT_MODEL = "/mnt/workspace/gitCode/yaoguangxiu/models/Qwen3-0.6B"
DEFAULT_WIKITEXT2 = "/mnt/workspace/gitCode/yaoguangxiu/models/wikitext/wikitext-2-raw/wiki.test.raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Local Qwen3 model directory.")
    parser.add_argument("--wikitext2", default=DEFAULT_WIKITEXT2, help="Local WikiText2 test raw file.")
    parser.add_argument("--output-dir", default="outputs/diagnostics", help="Result directory.")
    parser.add_argument("--device", default="npu:0", help="Evaluation device.")
    parser.add_argument("--seq-len", type=int, default=2048, help="PPL segment length.")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means full WikiText2 test set.")
    parser.add_argument("--bits", type=int, default=8, choices=(4, 8, 16), help="W/A fake quant bits.")
    parser.add_argument("--target", choices=("both", "attn-linear", "mlp"), default="both")
    parser.add_argument(
        "--metric",
        choices=("ppl", "nll"),
        default="ppl",
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


def _maybe_empty_cache() -> None:
    try:
        import torch.npu  # type: ignore

        torch.npu.empty_cache()
    except Exception:
        pass


def build_bit_policy(bits: int) -> BitPolicy:
    return BitPolicy(
        {
            "w_bits": bits,
            "a_bits": bits,
            "attn-linear": {
                "q_proj": {"w_bits": bits, "a_bits": bits},
                "k_proj": {"w_bits": bits, "a_bits": bits},
                "v_proj": {"w_bits": bits, "a_bits": bits},
                "o_proj": {"w_bits": bits, "a_bits": bits},
            },
            "attn-cache": {"q": bits, "k": bits, "p": bits, "v": bits},
            "mlp": {
                "gate_proj": {"w_bits": bits, "a_bits": bits},
                "up_proj": {"w_bits": bits, "a_bits": bits},
                "down_proj": {"w_bits": bits, "a_bits": bits},
            },
        }
    )


def make_quant_args(args: argparse.Namespace, quant_target: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        model=args.model,
        quant_target=quant_target,
        quant_dtype=args.quant_dtype,
        device=args.device,
        w_bits=args.bits,
        a_bits=args.bits,
        q_bits=args.bits,
        k_bits=args.bits,
        p_bits=args.bits,
        v_bits=args.bits,
        algos=[],
        output_dir="/tmp/qwen3_wikitext2_layer_ablation",
        data_dir="/tmp/qwen3_wikitext2_layer_ablation",
        attn_linear_param_dir="",
        attn_cache_param_dir="",
        moe_mlp_param_dir="",
        bit_policy=build_bit_policy(args.bits),
    )


def quant_targets(target: str) -> list[str]:
    if target == "both":
        return ["attn-linear", "mlp"]
    return [target]


def parse_layer_ids(value: str) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_exclude_cases(value: str) -> list[tuple[str, list[int]]]:
    if not value:
        return [("all", [])]

    cases = []
    for raw_case in value.split(";"):
        raw_case = raw_case.strip()
        if not raw_case:
            continue
        if ":" in raw_case:
            name, layers = raw_case.split(":", 1)
            name = name.strip()
        else:
            layers = raw_case
            name = f"exclude_{layers.replace(',', '_')}" if layers else "all"
        cases.append((name or "unnamed", parse_layer_ids(layers)))
    return cases


def load_wikitext2_tokens(path: Path, tokenizer) -> torch.Tensor:
    text = path.read_text(encoding="utf-8")
    encoded = tokenizer(text, return_tensors="pt")
    return encoded.input_ids


@torch.no_grad()
def evaluate_nll_metrics(
    model,
    tokens: torch.Tensor,
    seq_len: int,
    max_samples: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    nsamples = tokens.numel() // seq_len
    if max_samples > 0:
        nsamples = min(nsamples, max_samples)
    if nsamples <= 0:
        raise ValueError(f"WikiText2 tokens are shorter than seq_len={seq_len}.")

    nll_sum = torch.zeros((), device=device, dtype=torch.float32)
    token_count = 0
    for i in tqdm(range(nsamples), desc="ppl", leave=False):
        batch = tokens[:, i * seq_len : (i + 1) * seq_len].to(device)
        outputs = model(batch)
        logits = outputs.logits[:, :-1, :].contiguous().float()
        labels = batch[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )
        segment_tokens = seq_len - 1
        nll_sum += loss.float() * segment_tokens
        token_count += segment_tokens

    mean_nll = (nll_sum / token_count).item()
    return {
        "mean_nll": mean_nll,
        "ppl": math.exp(mean_nll),
        "token_count": float(token_count),
    }


@torch.no_grad()
def evaluate_ppl(model, tokens: torch.Tensor, seq_len: int, max_samples: int, device: torch.device) -> float:
    return evaluate_nll_metrics(model, tokens, seq_len, max_samples, device)["ppl"]


def patch_layer(layer, quant_args: SimpleNamespace, target: str) -> dict[str, Any]:
    originals = {"self_attn": layer.self_attn, "mlp": layer.mlp}
    if target in {"both", "attn-linear"}:
        apply_quant_to_attn(quant_args, layer, QuantQwen3Attn)
    if target in {"both", "mlp"}:
        apply_quant_to_moe_mlp(quant_args, layer, cls=QuantQwen3MLP)
    set_quantizer_state(layer, True)
    return originals


def restore_layer(layer, originals: dict[str, Any]) -> None:
    layer.self_attn = originals["self_attn"]
    layer.mlp = originals["mlp"]


def write_single_layer_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
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
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def write_exclude_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
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
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def write_single_layer_markdown(path: Path, rows: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    lines = [
        "# Qwen3-0.6B WikiText2 逐层量化消融报告",
        "",
        "## 实验口径",
        "",
        f"- 模型：`{meta['model']}`",
        f"- 数据集：`{meta['wikitext2']}`",
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
        "# Qwen3-0.6B WikiText2 敏感层保留精度消融报告",
        "",
        "## 实验口径",
        "",
        f"- 模型：`{meta['model']}`",
        f"- 数据集：`{meta['wikitext2']}`",
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


def run_single_layer_ablation(
    model,
    tokens: torch.Tensor,
    args: argparse.Namespace,
    baseline_metrics: dict[str, float],
    layer_ids: list[int],
    qargs: SimpleNamespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows = []
    for layer_idx in layer_ids:
        layer = model.model.layers[layer_idx]
        print(f"[layer {layer_idx}] single-layer W{args.bits}A{args.bits} target={args.target}")
        originals = patch_layer(layer, qargs, args.target)
        try:
            metrics = evaluate_nll_metrics(model, tokens, args.seq_len, args.max_samples, device)
        finally:
            restore_layer(layer, originals)
            gc.collect()
            _maybe_empty_cache()
        ppl = metrics["ppl"]
        baseline_ppl = baseline_metrics["ppl"]
        delta_ppl = ppl - baseline_ppl
        delta_nll = metrics["mean_nll"] - baseline_metrics["mean_nll"]
        relative_delta = delta_ppl / baseline_ppl if not math.isclose(baseline_ppl, 0.0) else 0.0
        row = {
            "layer": layer_idx,
            "target": args.target,
            "mean_nll": metrics["mean_nll"],
            "baseline_nll": baseline_metrics["mean_nll"],
            "delta_nll": delta_nll,
            "gold_logprob_delta": -delta_nll,
            "ppl": ppl,
            "baseline_ppl": baseline_ppl,
            "delta_ppl": delta_ppl,
            "relative_delta": relative_delta,
        }
        rows.append(row)
        print(f"[layer {layer_idx}] ppl={ppl:.6f} delta_ppl={delta_ppl:.6f} delta_nll={delta_nll:.6f}")

    sort_key = "delta_nll" if args.metric == "nll" else "delta_ppl"
    rows.sort(key=lambda row: row[sort_key], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def run_exclude_ablation(
    model,
    tokens: torch.Tensor,
    args: argparse.Namespace,
    baseline_metrics: dict[str, float],
    total_layers: int,
    qargs: SimpleNamespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows = []
    all_layer_ids = set(range(total_layers))
    for case_name, excluded_layers in parse_exclude_cases(args.exclude_cases):
        excluded = sorted(set(excluded_layers))
        unknown = sorted(set(excluded) - all_layer_ids)
        if unknown:
            raise ValueError(f"Unknown layer ids in case {case_name}: {unknown}")

        quantized = [idx for idx in range(total_layers) if idx not in excluded]
        print(
            f"[case {case_name}] quantized_layers={len(quantized)} "
            f"excluded={excluded or 'none'} target={args.target}"
        )
        patched_layers = []
        try:
            for layer_idx in quantized:
                layer = model.model.layers[layer_idx]
                patched_layers.append((layer, patch_layer(layer, qargs, args.target)))
            metrics = evaluate_nll_metrics(model, tokens, args.seq_len, args.max_samples, device)
        finally:
            for layer, originals in reversed(patched_layers):
                restore_layer(layer, originals)
            gc.collect()
            _maybe_empty_cache()

        ppl = metrics["ppl"]
        baseline_ppl = baseline_metrics["ppl"]
        delta_ppl = ppl - baseline_ppl
        delta_nll = metrics["mean_nll"] - baseline_metrics["mean_nll"]
        relative_delta = delta_ppl / baseline_ppl if not math.isclose(baseline_ppl, 0.0) else 0.0
        row = {
            "case": case_name,
            "excluded_layers": excluded,
            "quantized_layers": len(quantized),
            "target": args.target,
            "mean_nll": metrics["mean_nll"],
            "baseline_nll": baseline_metrics["mean_nll"],
            "delta_nll": delta_nll,
            "gold_logprob_delta": -delta_nll,
            "ppl": ppl,
            "baseline_ppl": baseline_ppl,
            "delta_ppl": delta_ppl,
            "relative_delta": relative_delta,
        }
        rows.append(row)
        print(f"[case {case_name}] ppl={ppl:.6f} delta_ppl={delta_ppl:.6f} delta_nll={delta_nll:.6f}")
    return rows


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
    tokens = load_wikitext2_tokens(Path(args.wikitext2), tokenizer)

    total_layers = len(model.model.layers)
    layer_ids = parse_layer_ids(args.layers) if args.layers else list(range(total_layers))

    nsamples = tokens.numel() // args.seq_len
    if args.max_samples > 0:
        nsamples = min(nsamples, args.max_samples)
    target = args.target
    qargs = make_quant_args(args, quant_targets(target))

    print(f"[baseline] evaluating {nsamples} WikiText2 segments")
    baseline_metrics = evaluate_nll_metrics(model, tokens, args.seq_len, args.max_samples, device)
    baseline_ppl = baseline_metrics["ppl"]
    print(f"[baseline] ppl={baseline_ppl:.6f}")

    sample_tag = "full" if args.max_samples == 0 else f"s{args.max_samples}"
    mode_tag = "layer_ablation" if args.mode == "single-layer" else "exclude_ablation"
    metric_tag = "nll" if args.metric == "nll" else "ppl"
    stem = (
        f"qwen3_0_6b_wikitext2_{mode_tag}_{metric_tag}_"
        f"w{args.bits}a{args.bits}_{target}_{sample_tag}_t{args.seq_len}"
    )
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    meta = {
        "model": args.model,
        "wikitext2": args.wikitext2,
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
        write_csv = write_single_layer_csv
        write_markdown = write_single_layer_markdown
    else:
        rows = run_exclude_ablation(model, tokens, args, baseline_metrics, total_layers, qargs, device)
        write_csv = write_exclude_csv
        write_markdown = write_exclude_markdown

    payload = {"meta": meta, "results": rows}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)
    write_markdown(md_path, rows, meta)
    print(f"json={json_path}")
    print(f"csv={csv_path}")
    print(f"report={md_path}")


if __name__ == "__main__":
    main()
