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

"""Run Pileval-backed fake-quant sensitivity diagnosis for local Qwen models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from amct_pytorch.quantization.bit_policy import BitPolicy
from amct_pytorch.quantization.diagnostics import (
    QuantDiagnosisConfig,
    diagnose_quantization_error,
)


DEFAULT_PILEVAL = "/mnt/workspace/gitCode/yaoguangxiu/models/pile-val-backup/val.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local model directory.")
    parser.add_argument(
        "--model-name",
        required=True,
        choices=("qwen3", "qwen3_6_moe"),
        help="AMCT model adapter name.",
    )
    parser.add_argument("--pileval", default=DEFAULT_PILEVAL, help="Local Pileval jsonl file.")
    parser.add_argument("--output-dir", default="outputs/diagnostics", help="Result directory.")
    parser.add_argument("--seq-len", type=int, default=128, help="Token length per sample.")
    parser.add_argument("--nsamples", type=int, default=4, help="Number of Pileval samples.")
    parser.add_argument("--layer-limit", type=int, default=0, help="0 means all layers.")
    parser.add_argument("--device", default="cpu", help="cpu or npu device.")
    parser.add_argument(
        "--quant-target",
        nargs="+",
        default=("attn-linear", "mlp"),
        help="Quant targets for diagnosis.",
    )
    parser.add_argument("--quant-dtype", default="int", help="Registered quant dtype.")
    parser.add_argument("--bits", type=int, default=8, choices=(4, 8, 16))
    return parser.parse_args()


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
                "in_proj_qkv": {"w_bits": bits, "a_bits": bits},
                "in_proj_z": {"w_bits": bits, "a_bits": bits},
                "in_proj_b": {"w_bits": bits, "a_bits": bits},
                "in_proj_a": {"w_bits": bits, "a_bits": bits},
                "out_proj": {"w_bits": bits, "a_bits": bits},
            },
            "attn-cache": {"q": bits, "k": bits, "p": bits, "v": bits},
            "mlp": {
                "gate_proj": {"w_bits": bits, "a_bits": bits},
                "up_proj": {"w_bits": bits, "a_bits": bits},
                "down_proj": {"w_bits": bits, "a_bits": bits},
            },
            "moe": {
                "routed": {
                    "gate_proj": {"w_bits": bits, "a_bits": bits},
                    "up_proj": {"w_bits": bits, "a_bits": bits},
                    "down_proj": {"w_bits": bits, "a_bits": bits},
                },
                "shared": {
                    "gate_proj": {"w_bits": bits, "a_bits": bits},
                    "up_proj": {"w_bits": bits, "a_bits": bits},
                    "down_proj": {"w_bits": bits, "a_bits": bits},
                },
            },
        }
    )


def make_adapter_args(args: argparse.Namespace, quant_target: list[str]) -> SimpleNamespace:
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
        data_dir="/tmp",
        output_dir="/tmp",
        attn_linear_param_dir="",
        attn_cache_param_dir="",
        moe_mlp_param_dir="",
        bit_policy=build_bit_policy(args.bits),
    )


def build_adapter(args: argparse.Namespace):
    quant_target = list(args.quant_target)
    if args.model_name == "qwen3":
        from amct_pytorch.common.models.llm.qwen.qwen3.qwen3 import Qwen3

        return Qwen3(make_adapter_args(args, quant_target))
    if args.model_name == "qwen3_6_moe":
        from amct_pytorch.common.models.llm.qwen.qwen3_6 import Qwen3_6Moe

        if "mlp" in quant_target:
            quant_target = ["moe" if item == "mlp" else item for item in quant_target]
        return Qwen3_6Moe(make_adapter_args(args, quant_target))
    raise ValueError(f"Unsupported model adapter: {args.model_name}")


def load_pileval_samples(path: Path, tokenizer, nsamples: int, seq_len: int) -> list[torch.Tensor]:
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
            while len(token_buffer) >= seq_len and len(samples) < nsamples:
                chunk = token_buffer[:seq_len]
                token_buffer = token_buffer[seq_len:]
                samples.append(torch.tensor([chunk], dtype=torch.long))
            if len(samples) >= nsamples:
                break
    if len(samples) < nsamples:
        raise ValueError(
            f"Not enough local Pileval tokens for {nsamples} samples with seq_len={seq_len}; "
            f"got {len(samples)} samples."
        )
    return samples


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> None:
    args = parse_args()
    adapter = build_adapter(args)
    samples = load_pileval_samples(Path(args.pileval), adapter.tokenizer, args.nsamples, args.seq_len)
    layer_limit = args.layer_limit if args.layer_limit > 0 else adapter.num_layers
    cfg = QuantDiagnosisConfig(layer_limit=layer_limit, sample_limit=args.nsamples)
    report = diagnose_quantization_error(adapter, samples, cfg)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model_name}_pileval_w{args.bits}a{args.bits}_l{layer_limit}_s{args.nsamples}_t{args.seq_len}"
    json_path = output_dir / f"{stem}.json"
    json_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    block_rows = [metric.__dict__ for metric in sorted(report.block_metrics, key=lambda item: item.score, reverse=True)]
    layer_rows = [metric.__dict__ for metric in sorted(report.layer_metrics, key=lambda item: item.score, reverse=True)]
    input_outlier_rows = [metric.__dict__ for metric in report.input_outlier_metrics]
    moe_rows = [metric.__dict__ for metric in sorted(report.moe_metrics, key=lambda item: item.score, reverse=True)]
    write_csv(
        output_dir / f"{stem}_blocks.csv",
        block_rows,
        [
            "name",
            "index",
            "qsnr",
            "nmse",
            "score",
            "precision",
        ],
    )
    write_csv(
        output_dir / f"{stem}_layers.csv",
        layer_rows,
        ["name", "index", "kind", "qsnr", "mse", "relative_error", "cosine_similarity", "max_error", "score", "precision"],
    )
    write_csv(
        output_dir / f"{stem}_input_outliers.csv",
        input_outlier_rows,
        [
            "name",
            "index",
            "kind",
            "max_abs",
            "p99_abs",
            "p999_abs",
            "token_ratio_p99",
            "token_ratio_max",
            "max_to_p999",
            "sample_count",
            "token_count",
        ],
    )
    if moe_rows:
        write_csv(
            output_dir / f"{stem}_moe.csv",
            moe_rows,
            ["name", "index", "routing_mismatch", "score", "precision"],
        )

    print(report.summary())
    print(f"json={json_path}")


if __name__ == "__main__":
    main()
