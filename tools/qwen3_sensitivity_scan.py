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

"""Run a local Qwen3 fake-quant sensitivity scan on selected linear blocks."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

from amct_pytorch.common.models.llm.common.quant_apply import (
    apply_quant_to_attn,
    apply_quant_to_moe_mlp,
)
from amct_pytorch.common.models.llm.qwen.qwen3.qwen3 import Qwen3
from amct_pytorch.common.models.llm.qwen.qwen3.quant_module import (
    QuantQwen3Attn,
    QuantQwen3MLP,
)
from amct_pytorch.quantization.bit_policy import BitPolicy
from amct_pytorch.quantization.dtypes import register_dtype


register_dtype()


DEFAULT_LAYERS = "2,23,24,25,26,27"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3 sensitivity scan")
    parser.add_argument(
        "--model",
        default="/mnt/workspace/gitCode/yaoguangxiu/models/Qwen3-0.6B",
        help="Local model directory.",
    )
    parser.add_argument(
        "--data-dir",
        default="ptq_data/qwen3",
        help="PTQ cache directory holding attn-linear/ and mlp/ inputs.",
    )
    parser.add_argument(
        "--outlier-json",
        default="qwen3_linear_stats.json",
        help="Outlier statistics JSON used to guide candidate selection.",
    )
    parser.add_argument(
        "--layers",
        default=DEFAULT_LAYERS,
        help="Comma-separated layer indices to evaluate.",
    )
    parser.add_argument(
        "--targets",
        default="mlp,attn-linear",
        help="Comma-separated targets to evaluate.",
    )
    parser.add_argument("--device", default="cpu", help="cpu or npu:0")
    parser.add_argument(
        "--sample-count",
        type=int,
        default=2,
        help="Number of cached samples used per layer.",
    )
    parser.add_argument(
        "--token-count",
        type=int,
        default=256,
        help="Token prefix used per sample.",
    )
    parser.add_argument(
        "--output",
        default="qwen3_sensitivity_scan.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def build_bit_policy() -> BitPolicy:
    return BitPolicy(
        {
            "w_bits": 8,
            "a_bits": 8,
            "attn-linear": {
                "q_proj": {"w_bits": 8, "a_bits": 8},
                "k_proj": {"w_bits": 8, "a_bits": 8},
                "v_proj": {"w_bits": 8, "a_bits": 8},
                "o_proj": {"w_bits": 8, "a_bits": 8},
            },
            "attn-cache": {"q": 8, "k": 8, "p": 8, "v": 8},
            "mlp": {
                "gate_proj": {"w_bits": 8, "a_bits": 8},
                "up_proj": {"w_bits": 8, "a_bits": 8},
                "down_proj": {"w_bits": 8, "a_bits": 8},
            },
        }
    )


def make_args(model: str, data_dir: str, device: str, target: str) -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        quant_target=[target],
        device=device,
        data_dir=data_dir,
        output_dir="/tmp/qwen3_sensitivity",
        quant_dtype="int",
        w_bits=8,
        a_bits=8,
        q_bits=8,
        k_bits=8,
        p_bits=8,
        v_bits=8,
        algos=[],
        cali_bsz=1,
        moe_mlp_param_dir="",
        bit_policy=build_bit_policy(),
    )


def load_tensor(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, weights_only=True, mmap=True)
    except TypeError:
        return torch.load(path, weights_only=True)


def metric(ref: torch.Tensor, out: torch.Tensor) -> dict[str, float]:
    ref_f = ref.float().reshape(-1)
    out_f = out.float().reshape(-1)
    mse = F.mse_loss(out_f, ref_f).item()
    ref_energy = torch.mean(ref_f.square()).item()
    rel_mse = mse / ref_energy if ref_energy else 0.0
    cos = F.cosine_similarity(ref_f, out_f, dim=0).item()
    max_abs = torch.max(torch.abs(ref_f - out_f)).item()
    return {
        "mse": mse,
        "rel_mse": rel_mse,
        "cosine": cos,
        "delta_cosine": 1.0 - cos,
        "max_abs_err": max_abs,
    }


def slice_input(tensor: torch.Tensor, sample_count: int, token_count: int) -> torch.Tensor:
    return tensor[:sample_count, :token_count].contiguous()


def run_target(
    model: Qwen3,
    layer_idx: int,
    target: str,
    data_dir: Path,
    sample_count: int,
    token_count: int,
    device: torch.device,
) -> dict[str, Any]:
    model.quant_target = [target]
    model.args.quant_target = [target]
    block = model.block(layer_idx)
    quant_block = copy.deepcopy(block)
    if target == "attn-linear":
        apply_quant_to_attn(model.args, quant_block, QuantQwen3Attn)
    elif target == "mlp":
        apply_quant_to_moe_mlp(model.args, quant_block, cls=QuantQwen3MLP)
    else:
        raise ValueError(f"Unsupported target: {target}")

    if target == "attn-linear":
        inp = load_tensor(data_dir / "attn-linear" / f"block_{layer_idx}_attn_in.pkl")
        pos = load_tensor(data_dir / "attn-linear" / "position_embeddings.pkl")
        inp = slice_input(inp, sample_count, token_count).to(device)
        pos = tuple(p[:, :token_count].contiguous().to(device) for p in pos)
        with torch.no_grad():
            ref, _ = block.self_attn(inp, position_embeddings=pos, attention_mask=None)
            out, _ = quant_block.self_attn(
                inp, position_embeddings=pos, attention_mask=None
            )
    else:
        inp = load_tensor(data_dir / "mlp" / f"block_{layer_idx}_mlp_in.pkl")
        inp = slice_input(inp, sample_count, token_count).to(device)
        with torch.no_grad():
            ref = block.mlp(inp)
            out = quant_block.mlp(inp)

    stats = metric(ref, out)
    return {
        "layer": layer_idx,
        "target": target,
        "shape": list(ref.shape),
        **stats,
    }


def layer_rank_from_outlier_json(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    ranked = []
    for layer_idx in range(28):
        scores = []
        for name, item in obj["layers"].items():
            if not name.startswith(f"model.layers.{layer_idx}."):
                continue
            inp = item.get("input")
            if not inp:
                continue
            q = inp["quantiles"]
            med = q["p50"]
            tail = max(abs(q["p01"] - med), abs(q["p99"] - med))
            mad = inp["mad"]
            max_abs = max(abs(inp["min"] - med), abs(inp["max"] - med))
            scores.append(
                {
                    "name": name,
                    "tail_mad": tail / mad if mad else 0.0,
                    "max_tail": max_abs / tail if tail else 0.0,
                }
            )
        if scores:
            ranked.append(
                {
                    "layer": layer_idx,
                    "score": max(max(x["tail_mad"], x["max_tail"]) for x in scores),
                    "best": max(scores, key=lambda x: max(x["tail_mad"], x["max_tail"])),
                }
            )
    return sorted(ranked, key=lambda x: x["score"], reverse=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["layer", "target", "mse", "rel_mse", "cosine", "delta_cosine", "max_abs_err"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    selected_layers = [int(x) for x in args.layers.split(",") if x]
    targets = [x.strip() for x in args.targets.split(",") if x.strip()]
    if not selected_layers:
        raise ValueError("layers cannot be empty")

    model = Qwen3(make_args(args.model, args.data_dir, args.device, targets[0]))

    ranked = layer_rank_from_outlier_json(Path(args.outlier_json))
    ranked_map = {item["layer"]: item for item in ranked}
    results = []
    for layer_idx in selected_layers:
        print(f"[layer {layer_idx}]")
        for target in targets:
            row = run_target(
                model,
                layer_idx,
                target,
                Path(args.data_dir),
                args.sample_count,
                args.token_count,
                device,
            )
            row["outlier_score"] = ranked_map.get(layer_idx, {}).get("score")
            row["outlier_best"] = ranked_map.get(layer_idx, {}).get("best", {})
            results.append(row)
            print(
                f"  {target:10s} cosine={row['cosine']:.6f} "
                f"rel_mse={row['rel_mse']:.6g} max_abs={row['max_abs_err']:.6g}"
            )

    # Stable ordering: lower cosine first, then higher rel_mse.
    results.sort(key=lambda x: (x["cosine"], -x["rel_mse"]))
    output = {
        "model": args.model,
        "data_dir": args.data_dir,
        "sample_count": args.sample_count,
        "token_count": args.token_count,
        "layers": selected_layers,
        "targets": targets,
        "results": results,
    }
    output_path = Path(args.output)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=True), encoding="utf-8")
    write_csv(output_path.with_suffix(".csv"), results)
    print(f"JSON: {output_path}")
    print(f"CSV:  {output_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
