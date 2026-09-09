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

"""统计 Qwen3 线性层输入激活和权重的分布特征。"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


DEFAULT_QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


class StatsAccumulator:
    """流式计算精确的一阶统计量，并保留固定大小样本估计分位数和 MAD。"""

    def __init__(self, reservoir_size: int, seed: int) -> None:
        self.reservoir_size = reservoir_size
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)
        self.count = 0
        self.sum = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.reservoir = torch.empty(0, dtype=torch.float32)

    @staticmethod
    def _sample(data: torch.Tensor, size: int, generator: torch.Generator) -> torch.Tensor:
        if size >= data.numel():
            return data
        indices = torch.randint(
            data.numel(), (size,), generator=generator, dtype=torch.int64
        )
        return data[indices]

    def update(self, tensor: torch.Tensor) -> None:
        data = tensor.detach().float().cpu().reshape(-1)
        if data.numel() == 0:
            return

        old_count = self.count
        new_count = old_count + data.numel()
        self.count = new_count
        self.sum += data.double().sum().item()
        self.minimum = min(self.minimum, data.min().item())
        self.maximum = max(self.maximum, data.max().item())

        current_size = self.reservoir.numel()
        if current_size < self.reservoir_size:
            if data.numel() <= self.reservoir_size - current_size:
                self.reservoir = torch.cat((self.reservoir, data))
                return
            old = self.reservoir
            new_part = self._sample(
                data, self.reservoir_size - current_size, self.generator
            )
            self.reservoir = torch.cat((old, new_part))
            return

        # Keep a size-weighted sample from the old reservoir and this chunk.
        new_reservoir_size = max(
            1,
            min(
                self.reservoir_size,
                round(self.reservoir_size * data.numel() / new_count),
            ),
        )
        old_reservoir_size = self.reservoir_size - new_reservoir_size
        old_part = self._sample(
            self.reservoir, old_reservoir_size, self.generator
        )
        new_part = self._sample(data, new_reservoir_size, self.generator)
        self.reservoir = torch.cat((old_part, new_part))

    def summarize(self, quantiles: tuple[float, ...]) -> dict[str, Any]:
        if self.count == 0:
            return {"count": 0}
        values = self.reservoir.double()
        requested_quantiles = tuple(sorted(set(quantiles + (0.5,))))
        quantile_tensor = torch.tensor(requested_quantiles, dtype=values.dtype)
        quantile_results = torch.quantile(values, quantile_tensor)
        median = quantile_results[requested_quantiles.index(0.5)].item()
        mad = torch.median(torch.abs(values - median)).item()
        quantile_values = {
            f"p{int(round(q * 100)):02d}": quantile_results[
                requested_quantiles.index(q)
            ].item()
            for q in quantiles
        }
        return {
            "count": self.count,
            "mean": self.sum / self.count,
            "min": self.minimum,
            "max": self.maximum,
            "quantiles": quantile_values,
            "mad": mad,
            "mad_definition": "median(abs(x - median(x)))",
            "quantile_source": "activation reservoir",
            "reservoir_count": int(self.reservoir.numel()),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Qwen3 Linear input activations and weights."
    )
    parser.add_argument("--model", required=True, help="Local model directory.")
    parser.add_argument(
        "--data-dir",
        default="ptq_data/qwen3",
        help="Directory containing attn-linear/ and mlp/ PTQ input files.",
    )
    parser.add_argument(
        "--output",
        default="qwen3_linear_stats.json",
        help="Output JSON path. A CSV with the same stem is also written.",
    )
    parser.add_argument("--device", default="cpu", help="cpu or npu:0.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum calibration samples per block; default uses all samples.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens per sample; default uses the complete sequence.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of calibration samples processed per block forward.",
    )
    parser.add_argument(
        "--reservoir-size",
        type=int,
        default=200_000,
        help="Activation values retained for quantiles and MAD.",
    )
    parser.add_argument(
        "--quantiles",
        default=",".join(str(q) for q in DEFAULT_QUANTILES),
        help="Comma-separated quantile probabilities.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_tensor(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, weights_only=True, mmap=True)
    except TypeError:
        return torch.load(path, weights_only=True)


def load_position_kwargs(data_dir: Path, max_tokens: int | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    position_ids_path = data_dir / "mlp" / "position_ids.pkl"
    embeddings_path = data_dir / "mlp" / "position_embeddings.pkl"
    attention_mask_path = data_dir / "mlp" / "attention_mask.pkl"
    if position_ids_path.exists():
        kwargs["position_ids"] = load_tensor(position_ids_path)
    if embeddings_path.exists():
        kwargs["position_embeddings"] = load_tensor(embeddings_path)
    if attention_mask_path.exists():
        kwargs["attention_mask"] = load_tensor(attention_mask_path)

    if max_tokens is None:
        return kwargs
    if "position_ids" in kwargs:
        kwargs["position_ids"] = kwargs["position_ids"][..., :max_tokens]
    if "position_embeddings" in kwargs:
        kwargs["position_embeddings"] = tuple(
            item[..., :max_tokens, :] for item in kwargs["position_embeddings"]
        )
    if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
        kwargs["attention_mask"] = kwargs["attention_mask"][
            ..., :max_tokens, :max_tokens
        ]
    return kwargs


def make_accumulator(
    accumulators: dict[str, StatsAccumulator],
    name: str,
    reservoir_size: int,
    seed: int,
) -> StatsAccumulator:
    if name not in accumulators:
        accumulators[name] = StatsAccumulator(reservoir_size, seed)
    return accumulators[name]


def collect_activation_stats(
    block: nn.Module,
    block_name: str,
    inputs: torch.Tensor,
    block_kwargs: dict[str, Any],
    target: str,
    accumulators: dict[str, StatsAccumulator],
    reservoir_size: int,
    seed: int,
    device: torch.device,
    batch_size: int,
) -> None:
    hooks = []

    def make_hook(name: str):
        def hook(_module: nn.Module, args: tuple[Any, ...]) -> None:
            if args:
                make_accumulator(
                    accumulators, name, reservoir_size, seed
                ).update(args[0])

        return hook

    target_module = block.self_attn if target == "attn-linear" else block.mlp
    for name, module in target_module.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        full_name = f"{block_name}.{'self_attn' if target == 'attn-linear' else 'mlp'}.{name}"
        hooks.append(module.register_forward_pre_hook(make_hook(full_name)))

    forward_kwargs = {}
    if target == "attn-linear":
        forward_kwargs["position_embeddings"] = tuple(
            value.to(device) for value in block_kwargs["position_embeddings"]
        )
        forward_kwargs["attention_mask"] = block_kwargs.get("attention_mask")
        if isinstance(forward_kwargs["attention_mask"], torch.Tensor):
            forward_kwargs["attention_mask"] = forward_kwargs["attention_mask"].to(device)
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            batch = inputs[start : start + batch_size].to(device)
            target_module(batch, **forward_kwargs)

    for hook in hooks:
        hook.remove()


def collect_weight_stats(
    model: nn.Module,
    quantiles: tuple[float, ...],
    reservoir_size: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    def summarize_weight(weight: torch.Tensor) -> dict[str, Any]:
        values = weight.detach().float().cpu().reshape(-1)
        exact_limit = 10_000_000
        if values.numel() <= exact_limit:
            values = values.double()
            requested_quantiles = tuple(sorted(set(quantiles + (0.5,))))
            quantile_tensor = torch.tensor(requested_quantiles, dtype=values.dtype)
            quantile_results = torch.quantile(values, quantile_tensor)
            median = quantile_results[requested_quantiles.index(0.5)].item()
            return {
                "count": int(values.numel()),
                "mean": values.mean().item(),
                "min": values.min().item(),
                "max": values.max().item(),
                "quantiles": {
                    f"p{int(round(q * 100)):02d}": quantile_results[
                        requested_quantiles.index(q)
                    ].item()
                    for q in quantiles
                },
                "mad": torch.median(torch.abs(values - median)).item(),
                "mad_definition": "median(abs(x - median(x)))",
                "quantile_source": "full tensor",
                "reservoir_count": int(values.numel()),
            }

        accumulator = StatsAccumulator(reservoir_size, seed)
        accumulator.update(values)
        result = accumulator.summarize(quantiles)
        result["quantile_source"] = "weight reservoir (tensor exceeds PyTorch quantile limit)"
        return result

    stats: dict[str, dict[str, Any]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        stats[name] = summarize_weight(module.weight)
        stats[name]["shape"] = list(module.weight.shape)
    return stats


def write_csv(path: Path, records: list[dict[str, Any]], quantiles: tuple[float, ...]) -> None:
    fields = ["name", "kind", "shape", "count", "mean", "min", "max", "mad"]
    fields.extend(f"p{int(round(q * 100)):02d}" for q in quantiles)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key) for key in fields}
            row["shape"] = json.dumps(row["shape"], ensure_ascii=True)
            row.update(record.get("quantiles", {}))
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    quantiles = tuple(float(item) for item in args.quantiles.split(","))
    if any(q < 0.0 or q > 1.0 for q in quantiles):
        raise ValueError("Quantiles must be within [0, 1].")

    model_path = Path(args.model)
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=None,
    ).eval()
    model.to(device)

    weight_stats = collect_weight_stats(model, quantiles, args.reservoir_size, args.seed)
    activation_stats: dict[str, StatsAccumulator] = {}
    position_kwargs = load_position_kwargs(data_dir, args.max_tokens)
    num_layers = len(model.model.layers)

    for layer_idx in range(num_layers):
        block = model.model.layers[layer_idx]
        block_name = f"model.layers.{layer_idx}"
        for target, subdir in (
            ("attn-linear", "attn-linear"),
            ("mlp", "mlp"),
        ):
            file_target = "attn" if target == "attn-linear" else target
            input_path = data_dir / subdir / f"block_{layer_idx}_{file_target}_in.pkl"
            if not input_path.exists():
                raise FileNotFoundError(input_path)
            raw_inputs = load_tensor(input_path)
            sample_count = (
                raw_inputs.shape[0]
                if args.max_samples is None
                else min(args.max_samples, raw_inputs.shape[0])
            )
            token_count = (
                raw_inputs.shape[1]
                if args.max_tokens is None
                else min(args.max_tokens, raw_inputs.shape[1])
            )
            inputs = raw_inputs[:sample_count, :token_count].contiguous()
            del raw_inputs
            collect_activation_stats(
                block,
                block_name,
                inputs,
                position_kwargs,
                target,
                activation_stats,
                args.reservoir_size,
                args.seed + layer_idx,
                device,
                args.batch_size,
            )
            del inputs
            gc.collect()
        print(f"processed layer {layer_idx + 1}/{num_layers}", flush=True)

    layers: dict[str, dict[str, Any]] = {}
    records = []
    for name, weight in weight_stats.items():
        item = {"shape": weight.pop("shape"), "weight": weight}
        if name in activation_stats:
            item["input"] = activation_stats[name].summarize(quantiles)
        layers[name] = item
        for kind, values in (("weight", item["weight"]), ("input", item.get("input"))):
            if values is None:
                continue
            records.append(
                {
                    "name": name,
                    "kind": kind,
                    "shape": item["shape"],
                    **values,
                }
            )

    result = {
        "model": str(model_path),
        "data_dir": str(data_dir),
        "device": str(device),
        "activation_scope": {
            "max_samples_per_block": args.max_samples,
            "max_tokens_per_sample": args.max_tokens,
            "batch_size": args.batch_size,
            "reservoir_size": args.reservoir_size,
        },
        "quantiles": list(quantiles),
        "layers": layers,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    write_csv(output_path.with_suffix(".csv"), records, quantiles)
    print(f"JSON: {output_path}")
    print(f"CSV:  {output_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
