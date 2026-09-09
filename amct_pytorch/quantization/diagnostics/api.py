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

from __future__ import annotations

import gc
from typing import Any, Iterable

import torch

from amct_pytorch.quantization.dtypes import register_dtype
from amct_pytorch.quantization.modules.quant_base import ActivationQuantizer

from .config import QuantDiagnosisConfig
from .decision import (
    decide_block_precision,
    decide_layer_precision,
    decide_moe_precision,
    pick_top_k,
)
from .metrics import (
    activation_stats,
    block_score,
    cosine_similarity,
    layer_score,
    max_error,
    mse,
    nmse,
    qsnr,
    relative_error,
)
from .report import BlockMetric, InputOutlierMetric, LayerMetric, MoEMetric, QuantDiagnosisReport

register_dtype()


def _as_samples(data: Any, limit: int) -> list[torch.Tensor]:
    if data is None:
        raise ValueError("diagnosis needs input samples.")
    if torch.is_tensor(data):
        return [data]
    if isinstance(data, (list, tuple)) and data and torch.is_tensor(data[0]):
        samples = list(data)
    else:
        samples = list(data)
    if not samples:
        raise ValueError("diagnosis received an empty sample set.")
    return samples[:limit]


def _maybe_empty_cache():
    try:
        import torch.npu  # type: ignore

        torch.npu.empty_cache()
    except Exception:
        pass


def _clone_state(dst, src):
    for name in ("position_ids", "attention_mask", "position_embeddings", "input_ids"):
        if hasattr(src, name):
            setattr(dst, name, getattr(src, name))


def _to_flat(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    flats = [t.detach().float().reshape(-1).cpu() for t in tensors if torch.is_tensor(t)]
    if not flats:
        return torch.empty(0)
    return torch.cat(flats, dim=0)


def _detach_to_cpu(output):
    if torch.is_tensor(output):
        return output.detach().cpu()
    if isinstance(output, tuple):
        return tuple(_detach_to_cpu(item) for item in output)
    if isinstance(output, list):
        return [_detach_to_cpu(item) for item in output]
    return output


def _first_tensor(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _resolve_attn_name(block):
    layer_type = getattr(block, "layer_type", None)
    if layer_type == "linear_attention" and hasattr(block, "linear_attn"):
        return "linear_attn"
    if hasattr(block, "self_attn"):
        return "self_attn"
    if hasattr(block, "linear_attn"):
        return "linear_attn"
    return None


def _infer_input_outlier_kind(module_name: str) -> str:
    if "self_attn" in module_name or "linear_attn" in module_name:
        return "attention"
    if "mlp" in module_name:
        return "mlp"
    return "activation"


def _run_block_forward(block, hidden_states, forward_kwargs, capture_input_outliers: bool = False):
    captures: dict[str, Any] = {"router": None}
    input_outliers: dict[str, list[torch.Tensor]] = {}
    hooks = []
    named_modules = dict(block.named_modules())

    def _attach(module_name: str | None, key: str):
        if not module_name:
            return
        module = named_modules.get(module_name)
        if module is None:
            return

        def _hook(_module, _inputs, output):
            captures[key] = _detach_to_cpu(output)

        hooks.append(module.register_forward_hook(_hook))

    _attach(_resolve_attn_name(block), "attn_out")
    _attach("mlp", "mlp_out")
    _attach("mlp.gate", "router")
    if capture_input_outliers:
        for module_name, module in named_modules.items():
            if not isinstance(module, ActivationQuantizer):
                continue

            def _pre_hook(_module, inputs, name=module_name):
                tensor = _first_tensor(inputs)
                if tensor is not None:
                    input_outliers.setdefault(name, []).append(tensor.detach().cpu())

            hooks.append(module.register_forward_pre_hook(_pre_hook))

    try:
        with torch.no_grad():
            output = block(hidden_states, **forward_kwargs)
    finally:
        for hook in hooks:
            hook.remove()

    captures["final"] = _detach_to_cpu(output[0] if isinstance(output, (tuple, list)) else output)
    captures["input_outliers"] = input_outliers
    return captures


def _top_layer_precision(metric, cfg: QuantDiagnosisConfig) -> int:
    precision = decide_layer_precision(metric, cfg)
    return max(cfg.precision_map["sensitive"], precision)


class QuantDiagnosisRunner:
    def __init__(self, adapter, config: QuantDiagnosisConfig | None = None):
        self.adapter = adapter
        self.config = config or QuantDiagnosisConfig()

    def diagnose(self, data: Any) -> QuantDiagnosisReport:
        samples = _as_samples(data, self.config.sample_limit)
        report = QuantDiagnosisReport()

        float_inputs = self.adapter.do_embedding_forward(samples)
        block_forward_kwargs = (
            self.adapter.get_block_forward_kwargs()
            if hasattr(self.adapter, "get_block_forward_kwargs")
            else {}
        )

        current_float = float_inputs
        current_quant = [tensor.clone() for tensor in float_inputs]

        layer_limit = min(getattr(self.adapter, "num_layers", len(float_inputs)), self.config.layer_limit)
        block_metrics: list[BlockMetric] = []
        layer_metrics: list[LayerMetric] = []
        input_outlier_tensors: dict[tuple[str, int, str], list[torch.Tensor]] = {}
        moe_metrics: list[MoEMetric] = []

        for layer_idx in range(layer_limit):
            block_name = f"layers.{layer_idx}"
            float_block = self.adapter.block(layer_idx).to(self.adapter.args.device).eval()
            quant_block = self.adapter.build_quant_block(layer_idx).to(self.adapter.args.device).eval()
            if hasattr(self.adapter, "load_selected_layer_ptq_params"):
                self.adapter.load_selected_layer_ptq_params(layer_idx, quant_block, strict=False)

            forward_kwargs = {
                key: value.to(self.adapter.args.device)
                if torch.is_tensor(value)
                else tuple(
                    item.to(self.adapter.args.device) if torch.is_tensor(item) else item
                    for item in value
                )
                if isinstance(value, (tuple, list))
                else value
                for key, value in block_forward_kwargs.items()
            }

            float_acts = []
            quant_acts = []
            float_attn_outs = []
            quant_attn_outs = []
            float_mlp_outs = []
            quant_mlp_outs = []
            float_router = None
            quant_router = None

            for float_hidden, quant_hidden in zip(current_float, current_quant):
                float_res = _run_block_forward(
                    float_block,
                    float_hidden.to(self.adapter.args.device),
                    forward_kwargs,
                )
                quant_res = _run_block_forward(
                    quant_block,
                    quant_hidden.to(self.adapter.args.device),
                    forward_kwargs,
                    capture_input_outliers=True,
                )
                float_acts.append(float_res["final"])
                quant_acts.append(quant_res["final"])
                if "attn_out" in float_res:
                    float_attn_outs.append(_first_tensor(float_res["attn_out"]))
                if "attn_out" in quant_res:
                    quant_attn_outs.append(_first_tensor(quant_res["attn_out"]))
                if "mlp_out" in float_res:
                    float_mlp_outs.append(_first_tensor(float_res["mlp_out"]))
                if "mlp_out" in quant_res:
                    quant_mlp_outs.append(_first_tensor(quant_res["mlp_out"]))
                if float_res.get("router") is not None:
                    float_router = float_res["router"]
                if quant_res.get("router") is not None:
                    quant_router = quant_res["router"]
                for module_name, tensors in quant_res.get("input_outliers", {}).items():
                    metric_name = f"{block_name}.{module_name}"
                    key = (
                        metric_name,
                        layer_idx,
                        _infer_input_outlier_kind(metric_name),
                    )
                    input_outlier_tensors.setdefault(key, []).extend(tensors)

            float_final = _to_flat(float_acts)
            quant_final = _to_flat(quant_acts)
            float_attn = _to_flat(float_attn_outs)
            quant_attn = _to_flat(quant_attn_outs)
            float_mlp = _to_flat(float_mlp_outs)
            quant_mlp = _to_flat(quant_mlp_outs)

            block_metric = BlockMetric(
                name=block_name,
                index=layer_idx,
                qsnr=qsnr(float_final, quant_final),
                nmse=nmse(float_final, quant_final),
                score=block_score(float_final, quant_final),
            )
            block_metrics.append(block_metric)

            attn_name = _resolve_attn_name(float_block) or "self_attn"
            layer_metrics.append(
                LayerMetric(
                    name=f"{block_name}.{attn_name}",
                    index=layer_idx,
                    kind="attention",
                    qsnr=qsnr(float_attn, quant_attn),
                    mse=mse(float_attn, quant_attn),
                    relative_error=relative_error(float_attn, quant_attn),
                    cosine_similarity=cosine_similarity(float_attn, quant_attn),
                    max_error=max_error(float_attn, quant_attn),
                    score=layer_score(float_attn, quant_attn),
                    precision=self.config.precision_map["normal"],
                )
            )
            layer_metrics.append(
                LayerMetric(
                    name=f"{block_name}.mlp",
                    index=layer_idx,
                    kind="mlp",
                    qsnr=qsnr(float_mlp, quant_mlp),
                    mse=mse(float_mlp, quant_mlp),
                    relative_error=relative_error(float_mlp, quant_mlp),
                    cosine_similarity=cosine_similarity(float_mlp, quant_mlp),
                    max_error=max_error(float_mlp, quant_mlp),
                    score=layer_score(float_mlp, quant_mlp),
                    precision=self.config.precision_map["normal"],
                )
            )

            if float_router is not None and quant_router is not None:
                float_router_indices = _first_tensor(float_router[-1] if isinstance(float_router, (tuple, list)) else float_router)
                quant_router_indices = _first_tensor(quant_router[-1] if isinstance(quant_router, (tuple, list)) else quant_router)
                if float_router_indices is not None and quant_router_indices is not None:
                    float_router_indices = float_router_indices.reshape(-1).to(torch.int64)
                    quant_router_indices = quant_router_indices.reshape(-1).to(torch.int64)
                    gate_module = getattr(getattr(float_block, "mlp", None), "gate", None)
                    if gate_module is None:
                        gate_module = getattr(getattr(quant_block, "mlp", None), "gate", None)
                    num_experts = int(getattr(gate_module, "num_experts", 0) or 0)
                    float_freq = torch.bincount(float_router_indices, minlength=num_experts).float()
                    quant_freq = torch.bincount(quant_router_indices, minlength=num_experts).float()
                    router_mismatch = (float_router_indices != quant_router_indices).float().mean().item()
                    expert_frequency = {
                        str(i): float(v.item())
                        for i, v in enumerate(float_freq / max(float_freq.sum().item(), 1e-12))
                        if v.item() > 0
                    }
                    expert_error = {
                        str(i): float(abs(float_freq[i].item() - quant_freq[i].item()))
                        for i in range(num_experts)
                        if float_freq[i].item() > 0 or quant_freq[i].item() > 0
                    }
                    moe_metrics.append(
                        MoEMetric(
                            name=f"{block_name}.moe",
                            index=layer_idx,
                            expert_frequency=expert_frequency,
                            expert_error=expert_error,
                            routing_mismatch=router_mismatch,
                            score=router_mismatch + block_metric.score,
                            precision=self.config.precision_map["sensitive"],
                        )
                    )

            current_float = [tensor.detach().cpu() for tensor in float_acts]
            current_quant = [tensor.detach().cpu() for tensor in quant_acts]

            if self.config.early_stop_enabled and layer_idx + 1 >= layer_limit:
                break

            del float_block, quant_block
            gc.collect()
            _maybe_empty_cache()

        if block_metrics:
            qsnr_sorted = sorted(m.qsnr for m in block_metrics)
            nmse_sorted = sorted(m.nmse for m in block_metrics)
            q_idx = max(
                0,
                min(
                    len(qsnr_sorted) - 1,
                    int(len(qsnr_sorted) * self.config.block_qsnr_percentile / 100.0),
                ),
            )
            n_idx = max(
                0,
                min(
                    len(nmse_sorted) - 1,
                    int(len(nmse_sorted) * self.config.block_nmse_percentile / 100.0),
                ),
            )
            qsnr_cutoff = qsnr_sorted[q_idx]
            nmse_cutoff = nmse_sorted[n_idx]
        else:
            qsnr_cutoff = 0.0
            nmse_cutoff = 0.0

        ranked_blocks = sorted(block_metrics, key=lambda m: m.score, reverse=True)
        top_k = pick_top_k(len(ranked_blocks), self.config)
        sensitive_blocks = ranked_blocks[:top_k]
        for metric in sensitive_blocks:
            metric.precision = decide_block_precision(metric, self.config)

        precision_plan: dict[str, int] = {}
        for metric in block_metrics:
            precision_plan[metric.name] = metric.precision

        ranked_layers = sorted(layer_metrics, key=lambda m: m.score, reverse=True)
        for metric in ranked_layers[: max(1, min(len(ranked_layers), top_k))]:
            metric.precision = _top_layer_precision(metric, self.config)
        for metric in layer_metrics:
            if metric.kind in {"attention", "mlp"}:
                precision_plan[metric.name] = metric.precision

        for metric in moe_metrics:
            metric.precision = decide_moe_precision(metric, self.config)
            precision_plan[metric.name] = metric.precision

        input_outlier_metrics = []
        for (name, layer_idx, kind), tensors in input_outlier_tensors.items():
            stats = activation_stats(tensors)
            token_ratio_p99 = stats["ratio"]
            input_outlier_metrics.append(
                InputOutlierMetric(
                    name=name,
                    index=layer_idx,
                    kind=kind,
                    max_abs=stats["max"],
                    p99_abs=stats["p99"],
                    p999_abs=stats["p999"],
                    token_ratio_p99=token_ratio_p99,
                    token_ratio_max=stats["ratio_max"],
                    max_to_p999=token_ratio_p99,
                    sample_count=len(tensors),
                    token_count=int(stats["token_count"]),
                )
            )
        input_outlier_metrics.sort(key=lambda metric: metric.token_ratio_p99, reverse=True)

        report.block_metrics = block_metrics
        report.layer_metrics = layer_metrics
        report.input_outlier_metrics = input_outlier_metrics
        report.moe_metrics = moe_metrics
        report.mixed_precision_plan = precision_plan
        report.notes.append(f"block_qsnr_cutoff={qsnr_cutoff:.4f}")
        report.notes.append(f"block_nmse_cutoff={nmse_cutoff:.4f}")
        if not input_outlier_metrics:
            report.notes.append("input_outlier_metrics=skipped(no ActivationQuantizer inputs captured)")
        report.notes.append(f"top_k={top_k}")
        return report


def diagnose_quantization_error(adapter, data: Any, config: QuantDiagnosisConfig | None = None) -> QuantDiagnosisReport:
    return QuantDiagnosisRunner(adapter, config=config).diagnose(data)
