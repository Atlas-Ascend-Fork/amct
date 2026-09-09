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

import math

import torch


def _flatten_pair(ref: torch.Tensor, cand: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ref = ref.detach().float().reshape(-1)
    cand = cand.detach().float().reshape(-1)
    if ref.numel() != cand.numel():
        raise ValueError(
            f"metric tensors must have the same size, got {ref.numel()} and {cand.numel()}."
        )
    return ref, cand


def mse(ref: torch.Tensor, cand: torch.Tensor) -> float:
    ref, cand = _flatten_pair(ref, cand)
    return torch.mean((ref - cand) ** 2).item()


def relative_error(ref: torch.Tensor, cand: torch.Tensor) -> float:
    ref, cand = _flatten_pair(ref, cand)
    denom = torch.norm(ref).clamp_min(1e-12)
    return (torch.norm(ref - cand) / denom).item()


def cosine_similarity(ref: torch.Tensor, cand: torch.Tensor) -> float:
    ref, cand = _flatten_pair(ref, cand)
    ref_norm = torch.norm(ref).clamp_min(1e-12)
    cand_norm = torch.norm(cand).clamp_min(1e-12)
    return torch.dot(ref, cand).div(ref_norm * cand_norm).item()


def max_error(ref: torch.Tensor, cand: torch.Tensor) -> float:
    ref, cand = _flatten_pair(ref, cand)
    return torch.max(torch.abs(ref - cand)).item()


def nmse(ref: torch.Tensor, cand: torch.Tensor) -> float:
    ref, cand = _flatten_pair(ref, cand)
    denom = torch.mean(ref**2).clamp_min(1e-12)
    return torch.mean((ref - cand) ** 2).div(denom).item()


def qsnr(ref: torch.Tensor, cand: torch.Tensor) -> float:
    ref, cand = _flatten_pair(ref, cand)
    noise = torch.mean((ref - cand) ** 2).clamp_min(1e-12)
    signal = torch.mean(ref**2).clamp_min(1e-12)
    return (10.0 * math.log10(signal.item() / noise.item()))


def activation_stats(tensors: torch.Tensor | list[torch.Tensor]) -> dict[str, float]:
    """Measure activation outliers at per-token granularity.

    Integer activation fake quant uses a dynamic scale per token over the last
    dimension. Therefore outlier statistics should first be computed inside
    each token vector, then aggregated across tokens.
    """
    if torch.is_tensor(tensors):
        tensor_list = [tensors]
    else:
        tensor_list = tensors

    token_max_values = []
    token_p99_values = []
    token_p999_values = []
    token_ratio_values = []
    for tensor in tensor_list:
        if not torch.is_tensor(tensor) or tensor.numel() == 0:
            continue
        abs_tensor = tensor.detach().float().abs()
        if abs_tensor.dim() == 0:
            abs_tokens = abs_tensor.reshape(1, 1)
        elif abs_tensor.dim() == 1:
            abs_tokens = abs_tensor.reshape(1, -1)
        else:
            abs_tokens = abs_tensor.reshape(-1, abs_tensor.shape[-1])
        token_max = abs_tokens.max(dim=-1).values
        token_p99 = torch.quantile(abs_tokens, 0.99, dim=-1)
        token_p999 = torch.quantile(abs_tokens, 0.999, dim=-1)
        token_ratio = token_max / token_p999.clamp_min(1e-12)
        token_max_values.append(token_max.cpu())
        token_p99_values.append(token_p99.cpu())
        token_p999_values.append(token_p999.cpu())
        token_ratio_values.append(token_ratio.cpu())

    if not token_ratio_values:
        return {
            "max": 0.0,
            "p99": 0.0,
            "p999": 0.0,
            "ratio": 0.0,
            "ratio_max": 0.0,
            "token_count": 0.0,
        }

    token_max_all = torch.cat(token_max_values, dim=0)
    token_p99_all = torch.cat(token_p99_values, dim=0)
    token_p999_all = torch.cat(token_p999_values, dim=0)
    token_ratio_all = torch.cat(token_ratio_values, dim=0)
    return {
        "max": token_max_all.max().item(),
        "p99": torch.quantile(token_p99_all, 0.99).item(),
        "p999": torch.quantile(token_p999_all, 0.99).item(),
        "ratio": torch.quantile(token_ratio_all, 0.99).item(),
        "ratio_max": token_ratio_all.max().item(),
        "token_count": float(token_ratio_all.numel()),
    }


def block_score(ref: torch.Tensor, cand: torch.Tensor) -> float:
    nmse_val = nmse(ref, cand)
    qsnr_val = qsnr(ref, cand)
    return nmse_val + max(0.0, 20.0 - qsnr_val) / 20.0


def layer_score(ref: torch.Tensor, cand: torch.Tensor) -> float:
    return (
        mse(ref, cand)
        + relative_error(ref, cand)
        + max_error(ref, cand)
        + max(0.0, 1.0 - cosine_similarity(ref, cand))
    )
