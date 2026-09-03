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
"""Pure Torch small-operator implementation for Kimi-K3 KDA."""

import math
from itertools import pairwise

import torch
import torch.nn.functional as F
from torch import nn


def _softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x) + torch.log1p(torch.exp(-torch.abs(x)))


def _l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.float().square().sum(dim=-1, keepdim=True) + eps)


class KimiShortConvolution(nn.Module):
    """Depthwise causal convolution implemented with standard Torch ops."""

    def __init__(self, hidden_size: int, kernel_size: int) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}.")
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(
            torch.empty(hidden_size, 1, kernel_size, dtype=torch.float32),
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if cu_seqlens is None:
            output, final_state = self._forward_dense(x, cache)
        else:
            output, final_state = self._forward_packed(x, cache, cu_seqlens)
        return output, final_state if output_final_state else None

    def _forward_dense(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, tokens, hidden_size = x.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, got {hidden_size}.",
            )
        state_size = self.kernel_size - 1
        channel_first = x.transpose(1, 2)
        if cache is None:
            cache = x.new_zeros(batch, hidden_size, state_size)
        expected_shape = (batch, hidden_size, state_size)
        if cache.shape != expected_shape:
            raise ValueError(
                f"Expected convolution state shape {expected_shape}, "
                f"got {tuple(cache.shape)}.",
            )
        if tokens == 0:
            return x, cache

        history = torch.cat((cache, channel_first), dim=-1)
        output = F.conv1d(
            history.float(),
            self.weight.float(),
            groups=self.hidden_size,
        ).transpose(1, 2)
        final_state = history[..., -state_size:] if state_size else history[..., :0]
        return F.silu(output).to(x.dtype), final_state

    def _forward_packed(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[0] != 1:
            raise ValueError("Packed convolution input must have batch size 1.")
        outputs = []
        states = []
        for request, (start, end) in enumerate(pairwise(cu_seqlens.tolist())):
            request_cache = None if cache is None else cache[request : request + 1]
            output, state = self._forward_dense(x[:, start:end], request_cache)
            outputs.append(output)
            states.append(state)
        return torch.cat(outputs, dim=1), torch.cat(states, dim=0)


def recurrent_kda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run KDA using small Torch ops and return ``[B, H, Dv, Dk]`` state."""
    if cu_seqlens is not None:
        if query.shape[0] != 1:
            raise ValueError("Packed KDA input must have batch size 1.")
        outputs = []
        states = []
        for request, (start, end) in enumerate(pairwise(cu_seqlens.tolist())):
            request_state = (
                None if initial_state is None else initial_state[request : request + 1]
            )
            output, state = recurrent_kda(
                query[:, start:end],
                key[:, start:end],
                value[:, start:end],
                decay[:, start:end],
                beta[:, start:end],
                initial_state=request_state,
            )
            outputs.append(output)
            states.append(state)
        return torch.cat(outputs, dim=1), torch.cat(states, dim=0)

    batch, tokens, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    q = _l2_normalize(query).transpose(1, 2).float() / math.sqrt(key_dim)
    k = _l2_normalize(key).transpose(1, 2).float()
    v = value.transpose(1, 2).float()
    g = decay.transpose(1, 2).float()
    b = beta.transpose(1, 2).float()
    if initial_state is None:
        state = torch.zeros(
            batch, heads, key_dim, value_dim,
            dtype=torch.float32,
            device=query.device,
        )
    else:
        expected_shape = (batch, heads, value_dim, key_dim)
        if initial_state.shape != expected_shape:
            raise ValueError(
                f"Expected recurrent state shape {expected_shape}, "
                f"got {tuple(initial_state.shape)}.",
            )
        state = initial_state.transpose(-1, -2).float()

    outputs = []
    for token_index in range(tokens):
        q_t = q[:, :, token_index]
        k_t = k[:, :, token_index]
        v_t = v[:, :, token_index]
        state = state * g[:, :, token_index].exp().unsqueeze(-1)
        memory = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - memory) * b[:, :, token_index].unsqueeze(-1)
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append((state * q_t.unsqueeze(-1)).sum(dim=-2))

    if outputs:
        output = torch.stack(outputs, dim=2).transpose(1, 2).to(value.dtype)
    else:
        output = value.new_empty(batch, 0, heads, value_dim)
    return output, state.transpose(-1, -2).contiguous()


def prepare_kda_gates(
    raw_decay: torch.Tensor,
    raw_beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Activate KDA decay per head and beta per token/head."""
    heads, head_dim = raw_decay.shape[-2:]
    if a_log.shape != (heads,):
        raise ValueError(
            f"Expected A_log shape ({heads},), got {tuple(a_log.shape)}.",
        )
    gate_input = raw_decay.float() + dt_bias.float().view(1, 1, heads, head_dim)
    gate_scale = a_log.float().exp().view(1, 1, heads, 1)
    if lower_bound is None:
        decay = -gate_scale * _softplus(gate_input)
    else:
        decay = float(lower_bound) * torch.sigmoid(gate_scale * gate_input)
    return decay, raw_beta.float().sigmoid()
