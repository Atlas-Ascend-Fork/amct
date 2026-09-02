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
"""Torch/NPU operators used by the Kimi-K3 delta-attention path."""

import math
from itertools import pairwise

import torch
import torch.nn.functional as F
from torch import nn

_KDA_CHUNK_SIZE = 64


def _is_npu_tensor(tensor: torch.Tensor) -> bool:
    return tensor.device.type == "npu"


def _get_cann_op(name: str):
    try:
        import cann_ops_transformer.ops  # noqa: F401
        return getattr(torch.ops.cann_ops_transformer, name)
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            f"Kimi-K3 NPU short convolution requires "
            f"torch.ops.cann_ops_transformer.{name}."
        ) from error


class KimiShortConvolution(nn.Module):
    """Depthwise causal convolution with CANN NPU and Torch fallbacks.

    The public cache layout is ``[batch, hidden_size, kernel_size]`` and stores
    raw inputs, matching the layout expected by KimiDynamicCache.
    """

    def __init__(self, hidden_size: int, kernel_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(hidden_size, 1, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if _is_npu_tensor(x):
            return self._forward_cann(x, cache, output_final_state, cu_seqlens)
        if cu_seqlens is None:
            return self._forward_torch(x, cache, output_final_state)
        if x.shape[0] != 1:
            raise ValueError("Packed KDA inputs require a singleton batch dimension.")
        outputs = []
        states = []
        for index, (start, end) in enumerate(pairwise(cu_seqlens.tolist())):
            request_cache = None if cache is None else cache[index : index + 1]
            request_output, request_cache = self._forward_torch(
                x[:, start:end], request_cache, True,
            )
            outputs.append(request_output)
            states.append(request_cache)
        final_state = torch.cat(states, dim=0) if output_final_state else None
        return torch.cat(outputs, dim=1), final_state

    def _forward_torch(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None,
        output_final_state: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, _, hidden_size = x.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, got {hidden_size}."
            )
        if cache is None:
            cache = x.new_zeros(batch, hidden_size, self.kernel_size)
        if x.shape[1] == 0:
            return x, cache if output_final_state else None
        history = cache[..., 1:]
        raw_inputs = torch.cat((history, x.transpose(1, 2)), dim=-1)
        output = F.conv1d(
            raw_inputs,
            self.weight,
            groups=self.hidden_size,
        ).transpose(1, 2)
        final_state = raw_inputs[..., -self.kernel_size :]
        return F.silu(output), final_state if output_final_state else None

    def _forward_cann(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, tokens, hidden_size = x.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, got {hidden_size}."
            )
        original_shape = x.shape
        packed_x = x.reshape(-1, hidden_size)
        if cu_seqlens is None:
            query_start_loc = torch.arange(
                0, (batch + 1) * tokens, tokens, device=x.device, dtype=torch.int32,
            )
            sequence_count = batch
        else:
            if batch != 1:
                raise ValueError("Packed KDA inputs require a singleton batch dimension.")
            query_start_loc = cu_seqlens.to(device=x.device, dtype=torch.int32)
            sequence_count = query_start_loc.numel() - 1

        has_initial_state = cache is not None
        if cache is None:
            cache = x.new_zeros(
                sequence_count, hidden_size, self.kernel_size,
            )
        cache_indices = torch.arange(
            sequence_count, device=x.device, dtype=torch.int32,
        )
        weight = self.weight.squeeze(1).transpose(0, 1).contiguous()
        is_single_token_batch = packed_x.shape[0] == sequence_count

        if has_initial_state and is_single_token_batch:
            output = _get_cann_op("causal_conv1d_update")(
                x=packed_x,
                conv_state=cache,
                conv_state_indices=cache_indices,
                weight=weight,
                bias=None,
                query_start_loc=query_start_loc,
                num_accepted_tokens=None,
            )
        else:
            initial_flags = torch.full(
                (sequence_count,),
                int(has_initial_state),
                device=x.device,
                dtype=torch.int32,
            )
            output = _get_cann_op("causal_conv1d_fn")(
                x=packed_x,
                conv_states=cache,
                cache_indices=cache_indices,
                weight=weight,
                bias=None,
                query_start_loc=query_start_loc,
                has_initial_state=initial_flags,
            )
        final_state = cache if output_final_state else None
        return output.reshape(original_shape), final_state


def _l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x.float() * x.float()).sum(dim=-1, keepdim=True) + eps)


def _softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x) + torch.log1p(torch.exp(-torch.abs(x)))


def _torch_chunk_kda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the FLA Torch chunk KDA algorithm on one padded request.

    Adapted from FLA ``naive_chunk_kda`` (MIT), copyright 2023-2026
    Songlin Yang, Yu Zhang and Zhiyuan Li.
    """
    output_dtype = value.dtype
    batch, tokens, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    pad_len = (-tokens) % _KDA_CHUNK_SIZE
    if pad_len:
        query, key, value, decay = (
            F.pad(tensor, (0, 0, 0, 0, 0, pad_len))
            for tensor in (query, key, value, decay)
        )
        beta = F.pad(beta, (0, 0, 0, pad_len))

    chunk_count = query.shape[1] // _KDA_CHUNK_SIZE

    def chunked(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(
            batch, chunk_count, _KDA_CHUNK_SIZE, heads, *tensor.shape[3:]
        ).permute(0, 3, 1, 2, *range(4, tensor.ndim + 1)).float()

    q = chunked(_l2_normalize(query)) / math.sqrt(key_dim)
    k = chunked(_l2_normalize(key))
    v = chunked(value)
    g = chunked(decay).cumsum(dim=-2)
    b = chunked(beta)
    device = q.device
    transition_mask = torch.triu(
        torch.ones(_KDA_CHUNK_SIZE, _KDA_CHUNK_SIZE, device=device, dtype=torch.bool),
    )
    attention_mask = torch.triu(
        torch.ones(_KDA_CHUNK_SIZE, _KDA_CHUNK_SIZE, device=device, dtype=torch.bool),
        diagonal=1,
    )
    identity = torch.eye(_KDA_CHUNK_SIZE, device=device, dtype=torch.float32)

    transition = torch.zeros(
        *g.shape[:-1], _KDA_CHUNK_SIZE, dtype=torch.float32, device=device,
    )
    for index in range(_KDA_CHUNK_SIZE):
        key_i = k[..., index, :]
        decay_i = g[..., index : index + 1, :]
        transition[..., index] = torch.matmul(
            k * (g - decay_i).exp(), key_i.unsqueeze(-1),
        ).squeeze(-1)
    transition = -(transition * b[..., None]).masked_fill(transition_mask, 0)
    for index in range(1, _KDA_CHUNK_SIZE):
        transition[..., index, :index] = transition[..., index, :index].clone() + (
            transition[..., index, :, None].clone()
            * transition[..., :, :index].clone()
        ).sum(-2)
    transition = (transition + identity) * b[..., None, :]

    corrected_key = transition @ (g.exp() * k)
    corrected_value = transition @ v
    state = initial_state.float()
    output = torch.zeros_like(v)
    for index in range(chunk_count):
        q_i = q[:, :, index]
        k_i = k[:, :, index]
        v_i = corrected_value[:, :, index]
        g_i = g[:, :, index]
        w_i = corrected_key[:, :, index]
        attention = torch.zeros(
            batch, heads, _KDA_CHUNK_SIZE, _KDA_CHUNK_SIZE,
            dtype=torch.float32, device=device,
        )
        for token_index in range(_KDA_CHUNK_SIZE):
            key_j = k_i[:, :, token_index]
            decay_j = g_i[:, :, token_index : token_index + 1]
            attention[..., token_index] = torch.matmul(
                q_i * (g_i - decay_j).exp(), key_j.unsqueeze(-1),
            ).squeeze(-1)
        attention = attention.masked_fill(attention_mask, 0)
        v_i = v_i - w_i @ state
        output[:, :, index] = (q_i * g_i.exp()) @ state + attention @ v_i
        final_decay = g_i[:, :, -1]
        state = state * final_decay.exp().unsqueeze(-1)
        state = state + (
            (final_decay.unsqueeze(-2) - g_i).exp() * k_i
        ).transpose(-1, -2) @ v_i

    output = output.permute(0, 2, 3, 1, 4).reshape(
        batch, -1, heads, value_dim,
    )
    return output[:, :tokens].to(output_dtype), state


def torch_chunk_kda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """KDA prefill with public state layout ``[B, H, V, K]``."""
    if cu_seqlens is None:
        if initial_state is None:
            initial_state = value.new_zeros(
                value.shape[0], value.shape[2], value.shape[3], query.shape[3],
                dtype=torch.float32,
            )
        output, state = _torch_chunk_kda(
            query, key, value, decay, beta, initial_state.transpose(-1, -2),
        )
        return output, state.transpose(-1, -2)

    boundaries = cu_seqlens.tolist()
    outputs = []
    states = []
    for index, (start, end) in enumerate(pairwise(boundaries)):
        request_state = None if initial_state is None else initial_state[index : index + 1]
        request_output, request_state = torch_chunk_kda(
            query[:, start:end], key[:, start:end], value[:, start:end],
            decay[:, start:end], beta[:, start:end], request_state,
        )
        outputs.append(request_output)
        states.append(request_state)
    return torch.cat(outputs, dim=1), torch.cat(states, dim=0)


def torch_recurrent_kda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache-aware recurrent KDA with state layout ``[B, H, V, K]``."""
    if cu_seqlens is not None:
        boundaries = cu_seqlens.tolist()
        outputs = []
        states = []
        for index, (start, end) in enumerate(pairwise(boundaries)):
            request_state = None if initial_state is None else initial_state[index : index + 1]
            request_output, request_state = torch_recurrent_kda(
                query[:, start:end], key[:, start:end], value[:, start:end],
                decay[:, start:end], beta[:, start:end], request_state,
            )
            outputs.append(request_output)
            states.append(request_state)
        return torch.cat(outputs, dim=1), torch.cat(states, dim=0)

    key_dim = query.shape[-1]
    q = _l2_normalize(query).float() / math.sqrt(key_dim)
    k = _l2_normalize(key).float()
    v = value.float()
    if initial_state is None:
        state = value.new_zeros(
            value.shape[0], value.shape[2], value.shape[3], key_dim,
            dtype=torch.float32,
        )
    else:
        state = initial_state.float()
    outputs = []
    for token_index in range(query.shape[1]):
        state = state * decay[:, token_index].float().exp().unsqueeze(-2)
        prediction = torch.einsum(
            "bhvk,bhk->bhv", state, k[:, token_index],
        )
        delta = (v[:, token_index] - prediction) * beta[:, token_index].float().unsqueeze(-1)
        state = state + delta.unsqueeze(-1) * k[:, token_index].unsqueeze(-2)
        outputs.append(torch.einsum("bhvk,bhk->bhv", state, q[:, token_index]))
    output = torch.stack(outputs, dim=1).to(value.dtype)
    return output, state


def prepare_kda_gates(
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert projected KDA gates into decay and beta tensors."""
    scale = a_log.float().exp().view(1, 1, -1, 1)
    gate_input = raw_gate.float() + dt_bias.float().view(
        1, 1, raw_gate.shape[-2], raw_gate.shape[-1],
    )
    if lower_bound is None:
        decay = -scale * _softplus(gate_input)
    else:
        decay = lower_bound * torch.sigmoid(scale * gate_input)
    return decay, raw_beta.float().sigmoid()


def rms_norm_gated(
    values: torch.Tensor,
    gate: torch.Tensor,
    norm: nn.Module,
) -> torch.Tensor:
    """Portable RMSNorm plus sigmoid gate used when the fused CUDA op is unavailable."""
    epsilon = getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6))
    values_float = values.float()
    variance = values_float.square().mean(dim=-1, keepdim=True)
    normalized = values_float * torch.rsqrt(variance + epsilon)
    output = normalized * norm.weight.float() * torch.sigmoid(gate.float())
    return output.to(values.dtype)
