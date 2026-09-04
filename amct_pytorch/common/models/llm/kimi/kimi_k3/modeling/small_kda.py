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
"""Small-operator adapters for Kimi-K3 KDA on Ascend NPU."""

import math
from itertools import pairwise

import torch
import torch.nn.functional as F
from torch import nn

from ops.cannbot_dsl.flash_kda import flash_kda as _flash_kda_impl
from ops.cannbot_dsl.fused_recurrent_kda import (
    fused_recurrent_kda_op as _recurrent_kda_impl,
)

_KDA_CHUNK_SIZE = 64


def _request_ranges(
    tensor: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> tuple[list[tuple[int, int, int]], bool]:
    if cu_seqlens is None:
        return [
            (request, 0, tensor.shape[1])
            for request in range(tensor.shape[0])
        ], False
    if tensor.shape[0] != 1:
        raise ValueError("Packed KDA input must have batch size 1.")
    boundaries = cu_seqlens.tolist()
    return [
        (request, start, end)
        for request, (start, end) in enumerate(pairwise(boundaries))
    ], True


def _initial_state(
    q: torch.Tensor,
    v: torch.Tensor,
    request_count: int,
    initial_state: torch.Tensor | None,
) -> torch.Tensor:
    expected_shape = (
        request_count, q.shape[2], v.shape[-1], q.shape[-1],
    )
    if initial_state is None:
        return torch.zeros(
            expected_shape, dtype=torch.float32, device=q.device,
        )
    if initial_state.shape != expected_shape:
        raise ValueError(
            f"Expected recurrent state shape {expected_shape}, "
            f"got {tuple(initial_state.shape)}.",
        )
    return initial_state


def _validate_fused_options(
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    use_beta_sigmoid_in_kernel: bool,
    transpose_state_layout: bool,
) -> None:
    if not all((
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        use_beta_sigmoid_in_kernel,
        transpose_state_layout,
    )):
        raise ValueError(
            "The Kimi-K3 NPU KDA adapter requires in-kernel Q/K normalization, "
            "gate activation, beta sigmoid, and V-first state layout.",
        )


def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    use_beta_sigmoid_in_kernel: bool,
    safe_gate: bool,
    lower_bound: float | None,
    transpose_state_layout: bool,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the external Flash-KDA prefill operator per logical request."""
    _validate_fused_options(
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        use_beta_sigmoid_in_kernel,
        transpose_state_layout,
    )
    if safe_gate != (lower_bound is not None):
        raise ValueError("safe_gate and lower_bound must select the same gate path.")
    ranges, packed = _request_ranges(q, cu_seqlens)
    state = _initial_state(q, v, len(ranges), initial_state)
    gate_bias = dt_bias.reshape(q.shape[2], q.shape[-1]).float().contiguous()
    a_log = A_log.detach().float().contiguous()
    outputs = []
    final_states = []

    for request, start, end in ranges:
        source_batch = 0 if packed else request
        request_slice = slice(source_batch, source_batch + 1)
        q_i = q[request_slice, start:end]
        k_i = k[request_slice, start:end]
        v_i = v[request_slice, start:end]
        g_i = g[request_slice, start:end]
        beta_i = beta[request_slice, start:end]
        tokens = end - start
        if tokens == 0:
            outputs.append(v_i)
            final_states.append(state[request : request + 1])
            continue

        pad_len = (-tokens) % _KDA_CHUNK_SIZE
        if pad_len:
            q_i, k_i, v_i = (
                F.pad(tensor, (0, 0, 0, 0, 0, pad_len))
                for tensor in (q_i, k_i, v_i)
            )
            g_i = F.pad(
                g_i, (0, 0, 0, 0, 0, pad_len), value=float("-inf"),
            )
            beta_i = F.pad(
                beta_i, (0, 0, 0, pad_len), value=float("-inf"),
            )
        q_i, k_i, v_i, g_i, beta_i = (
            tensor.contiguous() for tensor in (q_i, k_i, v_i, g_i, beta_i)
        )
        output, request_state = _flash_kda_impl(
            q_i,
            k_i,
            v_i,
            g=g_i,
            beta=beta_i,
            scale=1.0 / math.sqrt(q.shape[-1]),
            initial_state=state[request : request + 1],
            A_log=a_log,
            dt_bias=gate_bias,
            lower_bound=lower_bound,
            layout_qkv="BSND",
        )
        outputs.append(output[:, :tokens])
        final_states.append(request_state)

    output_dim = 1 if packed else 0
    output = torch.cat(outputs, dim=output_dim)
    final_state = torch.cat(final_states, dim=0)
    return output, final_state if output_final_state else None


def fused_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    use_beta_sigmoid_in_kernel: bool,
    lower_bound: float | None,
    transpose_state_layout: bool,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the external fused recurrent KDA operator with AMCT cache state."""
    _validate_fused_options(
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        use_beta_sigmoid_in_kernel,
        transpose_state_layout,
    )
    ranges, packed = _request_ranges(q, cu_seqlens)
    state = _initial_state(q, v, len(ranges), initial_state)
    gate_bias = dt_bias.reshape(q.shape[2], q.shape[-1]).float().contiguous()
    a_log = A_log.detach().float().contiguous()
    outputs = []

    if not packed:
        batch, tokens = q.shape[:2]
        state_indices = torch.arange(
            batch, device=q.device, dtype=torch.int32,
        ).repeat_interleave(tokens)
        output = _recurrent_kda_impl(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            state=state,
            beta=beta.unsqueeze(-1).contiguous(),
            g=g.contiguous(),
            scale=1.0 / math.sqrt(q.shape[-1]),
            A_log=a_log,
            dt_bias=gate_bias,
            lower_bound=lower_bound,
            layout_qkv="BSND",
            ssm_state_indices=state_indices,
            num_accepted_tokens=None,
        )
        return output, state if output_final_state else None

    for request, start, end in ranges:
        if start == end:
            outputs.append(v[:, start:end])
            continue
        state_indices = torch.full(
            (end - start,), request, device=q.device, dtype=torch.int32,
        )
        outputs.append(_recurrent_kda_impl(
            q[:, start:end].contiguous(),
            k[:, start:end].contiguous(),
            v[:, start:end].contiguous(),
            state=state,
            beta=beta[:, start:end].unsqueeze(-1).contiguous(),
            g=g[:, start:end].contiguous(),
            scale=1.0 / math.sqrt(q.shape[-1]),
            A_log=a_log,
            dt_bias=gate_bias,
            lower_bound=lower_bound,
            layout_qkv="BSND",
            ssm_state_indices=state_indices,
            num_accepted_tokens=None,
        ))
    return torch.cat(outputs, dim=1), state if output_final_state else None


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
