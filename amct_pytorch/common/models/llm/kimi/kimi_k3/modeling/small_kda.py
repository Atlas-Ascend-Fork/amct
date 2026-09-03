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
"""Pure Torch short-convolution implementation for Kimi-K3 KDA."""

import math
from itertools import pairwise

import torch
import torch.nn.functional as F
from torch import nn


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
