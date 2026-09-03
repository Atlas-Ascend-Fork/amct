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
"""Tests for the Kimi-K3 pure Torch small-operator KDA path."""

import importlib
import importlib.util
from types import SimpleNamespace

import torch
import torch.nn.functional as F

_MODULE_NAME = "amct_pytorch.common.models.llm.kimi.kimi_k3.modeling.small_kda"


def _load_small_kda():
    spec = importlib.util.find_spec(_MODULE_NAME)
    assert spec is not None, "Kimi-K3 small-operator KDA module is missing"
    return importlib.import_module(_MODULE_NAME)


def test_short_convolution_split_run_matches_full_run():
    small_kda = _load_small_kda()
    conv = small_kda.KimiShortConvolution(hidden_size=1, kernel_size=3)
    with torch.no_grad():
        conv.weight.copy_(torch.tensor([[[1.0, 2.0, 3.0]]]))
    inputs = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]]])

    full_output, full_state = conv(inputs, output_final_state=True)
    first_output, first_state = conv(inputs[:, :3], output_final_state=True)
    second_output, second_state = conv(
        inputs[:, 3:], cache=first_state, output_final_state=True,
    )

    torch.testing.assert_close(
        torch.cat((first_output, second_output), dim=1), full_output,
    )
    torch.testing.assert_close(second_state, full_state)
    torch.testing.assert_close(full_state, torch.tensor([[[4.0, 5.0]]]))


def test_short_convolution_keeps_packed_requests_independent():
    small_kda = _load_small_kda()
    conv = small_kda.KimiShortConvolution(hidden_size=1, kernel_size=3)
    with torch.no_grad():
        conv.weight.fill_(1)
    inputs = torch.tensor([[[1.0], [2.0], [10.0], [20.0], [30.0]]])

    output, state = conv(
        inputs,
        output_final_state=True,
        cu_seqlens=torch.tensor([0, 2, 5], dtype=torch.int32),
    )

    expected = F.silu(torch.tensor([[[1.0], [3.0], [10.0], [30.0], [60.0]]]))
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(
        state, torch.tensor([[[1.0, 2.0]], [[20.0, 30.0]]]),
    )


def test_delta_attention_uses_chunk_for_prefill_and_fused_for_decode(monkeypatch):
    modeling = importlib.import_module(
        "amct_pytorch.common.models.llm.kimi.kimi_k3.modeling.modeling_kimi",
    )
    config = SimpleNamespace(
        hidden_size=6,
        rms_norm_eps=1e-6,
        num_hidden_layers=1,
        linear_attn_config={
            "head_dim": 3,
            "num_heads": 2,
            "short_conv_kernel_size": 3,
            "use_full_rank_gate": False,
            "gate_lower_bound": -5.0,
        },
        is_kda_layer=lambda layer_idx: True,
    )
    torch.manual_seed(17)
    attention = modeling.KimiDeltaAttention(config, layer_idx=0).eval().bfloat16()
    hidden_states = torch.randn(1, 3, 6, dtype=torch.bfloat16)
    assert attention.A_log.shape == (2,)

    def fake_chunk_kda(*, v, **kwargs):
        assert kwargs["initial_state"] is None
        state = torch.full((1, 2, 3, 3), 3.0)
        return v, state

    def fake_fused_recurrent_kda(*, v, **kwargs):
        torch.testing.assert_close(
            kwargs["initial_state"], torch.full((1, 2, 3, 3), 3.0),
        )
        state = torch.full((1, 2, 3, 3), 7.0)
        return v, state

    monkeypatch.setattr(modeling, "chunk_kda", fake_chunk_kda)
    monkeypatch.setattr(
        modeling, "fused_recurrent_kda", fake_fused_recurrent_kda,
    )
    cache = modeling.KimiDynamicCache(config)

    with torch.no_grad():
        prefill_output = attention(hidden_states[:, :2], cache_params=cache)
        torch.testing.assert_close(
            cache.recurrent_states[0], torch.full((1, 2, 3, 3), 3.0),
        )
        decode_output = attention(hidden_states[:, 2:], cache_params=cache)

    assert prefill_output.shape == (1, 2, 6)
    assert decode_output.shape == (1, 1, 6)
    assert prefill_output.dtype == torch.bfloat16
    assert decode_output.dtype == torch.bfloat16
    assert cache.conv_states[0] is not None
    torch.testing.assert_close(
        cache.recurrent_states[0], torch.full((1, 2, 3, 3), 7.0),
    )


def test_get_unpad_data_does_not_require_fla_helpers():
    modeling = importlib.import_module(
        "amct_pytorch.common.models.llm.kimi.kimi_k3.modeling.modeling_kimi",
    )
    attention_mask = torch.tensor([[1, 0, 1], [0, 1, 1]])

    indices, cu_seqlens, max_length = modeling.get_unpad_data(attention_mask)

    torch.testing.assert_close(indices, torch.tensor([0, 2, 4, 5]))
    torch.testing.assert_close(
        cu_seqlens, torch.tensor([0, 2, 4], dtype=torch.int32),
    )
    assert max_length == 2


def test_delta_attention_padding_does_not_mix_request_states(monkeypatch):
    modeling = importlib.import_module(
        "amct_pytorch.common.models.llm.kimi.kimi_k3.modeling.modeling_kimi",
    )
    config = SimpleNamespace(
        hidden_size=6,
        rms_norm_eps=1e-6,
        num_hidden_layers=1,
        linear_attn_config={
            "head_dim": 3,
            "num_heads": 2,
            "short_conv_kernel_size": 3,
            "use_full_rank_gate": False,
            "gate_lower_bound": -5.0,
        },
        is_kda_layer=lambda layer_idx: True,
    )
    torch.manual_seed(23)
    attention = modeling.KimiDeltaAttention(config, layer_idx=0).eval()
    hidden_states = torch.randn(2, 3, 6)
    attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    def fake_kda(*, v, cu_seqlens=None, **kwargs):
        requests = 1 if cu_seqlens is None else len(cu_seqlens) - 1
        state = v.new_zeros(requests, 2, 3, 3)
        return v, state

    monkeypatch.setattr(modeling, "chunk_kda", fake_kda)
    monkeypatch.setattr(modeling, "fused_recurrent_kda", fake_kda)

    with torch.no_grad():
        batched_output = attention(hidden_states, attention_mask=attention_mask)
        first_output = attention(hidden_states[:1, :2])
        second_output = attention(hidden_states[1:, :3])

    torch.testing.assert_close(batched_output[:1, :2], first_output)
    torch.testing.assert_close(batched_output[1:, :3], second_output)
    torch.testing.assert_close(batched_output[0, 2], torch.zeros(6))
