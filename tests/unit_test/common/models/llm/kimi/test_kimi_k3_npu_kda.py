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
"""Tests for the Kimi-K3 Torch/NPU KDA fallback."""

import importlib
import importlib.util
from types import SimpleNamespace

import torch
import torch.nn.functional as F

_MODULE_NAME = "amct_pytorch.common.models.llm.kimi.kimi_k3.modeling.npu_kda"


def _load_npu_kda():
    spec = importlib.util.find_spec(_MODULE_NAME)
    assert spec is not None, "Kimi-K3 NPU KDA adapter is missing"
    return importlib.import_module(_MODULE_NAME)


def test_short_convolution_preserves_prefill_and_decode_cache():
    npu_kda = _load_npu_kda()
    conv = npu_kda.KimiShortConvolution(hidden_size=1, kernel_size=3)
    with torch.no_grad():
        conv.weight.copy_(torch.tensor([[[1.0, 2.0, 3.0]]]))

    prefill = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    output, cache = conv(prefill, output_final_state=True)
    expected = F.conv1d(
        prefill.transpose(1, 2), conv.weight, padding=2, groups=1,
    )[:, :, :4].transpose(1, 2)

    torch.testing.assert_close(output, F.silu(expected))
    torch.testing.assert_close(cache, torch.tensor([[[2.0, 3.0, 4.0]]]))

    decoded, cache = conv(
        torch.tensor([[[5.0]]]), cache=cache, output_final_state=True,
    )

    torch.testing.assert_close(decoded, F.silu(torch.tensor([[[26.0]]])))
    torch.testing.assert_close(cache, torch.tensor([[[3.0, 4.0, 5.0]]]))


def test_short_convolution_dispatches_prefill_to_cann(monkeypatch):
    npu_kda = _load_npu_kda()
    conv = npu_kda.KimiShortConvolution(hidden_size=2, kernel_size=2)
    calls = {}

    def fake_prefill(**kwargs):
        calls.update(kwargs)
        kwargs["conv_states"].fill_(7)
        return kwargs["x"] + 1

    monkeypatch.setattr(npu_kda, "_is_npu_tensor", lambda tensor: True)
    monkeypatch.setattr(npu_kda, "_get_cann_op", lambda name: fake_prefill)
    inputs = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)

    output, cache = conv(inputs, output_final_state=True)

    torch.testing.assert_close(output, inputs + 1)
    assert calls["x"].shape == (6, 2)
    assert calls["weight"].shape == (2, 2)
    torch.testing.assert_close(calls["query_start_loc"], torch.tensor([0, 3, 6], dtype=torch.int32))
    torch.testing.assert_close(calls["cache_indices"], torch.tensor([0, 1], dtype=torch.int32))
    torch.testing.assert_close(calls["has_initial_state"], torch.zeros(2, dtype=torch.int32))
    torch.testing.assert_close(cache, torch.full((2, 2, 2), 7.0))


def test_short_convolution_keeps_packed_sequences_independent():
    npu_kda = _load_npu_kda()
    conv = npu_kda.KimiShortConvolution(hidden_size=1, kernel_size=2)
    with torch.no_grad():
        conv.weight.fill_(1)
    inputs = torch.tensor([[[1.0], [2.0], [10.0], [20.0]]])

    output, cache = conv(
        inputs,
        output_final_state=True,
        cu_seqlens=torch.tensor([0, 2, 4], dtype=torch.int32),
    )

    expected = F.silu(torch.tensor([[[1.0], [3.0], [10.0], [30.0]]]))
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(
        cache, torch.tensor([[[1.0, 2.0]], [[10.0, 20.0]]]),
    )


def test_short_convolution_dispatches_decode_to_cann_update(monkeypatch):
    npu_kda = _load_npu_kda()
    conv = npu_kda.KimiShortConvolution(hidden_size=2, kernel_size=2)
    calls = {}

    def fake_update(**kwargs):
        calls.update(kwargs)
        return kwargs["x"]

    monkeypatch.setattr(npu_kda, "_is_npu_tensor", lambda tensor: True)
    monkeypatch.setattr(
        npu_kda,
        "_get_cann_op",
        lambda name: fake_update if name == "causal_conv1d_update" else None,
    )
    inputs = torch.ones(2, 1, 2)
    initial_cache = torch.zeros(2, 2, 2)

    output, cache = conv(
        inputs, cache=initial_cache, output_final_state=True,
    )

    torch.testing.assert_close(output, inputs)
    assert calls["conv_state"] is initial_cache
    torch.testing.assert_close(
        calls["conv_state_indices"], torch.tensor([0, 1], dtype=torch.int32),
    )
    assert cache is initial_cache


def test_recurrent_kda_matches_multi_token_chunk_kda():
    npu_kda = _load_npu_kda()
    torch.manual_seed(7)
    query = torch.randn(1, 5, 2, 3)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    decay = -torch.rand_like(query)
    beta = torch.sigmoid(torch.randn(1, 5, 2))
    cache = torch.randn(1, 2, 3, 3)

    chunk_output, chunk_cache = npu_kda.torch_chunk_kda(
        query, key, value, decay, beta, cache,
    )
    recurrent_output, recurrent_cache = npu_kda.torch_recurrent_kda(
        query, key, value, decay, beta, cache,
    )

    torch.testing.assert_close(recurrent_output, chunk_output, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(recurrent_cache, chunk_cache, rtol=1e-5, atol=1e-5)


def test_recurrent_kda_keeps_packed_sequence_caches_independent():
    npu_kda = _load_npu_kda()
    torch.manual_seed(11)
    query = torch.randn(1, 2, 1, 2)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    decay = -torch.rand_like(query)
    beta = torch.sigmoid(torch.randn(1, 2, 1))
    cache = torch.randn(2, 1, 2, 2)
    cu_seqlens = torch.tensor([0, 1, 2], dtype=torch.int32)

    packed_output, packed_cache = npu_kda.torch_recurrent_kda(
        query, key, value, decay, beta, cache, cu_seqlens,
    )
    expected = [
        npu_kda.torch_recurrent_kda(
            query[:, index:index + 1], key[:, index:index + 1],
            value[:, index:index + 1], decay[:, index:index + 1],
            beta[:, index:index + 1], cache[index:index + 1],
        )
        for index in range(2)
    ]

    torch.testing.assert_close(
        packed_output, torch.cat([item[0] for item in expected], dim=1),
    )
    torch.testing.assert_close(
        packed_cache, torch.cat([item[1] for item in expected], dim=0),
    )


def test_rms_norm_gated_matches_explicit_formula():
    npu_kda = _load_npu_kda()
    norm = SimpleNamespace(
        weight=torch.tensor([2.0, 0.5]),
        variance_epsilon=0.0,
    )
    values = torch.tensor([[[[3.0, 4.0]]]])
    gate = torch.zeros_like(values)

    output = npu_kda.rms_norm_gated(values, gate, norm)

    rms = torch.sqrt(torch.tensor(12.5))
    expected = values / rms * norm.weight * 0.5
    torch.testing.assert_close(output, expected)


def test_prepare_kda_gates_matches_reference_formula():
    npu_kda = _load_npu_kda()
    raw_gate = torch.tensor([[[[0.5, -0.5], [1.0, -1.0]]]])
    raw_beta = torch.tensor([[[0.0, 1.0]]])
    a_log = torch.log(torch.tensor([2.0, 3.0]))
    dt_bias = torch.tensor([0.1, 0.2, 0.3, 0.4])

    decay, beta = npu_kda.prepare_kda_gates(
        raw_gate, raw_beta, a_log, dt_bias, lower_bound=None,
    )

    gate_input = raw_gate + dt_bias.view(1, 1, 2, 2)
    expected_decay = -a_log.exp().view(1, 1, 2, 1) * F.softplus(gate_input)
    torch.testing.assert_close(decay, expected_decay)
    torch.testing.assert_close(beta, raw_beta.sigmoid())
