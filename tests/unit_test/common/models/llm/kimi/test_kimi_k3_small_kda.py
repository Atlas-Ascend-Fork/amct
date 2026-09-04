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

import builtins
import importlib
import importlib.util
import math
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

_MODULE_NAME = "amct_pytorch.common.models.llm.kimi.kimi_k3.modeling.small_kda"


@pytest.fixture(autouse=True)
def _stub_npu_kda_modules(monkeypatch):
    def unavailable_kernel(*args, **kwargs):
        raise AssertionError("NPU kernel must be replaced by the test")

    ops = ModuleType("ops")
    ops.__path__ = []
    cannbot_dsl = ModuleType("ops.cannbot_dsl")
    cannbot_dsl.__path__ = []
    flash_kda = ModuleType("ops.cannbot_dsl.flash_kda")
    flash_kda.flash_kda = unavailable_kernel
    recurrent_kda = ModuleType("ops.cannbot_dsl.fused_recurrent_kda")
    recurrent_kda.fused_recurrent_kda_op = unavailable_kernel

    monkeypatch.setitem(sys.modules, "ops", ops)
    monkeypatch.setitem(sys.modules, "ops.cannbot_dsl", cannbot_dsl)
    monkeypatch.setitem(sys.modules, "ops.cannbot_dsl.flash_kda", flash_kda)
    monkeypatch.setitem(
        sys.modules,
        "ops.cannbot_dsl.fused_recurrent_kda",
        recurrent_kda,
    )


def _load_small_kda():
    spec = importlib.util.find_spec(_MODULE_NAME)
    assert spec is not None, "Kimi-K3 small-operator KDA module is missing"
    return importlib.import_module(_MODULE_NAME)


def test_npu_kda_dependency_import_error_is_not_suppressed(monkeypatch):
    small_kda = _load_small_kda()
    spec = importlib.util.spec_from_file_location(
        "_small_kda_import_probe", small_kda.__file__,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    original_import = builtins.__import__

    def fail_flash_import(name, *args, **kwargs):
        if name == "ops.cannbot_dsl.flash_kda":
            raise ImportError("flash kernel root cause")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_flash_import)
    with pytest.raises(ImportError, match="flash kernel root cause"):
        spec.loader.exec_module(module)


def test_embedding_emits_bf16_with_legacy_output_dir_argument(monkeypatch):
    from amct_pytorch.common.models.llm.kimi.kimi_k3.kimi_k3 import KimiK3

    pipeline = KimiK3.__new__(KimiK3)
    pipeline.args = SimpleNamespace(device="cpu")
    pipeline.model = SimpleNamespace(
        model=SimpleNamespace(
            embed_tokens=torch.nn.Embedding(4, 2, device="meta"),
            norm=torch.nn.Linear(2, 2, bias=False, device="meta"),
            use_attn_residuals=True,
        ),
        lm_head=torch.nn.Linear(2, 4, bias=False, device="meta"),
    )
    weights = {
        "language_model.model.embed_tokens.": torch.arange(8).reshape(4, 2).bfloat16(),
        "language_model.model.norm.": torch.ones(2, 2, dtype=torch.bfloat16),
        "language_model.lm_head.": torch.ones(4, 2, dtype=torch.bfloat16),
    }
    monkeypatch.setattr(pipeline, "load_layer_weight", lambda prefix: {"weight": weights[prefix]})

    outputs = pipeline.do_embedding_forward([torch.tensor([[1, 3]])], "./outputs")

    torch.testing.assert_close(
        outputs[0], torch.tensor([[[2, 3], [6, 7]]], dtype=torch.bfloat16),
    )
    assert pipeline.model.model.embed_tokens.weight.dtype == torch.bfloat16
    assert pipeline._block_residuals[0].dtype == torch.bfloat16


@pytest.mark.parametrize("mode", ["chunk", "decode_dense", "decode_packed"])
def test_npu_kda_promotes_gate_parameters_to_fp32(monkeypatch, mode):
    small_kda = _load_small_kda()
    calls = []

    def kernel(q, k, v, *, A_log, dt_bias, g, beta, **kwargs):
        assert all(t.dtype == torch.bfloat16 for t in (q, k, v, g, beta))
        assert A_log.dtype == dt_bias.dtype == torch.float32
        assert A_log.is_contiguous() and dt_bias.is_contiguous()
        torch.testing.assert_close(A_log, torch.tensor([1.0, 3.0]))
        torch.testing.assert_close(dt_bias, torch.tensor([[0.0, 2.0], [4.0, 6.0]]))
        state = kwargs.get("initial_state", kwargs.get("state"))
        assert state.dtype == torch.float32
        calls.append(True)
        return (v, state) if mode == "chunk" else v

    monkeypatch.setattr(small_kda, "_flash_kda_impl", kernel)
    monkeypatch.setattr(small_kda, "_recurrent_kda_impl", kernel)
    q = torch.ones(1, 1, 2, 2, dtype=torch.bfloat16)
    kwargs = {
        "q": q, "k": q, "v": q, "g": q,
        "beta": torch.ones(1, 1, 2, dtype=torch.bfloat16),
        "A_log": torch.arange(1, 5, dtype=torch.bfloat16)[::2],
        "dt_bias": torch.arange(8, dtype=torch.bfloat16)[::2],
        "initial_state": None, "output_final_state": True,
        "use_qk_l2norm_in_kernel": True, "use_gate_in_kernel": True,
        "use_beta_sigmoid_in_kernel": True, "lower_bound": -5.0,
        "transpose_state_layout": True,
        "cu_seqlens": torch.tensor([0, 1]) if mode == "decode_packed" else None,
    }
    if mode == "chunk":
        output, _ = small_kda.chunk_kda(**kwargs, safe_gate=True)
    else:
        output, _ = small_kda.fused_recurrent_kda(**kwargs)
    torch.testing.assert_close(output, q)
    assert len(calls) == 1


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


def test_npu_chunk_kda_pads_each_packed_request_and_restores_layout(monkeypatch):
    small_kda = _load_small_kda()
    calls = []

    def fake_flash_kda(
        q, k, v, *, g, beta, scale, initial_state,
        A_log, dt_bias, lower_bound, layout_qkv,
    ):
        valid_tokens = (2, 3)[len(calls)]
        assert q.shape == k.shape == v.shape == g.shape == (1, 64, 1, 2)
        assert beta.shape == (1, 64, 1)
        assert torch.count_nonzero(q[:, valid_tokens:]) == 0
        assert torch.isneginf(g[:, valid_tokens:]).all()
        assert torch.isneginf(beta[:, valid_tokens:]).all()
        assert math.isclose(scale, 2 ** -0.5)
        assert A_log.shape == (1,)
        assert dt_bias.shape == (1, 2)
        assert lower_bound == -5.0
        assert layout_qkv == "BSND"
        calls.append(q.shape[1])
        return v + 10, initial_state + len(calls)

    monkeypatch.setattr(
        small_kda, "_flash_kda_impl", fake_flash_kda, raising=False,
    )
    query = torch.arange(10, dtype=torch.float32).view(1, 5, 1, 2)
    initial_state = torch.zeros(2, 1, 2, 2)

    output, state = small_kda.chunk_kda(
        q=query,
        k=query,
        v=query,
        g=query,
        beta=torch.zeros(1, 5, 1),
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(2),
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
        transpose_state_layout=True,
        cu_seqlens=torch.tensor([0, 2, 5], dtype=torch.int32),
    )

    torch.testing.assert_close(output, query + 10)
    torch.testing.assert_close(state[0], torch.ones(1, 2, 2))
    torch.testing.assert_close(state[1], torch.full((1, 2, 2), 2.0))
    assert calls == [64, 64]


def test_npu_fused_recurrent_kda_updates_existing_state(monkeypatch):
    small_kda = _load_small_kda()

    def fake_recurrent_kda(
        q, k, v, *, state, beta, g, scale, A_log, dt_bias,
        lower_bound, layout_qkv, ssm_state_indices, num_accepted_tokens,
    ):
        assert q.shape == k.shape == v.shape == g.shape == (2, 1, 1, 2)
        assert beta.shape == (2, 1, 1, 1)
        assert math.isclose(scale, 2 ** -0.5)
        assert A_log.shape == (1,)
        assert dt_bias.shape == (1, 2)
        assert lower_bound == -5.0
        assert layout_qkv == "BSND"
        assert num_accepted_tokens is None
        torch.testing.assert_close(
            ssm_state_indices, torch.tensor([0, 1], dtype=torch.int32),
        )
        state.index_add_(
            0, ssm_state_indices, torch.ones(2, 1, 2, 2),
        )
        return v + 20

    monkeypatch.setattr(
        small_kda, "_recurrent_kda_impl", fake_recurrent_kda, raising=False,
    )
    query = torch.arange(4, dtype=torch.float32).view(2, 1, 1, 2)
    initial_state = torch.zeros(2, 1, 2, 2)

    output, state = small_kda.fused_recurrent_kda(
        q=query,
        k=query,
        v=query,
        g=query,
        beta=torch.zeros(2, 1, 1),
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(2),
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=-5.0,
        transpose_state_layout=True,
        cu_seqlens=None,
    )

    torch.testing.assert_close(output, query + 20)
    torch.testing.assert_close(state, torch.ones_like(initial_state))
    assert state.data_ptr() == initial_state.data_ptr()


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
        assert kwargs["beta"].dtype == torch.bfloat16
        state = torch.full((1, 2, 3, 3), 3.0)
        return v, state

    def fake_fused_recurrent_kda(*, v, **kwargs):
        assert kwargs["beta"].dtype == torch.bfloat16
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
