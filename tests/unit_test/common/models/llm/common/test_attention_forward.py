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

import math

import pytest
import torch
import torch.nn as nn

from amct_pytorch.common.models.llm.common.attention_forward import (
    repeat_kv,
    scaled_dot_product_attention,
)

# ---- repeat_kv -----------------------------------------------------------


def test_repeat_kv_returns_input_when_n_rep_is_one():
    x = torch.randn(2, 4, 8, 16)
    out = repeat_kv(x, n_rep=1)
    assert out is x


def test_repeat_kv_expands_kv_heads_along_dim1():
    bs, kv_heads, seq, head_dim = 2, 3, 5, 4
    x = torch.arange(bs * kv_heads * seq * head_dim, dtype=torch.float32).reshape(
        bs, kv_heads, seq, head_dim
    )
    out = repeat_kv(x, n_rep=2)
    assert out.shape == (bs, kv_heads * 2, seq, head_dim)
    # Each kv head should appear `n_rep` times consecutively.
    for kv in range(kv_heads):
        assert torch.equal(out[:, kv * 2], x[:, kv])
        assert torch.equal(out[:, kv * 2 + 1], x[:, kv])


# ---- scaled_dot_product_attention helpers --------------------------------


class _StdAttnModule(nn.Module):
    """Provides qk_matmul / pv_matmul implementations matching the math
    expected by scaled_dot_product_attention."""

    def qk_matmul(self, q, k):
        return q @ k.transpose(-2, -1)

    def pv_matmul(self, attn, v_t):
        # v was transposed before the call: v_t has shape [..., head_dim, seq].
        # The standard math is attn @ v, so undo the transpose first.
        return attn @ v_t.transpose(-2, -1)


def _reference_sdpa(q, k, v, attn_mask=None, is_causal=False):
    seq_len, s_len = q.size(-2), k.size(-2)
    scale = 1 / math.sqrt(q.size(-1))
    bias = torch.zeros(seq_len, s_len, dtype=q.dtype, device=q.device)
    if is_causal:
        mask = torch.ones(seq_len, s_len, dtype=torch.bool, device=q.device).tril(0)
        bias.masked_fill_(mask.logical_not(), float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            bias = bias + attn_mask
    weights = torch.softmax((q @ k.transpose(-2, -1)) * scale + bias, dim=-1)
    out = weights @ v
    return out.transpose(1, 2).contiguous()


def test_sdpa_matches_reference_on_simple_input():
    torch.manual_seed(0)
    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)
    out = scaled_dot_product_attention(_StdAttnModule(), q, k, v)
    expected = _reference_sdpa(q, k, v)
    assert torch.allclose(out, expected, atol=1e-5)


def test_sdpa_is_causal_masks_upper_triangle():
    torch.manual_seed(0)
    q = torch.randn(1, 1, 4, 8)
    k = q.clone()
    v = q.clone()
    out = scaled_dot_product_attention(_StdAttnModule(), q, k, v, is_causal=True)
    expected = _reference_sdpa(q, k, v, is_causal=True)
    assert torch.allclose(out, expected, atol=1e-5)


def test_sdpa_with_bool_attn_mask_blocks_disallowed_positions():
    torch.manual_seed(0)
    q = torch.randn(1, 1, 2, 4)
    k = torch.randn(1, 1, 2, 4)
    v = torch.randn(1, 1, 2, 4)
    mask = torch.tensor([[True, False], [True, True]])
    out = scaled_dot_product_attention(_StdAttnModule(), q, k, v, attn_mask=mask)
    expected = _reference_sdpa(q, k, v, attn_mask=mask)
    assert torch.allclose(out, expected, atol=1e-5)


def test_sdpa_with_additive_float_mask_added_to_bias():
    torch.manual_seed(0)
    q = torch.randn(1, 1, 2, 4)
    k = torch.randn(1, 1, 2, 4)
    v = torch.randn(1, 1, 2, 4)
    mask = torch.tensor([[0.0, -1e4], [0.0, 0.0]])
    out = scaled_dot_product_attention(_StdAttnModule(), q, k, v, attn_mask=mask)
    expected = _reference_sdpa(q, k, v, attn_mask=mask)
    assert torch.allclose(out, expected, atol=1e-5)


def test_sdpa_repeats_kv_heads_when_module_declares_groups():
    torch.manual_seed(0)
    q = torch.randn(1, 4, 3, 8)            # 4 attention heads
    k = torch.randn(1, 2, 3, 8)            # 2 kv heads
    v = torch.randn(1, 2, 3, 8)
    module = _StdAttnModule()
    module.num_key_value_groups = 2        # repeat each kv head twice -> 4 heads
    out = scaled_dot_product_attention(module, q, k, v)

    expected_k = repeat_kv(k, 2)
    expected_v = repeat_kv(v, 2)
    expected = _reference_sdpa(q, expected_k, expected_v)
    assert torch.allclose(out, expected, atol=1e-5)


def test_sdpa_custom_scale_overrides_default():
    torch.manual_seed(0)
    q = torch.randn(1, 1, 2, 4)
    k = torch.randn(1, 1, 2, 4)
    v = torch.randn(1, 1, 2, 4)

    out_custom = scaled_dot_product_attention(_StdAttnModule(), q, k, v, scale=0.5)
    expected = (q @ k.transpose(-2, -1)) * 0.5
    expected = torch.softmax(expected, dim=-1) @ v
    expected = expected.transpose(1, 2).contiguous()
    assert torch.allclose(out_custom, expected, atol=1e-5)


def test_sdpa_enable_gqa_repeats_kv_along_head_axis():
    torch.manual_seed(0)
    q = torch.randn(1, 4, 3, 8)
    k = torch.randn(1, 2, 3, 8)
    v = torch.randn(1, 2, 3, 8)
    out = scaled_dot_product_attention(_StdAttnModule(), q, k, v, enable_gqa=True)

    expanded_k = k.repeat_interleave(2, dim=-3)
    expanded_v = v.repeat_interleave(2, dim=-3)
    expected = _reference_sdpa(q, expanded_k, expanded_v)
    assert torch.allclose(out, expected, atol=1e-5)
