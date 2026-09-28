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
import copy
from unittest.mock import patch

import pytest
import torch
from torch import nn
from transformers import Qwen3Config, ViTConfig

# Audio kernels are unrelated to these CPU model-semantic tests.
with patch('transformers.utils.is_torchaudio_available', return_value=False):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    from transformers.models.vit.modeling_vit import ViTLayer

from amct_pytorch.classic.optimizer.apply_scaling_fusion_pass import (
    _load_qwen3_fused_layer,
    _load_vit_fused_layer,
)


class IdentityQuant(nn.Module):
    # Isolate attention/layer semantics from the NPU quantization kernel.
    def __init__(self):
        super().__init__()
        self.register_buffer('marker', torch.ones(1))

    def forward(self, x):
        return x, torch.ones_like(x[..., 0])


class Projection(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, x, scale=None):
        return self.linear(x)


def _fuse_qwen(layer):
    for name in ('q_proj', 'k_proj', 'v_proj'):
        setattr(layer.self_attn, name, Projection(getattr(layer.self_attn, name)))
    for name in ('up_proj', 'gate_proj'):
        setattr(layer.mlp, name, Projection(getattr(layer.mlp, name)))
    return _load_qwen3_fused_layer().from_shared_map(
        layer,
        {
            'model.layers.0.input_layernorm': IdentityQuant(),
            'model.layers.0.post_attention_layernorm': IdentityQuant(),
        },
        'model.layers.0',
    )


def _set_vit_projection(attn, role, module):
    paths = {
        'q': (('q_proj',), ('attention', 'query')),
        'k': (('k_proj',), ('attention', 'key')),
        'v': (('v_proj',), ('attention', 'value')),
    }[role]
    for path in paths:
        parent = attn
        try:
            for attr in path[:-1]:
                parent = getattr(parent, attr)
            old = getattr(parent, path[-1])
        except AttributeError:
            continue
        setattr(parent, path[-1], module(old))
        return
    raise AttributeError(f'cannot resolve ViT {role} projection')


def _fuse_vit(layer):
    for role in ('q', 'k', 'v'):
        _set_vit_projection(layer.attention, role, Projection)
    layer.mlp.fc1 = Projection(layer.mlp.fc1)
    return _load_vit_fused_layer().from_shared_map(
        layer,
        {
            'layer.layernorm_before': IdentityQuant(),
            'layer.layernorm_after': IdentityQuant(),
        },
        'layer',
    )


def _qwen(backend, sliding=False):
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        layer_types=['sliding_attention' if sliding else 'full_attention'],
        sliding_window=3 if sliding else None,
        use_sliding_window=sliding,
        attention_dropout=0.2,
    )
    config._attn_implementation = backend
    return Qwen3ForCausalLM(config).eval()


@pytest.mark.parametrize('backend', ['eager', 'sdpa'])
@pytest.mark.parametrize('sliding', [False, True])
@pytest.mark.parametrize('padding', [False, True])
def test_qwen_logits_prefill_and_cached_decode_match(backend, sliding, padding):
    torch.manual_seed(11)
    original = _qwen(backend, sliding)
    fused = copy.deepcopy(original)
    fused.model.layers[0] = _fuse_qwen(fused.model.layers[0])
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    mask = torch.tensor([[0, 1, 1, 1, 1]]) if padding else None
    with torch.no_grad():
        expected = original(tokens, attention_mask=mask, use_cache=True)
        actual = fused(tokens, attention_mask=mask, use_cache=True)
        torch.testing.assert_close(actual.logits, expected.logits, atol=1e-6, rtol=1e-5)
        mask = (
            torch.cat([mask, torch.ones(1, 2, dtype=mask.dtype)], -1)
            if padding
            else None
        )
        # Multi-token decode exercises causal alignment in addition to KV reuse.
        expected = original(
            torch.tensor([[6, 7]]),
            attention_mask=mask,
            past_key_values=expected.past_key_values,
            use_cache=True,
        )
        actual = fused(
            torch.tensor([[6, 7]]),
            attention_mask=mask,
            past_key_values=actual.past_key_values,
            use_cache=True,
        )
        torch.testing.assert_close(actual.logits, expected.logits, atol=1e-6, rtol=1e-5)


def test_qwen_positional_arguments_and_dropout_match():
    original = _qwen('eager').train()
    layer = original.model.layers[0]
    fused = _fuse_qwen(copy.deepcopy(layer))
    x = torch.randn(1, 4, 16)
    positions = torch.arange(4).unsqueeze(0)
    rope = original.model.rotary_emb(x, positions)
    mask = torch.full((1, 1, 4, 4), float('-inf')).triu(1)
    torch.manual_seed(13)
    expected = layer(x, mask, positions, None, False, rope)
    torch.manual_seed(13)
    actual = fused(x, mask, positions, None, False, rope)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('training', [False, True])
@pytest.mark.parametrize('backend', ['eager', 'sdpa'])
def test_vit_positional_mask_and_dropout_match(training, backend):
    config = ViTConfig(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        hidden_dropout_prob=0.3,
        attention_probs_dropout_prob=0.2,
    )
    config._attn_implementation = backend
    original = ViTLayer(config).train(training)
    layer = copy.deepcopy(original)
    fused = _fuse_vit(layer)
    x = torch.randn(2, 5, 16)
    mask = torch.zeros(2, 1, 5, 5)
    mask[..., -1] = float('-inf')
    torch.manual_seed(15)
    expected = original(x, mask)
    torch.manual_seed(15)
    actual = fused(x, mask)
    torch.testing.assert_close(actual, expected)
    assert fused.training == original.training
    fused.double()
    assert all(
        q.marker.dtype == torch.float64 for q in fused._shared_by_pre_op.values()
    )


def test_qwen_all_shared_quantizers_are_registered():
    fused = _fuse_qwen(_qwen('eager').model.layers[0])
    fused.double()
    assert all(
        q.marker.dtype == torch.float64 for q in fused._shared_by_pre_op.values()
    )
    assert all(q in list(fused.modules()) for q in fused._shared_by_pre_op.values())


def test_qwen_fused_layer_preserves_layer_type_and_output_capture():
    from transformers.modeling_layers import GradientCheckpointingLayer
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    original = _qwen('eager')
    fused = copy.deepcopy(original)
    fused.model.layers[0] = _fuse_qwen(fused.model.layers[0])
    assert isinstance(fused.model.layers[0], Qwen3DecoderLayer)
    assert isinstance(fused.model.layers[0], GradientCheckpointingLayer)
    with torch.no_grad():
        expected = original(torch.tensor([[1, 2, 3]]), output_hidden_states=True)
        actual = fused(torch.tensor([[1, 2, 3]]), output_hidden_states=True)
    assert len(actual.hidden_states) == len(expected.hidden_states)


def test_vit_fused_layer_preserves_layer_type_and_output_capture():
    config = ViTConfig(hidden_size=16, intermediate_size=32, num_attention_heads=4)
    config._attn_implementation = 'eager'
    original = ViTLayer(config).eval()
    layer = copy.deepcopy(original)
    fused = _fuse_vit(layer)
    assert isinstance(fused, ViTLayer)


def test_qwen_fused_layer_requires_position_embeddings():
    layer = _fuse_qwen(_qwen('eager').model.layers[0])
    with pytest.raises(ValueError, match='position_embeddings'):
        layer(torch.randn(1, 2, 16))
