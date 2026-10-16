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
import inspect
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch
from torch import nn
from transformers import Qwen3Config, ViTConfig

# Audio kernels are unrelated to these CPU model-semantic tests.
with patch('transformers.utils.is_torchaudio_available', return_value=False):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    from transformers.models.vit.modeling_vit import ViTLayer

from amct_pytorch.classic.deploy_op.local_dynamic import build_local_adapter


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
    layer.self_attn = build_local_adapter('qwen3_attention', layer.self_attn, 'int8')
    layer.mlp = build_local_adapter('qwen3_mlp', layer.mlp, 'int8')
    layer.self_attn.shared_quantize = IdentityQuant()
    layer.mlp.shared_quantize = IdentityQuant()
    return layer


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


def _set_vit_mlp_input(layer, module):
    paths = (('mlp', 'fc1'), ('intermediate', 'dense'))
    for path in paths:
        parent = layer
        try:
            for attr in path[:-1]:
                parent = getattr(parent, attr)
            old = getattr(parent, path[-1])
        except AttributeError:
            continue
        setattr(parent, path[-1], module(old))
        return
    raise AttributeError('cannot resolve ViT MLP input projection')


def _vit_layer_accepts_attention_mask(layer):
    parameters = inspect.signature(type(layer).forward).parameters
    return 'attention_mask' in parameters


def _fuse_vit(layer):
    for role in ('q', 'k', 'v'):
        _set_vit_projection(layer.attention, role, Projection)
    _set_vit_mlp_input(layer, Projection)
    if hasattr(layer.attention, 'q_proj'):
        layer.attention = build_local_adapter('vit_attention', layer.attention, 'int8')
        attn = layer.attention
        layer.mlp = build_local_adapter('vit_mlp', layer.mlp, 'int8')
        mlp = layer.mlp
    else:
        layer.attention.attention = build_local_adapter(
            'vit_attention', layer.attention.attention, 'int8'
        )
        attn = layer.attention.attention
        layer.intermediate = build_local_adapter('vit_mlp', layer.intermediate, 'int8')
        mlp = layer.intermediate
    attn.shared_quantize = IdentityQuant()
    mlp.shared_quantize = IdentityQuant()
    return layer


def _shared_quantizers(layer):
    return [m for m in layer.modules() if isinstance(m, IdentityQuant)]


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
    if _vit_layer_accepts_attention_mask(original):
        expected = original(x, attention_mask=mask)
    else:
        expected = original(x)
    torch.manual_seed(15)
    if _vit_layer_accepts_attention_mask(fused):
        actual = fused(x, attention_mask=mask)
    else:
        actual = fused(x)
    torch.testing.assert_close(actual, expected)
    assert fused.training == original.training
    fused.double()
    assert all(q.marker.dtype == torch.float64 for q in _shared_quantizers(fused))


def test_qwen_all_shared_quantizers_are_registered():
    fused = _fuse_qwen(_qwen('eager').model.layers[0])
    fused.double()
    assert all(q.marker.dtype == torch.float64 for q in _shared_quantizers(fused))
    assert all(q in list(fused.modules()) for q in _shared_quantizers(fused))


def test_qwen_shared_quantizers_are_not_aliased_in_module_tree():
    fused = _fuse_qwen(_qwen('eager').model.layers[0])
    quantizers = list(_shared_quantizers(fused))
    shared_modules = [
        module for module in fused.modules() if isinstance(module, IdentityQuant)
    ]
    assert len(shared_modules) == len(quantizers)


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


def test_dynamic_adapters_are_serializable(tmp_path):
    layer = _qwen('eager').model.layers[0]
    fused = _fuse_qwen(layer)
    path = tmp_path / 'fused-qwen3.pt'

    torch.save(fused, path)
    loaded = torch.load(path, weights_only=False)

    assert isinstance(loaded.self_attn, type(fused.self_attn))
    assert isinstance(loaded.mlp, type(fused.mlp))
    assert isinstance(loaded.self_attn.shared_quantize, IdentityQuant)


def test_dynamic_adapters_load_in_fresh_process(tmp_path):
    layer = _qwen('eager').model.layers[0]
    layer.self_attn = build_local_adapter('qwen3_attention', layer.self_attn, 'int8')
    layer.mlp = build_local_adapter('qwen3_mlp', layer.mlp, 'int8')
    path = tmp_path / 'fused-qwen3-fresh-process.pt'
    torch.save(layer, path)

    child_code = '''
from unittest.mock import patch
with patch('transformers.utils.is_torchaudio_available', return_value=False):
    import torch
    import amct_pytorch.classic.deploy_op.local_dynamic as local_dynamic
    assert not hasattr(local_dynamic, 'FusedQwen3Attention')
    layer = torch.load(r'PATH', weights_only=False)
    assert type(layer.self_attn).__name__ == 'FusedQwen3Attention'
    assert type(layer.mlp).__name__ == 'FusedQwen3MLP'
    assert layer.self_attn._amct_adapter_id == 'qwen3_attention'
    assert layer.mlp._amct_adapter_id == 'qwen3_mlp'
    from amct_pytorch.classic.deploy_op.shared_quantize import SharedDynamicQuantize
    assert isinstance(layer.self_attn.shared_quantize, SharedDynamicQuantize)
    print('FRESH_PROCESS_LOAD_OK')
    '''.replace('PATH', str(path))
    result = subprocess.run(
        [sys.executable, '-c', child_code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'FRESH_PROCESS_LOAD_OK' in result.stdout


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
        layer.self_attn(torch.randn(1, 2, 16))


@pytest.fixture
def cpu_quant_kernels(monkeypatch):
    import torch_npu
    from types import SimpleNamespace

    counts = {'static': 0, 'dynamic': 0}

    def quantize(x, scale, offset, *, dtype, axis, div_mode):
        counts['static'] += 1
        shape = [1] * x.ndim
        shape[axis] = -1
        return (
            (x * scale.reshape(shape) + offset.reshape(shape))
            .round()
            .clamp(-128, 127)
            .to(dtype)
        )

    def dynamic(x, **kwargs):
        counts['dynamic'] += 1
        scale = x.abs().amax(-1).clamp_min(1e-10) / 127
        return (x / scale.unsqueeze(-1)).round().clamp(-128, 127).to(torch.int8), scale

    def matmul(x, weight, *, scale, pertoken_scale, bias, output_dtype):
        result = x.float() @ weight.float()
        if bias is not None and bias.dtype == torch.int32:
            result = result + bias
        result = result * scale
        if pertoken_scale is not None:
            result = result.reshape(-1, result.shape[-1])
            result = result * pertoken_scale.reshape(-1, 1)
            result = result.reshape(x.shape[:-1] + (weight.shape[1],))
        if bias is not None and bias.dtype != torch.int32:
            result = result + bias
        return result.to(output_dtype)

    matmul._schemas = {'default': SimpleNamespace(arguments=[])}
    monkeypatch.setattr(torch_npu, 'npu_quantize', quantize)
    monkeypatch.setattr(torch_npu, 'npu_dynamic_quant', dynamic)
    monkeypatch.setattr(torch_npu, 'npu_quant_matmul', matmul)
    monkeypatch.setattr(torch_npu, 'npu_trans_quant_param', lambda scale: scale)
    return counts


@pytest.mark.parametrize('family', ['qwen3', 'vit'])
@pytest.mark.parametrize('algorithm', ['minmax', 'smoothquant'])
@pytest.mark.parametrize('mode', ['tensor', 'static_token', 'dynamic_token'])
def test_quantize_calibrate_convert_real_model(
    family, algorithm, mode, cpu_quant_kernels
):
    from amct_pytorch.classic.quantize import quantize, convert
    from amct_pytorch.classic.optimizer.layer_norm_quant import LayerNormQuant

    torch.manual_seed(79)
    if family == 'qwen3':
        model = _qwen('eager')
        x = torch.tensor([[1, 2, 3, 4]])
        layer = model.model.layers[0]
        skip = {'lm_head', 'o_proj', 'down_proj'}
    else:
        config = ViTConfig(hidden_size=16, intermediate_size=32, num_attention_heads=4)
        config._attn_implementation = 'eager'
        model = ViTLayer(config).eval()
        layer = model
        x = torch.randn(2, 4, 16, dtype=torch.float16)
        skip = {'fc2', 'o_proj', 'output.dense'}
    original_type = type(layer)
    model.half()
    cfg = {
        'batch_num': 2,
        'quant_cfg': {
            'weights': {'type': 'int8', 'symmetric': True, 'strategy': 'channel'},
            'inputs': {
                'type': 'int8',
                'symmetric': True,
                'strategy': 'tensor' if mode == 'tensor' else 'token',
                'dynamic': (None if mode == 'tensor' else mode == 'dynamic_token'),
            },
        },
        'algorithm': {
            algorithm: {'smooth_strength': 0.5} if algorithm == 'smoothquant' else {}
        },
        'skip_layers': skip,
    }
    quantize(model, cfg)
    # Quantize accepts FP16/BF16 models. Promote this CPU algebra comparison
    # after insertion to isolate scaling correctness from FP16 norm rounding.
    model.float()
    if x.is_floating_point():
        x = x.float()
    wrappers = [m for m in model.modules() if isinstance(m, LayerNormQuant)]
    expected_wrapper_count = (
        0
        if mode == 'static_token'
        else (1 if family == 'vit' and algorithm == 'minmax' else 2)
    )
    assert len(wrappers) == expected_wrapper_count
    with torch.no_grad():
        model(x)
        model(x)
        expected = None if mode != 'tensor' else model(x)
    for wrapper in wrappers:
        assert wrapper.status == 'calibrated'
        for member in wrapper.scaling_layers:
            assert member.scale_w is not None
            assert (member.scale_d is None) == (mode == 'dynamic_token')
    convert(model)
    assert type(layer) is original_type
    assert not any(isinstance(m, LayerNormQuant) for m in model.modules())
    assert not hasattr(model, '_amct_fusion_registry')
    cpu_quant_kernels.update(static=0, dynamic=0)
    with torch.no_grad():
        actual = model(x)
    if family == 'qwen3':
        actual = actual.logits
        if expected is not None:
            expected = expected.logits
    if expected is None:
        assert torch.isfinite(actual).all()
    else:
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    if mode == 'tensor':
        assert cpu_quant_kernels == {'static': 2, 'dynamic': 0}
    else:
        # Issue #139: per-token deployment stays runtime dynamic. Dynamic groups
        # share one call per group; static-token groups are not built.
        expected_dynamic = (
            2 if mode == 'dynamic_token' else (5 if family == 'qwen3' else 4)
        )
        assert cpu_quant_kernels == {'static': 0, 'dynamic': expected_dynamic}


def test_local_vit_adapters_compile_without_graph_breaks():
    config = ViTConfig(hidden_size=16, intermediate_size=32, num_attention_heads=4)
    config._attn_implementation = 'eager'
    model = _fuse_vit(ViTLayer(config).eval())
    x = torch.randn(2, 4, 16)
    with torch.no_grad():
        expected = model(x)
        actual = torch.compile(model, backend='eager', fullgraph=True)(x)
    torch.testing.assert_close(actual, expected)
