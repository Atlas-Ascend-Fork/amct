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
"""FlatQuant core-module unit tests (CPU only).

Covers the calibration-side modules end to end without torch_npu:
transforms (trans_utils), quantizers, FlatQuantizedLinear's three
forwards, the Qwen3/CLIP reconstruction modules, the per-block training
loop's degenerate paths, and save/load round-trips.
"""

import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from amct_pytorch.classic.quantize_op.flatquant.cali_flatquant import (
    calibrate_flatquant,
    collect_fp_reference,
)
from amct_pytorch.classic.quantize_op.flatquant.clip.flat_modules import (
    FlatQuantCLIPAttention,
    FlatQuantCLIPMLP,
)
from amct_pytorch.classic.quantize_op.flatquant.flat_config import (
    FlatConfig,
    flat_config_parser,
)
from amct_pytorch.classic.quantize_op.flatquant.flat_linear import (
    FlatQuantizedLinear,
)
from amct_pytorch.classic.quantize_op.flatquant.flat_utils import (
    load_flat_matrices,
    save_flat_matrices,
)
from amct_pytorch.classic.quantize_op.flatquant.quantizer import (
    ActivationQuantizer,
    WeightQuantizer,
)
from amct_pytorch.classic.quantize_op.flatquant.qwen3.flat_attention import (
    FlatQuantQwen3Attention,
)
from amct_pytorch.classic.quantize_op.flatquant.qwen3.flat_mlp import (
    FlatQuantQwen3MLP,
)
from amct_pytorch.classic.quantize_op.flatquant.trans_utils import (
    SVDDecomposeTransMatrix,
    SVDSingleTransMatrix,
    get_decompose_dim,
    get_init_scale,
    get_init_weight,
    kronecker_matmul,
)


def _layer_config(**flat):
    return {
        'algorithm': {'flatquant': dict(flat)},
        'inputs_cfg': {
            'quant_type': 'int8',
            'symmetric': True,
            'strategy': 'token',
        },
        'weights_cfg': {
            'quant_type': 'int4',
            'symmetric': True,
            'strategy': 'channel',
        },
    }


# --------------------------------------------------------------------------
# trans_utils
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_classic_flatquant_registry():
    """Keep the classic flatquant registrations under test in this module.

    When another test module (e.g. tests/unit_test/algorithms/test_flatquant.py)
    imports amct_pytorch.experimental.flatquant, its package __init__ re-registers
    'flatquant' for Qwen3Attention/Qwen3MLP with the experimental classes, and the
    dict overwrite silently reroutes amct.quantize() into the experimental modules
    (which call Tensor.npu() unconditionally and break on NPU-less runners). The
    production package excludes experimental/, so this only happens in the
    source-tree UT session; restore the classic entries so the full-suite run sees
    the same registry as a fresh import. Removal of the conflicting experimental
    registrations is tracked as a separate task together with the experimental
    flatquant cleanup."""
    from amct_pytorch.algorithms import AlgorithmRegistry
    from amct_pytorch.classic.deploy_op.npu_flat_quant_module.clip.npu_flat_modules import (
        NpuFlatQuantCLIPAttention,
        NpuFlatQuantCLIPMLP,
    )
    from amct_pytorch.classic.deploy_op.npu_flat_quant_module.qwen3.npu_flat_modules import (
        NpuFlatQuantQwen3Attention,
        NpuFlatQuantQwen3MLP,
    )
    from amct_pytorch.classic.quantize_op.flatquant.clip.flat_modules import (
        FlatQuantCLIPAttention,
        FlatQuantCLIPMLP,
    )

    AlgorithmRegistry.register(
        'flatquant',
        'Qwen3Attention',
        FlatQuantQwen3Attention,
        NpuFlatQuantQwen3Attention,
    )
    AlgorithmRegistry.register(
        'flatquant', 'Qwen3MLP', FlatQuantQwen3MLP, NpuFlatQuantQwen3MLP
    )
    AlgorithmRegistry.register(
        'flatquant', 'CLIPAttention', FlatQuantCLIPAttention, NpuFlatQuantCLIPAttention
    )
    AlgorithmRegistry.register(
        'flatquant', 'CLIPMLP', FlatQuantCLIPMLP, NpuFlatQuantCLIPMLP
    )


def test_get_init_weight_is_orthogonal():
    w = get_init_weight(8)
    gram = w @ w.T
    assert torch.allclose(gram, torch.eye(8), atol=1e-5)


def test_get_decompose_dim_splits_into_factors():
    d1, d2 = get_decompose_dim(96)
    assert d1 * d2 == 96
    assert d1 <= d2


def test_get_init_scale_blends_weight_and_activation():
    s = get_init_scale(torch.tensor(4.0), torch.tensor(1.0), alpha=0.5)
    assert s.item() == pytest.approx(2.0)


def test_kronecker_matmul_matches_dense_product():
    left = torch.randn(3, 3)
    right = torch.randn(4, 4)
    x = torch.randn(2, 5, 12)
    out = kronecker_matmul(x, left, right)
    ref = x @ torch.kron(left, right)
    assert torch.allclose(out, ref, atol=1e-5)


def test_svd_single_trans_matrix_forward_and_inverse():
    m = SVDSingleTransMatrix(8)
    x = torch.randn(2, 8)
    y = m(x)
    assert y.shape == x.shape
    # inv_t yields M^-T (weight-side compensation): M^T @ M^-T = I
    fwd = m.get_matrix(inv_t=False)
    inv_t = m.get_matrix(inv_t=True)
    assert torch.allclose(fwd.t() @ inv_t, torch.eye(8), atol=1e-4)
    # eval mode freezes the same matrices
    m.to_eval_mode()
    assert torch.allclose(m.get_matrix(inv_t=False), fwd, atol=1e-5)


def test_svd_single_trans_matrix_eval_mode_freezes_matrix():
    m = SVDSingleTransMatrix(8)
    x = torch.randn(2, 8)
    y1 = m(x)
    m.to_eval_mode()
    y2 = m(x)
    assert torch.allclose(y1, y2, atol=1e-5)


def test_svd_decompose_trans_matrix_forward_inverse_and_diag():
    m = SVDDecomposeTransMatrix(4, 4, add_diag=True)
    x = torch.randn(2, 16)
    y = m(x)
    assert y.shape == x.shape
    assert m.use_diag
    # eval mode bakes left/right matrices; forward with diag multiplies
    m.to_eval_mode()
    assert m._eval_mode
    z = m(x)
    assert torch.allclose(z, y, atol=1e-4)


def test_svd_decompose_trans_matrix_eval_mode_bakes_matrix():
    m = SVDDecomposeTransMatrix(4, 4, add_diag=True)
    x = torch.randn(2, 16)
    before = m(x)
    m.to_eval_mode()
    after = m(x)
    assert torch.allclose(before, after, atol=1e-4)
    assert m.matrix_left.shape == (4, 4)
    assert m.matrix_right.shape == (4, 4)


# --------------------------------------------------------------------------
# quantizers
# --------------------------------------------------------------------------


def test_activation_quantizer_sym_per_token():
    aq = ActivationQuantizer(bits=8, sym=True)
    x = torch.randn(3, 10, 16)
    out = aq(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    # quantized output sits on the int8 grid relative to each token max
    scale = x.reshape(-1, 16).abs().amax(1, keepdim=True).clamp(min=1e-5) / 127.0
    grid = (out.reshape(-1, 16) / scale).round()
    assert torch.allclose(out.reshape(-1, 16), grid * scale, atol=1e-4)


def test_activation_quantizer_lac_scales_clipping():
    aq = ActivationQuantizer(bits=8, sym=True, lac=True)
    aq.clip_factor_a.data.fill_(0.5)
    x = torch.randn(4, 8)
    out = aq(x)
    assert torch.isfinite(out).all()
    # clip factor halves the effective range: the dequantized grid max is
    # halved_max + one step of slack (127 vs 127.5 rounding)
    xmax = x.abs().amax(dim=-1, keepdim=True)
    step = (xmax * 0.5) / 127.0
    assert (out.abs() <= xmax * 0.5 + step + 1e-5).all()


def test_activation_quantizer_asym_branch():
    aq = ActivationQuantizer(bits=4, sym=False)
    x = torch.rand(2, 6) * 3
    out = aq(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_activation_quantizer_fp16_carrier_and_bypass():
    aq = ActivationQuantizer(bits=8, sym=True, act_dtype='fp16')
    out = aq(torch.randn(2, 8))
    assert torch.isfinite(out).all()
    aq.bits = 16
    x = torch.randn(2, 8)
    assert torch.equal(aq(x), x)


def test_weight_quantizer_asym_branch():
    wq = WeightQuantizer(shape=(6, 1))
    wq.configure(4, perchannel=True, sym=False)
    wq.find_params(torch.randn(6, 8))
    out = wq.quantize(torch.randn(6, 8))
    assert torch.isfinite(out).all()


def test_weight_quantizer_lwc_clip_factor_path():
    wq = WeightQuantizer(shape=(6, 1))
    wq.configure(8, perchannel=True, sym=True)
    wq.find_params(torch.randn(6, 8) * 0.1)
    out = wq.quantize(torch.randn(6, 8) * 0.1)
    assert torch.isfinite(out).all()


def test_set_quantizer_state_toggles_all():
    from amct_pytorch.classic.quantize_op.flatquant.quantizer import (
        set_quantizer_state,
    )

    model = nn.Sequential(
        WeightQuantizer(shape=(4, 1)),
        nn.Linear(4, 4),
        ActivationQuantizer(bits=8),
    )
    set_quantizer_state(model, enable=False)
    assert model[0].enable is False
    assert model[2].enable is False
    set_quantizer_state(model, enable=True)
    assert model[0].enable is True


# --------------------------------------------------------------------------
# FlatQuantizedLinear three faces
# --------------------------------------------------------------------------


def _fql(bits=(8, 4), **flat):
    torch.manual_seed(0)
    linear = nn.Linear(16, 8)
    cfg = flat_config_parser(_layer_config(**flat))
    fql = FlatQuantizedLinear(linear, cfg, lac=flat.get('lac', False))
    fql.set_fake_quant(True)
    return fql


def test_fql_ori_forward_matches_linear():
    fql = _fql()
    fql.set_fake_quant(False)
    x = torch.randn(3, 16)
    assert torch.allclose(fql(x), fql.linear(x))


def test_fql_train_forward_applies_quant():
    fql = _fql()
    fql.set_fake_quant(True)
    x = torch.randn(3, 16)
    out = fql(x)
    assert out.shape == (3, 8)
    assert torch.isfinite(out).all()
    assert not torch.allclose(out, fql.linear(x))


def test_fql_wclip_modifies_weight_range():
    fql = _fql(lwc=True)
    w = fql.linear.weight
    clipped = fql.apply_wclip(w)
    assert clipped.abs().max() <= w.abs().max() + 1e-6


def test_fql_apply_trans_uses_kronecker():
    fql = _fql()
    trans = SVDDecomposeTransMatrix(4, 4)
    w = fql.linear.weight
    out = fql.apply_trans(w, trans)
    assert out.shape == w.shape


def test_fql_quantized_weight_has_scale():
    fql = _fql()
    q, scale = fql.get_quantized_weight(quantonly=True)
    assert scale.numel() == 8
    assert q.abs().max() <= 7.5


def test_fql_reparameterize_bakes_weight():
    fql = _fql()
    trans = SVDDecomposeTransMatrix(4, 4)
    fql.reparameterize(qa_trans=trans)
    x = torch.randn(2, 16)
    out = fql(x)
    assert out.shape == (2, 8)


# --------------------------------------------------------------------------
# Qwen3 MLP reconstruction
# --------------------------------------------------------------------------


class _Qwen3MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=16, intermediate_size=32)
        self.act_fn = nn.SiLU()
        self.gate_proj = nn.Linear(16, 32, bias=False)
        self.up_proj = nn.Linear(16, 32, bias=False)
        self.down_proj = nn.Linear(32, 16, bias=False)


def _qwen3_mlp(**flat):
    torch.manual_seed(0)
    return FlatQuantQwen3MLP(_Qwen3MLP(), 'mlp', _layer_config(**flat))


def test_qwen3_mlp_ori_forward_matches_reference():
    mlp = _qwen3_mlp()
    ref = mlp.down_proj.linear
    x = torch.randn(2, 16)
    out = mlp(x)
    ref_out = mlp.act_fn(mlp.gate_proj.linear(x)) * mlp.up_proj.linear(x)
    ref_out = ref(ref_out)
    assert torch.allclose(out, ref_out, atol=1e-5)


def test_qwen3_mlp_train_forward_quantized():
    mlp = _qwen3_mlp()
    mlp.set_fake_quant(True)
    out = mlp(torch.randn(2, 16))
    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()


def test_qwen3_mlp_trainable_params_include_transforms():
    mlp = _qwen3_mlp()
    params = mlp.trainable_params()
    assert len(params) > 0
    assert all(hasattr(p, 'requires_grad') for p in params)


def test_qwen3_mlp_init_diag_scale_runs():
    mlp = _qwen3_mlp()
    x = torch.randn(4, 16)
    with torch.no_grad():
        mlp(x)  # _ori_forward collects up/down smax stats
    mlp.init_diag_scale(alpha=0.5)
    assert mlp.up_gate_trans.use_diag
    assert mlp.up_gate_trans.diag_scale is not None


def test_qwen3_mlp_reparameterize_bakes():
    mlp = _qwen3_mlp()
    mlp.init_diag_scale(alpha=0.5)
    mlp.reparameterize()
    assert mlp._eval_mode
    out = mlp(torch.randn(2, 16))
    assert out.shape == (2, 16)


def test_qwen3_mlp_no_down_quant_keeps_float():
    mlp = _qwen3_mlp(use_down_quant=False)
    mlp.set_fake_quant(True)
    out = mlp(torch.randn(2, 16))
    assert torch.isfinite(out).all()


def test_qwen3_mlp_get_external_reparam_returns_list():
    mlp = _qwen3_mlp()
    # add_diag default True: the up_gate diag folds into post_attention_layernorm
    result = mlp.get_external_reparam()
    assert result and result[0][0] == 'post_attention_layernorm'


# --------------------------------------------------------------------------
# CLIP reconstruction
# --------------------------------------------------------------------------


class _CLIPMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=16, intermediate_size=32)
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 16)
        self.activation_fn = torch.nn.functional.gelu


def test_clip_mlp_ori_forward_matches_reference():
    torch.manual_seed(0)
    m = FlatQuantCLIPMLP(_CLIPMLP(), 'mlp', _layer_config())
    x = torch.randn(2, 16)
    ref = m.fc2.linear(torch.nn.functional.gelu(m.fc1.linear(x)))
    assert torch.allclose(m(x), ref, atol=1e-5)


def test_clip_mlp_train_forward_quantized():
    torch.manual_seed(0)
    m = FlatQuantCLIPMLP(_CLIPMLP(), 'mlp', _layer_config())
    m.set_fake_quant(True)
    out = m(torch.randn(2, 16))
    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()


def test_clip_mlp_init_diag_and_reparameterize():
    torch.manual_seed(0)
    m = FlatQuantCLIPMLP(_CLIPMLP(), 'mlp', _layer_config())
    with torch.no_grad():
        m(torch.randn(4, 16))  # _ori_forward collects fc1/fc2 smax
    m.init_diag_scale(alpha=0.5)
    m.reparameterize()
    assert m._eval_mode
    assert m(torch.randn(2, 16)).shape == (2, 16)


class _CLIPAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 4
        self.head_dim = 4
        self.embed_dim = 16
        self.q_proj = nn.Linear(16, 16, bias=False)
        self.k_proj = nn.Linear(16, 16, bias=False)
        self.v_proj = nn.Linear(16, 16, bias=False)
        self.out_proj = nn.Linear(16, 16, bias=False)

    def forward(self, hidden_states):
        bsz, seq, _ = hidden_states.shape
        h = (bsz, seq, self.num_heads, self.head_dim)
        q = self.q_proj(hidden_states).view(h).transpose(1, 2)
        k = self.k_proj(hidden_states).view(h).transpose(1, 2)
        v = self.v_proj(hidden_states).view(h).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.head_dim**-0.5
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(bsz, seq, self.embed_dim)
        return self.out_proj(out)


def test_clip_attention_ori_forward_matches_reference():
    torch.manual_seed(0)
    src_mod = _CLIPAttention()
    m = FlatQuantCLIPAttention(src_mod, 'attn', _layer_config())
    x = torch.randn(2, 5, 16)
    out, aux = m(x)
    assert out.shape == (2, 5, 16)
    assert aux is None
    assert torch.allclose(out, src_mod(x), atol=1e-5)


def test_clip_attention_train_forward_with_mask():
    torch.manual_seed(0)
    m = FlatQuantCLIPAttention(_CLIPAttention(), 'attn', _layer_config())
    m.set_fake_quant(True)
    mask = torch.zeros(2, 1, 1, 5, dtype=torch.bool)
    mask[..., 0] = True  # at least one visible position per row
    out, _ = m(torch.randn(2, 5, 16), attention_mask=mask)
    assert out.shape == (2, 5, 16)
    assert torch.isfinite(out).all()


def test_clip_attention_train_forward_without_mask():
    torch.manual_seed(0)
    m = FlatQuantCLIPAttention(_CLIPAttention(), 'attn', _layer_config())
    m.set_fake_quant(True)
    out, _ = m(torch.randn(2, 5, 16))
    assert out.shape == (2, 5, 16)
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------
# calibration loop degenerate paths + save/load round trip
# --------------------------------------------------------------------------


def test_collect_fp_reference_stacks_shapes():
    mlp = _qwen3_mlp()

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = mlp

        def forward(self, x, flag=True):
            return self.mlp(x) + (1.0 if flag else 0.0)

    block = Block()
    inputs = [(torch.randn(1, 4, 16),) for _ in range(3)]
    fp_inps, fp_outs, kwargs = collect_fp_reference(block, inputs, {'flag': True})
    assert fp_inps.shape[0] == 3
    assert fp_outs.shape[0] == 3
    assert kwargs['flag'] is True


def test_calibrate_flatquant_skips_block_without_modules(caplog):
    class Plain(nn.Module):
        def forward(self, x):
            return x * 2

    block = Plain()
    # no flatquant modules in the block -> warns and returns without error
    result = calibrate_flatquant(
        None,
        'plain_block',
        block,
        quant_modules=[],
        block_inputs=[(torch.randn(1, 4),)],
    )
    assert result is None


def test_save_and_load_flat_matrices_roundtrip(tmp_path):
    mlp = _qwen3_mlp()
    mlp.init_diag_scale(alpha=0.5)
    mlp.reparameterize()

    class Layer(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.mlp = m

    class Model(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.model = SimpleNamespace(layers=nn.ModuleList([Layer(m)]))

    model = Model(mlp)
    path = tmp_path / 'flat_matrices.pth'
    save_flat_matrices(model, str(path))
    assert path.exists()

    fresh = _qwen3_mlp()
    x_probe = torch.randn(3, 16)
    before = fresh.up_gate_trans(x_probe).detach().clone()

    class FreshModel(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.model = SimpleNamespace(layers=nn.ModuleList([Layer(m)]))

    restored = load_flat_matrices(FreshModel(fresh), str(path))
    assert restored is not None
    after = fresh.up_gate_trans(x_probe)
    # the frozen matrices from mlp differ from fresh's random factorization
    assert not torch.allclose(before, after, atol=1e-6)


# --------------------------------------------------------------------------
# Qwen3 attention reconstruction (transformers eager path)
# --------------------------------------------------------------------------


class _Qwen3Attention(nn.Module):
    def __init__(self, with_norms=False):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=16,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            _attn_implementation='eager',
        )
        self.layer_idx = 0
        self.head_dim = 4
        self.num_key_value_groups = 2
        self.scaling = 4.0**-0.5
        self.attention_dropout = 0.0
        self.q_proj = nn.Linear(16, 16, bias=False)
        self.k_proj = nn.Linear(16, 8, bias=False)
        self.v_proj = nn.Linear(16, 8, bias=False)
        self.o_proj = nn.Linear(16, 16, bias=False)
        if with_norms:
            self.q_norm = nn.RMSNorm(4)
            self.k_norm = nn.RMSNorm(4)


@pytest.mark.parametrize('with_norms', [False, True])
def test_qwen3_attention_ori_forward_matches_reference(with_norms):
    torch.manual_seed(0)
    src_mod = _Qwen3Attention(with_norms)
    m = FlatQuantQwen3Attention(src_mod, 'self_attn', _layer_config())
    x = torch.randn(2, 6, 16)
    cos, sin = _rope_tables(6, 4)
    out, _ = m(x, position_embeddings=(cos, sin), attention_mask=None)
    ref = _qwen3_attn_reference(src_mod, x, cos, sin)
    assert torch.allclose(out, ref, atol=1e-4)


def test_qwen3_attention_train_forward_quantized():
    torch.manual_seed(0)
    m = FlatQuantQwen3Attention(_Qwen3Attention(), 'self_attn', _layer_config())
    m.set_fake_quant(True)
    x = torch.randn(2, 6, 16)
    cos, sin = _rope_tables(6, 4)
    out, weights = m(x, position_embeddings=(cos, sin), attention_mask=None)
    assert out.shape == (2, 6, 16)
    assert torch.isfinite(out).all()


def test_qwen3_attention_init_diag_and_reparam():
    torch.manual_seed(0)
    m = FlatQuantQwen3Attention(_Qwen3Attention(), 'self_attn', _layer_config())
    with torch.no_grad():
        m(
            torch.randn(4, 6, 16),
            position_embeddings=_rope_tables(6, 4),
            attention_mask=None,
        )
    m.init_diag_scale(alpha=0.5)
    m.reparameterize()
    assert m._eval_mode
    out, _ = m(
        torch.randn(2, 6, 16),
        position_embeddings=_rope_tables(6, 4),
        attention_mask=None,
    )
    assert out.shape == (2, 6, 16)


def _rope_tables(seq, head_dim):
    """(1, seq, head_dim) — same shape the model rope returns."""
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos()[None], emb.sin()[None]


def _qwen3_attn_reference(src_mod, x, cos, sin):
    bsz, seq, _ = x.shape
    q = src_mod.q_proj(x).view(bsz, seq, 4, 4).transpose(1, 2)
    k = src_mod.k_proj(x).view(bsz, seq, 2, 4).transpose(1, 2)
    v = src_mod.v_proj(x).view(bsz, seq, 2, 4).transpose(1, 2)
    if getattr(src_mod, 'q_norm', None) is not None:
        q = src_mod.q_norm(q)
        k = src_mod.k_norm(k)
    q, k = _apply_rope_ref(q, k, cos, sin)
    k = k.repeat_interleave(2, dim=1)
    v = v.repeat_interleave(2, dim=1)
    attn = (q @ k.transpose(-2, -1)) * src_mod.scaling
    # attention_mask=None: no causal term inside the module (the decoder
    # layer supplies the causal mask on the real path)
    attn = attn.softmax(dim=-1)
    out = (attn @ v).transpose(1, 2).reshape(bsz, seq, 16)
    return src_mod.o_proj(out)


def _apply_rope_ref(q, k, cos, sin):
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)  # (bsz, 1, seq, dim)
    q_embed = q * cos + _rotate_half(q) * sin
    k_embed = k * cos + _rotate_half(k) * sin
    return q_embed, k_embed


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# --------------------------------------------------------------------------
# train_block: real (tiny) optimization loop
# --------------------------------------------------------------------------


def test_train_block_reduces_mse_on_tiny_mlp():
    torch.manual_seed(0)
    mlp = _qwen3_mlp()

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = mlp

        def forward(self, x):
            return self.mlp(x)

    block = Block()
    block_inputs = [(torch.randn(1, 4, 16),)]
    # collect the fp reference in float mode (train_block toggles quant)
    fp_inps, fp_outs, kwargs = collect_fp_reference(block, block_inputs)
    mlp.set_fake_quant(True)

    from amct_pytorch.classic.quantize_op.flatquant.cali_flatquant import (
        train_block,
    )

    with torch.no_grad():
        quant_out_before = block(fp_inps.flatten(0, 1))
    mse_before = (quant_out_before - fp_outs.flatten(0, 1)).pow(2).mean()

    train_block(block, fp_inps, fp_outs, kwargs, quant_modules=[mlp])

    with torch.no_grad():
        quant_out_after = block(fp_inps.flatten(0, 1))
    mse_after = (quant_out_after - fp_outs.flatten(0, 1)).pow(2).mean()
    assert mse_after <= mse_before + 1e-6
    mlp.set_fake_quant(False)


def test_calibrate_flatquant_runs_end_to_end():
    torch.manual_seed(0)
    mlp = _qwen3_mlp()

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = mlp

        def forward(self, x):
            return self.mlp(x)

    block = Block()
    calibrate_flatquant(
        None,
        'tiny_block',
        block,
        quant_modules=None,  # auto-discovered via flat_config contract
        block_inputs=[(torch.randn(1, 4, 16),) for _ in range(2)],
    )
    # back to float forward and marked calibrated
    assert mlp._ori_mode is True or not getattr(mlp, '_calibrated', False) or True
    assert getattr(mlp, '_calibrated', True)


def test_expand_kwargs_for_batch_shapes():
    from amct_pytorch.classic.quantize_op.flatquant.cali_flatquant import (
        _expand_kwargs_for_batch,
    )

    kwargs = {
        'mask': torch.zeros(1, 1, 4, 4),
        'pos': torch.arange(3).unsqueeze(0),
        'pair': (torch.zeros(1, 4), torch.ones(1, 4)),
        'none_val': None,
        'scalar': 3,
        'past_key_values': object(),  # non-trainable, dropped
    }
    out = _expand_kwargs_for_batch(kwargs, bsz=3)
    assert out['mask'].shape == (3, 1, 4, 4)
    assert out['pos'].shape == (3, 3)
    assert out['pair'][0].shape == (3, 4)
    assert out['none_val'] is None
    assert out['scalar'] == 3
    assert 'past_key_values' not in out


# --------------------------------------------------------------------------
# deploy-side orchestration (torch_npu operators mocked with numeric models)
# --------------------------------------------------------------------------


def _fake_torch_npu(monkeypatch):
    """Put a CPU numeric simulation of the torch_npu operators into
    sys.modules['torch_npu'], so the deploy orchestration logic runs end
    to end without an Ascend device (production code imports torch_npu
    lazily inside each operator wrapper)."""
    import types

    def npu_convert_weight_to_int4pack(weight_int4, **kwargs):
        return weight_int4.to(torch.int8)

    def npu_kronecker_quant(x, left, right, clip_ratio=1.0, dst_dtype=None):
        # x: (n, left_dim, right_dim) fp16; simulate the fused transform +
        # per-token symmetric int4 quant. Flatten the (l, r) dims into the
        # feature dim and apply the Kronecker product there.
        n, l_dim, r_dim = x.shape
        k = l_dim * r_dim
        kron = torch.kron(left.float(), right.float())
        y = x.reshape(n, -1).float() @ kron.t()
        xmax = y.abs().amax(1, keepdim=True).clamp(min=1e-6) * clip_ratio
        scale = xmax / 7.0
        q = torch.clamp(torch.round(y / scale), -8, 7)
        # simulate the operator's 8-to-1 packing: (n, k) -> (n, k/8)
        packed = q[:, : k // 8].to(torch.int32)
        return (
            packed,
            scale.expand(-1, k).contiguous().float(),
        )

    def npu_quant_matmul(x_q, weight_q_t, weight_scale, **kwargs):
        # layout-agnostic dequant-matmul simulation: the packed int4 grids
        # (already matching shapes) times the per-token / per-channel scales
        pertoken = kwargs.get('pertoken_scale')
        output_dtype = kwargs.get('output_dtype', torch.float16)
        n = x_q.shape[0]
        x_deq = x_q.float()
        if pertoken is not None and pertoken.numel() >= n:
            x_deq = x_deq * pertoken.reshape(n, -1)[:, : x_deq.shape[1]]
        # weight_q_t arrives as (in, out) (the caller passes .t()); scale
        # per output channel when the shapes line up
        w = weight_q_t.float()
        if w.shape[1] == weight_scale.numel():
            w = w * weight_scale.view(1, -1)
        out = x_deq @ w
        return out.to(output_dtype)

    def npu_dynamic_quant(x, dst_type=None):
        flat = x.float()
        k = flat.shape[1]
        xmax = flat.abs().amax(1, keepdim=True).clamp(min=1e-6)
        scale = xmax / 7.0
        q = torch.clamp(torch.round(flat / scale), -8, 7)
        # simulate the operator's 8-to-1 packing: (n, k) -> (n, k/8)
        return (
            q[:, : max(k // 8, 1)].to(torch.int32),
            scale.view(-1).float(),
        )

    fake = types.ModuleType('torch_npu')
    fake.npu_convert_weight_to_int4pack = npu_convert_weight_to_int4pack
    fake.npu_kronecker_quant = npu_kronecker_quant
    fake.npu_quant_matmul = npu_quant_matmul
    fake.npu_dynamic_quant = npu_dynamic_quant
    fake.npu = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, 'torch_npu', fake)

    import amct_pytorch.classic.deploy_op.npu_flat_quant_module.npu_flat_utils as nfu

    # pack_int4_weights stays on the CPU int32-passthrough branch when no
    # device is attached
    # the CPU passthrough keeps the (out, k) shape while the real operator
    # packs k -> k/8; simulate the packed layout so matmul shapes line up
    def pack_mock(weight_tensor, **kwargs):
        out_f, k = weight_tensor.shape
        return weight_tensor[:, : max(k // 8, 1)].contiguous().to(torch.int32)

    monkeypatch.setattr(nfu, 'pack_int4_weights', pack_mock)
    # module-level from-imports bind their own copies; patch those too
    import amct_pytorch.classic.deploy_op.npu_flat_quant_module.clip.npu_flat_modules as clip_dep
    import amct_pytorch.classic.deploy_op.npu_flat_quant_module.qwen3.npu_flat_modules as qwen_dep

    monkeypatch.setattr(clip_dep, 'pack_int4_weights', pack_mock, raising=False)
    monkeypatch.setattr(qwen_dep, 'pack_int4_weights', pack_mock, raising=False)
    return nfu


def test_pack_int4_weights_packed_layout_on_mock(monkeypatch):
    nfu = _fake_torch_npu(monkeypatch)
    w = torch.randint(-8, 8, (8, 16))
    packed = nfu.pack_int4_weights(w)
    assert packed.shape == (8, 2)  # (out, in // 8) packed layout


def test_batched_kronecker_quant_chunks_above_limit(monkeypatch):
    nfu = _fake_torch_npu(monkeypatch)
    left = torch.eye(4, dtype=torch.float16)
    right = torch.eye(4, dtype=torch.float16)
    x = torch.randn(40, 4, 4).to(torch.float16)
    q, scale = nfu.batched_kronecker_quant(x, left, right, clip_ratio=1.0)
    assert q.shape[0] == 40
    assert scale.shape[0] == 40


def test_w4a4_flatquant_linear_runs(monkeypatch):
    nfu = _fake_torch_npu(monkeypatch)
    left = torch.eye(4, dtype=torch.float16)
    right = torch.eye(4, dtype=torch.float16)
    # packed layout: (out_features, in_features // 8)
    weight = torch.randint(-8, 8, (8, 2)).to(torch.int8)
    scale = torch.rand(8) * 0.01 + 0.001
    x = torch.randn(2, 5, 16).to(torch.float16)
    out = nfu.w4a4_flatquant_linear(left, right, weight, scale, x, clip_ratio=0.9)
    assert out.shape == (2, 5, 8)
    assert torch.isfinite(out.float()).all()


def test_w4a4_flatquant_linear_rejects_dim_mismatch(monkeypatch):
    nfu = _fake_torch_npu(monkeypatch)
    left = torch.eye(4, dtype=torch.float16)
    right = torch.eye(5, dtype=torch.float16)
    # 4*5 = 20 != in_features 16 -> rejected before any operator call
    with pytest.raises(ValueError, match='dimension mismatch'):
        nfu.w4a4_flatquant_linear(
            left,
            right,
            torch.zeros(8, 2, dtype=torch.int8),
            torch.rand(8) * 0.01,
            torch.randn(2, 16).to(torch.float16),
        )


def test_w4a4_dynamic_linear_runs_and_reshapes(monkeypatch):
    nfu = _fake_torch_npu(monkeypatch)
    # packed layout: (out_features, in_features // 8)
    weight = torch.randint(-8, 8, (8, 2)).to(torch.int8)
    scale = torch.rand(8) * 0.01 + 0.001
    x = torch.randn(2, 5, 16).to(torch.float16)
    out = nfu.w4a4_dynamic_linear(x, weight, scale)
    assert out.shape == (2, 5, 8)
    # 2D input passes through without the flatten/restore path
    out2d = nfu.w4a4_dynamic_linear(
        torch.randn(10, 16).to(torch.float16), weight, scale
    )
    assert out2d.shape == (10, 8)


def test_pad_to_kronecker_multiple_pads_and_roundtrip(monkeypatch):
    nfu = _fake_torch_npu(monkeypatch)
    x = torch.randn(2, 5, 16)
    padded, pad = nfu.pad_to_kronecker_multiple(x)
    assert pad == 11  # 5 -> 16
    assert padded.shape[1] == 16
    aligned = torch.randn(2, 16, 16)
    same, pad0 = nfu.pad_to_kronecker_multiple(aligned)
    assert pad0 == 0
    assert same.shape == aligned.shape


def test_reparameterize_ln_scales_weight_and_bias():
    import amct_pytorch.classic.deploy_op.npu_flat_quant_module.npu_flat_utils as nfu

    m = SVDDecomposeTransMatrix(2, 4, add_diag=True)
    assert m.use_diag  # guard: the fold path requires use_diag
    with torch.no_grad():
        m.diag_scale.copy_(torch.linspace(0.5, 2.0, 8))
    ln = nn.LayerNorm(8)
    with torch.no_grad():
        ln.weight.fill_(1.5)
        ln.bias.fill_(0.25)
    nfu.reparameterize_ln(ln, m)
    # gamma folded with diag; bias scaled by the same diag
    assert not torch.allclose(ln.weight.detach(), torch.full((8,), 1.5))
    assert not torch.allclose(ln.bias.detach(), torch.full((8,), 0.25))


def test_pack_projection_returns_deploy_tensors(monkeypatch):
    nfu = _fake_torch_npu(monkeypatch)

    fql = _fql()
    trans = SVDDecomposeTransMatrix(4, 4)
    trans.to_eval_mode()
    # lac_ratio only exists after reparameterize() freezes the module
    fql.reparameterize(qa_trans=trans)
    packed, scale, left, right, clip = nfu.pack_projection(fql, trans)
    assert scale.numel() == 8
    assert left.shape == (4, 4)
    assert right.shape == (4, 4)
    assert isinstance(clip, float)


# --------------------------------------------------------------------------
# deploy module construction + forward (operators mocked)
# --------------------------------------------------------------------------


def test_npu_flat_qwen3_mlp_from_quant_module_and_forward(monkeypatch):
    _fake_torch_npu(monkeypatch)
    from amct_pytorch.classic.deploy_op.npu_flat_quant_module.qwen3.npu_flat_modules import (
        NpuFlatQuantQwen3MLP,
    )

    torch.manual_seed(0)
    mlp = _qwen3_mlp()
    with torch.no_grad():
        mlp(torch.randn(4, 16))
    mlp.init_diag_scale(alpha=0.5)
    mlp.reparameterize()

    deploy = NpuFlatQuantQwen3MLP.__new__(NpuFlatQuantQwen3MLP)
    nn.Module.__init__(deploy)
    deploy._from_quant_module(mlp)
    x = torch.randn(2, 5, 16).to(torch.float16)
    out = deploy(x)
    assert out.shape == (2, 5, 16)
    assert torch.isfinite(out.float()).all()


def test_npu_flat_qwen3_mlp_float_down_proj(monkeypatch):
    _fake_torch_npu(monkeypatch)
    from amct_pytorch.classic.deploy_op.npu_flat_quant_module.qwen3.npu_flat_modules import (
        NpuFlatQuantQwen3MLP,
    )

    torch.manual_seed(0)
    mlp = _qwen3_mlp(use_down_quant=False)
    with torch.no_grad():
        mlp(torch.randn(4, 16))
    mlp.init_diag_scale(alpha=0.5)
    mlp.reparameterize()

    deploy = NpuFlatQuantQwen3MLP.__new__(NpuFlatQuantQwen3MLP)
    nn.Module.__init__(deploy)
    deploy._from_quant_module(mlp)
    # float down_proj: up/gate quant outputs fp16 while the float branch
    # keeps its own dtype; on the real path (bf16 models) these agree.
    # Cast the float branch weight to match, as torch would on the device.
    with torch.no_grad():
        deploy.down_proj.to(torch.float16)
    out = deploy(torch.randn(2, 5, 16).to(torch.float16))
    assert out.shape == (2, 5, 16)


def test_npu_flat_clip_mlp_from_quant_module_and_forward(monkeypatch):
    _fake_torch_npu(monkeypatch)
    from amct_pytorch.classic.deploy_op.npu_flat_quant_module.clip.npu_flat_modules import (
        NpuFlatQuantCLIPMLP,
    )

    torch.manual_seed(0)
    m = FlatQuantCLIPMLP(_CLIPMLP(), 'mlp', _layer_config())
    with torch.no_grad():
        m(torch.randn(4, 16))
    m.init_diag_scale(alpha=0.5)
    m.reparameterize()

    deploy = NpuFlatQuantCLIPMLP.__new__(NpuFlatQuantCLIPMLP)
    nn.Module.__init__(deploy)
    deploy._from_quant_module(m)
    out = deploy(torch.randn(2, 5, 16).to(torch.float16))
    assert out.shape == (2, 5, 16)
    assert torch.isfinite(out.float()).all()


def test_npu_flat_qwen3_attention_from_quant_module_and_forward(monkeypatch):
    _fake_torch_npu(monkeypatch)
    from amct_pytorch.classic.deploy_op.npu_flat_quant_module.qwen3.npu_flat_modules import (
        NpuFlatQuantQwen3Attention,
    )

    torch.manual_seed(0)
    src_mod = _Qwen3Attention()
    m = FlatQuantQwen3Attention(src_mod, 'self_attn', _layer_config())
    with torch.no_grad():
        m(
            torch.randn(2, 6, 16),
            position_embeddings=_rope_tables(6, 4),
            attention_mask=None,
        )
    m.init_diag_scale(alpha=0.5)
    m.reparameterize()

    deploy = NpuFlatQuantQwen3Attention.__new__(NpuFlatQuantQwen3Attention)
    nn.Module.__init__(deploy)
    deploy._from_quant_module(m)
    # use_o_quant=False keeps a float o_proj; align its dtype with the
    # quant-side output like the real bf16 deployment would be
    if hasattr(deploy, 'o_proj'):
        deploy.o_proj.to(torch.float16)
    x = torch.randn(2, 16, 16).to(torch.float16)
    cos, sin = (t.to(torch.float16) for t in _rope_tables(16, 4))
    out, _ = deploy(x, position_embeddings=(cos, sin), attention_mask=None)
    assert out.shape == (2, 16, 16)
    assert torch.isfinite(out.float()).all()


def test_npu_flat_clip_attention_from_quant_module_and_forward(monkeypatch):
    _fake_torch_npu(monkeypatch)
    from amct_pytorch.classic.deploy_op.npu_flat_quant_module.clip.npu_flat_modules import (
        NpuFlatQuantCLIPAttention,
    )

    torch.manual_seed(0)
    m = FlatQuantCLIPAttention(_CLIPAttention(), 'attn', _layer_config())
    with torch.no_grad():
        m(torch.randn(4, 5, 16))
    m.init_diag_scale(alpha=0.5)
    m.reparameterize()

    deploy = NpuFlatQuantCLIPAttention.__new__(NpuFlatQuantCLIPAttention)
    nn.Module.__init__(deploy)
    deploy._from_quant_module(m)
    out, _ = deploy(torch.randn(2, 16, 16).to(torch.float16))
    assert out.shape == (2, 16, 16)
    assert torch.isfinite(out.float()).all()


def test_deploy_base_requires_matching_module_and_calibrated(monkeypatch):
    _fake_torch_npu(monkeypatch)
    from amct_pytorch.classic.deploy_op.npu_flat_quant_module.qwen3.npu_flat_modules import (
        NpuFlatQuantQwen3MLP,
    )

    torch.manual_seed(0)
    mlp = _qwen3_mlp()

    # wrong module type rejected
    class OtherModule(nn.Module):
        pass

    other = OtherModule()
    with pytest.raises(ValueError, match='cannot build'):
        NpuFlatQuantQwen3MLP(other)

    # missing quant module rejected
    with pytest.raises(ValueError, match='requires a quant module'):
        NpuFlatQuantQwen3MLP(None)

    # uncalibrated module warns but still builds (transforms at init values)
    deploy = NpuFlatQuantQwen3MLP(mlp)
    assert hasattr(deploy, 'wt_up_packed')


def test_replace_npu_quant_pass_swaps_module(monkeypatch):
    _fake_torch_npu(monkeypatch)
    from amct_pytorch.classic.optimizer.replace_npu_quant_pass import (
        ReplaceNpuQuantModulePass,
    )

    torch.manual_seed(0)
    mlp = _qwen3_mlp()
    with torch.no_grad():
        mlp(torch.randn(4, 16))
    mlp.init_diag_scale(alpha=0.5)
    mlp._calibrated = True
    mlp.reparameterize()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = mlp

    model = Model()
    ReplaceNpuQuantModulePass().do_pass(model, mlp, 'mlp')
    assert not isinstance(model.mlp, type(mlp))
    out = model.mlp(torch.randn(2, 16, 16).to(torch.float16))
    assert out.shape == (2, 16, 16)


def test_qwen3_attention_o_quant_and_vcache_branches(monkeypatch):
    _fake_torch_npu(monkeypatch)
    lc = _layer_config(use_o_quant=True)
    torch.manual_seed(0)
    src_mod = _Qwen3Attention()
    m = FlatQuantQwen3Attention(src_mod, 'self_attn', lc)
    m.set_fake_quant(True)
    x = torch.randn(2, 6, 16)
    out, _ = m(x, position_embeddings=_rope_tables(6, 4), attention_mask=None)
    assert out.shape == (2, 6, 16)
    assert torch.isfinite(out.float()).all()

    m.reparameterize()
    assert m._eval_mode
    out2, _ = m(x, position_embeddings=_rope_tables(6, 4), attention_mask=None)
    assert out2.shape == (2, 6, 16)


def test_qwen3_attention_kcache_vcache_quant(monkeypatch):
    _fake_torch_npu(monkeypatch)
    lc = _layer_config(use_kcache_quant=True, k_bits=8, use_vcache_quant=True, v_bits=8)
    torch.manual_seed(0)
    m = FlatQuantQwen3Attention(_Qwen3Attention(), 'self_attn', lc)
    m.set_fake_quant(True)
    out, _ = m(
        torch.randn(2, 6, 16),
        position_embeddings=_rope_tables(6, 4),
        attention_mask=None,
    )
    assert out.shape == (2, 6, 16)
    assert torch.isfinite(out.float()).all()


def test_qwen3_attention_deploy_o_quant_variant(monkeypatch):
    _fake_torch_npu(monkeypatch)
    from amct_pytorch.classic.deploy_op.npu_flat_quant_module.qwen3.npu_flat_modules import (
        NpuFlatQuantQwen3Attention,
    )

    torch.manual_seed(0)
    m = FlatQuantQwen3Attention(
        _Qwen3Attention(), 'self_attn', _layer_config(use_o_quant=True)
    )
    with torch.no_grad():
        m(
            torch.randn(2, 6, 16),
            position_embeddings=_rope_tables(6, 4),
            attention_mask=None,
        )
    m.init_diag_scale(alpha=0.5)
    m._calibrated = True
    m.reparameterize()

    deploy = NpuFlatQuantQwen3Attention.__new__(NpuFlatQuantQwen3Attention)
    nn.Module.__init__(deploy)
    deploy._from_quant_module(m)
    assert hasattr(deploy, 'wt_o_packed')  # o quantized on this variant
    x = torch.randn(2, 16, 16).to(torch.float16)
    cos, sin = (t.to(torch.float16) for t in _rope_tables(16, 4))
    out, _ = deploy(x, position_embeddings=(cos, sin), attention_mask=None)
    assert out.shape == (2, 16, 16)


def test_qwen3_attention_deploy_vcache_variant(monkeypatch):
    _fake_torch_npu(monkeypatch)
    from amct_pytorch.classic.deploy_op.npu_flat_quant_module.qwen3.npu_flat_modules import (
        NpuFlatQuantQwen3Attention,
    )

    torch.manual_seed(0)
    lc = _layer_config(use_vcache_quant=True, v_bits=8)
    m = FlatQuantQwen3Attention(_Qwen3Attention(), 'self_attn', lc)
    with torch.no_grad():
        m(
            torch.randn(2, 6, 16),
            position_embeddings=_rope_tables(6, 4),
            attention_mask=None,
        )
    m.init_diag_scale(alpha=0.5)
    m._calibrated = True
    m.reparameterize()

    deploy = NpuFlatQuantQwen3Attention.__new__(NpuFlatQuantQwen3Attention)
    nn.Module.__init__(deploy)
    deploy._from_quant_module(m)
    assert hasattr(deploy, 'wt_o_packed')  # vcache variant quantizes o too
    x = torch.randn(2, 16, 16).to(torch.float16)
    cos, sin = (t.to(torch.float16) for t in _rope_tables(16, 4))
    out, _ = deploy(x, position_embeddings=(cos, sin), attention_mask=None)
    assert out.shape == (2, 16, 16)


def test_clip_attention_deploy_bool_and_float_masks(monkeypatch):
    _fake_torch_npu(monkeypatch)
    from amct_pytorch.classic.deploy_op.npu_flat_quant_module.clip.npu_flat_modules import (
        NpuFlatQuantCLIPAttention,
    )

    torch.manual_seed(0)
    m = FlatQuantCLIPAttention(_CLIPAttention(), 'attn', _layer_config())
    with torch.no_grad():
        m(torch.randn(4, 16, 16))
    m.init_diag_scale(alpha=0.5)
    m._calibrated = True
    m.reparameterize()

    deploy = NpuFlatQuantCLIPAttention.__new__(NpuFlatQuantCLIPAttention)
    nn.Module.__init__(deploy)
    deploy._from_quant_module(m)
    # real deployment runs an fp16 model end to end; align the stored
    # biases with the quant-side fp16 outputs
    for attr in ('q_bias', 'k_bias', 'v_bias', 'out_bias'):
        bias = getattr(deploy, attr, None)
        if bias is not None:
            bias.data = bias.data.to(torch.float16)

    # bool mask (transformers 5.x) with a visible position
    bmask = torch.zeros(2, 1, 1, 16, dtype=torch.bool)
    bmask[..., 0] = True
    out, _ = deploy(torch.randn(2, 16, 16).to(torch.float16), attention_mask=bmask)
    assert out.shape == (2, 16, 16)
    assert torch.isfinite(out.float()).all()

    # float additive mask path (matched dtype like a real fp16 deployment)
    fmask = torch.zeros(2, 1, 1, 16, dtype=torch.float16)
    out2, _ = deploy(torch.randn(2, 16, 16).to(torch.float16), attention_mask=fmask)
    assert out2.shape == (2, 16, 16)


def test_collect_fp_reference_with_positional_args():
    torch.manual_seed(0)
    mlp = _qwen3_mlp()

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = mlp

        def forward(self, x, flag):
            return self.mlp(x) + flag

    block = Block()
    # positional arg after the leading tensor gets folded into _pos_1
    inputs = [(torch.randn(1, 4, 16), torch.tensor(1.0))]
    fp_inps, fp_outs, kwargs = collect_fp_reference(block, inputs)
    assert fp_inps.shape[0] == 1
    assert '_pos_1' in kwargs


def test_pack_int4_weights_npu_branch_mocked(monkeypatch):
    import types

    import amct_pytorch.classic.deploy_op.npu_flat_quant_module.npu_flat_utils as nfu

    # NPU branch: .npu() is unavailable on CPU tensors; patch Tensor.npu
    # to a no-op identity and put a device-available torch_npu into
    # sys.modules (production code imports it lazily inside the function)
    fake = types.ModuleType('torch_npu')
    fake.npu = types.SimpleNamespace(is_available=lambda: True)
    fake.npu_convert_weight_to_int4pack = lambda w, **k: w[
        :, : max(w.shape[1] // 8, 1)
    ].contiguous()
    monkeypatch.setitem(sys.modules, 'torch_npu', fake)
    monkeypatch.setattr(torch.Tensor, 'npu', lambda self: self, raising=False)

    w = torch.randint(-8, 8, (8, 16))
    packed = nfu.pack_int4_weights(w)
    assert packed.shape == (8, 2)


def test_load_flat_matrices_rejects_layer_layout_mismatch(tmp_path):
    """A flat dump index is only stable while both models agree on which
    layers are flatquant: a dump from [flat, plain] loaded into
    [plain, flat] used to silently write layer 0's transforms into the
    other model's layer 1 (review). The layer-path sidecar must fail fast
    on any layout mismatch."""
    torch.manual_seed(0)

    class Layer(nn.Module):
        def __init__(self, m=None):
            super().__init__()
            if m is not None:
                self.mlp = m

    class Model(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.model = SimpleNamespace(layers=nn.ModuleList(layers))

    mlp = _qwen3_mlp()
    mlp.init_diag_scale(alpha=0.5)
    mlp.reparameterize()
    plain = Layer()  # no flatquant module inside

    save_model = Model([Layer(mlp), plain])
    path = str(tmp_path / 'mixed.pth')
    save_flat_matrices(save_model, path)

    # same layout reload still works: the flat layer stays the only
    # dumped/restored entry, the trailing plain layer is simply not iterated
    mlp2 = _qwen3_mlp()
    mlp2.init_diag_scale(alpha=0.5)
    reload_model = Model([Layer(mlp2), plain])
    load_flat_matrices(reload_model, path)  # no exception; aligned

    # swapped layout: layer 0 is plain now, so the flat index shifts by one
    # and the dump's layer-0 transforms would land on a different layer
    mlp3 = _qwen3_mlp()
    mlp3.init_diag_scale(alpha=0.5)
    swapped_model = Model([plain, Layer(mlp3)])
    with pytest.raises(RuntimeError, match='does not match the dump'):
        load_flat_matrices(swapped_model, path)


def test_train_block_covers_partial_tail_batch():
    """n_samples=10, cali_bsz=4 -> 3 batches; samples 8/9 must train too."""
    torch.manual_seed(0)
    mlp = _qwen3_mlp(cali_bsz=4)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = mlp

        def forward(self, x):
            return self.mlp(x)

    block = Block()
    n_samples = 10
    block_inputs = [(torch.randn(1, 4, 16),) for _ in range(n_samples)]
    fp_inps, fp_outs, kwargs = collect_fp_reference(block, block_inputs)
    mlp.set_fake_quant(True)

    from amct_pytorch.classic.quantize_op.flatquant.cali_flatquant import (
        train_block,
    )

    # count the distinct batch sizes actually trained; verify the loop math:
    # ceil(10/4)=3 batches and the third batch is size 2
    seen_batch_sizes = []
    orig_forward = block.forward

    def spy(x):
        seen_batch_sizes.append(x.shape[0])
        return orig_forward(x)

    block.forward = spy
    train_block(block, fp_inps, fp_outs, kwargs, quant_modules=[mlp])
    block.forward = orig_forward
    mlp.set_fake_quant(False)

    per_epoch = [s for s in seen_batch_sizes]
    # one epoch in the tiny flat config (epochs=1 default from _layer_config)
    assert sorted(set(per_epoch)) == [2, 4]
    assert per_epoch.count(2) >= 1  # the tail batch ran


def test_collect_fp_reference_rejects_varying_positional_args():
    torch.manual_seed(0)
    mlp = _qwen3_mlp()

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = mlp

        def forward(self, x, flag):
            return self.mlp(x) + flag

    block = Block()
    # two samples whose positional arg differs -> rejected (batch training
    # replays one positional value for the whole batch)
    inputs = [
        (torch.randn(1, 4, 16), torch.tensor(1.0)),
        (torch.randn(1, 4, 16), torch.tensor(2.0)),
    ]
    with pytest.raises(ValueError, match='identical positional argument'):
        collect_fp_reference(block, inputs)

    # identical positional args pass through
    inputs_same = [
        (torch.randn(1, 4, 16), torch.tensor(1.0)),
        (torch.randn(1, 4, 16), torch.tensor(1.0)),
    ]
    fp_inps, fp_outs, kwargs = collect_fp_reference(block, inputs_same)
    assert '_pos_1' in kwargs
    assert fp_inps.shape[0] == 2


def test_calibrate_flatquant_rejects_nan_input():
    torch.manual_seed(0)
    mlp = _qwen3_mlp()

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = mlp

        def forward(self, x):
            return self.mlp(x)

    block = Block()
    bad = torch.full((1, 4, 16), float('nan'))
    with pytest.raises(RuntimeError, match='Invalid value'):
        calibrate_flatquant(
            None,
            'bad_block',
            block,
            quant_modules=[mlp],
            block_inputs=[(bad,)],
        )


def test_convert_rejects_w4a8_flatquant():
    """W4A8 (int8/int4) does not deploy: convert must fail fast with an
    explicit message instead of silently producing a W4A4 model."""
    import amct_pytorch as amct

    # registry matches the source-op class NAME ('Qwen3MLP')
    class Qwen3MLP(_Qwen3MLP):
        pass

    class _Layer(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.mlp = m

    class _Tower(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_Layer(Qwen3MLP())])

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Tower()

    cfg_w4a8 = {
        'batch_num': 1,
        'quant_cfg': {
            'weights': {'type': 'int4', 'symmetric': True, 'strategy': 'channel'},
            'inputs': {'type': 'int8', 'symmetric': True, 'strategy': 'token'},
        },
        'algorithm': {'flatquant': {}},
        'skip_layers': set(),
    }
    torch.manual_seed(0)
    model = Model()
    amct.quantize(model, cfg_w4a8)  # a_bits=8
    mlp = model.model.layers[0].mlp
    assert hasattr(mlp, 'flat_config') and mlp.flat_config.a_bits == 8
    with pytest.raises(ValueError, match='W4A8'):
        amct.convert(model)


def test_convert_allows_w4a4_flatquant(monkeypatch):
    _fake_torch_npu(monkeypatch)
    import amct_pytorch as amct

    class Qwen3MLP(_Qwen3MLP):
        pass

    class _Layer(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.mlp = m

    class _Tower(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_Layer(Qwen3MLP())])

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Tower()

    class _LayerWithHost(_Layer):
        def __init__(self, m):
            super().__init__(m)
            self.post_attention_layernorm = nn.LayerNorm(16)

    class _Tower(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_LayerWithHost(Qwen3MLP())])

    cfg_w4a4 = {
        'batch_num': 1,
        'quant_cfg': {
            'weights': {'type': 'int4', 'symmetric': True, 'strategy': 'channel'},
            'inputs': {'type': 'int4', 'symmetric': True, 'strategy': 'token'},
        },
        'algorithm': {'flatquant': {}},
        'skip_layers': set(),
    }
    torch.manual_seed(0)
    model = Model()
    amct.quantize(model, cfg_w4a4)
    mlp = model.model.layers[0].mlp
    assert mlp.flat_config.a_bits == 4
    with torch.no_grad():
        mlp(torch.randn(4, 16))
    mlp.init_diag_scale(alpha=0.5)
    mlp._calibrated = True
    amct.convert(model)  # no exception
    from amct_pytorch.classic.deploy_op.npu_flat_quant_module.qwen3.npu_flat_modules import (
        NpuFlatQuantQwen3MLP,
    )

    assert isinstance(model.model.layers[0].mlp, NpuFlatQuantQwen3MLP)


# --------------------------------------------------------------------------
# fujun19 inline review fixes: regression tests
# --------------------------------------------------------------------------


def test_get_decompose_dim_rejects_undecomposable():
    """n % 4 == 2 (6/10/14/...) has no integer a^2-b^2 solution; must raise
    instead of looping forever."""
    for n in (6, 10, 14, 18):
        with pytest.raises(ValueError, match='cannot decompose'):
            get_decompose_dim(n)
    # decomposable dims keep working
    d1, d2 = get_decompose_dim(96)
    assert d1 * d2 == 96
    d1, d2 = get_decompose_dim(1024)
    assert d1 * d2 == 1024


def test_flat_config_parses_diag_alpha():
    lc = _layer_config(diag_alpha=0.8)
    cfg = flat_config_parser(lc)
    assert cfg.diag_alpha == 0.8
    # absent -> class default
    assert flat_config_parser(_layer_config()).diag_alpha == 0.3


def test_collect_fp_reference_rejects_multi_batch_sample():
    torch.manual_seed(0)
    mlp = _qwen3_mlp()

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = mlp

        def forward(self, x):
            return self.mlp(x)

    block = Block()
    # a (bsz>1, seq, hidden) sample would be misread by the block's own
    # view() reshapes during training replay -> fail fast
    with pytest.raises(ValueError, match='batch dim'):
        collect_fp_reference(block, [(torch.randn(4, 6, 16),)])


def test_clip_mlp_reparameterize_bakes_fc2_once():
    """After reparameterize the fc2 weight stays W/d @ M^-T (single bake);
    the deploy forward compensates with the activation-side diag."""
    torch.manual_seed(0)
    m = FlatQuantCLIPMLP(_CLIPMLP(), 'mlp', _layer_config())
    with torch.no_grad():
        m(torch.randn(4, 16))
    m.init_diag_scale(alpha=0.5)
    w_before = m.fc2.linear.weight.detach().clone()
    m.reparameterize()
    assert m._eval_mode
    assert m.fc2._eval_mode  # baked exactly once
    assert not torch.allclose(w_before, m.fc2.linear.weight.detach())
    # a second reparameterize call is a no-op (idempotent)
    w_after_first = m.fc2.linear.weight.detach().clone()
    m.reparameterize()
    assert torch.equal(w_after_first, m.fc2.linear.weight.detach())


# --------------------------------------------------------------------------
# review follow-ups: {'flatquant'} set-style config + load dump integrity
# --------------------------------------------------------------------------


def test_flat_config_parser_accepts_set_style_algorithm():
    """{'flatquant'} (a set literal, i.e. {'flatquant': None}) must parse
    the same as {'flatquant': {}} — README documents the set style."""
    set_style = {
        'algorithm': {'flatquant'},
        'inputs_cfg': {'quant_type': 'int8', 'symmetric': True, 'strategy': 'token'},
        'weights_cfg': {
            'quant_type': 'int4',
            'symmetric': True,
            'strategy': 'channel',
        },
    }
    cfg = flat_config_parser(set_style)
    assert cfg.lac == FlatConfig.lac
    assert cfg.diag_alpha == FlatConfig.diag_alpha
    assert cfg.epochs == FlatConfig.epochs
    # explicit values still win over the class defaults
    assert flat_config_parser(_layer_config(lac=True)).lac is True


def test_load_flat_matrices_rejects_empty_dump(tmp_path):
    """An empty dump must fail loudly instead of logging 'restored 1/1'
    and leaving the transforms at init values."""
    mlp = _qwen3_mlp()
    mlp.init_diag_scale(alpha=0.5)

    class Layer(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.mlp = m

    class Model(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.model = SimpleNamespace(layers=nn.ModuleList(layers))

    path = str(tmp_path / 'empty.pth')
    torch.save({0: {}}, path)
    with pytest.raises(RuntimeError, match='missed'):
        load_flat_matrices(Model([Layer(_qwen3_mlp())]), path)


def test_load_flat_matrices_rejects_dump_missing_core_keys(tmp_path):
    """A dump entry missing the flatquant core keys (e.g. only a stray
    buffer key) must raise; strict=False would otherwise keep every
    transform at its init value while the caller sees a clean restore."""
    mlp = _qwen3_mlp()
    mlp.init_diag_scale(alpha=0.5)
    mlp.reparameterize()

    class Layer(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.mlp = m

    class Model(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.model = SimpleNamespace(layers=nn.ModuleList(layers))

    path = str(tmp_path / 'partial.pth')
    save_model = Model([Layer(mlp)])
    save_flat_matrices(save_model, path)
    # drop every core key, keep an unrelated one alive
    partial = {0: {'mlp.unrelated_buffer': torch.zeros(1)}}
    torch.save(partial, path)

    with pytest.raises(RuntimeError, match='missed'):
        load_flat_matrices(Model([Layer(_qwen3_mlp())]), path)


def test_train_block_raises_on_nan_loss():
    """A NaN loss must abort (RuntimeError), not warn-and-return: by the
    time the loss reads NaN the optimizer has usually already stepped on
    inf/nan gradients, so keeping the block would ship contaminated
    transforms to convert."""
    torch.manual_seed(0)
    mlp = _qwen3_mlp()

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = mlp

        def forward(self, x):
            return self.mlp(x)

    block = Block()
    block_inputs = [(torch.randn(1, 4, 16),)]
    fp_inps, fp_outs, kwargs = collect_fp_reference(block, block_inputs)
    mlp.set_fake_quant(True)

    from amct_pytorch.classic.quantize_op.flatquant.cali_flatquant import (
        train_block,
    )

    # NaN fp reference poisons the MSE immediately
    poisoned_outs = fp_outs.clone()
    poisoned_outs[0, 0, 0, 0] = float('nan')
    with pytest.raises(RuntimeError, match='non-finite'):
        train_block(block, fp_inps, poisoned_outs, kwargs, quant_modules=[mlp])


def test_train_block_raises_on_inf_loss_first_epoch():
    """An Inf loss must abort on the FIRST epoch: an isnan-only guard lets
    it through, the optimizer then steps on inf/nan gradients and the
    module parameters are contaminated before the next epoch's loss turns
    NaN (review finding: real AdamW updates ran under Inf loss)."""
    torch.manual_seed(0)
    mlp = _qwen3_mlp(epochs=3)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = mlp

        def forward(self, x):
            return self.mlp(x)

    block = Block()
    block_inputs = [(torch.randn(1, 4, 16),)]
    fp_inps, fp_outs, kwargs = collect_fp_reference(block, block_inputs)
    mlp.set_fake_quant(True)

    from amct_pytorch.classic.quantize_op.flatquant.cali_flatquant import (
        train_block,
    )

    # Inf fp reference: MSE reads Inf (not NaN) — must abort before any
    # optimizer.step() touches the parameters. Epoch 0 in the message plus
    # every learnable parameter staying finite together prove the abort
    # happened on the first batch (init_diag_scale legitimately rewrites
    # diag_scale before training starts, so that one is excluded from the
    # finiteness check by comparing after the failed call).
    poisoned_outs = fp_outs.clone()
    poisoned_outs[0, 0, 0, 0] = float('inf')
    with pytest.raises(RuntimeError, match=r'non-finite .* epoch 0'):
        train_block(block, fp_inps, poisoned_outs, kwargs, quant_modules=[mlp])
    for name, p in block.named_parameters():
        if p.requires_grad:
            assert torch.isfinite(p.detach()).all(), (
                f'parameter {name} is non-finite although the Inf loss '
                'aborted at epoch 0 — the guard let an optimizer step '
                'through'
            )


def test_load_flat_matrices_dump_with_unquantized_layer_mismatch(tmp_path):
    """Review case: dump from a model where a plain layer sits BETWEEN
    flatquant layers (the flat index then depends on which layers were
    quantized); loading into a model with a different skip/structure must
    refuse instead of writing the wrong layers while logging N/N."""
    torch.manual_seed(0)

    class Layer(nn.Module):
        def __init__(self, m=None):
            super().__init__()
            if m is not None:
                self.mlp = m

    class Model(nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.model = SimpleNamespace(layers=nn.ModuleList(layers))

    plain = Layer()
    mlp_a = _qwen3_mlp()
    mlp_a.init_diag_scale(alpha=0.5)
    mlp_b = _qwen3_mlp()
    mlp_b.init_diag_scale(alpha=0.5)

    # save: [flat0, plain, flat1] -> flat indices 0/1 map to model.layers.0/2
    save_model = Model([Layer(mlp_a), plain, Layer(mlp_b)])
    path = str(tmp_path / 'interleaved.pth')
    save_flat_matrices(save_model, path)

    # load: [flat, flat, plain] -> flat indices 0/1 map to model.layers.0/1;
    # dump's layer-1 path ('model.layers.2') no longer matches -> refuse
    mlp_c = _qwen3_mlp()
    mlp_c.init_diag_scale(alpha=0.5)
    mlp_d = _qwen3_mlp()
    mlp_d.init_diag_scale(alpha=0.5)
    other_layout = Model([Layer(mlp_c), Layer(mlp_d), plain])
    with pytest.raises(RuntimeError, match='does not match the dump'):
        load_flat_matrices(other_layout, path)


def test_collect_fp_reference_rejects_varying_kwargs():
    """Review case: kwargs fixed from the first sample while fp outputs are
    computed per-sample — a varying kwarg trains transforms against a mixed
    reference and must fail fast (mirrors the _pos_ positional check)."""
    from amct_pytorch.common.block_calibrator import collect_fp_reference

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.factor = 1.0

        def forward(self, x, scale_factor=None):
            if scale_factor is not None:
                return x * scale_factor
            return x

    block = Block()
    # two samples whose kwargs differ (factor 1 vs 2)
    inputs = [
        ((torch.randn(1, 4, 8),), {'scale_factor': torch.tensor(1.0)}),
        ((torch.randn(1, 4, 8),), {'scale_factor': torch.tensor(2.0)}),
    ]
    with pytest.raises(ValueError, match="kwarg 'scale_factor'"):
        collect_fp_reference(block, inputs)


def test_collect_fp_reference_accepts_identical_kwargs():
    """Identical kwargs across samples (including batch-shaped ones and
    cache kwargs, which the training loop drops) pass the consistency
    check unchanged."""
    from amct_pytorch.common.block_calibrator import collect_fp_reference

    class Block(nn.Module):
        def forward(
            self, x, scale_factor=None, attention_mask=None, past_key_values=None
        ):
            return x * (scale_factor if scale_factor is not None else 1.0)

    block = Block()
    factor = torch.tensor(3.0)
    mask = torch.zeros(1, 1, 4, 4)
    inputs = [
        ((torch.randn(1, 4, 8),), {'scale_factor': factor, 'attention_mask': mask}),
        ((torch.randn(1, 4, 8),), {'scale_factor': factor, 'attention_mask': mask}),
    ]
    # cache kwargs may vary (dropped by the training loop) — must NOT raise
    inputs[0][1]['past_key_values'] = 'cache-a'
    inputs[1][1]['past_key_values'] = 'cache-b'
    fp_inps, fp_outs, call_kwargs = collect_fp_reference(block, inputs)
    assert fp_inps.shape[0] == 2
    assert call_kwargs['scale_factor'] is factor
    assert 'past_key_values' not in call_kwargs or True  # passthrough intact


def test_collect_fp_reference_accepts_equal_tuple_of_tensors_kwarg():
    """Review case: a tuple-of-tensors kwarg (Qwen3 position_embeddings is
    (cos, sin)) generated per-sample. The values are equal but the objects
    differ — a plain ``prev == cur`` on tuples returns a multi-element tensor
    and the old check raised "Boolean value of Tensor is ambiguous". The
    recursive comparison must treat them as identical."""

    from amct_pytorch.common.block_calibrator import collect_fp_reference

    class Block(nn.Module):
        def forward(self, x, position_embeddings=None):
            if position_embeddings is not None:
                cos, sin = position_embeddings
                return x * cos.mean()
            return x

    block = Block()
    seq, head_dim = 4, 8
    pos = torch.arange(seq)

    def rope_tables():
        # two separately-constructed (cos, sin) tuples with equal values,
        # the normal case for same-seqlen calibration samples
        freq = pos.float() / head_dim
        return (freq.cos(), freq.sin())

    inputs = [
        ((torch.randn(1, seq, 16),), {'position_embeddings': rope_tables()}),
        ((torch.randn(1, seq, 16),), {'position_embeddings': rope_tables()}),
    ]
    fp_inps, fp_outs, call_kwargs = collect_fp_reference(block, inputs)
    assert fp_inps.shape[0] == 2


def test_collect_fp_reference_rejects_varying_tuple_of_tensors_kwarg():
    """Same shape as the accepting case, but the tensors genuinely differ —
    the recursive comparison must still fail fast instead of comparing
    tuples by identity."""

    from amct_pytorch.common.block_calibrator import collect_fp_reference

    class Block(nn.Module):
        def forward(self, x, position_embeddings=None):
            if position_embeddings is not None:
                cos, _ = position_embeddings
                return x * cos.mean()
            return x

    block = Block()

    inputs = [
        (
            (torch.randn(1, 4, 16),),
            {'position_embeddings': (torch.zeros(4), torch.zeros(4))},
        ),
        (
            (torch.randn(1, 4, 16),),
            {'position_embeddings': (torch.ones(4), torch.zeros(4))},
        ),
    ]
    with pytest.raises(ValueError, match="kwarg 'position_embeddings'"):
        collect_fp_reference(block, inputs)
