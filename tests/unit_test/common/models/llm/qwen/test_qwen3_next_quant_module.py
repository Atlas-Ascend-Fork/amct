"""Tests for QuantQwen3NextAttn / QuantQwen3NextMLP using tiny configs."""

from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextConfig,
    Qwen3NextMLP,
)

from amct_pytorch.common.models.llm.common.quant_apply import PlainLinear
from amct_pytorch.common.models.llm.qwen.qwen3_next.quant_module import (
    QuantQwen3NextAttn,
    QuantQwen3NextMLP,
)
from amct_pytorch.quantization.dtypes import register_dtype
from amct_pytorch.quantization.modules.quant_linear import QuantLinear

register_dtype()


def _tiny_config():
    return Qwen3NextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=32,
        max_position_embeddings=64,
    )


def _quant_args(quant_target=("attn-linear",)):
    return SimpleNamespace(
        algos=[],
        quant_dtype="int",
        w_bits=8,
        a_bits=8,
        q_bits=8, k_bits=8, p_bits=8, v_bits=8,
        quant_target=list(quant_target),
    )


# ---- QuantQwen3NextMLP ---------------------------------------------------


def test_quant_qwen3_next_mlp_forward_preserves_shape():
    cfg = _tiny_config()
    qmlp = QuantQwen3NextMLP(_quant_args(quant_target=["mlp"]), Qwen3NextMLP(cfg))
    x = torch.randn(1, 4, cfg.hidden_size)
    assert qmlp(x).shape == x.shape


def test_quant_qwen3_next_mlp_uses_quant_linear():
    cfg = _tiny_config()
    qmlp = QuantQwen3NextMLP(_quant_args(quant_target=["mlp"]), Qwen3NextMLP(cfg))
    assert isinstance(qmlp.up_proj, QuantLinear)
    assert isinstance(qmlp.gate_proj, QuantLinear)
    assert isinstance(qmlp.down_proj, QuantLinear)


# ---- QuantQwen3NextAttn --------------------------------------------------


def test_quant_qwen3_next_attn_attn_linear_branch_uses_quant_linear():
    cfg = _tiny_config()
    attn = Qwen3NextAttention(cfg, layer_idx=0)
    qattn = QuantQwen3NextAttn(_quant_args(quant_target=["attn-linear"]), attn)
    for proj in (qattn.q_proj, qattn.k_proj, qattn.v_proj, qattn.o_proj):
        assert isinstance(proj, QuantLinear)


def test_quant_qwen3_next_attn_passthrough_branch_uses_plain_linear():
    cfg = _tiny_config()
    attn = Qwen3NextAttention(cfg, layer_idx=0)
    qattn = QuantQwen3NextAttn(_quant_args(quant_target=["attn-cache"]), attn)
    for proj in (qattn.q_proj, qattn.k_proj, qattn.v_proj, qattn.o_proj):
        assert isinstance(proj, PlainLinear)


@pytest.mark.parametrize("quant_target", [["attn-linear"], ["attn-cache"]])
def test_quant_qwen3_next_attn_forward_returns_correct_shape(quant_target):
    cfg = _tiny_config()
    attn = Qwen3NextAttention(cfg, layer_idx=0)
    qattn = QuantQwen3NextAttn(_quant_args(quant_target=quant_target), attn)
    bs, seq = 1, 4
    h = torch.randn(bs, seq, cfg.hidden_size)
    cos = torch.randn(bs, seq, cfg.head_dim)
    sin = torch.randn(bs, seq, cfg.head_dim)
    out, weights = qattn(h, position_embeddings=(cos, sin))
    assert out.shape == h.shape
    assert weights is None
