"""Tests for QuantQwen3NextAttn / QuantQwen3NextMLP / QuantQwen3NextLinearAttn using tiny configs."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextConfig,
    Qwen3NextGatedDeltaNet,
    Qwen3NextMLP,
)

from amct_pytorch.common.models.llm.common.quant_apply import PlainLinear
from amct_pytorch.common.models.llm.qwen.qwen3_next.quant_module import (
    QuantQwen3NextAttn,
    QuantQwen3NextLinearAttn,
    QuantQwen3NextMLP,
)
from amct_pytorch.quantization.bit_policy import BitPolicy
from amct_pytorch.quantization.dtypes import register_dtype
from amct_pytorch.quantization.modules.quant_base import ActivationQuantizer
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


def _linear_attn_config():
    return Qwen3NextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=32,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
    )


def _quant_args(quant_target=("attn-linear",)):
    return SimpleNamespace(
        algos=[],
        quant_dtype="int",
        w_bits=8,
        a_bits=8,
        q_bits=8, k_bits=8, p_bits=8, v_bits=8,
        quant_target=list(quant_target),
        bit_policy=BitPolicy({
            "attn-linear": {
                "q_proj": {"w_bits": 8, "a_bits": 8},
                "k_proj": {"w_bits": 8, "a_bits": 8},
                "v_proj": {"w_bits": 8, "a_bits": 8},
                "o_proj": {"w_bits": 8, "a_bits": 8},
                "out_proj": {"w_bits": 8, "a_bits": 8},
                "in_proj_qkv": {"w_bits": 8, "a_bits": 8},
                "in_proj_z": {"w_bits": 8, "a_bits": 8},
                "in_proj_b": {"w_bits": 8, "a_bits": 8},
                "in_proj_a": {"w_bits": 8, "a_bits": 8},
            },
            "attn-cache": {"q": 8, "k": 8, "p": 8, "v": 8},
            "mlp": {
                "gate_proj": {"w_bits": 8, "a_bits": 8},
                "up_proj": {"w_bits": 8, "a_bits": 8},
                "down_proj": {"w_bits": 8, "a_bits": 8},
            },
        }),
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


def test_quant_qwen3_next_mlp_has_activation_quantizers():
    cfg = _tiny_config()
    qmlp = QuantQwen3NextMLP(_quant_args(quant_target=["mlp"]), Qwen3NextMLP(cfg))
    assert isinstance(qmlp.input_quant, ActivationQuantizer)
    assert isinstance(qmlp.hidden_quant, ActivationQuantizer)


def test_quant_qwen3_next_mlp_export_ptq_params():
    cfg = _tiny_config()
    qmlp = QuantQwen3NextMLP(_quant_args(quant_target=["mlp"]), Qwen3NextMLP(cfg))
    params = qmlp.export_ptq_params()
    assert isinstance(params, dict)


# ---- QuantQwen3NextAttn --------------------------------------------------


def test_quant_qwen3_next_attn_attn_linear_branch_uses_quant_linear():
    cfg = _tiny_config()
    attn = Qwen3NextAttention(cfg, layer_idx=0)
    qattn = QuantQwen3NextAttn(_quant_args(quant_target=["attn-linear"]), attn)
    for proj in (qattn.q_proj, qattn.k_proj, qattn.v_proj, qattn.o_proj):
        assert isinstance(proj, QuantLinear)


def test_quant_qwen3_next_attn_attn_linear_branch_has_activation_quantizers():
    cfg = _tiny_config()
    attn = Qwen3NextAttention(cfg, layer_idx=0)
    qattn = QuantQwen3NextAttn(_quant_args(quant_target=["attn-linear"]), attn)
    assert isinstance(qattn.inp_afq, ActivationQuantizer)
    assert isinstance(qattn.o_proj_afq, ActivationQuantizer)


def test_quant_qwen3_next_attn_passthrough_branch_uses_plain_linear():
    cfg = _tiny_config()
    attn = Qwen3NextAttention(cfg, layer_idx=0)
    qattn = QuantQwen3NextAttn(_quant_args(quant_target=["attn-cache"]), attn)
    for proj in (qattn.q_proj, qattn.k_proj, qattn.v_proj, qattn.o_proj):
        assert isinstance(proj, PlainLinear)


def test_quant_qwen3_next_attn_passthrough_branch_uses_identity_afq():
    cfg = _tiny_config()
    attn = Qwen3NextAttention(cfg, layer_idx=0)
    qattn = QuantQwen3NextAttn(_quant_args(quant_target=["attn-cache"]), attn)
    assert isinstance(qattn.inp_afq, nn.Identity)
    assert isinstance(qattn.o_proj_afq, nn.Identity)


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


def test_quant_qwen3_next_attn_batch_forward():
    cfg = _tiny_config()
    attn = Qwen3NextAttention(cfg, layer_idx=0)
    qattn = QuantQwen3NextAttn(_quant_args(quant_target=["attn-linear"]), attn)
    bs, seq = 2, 8
    h = torch.randn(bs, seq, cfg.hidden_size)
    cos = torch.randn(bs, seq, cfg.head_dim)
    sin = torch.randn(bs, seq, cfg.head_dim)
    out, weights = qattn(h, position_embeddings=(cos, sin))
    assert out.shape == (bs, seq, cfg.hidden_size)
    assert weights is None


# ---- QuantQwen3NextLinearAttn --------------------------------------------


def _make_linear_attn(quant_target):
    cfg = _linear_attn_config()
    gm = Qwen3NextGatedDeltaNet(cfg, layer_idx=0)
    gm.config = cfg
    return QuantQwen3NextLinearAttn(_quant_args(quant_target=quant_target), gm)


def test_quant_qwen3_next_linear_attn_attn_linear_branch_uses_quant_linear():
    qla = _make_linear_attn(["attn-linear"])
    for proj in (qla.in_proj_qkvz, qla.in_proj_ba, qla.out_proj):
        assert isinstance(proj, QuantLinear)


def test_quant_qwen3_next_linear_attn_attn_linear_branch_has_activation_quantizers():
    qla = _make_linear_attn(["attn-linear"])
    assert isinstance(qla.inp_afq, ActivationQuantizer)
    assert isinstance(qla.o_proj_afq, ActivationQuantizer)


def test_quant_qwen3_next_linear_attn_passthrough_branch_uses_plain_linear():
    qla = _make_linear_attn(["attn-cache"])
    for proj in (qla.in_proj_qkvz, qla.in_proj_ba, qla.out_proj):
        assert isinstance(proj, PlainLinear)
    assert qla.input_transform is None
    assert qla.out_transform is None


def test_quant_qwen3_next_linear_attn_passthrough_branch_uses_identity_afq():
    qla = _make_linear_attn(["attn-cache"])
    assert isinstance(qla.inp_afq, nn.Identity)
    assert isinstance(qla.o_proj_afq, nn.Identity)


def test_quant_qwen3_next_linear_attn_enable_attn_linear_flag():
    qla_linear = _make_linear_attn(["attn-linear"])
    assert qla_linear.enable_attn_linear is True
    qla_cache = _make_linear_attn(["attn-cache"])
    assert qla_cache.enable_attn_linear is False


def test_quant_qwen3_next_linear_attn_hidden_size():
    qla = _make_linear_attn(["attn-linear"])
    assert qla.hidden_size == 64


@pytest.mark.parametrize("quant_target", [["attn-linear"], ["attn-cache"]])
def test_quant_qwen3_next_linear_attn_forward_returns_correct_shape(quant_target):
    qla = _make_linear_attn(quant_target)
    x = torch.randn(1, 4, 64)
    out = qla(x)
    assert out.shape == x.shape


def test_quant_qwen3_next_linear_attn_forward_batch():
    qla = _make_linear_attn(["attn-linear"])
    bs, seq = 2, 8
    x = torch.randn(bs, seq, 64)
    out = qla(x)
    assert out.shape == (bs, seq, 64)


def test_quant_qwen3_next_linear_attn_delegated_attributes():
    qla = _make_linear_attn(["attn-linear"])
    assert hasattr(qla, "conv1d")
    assert hasattr(qla, "dt_bias")
    assert hasattr(qla, "A_log")
    assert hasattr(qla, "norm")
