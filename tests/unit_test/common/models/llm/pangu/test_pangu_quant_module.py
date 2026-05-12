"""Tests for QuantPanguAttn / QuantPanguMLP using fake attn_module objects.

QuantPanguAttn doesn't subclass an HF attention class — it pulls attributes
from `attn_module`, so we can fake it without loading Pangu transformers code.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from amct_pytorch.common.models.llm.common.quant_apply import PlainLinear
from amct_pytorch.common.models.llm.pangu.pangu_7b.quant_module import (
    QuantPanguAttn,
    QuantPanguMLP,
    apply_rotary_pos_emb,
    rotate_half,
)
from amct_pytorch.quantization.bit_policy import BitPolicy
from amct_pytorch.quantization.dtypes import register_dtype
from amct_pytorch.quantization.modules.quant_linear import QuantLinear

register_dtype()


def _quant_args(quant_target=("attn-linear",)):
    return SimpleNamespace(
        algos=[],
        quant_dtype="int",
        w_bits=8,
        a_bits=8,
        q_bits=8, k_bits=8, p_bits=8, v_bits=8,
        quant_target=list(quant_target),
        bit_policy=BitPolicy({
            "mlp": {
                "gate_proj": {"w_bits": 8, "a_bits": 8},
                "up_proj": {"w_bits": 8, "a_bits": 8},
                "down_proj": {"w_bits": 8, "a_bits": 8},
            },
            "attn-linear": {
                "q_proj": {"w_bits": 8, "a_bits": 8},
                "k_proj": {"w_bits": 8, "a_bits": 8},
                "v_proj": {"w_bits": 8, "a_bits": 8},
                "o_proj": {"w_bits": 8, "a_bits": 8},
            },
            "attn-cache": {"q": 8, "k": 8, "p": 8, "v": 8},
        }),
    )


def _fake_attn_module(hidden_size=64, num_heads=4, num_kv_heads=2, head_dim=16):
    attn = SimpleNamespace()
    attn.config = SimpleNamespace(hidden_size=hidden_size)
    attn.layer_idx = 0
    attn.hidden_size = hidden_size
    attn.head_dim = head_dim
    attn.num_heads = num_heads
    attn.num_key_value_heads = num_kv_heads
    attn.num_key_value_groups = num_heads // num_kv_heads
    attn.scaling = head_dim ** -0.5
    attn.attention_dropout = 0.0
    attn.is_causal = True
    attn.value_dim = num_kv_heads * head_dim
    attn.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
    attn.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
    attn.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
    attn.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
    return attn


class _FakeMLP(nn.Module):
    """Minimal substitute for Pangu's gated MLP module that QuantGatedMLP wraps."""

    def __init__(self, hidden_size=64, intermediate_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()


# ---- rotate_half / apply_rotary_pos_emb ----------------------------------


def test_rotate_half_swaps_and_negates_first_half():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # last dim split: [1,2] | [3,4]
    out = rotate_half(x)
    # rotate_half returns concat(-x2, x1) -> [-3, -4, 1, 2]
    assert torch.equal(out, torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))


def test_apply_rotary_pos_emb_with_cos_one_sin_zero_is_identity():
    bs, h, s, d = 1, 2, 3, 4
    q = torch.randn(bs, h, s, d)
    k = torch.randn(bs, h, s, d)
    cos = torch.ones(bs, s, d)
    sin = torch.zeros(bs, s, d)
    q_e, k_e = apply_rotary_pos_emb(q, k, cos, sin)
    assert torch.allclose(q_e, q, atol=1e-6)
    assert torch.allclose(k_e, k, atol=1e-6)


# ---- QuantPanguMLP -------------------------------------------------------


def test_quant_pangu_mlp_forward_preserves_shape():
    qmlp = QuantPanguMLP(_quant_args(quant_target=["mlp"]), _FakeMLP())
    x = torch.randn(1, 4, 64)
    assert qmlp(x).shape == x.shape


def test_quant_pangu_mlp_uses_quant_linear():
    qmlp = QuantPanguMLP(_quant_args(quant_target=["mlp"]), _FakeMLP())
    assert isinstance(qmlp.up_proj, QuantLinear)
    assert isinstance(qmlp.gate_proj, QuantLinear)
    assert isinstance(qmlp.down_proj, QuantLinear)


# ---- QuantPanguAttn ------------------------------------------------------


def test_quant_pangu_attn_attn_linear_branch_uses_quant_linear():
    qattn = QuantPanguAttn(_quant_args(quant_target=["attn-linear"]), _fake_attn_module())
    for proj in (qattn.q_proj, qattn.k_proj, qattn.v_proj, qattn.o_proj):
        assert isinstance(proj, QuantLinear)
    assert qattn.input_transform is None  # algos=[] => no structure algo
    assert qattn.out_transform is None


def test_quant_pangu_attn_passthrough_branch_uses_plain_linear():
    qattn = QuantPanguAttn(_quant_args(quant_target=["attn-cache"]), _fake_attn_module())
    for proj in (qattn.q_proj, qattn.k_proj, qattn.v_proj, qattn.o_proj):
        assert isinstance(proj, PlainLinear)
    assert isinstance(qattn.inp_afq, nn.Identity)
    assert isinstance(qattn.o_proj_afq, nn.Identity)


def test_quant_pangu_attn_metadata_propagates_from_attn_module():
    attn = _fake_attn_module(num_heads=4, num_kv_heads=2, head_dim=16)
    qattn = QuantPanguAttn(_quant_args(), attn)
    assert qattn.head_dim == 16
    assert qattn.num_heads == 4
    assert qattn.num_key_value_heads == 2
    assert qattn.num_key_value_groups == 2
    assert qattn.is_causal is True


@pytest.mark.parametrize("quant_target", [["attn-linear"], ["attn-cache"]])
def test_quant_pangu_attn_forward_returns_correct_shape(quant_target):
    attn = _fake_attn_module()
    qattn = QuantPanguAttn(_quant_args(quant_target=quant_target), attn)
    bs, seq, hidden, head_dim = 1, 4, 64, 16
    h = torch.randn(bs, seq, hidden)
    cos = torch.randn(bs, seq, head_dim)
    sin = torch.randn(bs, seq, head_dim)
    out, weights = qattn(h, position_embeddings=(cos, sin), attention_mask=None)
    assert out.shape == h.shape
    assert weights is None
