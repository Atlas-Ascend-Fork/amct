"""Structural tests for QuantGLM5Attn / QuantGLM5MLP / QuantGLM5Indexer.

Forward of QuantGLM5Attn requires a real attention-implementation backend to
match `_attn_implementation`; here we cover constructor wiring and the parts
of QuantGLM5Indexer that don't need full attention dispatch.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaAttention

from amct_pytorch.common.models.llm.glm.glm5.glm5 import (
    QuantGLM5Attn,
    QuantGLM5Indexer,
    QuantGLM5MLP,
    _linear_with_structure,
)
from amct_pytorch.quantization.bit_policy import BitPolicy
from amct_pytorch.quantization.dtypes import register_dtype
from amct_pytorch.quantization.modules.quant_linear import QuantLinear

register_dtype()


def _tiny_config():
    return GlmMoeDsaConfig(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=4,
        routed_scaling_factor=1.0,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=16,
        vocab_size=32,
        max_position_embeddings=64,
        index_n_heads=4,
        index_head_dim=16,
        index_topk=2,
        first_k_dense_replace=1,
        moe_layer_freq=1,
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


# ---- _linear_with_structure ----------------------------------------------


def test_linear_with_structure_no_transform_matches_F_linear():
    linear = nn.Linear(4, 8)
    x = torch.randn(2, 4)
    out = _linear_with_structure(x, linear)
    assert torch.allclose(out, linear(x), atol=1e-6)


def test_linear_with_structure_passes_inv_t_and_name_to_transform():
    linear = nn.Linear(4, 4)
    seen = {}

    def transform(weight, inv_t=False, name=None):
        seen["inv_t"] = inv_t
        seen["name"] = name
        return weight

    _linear_with_structure(torch.zeros(1, 4), linear, structure_transform=transform, name="proj")
    assert seen == {"inv_t": True, "name": "proj"}


# ---- QuantGLM5MLP --------------------------------------------------------


class _FakeMLP(nn.Module):
    def __init__(self, hidden_size=64, intermediate_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()


def test_quant_glm5_mlp_forward_preserves_shape():
    qmlp = QuantGLM5MLP(_quant_args(quant_target=["mlp"]), _FakeMLP())
    x = torch.randn(1, 4, 64)
    assert qmlp(x).shape == x.shape


def test_quant_glm5_mlp_uses_quant_linear():
    qmlp = QuantGLM5MLP(_quant_args(quant_target=["mlp"]), _FakeMLP())
    assert isinstance(qmlp.up_proj, QuantLinear)
    assert isinstance(qmlp.gate_proj, QuantLinear)
    assert isinstance(qmlp.down_proj, QuantLinear)


# ---- QuantGLM5Attn (constructor only) ------------------------------------


def test_quant_glm5_attn_attn_linear_branch_uses_quant_linear():
    cfg = _tiny_config()
    attn = GlmMoeDsaAttention(cfg, layer_idx=0)
    qattn = QuantGLM5Attn(_quant_args(quant_target=["attn-linear"]), attn)
    # With q_lora_rank set the LoRA path uses q_a/q_b instead of q_proj.
    assert not hasattr(qattn, "q_proj")
    assert isinstance(qattn.q_a_proj, QuantLinear)
    assert isinstance(qattn.q_b_proj, QuantLinear)
    assert isinstance(qattn.kv_a_proj_with_mqa, QuantLinear)
    assert isinstance(qattn.kv_b_proj, QuantLinear)
    assert isinstance(qattn.o_proj, QuantLinear)
    assert isinstance(qattn.indexer, QuantGLM5Indexer)


def test_quant_glm5_attn_without_attn_linear_still_uses_quant_linear_for_projections():
    """GLM5 always wraps projections in QuantLinear regardless of attn-linear flag.
    The flag only controls structure transforms and activation quantizers."""
    cfg = _tiny_config()
    attn = GlmMoeDsaAttention(cfg, layer_idx=0)
    qattn = QuantGLM5Attn(_quant_args(quant_target=["attn-cache"]), attn)
    assert isinstance(qattn.q_a_proj, QuantLinear)
    assert isinstance(qattn.kv_a_proj_with_mqa, QuantLinear)
    assert isinstance(qattn.kv_b_proj, QuantLinear)
    assert isinstance(qattn.o_proj, QuantLinear)
    # Without attn-linear, transforms/afq stay disabled.
    assert qattn.input_transform is None
    assert qattn.out_transform is None
    assert isinstance(qattn.inp_afq, nn.Identity)
    assert isinstance(qattn.o_proj_afq, nn.Identity)


# ---- QuantGLM5Indexer ----------------------------------------------------


def test_quant_glm5_indexer_constructor_wraps_wq_b_with_quant_linear():
    cfg = _tiny_config()
    attn = GlmMoeDsaAttention(cfg, layer_idx=0)
    indexer = QuantGLM5Indexer(_quant_args(), attn.indexer)
    assert isinstance(indexer.wq_b, QuantLinear)
    # Other modules are reused by reference (not wrapped).
    assert indexer.wk is attn.indexer.wk
    assert indexer.k_norm is attn.indexer.k_norm
    assert indexer.weights_proj is attn.indexer.weights_proj


def test_quant_glm5_indexer_records_geometry_from_module():
    cfg = _tiny_config()
    attn = GlmMoeDsaAttention(cfg, layer_idx=0)
    indexer = QuantGLM5Indexer(_quant_args(), attn.indexer)
    assert indexer.hidden_size == cfg.hidden_size
    assert indexer.n_heads == cfg.index_n_heads
    assert indexer.head_dim == cfg.index_head_dim
    assert indexer.qk_rope_head_dim == cfg.qk_rope_head_dim
    assert indexer.index_topk == cfg.index_topk
    assert indexer.q_lora_rank == cfg.q_lora_rank
    assert indexer._cached_keys is None
