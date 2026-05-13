"""Lightweight adapter tests for Qwen3 / Qwen3-MoE / Qwen3.5 families.

These tests bypass `BaseModel.__init__` (which loads HF tokenizer + config) and
exercise the adapter-level overrides that are pure logic: `parse_quant_mode`,
`get_layer_weight_prefix`, `_embed_base_prefix`, and the MoE-specific
`iter_deploy_bindings` name parsing.
"""

from types import SimpleNamespace

import pytest
import torch.nn as nn

from amct_pytorch.common.models.llm.qwen.qwen3.qwen3 import Qwen3
from amct_pytorch.common.models.llm.qwen.qwen3.qwen3_moe import Qwen3Moe
from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5 import Qwen3_5
from amct_pytorch.common.models.llm.qwen.qwen3_5.qwen3_5_moe import Qwen3_5Moe
from amct_pytorch.common.models.llm.qwen.qwen3_next.qwen3_next import Qwen3Next
from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.quantization.dtypes import register_dtype
from amct_pytorch.quantization.modules.quant_linear import QuantLinear

register_dtype()


def _stub_adapter(adapter_cls, quant_target):
    """Bypass BaseModel.__init__ and seed only the attributes the adapter
    overrides actually rely on."""
    obj = adapter_cls.__new__(adapter_cls)
    obj.args = SimpleNamespace(quant_target=list(quant_target))
    obj.quant_target = list(quant_target)
    return obj


# ---- parse_quant_mode ----------------------------------------------------


def test_qwen3_dense_rejects_moe_target():
    obj = _stub_adapter(Qwen3, ["moe"])
    with pytest.raises(ValueError, match="dense model does not support"):
        obj.parse_quant_mode()


def test_qwen3_dense_accepts_attn_and_mlp_targets():
    for target in (["attn-linear"], ["attn-cache"], ["mlp"]):
        obj = _stub_adapter(Qwen3, target)
        assert obj.parse_quant_mode() is None


def test_qwen3_moe_rejects_mlp_target():
    obj = _stub_adapter(Qwen3Moe, ["mlp"])
    with pytest.raises(ValueError, match="does not support quant_target='mlp'"):
        obj.parse_quant_mode()


def test_qwen3_moe_accepts_moe_and_attn_targets():
    for target in (["moe"], ["attn-linear"], ["attn-cache"]):
        obj = _stub_adapter(Qwen3Moe, target)
        assert obj.parse_quant_mode() is None


def test_qwen3_5_dense_rejects_moe_target():
    obj = _stub_adapter(Qwen3_5, ["moe"])
    with pytest.raises(ValueError, match="dense model and does not support"):
        obj.parse_quant_mode()


def test_qwen3_5_moe_rejects_mlp_target():
    obj = _stub_adapter(Qwen3_5Moe, ["mlp"])
    with pytest.raises(ValueError, match="does not support quant_target='mlp'"):
        obj.parse_quant_mode()


def test_qwen3_next_rejects_mlp_target():
    obj = _stub_adapter(Qwen3Next, ["mlp"])
    with pytest.raises(ValueError, match="does not support quant_target='mlp'"):
        obj.parse_quant_mode()


# ---- get_layer_weight_prefix ---------------------------------------------


def test_qwen3_layer_weight_prefix_uses_model_layers():
    obj = _stub_adapter(Qwen3, ["mlp"])
    assert obj.get_layer_weight_prefix(7) == "model.layers.7."


def test_qwen3_moe_layer_weight_prefix_uses_model_layers():
    obj = _stub_adapter(Qwen3Moe, ["moe"])
    assert obj.get_layer_weight_prefix(0) == "model.layers.0."


def test_qwen3_5_uses_model_language_model_prefix():
    obj = _stub_adapter(Qwen3_5, ["mlp"])
    assert obj.get_layer_weight_prefix(2) == "model.language_model.layers.2."
    assert obj._embed_base_prefix() == "model.language_model."


def test_qwen3_next_uses_model_layers_prefix():
    obj = _stub_adapter(Qwen3Next, ["moe"])
    assert obj.get_layer_weight_prefix(5) == "model.layers.5."


# ---- registry registration -----------------------------------------------


@pytest.mark.parametrize(
    "key,expected_cls",
    [
        ("qwen3", Qwen3),
        ("qwen3_moe", Qwen3Moe),
        ("qwen3_5", Qwen3_5),
    ],
)
def test_adapters_register_under_expected_keys(key, expected_cls):
    assert MODEL_REGISTRY.get(key) is expected_cls


# ---- iter_deploy_bindings (Qwen3Moe) -------------------------------------


def _build_moe_block_with_quant_linears():
    """Build a fake decoder layer matching Qwen3Moe's iter_deploy_bindings name
    scheme (`mlp.experts.expert_modules.<idx>.<proj>`)."""
    block = nn.Module()
    block.self_attn = nn.Module()
    block.self_attn.q_proj = QuantLinear(
        SimpleNamespace(algos=[], quant_dtype="int", w_bits=8),
        nn.Linear(4, 4),
        w_bits=8,
        name="q_proj",
    )

    # MoE expert hierarchy: mlp.experts.expert_modules.{idx}.{proj_name}
    block.mlp = nn.Module()
    block.mlp.experts = nn.Module()
    block.mlp.experts.expert_modules = nn.ModuleList()
    for _ in range(2):
        expert = nn.Module()
        expert.up_proj = QuantLinear(
            SimpleNamespace(algos=[], quant_dtype="int", w_bits=8),
            nn.Linear(4, 4),
            w_bits=8,
            name="up_proj",
        )
        block.mlp.experts.expert_modules.append(expert)
    return block


def test_qwen3_moe_iter_deploy_bindings_rewrites_expert_path():
    obj = _stub_adapter(Qwen3Moe, ["moe"])
    block = _build_moe_block_with_quant_linears()
    bindings = list(obj.iter_deploy_bindings(layer_idx=3, block=block))
    names = [b[0] for b in bindings]
    # Self-attn QuantLinear stays untouched (just gets the prefix prepended).
    assert "model.layers.3.self_attn.q_proj.weight" in names
    # Each expert is rewritten to model.layers.3.mlp.experts.<idx>.up_proj.weight.
    assert "model.layers.3.mlp.experts.0.up_proj.weight" in names
    assert "model.layers.3.mlp.experts.1.up_proj.weight" in names
    # The original expert_modules.<idx> path must NOT appear.
    assert all("expert_modules" not in n for n in names)


def test_qwen3_moe_iter_deploy_bindings_raises_on_malformed_expert_name():
    """Construct a name that starts with the expert prefix but has the wrong arity."""
    from amct_pytorch.quantization.modules.quant_linear import QuantLinear

    obj = _stub_adapter(Qwen3Moe, ["moe"])
    block = nn.Module()
    block.mlp = nn.Module()
    block.mlp.experts = nn.Module()
    block.mlp.experts.expert_modules = nn.Module()
    # Single attribute `bad` -> name `mlp.experts.expert_modules.bad`, only 4 parts.
    block.mlp.experts.expert_modules.bad = QuantLinear(
        SimpleNamespace(algos=[], quant_dtype="int", w_bits=8),
        nn.Linear(4, 4),
        w_bits=8,
        name="bad",
    )
    with pytest.raises(ValueError, match="Unexpected Qwen3 MoE expert module name"):
        list(obj.iter_deploy_bindings(layer_idx=0, block=block))
