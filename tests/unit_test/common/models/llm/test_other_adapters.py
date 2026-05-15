"""Lightweight adapter tests for  glm5 / longcat / qwen3.6 families."""

from types import SimpleNamespace

import pytest
import torch.nn as nn

from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm.common.ptq_units import make_ptq_unit
from amct_pytorch.common.models.llm.glm.glm5.glm5 import GLM5
from amct_pytorch.common.models.llm.longcat.longcat_lite.longcat_lite import LongcatLite
from amct_pytorch.common.models.llm.longcat.longcat_next.longcat_next import LongcatNext
from amct_pytorch.common.models.llm.qwen.qwen3_6.qwen3_6_moe import Qwen3_6Moe
from amct_pytorch.quantization.dtypes import register_dtype
from amct_pytorch.quantization.modules.quant_linear import QuantLinear

register_dtype()

DENSE = 'dense'

def _stub(adapter_cls, **attrs):
    obj = adapter_cls.__new__(adapter_cls)
    obj.args = SimpleNamespace(quant_target=attrs.get("quant_target", []))
    obj.quant_target = attrs.get("quant_target", [])
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


# ---- LongcatLite / LongcatNext ------------------------------------------


def test_longcat_lite_layer_weight_prefix():
    obj = _stub(LongcatLite, quant_target=["moe"])
    assert obj.get_layer_weight_prefix(0) == "model.layers.0."


def test_longcat_next_layer_weight_prefix_delegates_to_super():
    # LongcatNext inherits LongcatLite's prefix scheme via super().
    obj = _stub(LongcatNext, quant_target=["moe"])
    assert obj.get_layer_weight_prefix(2) == "model.layers.2."


def test_longcat_lite_set_safe_attn_impl_writes_eager():
    cfg = SimpleNamespace()
    obj = _stub(LongcatLite, config=cfg)
    obj._set_safe_attn_impl()
    assert cfg._attn_implementation == "eager"


# ---- GLM5 ---------------------------------------------------------------


def test_glm5_layer_weight_prefix():
    obj = _stub(GLM5, quant_target=[])
    assert obj.get_layer_weight_prefix(4) == "model.layers.4."


@pytest.mark.parametrize(
    "first_dense,moe_freq,layer_idx,expected",
    [
        (3, 1, 0, False),    # before first_dense
        (3, 1, 2, False),
        (3, 1, 3, True),     # at first_dense
        (3, 1, 5, True),
        (3, 2, 3, True),     # frequency 2 -> hits at offsets 0, 2, 4 ...
        (3, 2, 4, False),
        (3, 2, 5, True),
        (0, 1, 0, True),     # no dense prefix -> all layers are moe
    ],
)
def test_glm5_is_moe_layer_uses_first_k_dense_and_freq(first_dense, moe_freq, layer_idx, expected):
    obj = _stub(
        GLM5,
        quant_target=[],
        config=SimpleNamespace(first_k_dense_replace=first_dense, moe_layer_freq=moe_freq),
    )
    assert obj._is_moe_layer(layer_idx) is expected


def test_glm5_ensure_config_writes_mlp_layer_types_when_missing():
    obj = _stub(
        GLM5,
        quant_target=[],
        config=SimpleNamespace(num_hidden_layers=6, first_k_dense_replace=2, moe_layer_freq=2, mlp_layer_types=None),
    )
    obj._ensure_glm5_config()
    assert obj.config.mlp_layer_types == [DENSE, DENSE, "sparse", DENSE, "sparse", DENSE]


def test_glm5_ensure_config_no_op_when_mlp_layer_types_already_set():
    existing = [DENSE, "sparse"]
    obj = _stub(
        GLM5,
        quant_target=[],
        config=SimpleNamespace(
            num_hidden_layers=2, first_k_dense_replace=1,
            moe_layer_freq=1, mlp_layer_types=existing),
    )
    obj._ensure_glm5_config()
    assert obj.config.mlp_layer_types is existing


# ---- GLM5.iter_ptq_units (key dispatch logic) ----------------------------


def test_glm5_iter_ptq_units_yields_self_attn_when_attn_target():
    block = nn.Module()
    block.self_attn = nn.Linear(4, 4)
    obj = _stub(GLM5, quant_target=["attn-linear"])
    units = list(obj.iter_ptq_units(layer_idx=1, block=block))
    assert len(units) == 1
    assert units[0].name == "self_attn"
    assert units[0].module is block.self_attn


def test_glm5_iter_ptq_units_falls_back_to_super_when_self_attn_missing():
    block = nn.Module()
    block.mlp = nn.Linear(4, 4)
    obj = _stub(GLM5, quant_target=["attn-linear"])
    # super().iter_ptq_units with target attn-linear would still try to read
    # `self_attn` and yield None — verify we degrade gracefully.
    units = list(obj.iter_ptq_units(layer_idx=0, block=block))
    # With no self_attn at all, the super fallback yields a unit with module=None.
    assert len(units) == 1
    assert units[0].module is None


def test_glm5_iter_ptq_units_raises_when_no_mlp_for_unknown_target():
    block = nn.Module()
    obj = _stub(GLM5, quant_target=["nonsense"])
    with pytest.raises(ValueError, match="Unsupported quant target"):
        list(obj.iter_ptq_units(layer_idx=0, block=block))


def test_glm5_iter_ptq_units_yields_shared_experts_when_mlp_target_and_present():
    block = nn.Module()
    block.mlp = nn.Module()
    block.mlp.shared_experts = nn.Linear(4, 4)
    obj = _stub(GLM5, quant_target=["mlp"])
    units = list(obj.iter_ptq_units(layer_idx=2, block=block))
    assert len(units) == 1
    assert units[0].name == "shared_experts"


def test_glm5_iter_ptq_units_yields_plain_mlp_when_no_shared_experts():
    block = nn.Module()
    block.mlp = nn.Linear(4, 4)
    obj = _stub(GLM5, quant_target=["mlp"])
    units = list(obj.iter_ptq_units(layer_idx=0, block=block))
    assert units[0].name == "mlp"
    assert units[0].module is block.mlp


# ---- GLM5.iter_deploy_bindings expert path rewriting ---------------------


def _make_block_with_quant_linears_for_glm5():
    args = SimpleNamespace(algos=[], quant_dtype="int", w_bits=8)
    block = nn.Module()
    block.self_attn = nn.Module()
    block.self_attn.q_proj = QuantLinear(args, nn.Linear(4, 4), w_bits=args.w_bits, name="q_proj")

    block.mlp = nn.Module()
    block.mlp.experts = nn.Module()
    block.mlp.experts.expert_modules = nn.ModuleList()
    for _ in range(2):
        e = nn.Module()
        e.up_proj = QuantLinear(args, nn.Linear(4, 4), w_bits=args.w_bits, name="up_proj")
        block.mlp.experts.expert_modules.append(e)
    return block


def test_glm5_iter_deploy_bindings_rewrites_expert_path():
    obj = _stub(GLM5, quant_target=["moe"])
    block = _make_block_with_quant_linears_for_glm5()
    names = [w for w, _ in obj.iter_deploy_bindings(layer_idx=4, block=block)]
    assert "model.layers.4.self_attn.q_proj.weight" in names
    assert "model.layers.4.mlp.experts.0.up_proj.weight" in names
    assert "model.layers.4.mlp.experts.1.up_proj.weight" in names
    assert all("expert_modules" not in n for n in names)


# ---- Qwen3.6Moe is essentially a stub class ------------------------------


def test_qwen3_6_moe_registers_in_model_registry():
    assert MODEL_REGISTRY.get("qwen3_6_moe") is Qwen3_6Moe


# ---- registry coverage check --------------------------------------------


@pytest.mark.parametrize(
    "key,expected_cls",
    [
        ("longcat_lite", LongcatLite),
        ("glm5", GLM5),
    ],
)
def test_other_adapters_register_under_expected_keys(key, expected_cls):
    assert MODEL_REGISTRY.get(key) is expected_cls

