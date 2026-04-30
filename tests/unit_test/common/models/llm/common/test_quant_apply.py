import argparse

import pytest
import torch
import torch.nn as nn

from amct_pytorch.common.models.llm.common.quant_apply import (
    PlainLinear,
    apply_quant_to_attn,
    apply_quant_to_moe_mlp,
    build_direct_quant_args,
    set_model_act_quant_state,
    set_model_to_observe,
    set_model_weight_quant_state,
)
from amct_pytorch.quantization.modules.quant_base import (
    ActivationQuantizer,
    WeightQuantizer,
)


# ---- PlainLinear ---------------------------------------------------------


def test_plain_linear_forward_matches_inner_linear():
    inner = nn.Linear(4, 6)
    wrap = PlainLinear(inner)
    x = torch.randn(2, 4)
    assert torch.equal(wrap(x), inner(x))


def test_plain_linear_ignores_structure_transform_kwarg():
    inner = nn.Linear(4, 6)
    wrap = PlainLinear(inner)
    x = torch.randn(2, 4)
    out = wrap(x, structure_transform=lambda t: t * 0)
    assert torch.equal(out, inner(x))


# ---- set_model_*_quant_state ---------------------------------------------


class _StubActQuantizer(ActivationQuantizer):
    def __init__(self):
        nn.Module.__init__(self)
        self.enable = False
        self.is_observe = True


class _StubWeightQuantizer(WeightQuantizer):
    def __init__(self):
        nn.Module.__init__(self)
        self.enable = False


def _model_with_quantizers():
    m = nn.Module()
    m.act_a = _StubActQuantizer()
    m.act_b = _StubActQuantizer()
    m.wq = _StubWeightQuantizer()
    m.linear = nn.Linear(4, 4)  # should be ignored
    return m


@pytest.mark.parametrize("flag", [True, False])
def test_set_model_act_quant_state_toggles_only_activation_modules(flag):
    m = _model_with_quantizers()
    set_model_act_quant_state(m, flag)
    assert m.act_a.enable is flag
    assert m.act_b.enable is flag
    assert m.act_a.is_observe is (not flag)
    assert m.act_b.is_observe is (not flag)
    # Non-activation modules untouched.
    assert m.wq.enable is False


@pytest.mark.parametrize("flag", [True, False])
def test_set_model_weight_quant_state_toggles_only_weight_quantizer(flag):
    m = _model_with_quantizers()
    set_model_weight_quant_state(m, flag)
    assert m.wq.enable is flag
    # Activation flags should not be touched.
    assert m.act_a.enable is False


@pytest.mark.parametrize("flag", [True, False])
def test_set_model_to_observe_only_targets_modules_with_attribute(flag):
    m = _model_with_quantizers()
    set_model_to_observe(m, flag)
    assert m.act_a.is_observe is flag
    assert m.act_b.is_observe is flag
    # weight quantizer has no `is_observe` attribute and must remain unchanged.
    assert not hasattr(m.wq, "is_observe")


# ---- build_direct_quant_args ---------------------------------------------


def test_build_direct_quant_args_returns_independent_copy_with_overrides():
    args = argparse.Namespace(w_bits=4, a_bits=4, algos=["awq", "smooth"], extra="keep")
    new = build_direct_quant_args(args, bits=8)
    assert new is not args
    assert new.w_bits == 8 and new.a_bits == 8
    assert new.algos == []
    assert new.extra == "keep"
    # Source args should not be mutated.
    assert args.w_bits == 4 and args.algos == ["awq", "smooth"]


# ---- apply_quant_to_attn / apply_quant_to_moe_mlp -------------------------


class _FakeQuantWrapper(nn.Module):
    """Records the wrapped child so we can assert replacement behavior."""
    def __init__(self, args, original):
        super().__init__()
        self.args = args
        self.original = original


def _decoder_layer():
    layer = nn.Module()
    layer.self_attn = nn.Linear(8, 8)
    layer.mlp = nn.Linear(8, 8)
    return layer


def test_apply_quant_to_attn_replaces_self_attn_child():
    layer = _decoder_layer()
    original_attn = layer.self_attn

    apply_quant_to_attn(args=object(), model=layer, cls=_FakeQuantWrapper)

    assert isinstance(layer.self_attn, _FakeQuantWrapper)
    assert layer.self_attn.original is original_attn
    # Sibling MLP should remain untouched.
    assert isinstance(layer.mlp, nn.Linear)


def test_apply_quant_to_attn_recurses_into_grandchildren():
    outer = nn.Module()
    outer.block = _decoder_layer()
    apply_quant_to_attn(args=object(), model=outer, cls=_FakeQuantWrapper)
    assert isinstance(outer.block.self_attn, _FakeQuantWrapper)


def test_apply_quant_to_moe_mlp_wraps_dense_mlp():
    layer = nn.Module()
    layer.mlp = nn.Linear(8, 8)  # dense mlp without `experts`
    original_mlp = layer.mlp
    apply_quant_to_moe_mlp(args=object(), model=layer, cls=_FakeQuantWrapper)
    assert isinstance(layer.mlp, _FakeQuantWrapper)
    assert layer.mlp.original is original_mlp


def test_apply_quant_to_moe_mlp_does_not_wrap_when_mlp_has_experts():
    layer = nn.Module()
    layer.mlp = nn.Module()
    layer.mlp.experts = nn.ModuleList([nn.Linear(4, 4)])
    original_mlp = layer.mlp
    apply_quant_to_moe_mlp(args=object(), model=layer, cls=_FakeQuantWrapper)
    # The MLP itself stays the same — experts are wrapped in-place instead.
    assert layer.mlp is original_mlp
    assert isinstance(layer.mlp.experts[0], _FakeQuantWrapper)


def test_apply_quant_to_moe_mlp_skips_none_experts():
    layer = nn.Module()
    layer.experts = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
    layer.experts[1] = None  # sparse expert slot
    apply_quant_to_moe_mlp(args=object(), model=layer, cls=_FakeQuantWrapper)
    assert isinstance(layer.experts[0], _FakeQuantWrapper)
    assert layer.experts[1] is None
