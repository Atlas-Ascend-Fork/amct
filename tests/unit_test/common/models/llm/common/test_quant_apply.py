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

import argparse
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from amct_pytorch.common.models.llm.common.quant_apply import (
    PlainLinear,
    QuantGatedMLP,
    apply_quant_to_attn,
    apply_quant_to_moe_mlp,
    set_model_act_quant_state,
    set_model_to_observe,
    set_model_weight_quant_state,
)
from amct_pytorch.quantization.bit_policy import BitPolicy
from amct_pytorch.quantization.dtypes import register_dtype
from amct_pytorch.quantization.modules.quant_base import (
    ActivationQuantizer,
    WeightQuantizer,
)

_MLP_BIT_POLICY = BitPolicy({
    "mlp": {
        "gate_proj": {"w_bits": 8, "a_bits": 8},
        "up_proj": {"w_bits": 8, "a_bits": 8},
        "down_proj": {"w_bits": 8, "a_bits": 8},
    },
    "moe": {
        "routed": {
            "gate_proj": {"w_bits": 8, "a_bits": 8},
            "up_proj": {"w_bits": 8, "a_bits": 8},
            "down_proj": {"w_bits": 8, "a_bits": 8},
        },
        "shared": {
            "gate_proj": {"w_bits": 8, "a_bits": 8},
            "up_proj": {"w_bits": 8, "a_bits": 8},
            "down_proj": {"w_bits": 8, "a_bits": 8},
        },
    },
    "attn-linear": {
        "q_proj": {"w_bits": 8, "a_bits": 8},
        "k_proj": {"w_bits": 8, "a_bits": 8},
        "v_proj": {"w_bits": 8, "a_bits": 8},
        "o_proj": {"w_bits": 8, "a_bits": 8},
    },
})

register_dtype()


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


_MOCK_QUANT_ARGS = SimpleNamespace(w_bits=8, quant_dtype="int", algos=[])


class _StubActQuantizer(ActivationQuantizer):
    def __init__(self):
        super().__init__(_MOCK_QUANT_ARGS, bits=8)
        self.enable = False
        self.is_observe = True


class _StubWeightQuantizer(WeightQuantizer):
    def __init__(self):
        super().__init__(_MOCK_QUANT_ARGS)
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


# ---- apply_quant_to_attn / apply_quant_to_moe_mlp -------------------------


class _FakeQuantWrapper(nn.Module):
    """Records the wrapped child so we can assert replacement behavior."""
    def __init__(self, args, original, group=None):
        super().__init__()
        self.args = args
        self.original = original
        self.group = group


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


def test_apply_quant_to_moe_mlp_with_routed_expert_bits():
    layer = nn.Module()
    layer.mlp = nn.Module()
    layer.mlp.experts = nn.ModuleList([nn.Linear(4, 4)])
    args = argparse.Namespace(
        w_bits=8, a_bits=8, bit_policy=_MLP_BIT_POLICY,
        quant_target=["moe"], algos=[], quant_dtype="int",
    )
    apply_quant_to_moe_mlp(args=args, model=layer, cls=_FakeQuantWrapper)
    assert isinstance(layer.mlp.experts[0], _FakeQuantWrapper)
    assert layer.mlp.experts[0].group == "moe.routed"


def test_apply_quant_to_moe_mlp_with_shared_experts():
    layer = nn.Module()
    layer.shared_experts = nn.Linear(4, 4)
    args = argparse.Namespace(
        w_bits=8, a_bits=8, bit_policy=_MLP_BIT_POLICY,
        quant_target=["moe"], algos=[], quant_dtype="int",
    )
    apply_quant_to_moe_mlp(args=args, model=layer, cls=_FakeQuantWrapper)
    assert isinstance(layer.shared_experts, _FakeQuantWrapper)
    assert layer.shared_experts.group == "moe.shared"


class _FakeMLP(nn.Module):
    def __init__(self, hidden_size=8, intermediate_size=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.act_fn = nn.SiLU()
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)


def test_quant_gated_mlp_forward_uses_input_and_hidden_transform(monkeypatch):
    cmds = []

    args = argparse.Namespace(
        w_bits=8, a_bits=8, quant_dtype="int", quant_target=["mlp"], algos=[], bit_policy=_MLP_BIT_POLICY,
    )
    mlp = _FakeMLP()
    gated = QuantGatedMLP(args, mlp)

    def input_transform(x, **kw):
        cmds.append("input_xform")
        return x

    gated.input_transform = input_transform

    def hidden_transform(x, **kw):
        cmds.append("hidden_xform")
        return x

    gated.hidden_transform = hidden_transform
    gated.input_quant.enable = True
    gated.hidden_quant.enable = True

    x = torch.randn(2, 8)
    gated(x)
    assert "input_xform" in cmds
    assert "hidden_xform" in cmds


def test_quant_gated_mlp_forward_passthrough_when_quant_disabled():
    args = argparse.Namespace(
        w_bits=8, a_bits=8, quant_dtype="int", quant_target=["mlp"], algos=[], bit_policy=_MLP_BIT_POLICY,
    )
    mlp = _FakeMLP()
    gated = QuantGatedMLP(args, mlp)
    x = torch.randn(2, 8)
    y = gated(x)
    assert y.shape == (2, mlp.hidden_size)


def test_quant_gated_mlp_export_ptq_params_returns_module_params(monkeypatch):
    from amct_pytorch.common.models.llm.common.ptq_params import PtqParamHandler

    monkeypatch.setattr(PtqParamHandler, "export_module", lambda m: {"sub": {"k": 1}})
    monkeypatch.setattr(PtqParamHandler, "export_trainable_module", lambda m: {})

    args = argparse.Namespace(
        w_bits=8, a_bits=8, quant_dtype="int", quant_target=["mlp"], algos=[], bit_policy=_MLP_BIT_POLICY,
    )
    mlp = _FakeMLP()
    gated = QuantGatedMLP(args, mlp)
    result = gated.export_ptq_params()
    assert result == {"sub": {"k": 1}}
    original = False
    trainable = False

    from amct_pytorch.common.models.llm.common.ptq_params import PtqParamHandler

    monkeypatch.setattr(PtqParamHandler, "export_module", lambda m: {})
    monkeypatch.setattr(PtqParamHandler, "export_trainable_module",
                        lambda m: trainable or {"trainable": [1]})

    args = argparse.Namespace(
        w_bits=8, a_bits=8, quant_dtype="int", quant_target=["mlp"], algos=[], bit_policy=_MLP_BIT_POLICY,
    )
    mlp = _FakeMLP()
    gated = QuantGatedMLP(args, mlp)
    result = gated.export_ptq_params()
    assert result == {"trainable": [1]}


def test_quant_gated_mlp_load_ptq_params_nested_dicts(monkeypatch):
    loaded = {}
    from amct_pytorch.common.models.llm.common.ptq_params import PtqParamHandler

    monkeypatch.setattr(PtqParamHandler, "load_module", lambda m, p: loaded.update({"module": True}))
    monkeypatch.setattr(PtqParamHandler, "load_trainable_module", lambda m, p: loaded.update({"trainable": True}))

    args = argparse.Namespace(
        w_bits=8, a_bits=8, quant_dtype="int", quant_target=["mlp"], algos=[], bit_policy=_MLP_BIT_POLICY,
    )
    mlp = _FakeMLP()
    gated = QuantGatedMLP(args, mlp)
    gated.load_ptq_params({"a": {"b": "c"}, "d": {"e": "f"}})
    assert loaded.get("module") is True


def test_quant_gated_mlp_load_ptq_params_non_nested(monkeypatch):
    loaded = {}
    from amct_pytorch.common.models.llm.common.ptq_params import PtqParamHandler

    monkeypatch.setattr(PtqParamHandler, "load_module", lambda m, p: loaded.update({"module": True}))
    monkeypatch.setattr(PtqParamHandler, "load_trainable_module", lambda m, p: loaded.update({"trainable": True}))

    args = argparse.Namespace(
        w_bits=8, a_bits=8, quant_dtype="int", quant_target=["mlp"], algos=[], bit_policy=_MLP_BIT_POLICY,
    )
    mlp = _FakeMLP()
    gated = QuantGatedMLP(args, mlp)
    gated.load_ptq_params({"some_key": [1, 2, 3]})
    assert loaded.get("trainable") is True


def test_quant_gated_mlp_forward():
    args = SimpleNamespace(algos=[], quant_dtype="int", w_bits=8, a_bits=8, quant_target=["mlp"])
    mlp = QuantGatedMLP(args, _FakeMLP(hidden_size=4, intermediate_size=8))
    x = torch.randn(2, 4)
    out = mlp(x)
    assert out.shape == (2, 4)
