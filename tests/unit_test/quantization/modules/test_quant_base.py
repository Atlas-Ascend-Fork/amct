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
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from amct_pytorch.algorithms.registry_factory import ALGO_REGISTRY
from amct_pytorch.quantization.dtypes import register_dtype
from amct_pytorch.quantization.modules.quant_base import (
    ActivationQuantizer,
    WeightQuantizer,
    build_algorithms_by_target,
    get_algo_names_by_target,
    set_act_quantizer_state,
    set_quantizer_state,
    set_weight_quantizer_state,
)

# DTYPE_REGISTRY entries are registered lazily — pull them in once at import.
register_dtype()


UT_OBSERVE_ALGO = '_ut_observe'
UT_DOUBLE_ALGO = '_ut_double'
UT_QUANT_HOOK_ALGO = '_ut_quant_hook'
UT_QH_A_ALGO = '_ut_qh_a'
UT_QH_EXPORT_ALGO = '_ut_qh_export'

UT_QH_B = '_ut_qh_b'


def _args(algos=(), quant_dtype="int", w_bits=8, quant_target=()):
    return SimpleNamespace(
        algos=list(algos),
        quant_dtype=quant_dtype,
        w_bits=w_bits,
        quant_target=list(quant_target),
    )


# ---- get_algo_names_by_target / build_algorithms_by_target ---------------


@pytest.fixture
def _ephemeral_algo():
    """Register a temporary algorithm with explicit targets; remove after test."""
    name = "_ut_lwc_like"

    @ALGO_REGISTRY.register(name=name, targets=("weight", "activation"))
    class _Algo(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.args = args

        def forward(self, x):
            return x * 2

    yield name
    # Hard-clean the registry to keep tests isolated.
    ALGO_REGISTRY._items.pop(name, None)


def test_get_algo_names_filters_by_target(_ephemeral_algo):
    args = _args(algos=[_ephemeral_algo])
    assert get_algo_names_by_target(args, "weight") == [_ephemeral_algo]
    # Activation also valid.
    assert get_algo_names_by_target(args, "activation") == [_ephemeral_algo]
    # Structure not in declared targets.
    assert not get_algo_names_by_target(args, "structure")


def test_get_algo_names_raises_on_algo_without_targets():
    name = "_ut_no_targets"
    ALGO_REGISTRY.register(name=name)(type("T", (), {}))
    try:
        with pytest.raises(ValueError, match="missing registry metadata 'targets'"):
            get_algo_names_by_target(_args(algos=[name]), "weight")
    finally:
        ALGO_REGISTRY._items.pop(name, None)


def test_build_algorithms_returns_module_dict_for_non_structure(_ephemeral_algo):
    out = build_algorithms_by_target(_args(algos=[_ephemeral_algo]), "weight")
    assert isinstance(out, nn.ModuleDict)
    assert _ephemeral_algo in out


def test_build_algorithms_structure_returns_none_when_no_match(_ephemeral_algo):
    out = build_algorithms_by_target(_args(algos=[_ephemeral_algo]), "structure")
    assert out is None


def test_build_algorithms_structure_returns_single_algorithm():
    name = "_ut_struct_one"

    @ALGO_REGISTRY.register(name=name, targets=("structure",))
    class _Algo(nn.Module):
        def __init__(self, args, ctx):
            super().__init__()
    try:
        out = build_algorithms_by_target(
            _args(algos=[name]), "structure", SimpleNamespace()
        )
        assert isinstance(out, _Algo)
    finally:
        ALGO_REGISTRY._items.pop(name, None)


def test_build_algorithms_structure_raises_on_multiple_matches():
    n1, n2 = "_ut_struct_a", "_ut_struct_b"
    for n in (n1, n2):
        @ALGO_REGISTRY.register(name=n, targets=("structure",))
        class _Algo(nn.Module):
            def __init__(self, args, ctx):
                super().__init__()
    try:
        with pytest.raises(ValueError, match="Only one 'structure' algorithm"):
            build_algorithms_by_target(
                _args(algos=[n1, n2]), "structure", SimpleNamespace()
            )
    finally:
        ALGO_REGISTRY._items.pop(n1, None)
        ALGO_REGISTRY._items.pop(n2, None)


# ---- ActivationQuantizer / WeightQuantizer state toggles -----------------


def _model_with_quantizers():
    m = nn.Module()
    m.act = ActivationQuantizer(_args(), bits=8)
    m.weight = WeightQuantizer(_args(), w_bits=8)
    m.linear = nn.Linear(4, 4)
    return m


@pytest.mark.parametrize("flag", [True, False])
def test_set_quantizer_state_toggles_both_kinds(flag):
    m = _model_with_quantizers()
    set_quantizer_state(m, enable=flag)
    assert m.act.enable is flag
    assert m.weight.enable is flag


def test_set_weight_quantizer_state_only_touches_weight():
    m = _model_with_quantizers()
    set_weight_quantizer_state(m, enable=True)
    assert m.weight.enable is True
    assert m.act.enable is False


def test_set_act_quantizer_state_only_touches_activation():
    m = _model_with_quantizers()
    set_act_quantizer_state(m, enable=True)
    assert m.act.enable is True
    assert m.weight.enable is False


# ---- ActivationQuantizer behavior ---------------------------------------


def test_activation_quantizer_forward_passthrough_when_disabled():
    aq = ActivationQuantizer(_args(), bits=8)
    x = torch.randn(2, 32)
    assert torch.equal(aq(x), x)


def test_activation_quantizer_forward_quantizes_when_enabled():
    aq = ActivationQuantizer(_args(), bits=8)
    aq.enable = True
    x = torch.randn(2, 32, dtype=torch.float32)
    out = aq(x)
    # Same shape and dtype (int dtype quantizer is fake-quant).
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_activation_quantizer_trainable_params_collects_from_algorithms(_ephemeral_algo):
    aq = ActivationQuantizer(_args(algos=[_ephemeral_algo]), bits=8)
    # The ephemeral algo has no `trainable_params` -> empty.
    assert not aq.trainable_params()


def test_activation_quantizer_deploy_hooks_are_no_ops():
    aq = ActivationQuantizer(_args(), bits=8)
    assert aq.deploy() is None
    assert aq.load_deploy(scale=1.0, zero=0.0) is None


# ---- WeightQuantizer behavior --------------------------------------------


def test_weight_quantizer_forward_passthrough_when_disabled():
    wq = WeightQuantizer(_args(w_bits=8), w_bits=8)
    w = torch.randn(4, 8)
    assert torch.equal(wq(w), w)


def test_weight_quantizer_forward_quantizes_when_enabled():
    wq = WeightQuantizer(_args(w_bits=8), w_bits=8)
    wq.enable = True
    w = torch.randn(4, 8, dtype=torch.float32)
    out = wq(w)
    assert out.shape == w.shape


def test_weight_quantizer_observe_input_dispatches_to_algorithms_with_hook():
    seen = []

    @ALGO_REGISTRY.register(name=UT_OBSERVE_ALGO, targets=("weight",))
    class _Obs(nn.Module):
        def __init__(self, args, *_):
            super().__init__()

        def observe_input(self, x, weight):
            seen.append((x, weight))

    try:
        wq = WeightQuantizer(_args(algos=[UT_OBSERVE_ALGO], w_bits=8), w_bits=8)
        x = torch.zeros(1, 4)
        w = torch.ones(4, 4)
        wq.observe_input(x, w)
        assert len(seen) == 1
        assert torch.equal(seen[0][0], x) and torch.equal(seen[0][1], w)
    finally:
        ALGO_REGISTRY._items.pop(UT_OBSERVE_ALGO, None)


def test_weight_quantizer_algo_forward_chains_non_quantize_algos():
    @ALGO_REGISTRY.register(name=UT_DOUBLE_ALGO, targets=("weight",))
    class _Double(nn.Module):
        def __init__(self, args, *_):
            super().__init__()

        def forward(self, x):
            return x * 2

    try:
        wq = WeightQuantizer(_args(algos=[UT_DOUBLE_ALGO], w_bits=8), w_bits=8)
        out, qa = wq.algo_forward(torch.ones(1, 4))
        assert qa is None
        assert torch.equal(out, torch.full((1, 4), 2.0))
    finally:
        ALGO_REGISTRY._items.pop(UT_DOUBLE_ALGO, None)


def test_weight_quantizer_algo_forward_picks_quantize_hook_separately():
    @ALGO_REGISTRY.register(name=UT_QUANT_HOOK_ALGO, targets=("weight",))
    class _Q(nn.Module):
        def __init__(self, args, *_):
            super().__init__()

        def quantize(self, x, quant_obj):
            return x * 0

    try:
        wq = WeightQuantizer(_args(algos=[UT_QUANT_HOOK_ALGO], w_bits=8), w_bits=8)
        x = torch.ones(1, 4)
        out, qa = wq.algo_forward(x)
        assert torch.equal(out, x)        # passthrough — quantize hook is deferred
        assert isinstance(qa, _Q)
    finally:
        ALGO_REGISTRY._items.pop(UT_QUANT_HOOK_ALGO, None)


def test_weight_quantizer_algo_forward_rejects_multiple_quantize_hooks():
    for n in (UT_QH_A_ALGO, UT_QH_B):
        @ALGO_REGISTRY.register(name=n, targets=("weight",))
        class _Q(nn.Module):
            def __init__(self, args, *_):
                super().__init__()

            def quantize(self, x, q):
                return x

    try:
        wq = WeightQuantizer(_args(algos=[UT_QH_A_ALGO, UT_QH_B], w_bits=8), w_bits=8)
        with pytest.raises(ValueError, match="Only one weight algorithm"):
            wq.algo_forward(torch.zeros(1, 4))
    finally:
        for n in (UT_QH_A_ALGO, UT_QH_B):
            ALGO_REGISTRY._items.pop(n, None)


def test_weight_quantizer_export_deploy_uses_quant_obj_export():
    wq = WeightQuantizer(_args(w_bits=8), w_bits=8)
    out = wq.export_deploy(torch.randn(4, 8))
    # int dtype's export_deploy returns dict with qweight/weight_scale.
    assert "qweight" in out and "weight_scale" in out


def test_weight_quantizer_export_deploy_rejects_quantize_hook_path():
    @ALGO_REGISTRY.register(name=UT_QH_EXPORT_ALGO, targets=("weight",))
    class _Q(nn.Module):
        def __init__(self, args, *_):
            super().__init__()

        def quantize(self, x, q):
            return x
    try:
        wq = WeightQuantizer(_args(algos=[UT_QH_EXPORT_ALGO], w_bits=8), w_bits=8)
        with pytest.raises(NotImplementedError, match="custom weight quantize"):
            wq.export_deploy(torch.zeros(4, 8))
    finally:
        ALGO_REGISTRY._items.pop(UT_QH_EXPORT_ALGO, None)


def test_build_algorithms_raises_when_algo_declares_targets_but_mismatches():

    name = "_ut_struct_mis"

    @ALGO_REGISTRY.register(name=name, targets=("weight",))
    class _Algo(nn.Module):
        def __init__(self, args, ctx=None):
            super().__init__()
    try:
        out = build_algorithms_by_target(_args(algos=[name]), "activation")
        assert isinstance(out, nn.ModuleDict)
        assert len(out) == 0
    finally:
        ALGO_REGISTRY._items.pop(name, None)


def test_activation_quantizer_trainable_params_returns_params_from_algo():

    name = "_ut_act_tp"

    @ALGO_REGISTRY.register(name=name, targets=("activation",))
    class _AlgoWithParams(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.p = nn.Parameter(torch.tensor(1.0))

        def forward(self, x):
            return x

        def trainable_params(self):
            return [self.p]
    try:
        aq = ActivationQuantizer(_args(algos=[name]), bits=8)
        params = aq.trainable_params()
        assert len(params) == 1
    finally:
        ALGO_REGISTRY._items.pop(name, None)


def test_activation_quantizer_forward_applies_algo_when_enabled():

    name = "_ut_act_fwd"

    @ALGO_REGISTRY.register(name=name, targets=("activation",))
    class _DoubleAlgo(nn.Module):
        def __init__(self, args):
            super().__init__()

        def forward(self, x):
            return x * 2
    try:
        aq = ActivationQuantizer(_args(algos=[name]), bits=8)
        aq.enable = True
        x = torch.tensor([1.0, 2.0, 3.0])
        out = aq(x)
        assert out.dtype == x.dtype
        assert out.shape == x.shape
    finally:
        ALGO_REGISTRY._items.pop(name, None)


def test_weight_quantizer_trainable_params_returns_params_from_algo():

    name = "_ut_wt_tp"

    @ALGO_REGISTRY.register(name=name, targets=("weight",))
    class _WtAlgoWithParams(nn.Module):
        def __init__(self, args, *_):
            super().__init__()
            self.p = nn.Parameter(torch.tensor(2.0))

        def forward(self, x):
            return x

        def trainable_params(self):
            return [self.p]
    try:
        wq = WeightQuantizer(_args(algos=[name], w_bits=8), w_bits=8)
        params = wq.trainable_params()
        assert len(params) == 1
    finally:
        ALGO_REGISTRY._items.pop(name, None)


def test_weight_quantizer_forward_uses_quantize_algo_when_enabled():

    name = "_ut_wt_qalgo"

    @ALGO_REGISTRY.register(name=name, targets=("weight",))
    class _QAlgo(nn.Module):
        def __init__(self, args, *_):
            super().__init__()

        def quantize(self, x, quant_obj):
            return x * 100

    try:
        wq = WeightQuantizer(_args(algos=[name], w_bits=8), w_bits=8)
        wq.enable = True
        x = torch.tensor([1.0, 2.0])
        out = wq(x)
        assert torch.equal(out, x * 100)
    finally:
        ALGO_REGISTRY._items.pop(name, None)


def test_weight_quantizer_export_deploy_rejects_unsupported_dtype(monkeypatch):
    wq = WeightQuantizer(_args(w_bits=8), w_bits=8)
    wq.quant_obj.export_deploy = None
    with pytest.raises(NotImplementedError, match="does not implement export_deploy"):
        wq.export_deploy(torch.randn(4, 8))


def test_build_algorithms_raises_with_missing_targets():
    name = "_ut_missing_targets"
    ALGO_REGISTRY.register(name=name)(type("T", (), {}))
    try:
        with pytest.raises(ValueError, match="missing registry metadata"):
            build_algorithms_by_target(_args(algos=[name]), "weight")
    finally:
        ALGO_REGISTRY._items.pop(name, None)


def test_build_algorithms_raises_when_target_not_in_algo_targets():

    name = "_ut_struct_nonmatch"

    @ALGO_REGISTRY.register(name=name, targets=("weight",))
    class _Algo(nn.Module):
        def __init__(self, args, ctx=None):
            super().__init__()
    try:
        out = build_algorithms_by_target(_args(algos=[name]), "activation")
        assert isinstance(out, nn.ModuleDict)
        assert len(out) == 0
    finally:
        ALGO_REGISTRY._items.pop(name, None)


def test_build_algorithms_by_target_raises_on_missing_targets_metadata(monkeypatch):
    from types import SimpleNamespace as simple_ns

    from amct_pytorch.algorithms.registry_factory import ALGO_REGISTRY as algo_registry
    from amct_pytorch.quantization.modules import quant_base as quant_base_mod
    monkeypatch.setattr(
        quant_base_mod, "get_algo_names_by_target",
        lambda args, target: ["fake_algo"],
    )
    monkeypatch.setattr(
        algo_registry, "get_item",
        lambda name: simple_ns(metadata={}, target=lambda *a: None),
    )
    args = SimpleNamespace()
    with pytest.raises(ValueError, match="missing registry metadata"):
        quant_base_mod.build_algorithms_by_target(args, "mlp")


def test_build_algorithms_by_target_raises_on_mismatched_target(monkeypatch):
    from types import SimpleNamespace as simple_ns

    from amct_pytorch.algorithms.registry_factory import ALGO_REGISTRY as algo_registry
    from amct_pytorch.quantization.modules import quant_base as quant_base_mod
    monkeypatch.setattr(
        quant_base_mod, "get_algo_names_by_target",
        lambda args, target: ["fake_algo"],
    )
    monkeypatch.setattr(
        algo_registry, "get_item",
        lambda name: simple_ns(metadata={"targets": ("attn",)}, target=lambda *a: None),
    )
    args = SimpleNamespace()
    with pytest.raises(ValueError, match="cannot be used for target"):
        quant_base_mod.build_algorithms_by_target(args, "mlp")

