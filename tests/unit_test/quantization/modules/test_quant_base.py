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
from amct_pytorch.algorithms.quant.base import QuantAlgorithmBase
from amct_pytorch.algorithms.quant.auto_round import AutoRound  # noqa: F401
from amct_pytorch.quantization.dtypes import register_dtype
from amct_pytorch.quantization.modules.quant_base import (
    ActivationQuantizer,
    WeightQuantizer,
    build_algorithms_by_target,
    get_algo_names_by_target,
)

# DTYPE_REGISTRY entries are registered lazily — pull them in once at import.
register_dtype()


UT_OBSERVE_ALGO = '_ut_observe'
UT_DOUBLE_ALGO = '_ut_double'
UT_QUANT_HOOK_ALGO = '_ut_quant_hook'
UT_QH_A_ALGO = '_ut_qh_a'
UT_QH_EXPORT_ALGO = '_ut_qh_export'

UT_QH_B = '_ut_qh_b'


def _args(algos=(), quant_dtype="int", w_bits=8, quant_target=(), w_size=(4, 8)):
    return SimpleNamespace(
        algos=list(algos),
        quant_dtype=quant_dtype,
        w_bits=w_bits,
        quant_target=list(quant_target),
        w_size=w_size,
    )


class _ObserveActivationAlgo(QuantAlgorithmBase):
    def __init__(self, args):
        super().__init__()
        self.calib_call_count = 0
        self.quant_call_count = 0
        self.received = None

    def forward(self, x):
        self.quant_call_count += 1
        self.received = x
        return x

    def calib_forward(self, x, *args, **kwargs):
        self.calib_call_count += 1
        self.received = x
        return x


class _RegularActivationAlgo(QuantAlgorithmBase):
    def __init__(self, args):
        super().__init__()
        self.call_count = 0

    def forward(self, x):
        self.call_count += 1
        return x + 1


class _FakeQuantSpy(nn.Module):
    def __init__(self):
        super().__init__()
        self.call_count = 0
        self.received = None

    def forward(self, x):
        self.call_count += 1
        self.received = x
        return x - 3


class _PassthroughAlgorithm(QuantAlgorithmBase):
    def __init__(self, args=None, *ctor_args):
        super().__init__()

    def forward(self, x):
        return x


# ---- get_algo_names_by_target / build_algorithms_by_target ---------------


@pytest.fixture
def _ephemeral_algo():
    """Register a temporary algorithm with explicit targets; remove after test."""
    name = "_ut_lwc_like"

    @ALGO_REGISTRY.register(name=name, targets=("weight", "activation"))
    class _Algo(QuantAlgorithmBase):
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
    ALGO_REGISTRY.register(name=name)(_PassthroughAlgorithm)
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
    class _Algo(QuantAlgorithmBase):
        def __init__(self, args, ctx):
            super().__init__()

        def forward(self, x):
            return x

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
        class _Algo(QuantAlgorithmBase):
            def __init__(self, args, ctx):
                super().__init__()

            def forward(self, x):
                return x

    try:
        with pytest.raises(ValueError, match="Only one 'structure' algorithm"):
            build_algorithms_by_target(
                _args(algos=[n1, n2]), "structure", SimpleNamespace()
            )
    finally:
        ALGO_REGISTRY._items.pop(n1, None)
        ALGO_REGISTRY._items.pop(n2, None)


# ---- ActivationQuantizer behavior ---------------------------------------


def test_activation_quantizer_observe_returns_without_fake_quant():
    aq = ActivationQuantizer(_args(), bits=8)
    aq.is_observe = True
    x = torch.randn(2, 32)
    assert aq(x) is x


def test_activation_quantizer_forward_quantizes_when_not_observing():
    aq = ActivationQuantizer(_args(), bits=8)
    x = torch.randn(2, 32, dtype=torch.float32)
    out = aq(x)
    # Same shape and dtype (int dtype quantizer is fake-quant).
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_activation_quantizer_observe_dispatches_only_calib_forward():
    name = "_ut_act_calib_dispatch"
    ALGO_REGISTRY.register(name=name, targets=("activation",))(_ObserveActivationAlgo)

    try:
        aq = ActivationQuantizer(_args(algos=[name]), bits=8)
        aq.quant_obj = _FakeQuantSpy()
        aq.is_observe = True
        x = torch.randn(2, 8)
        snapshot = x.clone()

        out = aq(x)

        algo = aq.algorithms[name]
        assert algo.calib_call_count == 1
        assert algo.quant_call_count == 0
        assert algo.received is x
        assert aq.quant_obj.call_count == 0
        assert out is x
        assert torch.equal(x, snapshot)
    finally:
        ALGO_REGISTRY._items.pop(name, None)


def test_activation_quantizer_non_observe_runs_all_algorithms_and_fake_quant():
    observe_name = "_ut_act_non_observe_aware"
    regular_name = "_ut_act_non_observe_regular"
    ALGO_REGISTRY.register(name=observe_name, targets=("activation",))(
        _ObserveActivationAlgo
    )
    ALGO_REGISTRY.register(name=regular_name, targets=("activation",))(
        _RegularActivationAlgo
    )

    try:
        aq = ActivationQuantizer(_args(algos=[observe_name, regular_name]), bits=8)
        aq.quant_obj = _FakeQuantSpy()
        aq.is_observe = False
        x = torch.tensor([1.0, 2.0])

        out = aq(x)

        observe_algo = aq.algorithms[observe_name]
        regular_algo = aq.algorithms[regular_name]
        expected_algo_out = x + 1
        assert observe_algo.quant_call_count == 1
        assert observe_algo.calib_call_count == 0
        assert observe_algo.is_observe is False
        assert regular_algo.call_count == 1
        assert aq.quant_obj.call_count == 1
        assert torch.equal(aq.quant_obj.received, expected_algo_out)
        assert torch.equal(out, expected_algo_out - 3)
    finally:
        ALGO_REGISTRY._items.pop(observe_name, None)
        ALGO_REGISTRY._items.pop(regular_name, None)


def test_activation_quantizer_trainable_params_collects_from_algorithms(
    _ephemeral_algo,
):
    aq = ActivationQuantizer(_args(algos=[_ephemeral_algo]), bits=8)
    # The ephemeral algo has no `trainable_params` -> empty.
    assert not aq.trainable_params()


def test_activation_quantizer_deploy_hooks_are_no_ops():
    aq = ActivationQuantizer(_args(), bits=8)
    assert aq.deploy() is None
    assert aq.load_deploy(scale=1.0, zero=0.0) is None


# ---- WeightQuantizer behavior --------------------------------------------


def test_weight_quantizer_observe_dispatches_calib_and_skips_quantize_hook():
    normal_name = "_ut_weight_calib_dispatch"
    hook_name = "_ut_weight_calib_hook"
    ALGO_REGISTRY.register(name=normal_name, targets=("weight",))(
        _ObserveActivationAlgo
    )

    @ALGO_REGISTRY.register(name=hook_name, targets=("weight",))
    class _QuantizeHook(QuantAlgorithmBase):
        def __init__(self, args, *_):
            super().__init__()
            self.calib_call_count = 0
            self.forward_call_count = 0
            self.quantize_call_count = 0

        def forward(self, x):
            self.forward_call_count += 1
            return x

        def calib_forward(self, x, *args, **kwargs):
            self.calib_call_count += 1
            return x

        def quantize(self, x, quant_obj):
            self.quantize_call_count += 1
            return x * 0

    try:
        wq = WeightQuantizer(_args(algos=[normal_name, hook_name], w_bits=8), w_bits=8)
        wq.quant_obj = _FakeQuantSpy()
        wq.is_observe = True
        w = torch.randn(4, 8)
        snapshot = w.clone()

        out = wq(w)

        normal_algo = wq.algorithms[normal_name]
        hook_algo = wq.algorithms[hook_name]
        assert normal_algo.calib_call_count == 1
        assert normal_algo.quant_call_count == 0
        assert hook_algo.calib_call_count == 1
        assert hook_algo.forward_call_count == 0
        assert hook_algo.quantize_call_count == 0
        assert wq.quant_obj.call_count == 0
        assert out is w
        assert torch.equal(out, snapshot)
    finally:
        ALGO_REGISTRY._items.pop(normal_name, None)
        ALGO_REGISTRY._items.pop(hook_name, None)


def test_weight_quantizer_forward_quantizes_when_not_observing():
    wq = WeightQuantizer(_args(w_bits=8), w_bits=8)
    w = torch.randn(4, 8, dtype=torch.float32)
    out = wq(w)
    assert out.shape == w.shape


def test_weight_quantizer_non_observe_uses_forward_and_fake_quant():
    name = "_ut_weight_quant_dispatch"
    ALGO_REGISTRY.register(name=name, targets=("weight",))(_ObserveActivationAlgo)

    try:
        wq = WeightQuantizer(_args(algos=[name], w_bits=8), w_bits=8)
        wq.quant_obj = _FakeQuantSpy()
        wq.is_observe = False
        w = torch.tensor([1.0, 2.0])

        out = wq(w)

        algo = wq.algorithms[name]
        assert algo.quant_call_count == 1
        assert algo.calib_call_count == 0
        assert wq.quant_obj.call_count == 1
        assert wq.quant_obj.received is w
        assert torch.equal(out, w - 3)
    finally:
        ALGO_REGISTRY._items.pop(name, None)


def test_weight_quantizer_observe_input_dispatches_to_algorithms_with_hook():
    seen = []

    @ALGO_REGISTRY.register(name=UT_OBSERVE_ALGO, targets=("weight",))
    class _Obs(QuantAlgorithmBase):
        def __init__(self, args, *_):
            super().__init__()

        def observe_input(self, x, weight):
            seen.append((x, weight))

        def forward(self, x):
            return x

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
    class _Double(QuantAlgorithmBase):
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
    class _Q(QuantAlgorithmBase):
        def __init__(self, args, *_):
            super().__init__()

        def quantize(self, x, quant_obj):
            return x * 0

        def forward(self, x):
            return x

    try:
        wq = WeightQuantizer(_args(algos=[UT_QUANT_HOOK_ALGO], w_bits=8), w_bits=8)
        x = torch.ones(1, 4)
        out, qa = wq.algo_forward(x)
        assert torch.equal(out, x)  # passthrough — quantize hook is deferred
        assert isinstance(qa, _Q)
    finally:
        ALGO_REGISTRY._items.pop(UT_QUANT_HOOK_ALGO, None)


def test_weight_quantizer_algo_forward_rejects_multiple_quantize_hooks():
    for n in (UT_QH_A_ALGO, UT_QH_B):

        @ALGO_REGISTRY.register(name=n, targets=("weight",))
        class _Q(QuantAlgorithmBase):
            def __init__(self, args, *_):
                super().__init__()

            def quantize(self, x, q):
                return x

            def forward(self, x):
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
    class _Q(QuantAlgorithmBase):
        def __init__(self, args, *_):
            super().__init__()

        def quantize(self, x, q):
            return x

        def forward(self, x):
            return x

    try:
        wq = WeightQuantizer(_args(algos=[UT_QH_EXPORT_ALGO], w_bits=8), w_bits=8)
        with pytest.raises(NotImplementedError, match="custom weight quantize"):
            wq.export_deploy(torch.zeros(4, 8))
    finally:
        ALGO_REGISTRY._items.pop(UT_QH_EXPORT_ALGO, None)


def test_weight_quantizer_export_deploy_supports_autoround_hook():
    wq = WeightQuantizer(_args(algos=["autoround"], w_bits=8), w_bits=8)
    out = wq.export_deploy(torch.randn(4, 8))

    assert "qweight" in out
    assert "weight_scale" in out
    assert "weight_bias" in out


def test_weight_quantizer_export_deploy_supports_autoround_hook_for_mxfp():
    args = _args(algos=["autoround"], quant_dtype="mxfp", w_bits=4, w_size=(4, 32))
    wq = WeightQuantizer(args, w_bits=4)
    out = wq.export_deploy(torch.randn(4, 32))

    assert "qweight" in out
    assert "weight_scale" in out


def test_build_algorithms_raises_when_algo_declares_targets_but_mismatches():
    name = "_ut_struct_mis"

    @ALGO_REGISTRY.register(name=name, targets=("weight",))
    class _Algo(QuantAlgorithmBase):
        def __init__(self, args, ctx=None):
            super().__init__()

        def forward(self, x):
            return x

    try:
        out = build_algorithms_by_target(_args(algos=[name]), "activation")
        assert isinstance(out, nn.ModuleDict)
        assert len(out) == 0
    finally:
        ALGO_REGISTRY._items.pop(name, None)


def test_activation_quantizer_trainable_params_returns_params_from_algo():
    name = "_ut_act_tp"

    @ALGO_REGISTRY.register(name=name, targets=("activation",))
    class _AlgoWithParams(QuantAlgorithmBase):
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
    class _DoubleAlgo(QuantAlgorithmBase):
        def __init__(self, args):
            super().__init__()

        def forward(self, x):
            return x * 2

    try:
        aq = ActivationQuantizer(_args(algos=[name]), bits=8)
        x = torch.tensor([1.0, 2.0, 3.0])
        out = aq(x)
        assert out.dtype == x.dtype
        assert out.shape == x.shape
    finally:
        ALGO_REGISTRY._items.pop(name, None)


def test_weight_quantizer_trainable_params_returns_params_from_algo():
    name = "_ut_wt_tp"

    @ALGO_REGISTRY.register(name=name, targets=("weight",))
    class _WtAlgoWithParams(QuantAlgorithmBase):
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
    class _QAlgo(QuantAlgorithmBase):
        def __init__(self, args, *_):
            super().__init__()

        def quantize(self, x, quant_obj):
            return x * 100

        def forward(self, x):
            return x

    try:
        wq = WeightQuantizer(_args(algos=[name], w_bits=8), w_bits=8)
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
    ALGO_REGISTRY.register(name=name)(_PassthroughAlgorithm)
    try:
        with pytest.raises(ValueError, match="missing registry metadata"):
            build_algorithms_by_target(_args(algos=[name]), "weight")
    finally:
        ALGO_REGISTRY._items.pop(name, None)


def test_build_algorithms_raises_when_target_not_in_algo_targets():
    name = "_ut_struct_nonmatch"

    @ALGO_REGISTRY.register(name=name, targets=("weight",))
    class _Algo(QuantAlgorithmBase):
        def __init__(self, args, ctx=None):
            super().__init__()

        def forward(self, x):
            return x

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
        quant_base_mod,
        "get_algo_names_by_target",
        lambda args, target: ["fake_algo"],
    )
    monkeypatch.setattr(
        algo_registry,
        "get_item",
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
        quant_base_mod,
        "get_algo_names_by_target",
        lambda args, target: ["fake_algo"],
    )
    monkeypatch.setattr(
        algo_registry,
        "get_item",
        lambda name: simple_ns(metadata={"targets": ("attn",)}, target=lambda *a: None),
    )
    args = SimpleNamespace()
    with pytest.raises(ValueError, match="cannot be used for target"):
        quant_base_mod.build_algorithms_by_target(args, "mlp")
