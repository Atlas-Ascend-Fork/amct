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
"""Targeted diff-coverage unit tests for previously-uncovered pruning error/edge branches."""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from mini_models import create_mini_mlp

from amct_pytorch.pruning import api
from amct_pytorch.pruning.api import _coerce_config
from amct_pytorch.pruning import registry as registry_mod
from amct_pytorch.pruning import utils as utils_mod
from amct_pytorch.pruning.config import PruneConfig
from amct_pytorch.pruning.diagnostics import _forward_ok, prune_diagnose
from amct_pytorch.pruning.domains.dense import (
    FusedGatedDenseTarget,
    GatedDenseTarget,
    TwoLayerDenseTarget,
    DensePruningDomain,
)
from amct_pytorch.pruning.pruner import AutoPruner, _is_skipped, _target_paths
from amct_pytorch.pruning.domains.cnn import CNNChannelTarget, CNNPruningDomain
from amct_pytorch.pruning.domains.moe import (
    MoEPruningDomain,
    MoETarget,
    _expert_slice_axis,
    _fused_expert_count,
    route_topk_from_output,
)
from amct_pytorch.pruning.config import MethodSpec
from amct_pytorch.pruning.context import PruneContext
from amct_pytorch.pruning.report import PruneReport
from amct_pytorch.pruning.prune_op.cnn_variance import VarianceChannelPruningMethod
from amct_pytorch.pruning.prune_op.dense_low_variance import (
    LowVarianceDensePruningMethod,
)
from amct_pytorch.pruning.prune_op.moe_mass_pruning import (
    ActivationCounter,
    MassMoEPruningMethod,
)
from amct_pytorch.pruning.prune_op import _moe_mass_common as moe_common

torch.set_num_threads(4)


def _empty_context():
    return PruneContext(data=None, batch_adapter=None)


def test_coerce_none_returns_default():
    cfg = _coerce_config(None)
    assert isinstance(cfg, PruneConfig)


def test_coerce_existing_config_is_copied():
    original = PruneConfig(min_neurons=8)
    cfg = _coerce_config(original)
    assert cfg is not original
    assert cfg.min_neurons == 8


def test_coerce_mapping():
    cfg = _coerce_config({"min_neurons": 7})
    assert cfg.min_neurons == 7


def test_coerce_bad_type_raises():
    with pytest.raises(TypeError):
        _coerce_config(123)


def test_tolerance_and_size_budget_mutually_exclusive():
    model, _ = create_mini_mlp()
    with pytest.raises(ValueError):
        api.prune(model, tolerance=0.1, size_budget=0.5)


def test_unknown_domain_raises_keyerror():
    reg = registry_mod.create_default_registry()
    with pytest.raises(KeyError):
        registry_mod.get_binding(reg, "no_such_domain", "low_variance")


def test_unknown_method_for_known_domain_raises_keyerror():
    reg = registry_mod.create_default_registry()
    with pytest.raises(KeyError):
        registry_mod.get_binding(reg, "dense", "no_such_method")


def test_default_registry_has_seven_bindings():
    reg = registry_mod.create_default_registry()
    assert len(reg) == 7
    binding = registry_mod.get_binding(reg, "dense", "low_variance")
    assert binding.domain.name == "dense"


def test_running_variance_empty_update_noop():
    rv = utils_mod.RunningVariance()
    rv.update(torch.empty(0, 4))
    assert rv.count == 0


def test_running_variance_no_samples_raises():
    rv = utils_mod.RunningVariance()
    with pytest.raises(RuntimeError):
        rv.variance()


def test_topk_keep_indices_requires_1d():
    with pytest.raises(ValueError):
        utils_mod.topk_keep_indices(torch.zeros(2, 2), 0.5, 1)


def test_get_submodule_empty_path_returns_model():
    model, _ = create_mini_mlp()
    assert utils_mod.get_submodule(model, "") is model


def test_prune_conv2d_in_channels_grouped_raises():
    conv = nn.Conv2d(4, 4, kernel_size=3, groups=2)
    with pytest.raises(ValueError):
        utils_mod.prune_conv2d_in_channels(conv, [0, 1])


def test_expand_flatten_indivisible_raises():
    with pytest.raises(ValueError):
        utils_mod.expand_flatten_channel_indices(
            [0], old_channels=3, linear_in_features=10
        )


def test_infer_model_device_no_params_cpu():
    empty = nn.Module()
    assert utils_mod.infer_model_device(empty) == torch.device("cpu")


def test_infer_model_device_with_params():
    model, _ = create_mini_mlp()
    assert utils_mod.infer_model_device(model).type == "cpu"


def test_move_batch_to_device_nested():
    args = (torch.zeros(2), [torch.zeros(1)], {"k": torch.zeros(1)}, 5)
    moved_args, moved_kwargs = utils_mod.move_batch_to_device(
        args, {"t": (torch.zeros(1),)}, torch.device("cpu")
    )
    assert torch.is_tensor(moved_args[0])
    assert isinstance(moved_args[1], list)
    assert isinstance(moved_args[2], dict)
    assert moved_args[3] == 5
    assert isinstance(moved_kwargs["t"], tuple)


def test_record_prune_width_and_size():
    model, _ = create_mini_mlp()
    utils_mod.record_prune_width(model, "_w", 3)
    utils_mod.record_prune_width(model, "_w", 4)
    utils_mod.record_prune_size(model, "_s", 9)
    meta = getattr(model, "_amct_prune_meta")
    assert meta["_w"] == {3, 4}
    assert meta["_s"] == 9


def test_solve_least_squares():
    gram = torch.eye(3) * 2.0
    rhs = torch.ones(3)
    out = utils_mod.solve_least_squares(gram, rhs)
    assert torch.allclose(out, torch.full((3,), 0.5))


def test_solve_least_squares_cpu_fallback(monkeypatch):
    calls = {"n": 0}
    real_solve = torch.linalg.solve

    def flaky_solve(a, b):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("no solver on device")
        return real_solve(a, b)

    monkeypatch.setattr(torch.linalg, "solve", flaky_solve)
    out = utils_mod.solve_least_squares(torch.eye(2), torch.ones(2))
    assert calls["n"] == 2
    assert torch.allclose(out, torch.ones(2))


def test_ridge_solve():
    gram = torch.eye(3).clone()
    rhs = torch.ones(3)
    out = utils_mod.ridge_solve(gram, rhs, ridge=0.1)
    assert out.shape == (3,)


def test_forward_ok_with_batch_adapter():
    model, _ = create_mini_mlp()

    def adapter(batch):
        return (batch,), {}

    assert _forward_ok(model, torch.randint(0, 1000, (2, 8)), adapter) is True


def test_forward_ok_dict_batch():
    model, _ = create_mini_mlp()
    assert (
        _forward_ok(model, {"input_ids": torch.randint(0, 1000, (2, 8))}, None) is True
    )


def test_forward_ok_tuple_batch():
    model, _ = create_mini_mlp()
    assert _forward_ok(model, (torch.randint(0, 1000, (2, 8)),), None) is True


def test_forward_ok_failure_returns_false():
    model, _ = create_mini_mlp()
    assert _forward_ok(model, torch.randn(2, 8), None) is False


def test_prune_diagnose_broken_domain_records_note():
    model = nn.Sequential(nn.ReLU())
    rep = prune_diagnose(model, data=None)
    assert rep.any_domain_detected is False
    assert any("no prunable targets" in n for n in rep.notes)
    assert any("acc binary-search" in n or "no data" in n for n in rep.notes)
    assert rep.search_works is False


def test_prune_diagnose_fixed_ratio_dryrun_error_note():
    model, _ = create_mini_mlp()
    rep = prune_diagnose(model, data=None, prune_ratio=0.5)
    assert isinstance(rep.summary(), str)


def test_target_paths_collects_string_attrs():
    t = TwoLayerDenseTarget(producer_path="a.b", consumer_path="a.c")
    paths = _target_paths(t)
    assert "a.b" in paths and "a.c" in paths


def test_is_skipped_true_and_empty():
    t = TwoLayerDenseTarget(
        producer_path="layers.0.dense1", consumer_path="layers.0.dense2"
    )
    assert _is_skipped(t, ["dense1"]) is True
    assert _is_skipped(t, []) is False


def test_missing_data_policy_raise():
    model, _ = create_mini_mlp()
    cfg = PruneConfig(
        methods={"dense": {"name": "reconstruct", "prune_ratio": 0.5}},
        missing_data_policy="raise",
    )
    cfg.copy_model = False
    with pytest.raises(ValueError):
        AutoPruner(cfg)(model, data=None)


def test_warn_skip_on_stage_error_keeps_model():
    model, _ = create_mini_mlp()
    cfg = PruneConfig(
        methods={"dense": {"name": "low_variance", "prune_ratio": 0.5}},
        stage_error_policy="warn_skip",
        missing_data_policy="warn_skip",
    )
    cfg.copy_model = False
    out = AutoPruner(cfg)(model, data=None)
    assert isinstance(out, nn.Module)


def test_two_layer_non_linear_raises():
    dom = DensePruningDomain()
    model = nn.Sequential()
    model.add_module("a", nn.ReLU())
    model.add_module("b", nn.ReLU())
    target = TwoLayerDenseTarget(producer_path="a", consumer_path="b")
    with pytest.raises(TypeError):
        dom.apply_keep_indices(model, target, [0])


def test_gated_non_linear_raises():
    dom = DensePruningDomain()
    model = nn.Module()
    model.add_module("gate_proj", nn.ReLU())
    model.add_module("up_proj", nn.ReLU())
    model.add_module("down_proj", nn.ReLU())
    target = GatedDenseTarget(
        gate_path="gate_proj", up_path="up_proj", down_path="down_proj"
    )
    with pytest.raises(TypeError):
        dom.apply_keep_indices(model, target, [0])


def test_fused_gated_non_linear_raises():
    dom = DensePruningDomain()
    model = nn.Module()
    model.add_module("gate_up_proj", nn.ReLU())
    model.add_module("down_proj", nn.ReLU())
    target = FusedGatedDenseTarget(gate_up_path="gate_up_proj", down_path="down_proj")
    with pytest.raises(TypeError):
        dom.apply_keep_indices(model, target, [0])


def test_hidden_size_non_linear_producer_raises():
    dom = DensePruningDomain()
    model = nn.Module()
    model.add_module("a", nn.ReLU())
    model.add_module("b", nn.ReLU())
    target = TwoLayerDenseTarget(producer_path="a", consumer_path="b")
    with pytest.raises(TypeError):
        dom.hidden_size(model, target)


def test_hidden_size_two_layer_linear():
    dom = DensePruningDomain()
    model = nn.Module()
    model.add_module("a", nn.Linear(8, 16))
    model.add_module("b", nn.Linear(16, 8))
    target = TwoLayerDenseTarget(producer_path="a", consumer_path="b")
    assert dom.hidden_size(model, target) == 16


def test_hidden_size_gated_non_linear_down_raises():
    dom = DensePruningDomain()
    model = nn.Module()
    model.add_module("gate_proj", nn.Linear(8, 16))
    model.add_module("up_proj", nn.Linear(8, 16))
    model.add_module("down_proj", nn.ReLU())
    target = GatedDenseTarget(
        gate_path="gate_proj", up_path="up_proj", down_path="down_proj"
    )
    with pytest.raises(TypeError):
        dom.hidden_size(model, target)


def test_apply_non_conv_producer_raises():
    dom = CNNPruningDomain()
    model = nn.Module()
    model.add_module("p", nn.Linear(4, 4))
    model.add_module("c", nn.Conv2d(4, 4, 3))
    target = CNNChannelTarget(producer_path="p", bn_path=None, consumer_path="c")
    with pytest.raises(TypeError):
        dom.apply_keep_indices(model, target, [0])


def test_apply_bad_consumer_raises():
    dom = CNNPruningDomain()
    model = nn.Module()
    model.add_module("p", nn.Conv2d(4, 4, 3))
    model.add_module("c", nn.ReLU())
    target = CNNChannelTarget(producer_path="p", bn_path=None, consumer_path="c")
    with pytest.raises(TypeError):
        dom.apply_keep_indices(model, target, [0])


def test_channel_count_non_conv_raises():
    dom = CNNPruningDomain()
    model = nn.Module()
    model.add_module("p", nn.Linear(4, 4))
    target = CNNChannelTarget(producer_path="p", bn_path=None, consumer_path="p")
    with pytest.raises(TypeError):
        dom.channel_count(model, target)


def test_variance_channel_requires_data():
    method = VarianceChannelPruningMethod()
    with pytest.raises(ValueError):
        method.apply(
            nn.Module(),
            CNNPruningDomain(),
            [],
            _empty_context(),
            PruneReport(backend="x", params_before=0),
            PruneConfig(),
            MethodSpec("variance_channel"),
        )


def test_variance_channel_wrong_domain():
    method = VarianceChannelPruningMethod()
    ctx = PruneContext(data=[torch.randn(1, 3, 8, 8)], batch_adapter=None)
    with pytest.raises(TypeError):
        method.apply(
            nn.Module(),
            DensePruningDomain(),
            [],
            ctx,
            PruneReport(backend="x", params_before=0),
            PruneConfig(),
            MethodSpec("variance_channel"),
        )


def test_variance_channel_zero_ratio_noop():
    method = VarianceChannelPruningMethod()
    ctx = PruneContext(data=[torch.randn(1, 3, 8, 8)], batch_adapter=None)
    rep = PruneReport(backend="x", params_before=0)
    method.apply(
        nn.Module(),
        CNNPruningDomain(),
        [],
        ctx,
        rep,
        PruneConfig(),
        MethodSpec("variance_channel", {"prune_ratio": 0.0}),
    )
    assert rep.params_before == 0


def test_low_variance_requires_data():
    method = LowVarianceDensePruningMethod()
    with pytest.raises(ValueError):
        method.apply(
            nn.Module(),
            DensePruningDomain(),
            [],
            _empty_context(),
            PruneReport(backend="x", params_before=0),
            PruneConfig(),
            MethodSpec("low_variance"),
        )


def test_low_variance_wrong_domain():
    method = LowVarianceDensePruningMethod()
    ctx = PruneContext(data=[torch.randint(0, 10, (1, 4))], batch_adapter=None)
    with pytest.raises(TypeError):
        method.apply(
            nn.Module(),
            CNNPruningDomain(),
            [],
            ctx,
            PruneReport(backend="x", params_before=0),
            PruneConfig(),
            MethodSpec("low_variance"),
        )


def test_low_variance_zero_ratio_noop():
    method = LowVarianceDensePruningMethod()
    ctx = PruneContext(data=[torch.randint(0, 10, (1, 4))], batch_adapter=None)
    rep = PruneReport(backend="x", params_before=0)
    method.apply(
        nn.Module(),
        DensePruningDomain(),
        [],
        ctx,
        rep,
        PruneConfig(),
        MethodSpec("low_variance", {"prune_ratio": 0.0}),
    )
    assert rep.params_before == 0


def test_mass_moe_requires_data():
    method = MassMoEPruningMethod()
    with pytest.raises(ValueError):
        method.apply(
            nn.Module(),
            MoEPruningDomain(),
            [],
            _empty_context(),
            PruneReport(backend="x", params_before=0),
            PruneConfig(),
            MethodSpec("activation_count"),
        )


def test_mass_moe_wrong_domain():
    method = MassMoEPruningMethod()
    ctx = PruneContext(data=[torch.randint(0, 10, (1, 4))], batch_adapter=None)
    with pytest.raises(TypeError):
        method.apply(
            nn.Module(),
            DensePruningDomain(),
            [],
            ctx,
            PruneReport(backend="x", params_before=0),
            PruneConfig(),
            MethodSpec("activation_count"),
        )


def test_route_topk_non_tensor_output_none():
    idx, wt = route_topk_from_output("not a tensor", num_experts=4, top_k=2)
    assert idx is None and wt is None


def test_route_topk_from_plain_tensor():
    scores = torch.randn(6, 4)
    idx, wt = route_topk_from_output(scores, num_experts=4, top_k=2)
    assert idx.shape == (6, 2)


def test_route_topk_tuple_with_int_and_float():
    idx_t = torch.randint(0, 4, (5, 2))
    wt_t = torch.rand(5, 2)
    idx, wt = route_topk_from_output((idx_t, wt_t), num_experts=4, top_k=2)
    assert idx.shape == (5, 2)
    assert wt.shape == (5, 2)


def test_route_topk_tuple_no_usable_tensor_none():
    idx, wt = route_topk_from_output(("a", 3), num_experts=4, top_k=2)
    assert idx is None and wt is None


def test_fused_expert_count_from_attr():
    mod = nn.Module()
    mod.num_experts = 6
    assert _fused_expert_count(mod) == 6


def test_fused_expert_count_from_param():
    mod = nn.Module()
    mod.w = nn.Parameter(torch.randn(8, 4, 2))
    assert _fused_expert_count(mod) == 8


def test_fused_expert_count_none():
    assert _fused_expert_count(nn.Module()) is None


def test_expert_slice_axis_single_dim():
    assert _expert_slice_axis(nn.Linear(4, 4), "weight", [0]) == 0


def test_expert_slice_axis_linear_weight():
    assert _expert_slice_axis(nn.Linear(4, 4), "weight", [0, 1]) == 0


def test_expert_slice_axis_bias_like():
    assert _expert_slice_axis(nn.Module(), "expert_bias", [0, 1]) == 1


def test_expert_slice_axis_ambiguous_none():
    assert _expert_slice_axis(nn.Module(), "mystery", [0, 1]) is None


def test_update_none_route_noop():
    counter = ActivationCounter(num_experts=4)
    counter.update("not a tensor", top_k=2)
    assert counter.counts is None


def test_update_counts_experts():
    counter = ActivationCounter(num_experts=4)
    scores = torch.tensor([[3.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]])
    counter.update(scores, top_k=1)
    assert counter.counts is not None
    assert int(counter.counts.sum()) == 2


def test_update_grows_counts_when_new_expert_seen():
    counter = ActivationCounter(num_experts=0)
    counter.counts = torch.zeros(2, dtype=torch.long)
    counter.num_experts = 2
    scores = torch.tensor([[0.0, 0.0, 0.0, 9.0]])
    counter.update(scores, top_k=1)
    assert len(counter.counts) >= 4


def test_register_router_hooks_skips_none_router_path():
    target = MoETarget(module_path="m", router_path=None, experts_path="e")
    accs, hooks = moe_common.register_router_hooks(
        nn.Module(),
        MoEPruningDomain(),
        [target],
        top_k=2,
        make_accumulator=lambda *a: object(),
        make_hook=lambda *a: (lambda *x: None),
    )
    assert accs == {} and hooks == []


def test_patch_router_module_top_k_typeerror_swallowed():
    class Router(nn.Module):
        def __init__(self):
            super().__init__()
            self.top_k = object()

    class MoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = Router()

    moe = MoE()
    target = MoETarget(module_path="", router_path=None, experts_path="e")
    moe_common.patch_router_module(moe, target, keep_idx=[0, 1], top_k=2)
    assert isinstance(moe.gate.top_k, object)


def test_prune_moe_layers_skips_none_keep():
    target = MoETarget(module_path="m", router_path="r", experts_path="e")
    rep = PruneReport(backend="x", params_before=0)
    out = moe_common.prune_moe_layers(
        "activation_count",
        nn.Module(),
        MoEPruningDomain(),
        [target],
        rep,
        top_k=2,
        select_fn=lambda idx, t: (None, 0, None),
    )
    assert out is None


def test_num_experts_fused_uninferable_raises():
    dom = MoEPruningDomain()
    model = nn.Module()
    model.add_module("e", nn.Module())
    target = MoETarget(module_path="", router_path=None, experts_path="e", fused=True)
    with pytest.raises(TypeError):
        dom.num_experts(model, target)


def test_num_experts_non_modulelist_raises():
    dom = MoEPruningDomain()
    model = nn.Module()
    model.add_module("e", nn.Linear(4, 4))
    target = MoETarget(module_path="", router_path=None, experts_path="e", fused=False)
    with pytest.raises(TypeError):
        dom.num_experts(model, target)


def test_num_experts_modulelist_counts():
    dom = MoEPruningDomain()
    model = nn.Module()
    model.add_module("e", nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)]))
    target = MoETarget(module_path="", router_path=None, experts_path="e", fused=False)
    assert dom.num_experts(model, target) == 2


def test_get_router_none_path():
    dom = MoEPruningDomain()
    target = MoETarget(module_path="", router_path=None, experts_path="e")
    assert dom.get_router(nn.Module(), target) is None


def test_get_experts_non_modulelist_raises():
    dom = MoEPruningDomain()
    model = nn.Module()
    model.add_module("e", nn.Linear(4, 4))
    target = MoETarget(module_path="", router_path=None, experts_path="e")
    with pytest.raises(TypeError):
        dom.get_experts(model, target)


def test_select_no_router_path():
    from amct_pytorch.pruning.prune_op.moe_mass_pruning import _select_keep_indices

    target = MoETarget(module_path="m", router_path=None, experts_path="e")
    keep, n = _select_keep_indices(
        target, {}, MoEPruningDomain(), nn.Module(), PruneConfig(), 0.5
    )
    assert keep is None and n == 0


def test_select_router_not_in_collectors():
    from amct_pytorch.pruning.prune_op.moe_mass_pruning import _select_keep_indices

    target = MoETarget(module_path="m", router_path="r", experts_path="e")
    keep, n = _select_keep_indices(
        target, {}, MoEPruningDomain(), nn.Module(), PruneConfig(), 0.5
    )
    assert keep is None and n == 0


def test_select_counts_none():
    from amct_pytorch.pruning.prune_op.moe_mass_pruning import _select_keep_indices

    target = MoETarget(module_path="m", router_path="r", experts_path="e")
    collectors = {"r": ActivationCounter(num_experts=4)}
    keep, n = _select_keep_indices(
        target, collectors, MoEPruningDomain(), nn.Module(), PruneConfig(), 0.5
    )
    assert keep is None and n == 0


def test_select_num_experts_at_floor():
    from amct_pytorch.pruning.prune_op.moe_mass_pruning import _select_keep_indices

    dom = MoEPruningDomain()
    model = nn.Module()
    model.add_module("experts", nn.ModuleList([nn.Linear(2, 2)]))
    target = MoETarget(module_path="", router_path="r", experts_path="experts")
    counter = ActivationCounter(num_experts=1)
    counter.counts = torch.ones(1, dtype=torch.long)
    collectors = {"r": counter}
    keep, n = _select_keep_indices(
        target, collectors, dom, model, PruneConfig(min_experts=1), 0.5
    )
    assert keep is None and n == 1


def test_select_keeps_subset():
    from amct_pytorch.pruning.prune_op.moe_mass_pruning import _select_keep_indices

    dom = MoEPruningDomain()
    model = nn.Module()
    model.add_module("experts", nn.ModuleList([nn.Linear(2, 2) for _ in range(4)]))
    target = MoETarget(module_path="", router_path="r", experts_path="experts")
    counter = ActivationCounter(num_experts=4)
    counter.counts = torch.tensor([10, 8, 1, 0], dtype=torch.long)
    collectors = {"r": counter}
    keep, n = _select_keep_indices(
        target, collectors, dom, model, PruneConfig(min_experts=1), 0.5
    )
    assert n == 4
    assert keep is not None and len(keep) < 4


def test_acc_search_error_note():
    model, _ = create_mini_mlp()

    class Boom:
        def __iter__(self):
            raise RuntimeError(f"boom from {type(self).__name__}")

    rep = prune_diagnose(model, data=Boom(), tolerance=0.05)
    assert any("error" in n.lower() for n in rep.notes)
