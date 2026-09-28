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
import torch
from torch import nn

from amct_pytorch.classic.optimizer.fusion_group import FusionGroup
from amct_pytorch.classic.quantize_op import MinMaxQuant, SmoothQuant


def _layer_config(algorithm='smoothquant', strategy='tensor', batch_num=2):
    inputs = {
        'enable_quant': True,
        'quant_type': 'int8',
        'symmetric': False,
        'strategy': strategy,
    }
    cfg = {
        'batch_num': batch_num,
        'weights_cfg': {'quant_type': 'int8', 'symmetric': True, 'strategy': 'channel'},
        'inputs_cfg': inputs,
        'algorithm': {'smoothquant': {'smooth_strength': 0.5}},
    }
    if algorithm == 'minmax':
        cfg['algorithm'] = {'minmax'}
    return cfg


def _quant(cls, in_features=6, out_features=5, **kwargs):
    return cls(nn.Linear(in_features, out_features), 'layer', _layer_config(**kwargs))


def _group(members, algorithm='smoothquant'):
    group = FusionGroup(
        group_id='g',
        group_type='norm',
        pre_op_ref='norm',
        scaling_layer_refs=[f'm{i}' for i in range(len(members))],
        algorithm=algorithm,
        fusion_rule='norm',
    )
    for member in members:
        member.fusion_group = group
    return group


def test_shared_activation_statistics_count_each_batch_once():
    group = _group(
        [SmoothQuant(nn.Linear(6, 5), 'a', _layer_config()) for _ in range(3)]
    )
    x = torch.randn(3, 6)
    group.update_activation_stat(x)
    group.update_activation_stat(x)
    group.update_activation_stat(x)
    assert group.batch_count == 1
    expected_max = x.reshape(-1, x.shape[-1]).amax(dim=0, keepdim=True)
    expected_min = x.reshape(-1, x.shape[-1]).amin(dim=0, keepdim=True)
    assert torch.equal(group.act_max, expected_max)
    assert torch.equal(group.act_min, expected_min)

    y = torch.randn(3, 6)
    group.update_activation_stat(y)
    assert group.batch_count == 2


def test_smoothquant_group_params_are_common_and_weight_scales_independent():
    members = [SmoothQuant(nn.Linear(6, 5), f'm{i}', _layer_config()) for i in range(3)]
    group = _group(members)
    for _ in range(2):
        x = torch.randn(4, 6)
        for member in members:
            member(x)
            group.update_activation_stat(x)

    group.finalize_quant_params(members)
    for member in members:
        # Re-enter the member's batch_num branch to exercise member-owned
        # weight-scale calculation after group activation writeback.
        member.cur_batch = member.batch_num - 1
        member.forward(torch.randn(4, 6))
    assert group.status == 'calibrated'
    assert group.smooth_factor is not None
    assert group.activation_scale is not None

    scale_values = [member.scale for member in members]
    assert all(torch.equal(scale_values[0], value) for value in scale_values[1:])
    assert all(
        torch.equal(members[0].scale_d, member.scale_d) for member in members[1:]
    )
    assert any(
        not torch.equal(members[0].scale_w, member.scale_w) for member in members[1:]
    )


def test_minmax_group_receives_common_activation_scale():
    members = [_quant(MinMaxQuant, algorithm='minmax') for _ in range(2)]
    group = _group(members, algorithm='minmax')
    for _ in range(2):
        x = torch.randn(4, 6)
        for member in members:
            member(x)
            group.update_activation_stat(x)
    group.finalize_quant_params(members)
    assert group.smooth_factor is None
    assert all(
        torch.equal(members[0].scale_d, member.scale_d) for member in members[1:]
    )
    assert all(
        torch.equal(members[0].offset_d, member.offset_d) for member in members[1:]
    )


def test_batch_num_one_group_overrides_member_parameters_and_clears_cache():
    members = [
        SmoothQuant(nn.Linear(6, 5), f'm{i}', _layer_config(batch_num=1))
        for i in range(2)
    ]
    group = _group(members)
    x = torch.randn(4, 6)
    for member in members:
        member(x)
        group.update_activation_stat(x)
    group.finalize_quant_params(members)

    assert group.batch_count == 1
    assert group.status == 'calibrated'
    assert all(torch.equal(group.smooth_factor, member.scale) for member in members)
    assert not getattr(members[0], 'fake_quant_cache_ready', False)


def test_failed_group_falls_back_to_independent_calculation():
    members = [SmoothQuant(nn.Linear(6, 5), f'm{i}', _layer_config()) for i in range(2)]
    group = _group(members)
    group.act_max = None
    group.finalize_quant_params(members)
    assert group.status == 'skipped'
    assert group.skip_reason
    # Existing independent parameters remain untouched.
    assert all(member.scale is None for member in members)


def test_quant_module_skips_own_calculation_when_group_is_healthy():
    members = [
        SmoothQuant(nn.Linear(6, 5), f'm{i}', _layer_config(batch_num=2))
        for i in range(2)
    ]
    group = _group(members)
    x = torch.randn(4, 6)
    for _ in range(2):
        for member in members:
            member(x)
            group.update_activation_stat(x)
    group.finalize_quant_params(members)
    assert all(torch.equal(group.smooth_factor, member.scale) for member in members)


def _bound_group(members, algorithm='smoothquant'):
    group = _group(members, algorithm)
    for member in members:
        group.bind_member(member)
    return group


def test_smooth_token_multiple_batches_match_independent_statistics():
    import copy

    torch.manual_seed(7)
    member = _quant(SmoothQuant, strategy='token', batch_num=2)
    reference = copy.deepcopy(member)
    group = _bound_group([member])
    for x in [torch.randn(2, 4, 6) - 2, torch.randn(2, 4, 6)]:
        reference(x)
        member(x)
    torch.testing.assert_close(member.scale, reference.scale)
    torch.testing.assert_close(member.scale_d, reference.scale_d)
    torch.testing.assert_close(member.offset_d, reference.offset_d)
    assert group.status == 'calibrated'


def test_minmax_token_matches_independent_statistics():
    import copy

    member = _quant(MinMaxQuant, algorithm='minmax', strategy='token', batch_num=2)
    reference = copy.deepcopy(member)
    _bound_group([member], 'minmax')
    for x in [torch.randn(2, 4, 6), torch.randn(2, 4, 6) * 2]:
        reference(x)
        member(x)
    assert member.scale_d.shape == (1, 4, 1)
    torch.testing.assert_close(member.scale_d, reference.scale_d)
    torch.testing.assert_close(member.offset_d, reference.offset_d)


def test_one_batch_convert_finishes_all_smooth_members():
    members = [_quant(SmoothQuant, batch_num=1) for _ in range(3)]
    group = _bound_group(members)
    x = torch.randn(2, 4, 6)
    for member in members:
        member(x)
    from amct_pytorch.classic.optimizer.apply_scaling_fusion_pass import (
        ApplyScalingFusionPass,
    )
    from amct_pytorch.classic.optimizer.fusion_group import FusionRegistry

    model = nn.Module()
    model.norm = nn.LayerNorm(6)
    for i, member in enumerate(members):
        model.add_module(f'm{i}', member)
    model._amct_fusion_registry = FusionRegistry()
    model._amct_fusion_registry.add_group(group)
    original_weights = [m.weight.detach().clone() for m in members]
    ApplyScalingFusionPass().run(model)
    for member, weight in zip(members, original_weights):
        assert member.scale is not None
        assert member.scale_d is not None
        assert member.scale_w is not None
        torch.testing.assert_close(member.scale, group.smooth_factor)
        expected, _ = member.calculate_weights_scale_factor(
            weight * member.scale, 'channel'
        )
        torch.testing.assert_close(member.scale_w, expected)


def test_one_batch_fakequant_finishes_pending_member_without_another_calibration():
    members = [_quant(SmoothQuant, batch_num=1) for _ in range(3)]
    _bound_group(members)
    x = torch.randn(2, 4, 6)
    for member in members:
        member(x)
    for member in members:
        assert torch.isfinite(member(x)).all()
        assert member.scale_w is not None


def test_group_calibration_does_not_keep_full_inputs_alive():
    import weakref
    import gc

    members = [_quant(SmoothQuant, strategy='token', batch_num=3) for _ in range(3)]
    group = _bound_group(members)
    x = torch.randn(2, 4, 6)
    ref = weakref.ref(x)
    for member in members:
        member(x)
        assert member.batch_input is None
    del x
    gc.collect()
    assert ref() is None
    assert group.batch_input is None


def test_dynamic_group_does_not_cache_calibration_inputs():
    members = [_quant(SmoothQuant, strategy='token', batch_num=2) for _ in range(3)]
    group = _bound_group(members)
    group.dynamic_per_token = True
    for member in members:
        member.dynamic = True
    for _ in range(2):
        x = torch.randn(2, 4, 6)
        for member in members:
            member(x)
    assert group.batch_input is None
    assert all(member.batch_input is None for member in members)


def test_reusing_input_tensor_counts_distinct_calibration_calls():
    members = [_quant(SmoothQuant, batch_num=2) for _ in range(3)]
    group = _bound_group(members)
    x = torch.randn(2, 4, 6)
    for _ in range(2):
        for member in members:
            member(x)
    assert group.batch_count == 2
    assert group.status == 'calibrated'


def test_dynamic_group_fakequant_uses_current_token_scales():
    from amct_pytorch.classic.quantize_op.utils import calculate_scale_offset
    from amct_pytorch.common.utils.quant_util import quant_dequant_tensor

    for cls, algorithm in [(SmoothQuant, 'smoothquant'), (MinMaxQuant, 'minmax')]:
        member = _quant(cls, algorithm=algorithm, strategy='token', batch_num=1)
        member.dynamic = True
        member.act_symmetric = True
        group = _bound_group([member], algorithm)
        group.dynamic_per_token = True
        member(torch.randn(1, 2, 6))
        x = torch.randn(1, 3, 6) * torch.tensor([0.001, 1.0, 100.0]).reshape(1, 3, 1)
        actual = member(x)
        smooth_x = x / member.scale if algorithm == 'smoothquant' else x
        scale, offset = calculate_scale_offset(
            smooth_x.amax(dim=-1, keepdim=True),
            smooth_x.amin(dim=-1, keepdim=True),
            True,
            'int8',
        )
        expected_x = quant_dequant_tensor(smooth_x, 'int8', scale, offset)
        expected = torch.nn.functional.linear(
            expected_x, member.cached_dq_w, member.cached_bias
        )
        torch.testing.assert_close(actual, expected)
