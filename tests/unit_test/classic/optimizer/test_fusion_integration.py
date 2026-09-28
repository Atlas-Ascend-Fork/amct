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
import pytest
from dataclasses import replace
from amct_pytorch.classic.optimizer.fusion_pattern import FUSION_PATTERNS

import torch
from torch import nn

from amct_pytorch.classic.quantize import quantize
from amct_pytorch.classic.optimizer.fusion_group import FusionGroup
from amct_pytorch.classic.quantize_op import SmoothQuant


class TinyQwenLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(8)
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(8, 8)
        self.self_attn.k_proj = nn.Linear(8, 8)
        self.self_attn.v_proj = nn.Linear(8, 8)
        self.post_attention_layernorm = nn.LayerNorm(8)
        self.mlp = nn.Module()
        self.mlp.up_proj = nn.Linear(8, 8)
        self.mlp.gate_proj = nn.Linear(8, 8)

    def forward(self, x):
        h = self.input_layernorm(x)
        attn = self.self_attn.q_proj(h) + self.self_attn.k_proj(h)
        attn = attn + self.self_attn.v_proj(h)
        h2 = self.post_attention_layernorm(attn)
        return self.mlp.up_proj(h2) + self.mlp.gate_proj(h2)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([TinyQwenLayer()])

    def forward(self, x):
        return self.layers[0](x)


def _config(algorithm, batch_num=1):
    return {
        'batch_num': batch_num,
        'quant_cfg': {
            'weights': {'type': 'int8', 'symmetric': True, 'strategy': 'channel'},
            'inputs': {'type': 'int8', 'symmetric': True, 'strategy': 'tensor'},
        },
        'algorithm': {
            algorithm: ({'smooth_strength': 0.5} if algorithm == 'smoothquant' else {})
        },
    }


def test_real_minmax_group_is_identified_and_calibrated_after_first_forward():
    model = TinyModel().to(torch.float16)
    quantize(model, _config('minmax'))
    model(torch.randn(2, 3, 8, dtype=torch.float16))

    groups = model._amct_fusion_registry.groups
    assert len(groups) == 2
    group = groups[0]
    assert group.algorithm == 'minmax'
    assert group.status == 'calibrated'
    assert group.batch_count == 1


def test_real_smoothquant_group_uses_string_algorithm_and_calibrates_after_two_forwards():
    model = TinyModel().to(torch.float16)
    quantize(model, _config('smoothquant', batch_num=2))
    model(torch.randn(2, 3, 8, dtype=torch.float16))
    model(torch.randn(2, 3, 8, dtype=torch.float16))

    groups = model._amct_fusion_registry.groups
    assert len(groups) == 2
    group = groups[0]
    assert group.algorithm == 'smoothquant'
    assert group.status == 'calibrated'
    assert group.batch_count == 2
    assert group.smooth_factor is not None


def test_smoothquant_finalize_waits_for_all_member_weight_stats():
    members = [
        SmoothQuant(
            nn.Linear(8, 8),
            f'm{i}',
            {
                'algorithm': {'smoothquant': {'smooth_strength': 0.5}},
                'batch_num': 1,
                'inputs_cfg': {
                    'enable_quant': True,
                    'quant_type': 'int8',
                    'symmetric': True,
                    'strategy': 'tensor',
                },
                'weights_cfg': {
                    'quant_type': 'int8',
                    'symmetric': True,
                    'strategy': 'channel',
                },
            },
        )
        for i in range(3)
    ]
    group = FusionGroup(
        'g', 'norm', 'norm', [f'm{i}' for i in range(3)], 'smoothquant', 'norm'
    )
    for member in members:
        member.fusion_group = group
        group.bind_member(member)

    x = torch.randn(2, 8)
    group.update_activation_stat(x)
    group.update_weight_stat(members[0])
    assert group.finalize_if_ready(members[0]) is False
    assert group.status == 'pending'

    group.update_weight_stat(members[1])
    group.update_weight_stat(members[2])
    assert group.finalize_if_ready(members[2]) is True
    assert group.status == 'calibrated'


class RealWeightMember(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 8))
        self.act_type = 'int8'


def _smooth_group(pre_ref, members, dynamic=False):
    group = FusionGroup('static-g', 'norm', pre_ref, members, 'smoothquant', 'norm')
    group.status = 'calibrated'
    group.smooth_factor = torch.tensor([[1.0, 2.0, 4.0, 8.0, 1.0, 2.0, 4.0, 8.0]])
    group.activation_scale = torch.tensor([0.1])
    group.dynamic_per_token = dynamic
    if dynamic:
        group.dynamic_layer = 'decoder_layer'
        group.layer_ref = 'layer'
    return group


def test_real_weight_smoothquant_static_group_fuses_without_linear_attribute():
    from amct_pytorch.classic.optimizer.apply_scaling_fusion_pass import (
        ApplyScalingFusionPass,
    )
    from amct_pytorch.classic.optimizer.fusion_group import FusionRegistry

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(8)
            self.q_proj = RealWeightMember()
            self.k_proj = RealWeightMember()
            self.v_proj = RealWeightMember()

        def forward(self, x):
            return x

    model = Model()
    group = _smooth_group('norm', ['q_proj', 'k_proj', 'v_proj'])
    registry = FusionRegistry()
    registry.add_group(group)
    model._amct_fusion_registry = registry
    group._writeback_done = {
        id(model.get_submodule(ref)): True for ref in group.scaling_layer_refs
    }
    ApplyScalingFusionPass().run(model)
    assert group.status == 'fused'


def test_dynamic_smoothquant_group_fuses_scaling_before_layer_replacement():
    from amct_pytorch.classic.optimizer.apply_scaling_fusion_pass import (
        ApplyScalingFusionPass,
    )
    from amct_pytorch.classic.optimizer.fusion_group import FusionRegistry

    class DynamicLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = nn.LayerNorm(8)
            self.self_attn = nn.Module()
            self.self_attn.q_proj = RealWeightMember()
            self.self_attn.k_proj = RealWeightMember()
            self.self_attn.v_proj = RealWeightMember()
            self.post_attention_layernorm = nn.LayerNorm(8)
            self.mlp = nn.Module()
            self.mlp.up_proj = RealWeightMember()
            self.mlp.gate_proj = RealWeightMember()

        def forward(self, x):
            return x

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = DynamicLayer()

        def forward(self, x):
            return x

    model = Model()
    group = _smooth_group(
        'layer.input_layernorm',
        [
            'layer.self_attn.q_proj',
            'layer.self_attn.k_proj',
            'layer.self_attn.v_proj',
        ],
        dynamic=True,
    )
    registry = FusionRegistry()
    registry.add_group(group)
    model._amct_fusion_registry = registry
    old_norm = model.layer.input_layernorm.weight.detach().clone()
    old_weight = model.layer.self_attn.q_proj.weight.detach().clone()

    group._writeback_done = {
        id(model.get_submodule(ref)): True for ref in group.scaling_layer_refs
    }
    ApplyScalingFusionPass().run(model)
    assert group.status == 'fused'
    assert not torch.allclose(model.layer.input_layernorm.weight, old_norm)
    assert not torch.allclose(model.layer.self_attn.q_proj.weight, old_weight)
    assert model.layer.self_attn.q_proj.smooth_scale_skipped is True


def test_qwen3_fused_layer_single_quantizer_constructor():
    import importlib.util
    from pathlib import Path

    from amct_pytorch.classic.deploy_op.shared_quantize import SharedDynamicQuantize

    spec = importlib.util.spec_from_file_location(
        'amct_qwen3_fused_layer_integration',
        Path(__file__).resolve().parents[4]
        / 'amct_pytorch/common/models/llm/qwen/qwen3/fused_layer.py',
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    Qwen3FusedDecoderLayer = module.Qwen3FusedDecoderLayer

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = nn.LayerNorm(8)
            self.self_attn = nn.Module()
            self.self_attn.q_proj = nn.Linear(8, 8)
            self.self_attn.k_proj = nn.Linear(8, 8)
            self.self_attn.v_proj = nn.Linear(8, 8)
            self.post_attention_layernorm = nn.LayerNorm(8)
            self.mlp = nn.Module()
            self.mlp.up_proj = nn.Linear(8, 8)
            self.mlp.gate_proj = nn.Linear(8, 8)

        def forward(self, x):
            return x

    layer = Qwen3FusedDecoderLayer(Layer(), SharedDynamicQuantize())
    assert layer._shared_by_pre_op['_default'] is not None


# Stub structures are explicitly registered for these unit tests only.


@pytest.fixture(autouse=True)
def register_test_patterns(monkeypatch):
    patterns = dict(FUSION_PATTERNS._patterns)
    patterns['qwen3'] = replace(
        patterns['qwen3'],
        layer_types=(f'{TinyQwenLayer.__module__}.{TinyQwenLayer.__qualname__}',),
    )
    monkeypatch.setattr(FUSION_PATTERNS, '_patterns', patterns)
