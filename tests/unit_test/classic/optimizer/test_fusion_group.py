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
from unittest.mock import patch

from torch import nn

from amct_pytorch.classic.optimizer.fusion_group import (
    FusionGroup,
    FusionRegistry,
)
from amct_pytorch.classic.optimizer.fusion_pattern import (
    build_fusion_groups_from_patterns,
)


class Quant(nn.Module):
    def __init__(self, algorithm='smoothquant', strategy='tensor'):
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.batch_num = 2
        self.act_type = 'int8'
        self.act_symmetric = True
        self.act_granularity = strategy
        self.fusion_group = None
        self.quant_config = {
            'algorithm': (
                {'smoothquant': {'smooth_strength': 0.5}}
                if algorithm == 'smoothquant'
                else {'minmax': {}}
            ),
            'batch_num': self.batch_num,
            'inputs_cfg': {
                'quant_type': self.act_type,
                'symmetric': self.act_symmetric,
                'strategy': strategy,
            },
        }

    def forward(self, x):
        return self.linear(x)


class QwenLayer(nn.Module):
    def __init__(self, algorithm='smoothquant'):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(8)
        self.post_attention_layernorm = nn.LayerNorm(8)
        self.self_attn = nn.Module()
        self.self_attn.q_proj = Quant(algorithm)
        self.self_attn.k_proj = Quant(algorithm)
        self.self_attn.v_proj = Quant(algorithm)
        self.mlp = nn.Module()
        self.mlp.up_proj = Quant(algorithm)
        self.mlp.gate_proj = Quant(algorithm)

    def forward(self, x):
        h = self.input_layernorm(x)
        out = self.self_attn.q_proj(h) + self.self_attn.k_proj(h)
        out = out + self.self_attn.v_proj(h)
        h2 = self.post_attention_layernorm(out)
        return self.mlp.up_proj(h2) + self.mlp.gate_proj(h2)


class VitLayer(nn.Module):
    def __init__(self, algorithm='smoothquant'):
        super().__init__()
        self.layernorm_before = nn.LayerNorm(8)
        self.layernorm_after = nn.LayerNorm(8)
        self.attention = nn.Module()
        self.attention.query = Quant(algorithm)
        self.attention.key = Quant(algorithm)
        self.attention.value = Quant(algorithm)
        self.mlp = nn.Module()
        self.mlp.fc1 = Quant(algorithm)

    def forward(self, x):
        h = self.layernorm_before(x)
        out = self.attention.query(h) + self.attention.key(h)
        out = out + self.attention.value(h)
        return self.mlp.fc1(self.layernorm_after(out))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([QwenLayer()])

    def forward(self, x):
        return self.layers[0](x)


def _registry_for(model):
    registry = build_fusion_groups_from_patterns(model)
    model._amct_fusion_registry = registry
    assert isinstance(registry, FusionRegistry)
    return registry


def test_pattern_whitelist_builds_qwen_groups_without_forward():
    model = Model()
    registry = _registry_for(model)

    groups = {(g.pre_op_ref, tuple(g.scaling_layer_refs)): g for g in registry.groups}
    attention = (
        'layers.0.input_layernorm',
        (
            'layers.0.self_attn.q_proj',
            'layers.0.self_attn.k_proj',
            'layers.0.self_attn.v_proj',
        ),
    )
    mlp = (
        'layers.0.post_attention_layernorm',
        ('layers.0.mlp.up_proj', 'layers.0.mlp.gate_proj'),
    )
    assert attention in groups
    assert mlp in groups
    assert all(group.status == 'pending' for group in registry.groups)
    assert all(group.group_type == 'norm' for group in registry.groups)
    assert all(
        model.get_submodule(ref).fusion_group is not None
        for group in registry.groups
        for ref in group.scaling_layer_refs
    )


def test_pattern_whitelist_builds_vit_groups_and_single_member_smooth_group():
    model = VitLayer()
    registry = _registry_for(model)
    groups = {(g.pre_op_ref, tuple(g.scaling_layer_refs)): g for g in registry.groups}
    assert (
        'layernorm_before',
        ('attention.query', 'attention.key', 'attention.value'),
    ) in groups
    assert ('layernorm_after', ('mlp.fc1',)) in groups


def test_unregistered_model_has_no_registry_groups():
    class Unknown(nn.Module):
        def forward(self, x):
            return x

    assert _registry_for(Unknown()).groups == []


def test_dynamic_group_uses_containing_layer_ref():
    model = Model()
    registry = _registry_for(model)
    for group in registry.groups:
        group.dynamic_per_token = True
        group.layer_ref = 'layers.0'
    assert all(group.layer_ref == 'layers.0' for group in registry.groups)


def test_minmax_single_member_is_rejected():
    model = VitLayer(algorithm='minmax')
    registry = _registry_for(model)
    assert all(len(group.scaling_layer_refs) > 1 for group in registry.groups)


def test_config_mismatch_does_not_create_group():
    model = Model()
    model.layers[0].self_attn.k_proj.quant_config['algorithm']['smoothquant'][
        'smooth_strength'
    ] = 0.6
    registry = _registry_for(model)
    assert all(
        'self_attn.k_proj' not in group.scaling_layer_refs for group in registry.groups
    )


def test_registry_uses_plain_attribute_and_group_ids_are_unique():
    model = Model()
    registry = _registry_for(model)
    assert '_amct_fusion_registry' not in dict(model.named_buffers())
    assert '_amct_fusion_registry' not in dict(model.named_parameters())
    ids = [group.group_id for group in registry.groups]
    assert len(ids) == len(set(ids))


def test_fusion_group_status_and_fields():
    group = FusionGroup(
        group_id='g',
        group_type='norm',
        pre_op_ref='norm',
        scaling_layer_refs=['a'],
        algorithm='smoothquant',
        fusion_rule='norm',
    )
    assert group.status == 'pending'
    assert group.skip_reason is None
    assert group.activation_scale is None
    assert group.smooth_factor is None
    assert group.act_max is None
    assert group.act_min is None
    assert group.wts_max_by_layer == {}


def test_quantize_builds_pattern_registry_directly():
    import inspect

    import amct_pytorch.classic.quantize as quantize_module

    quantize_module = inspect.getmodule(quantize_module.__wrapped__)
    model = nn.Linear(8, 8)
    with (
        patch.object(quantize_module, 'set_default_config', return_value={}),
        patch.object(quantize_module, 'parse_config', return_value={}),
        patch.object(quantize_module, 'ModelOptimizer'),
        patch.object(quantize_module, 'InsertQuantizeModulePass'),
        patch.object(
            quantize_module,
            'build_fusion_groups_from_patterns',
            return_value=FusionRegistry(),
        ) as mock_build,
    ):
        quantize_module.quantize(model)
    mock_build.assert_called_once_with(model)
