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

from amct_pytorch.classic.optimizer.fusion_group import (
    FusionGroup,
    FusionRegistry,
)
from amct_pytorch.classic.optimizer.fusion_hooks import (
    install_fusion_hooks,
)


class Quant(nn.Module):
    def __init__(self, alpha=0.5, quant_type='int8', strategy='tensor'):
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.algorithm = 'smoothquant' if alpha is not None else 'minmax'
        self.quant_config = {
            'algorithm': self.algorithm,
            'inputs_cfg': {
                'quant_type': quant_type,
                'symmetric': False,
                'strategy': strategy,
            },
            'batch_num': 2,
            'smooth_strength': alpha,
        }
        self.fusion_group = None

    def forward(self, x):
        return self.linear(x)


class Attention(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.norm = nn.LayerNorm(8)
        self.q = Quant(alpha)
        self.k = Quant(alpha)
        self.v = Quant(alpha)

    def forward(self, x):
        h = self.norm(x)
        return self.q(h) + self.k(h) + self.v(h)


class Mlp(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.norm = nn.LayerNorm(8)
        self.up = Quant(alpha)
        self.gate = Quant(alpha)

    def forward(self, x):
        h = self.norm(x)
        return self.up(h) + self.gate(h)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attention()
        self.mlp = Mlp()

    def forward(self, x):
        return self.attn(x) + self.mlp(x)


def _registry_for(model):
    install_fusion_hooks(model)
    model(torch.randn(2, 8))
    assert isinstance(model._amct_fusion_registry, FusionRegistry)
    return model._amct_fusion_registry


def test_hooks_build_qkv_and_up_gate_groups():
    model = Model()
    registry = _registry_for(model)

    groups = {(g.pre_op_ref, tuple(g.scaling_layer_refs)) for g in registry.groups}

    assert ('attn.norm', ('attn.q', 'attn.k', 'attn.v')) in groups
    assert ('mlp.norm', ('mlp.up', 'mlp.gate')) in groups
    assert all(g.status == 'pending' for g in registry.groups)
    assert all(g.group_type == 'norm' for g in registry.groups)
    assert all(g.fusion_rule == 'norm' for g in registry.groups)
    assert all(
        model.get_submodule(ref).fusion_group is not None
        for g in registry.groups
        for ref in g.scaling_layer_refs
    )


def test_single_member_smoothquant_is_allowed_and_layout_transform_is_rejected():
    class Single(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(8)
            self.fc = Quant(0.5)

        def forward(self, x):
            return self.fc(self.norm(x))

    model = Single()
    registry = _registry_for(model)
    assert len(registry.groups) == 1
    assert registry.groups[0].scaling_layer_refs == ['fc']


def test_config_mismatch_does_not_create_group():
    class Mixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(8)
            self.a = Quant(0.5)
            self.b = Quant(0.6)

        def forward(self, x):
            h = self.norm(x)
            return self.a(h) + self.b(h)

    model = Mixed()
    registry = _registry_for(model)
    assert registry.groups == []


def test_minmax_single_member_is_rejected():
    class Single(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(8)
            self.fc = Quant(None)

        def forward(self, x):
            return self.fc(self.norm(x))

    model = Single()
    registry = _registry_for(model)
    assert registry.groups == []


def test_outside_consumer_is_rejected():
    class Outside(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(8)
            self.a = Quant(0.5)
            self.b = Quant(0.5)
            self.other = nn.Linear(8, 8)

        def forward(self, x):
            h = self.norm(x)
            return self.a(h) + self.b(h) + self.other(h)

    model = Outside()
    registry = _registry_for(model)
    assert registry.groups == []


def test_registry_uses_plain_attribute_and_group_ids_are_unique():
    model = Model()
    registry = _registry_for(model)
    assert '_amct_fusion_registry' not in dict(model.named_buffers())
    assert '_amct_fusion_registry' not in dict(model.named_parameters())
    ids = [g.group_id for g in registry.groups]
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


def test_quantize_installs_fusion_hooks():
    from unittest.mock import patch
    import inspect
    import amct_pytorch.classic.quantize as quantize_module

    quantize_module = inspect.getmodule(quantize_module.__wrapped__)

    model = nn.Linear(8, 8)
    with (
        patch.object(quantize_module, 'set_default_config', return_value={}),
        patch.object(quantize_module, 'parse_config', return_value={}),
        patch.object(quantize_module, 'ModelOptimizer'),
        patch.object(quantize_module, 'InsertQuantizeModulePass'),
        patch.object(quantize_module, 'install_fusion_hooks') as mock_install,
    ):
        quantize_module.quantize(model)
    mock_install.assert_called_once_with(model)
