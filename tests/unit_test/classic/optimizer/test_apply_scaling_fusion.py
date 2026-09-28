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

import pytest

import torch
from torch import nn

from amct_pytorch.classic.optimizer.apply_scaling_fusion_pass import (
    ApplyScalingFusionPass,
)
from amct_pytorch.classic.optimizer.fusion_group import FusionGroup, FusionRegistry


def _group(pre_op, members, algorithm='smoothquant'):
    group = FusionGroup(
        group_id='g',
        group_type='norm' if isinstance(pre_op, nn.LayerNorm) else 'linear',
        pre_op_ref='pre',
        scaling_layer_refs=[f'm{i}' for i in range(len(members))],
        algorithm=algorithm,
        fusion_rule='norm' if isinstance(pre_op, nn.LayerNorm) else 'linear',
    )
    group.status = 'calibrated'
    group._writeback_done = {id(member): True for member in members}
    if algorithm == 'smoothquant':
        group.smooth_factor = torch.tensor([[1.0, 2.0, 4.0]])
    group.activation_scale = torch.tensor([0.1])
    return group


class Member(nn.Module):
    def __init__(self, smooth=True):
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.scale = torch.tensor([[1.0, 2.0, 4.0]]) if smooth else None
        self.scale_d = torch.tensor([0.1])
        self.offset_d = None
        self.scale_w = torch.tensor([[0.2], [0.3]])
        self.input_quant_disabled = False
        self.smooth_scale_skipped = False
        self.act_type = 'int8'
        self.act_granularity = 'tensor'


class NormModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.LayerNorm(3)
        self.m0 = Member()
        self.m1 = Member()

    def forward(self, x):
        h = self.pre(x)
        return self.m0.linear(h) + self.m1.linear(h)


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.Linear(3, 3)
        self.m0 = Member()
        self.m1 = Member()

    def forward(self, x):
        h = self.pre(x)
        return self.m0.linear(h) + self.m1.linear(h)


def test_norm_fusion_modifies_preop_and_member_weights():
    model = NormModel()
    group = _group(model.pre, [model.m0, model.m1])
    registry = FusionRegistry()
    registry.add_group(group)
    model._amct_fusion_registry = registry
    old_gamma = model.pre.weight.detach().clone()
    old_beta = model.pre.bias.detach().clone()
    old_w0 = model.m0.linear.weight.detach().clone()
    old_w1 = model.m1.linear.weight.detach().clone()

    ApplyScalingFusionPass().run(model)
    s = group.smooth_factor
    assert torch.allclose(model.pre.pre_op.weight, old_gamma / s, atol=1e-6)
    assert torch.allclose(model.pre.pre_op.bias, old_beta / s, atol=1e-6)
    assert torch.allclose(model.m0.linear.weight, old_w0 * s, atol=1e-6)
    assert torch.allclose(model.m1.linear.weight, old_w1 * s, atol=1e-6)
    assert group.status == 'fused'
    assert model.m0.smooth_scale_skipped
    assert model.m1.smooth_scale_skipped


def test_linear_preop_fusion_modifies_output_channels():
    model = LinearModel()
    group = _group(model.pre, [model.m0, model.m1])
    registry = FusionRegistry()
    registry.add_group(group)
    model._amct_fusion_registry = registry
    old_weight = model.pre.weight.detach().clone()
    old_bias = model.pre.bias.detach().clone()
    old_w0 = model.m0.linear.weight.detach().clone()

    ApplyScalingFusionPass().run(model)
    s = group.smooth_factor.reshape(-1)
    assert torch.allclose(model.pre.pre_op.weight, old_weight / s[:, None], atol=1e-6)
    assert torch.allclose(model.pre.pre_op.bias, old_bias / s, atol=1e-6)
    assert torch.allclose(
        model.m0.linear.weight, old_w0 * group.smooth_factor, atol=1e-6
    )


def test_minmax_group_uses_shared_quantize_without_parameter_fusion():
    model = NormModel()
    group = _group(model.pre, [model.m0, model.m1], algorithm='minmax')
    registry = FusionRegistry()
    registry.add_group(group)
    model._amct_fusion_registry = registry
    old_gamma = model.pre.weight.detach().clone()
    old_w0 = model.m0.linear.weight.detach().clone()

    ApplyScalingFusionPass().run(model)
    assert torch.equal(model.pre.pre_op.weight, old_gamma)
    assert torch.equal(model.m0.linear.weight, old_w0)
    assert model.m0.input_quant_disabled
    assert not model.m0.smooth_scale_skipped
    from amct_pytorch.classic.deploy_op.shared_quantize import FusedPreOp

    assert isinstance(model.pre, FusedPreOp)


def test_single_member_smoothquant_fuses_without_shared_quantize():
    model = NormModel()
    model.m1 = None
    group = _group(model.pre, [model.m0])
    registry = FusionRegistry()
    registry.add_group(group)
    model._amct_fusion_registry = registry
    old_gamma = model.pre.weight.detach().clone()
    old_w0 = model.m0.linear.weight.detach().clone()

    ApplyScalingFusionPass().run(model)
    assert torch.allclose(model.pre.weight, old_gamma / group.smooth_factor, atol=1e-6)
    assert torch.allclose(
        model.m0.linear.weight, old_w0 * group.smooth_factor, atol=1e-6
    )
    from amct_pytorch.classic.deploy_op.shared_quantize import FusedPreOp

    assert not isinstance(model.pre, FusedPreOp)


def test_repeat_convert_is_idempotent():
    model = NormModel()
    group = _group(model.pre, [model.m0, model.m1])
    registry = FusionRegistry()
    registry.add_group(group)
    model._amct_fusion_registry = registry
    ApplyScalingFusionPass().run(model)
    weight = model.pre.pre_op.weight.detach().clone()
    ApplyScalingFusionPass().run(model)
    assert torch.equal(model.pre.pre_op.weight, weight)


def test_failed_group_is_skipped_without_model_changes():
    model = NormModel()
    group = _group(model.pre, [model.m0, model.m1])
    group.status = 'skipped'
    registry = FusionRegistry()
    registry.add_group(group)
    model._amct_fusion_registry = registry
    old_weight = model.pre.weight.detach().clone()
    ApplyScalingFusionPass().run(model)
    assert torch.equal(model.pre.weight, old_weight)
    assert group.status == 'skipped'


def test_convert_runs_fusion_before_replacement():
    import inspect
    import amct_pytorch.classic.quantize as quantize_module

    quantize_module = inspect.getmodule(quantize_module)

    model = nn.Linear(3, 3)
    with (
        patch.object(quantize_module, 'ModelOptimizer') as mock_opt_cls,
        patch.object(quantize_module, 'ApplyScalingFusionPass') as mock_fusion_cls,
        patch.object(quantize_module, 'ReplaceNpuQuantModulePass') as mock_replace_cls,
    ):
        quantize_module.convert(model)

    calls = [call.args[0] for call in mock_opt_cls.return_value.add_pass.call_args_list]
    assert calls == [mock_fusion_cls.return_value, mock_replace_cls.return_value]


def test_deploy_linear_skips_input_quant_and_smooth_scale(monkeypatch):
    from unittest.mock import MagicMock
    import sys

    mock_npu = MagicMock()
    schema = MagicMock()
    schema.arguments = []
    mock_npu.npu_quant_matmul._schemas = {'default': schema}
    sys.modules['torch_npu'] = mock_npu
    from amct_pytorch.classic.deploy_op.npu_quantization_linear import (
        NpuQuantizationLinear,
    )

    class QuantModule(nn.Module):
        weight = nn.Parameter(torch.randn(3, 4))
        bias = nn.Parameter(torch.randn(3))
        scale = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        scale_d = torch.tensor([0.1])
        scale_w = torch.tensor([[0.2], [0.3], [0.4]])
        offset_d = None
        act_type = 'int8'
        wts_type = 'int8'
        act_granularity = 'tensor'
        act_symmetric = False
        group_size = None
        dynamic = False
        input_quant_disabled = True
        smooth_scale_skipped = True

    module = QuantModule()
    deploy = NpuQuantizationLinear(module)
    assert not hasattr(deploy, 'scale_factor')
    assert deploy.input_quant_disabled is True


class DynamicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Module()
        self.layer.input_layernorm = nn.LayerNorm(3)
        self.layer.self_attn = nn.Module()
        self.layer.self_attn.q_proj = Member()
        self.layer.self_attn.k_proj = Member()
        self.layer.self_attn.v_proj = Member()
        self.layer.post_attention_layernorm = nn.LayerNorm(3)
        self.layer.mlp = nn.Module()
        self.layer.mlp.up_proj = Member()
        self.layer.mlp.gate_proj = Member()

    def forward(self, x):
        h = self.layer.input_layernorm(x)
        return self.layer.self_attn.q_proj.linear(
            h
        ) + self.layer.self_attn.k_proj.linear(h)


def _dynamic_group(model, members):
    group = FusionGroup(
        group_id='dynamic',
        group_type='norm',
        pre_op_ref='layer.input_layernorm',
        scaling_layer_refs=[
            'layer.self_attn.q_proj',
            'layer.self_attn.k_proj',
            'layer.self_attn.v_proj',
        ],
        algorithm='minmax',
        fusion_rule='qwen3_dynamic_layer',
    )
    group.status = 'calibrated'
    group.activation_scale = torch.tensor([0.1])
    group.dynamic_layer = 'decoder_layer'
    group.dynamic_per_token = True
    group.layer_ref = 'layer'
    return group


def test_dynamic_per_token_replaces_qwen3_layer_and_shares_quantization():
    model = DynamicModel()
    group = _dynamic_group(model, [])
    registry = FusionRegistry()
    registry.add_group(group)
    model._amct_fusion_registry = registry

    ApplyScalingFusionPass().run(model)

    assert group.status == 'fused'
    assert type(model.layer).__name__.startswith('Fused')
    assert model.layer._shared_by_pre_op['input_layernorm'].act_type == 'int8'
    for name in ('q_proj', 'k_proj', 'v_proj'):
        member = getattr(model.layer.self_attn, name)
        assert member.input_quant_disabled is True
    # Repeat fusion remains idempotent.
    ApplyScalingFusionPass().run(model)
    assert group.status == 'fused'


def test_dynamic_per_token_passes_shared_scale_to_all_members(monkeypatch):
    class SharedMember(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 3)
            self.act_type = 'int8'
            self.input_quant_disabled = False

        def forward(self, x, pertoken_scale=None):
            self.last_input = x
            self.last_scale = pertoken_scale
            return self.linear(x)

    class SharedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Module()
            self.layer.input_layernorm = nn.LayerNorm(3)
            self.layer.self_attn = nn.Module()
            self.layer.self_attn.q_proj = SharedMember()
            self.layer.self_attn.k_proj = SharedMember()
            self.layer.self_attn.v_proj = SharedMember()
            self.layer.self_attn.head_dim = 3
            self.layer.self_attn.num_attention_heads = 1
            self.layer.self_attn.num_key_value_groups = 1
            self.layer.post_attention_layernorm = nn.LayerNorm(3)
            self.layer.mlp = nn.Module()
            self.layer.mlp.up_proj = SharedMember()
            self.layer.mlp.gate_proj = SharedMember()
            self.layer.mlp.down_proj = SharedMember()

        def forward(self, x):
            return self.layer(x)

    model = SharedModel()
    group = _dynamic_group(model, [])
    registry = FusionRegistry()
    registry.add_group(group)
    model._amct_fusion_registry = registry

    ApplyScalingFusionPass().run(model)
    assert group.status == 'fused'

    calls = []

    class CountingQuant(nn.Module):
        def forward(self, x):
            calls.append(x)
            return x, torch.full((x.numel() // x.shape[-1],), 0.5)

    model.layer._shared_by_pre_op['input_layernorm'] = CountingQuant()
    x = torch.randn(2, 4, 3)
    rope = (torch.ones(2, 4, 3), torch.zeros(2, 4, 3))
    output = model.layer(x, position_embeddings=rope)
    assert output.shape == (2, 4, 3)
    assert len(calls) == 1

    attention_members = [
        model.layer.self_attn.q_proj,
        model.layer.self_attn.k_proj,
        model.layer.self_attn.v_proj,
    ]
    assert all(
        member.last_input is attention_members[0].last_input
        for member in attention_members[1:]
    )
    assert all(
        torch.equal(member.last_scale, attention_members[0].last_scale)
        for member in attention_members[1:]
    )

    mlp_members = [model.layer.mlp.up_proj, model.layer.mlp.gate_proj]
    assert mlp_members[0].last_input is mlp_members[1].last_input
    assert all(member.last_scale is None for member in mlp_members)


def test_shared_quantize_uses_granularity_axis_and_one_offset_per_scale(monkeypatch):
    import sys
    from amct_pytorch.classic.deploy_op.shared_quantize import SharedQuantize

    calls = []
    monkeypatch.setattr(
        sys.modules['torch_npu'],
        'npu_quantize',
        lambda x, scale, offset, **kwargs: calls.append((scale, offset, kwargs)),
    )
    for act_type in ('int8', 'int4'):
        for granularity, axis in (('tensor', -1), ('token', -2)):
            scales = (
                torch.tensor([0.1, 0.2])
                if granularity == 'token'
                else torch.tensor([0.1])
            )
            offsets = torch.arange(scales.numel())
            shared = SharedQuantize(
                scales, offsets, act_type, act_granularity=granularity
            )
            shared(torch.randn(1, 2, 4))
            assert calls[-1][2]['axis'] == axis
            assert calls[-1][2]['dtype'] == (
                torch.int8 if act_type == 'int8' else sys.modules['torch_npu'].int4
            )
            assert calls[-1][1].numel() == scales.numel()


def test_fused_preop_loads_legacy_norm_paths_with_fused_parameters():
    from amct_pytorch.classic.deploy_op.shared_quantize import (
        FusedPreOp,
        SharedQuantize,
    )

    source = nn.Module()
    source.norm = FusedPreOp(nn.LayerNorm(3), SharedQuantize(torch.tensor([0.2])))
    state = source.state_dict()
    legacy_state = {
        key.replace('norm.pre_op.', 'norm.'): value.clone()
        for key, value in state.items()
    }
    target = nn.Module()
    target.norm = FusedPreOp(nn.LayerNorm(3), SharedQuantize(torch.tensor([0.1])))
    target.load_state_dict(legacy_state, strict=True)
    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, state[key])


def test_shared_buffers_follow_deployment_member_device():
    model = NormModel().to('meta')
    group = _group(model.pre, [model.m0, model.m1], algorithm='minmax')
    registry = FusionRegistry()
    registry.add_group(group)
    model._amct_fusion_registry = registry
    ApplyScalingFusionPass().run(model)
    assert model.pre.shared.act_scale.device.type == 'meta'
    assert model.pre.shared.act_offset.device.type == 'meta'


def test_shared_token_matmul_flattens_batches_and_preserves_static_scales(monkeypatch):
    import sys
    from types import SimpleNamespace
    from amct_pytorch.classic.deploy_op.npu_quantization_linear import (
        NpuQuantizationLinear,
    )
    from amct_pytorch.classic.quantize_op import MinMaxQuant

    def matmul(x, weight, *, scale, pertoken_scale, bias, output_dtype):
        assert x.ndim == 2
        assert pertoken_scale.numel() == x.shape[0]
        return ((x.float() @ weight.float()) * scale * pertoken_scale[:, None]).to(
            output_dtype
        )

    matmul._schemas = {'default': SimpleNamespace(arguments=[])}
    monkeypatch.setattr(sys.modules['torch_npu'], 'npu_quant_matmul', matmul)
    for dynamic in (False, True):
        config = {
            'batch_num': 1,
            'algorithm': {'minmax': {}},
            'inputs_cfg': {
                'quant_type': 'int8',
                'symmetric': True,
                'strategy': 'token',
                'dynamic': dynamic,
            },
            'weights_cfg': {
                'quant_type': 'int8',
                'symmetric': True,
                'strategy': 'channel',
            },
        }
        quant = MinMaxQuant(nn.Linear(4, 3, bias=False), 'linear', config)
        quant(torch.randn(2, 2, 4))
        quant.input_quant_disabled = True
        deploy = NpuQuantizationLinear(quant)
        x = torch.ones(2, 2, 4, dtype=torch.int8)
        runtime_scale = torch.tensor([[0.1, 0.2], [0.3, 0.4]]) if dynamic else None
        before = None if dynamic else deploy.pertoken_scale.clone()
        actual = deploy(x, runtime_scale)
        scales = runtime_scale if dynamic else before.reshape(1, 2).expand(2, 2)
        expected = (
            (x.float() @ deploy.quantized_weight.float())
            * deploy.deq_scale
            * scales[..., None]
        )
        torch.testing.assert_close(actual, expected)
        if not dynamic:
            torch.testing.assert_close(deploy.pertoken_scale, before)


def test_shared_dynamic_quantize_rejects_unsupported_dtype(monkeypatch):
    import sys
    from types import SimpleNamespace
    from amct_pytorch.classic.deploy_op.shared_quantize import SharedDynamicQuantize

    calls = []

    def dynamic_quant(x, *, dst_type, **kwargs):
        calls.append((dst_type, kwargs))
        return x, torch.ones_like(x[..., 0])

    torch_npu = sys.modules['torch_npu']
    monkeypatch.setattr(torch_npu, 'npu_dynamic_quant', dynamic_quant)
    monkeypatch.setattr(torch_npu, 'hifloat8', SimpleNamespace(marker='hifloat8'))
    SharedDynamicQuantize('int8')(torch.randn(1, 2, 3))
    SharedDynamicQuantize('hifloat8')(torch.randn(1, 2, 3))
    assert calls[-1][1] == {'dst_type_max': 15}
    with pytest.raises(ValueError, match='Unsupported activation dtype'):
        SharedDynamicQuantize('int4')(torch.randn(1, 2, 3))
