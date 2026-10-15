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

from amct_pytorch.classic.deploy_op.shared_quantize import FusedPreOp
from amct_pytorch.classic.deploy_op.local_dynamic import (
    _HOOK_DICT_NAMES,
    make_local_adapter,
)
from amct_pytorch.classic.optimizer.apply_scaling_fusion_pass import (
    ApplyScalingFusionPass,
)
from amct_pytorch.classic.optimizer.layer_norm_quant import LayerNormQuant


class Member(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.weight = self.linear.weight
        self.act_type = 'int8'
        self.scale = torch.tensor([[1.0, 2.0, 4.0, 8.0]])
        self.scale_d = torch.tensor([0.1])
        self.offset_d = None
        self.input_quant_disabled = False
        self.smooth_scale_skipped = False
        self.shared_param_mode = None

    def forward(self, x, pertoken_scale=None):
        return self.linear(x)


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(4)
        self.self_attn = nn.Module()
        self.self_attn.q_proj = Member()
        self.self_attn.k_proj = Member()
        self.self_attn.v_proj = Member()
        self.post_attention_layernorm = nn.LayerNorm(4)
        self.mlp = nn.Module()
        self.mlp.gate_proj = Member()
        self.mlp.up_proj = Member()

    def forward(self, x):
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = Layer()

    def forward(self, x):
        return self.layer(x)


def _dynamic_norm_quant(model, norm_name, members, target_ref, adapter):
    old_norm = model.get_submodule(norm_name)
    module = LayerNormQuant(
        pre_op=old_norm,
        scaling_layers=tuple(model.get_submodule(ref) for ref in members),
        member_refs=members,
        target_ref=target_ref,
        adapter=adapter,
        algorithm='smoothquant',
        act_granularity='token',
        batch_num=1,
        dynamic_per_token=True,
    )
    for member in module.scaling_layers:
        member.scale_d = None
        member.shared_param_mode = 'dynamic'
    return module


def test_dynamic_fusion_replaces_attention_and_mlp_without_replacing_layer():
    model = Model()
    model.layer.input_layernorm = _dynamic_norm_quant(
        model,
        'layer.input_layernorm',
        (
            'layer.self_attn.q_proj',
            'layer.self_attn.k_proj',
            'layer.self_attn.v_proj',
        ),
        'layer.self_attn',
        'qwen3_attention',
    )
    model.layer.post_attention_layernorm = _dynamic_norm_quant(
        model,
        'layer.post_attention_layernorm',
        ('layer.mlp.gate_proj', 'layer.mlp.up_proj'),
        'layer.mlp',
        'qwen3_mlp',
    )
    for norm in (model.layer.input_layernorm, model.layer.post_attention_layernorm):
        norm.status = 'calibrated'
        norm.smooth_factor = torch.tensor([[1.0, 2.0, 4.0, 8.0]])

    ApplyScalingFusionPass().run(model)

    assert isinstance(model.layer.input_layernorm, nn.LayerNorm)
    assert isinstance(model.layer.post_attention_layernorm, nn.LayerNorm)
    assert type(model.layer) is Layer
    assert hasattr(model.layer.self_attn, 'shared_quantize')
    assert hasattr(model.layer.mlp, 'shared_quantize')
    assert model.layer.self_attn.q_proj.input_quant_disabled
    assert model.layer.mlp.gate_proj.input_quant_disabled


def test_local_adapter_isolates_hook_registration():
    original = nn.Module()
    original.register_forward_hook(lambda *args: None)

    fused = make_local_adapter('qwen3_attention', original, 'int8')

    for name in _HOOK_DICT_NAMES:
        assert getattr(fused, name) is not getattr(original, name)
    assert len(fused._forward_hooks) == len(original._forward_hooks)

    fused.register_forward_hook(lambda *args: None)
    original.register_forward_hook(lambda *args: None)

    assert len(original._forward_hooks) == 2
    assert len(fused._forward_hooks) == 2


def test_static_fusion_keeps_fused_pre_op_and_disables_member_inputs():
    model = Model()
    model.layer.input_layernorm = LayerNormQuant(
        pre_op=model.layer.input_layernorm,
        scaling_layers=(model.layer.self_attn.q_proj, model.layer.self_attn.k_proj),
        member_refs=('layer.self_attn.q_proj', 'layer.self_attn.k_proj'),
        target_ref='layer.self_attn',
        adapter='qwen3_attention',
        algorithm='smoothquant',
        act_granularity='tensor',
        batch_num=1,
    )
    model.layer.input_layernorm.status = 'calibrated'
    model.layer.input_layernorm.smooth_factor = torch.tensor([[1.0, 2.0, 4.0, 8.0]])
    model.layer.input_layernorm.activation_scale = torch.tensor([0.1])
    for member in (
        model.layer.self_attn.q_proj,
        model.layer.self_attn.k_proj,
    ):
        member.shared_param_mode = 'static'

    ApplyScalingFusionPass().run(model)

    assert isinstance(model.layer.input_layernorm, FusedPreOp)
    assert model.layer.self_attn.q_proj.input_quant_disabled
    assert model.layer.self_attn.k_proj.input_quant_disabled


def test_adapter_failure_keeps_norm_and_weights_unmodified(monkeypatch):
    import amct_pytorch.classic.optimizer.apply_scaling_fusion_pass as fusion

    model = Model()
    model.layer.input_layernorm = _dynamic_norm_quant(
        model,
        'layer.input_layernorm',
        ('layer.self_attn.q_proj', 'layer.self_attn.k_proj'),
        'layer.self_attn',
        'qwen3_attention',
    )
    wrapper = model.layer.input_layernorm
    wrapper.status = 'calibrated'
    wrapper.smooth_factor = torch.tensor([[1.0, 2.0, 4.0, 8.0]])
    before = {name: p.clone() for name, p in model.named_parameters()}

    def unavailable(*args):
        raise ValueError('adapter unavailable')

    monkeypatch.setattr(fusion, 'build_local_adapter', unavailable)
    ApplyScalingFusionPass().run(model)
    for name, p in model.named_parameters():
        old_name = name.replace('input_layernorm.', 'input_layernorm.pre_op.')
        torch.testing.assert_close(p, before[old_name])
    assert not model.layer.self_attn.q_proj.input_quant_disabled
    assert not model.layer.self_attn.q_proj.smooth_scale_skipped
    assert isinstance(model.layer.input_layernorm, nn.LayerNorm)


def test_replacement_failure_rolls_back_parameters_and_target(monkeypatch):
    import amct_pytorch.classic.optimizer.apply_scaling_fusion_pass as fusion
    from amct_pytorch.common.utils.model_util import (
        ModuleHelper as RealModuleHelper,
    )

    model = Model()
    model.layer.input_layernorm = _dynamic_norm_quant(
        model,
        'layer.input_layernorm',
        (
            'layer.self_attn.q_proj',
            'layer.self_attn.k_proj',
            'layer.self_attn.v_proj',
        ),
        'layer.self_attn',
        'qwen3_attention',
    )
    wrapper = model.layer.input_layernorm
    wrapper.status = 'calibrated'
    wrapper.smooth_factor = torch.tensor([[1.0, 2.0, 4.0, 8.0]])
    original_target = model.layer.self_attn
    before = {
        id(parameter): parameter.detach().clone() for parameter in model.parameters()
    }
    failed = {'once': False}

    class FlakyModuleHelper:
        @staticmethod
        def replace_module_by_name(model, name, mod):
            if (
                name == 'layer.input_layernorm'
                and mod is wrapper.pre_op
                and not failed['once']
            ):
                failed['once'] = True
                raise RuntimeError('wrapper replacement failed')
            RealModuleHelper.replace_module_by_name(model, name, mod)

    monkeypatch.setattr(fusion, 'ModuleHelper', FlakyModuleHelper)
    ApplyScalingFusionPass().run(model)

    assert model.layer.self_attn is original_target
    assert model.layer.input_layernorm is wrapper.pre_op
    assert not hasattr(original_target, 'shared_quantize')
    assert not original_target.q_proj.input_quant_disabled
    assert not original_target.q_proj.smooth_scale_skipped
    for parameter in model.parameters():
        torch.testing.assert_close(parameter, before[id(parameter)])


def test_shared_parameter_mode_mismatch_skips_fusion_without_weight_changes():
    model = Model()
    model.layer.input_layernorm = LayerNormQuant(
        model.layer.input_layernorm,
        (model.layer.self_attn.q_proj, model.layer.self_attn.k_proj),
        ('layer.self_attn.q_proj', 'layer.self_attn.k_proj'),
        'layer.self_attn',
        'qwen3_attention',
        'smoothquant',
        'tensor',
        1,
    )
    wrapper = model.layer.input_layernorm
    wrapper.status = 'calibrated'
    wrapper.smooth_factor = torch.tensor([[1.0, 2.0, 4.0, 8.0]])
    wrapper.activation_scale = torch.tensor([0.1])
    before = {id(value): value.clone() for value in model.parameters()}

    ApplyScalingFusionPass().run(model)

    assert isinstance(model.layer.input_layernorm, nn.LayerNorm)
    for value in model.parameters():
        torch.testing.assert_close(value, before[id(value)])
    assert not model.layer.self_attn.q_proj.smooth_scale_skipped


def test_single_member_static_smooth_retains_input_quantization():
    model = Model()
    model.layer.input_layernorm = LayerNormQuant(
        model.layer.input_layernorm,
        (model.layer.self_attn.q_proj,),
        ('layer.self_attn.q_proj',),
        'layer.self_attn',
        'qwen3_attention',
        'smoothquant',
        'tensor',
        1,
    )
    wrapper = model.layer.input_layernorm
    wrapper.status = 'calibrated'
    wrapper.smooth_factor = torch.tensor([[1.0, 2.0, 4.0, 8.0]])
    wrapper.activation_scale = torch.tensor([0.1])
    wrapper.scaling_layers[0].shared_param_mode = 'static'
    ApplyScalingFusionPass().run(model)
    assert not model.layer.self_attn.q_proj.input_quant_disabled
    assert model.layer.self_attn.q_proj.smooth_scale_skipped
    assert isinstance(model.layer.input_layernorm, nn.LayerNorm)


def test_smooth_fusion_preserves_norm_bias_and_projection_algebra():
    model = Model()
    model.layer.input_layernorm.bias.data.copy_(torch.tensor([1.0, -2.0, 3.0, -4.0]))
    original_norm = model.layer.input_layernorm
    x = torch.randn(2, 3, 4)
    members = (model.layer.self_attn.q_proj, model.layer.self_attn.k_proj)
    expected = [member(original_norm(x)).detach() for member in members]
    model.layer.input_layernorm = LayerNormQuant(
        original_norm,
        members,
        ('layer.self_attn.q_proj', 'layer.self_attn.k_proj'),
        'layer.self_attn',
        'qwen3_attention',
        'smoothquant',
        'tensor',
        1,
    )
    wrapper = model.layer.input_layernorm
    wrapper.status = 'calibrated'
    wrapper.smooth_factor = members[0].scale
    wrapper.activation_scale = members[0].scale_d
    ApplyScalingFusionPass().run(model)
    # Isolate the exact smooth algebra from the separate rounding operation.
    model.layer.input_layernorm.shared = nn.Identity()
    actual = [member(model.layer.input_layernorm(x)) for member in members]
    for actual_output, expected_output in zip(actual, expected):
        torch.testing.assert_close(actual_output, expected_output)
