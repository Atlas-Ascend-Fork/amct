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
from amct_pytorch.classic.optimizer.base_module_fusion_pass import (
    BaseModuleFusionPass,
)
from amct_pytorch.common.utils.model_util import ModuleHelper


class Member(nn.Module):
    def __init__(self, smooth=True):
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.weight = self.linear.weight
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


def test_scaling_fusion_pass_uses_module_fusion_interface():
    model = NormModel()
    from amct_pytorch.classic.optimizer.layer_norm_quant import LayerNormQuant

    wrapper = LayerNormQuant(
        pre_op=model.pre,
        scaling_layers=(model.m0, model.m1),
        member_refs=('m0', 'm1'),
        target_ref='m0',
        adapter='qwen3_attention',
        algorithm='minmax',
        act_granularity='tensor',
        batch_num=1,
    )
    fusion_pass = ApplyScalingFusionPass()

    assert isinstance(fusion_pass, BaseModuleFusionPass)
    assert fusion_pass.match_pattern(wrapper, 'pre')
    assert not fusion_pass.match_pattern(model.pre, 'pre')


def test_replace_module_rejects_empty_module_name():
    model = NormModel()
    with pytest.raises(ValueError, match='module name must be a non-empty string'):
        ModuleHelper.replace_module_by_name(model, '', nn.Linear(3, 3))


def test_replace_module_rejects_non_string_module_name():
    model = NormModel()
    with pytest.raises(ValueError, match='module name must be a non-empty string'):
        ModuleHelper.replace_module_by_name(model, None, nn.Linear(3, 3))


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
    monkeypatch.setitem(sys.modules, 'torch_npu', mock_npu)
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
    from amct_pytorch.classic.optimizer.layer_norm_quant import LayerNormQuant

    model.pre = LayerNormQuant(
        pre_op=model.pre,
        scaling_layers=(model.m0, model.m1),
        member_refs=('m0', 'm1'),
        target_ref='m0',
        adapter='qwen3_attention',
        algorithm='minmax',
        act_granularity='tensor',
        batch_num=1,
    )
    model.pre.status = 'calibrated'
    model.pre.activation_scale = torch.tensor([0.1])
    model.pre.activation_offset = None
    for member in (model.m0, model.m1):
        member.shared_param_mode = 'static'
    ApplyScalingFusionPass().run(model)
    assert model.pre.shared.act_scale.device.type == 'meta'
    assert model.pre.shared.act_offset.device.type == 'meta'


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
