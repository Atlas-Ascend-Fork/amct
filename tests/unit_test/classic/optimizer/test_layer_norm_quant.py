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

from amct_pytorch.classic.optimizer.layer_norm_quant import LayerNormQuant
from amct_pytorch.classic.quantize_op import MinMaxQuant, SmoothQuant


def _config(algorithm='smoothquant', strategy='tensor', dynamic=None):
    inputs = {
        'enable_quant': True,
        'quant_type': 'int8',
        'symmetric': True,
        'strategy': strategy,
    }
    if dynamic is not None:
        inputs['dynamic'] = dynamic
    return {
        'batch_num': 1,
        'weights_cfg': {'quant_type': 'int8', 'symmetric': True, 'strategy': 'channel'},
        'inputs_cfg': inputs,
        'algorithm': (
            {'smoothquant': {'smooth_strength': 0.5}}
            if algorithm == 'smoothquant'
            else {'minmax': {}}
        ),
    }


class TinyLayer(nn.Module):
    def __init__(self, algorithm='smoothquant', strategy='tensor', dynamic=None):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(4)
        self.self_attn = nn.Module()
        quant_cls = SmoothQuant if algorithm == 'smoothquant' else MinMaxQuant
        self.self_attn.q_proj = quant_cls(
            nn.Linear(4, 4), 'self_attn.q_proj', _config(algorithm, strategy, dynamic)
        )
        self.self_attn.k_proj = quant_cls(
            nn.Linear(4, 4), 'self_attn.k_proj', _config(algorithm, strategy, dynamic)
        )

    def forward(self, x):
        h = self.input_layernorm(x)
        return self.self_attn.q_proj(h) + self.self_attn.k_proj(h)


def _layer_norm_quant(layer, algorithm='smoothquant', strategy='tensor', dynamic=None):
    return LayerNormQuant(
        pre_op=layer.input_layernorm,
        scaling_layers=(layer.self_attn.q_proj, layer.self_attn.k_proj),
        member_refs=('self_attn.q_proj', 'self_attn.k_proj'),
        target_ref='self_attn',
        adapter='qwen3_attention',
        algorithm=algorithm,
        act_granularity=strategy,
        dynamic_per_token=dynamic is True,
        batch_num=1,
    )


def test_layer_norm_quant_does_not_register_members_as_children():
    layer = TinyLayer()
    layer.input_layernorm = _layer_norm_quant(layer)

    assert isinstance(layer.input_layernorm.pre_op, nn.LayerNorm)
    assert layer.input_layernorm.member_refs == (
        'self_attn.q_proj',
        'self_attn.k_proj',
    )
    assert set(dict(layer.input_layernorm.named_children())) == {'pre_op'}
    assert set(layer.input_layernorm.state_dict()) == {'pre_op.weight', 'pre_op.bias'}
    assert layer.input_layernorm.scaling_layers[0] is layer.self_attn.q_proj
    assert layer.input_layernorm.scaling_layers[1] is layer.self_attn.k_proj


def test_shared_smooth_and_activation_params_are_written_to_members():
    layer = TinyLayer()
    layer.input_layernorm = _layer_norm_quant(layer)
    x = torch.randn(2, 4)

    h = layer.input_layernorm(x)
    assert layer.input_layernorm.status == 'calibrated'
    assert layer.input_layernorm.batch_count == 1
    assert layer.input_layernorm.smooth_factor is not None

    layer.self_attn.q_proj(h)
    layer.self_attn.k_proj(h)
    assert layer.self_attn.q_proj.shared_param_mode == 'static'
    assert layer.self_attn.k_proj.shared_param_mode == 'static'
    assert torch.equal(
        layer.self_attn.q_proj.scale, layer.input_layernorm.smooth_factor
    )
    assert torch.equal(
        layer.self_attn.k_proj.scale, layer.input_layernorm.smooth_factor
    )
    assert torch.equal(
        layer.self_attn.q_proj.scale_d, layer.input_layernorm.activation_scale
    )
    assert torch.equal(
        layer.self_attn.k_proj.scale_d, layer.input_layernorm.activation_scale
    )


def test_dynamic_group_writes_smooth_but_not_static_activation_scale():
    layer = TinyLayer(strategy='token', dynamic=True)
    layer.input_layernorm = _layer_norm_quant(layer, strategy='token', dynamic=True)
    x = torch.randn(2, 4)

    h = layer.input_layernorm(x)
    layer.self_attn.q_proj(h)
    layer.self_attn.k_proj(h)

    assert layer.input_layernorm.dynamic_per_token is True
    assert layer.input_layernorm.activation_scale is None
    assert layer.self_attn.q_proj.shared_param_mode == 'dynamic'
    assert layer.self_attn.q_proj.scale is not None
    assert layer.self_attn.q_proj.scale_d is None
    assert layer.self_attn.k_proj.scale_d is None


def test_minmax_dynamic_group_only_marks_shared_mode():
    layer = TinyLayer(algorithm='minmax', strategy='token', dynamic=True)
    for name in ('q_proj', 'k_proj'):
        setattr(
            layer.self_attn,
            name,
            MinMaxQuant(
                nn.Linear(4, 4),
                f'self_attn.{name}',
                _config('minmax', 'token', True),
            ),
        )
    layer.input_layernorm = _layer_norm_quant(
        layer, algorithm='minmax', strategy='token', dynamic=True
    )

    h = layer.input_layernorm(torch.randn(2, 4))
    layer.self_attn.q_proj(h)
    layer.self_attn.k_proj(h)

    assert layer.self_attn.q_proj.shared_param_mode == 'dynamic'
    assert layer.self_attn.q_proj.scale_d is None
    assert layer.self_attn.k_proj.scale_d is None


def test_finalize_executes_once_and_releases_statistics():
    from unittest.mock import patch

    layer = TinyLayer()
    layer.input_layernorm = _layer_norm_quant(layer)
    wrapper = layer.input_layernorm
    with patch.object(
        wrapper, 'finalize_quant_params', wraps=wrapper.finalize_quant_params
    ) as finalize:
        layer(torch.randn(2, 4))
        layer(torch.randn(2, 4))
        assert finalize.call_count == 1
    assert wrapper.act_max is None
    assert wrapper.act_min is None


def test_dynamic_per_token_keeps_activation_unquantized_before_convert():
    from unittest.mock import patch
    import importlib

    for algorithm in ('smoothquant', 'minmax'):
        layer = TinyLayer(algorithm=algorithm, strategy='token', dynamic=True)
        layer.input_layernorm = _layer_norm_quant(layer, algorithm, 'token', True)
        layer(torch.randn(2, 4))
        path = 'smooth_quant_module' if algorithm == 'smoothquant' else 'minmax_module'
        module = importlib.import_module('amct_pytorch.classic.quantize_op.' + path)
        with patch.object(
            module, 'quant_dequant_tensor', wraps=module.quant_dequant_tensor
        ) as quant:
            layer(torch.randn(2, 4))
            assert quant.call_count == 0


def test_deepcopy_calibration_does_not_modify_source_members():
    import copy

    source = TinyLayer()
    source.input_layernorm = _layer_norm_quant(source)
    copied = copy.deepcopy(source)
    copied(torch.randn(2, 4))
    assert copied.input_layernorm.status == 'calibrated'
    assert copied.self_attn.q_proj.scale is not None
    assert source.input_layernorm.status == 'pending'
    assert source.self_attn.q_proj.scale is None
    assert copied.input_layernorm.scaling_layers[0] is copied.self_attn.q_proj
    assert copied.input_layernorm.scaling_layers[1] is copied.self_attn.k_proj
    assert copied.self_attn.k_proj.scale is not None
    assert source.self_attn.k_proj.scale is None


def test_common_smooth_uses_maximum_of_all_member_weights():
    layer = TinyLayer()
    layer.self_attn.q_proj.weight.data.fill_(1)
    layer.self_attn.k_proj.weight.data.fill_(4)
    layer.input_layernorm = _layer_norm_quant(layer)
    x = torch.randn(2, 3, 4)
    h = layer.input_layernorm.pre_op(x).detach()
    layer(x)
    expected = (h.abs().amax((0, 1), keepdim=False).reshape(1, -1) / 4).sqrt()
    torch.testing.assert_close(layer.input_layernorm.smooth_factor, expected)
    assert layer.self_attn.q_proj.scale_w is not None
    assert layer.self_attn.k_proj.scale_w is not None
    assert not torch.equal(
        layer.self_attn.q_proj.scale_w, layer.self_attn.k_proj.scale_w
    )


def test_activation_parameters_are_calculated_once_per_preop():
    from unittest.mock import patch
    import amct_pytorch.classic.optimizer.layer_norm_quant as module

    layer = TinyLayer()
    layer.input_layernorm = _layer_norm_quant(layer)
    with patch.object(
        module, 'calculate_scale_offset', wraps=module.calculate_scale_offset
    ) as calculate:
        layer(torch.randn(2, 4))
        assert calculate.call_count == 1


def test_convert_parameter_mismatch_keeps_independent_quantization():
    from amct_pytorch.classic.optimizer.apply_scaling_fusion_pass import (
        ApplyScalingFusionPass,
    )

    layer = TinyLayer()
    layer.input_layernorm = _layer_norm_quant(layer)
    layer(torch.randn(2, 4))
    norm = layer.input_layernorm.pre_op
    old_weight = norm.weight.clone()
    layer.self_attn.k_proj.shared_param_mode = None
    ApplyScalingFusionPass().run(layer)
    assert layer.input_layernorm is norm
    torch.testing.assert_close(norm.weight, old_weight)
    assert not getattr(layer.self_attn.q_proj, 'input_quant_disabled', False)


def test_convert_incomplete_calibration_does_not_partially_fuse():
    import pytest
    from amct_pytorch.classic.optimizer.apply_scaling_fusion_pass import (
        ApplyScalingFusionPass,
    )

    layer = TinyLayer()
    layer.input_layernorm = _layer_norm_quant(layer)
    before = layer.input_layernorm.pre_op.weight.clone()
    with pytest.raises(RuntimeError, match='has not finished calibration'):
        ApplyScalingFusionPass().run(layer)
    torch.testing.assert_close(layer.input_layernorm.pre_op.weight, before)
    assert isinstance(layer.input_layernorm, LayerNormQuant)


def test_weight_only_minmax_retains_fakequant_forward():
    config = _config('minmax')
    config['inputs_cfg'] = {'enable_quant': False}
    module = MinMaxQuant(nn.Linear(4, 3), 'linear', config)
    result = module(torch.randn(2, 4))
    assert result.shape == (2, 3)
    assert torch.isfinite(result).all()


def test_static_token_does_not_build_shared_group():
    from amct_pytorch.classic.optimizer.model_structure import (
        build_layer_norm_quant_specs,
    )

    layer = TinyLayer(strategy='token', dynamic=False)
    assert build_layer_norm_quant_specs(layer) == []


def test_pre_op_whitelist_matches_exact_type_paths():
    from amct_pytorch.classic.optimizer.model_structure import _is_norm

    class FakeQwen3RMSNorm(nn.Module):
        pass

    FakeQwen3RMSNorm.__module__ = 'transformers.models.qwen3.modeling_qwen3'
    FakeQwen3RMSNorm.__qualname__ = 'Qwen3RMSNorm'

    assert _is_norm(nn.LayerNorm(4))
    assert _is_norm(FakeQwen3RMSNorm())
    assert not _is_norm(nn.Identity())


def test_static_token_layer_norm_quant_is_rejected():
    import pytest

    layer = TinyLayer(strategy='token', dynamic=False)
    with pytest.raises(ValueError, match='static per-token quantization'):
        _layer_norm_quant(layer, strategy='token', dynamic=False)
