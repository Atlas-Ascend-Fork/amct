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
from amct_pytorch.classic.optimizer.model_structure import MODEL_STRUCTURES

import torch
from torch import nn

from amct_pytorch.classic.quantize import quantize


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


def test_real_minmax_relation_is_identified_and_calibrated_after_first_forward():
    model = TinyModel().to(torch.float16)
    quantize(model, _config('minmax'))
    model(torch.randn(2, 3, 8, dtype=torch.float16))

    norms = [
        model.layers[0].input_layernorm,
        model.layers[0].post_attention_layernorm,
    ]
    assert len(norms) == 2
    assert model.layers[0].self_attn.q_proj.shared_param_mode == 'static'
    assert model.layers[0].self_attn.q_proj.scale_d is not None
    assert model.layers[0].mlp.up_proj.shared_param_mode == 'static'
    assert model.layers[0].mlp.up_proj.scale_d is not None


def test_real_smoothquant_relation_uses_string_algorithm_and_calibrates_after_two_forwards():
    model = TinyModel().to(torch.float16)
    quantize(model, _config('smoothquant', batch_num=2))
    model(torch.randn(2, 3, 8, dtype=torch.float16))
    model(torch.randn(2, 3, 8, dtype=torch.float16))

    norms = [
        model.layers[0].input_layernorm,
        model.layers[0].post_attention_layernorm,
    ]
    assert len(norms) == 2
    assert model.layers[0].self_attn.q_proj.shared_param_mode == 'static'
    assert model.layers[0].self_attn.q_proj.scale is not None
    assert model.layers[0].mlp.up_proj.shared_param_mode == 'static'
    assert model.layers[0].mlp.up_proj.scale is not None


def register_test_structures(monkeypatch):
    structures = dict(MODEL_STRUCTURES._structures)
    structures['qwen3'] = replace(
        structures['qwen3'],
        layer_types=(f'{TinyQwenLayer.__module__}.{TinyQwenLayer.__qualname__}',),
    )
    monkeypatch.setattr(MODEL_STRUCTURES, '_structures', structures)


register_test_structures = pytest.fixture(autouse=True)(register_test_structures)


@pytest.mark.parametrize('reason', ['partial', 'config', 'missing'])
def test_invalid_candidate_keeps_independent_members_and_other_relation(reason):
    from amct_pytorch.classic.optimizer.layer_norm_quant import (
        LayerNormQuant,
    )
    from amct_pytorch.classic.optimizer import (
        InsertQuantizeModulePass,
        InsertLayerNormQuantPass,
    )
    from amct_pytorch.common.config import parse_config
    from amct_pytorch.algorithms import AlgorithmRegistry

    model = TinyModel().half()
    config = _config('smoothquant')
    if reason == 'partial':
        config['skip_layers'] = {'k_proj'}
    InsertQuantizeModulePass(parse_config(model, config, AlgorithmRegistry)).run(model)
    if reason == 'config':
        import copy

        model.layers[0].self_attn.k_proj.quant_config = copy.deepcopy(
            model.layers[0].self_attn.k_proj.quant_config
        )
        model.layers[0].self_attn.k_proj.quant_config['algorithm']['smoothquant'][
            'smooth_strength'
        ] = 0.3
    if reason == 'missing':
        del model.layers[0].self_attn.k_proj
    InsertLayerNormQuantPass().run(model)
    assert not isinstance(model.layers[0].input_layernorm, LayerNormQuant)
    assert isinstance(model.layers[0].post_attention_layernorm, LayerNormQuant)
    assert model.layers[0].self_attn.q_proj.shared_param_mode is None


def test_unregistered_structure_is_not_wrapped(monkeypatch):
    from amct_pytorch.classic.optimizer.layer_norm_quant import LayerNormQuant

    monkeypatch.setattr(MODEL_STRUCTURES, '_structures', {})
    model = TinyModel().half()
    quantize(model, _config('minmax'))
    model(torch.randn(2, 3, 8, dtype=torch.float16))
    assert not any(isinstance(m, LayerNormQuant) for m in model.modules())
    assert model.layers[0].self_attn.q_proj.scale_d is not None


def test_non_int8_activation_candidate_is_not_wrapped():
    from amct_pytorch.classic.optimizer.layer_norm_quant import (
        LayerNormQuant,
    )
    from amct_pytorch.classic.optimizer import (
        InsertQuantizeModulePass,
        InsertLayerNormQuantPass,
    )
    from amct_pytorch.common.config import parse_config
    from amct_pytorch.algorithms import AlgorithmRegistry

    model = TinyModel().half()
    InsertQuantizeModulePass(
        parse_config(model, _config('minmax'), AlgorithmRegistry)
    ).run(model)
    for member in (
        model.layers[0].self_attn.q_proj,
        model.layers[0].self_attn.k_proj,
        model.layers[0].self_attn.v_proj,
    ):
        member.act_type = 'hifloat8'
        member.quant_config['inputs_cfg']['quant_type'] = 'hifloat8'
    InsertLayerNormQuantPass().run(model)
    assert not isinstance(model.layers[0].input_layernorm, LayerNormQuant)
    assert isinstance(model.layers[0].post_attention_layernorm, LayerNormQuant)
