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

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
import pytest
import torch
import torch.nn as nn

from amct_pytorch.algorithms.quant.awq import (
    apply_scale,
    calculate_scale_offset_by_granularity,
    process_weights_for_layers,
    search_scale,
)


def test_calculate_scale_offset_by_granularity_int_symmetric():
    weight = torch.randn(8, 32, dtype=torch.float32)
    quant_config = {
        "weights_cfg": {
            "quant_type": "int8",
            "symmetric": True,
            "strategy": "channel",
        }
    }
    scale, offset = calculate_scale_offset_by_granularity(weight, quant_config)
    assert scale.shape == (8, 1)
    assert offset is None


def test_calculate_scale_offset_by_granularity_int_asymmetric():
    weight = torch.randn(8, 32, dtype=torch.float32)
    quant_config = {
        "weights_cfg": {
            "quant_type": "int8",
            "symmetric": False,
            "strategy": "channel",
        }
    }
    scale, offset = calculate_scale_offset_by_granularity(weight, quant_config)
    assert scale.shape == (8, 1)
    assert offset.shape == (8, 1)


def test_apply_scale_updates_weight_and_input():
    weight = torch.randn(4, 8, dtype=torch.float32)
    scale = torch.full((1, 8), 2.0, dtype=torch.float32)
    inp = torch.randn(2, 4, 8, dtype=torch.float32)
    weight_before = weight.clone()
    input_before = inp.clone()

    class FakeModule:
        def __init__(self):
            self.weight = None

    mod = FakeModule()
    mod.weight = torch.nn.Parameter(weight_before.clone())
    apply_scale(scale, mod, inp)
    assert torch.allclose(mod.weight.data, weight_before * 2.0)
    assert torch.allclose(inp, input_before / 2.0)


def test_process_weights_for_layers_int8(monkeypatch):
    qdq_calls = []

    def fake_qdq(tensor, wts_type, scale, offset, group_size):
        qdq_calls.append((wts_type, group_size))
        return tensor

    monkeypatch.setattr(
        "amct_pytorch.algorithms.quant.awq.quant_dequant_tensor", fake_qdq
    )

    layer = nn.Linear(4, 8)
    quant_config = {
        "weights_cfg": {
            "quant_type": "int8",
            "strategy": "channel",
            "group_size": None,
            "symmetric": True,
        }
    }
    scale_awq = torch.full((1, 4), 2.0)
    process_weights_for_layers([layer], scale_awq, quant_config)
    assert len(qdq_calls) == 1
    assert qdq_calls[0][0] == "int8"


def test_process_weights_for_layers_mxfp4(monkeypatch):
    qdq_calls = []

    def fake_qdq(tensor, wts_type, group_size=None):
        qdq_calls.append((wts_type, group_size))
        return tensor

    monkeypatch.setattr(
        "amct_pytorch.algorithms.quant.awq.quant_dequant_tensor", fake_qdq
    )
    from amct_pytorch.common.utils.vars import MXFP4_E2M1

    layer = nn.Linear(8, 32)
    quant_config = {
        "weights_cfg": {
            "quant_type": MXFP4_E2M1,
            "strategy": "channel",
            "group_size": 32,
        }
    }
    scale_awq = torch.full((1, 8), 2.0)
    process_weights_for_layers([layer], scale_awq, quant_config)
    assert len(qdq_calls) == 1
    assert qdq_calls[0][0] == MXFP4_E2M1


def test_search_scale_grid_returns_best_scale(monkeypatch):
    weight = nn.Parameter(torch.ones(4, 8))
    inputs = torch.randn(2, 4, 8)

    layer = nn.Module()
    layer.weight = nn.Parameter(weight.clone())

    block = nn.Module()
    block.weight = weight
    block.linear = nn.Linear(8, 8)

    ori_out_calls = []
    quant_out_calls = []

    def block_forward(x, **kwargs):
        if not ori_out_calls:
            ori_out_calls.append(1)
            return x * 2
        quant_out_calls.append(1)
        return x * 2.1

    block.forward = block_forward

    def fake_qdq(tensor, wts_type, scale, offset, group_size):
        return tensor

    monkeypatch.setattr(
        "amct_pytorch.algorithms.quant.awq.quant_dequant_tensor", fake_qdq
    )

    quant_config = {
        "algorithm": {"awq": {"grids_num": 3}},
        "weights_cfg": {
            "quant_type": "int8",
            "strategy": "channel",
            "group_size": None,
            "symmetric": True,
        },
    }

    scale = search_scale(inputs, [layer], block, quant_config)
    assert scale.shape == (1, 8)
    assert len(quant_out_calls) == 3


def test_search_scale_rejects_nan_input():
    inputs = torch.tensor([[float("nan"), 1.0]])
    layer = nn.Module()
    layer.weight = nn.Parameter(torch.ones(1, 2))
    block = nn.Module()
    with pytest.raises(RuntimeError, match="Invalid value.*activation"):
        search_scale(inputs, [layer], block, {"algorithm": {}, "weights_cfg": {}})


def test_search_scale_rejects_nan_weight():
    inputs = torch.ones(1, 2)
    layer = nn.Module()
    layer.weight = nn.Parameter(torch.tensor([[float("nan"), 1.0]]))
    block = nn.Module()
    with pytest.raises(RuntimeError, match="Invalid value.*weight"):
        search_scale(inputs, [layer], block, {"algorithm": {}, "weights_cfg": {}})


def test_search_scale_handles_tuple_block_output(monkeypatch):
    weight = nn.Parameter(torch.ones(4, 8))
    inputs = torch.randn(2, 4, 8)

    layer = nn.Module()
    layer.weight = nn.Parameter(weight.clone())

    block = nn.Module()
    block.weight = weight
    block.linear = nn.Linear(8, 8)

    ori_out_calls = []
    quant_out_calls = []

    def block_forward(x, **kwargs):
        if not ori_out_calls:
            ori_out_calls.append(1)
            return (x * 2, "aux")
        quant_out_calls.append(1)
        return (x * 2.1, "aux")

    block.forward = block_forward

    def fake_qdq(tensor, wts_type, scale, offset, group_size):
        return tensor

    monkeypatch.setattr(
        "amct_pytorch.algorithms.quant.awq.quant_dequant_tensor", fake_qdq
    )

    quant_config = {
        "algorithm": {"awq": {"grids_num": 3}},
        "weights_cfg": {
            "quant_type": "int8",
            "strategy": "channel",
            "group_size": None,
            "symmetric": True,
        },
    }

    scale = search_scale(inputs, [layer], block, quant_config)
    assert scale.shape == (1, 8)
    assert len(quant_out_calls) == 3


def test_search_best_scale_raises_on_invalid_loss():
    class _NanBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))

        def forward(self, x, **kwargs):
            return torch.tensor(float("nan"))

    block = _NanBlock()
    layer = nn.Module()
    layer.weight = nn.Parameter(torch.ones(2, 4))
    with pytest.raises(RuntimeError, match="Run AWQ error"):
        search_scale(
            torch.randn(2, 4, 4),
            [layer],
            block,
            {
                "algorithm": {"awq": {"grids_num": 3}},
                "weights_cfg": {
                    "quant_type": "int8",
                    "strategy": "channel",
                    "group_size": None,
                },
            },
        )
