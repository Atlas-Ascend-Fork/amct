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
import torch
import torch.nn as nn

from amct_pytorch.common.models.llm.common.capture import (
    Catcher,
    register_forward_hooks,
)


class _DecoderWithMeta(nn.Module):
    layer_type = "self_attention"
    attention_type = "softmax"

    def forward(self, x, **kwargs):
        return x


class _PlainDecoder(nn.Module):
    def forward(self, x, **kwargs):
        return x


# ---- Catcher --------------------------------------------------------------


def test_catcher_inherits_layer_and_attention_metadata():
    inner = _DecoderWithMeta()
    catcher = Catcher(inner, dataset=[])
    assert catcher.layer_type == "self_attention"
    assert catcher.attention_type == "softmax"


def test_catcher_layer_type_default_when_module_missing_attr():
    catcher = Catcher(_PlainDecoder(), dataset=[])
    # Default per the implementation is "linear_attention".
    assert catcher.layer_type == "linear_attention"
    assert catcher.attention_type is None


def test_catcher_forward_appends_input_to_dataset_then_raises():
    dataset = []
    catcher = Catcher(_PlainDecoder(), dataset=dataset)
    x = torch.tensor([[1.0, 2.0]])
    with pytest.raises(ValueError):
        catcher(x)
    assert len(dataset) == 1
    assert torch.equal(dataset[0], x)
    assert catcher.index == 1


def test_catcher_forward_captures_kwargs_on_first_pass():
    catcher = Catcher(_PlainDecoder(), dataset=[])
    pos = torch.tensor([[0, 1, 2]])
    mask = torch.tensor([[1, 1, 1]])
    pe = (torch.ones(1, 3, 4), torch.zeros(1, 3, 4))

    with pytest.raises(ValueError):
        catcher(
            torch.zeros(1, 3, 4),
            position_ids=pos,
            attention_mask=mask,
            position_embeddings=pe,
        )
    assert torch.equal(catcher.position_ids, pos)
    assert torch.equal(catcher.attention_mask, mask)
    assert catcher.position_embeddings is pe


def test_catcher_does_not_overwrite_already_captured_kwargs():
    catcher = Catcher(_PlainDecoder(), dataset=[])
    first_pos = torch.tensor([[0, 1, 2]])
    second_pos = torch.tensor([[9, 9, 9]])

    with pytest.raises(ValueError):
        catcher(torch.zeros(1, 3, 4), position_ids=first_pos)
    with pytest.raises(ValueError):
        catcher(torch.zeros(1, 3, 4), position_ids=second_pos)

    assert torch.equal(catcher.position_ids, first_pos)


def test_catcher_index_increments_per_call():
    dataset = []
    catcher = Catcher(_PlainDecoder(), dataset=dataset)
    for _ in range(3):
        with pytest.raises(ValueError):
            catcher(torch.zeros(1, 2))
    assert catcher.index == 3
    assert len(dataset) == 3


def test_catcher_proxies_inner_module_attribute_access():
    class _WithExtra(nn.Module):
        def __init__(self):
            super().__init__()
            self.extra = "marker"

        def forward(self, x):
            return x

    inner = _WithExtra()
    catcher = Catcher(inner, dataset=[])
    # `extra` lives on the inner module; Catcher's __getattr__ must forward.
    assert catcher.extra == "marker"


def test_catcher_dataset_stores_tensor_on_cpu():
    catcher = Catcher(_PlainDecoder(), dataset=[])
    x = torch.zeros(1, 2)  # already on CPU; the .to('cpu') call is a no-op
    with pytest.raises(ValueError):
        catcher(x)
    assert catcher.dataset[0].device.type == "cpu"


# ---- register_forward_hooks ----------------------------------------------


def _make_block_with_named_children():
    block = nn.Module()
    block.up_proj = nn.Linear(4, 8)
    block.gate_proj = nn.Linear(4, 8)
    block.down_proj = nn.Linear(8, 4)
    return block


def test_register_forward_hooks_filters_by_substring_match():
    block = _make_block_with_named_children()
    hooks = []
    act_stat = {}
    register_forward_hooks(block, target_name="up_proj", hooks=hooks, act_stat=act_stat)
    # Only the up_proj submodule matches the substring.
    assert len(hooks) == 1
    block.up_proj(torch.randn(2, 4))
    assert "up_proj_out" in act_stat
    # Other modules were not hooked.
    block.gate_proj(torch.randn(2, 4))
    assert "gate_proj_out" not in act_stat


def test_register_forward_hooks_writes_outputs_to_act_stat():
    block = _make_block_with_named_children()
    hooks, act_stat = [], {}
    register_forward_hooks(block, target_name="proj", hooks=hooks, act_stat=act_stat)
    # Substring "proj" matches all three children.
    assert len(hooks) == 3

    block.up_proj(torch.randn(2, 4))
    block.up_proj(torch.randn(2, 4))
    assert len(act_stat["up_proj_out"]) == 2
    assert all(t.device.type == "cpu" for t in act_stat["up_proj_out"])


def test_register_forward_hooks_handle_remove_stops_capture():
    block = _make_block_with_named_children()
    hooks, act_stat = [], {}
    register_forward_hooks(block, target_name="up_proj", hooks=hooks, act_stat=act_stat)
    block.up_proj(torch.randn(2, 4))
    assert len(act_stat["up_proj_out"]) == 1
    for h in hooks:
        h.remove()
    block.up_proj(torch.randn(2, 4))
    # No new entries after removal.
    assert len(act_stat["up_proj_out"]) == 1
