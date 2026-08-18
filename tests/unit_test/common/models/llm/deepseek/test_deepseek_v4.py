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

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from amct_pytorch.common.models.llm.deepseek.deepseek_v4.deepseekv4 import (
    DeepseekV4,
)


def _make_model(model_path, weight_map):
    model = DeepseekV4.__new__(DeepseekV4)
    model.model_path = str(model_path)
    model._weight_map = weight_map
    return model


def test_block_sharded_rejects_shard_outside_model_directory(tmp_path, monkeypatch):
    from amct_pytorch.common.models.llm.deepseek.deepseek_v4 import deepseekv4

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    weight_name = "model.layers.0.weight"
    save_file({weight_name: torch.ones(1, 1)}, str(tmp_path / "outside.safetensors"))
    model = _make_model(model_dir, {weight_name: "../outside.safetensors"})
    model.cls = lambda config, layer_idx: nn.Linear(1, 1, bias=False)
    model.config = SimpleNamespace()
    model.args = SimpleNamespace(device="cpu")
    model._build_block_device_map = lambda block: {"": "cpu"}
    model.get_layer_weight_prefix = lambda layer_idx: "model.layers.0."
    monkeypatch.setattr(deepseekv4, "init_empty_weights", nullcontext)

    with pytest.raises(ValueError, match="plain file name"):
        model._block_sharded(0)


def test_top_level_hc_params_reject_shard_outside_model_directory(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    save_file({"hc_head_fn": torch.ones(1)}, str(tmp_path / "outside.safetensors"))
    model = _make_model(model_dir, {"hc_head_fn": "../outside.safetensors"})
    model.model = nn.Module()

    with pytest.raises(ValueError, match="plain file name"):
        model._load_top_level_hc_head_params()
