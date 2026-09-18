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
from unittest.mock import patch

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
    model.safetensors_files = None
    model._weight_map = weight_map
    return model


def test_init_requires_trust_remote_code():
    args = SimpleNamespace(
        model_name="deepseek_v4",
        trust_remote_code=False,
    )

    with pytest.raises(
        ValueError,
        match="deepseek_v4 requires --trust_remote_code",
    ):
        DeepseekV4(args)


def test_empty_weights_model_uses_trust_remote_code_from_args(monkeypatch):
    from amct_pytorch.common.models.llm.deepseek.deepseek_v4 import deepseekv4

    model = DeepseekV4.__new__(DeepseekV4)
    model.config = SimpleNamespace()
    model.trust_remote_code = True
    monkeypatch.setattr(deepseekv4, "init_empty_weights", lambda **_: nullcontext())

    with patch.object(deepseekv4.AutoModelForCausalLM, "from_config") as loader:
        model.empty_weights_model()

    assert loader.call_args.kwargs["trust_remote_code"] is True


def test_block_sharded_rejects_shard_outside_model_directory(tmp_path, monkeypatch):
    from amct_pytorch.common.models.llm.deepseek.deepseek_v4 import deepseekv4

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    weight_name = "model.layers.0.weight"
    save_file({"allowed": torch.ones(1)}, str(model_dir / "allowed.safetensors"))
    save_file({weight_name: torch.ones(1, 1)}, str(tmp_path / "outside.safetensors"))
    model = _make_model(model_dir, {weight_name: "../outside.safetensors"})
    model.cls = lambda config, layer_idx: nn.Linear(1, 1, bias=False)
    model.config = SimpleNamespace()
    model.args = SimpleNamespace(device="cpu")
    model._build_block_device_map = lambda block: {"": "cpu"}
    model.get_layer_weight_prefix = lambda layer_idx: "model.layers.0."
    monkeypatch.setattr(deepseekv4, "init_empty_weights", nullcontext)

    with pytest.raises(ValueError, match="not in the model directory allowlist"):
        model._block_sharded(0)


def test_top_level_hc_params_reject_shard_outside_model_directory(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    save_file({"allowed": torch.ones(1)}, str(model_dir / "allowed.safetensors"))
    save_file({"hc_head_fn": torch.ones(1)}, str(tmp_path / "outside.safetensors"))
    model = _make_model(model_dir, {"hc_head_fn": "../outside.safetensors"})
    model.model = nn.Module()

    with pytest.raises(ValueError, match="not in the model directory allowlist"):
        model._load_top_level_hc_head_params()


@pytest.mark.parametrize(
    "target,attr", [("moe", "ffn"), ("attn-linear", "attn"), ("attn-cache", "attn")]
)
def test_iter_ptq_units_for_load_preserves_v4_module_paths(target, attr, tmp_path):
    from amct_pytorch.common.models.llm.common.base import BaseModel

    model = DeepseekV4.__new__(DeepseekV4)
    args = SimpleNamespace(
        model="/fake/model", quant_target=[target], quant_dtype="int"
    )
    with (
        patch("amct_pytorch.common.models.llm.common.base.AutoConfig.from_pretrained"),
        patch(
            "amct_pytorch.common.models.llm.common.base.AutoTokenizer.from_pretrained"
        ),
    ):
        BaseModel.__init__(model, args)
    module = nn.Linear(2, 2, bias=False)
    block = SimpleNamespace(
        **{attr: SimpleNamespace(experts=[module]) if target == "moe" else module}
    )
    train_unit = list(model.iter_ptq_units(0, block))[0]
    loaded_unit = list(model.iter_ptq_units(0, block, for_load=True))[0]
    assert train_unit.module is loaded_unit.module is module
    assert train_unit.save_name == loaded_unit.save_name
    expected = torch.full_like(module.weight, 0.75)
    torch.save({"weight": expected}, tmp_path / f"layer_0_{train_unit.save_name}.pt")
    result = model.load_layer_ptq_params(0, block, str(tmp_path), strict=True)
    assert result == {"loaded": [train_unit.name], "missing": []}
    torch.testing.assert_close(module.weight, expected)
