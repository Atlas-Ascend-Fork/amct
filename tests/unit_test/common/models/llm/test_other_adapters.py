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
"""Lightweight adapter tests for longcat / qwen3.6 families."""

import json
from types import SimpleNamespace

import pytest
import torch.nn as nn

from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm.longcat.longcat_lite.longcat_lite import LongcatLite
from amct_pytorch.common.models.llm.longcat.longcat_next.longcat_next import LongcatNext
from amct_pytorch.common.models.llm.kimi.kimi_k3.kimi_k3 import KimiK3
from amct_pytorch.common.models.llm.qwen.qwen3_6.qwen3_6_moe import Qwen3_6Moe
from amct_pytorch.quantization.dtypes import register_dtype

register_dtype()


def _stub(adapter_cls, **attrs):
    obj = adapter_cls.__new__(adapter_cls)
    obj.args = SimpleNamespace(quant_target=attrs.get("quant_target", []))
    obj.quant_target = attrs.get("quant_target", [])
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


# ---- LongcatLite / LongcatNext ------------------------------------------


def test_longcat_lite_layer_weight_prefix():
    obj = _stub(LongcatLite, quant_target=["moe"])
    assert obj.get_layer_weight_prefix(0) == "model.layers.0."


def test_longcat_next_layer_weight_prefix_delegates_to_super():
    # LongcatNext inherits LongcatLite's prefix scheme via super().
    obj = _stub(LongcatNext, quant_target=["moe"])
    assert obj.get_layer_weight_prefix(2) == "model.layers.2."


def test_longcat_lite_set_safe_attn_impl_writes_eager():
    cfg = SimpleNamespace()
    obj = _stub(LongcatLite, config=cfg)
    obj._set_safe_attn_impl()
    assert cfg._attn_implementation == "eager"


# ---- Qwen3.6Moe is essentially a stub class ------------------------------


def test_qwen3_6_moe_registers_in_model_registry():
    assert MODEL_REGISTRY.get("qwen3_6_moe") is Qwen3_6Moe


def test_kimi_k3_registers_in_model_registry():
    assert MODEL_REGISTRY.get("kimi_k3") is KimiK3


def test_kimi_k3_placeholder_fails_outside_tensor_deploy():
    args = SimpleNamespace(granularity="block", model="/fake", quant_target=[])
    with pytest.raises(NotImplementedError, match="model_name='kimi_k3'"):
        KimiK3(args)


def test_kimi_k3_tensor_deploy_builds_moe_quant_layers_from_index(tmp_path):
    index = {
        "weight_map": {
            "language_model.model.layers.19.block_sparse_moe.experts.895.w3.weight_packed": "shard.safetensors",
            "language_model.model.layers.19.block_sparse_moe.experts.895.w3.weight_scale": "shard.safetensors",
            "language_model.model.layers.19.self_attn.q_proj.weight": "shard.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    args = SimpleNamespace(granularity="tensor", model=str(tmp_path), quant_target=["moe"])

    model = KimiK3(args)

    assert model.generate_tensorwise_quant_layers() == {
        "language_model.model.layers.19.block_sparse_moe.experts.895.w3": 4,
    }
    assert model.generate_tensorwise_ignore_layers() == []


def test_kimi_k3_tensor_deploy_respects_non_moe_quant_target(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "language_model.model.layers.0.block_sparse_moe.experts.0.w1.weight_packed": "shard.safetensors",
    }}))
    args = SimpleNamespace(granularity="tensor", model=str(tmp_path), quant_target=["attn-linear"])

    model = KimiK3(args)

    assert model.generate_tensorwise_quant_layers() == {}


def test_kimi_k3_scale_and_output_names_for_packed_weight(tmp_path):
    args = SimpleNamespace(granularity="tensor", model=str(tmp_path), quant_target=["moe"])
    model = KimiK3(args)
    weight_name = "language_model.model.layers.0.block_sparse_moe.experts.0.w1.weight_packed"

    assert model.get_scale_name(weight_name) == (
        ".weight_scale",
        "language_model.model.layers.0.block_sparse_moe.experts.0.w1.weight_scale",
    )
    assert model.tensorwise_output_weight_name(weight_name) == (
        "language_model.model.layers.0.block_sparse_moe.experts.0.w1.weight"
    )


# ---- registry coverage check --------------------------------------------


@pytest.mark.parametrize(
    "key,expected_cls",
    [
        ("longcat_lite", LongcatLite),
    ],
)
def test_other_adapters_register_under_expected_keys(key, expected_cls):
    assert MODEL_REGISTRY.get(key) is expected_cls

