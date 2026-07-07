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

"""Tests for deploy_export pure-config functions.

`get_quant_ignore_linear_names` and `export_block_deploy` exercise QuantLinear
and the full pipeline; they are deferred to the tiny-config integration batch.
"""

import importlib

import pytest
import torch.nn as nn

from amct_pytorch.common.models.llm.common.deploy_export import (
    export_block_deploy,
    generate_quant_config,
    generate_quant_group,
    get_quant_ignore_linear_names,
)
from amct_pytorch.quantization.modules.quant_linear import QuantLinear

CONFIG_GROUPS_KEY = 'config_groups'
GROUP_0 = 'group_0'
GROUP_1 = 'group_1'
GROUP_2 = 'group_2'

# ---- generate_quant_group ------------------------------------------------

INPUT_ACTIVATIONS = 'input_activations'
INT = 'int'
KV_CACHE_SCHEME = 'kv_cache_scheme'
NUM_BITS = 'num_bits'
P_KEY_WEIGHT = 'p.key.weight'

WEIGHTS = 'weights'


def test_generate_quant_group_default_int():
    group = generate_quant_group()
    assert group[INPUT_ACTIVATIONS][NUM_BITS] == 8
    assert group[INPUT_ACTIVATIONS]["strategy"] == "token"
    assert group[INPUT_ACTIVATIONS]["dynamic"] is True
    assert group[WEIGHTS][NUM_BITS] == 8
    assert group[WEIGHTS]["strategy"] == "channel"
    assert group[WEIGHTS]["dynamic"] is False
    assert group["activation_use_clip"] is False
    assert group["output_activations"] is None
    assert group[INPUT_ACTIVATIONS]["type"] == "float"


def test_generate_quant_group_custom_bits_and_qtype():
    group = generate_quant_group(a_num_bits=4, w_num_bits=4, qtype=INT, activation_use_clip=True)
    assert group[INPUT_ACTIVATIONS][NUM_BITS] == 4
    assert group[WEIGHTS][NUM_BITS] == 4
    assert group[INPUT_ACTIVATIONS]["type"] == INT
    assert group[WEIGHTS]["type"] == INT
    assert group["activation_use_clip"] is True


# ---- generate_quant_config -----------------------------------------------


def test_generate_quant_config_int_path_only_has_group_0():
    cfg = generate_quant_config(cache_scheme={}, ignores=["lm_head"])
    assert cfg["format"] == "int-quantized"
    assert cfg["ignore"] == ["lm_head"]
    assert cfg["quantization_status"] == "compressed"
    assert cfg["quant_method"] == "compressed-tensors"
    assert set(cfg[CONFIG_GROUPS_KEY]) == {GROUP_0}
    assert cfg[CONFIG_GROUPS_KEY][GROUP_0]["targets"] == ["Linear"]
    # int path: no weight_block_size key emitted.
    assert "weight_block_size" not in cfg


def test_generate_quant_config_mx_w8_adds_moegmm_group():
    cfg = generate_quant_config(cache_scheme={}, ignores=[], is_mx=True)
    assert cfg["format"] == "float-quantized"
    assert set(cfg[CONFIG_GROUPS_KEY]) == {GROUP_0, GROUP_1}
    assert cfg[CONFIG_GROUPS_KEY][GROUP_1]["targets"] == ["MoEGMM"]
    assert cfg[CONFIG_GROUPS_KEY][GROUP_1][WEIGHTS][NUM_BITS] == 8
    assert cfg["weight_block_size"] == [1, 32]


def test_generate_quant_config_mx_w4a8_uses_4bit_weights_in_moegmm():
    cfg = generate_quant_config(cache_scheme={}, ignores=[], w4a8=True, is_mx=True)
    assert cfg[CONFIG_GROUPS_KEY][GROUP_1][WEIGHTS][NUM_BITS] == 4
    assert cfg[CONFIG_GROUPS_KEY][GROUP_1][INPUT_ACTIVATIONS][NUM_BITS] == 8


def test_generate_quant_config_mx_w4a4_splits_up_gate_and_down():
    cfg = generate_quant_config(cache_scheme={}, ignores=[], w4a4=True, is_mx=True)
    assert set(cfg[CONFIG_GROUPS_KEY]) == {GROUP_0, GROUP_1, GROUP_2}
    assert cfg[CONFIG_GROUPS_KEY][GROUP_1]["targets"] == ["MoEGMMUpGate"]
    assert cfg[CONFIG_GROUPS_KEY][GROUP_1][WEIGHTS][NUM_BITS] == 4
    assert cfg[CONFIG_GROUPS_KEY][GROUP_1][INPUT_ACTIVATIONS][NUM_BITS] == 4
    assert cfg[CONFIG_GROUPS_KEY][GROUP_2]["targets"] == ["MoEGMMDown"]
    assert cfg[CONFIG_GROUPS_KEY][GROUP_2][WEIGHTS][NUM_BITS] == 4
    assert cfg[CONFIG_GROUPS_KEY][GROUP_2][INPUT_ACTIVATIONS][NUM_BITS] == 8


def test_generate_quant_config_merges_cache_scheme_into_top_level():
    cache = {KV_CACHE_SCHEME: {"strategy": "channel", NUM_BITS: 8}}
    cfg = generate_quant_config(cache_scheme=cache, ignores=[])
    assert cfg[KV_CACHE_SCHEME] == cache[KV_CACHE_SCHEME]


def test_generate_quant_config_has_global_compression_ratio_one():
    cfg = generate_quant_config(cache_scheme={}, ignores=[])
    assert cfg["global_compression_ratio"] == 1


def test_get_quant_ignore_linear_names_finds_plain_linear():
    block = nn.Sequential(nn.Linear(4, 4))
    names = get_quant_ignore_linear_names(block, "prefix.")
    assert names == ["prefix.0"]


def test_get_quant_ignore_linear_names_no_quant_linear_all_plain():
    block = nn.Module()
    block.a = nn.Linear(4, 4)
    block.b = nn.Linear(8, 8)
    names = get_quant_ignore_linear_names(block, "")
    assert sorted(names) == ["a", "b"]


def test_get_quant_ignore_linear_names_skips_quant_linear_prefix():
    class _NestedQL(QuantLinear):
        def __init__(self):
            from types import SimpleNamespace

            from amct_pytorch.quantization.dtypes import register_dtype
            register_dtype()
            args = SimpleNamespace(w_bits=4, quant_dtype=INT, algos=[])
            super().__init__(args, nn.Linear(4, 4))
            self.nested_fc = nn.Linear(4, 4)

    block = nn.Module()
    block.q_proj = _NestedQL()
    names = get_quant_ignore_linear_names(block, "")
    assert len(names) == 0


def test_export_block_deploy_basic():
    class FakeQuantLinear:
        def export_deploy(self):
            return {"qweight": "qw_val", "weight_scale": "ws_val"}

    fql = FakeQuantLinear()
    pipeline = importlib.import_module("types").SimpleNamespace(
        args=importlib.import_module("types").SimpleNamespace(device="cpu"),
        build_quant_block=lambda idx: nn.Linear(4, 4),
        load_selected_layer_ptq_params=lambda idx, block, strict: None,
        get_layer_weight_prefix=lambda idx: "p.",
        iter_deploy_bindings=lambda idx, block: [(P_KEY_WEIGHT, fql)],
    )
    deploy_tensors, tensor_routes = export_block_deploy(pipeline, 0, [])
    assert deploy_tensors[P_KEY_WEIGHT] == "qw_val"
    assert deploy_tensors["p.key.weight_scale"] == "ws_val"
    assert tensor_routes["p.key.weight_scale"] == P_KEY_WEIGHT


def test_export_block_deploy_skips_none_extra_tensor():
    class FakeQuantLinear:
        def export_deploy(self):
            return {"qweight": "qw", "extra": None}

    fql = FakeQuantLinear()
    pipeline = importlib.import_module("types").SimpleNamespace(
        args=importlib.import_module("types").SimpleNamespace(device="cpu"),
        build_quant_block=lambda idx: nn.Linear(4, 4),
        load_selected_layer_ptq_params=lambda idx, block, strict: None,
        get_layer_weight_prefix=lambda idx: "",
        iter_deploy_bindings=lambda idx, block: [("k.weight", fql)],
    )
    deploy_tensors, _ = export_block_deploy(pipeline, 0, [])
    assert "k.extra" not in deploy_tensors

