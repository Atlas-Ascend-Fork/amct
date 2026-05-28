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

from amct_pytorch.common.config.utils import (
    get_alg_name_from_config,
    match_fuzzy_pattern,
)

MODEL_LAYERS_0_SELF_ATTN_Q_PROJ = 'model.layers.0.self_attn.q_proj'


def test_get_alg_name_from_dict():
    names, attrs = get_alg_name_from_config({"minmax": {}, "awq": {"grids_num": 20}})
    assert names == ["minmax", "awq"]
    assert attrs == [{}, {"grids_num": 20}]


def test_get_alg_name_from_set():
    names, attrs = get_alg_name_from_config({"gptq"})
    assert names == ["gptq"]
    assert attrs == [None]


def test_get_alg_name_from_str():
    names, attrs = get_alg_name_from_config("minmax")
    assert names == ["minmax"]
    assert attrs == [None]


def test_get_alg_name_from_invalid():
    with pytest.raises(ValueError, match="invalid algo name"):
        get_alg_name_from_config(123)


def test_match_fuzzy_exact():
    assert match_fuzzy_pattern(MODEL_LAYERS_0_SELF_ATTN_Q_PROJ, MODEL_LAYERS_0_SELF_ATTN_Q_PROJ)
    assert not match_fuzzy_pattern(MODEL_LAYERS_0_SELF_ATTN_Q_PROJ, "model.layers.1.self_attn.q_proj")


def test_match_fuzzy_wildcard():
    assert match_fuzzy_pattern(MODEL_LAYERS_0_SELF_ATTN_Q_PROJ, "*self_attn.q_proj")
    assert match_fuzzy_pattern("model.layers.1.self_attn.k_proj", "*self_attn.k_proj")
    assert not match_fuzzy_pattern("model.layers.0.mlp.gate_proj", "*self_attn.q_proj")


def test_match_fuzzy_with_suffix():
    assert match_fuzzy_pattern(MODEL_LAYERS_0_SELF_ATTN_Q_PROJ, "*self_attn.q_proj.weights")
    assert match_fuzzy_pattern("model.layers.1.self_attn.q_proj", "*self_attn.q_proj.inputs")
    assert not match_fuzzy_pattern("model.layers.0.mlp.gate_proj", "*self_attn.q_proj.weights")

