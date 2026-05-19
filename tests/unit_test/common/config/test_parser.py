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

from amct_pytorch.algorithms import AlgorithmRegistry
from amct_pytorch.common.config.config import INT8_MINMAX_WEIGHT_QUANT_CFG
from amct_pytorch.common.config.fields import QuantConfig
from amct_pytorch.common.config.parser import (
    _build_layer_types_and_quant_type,
    _check_fuzzy_config_warnings,
    _check_layer_constraints,
    _is_layer_supported,
    check_config,
    check_quant_op_constraint,
    check_skip_layer,
    get_supported_layers,
    parse_config,
    set_default_config,
)

# ---- set_default_config ----------------------------------------------------

MINMAX = 'minmax'
MY_CUSTOM_PARSER = 'my_custom_parser'


def test_set_default_config_returns_int8_minmax():
    assert set_default_config() == INT8_MINMAX_WEIGHT_QUANT_CFG


# ---- check_skip_layer ------------------------------------------------------


def test_check_skip_layer_empty_returns_false():
    assert check_skip_layer("layer.0", []) is False
    assert check_skip_layer("layer.0", None) is False


def test_check_skip_layer_matched():
    assert check_skip_layer("model.layers.0.lm_head", ["lm_head"]) is True


def test_check_skip_layer_substring_match():
    assert check_skip_layer("model.layers.0.self_attn", ["self_attn"]) is True


def test_check_skip_layer_no_match():
    assert check_skip_layer("model.layers.0.mlp", ["lm_head"]) is False


# ---- check_config ----------------------------------------------------------


def _make_quant_config(*, weight_type="int8", weight_strategy="channel", group_size=None,
                       input_type="int8", input_strategy="tensor", enable_input=True):
    cfg = {
        "batch_num": 1,
        "quant_cfg": {
            "weights": {"type": weight_type, "symmetric": True, "strategy": weight_strategy},
            "inputs": {"type": input_type, "symmetric": True, "strategy": input_strategy, "enable_quant": enable_input},
        },
        "algorithm": {MINMAX: {}},
    }
    if group_size is not None:
        cfg["quant_cfg"]["weights"]["group_size"] = group_size
    return QuantConfig(cfg, AlgorithmRegistry)


def test_check_config_valid_int8_int8():
    qc = _make_quant_config()
    check_config("int8 int8", qc, MINMAX)


def test_check_config_invalid_comb():
    with pytest.raises(ValueError, match="Do not support combination"):
        check_config("float64 float64", _make_quant_config(), MINMAX)


def test_check_config_algo_not_support_comb():
    with pytest.raises(ValueError, match="do not support act and weight quant dtype"):
        check_config("mxfp8_e4m3fn mxfp8_e4m3fn", _make_quant_config(), MINMAX)


def test_check_config_weight_strategy_not_supported():
    qc = _make_quant_config(weight_type="int8", weight_strategy="group", group_size=64)
    with pytest.raises(ValueError, match="do not support weight quant strategy"):
        check_config("int8 int8", qc, MINMAX)


def test_check_config_act_strategy_not_supported():
    qc = _make_quant_config(input_strategy="token", weight_type="float8_e4m3fn",
                            input_type="float8_e4m3fn")
    with pytest.raises(ValueError, match="do not support activation quant strategy"):
        check_config("float8_e4m3fn float8_e4m3fn", qc, "ofmr")


def test_check_config_mxfp8_act_strategy_not_group():
    qc = _make_quant_config(input_type="mxfp8_e4m3fn", input_strategy="tensor",
                            weight_type="mxfp8_e4m3fn", weight_strategy="group", group_size=32)
    with pytest.raises(ValueError, match="only support activation quant strategy group"):
        check_config("mxfp8_e4m3fn mxfp8_e4m3fn", qc, "mxquant")


def test_check_config_group_size_not_multiple_of_32():
    qc = _make_quant_config(weight_type="int4", weight_strategy="group", group_size=33, enable_input=False)
    with pytest.raises(ValueError, match="integer multiple of 32"):
        check_config("NOT_QUANTIZE int4", qc, MINMAX)


def test_check_config_group_size_less_than_32():
    qc = _make_quant_config(weight_type="int4", weight_strategy="group", group_size=16, enable_input=False)
    with pytest.raises(ValueError, match="group_size larger than 32"):
        check_config("NOT_QUANTIZE int4", qc, MINMAX)


# ---- check_quant_op_constraint ---------------------------------------------


def _make_mock_linear(in_features=64, out_features=64, has_bias=False):
    mod = nn.Linear(in_features, out_features, bias=has_bias)
    return mod


def test_check_quant_op_constraint_non_linear():
    mod = nn.Conv2d(3, 3, 1)
    assert check_quant_op_constraint(mod, "conv", "int8 int8", _make_quant_config()) is True


def test_check_quant_op_constraint_cin_not_multiple_of_64():
    mod = _make_mock_linear(in_features=63, out_features=64)
    result = check_quant_op_constraint(
        mod, "layer.0", "float8_e4m3fn float4_e2m1",
        _make_quant_config(
            weight_type="float4_e2m1", weight_strategy="group", group_size=64),
    )
    assert result is False


def test_check_quant_op_constraint_has_bias():
    mod = _make_mock_linear(in_features=64, out_features=64, has_bias=True)
    result = check_quant_op_constraint(
        mod, "layer.0", "float8_e4m3fn float4_e2m1",
        _make_quant_config(
            weight_type="float4_e2m1", weight_strategy="group", group_size=64),
    )
    assert result is False


def test_check_quant_op_constraint_no_bias_no_group_size():
    mod = _make_mock_linear(in_features=64, out_features=64, has_bias=False)
    result = check_quant_op_constraint(mod, "layer.0", "int8 int8", _make_quant_config(weight_strategy="channel"))
    assert result is True


def test_check_quant_op_constraint_group_size_none():
    mod = _make_mock_linear(in_features=32, out_features=32)
    qc = _make_quant_config(weight_type="int8", weight_strategy="channel", enable_input=False)
    result = check_quant_op_constraint(mod, "layer.0", "NOT_QUANTIZE int8", qc)
    assert result is True


def test_check_quant_op_constraint_mxfp8_ceiling_odd():
    mod = _make_mock_linear(in_features=31, out_features=64)
    qc = _make_quant_config(weight_type="mxfp8_e4m3fn", weight_strategy="group", group_size=32,
                            input_type="mxfp8_e4m3fn", input_strategy="group")
    result = check_quant_op_constraint(mod, "layer.0", "mxfp8_e4m3fn mxfp8_e4m3fn", qc)
    assert result is False


def test_check_quant_op_constraint_mxfp4_shape_cout_not_64():
    mod = _make_mock_linear(in_features=64, out_features=63)
    result = check_quant_op_constraint(
        mod, "layer.0", "NOT_QUANTIZE mxfp4_e2m1",
        _make_quant_config(
            weight_type="mxfp4_e2m1", weight_strategy="group",
            group_size=32, enable_input=False),
    )
    assert result is False


def test_check_quant_op_constraint_int4_cin_not_8():
    mod = _make_mock_linear(in_features=7, out_features=8)
    result = check_quant_op_constraint(
        mod, "layer.0", "NOT_QUANTIZE int4",
        _make_quant_config(
            weight_type="int4", weight_strategy="channel", enable_input=False),
    )
    assert result is False


def test_check_fuzzy_config_warnings_skip_interaction(caplog):
    qc = QuantConfig(
        {
            "batch_num": 1,
            "quant_cfg": {
                "*self_attn.q_proj.weights": {
                    "type": "int4", "symmetric": True, "strategy": "channel",
                },
            },
            "algorithm": {MINMAX: {}},
            "skip_layers": ["model.layers.0.self_attn.q_proj"],
        },
        AlgorithmRegistry,
    )
    _check_fuzzy_config_warnings(["model.layers.0.self_attn.q_proj"], qc)


def test_is_layer_supported_conv2d_padding_mode():
    mod = nn.Conv2d(3, 3, 1, padding_mode='reflect')
    lt = {"Conv2d": "ofmr"}
    qc = _make_quant_config(weight_type="int8", input_type="int8")
    assert _is_layer_supported(mod, "conv.0", lt, "int8 int8", qc) is False


def test_check_layer_constraints_custom_algo():
    mod = _make_mock_linear(in_features=64, out_features=64)
    qc = _make_quant_config()
    assert _check_layer_constraints(mod, "layer.0", "custom_algo", "int8 int8", qc) is True


def test_check_layer_constraints_weight_dtype_mismatch():
    mod = nn.Linear(64, 64, dtype=torch.float64)
    qc = _make_quant_config(weight_type="int4", enable_input=False)
    result = _check_layer_constraints(mod, "layer.0", MINMAX, "NOT_QUANTIZE int4", qc)
    assert result is False


def test_check_quant_op_constraint_group_size_too_large():
    qc = _make_quant_config(weight_type="int4", weight_strategy="group", group_size=128)
    mod = _make_mock_linear(in_features=32, out_features=32)
    result = check_quant_op_constraint(mod, "layer.0", "NOT_QUANTIZE int4", qc)
    assert result is False


def test_check_quant_op_constraint_int4_shape():
    mod = _make_mock_linear(in_features=7, out_features=7)
    result = check_quant_op_constraint(mod, "layer.0", "NOT_QUANTIZE int4",
                                       _make_quant_config(weight_type="int4", weight_strategy="channel"))
    assert result is False


# ---- _check_fuzzy_config_warnings ------------------------------------------


def test_check_fuzzy_config_warnings_no_fuzzy():
    qc = QuantConfig(
        {
            "batch_num": 1,
            "quant_cfg": {
                "weights": {
                    "type": "int8", "symmetric": True, "strategy": "channel",
                },
            },
            "algorithm": {MINMAX: {}},
        },
        AlgorithmRegistry,
    )
    _check_fuzzy_config_warnings(["layer.0", "layer.1"], qc)


def test_check_fuzzy_config_warnings_with_match(caplog):
    qc = QuantConfig(
        {
            "batch_num": 1,
            "quant_cfg": {
                "*self_attn.q_proj.weights": {
                    "type": "int4", "symmetric": True, "strategy": "channel",
                },
            },
            "algorithm": {MINMAX: {}},
        },
        AlgorithmRegistry,
    )
    _check_fuzzy_config_warnings(["model.layers.0.self_attn.q_proj"], qc)


def test_check_fuzzy_config_warnings_no_match(caplog):
    qc = QuantConfig(
        {
            "batch_num": 1,
            "quant_cfg": {
                "*self_attn.q_proj.weights": {
                    "type": "int4", "symmetric": True, "strategy": "channel",
                },
            },
            "algorithm": {MINMAX: {}},
        },
        AlgorithmRegistry,
    )
    _check_fuzzy_config_warnings(["model.layers.0.mlp.gate_proj"], qc)


# ---- _build_layer_types_and_quant_type -------------------------------------


def test_build_layer_types_single_algo():
    qc = QuantConfig({"batch_num": 1,
                      "quant_cfg": {"weights": {"type": "int8", "symmetric": True, "strategy": "channel"}},
                      "algorithm": {MINMAX: {}}}, AlgorithmRegistry)
    lt, qtc = _build_layer_types_and_quant_type(qc, AlgorithmRegistry)
    assert "Linear" in lt
    assert qtc == "NOT_QUANTIZE int8"


def test_build_layer_types_multi_algo():
    qc = QuantConfig({"batch_num": 1,
                      "quant_cfg": {"weights": {"type": "int4", "symmetric": True, "strategy": "channel"}},
                      "algorithm": {"awq": {"grids_num": 20}}}, AlgorithmRegistry)
    lt, qtc = _build_layer_types_and_quant_type(qc, AlgorithmRegistry)
    assert "Linear" in lt
    assert qtc == "NOT_QUANTIZE int4"


def test_build_layer_types_weight_none():
    qc = QuantConfig({"batch_num": 1,
                      "quant_cfg": {"weights": {"type": "int8", "symmetric": True, "strategy": "channel"},
                                    "inputs": {"enable_quant": False}},
                      "algorithm": {MINMAX: {}}}, AlgorithmRegistry)
    lt, qtc = _build_layer_types_and_quant_type(qc, AlgorithmRegistry)
    assert qtc == "NOT_QUANTIZE int8"


# ---- _is_layer_supported ---------------------------------------------------


def test_is_layer_supported_linear():
    mod = nn.Linear(4, 4)
    lt = {"Linear": MINMAX}
    qc = QuantConfig({"batch_num": 1,
                      "quant_cfg": {"weights": {"type": "int8", "symmetric": True, "strategy": "channel"}},
                      "algorithm": {MINMAX: {}}}, AlgorithmRegistry)
    assert _is_layer_supported(mod, "layer.0", lt, "NOT_QUANTIZE int8", qc) is True


def test_is_layer_supported_skip_layer():
    mod = nn.Linear(4, 4)
    lt = {"Linear": MINMAX}
    qc = QuantConfig({"batch_num": 1,
                      "quant_cfg": {"weights": {"type": "int8", "symmetric": True, "strategy": "channel"}},
                      "algorithm": {MINMAX: {}},
                      "skip_layers": ["layer.0"]}, AlgorithmRegistry)
    assert _is_layer_supported(mod, "layer.0", lt, "NOT_QUANTIZE int8", qc) is False


# ---- _check_layer_constraints ----------------------------------------------


def test_check_layer_constraints_no_weight():
    class NoWeightModule(nn.Module):
        pass
    mod = NoWeightModule()
    lt = {"Linear": MINMAX}
    qc = QuantConfig({"batch_num": 1,
                      "quant_cfg": {"weights": {"type": "int8", "symmetric": True, "strategy": "channel"}},
                      "algorithm": {MINMAX: {}}}, AlgorithmRegistry)
    assert _check_layer_constraints(mod, "norm.0", MINMAX, "NOT_QUANTIZE int8", qc) is True


# ---- get_supported_layers / parse_config -----------------------------------


class _MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Module(),
            nn.Module(),
        ])
        self.layers[0].self_attn = nn.Module()
        self.layers[0].self_attn.q_proj = nn.Linear(64, 64, dtype=torch.bfloat16)
        self.layers[0].mlp = nn.Module()
        self.layers[0].mlp.gate_proj = nn.Linear(64, 64, dtype=torch.bfloat16)
        self.layers[1].self_attn = nn.Module()
        self.layers[1].self_attn.q_proj = nn.Linear(64, 64, dtype=torch.bfloat16)
        self.layers[1].mlp = nn.Module()
        self.layers[1].mlp.gate_proj = nn.Linear(64, 64, dtype=torch.bfloat16)


def test_get_supported_layers_finds_linears():
    model = _MockModel()
    qc = QuantConfig({"batch_num": 1,
                      "quant_cfg": {"weights": {"type": "int8", "symmetric": True, "strategy": "channel"}},
                      "algorithm": {MINMAX: {}}}, AlgorithmRegistry)
    result = get_supported_layers(model, qc, AlgorithmRegistry)
    assert len(result) == 4


def test_parse_config_returns_detail():
    model = _MockModel()
    config = {"batch_num": 1,
              "quant_cfg": {"weights": {"type": "int8", "symmetric": True, "strategy": "channel"}},
              "algorithm": {MINMAX: {}}}
    result = parse_config(model, config, AlgorithmRegistry)
    assert len(result) == 4
    for name in result:
        assert "batch_num" in result[name]
        assert "weights_cfg" in result[name]
        assert "algorithm" in result[name]


def test_check_quant_op_constraint_group_size_valid_returns_true():
    mod = _make_mock_linear(in_features=128, out_features=64)
    qc = _make_quant_config(weight_type="int4", weight_strategy="group", group_size=64)
    result = check_quant_op_constraint(mod, "layer.0", "NOT_QUANTIZE int4", qc)
    assert result is True


def test_check_quant_op_constraint_group_size_valid_less_than_64_returns_true():
    mod = _make_mock_linear(in_features=64, out_features=32)
    qc = _make_quant_config(weight_type="int8", weight_strategy="group", group_size=32)
    result = check_quant_op_constraint(mod, "layer.0", "NOT_QUANTIZE int8", qc)
    assert result is True


def test_check_fuzzy_config_warnings_pattern_no_match(caplog):
    qc = QuantConfig({"batch_num": 1,
                      "quant_cfg": {"*nonexistent.weights": {"type": "int4", "symmetric": True, "strategy": "channel"}},
                      "algorithm": {MINMAX: {}}}, AlgorithmRegistry)
    _check_fuzzy_config_warnings(["model.layers.0.self_attn.q_proj"], qc)


def test_build_layer_types_customized_algo(monkeypatch):
    AlgorithmRegistry.algo[MY_CUSTOM_PARSER] = {"Linear": object()}
    monkeypatch.setattr("amct_pytorch.common.config.parser.BUILT_IN_ALGORITHM", [])
    try:
        qc = QuantConfig({"batch_num": 1,
                          "quant_cfg": {"weights": {"type": "int8", "symmetric": True, "strategy": "channel"}},
                          "algorithm": {MY_CUSTOM_PARSER: {}}}, AlgorithmRegistry)
        _build_layer_types_and_quant_type(qc, AlgorithmRegistry)
    finally:
        del AlgorithmRegistry.algo[MY_CUSTOM_PARSER]


def test_get_supported_layers_no_weights():
    model = _MockModel()
    qc = QuantConfig({"batch_num": 1,
                      "quant_cfg": {"weights": {"type": "int4", "symmetric": True, "strategy": "channel"}},
                      "algorithm": {MINMAX: {}}}, AlgorithmRegistry)
    result = get_supported_layers(model, qc, AlgorithmRegistry)


def test_get_supported_layers_constraint_skip():
    model = _MockModel()

    class BadWeightLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(64, 63).to(torch.float32))
            self.bias = None
    model.layers[0].self_attn.q_proj = BadWeightLinear()
    qc = _make_quant_config(weight_type="float4_e2m1", weight_strategy="group", group_size=64, enable_input=False)
    result = get_supported_layers(model, qc, AlgorithmRegistry)
    assert "layers.0.self_attn.q_proj" not in result


def test_build_layer_types_and_quant_type_when_wts_type_is_none():
    from unittest.mock import MagicMock

    from amct_pytorch.common.config.parser import _build_layer_types_and_quant_type
    quant_config = MagicMock()
    quant_config.quant_cfg.inputs_cfg.quant_input = False
    quant_config.quant_cfg.inputs_cfg.quant_type = "NOT_QUANTIZE"
    quant_config.quant_cfg.weights_cfg.quant_type = None
    registed_alg = MagicMock()
    registed_alg.algo = {"awq": {"Linear": {}}}
    layer_types, quant_type_comb = _build_layer_types_and_quant_type(quant_config, registed_alg)
    assert quant_type_comb is None

