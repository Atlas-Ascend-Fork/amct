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
import logging
from unittest.mock import MagicMock

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


def _make_quant_config(
    *,
    weight_type="int8",
    weight_strategy="channel",
    group_size=None,
    input_type="int8",
    input_strategy="tensor",
    enable_input=True,
):
    cfg = {
        "batch_num": 1,
        "quant_cfg": {
            "weights": {
                "type": weight_type,
                "symmetric": True,
                "strategy": weight_strategy,
            },
            "inputs": {
                "type": input_type,
                "symmetric": True,
                "strategy": input_strategy,
                "enable_quant": enable_input,
            },
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
    qc = _make_quant_config(
        input_strategy="token", weight_type="float8_e4m3fn", input_type="float8_e4m3fn"
    )
    with pytest.raises(ValueError, match="do not support activation quant strategy"):
        check_config("float8_e4m3fn float8_e4m3fn", qc, "ofmr")


def test_check_config_mxfp8_act_strategy_not_group():
    qc = _make_quant_config(
        input_type="mxfp8_e4m3fn",
        input_strategy="tensor",
        weight_type="mxfp8_e4m3fn",
        weight_strategy="group",
        group_size=32,
    )
    with pytest.raises(
        ValueError, match="only support activation quant strategy group"
    ):
        check_config("mxfp8_e4m3fn mxfp8_e4m3fn", qc, "mxquant")


def test_check_config_group_size_not_multiple_of_32():
    qc = _make_quant_config(
        weight_type="int4", weight_strategy="group", group_size=33, enable_input=False
    )
    with pytest.raises(ValueError, match="integer multiple of 32"):
        check_config("NOT_QUANTIZE int4", qc, MINMAX)


def test_check_config_group_size_less_than_32():
    qc = _make_quant_config(
        weight_type="int4", weight_strategy="group", group_size=16, enable_input=False
    )
    with pytest.raises(ValueError, match="group_size larger than 32"):
        check_config("NOT_QUANTIZE int4", qc, MINMAX)


# ---- check_quant_op_constraint ---------------------------------------------


def _make_mock_linear(in_features=64, out_features=64, has_bias=False):
    mod = nn.Linear(in_features, out_features, bias=has_bias)
    return mod


def test_check_quant_op_constraint_non_linear():
    mod = nn.Conv2d(3, 3, 1)
    assert (
        check_quant_op_constraint(mod, "conv", "int8 int8", _make_quant_config())
        is True
    )


def test_check_quant_op_constraint_cin_not_multiple_of_64():
    mod = _make_mock_linear(in_features=63, out_features=64)
    result = check_quant_op_constraint(
        mod,
        "layer.0",
        "float8_e4m3fn float4_e2m1",
        _make_quant_config(
            weight_type="float4_e2m1", weight_strategy="group", group_size=64
        ),
    )
    assert result is False


def test_check_quant_op_constraint_has_bias():
    mod = _make_mock_linear(in_features=64, out_features=64, has_bias=True)
    result = check_quant_op_constraint(
        mod,
        "layer.0",
        "float8_e4m3fn float4_e2m1",
        _make_quant_config(
            weight_type="float4_e2m1", weight_strategy="group", group_size=64
        ),
    )
    assert result is False


def _warning_records(caplog, keyword):
    return [
        rec.getMessage()
        for rec in caplog.records
        if rec.levelno == logging.WARNING and keyword in rec.getMessage()
    ]


def test_check_quant_op_constraint_shape_skip_logs_warning(caplog):
    '''形状约束跳层应打 WARNING，用户配置未生效时控制台/日志需有痕迹'''
    mod = _make_mock_linear(in_features=96, out_features=64)
    qc = _make_quant_config(
        weight_type="float4_e2m1", weight_strategy="group", group_size=64
    )
    with caplog.at_level(logging.DEBUG, logger="Log"):
        result = check_quant_op_constraint(mod, "wide", "NOT_QUANTIZE float4_e2m1", qc)
    assert result is False
    assert _warning_records(caplog, "layer:wide")
    assert _warning_records(caplog, "integer multiple of 64")


def test_check_quant_op_constraint_bias_skip_logs_warning(caplog):
    '''bias 约束跳层同样打 WARNING'''
    mod = _make_mock_linear(in_features=64, out_features=64, has_bias=True)
    qc = _make_quant_config(
        weight_type="float4_e2m1", weight_strategy="group", group_size=64
    )
    with caplog.at_level(logging.DEBUG, logger="Log"):
        result = check_quant_op_constraint(
            mod, "layer.0", "float8_e4m3fn float4_e2m1", qc
        )
    assert result is False
    assert _warning_records(caplog, "bias is not supported")


def test_check_quant_op_constraint_int4_shape_skip_logs_warning(caplog):
    '''int4 的 8 倍数约束跳层打 WARNING'''
    mod = _make_mock_linear(in_features=60, out_features=64)
    qc = _make_quant_config(
        weight_type="int4", weight_strategy="channel", enable_input=False
    )
    with caplog.at_level(logging.DEBUG, logger="Log"):
        result = check_quant_op_constraint(mod, "layer.0", "NOT_QUANTIZE int4", qc)
    assert result is False
    assert _warning_records(caplog, "integer multiples of 8")


def test_get_supported_layers_fuzzy_shape_skip_logs_warning(caplog):
    '''通配符改 dtype 导致形状不满足而跳层时，控制台需可见告警'''
    model = _FuzzyModel()
    qc = _make_fuzzy_quant_config(
        {
            "weights": {"type": "int8", "symmetric": True, "strategy": "channel"},
            "*wide.weights": {
                "type": "float4_e2m1",
                "symmetric": True,
                "strategy": "group",
                "group_size": 32,
            },
        }
    )
    with caplog.at_level(logging.DEBUG, logger="Log"):
        result = get_supported_layers(model, qc, AlgorithmRegistry)
    assert "wide" not in result
    assert _warning_records(caplog, "layer:wide")


def test_check_layer_constraints_dtype_mismatch_skip_logs_warning(caplog):
    '''原始权重 dtype 不在组合白名单内而跳层时，应打 WARNING 而非 DEBUG'''
    mod = nn.Linear(64, 64, dtype=torch.float32)
    qc = _make_quant_config(weight_type="int8", enable_input=False)
    with caplog.at_level(logging.DEBUG, logger="Log"):
        result = _check_layer_constraints(mod, "wide", MINMAX, "NOT_QUANTIZE int8", qc)
    assert result is False
    assert _warning_records(caplog, "Layer wide cannot be quantized")
    assert _warning_records(caplog, "only supports original dtypes")


def test_get_supported_layers_fuzzy_dtype_whitelist_skip_logs_warning(caplog):
    '''通配符改 dtype 后按逐层白名单跳层时，控制台需可见告警'''
    model = _Fp32FuzzyModel()
    qc = _make_fuzzy_quant_config(
        {
            "weights": {"type": "int4", "symmetric": True, "strategy": "channel"},
            "*wide.weights": {"type": "int8", "symmetric": True, "strategy": "channel"},
        }
    )
    with caplog.at_level(logging.DEBUG, logger="Log"):
        result = get_supported_layers(model, qc, AlgorithmRegistry)
    assert "wide" not in result
    assert "small" in result
    assert _warning_records(caplog, "Layer wide cannot be quantized")


def test_check_quant_op_constraint_no_bias_no_group_size():
    mod = _make_mock_linear(in_features=64, out_features=64, has_bias=False)
    result = check_quant_op_constraint(
        mod, "layer.0", "int8 int8", _make_quant_config(weight_strategy="channel")
    )
    assert result is True


def test_check_quant_op_constraint_group_size_none():
    mod = _make_mock_linear(in_features=32, out_features=32)
    qc = _make_quant_config(
        weight_type="int8", weight_strategy="channel", enable_input=False
    )
    result = check_quant_op_constraint(mod, "layer.0", "NOT_QUANTIZE int8", qc)
    assert result is True


def test_check_quant_op_constraint_mxfp8_ceiling_odd():
    mod = _make_mock_linear(in_features=31, out_features=64)
    qc = _make_quant_config(
        weight_type="mxfp8_e4m3fn",
        weight_strategy="group",
        group_size=32,
        input_type="mxfp8_e4m3fn",
        input_strategy="group",
    )
    result = check_quant_op_constraint(mod, "layer.0", "mxfp8_e4m3fn mxfp8_e4m3fn", qc)
    assert result is False


def test_check_quant_op_constraint_mxfp4_shape_cout_not_64():
    mod = _make_mock_linear(in_features=64, out_features=63)
    result = check_quant_op_constraint(
        mod,
        "layer.0",
        "NOT_QUANTIZE mxfp4_e2m1",
        _make_quant_config(
            weight_type="mxfp4_e2m1",
            weight_strategy="group",
            group_size=32,
            enable_input=False,
        ),
    )
    assert result is False


def test_check_quant_op_constraint_int4_cin_not_8():
    mod = _make_mock_linear(in_features=7, out_features=8)
    result = check_quant_op_constraint(
        mod,
        "layer.0",
        "NOT_QUANTIZE int4",
        _make_quant_config(
            weight_type="int4", weight_strategy="channel", enable_input=False
        ),
    )
    assert result is False


def test_check_fuzzy_config_warnings_skip_interaction(caplog):
    qc = QuantConfig(
        {
            "batch_num": 1,
            "quant_cfg": {
                "*self_attn.q_proj.weights": {
                    "type": "int4",
                    "symmetric": True,
                    "strategy": "channel",
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
    assert (
        _check_layer_constraints(mod, "layer.0", "custom_algo", "int8 int8", qc) is True
    )


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
    result = check_quant_op_constraint(
        mod,
        "layer.0",
        "NOT_QUANTIZE int4",
        _make_quant_config(weight_type="int4", weight_strategy="channel"),
    )
    assert result is False


# ---- _check_fuzzy_config_warnings ------------------------------------------


def test_check_fuzzy_config_warnings_no_fuzzy():
    qc = QuantConfig(
        {
            "batch_num": 1,
            "quant_cfg": {
                "weights": {
                    "type": "int8",
                    "symmetric": True,
                    "strategy": "channel",
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
                    "type": "int4",
                    "symmetric": True,
                    "strategy": "channel",
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
                    "type": "int4",
                    "symmetric": True,
                    "strategy": "channel",
                },
            },
            "algorithm": {MINMAX: {}},
        },
        AlgorithmRegistry,
    )
    _check_fuzzy_config_warnings(["model.layers.0.mlp.gate_proj"], qc)


# ---- _build_layer_types_and_quant_type -------------------------------------


def test_build_layer_types_single_algo():
    qc = QuantConfig(
        {
            "batch_num": 1,
            "quant_cfg": {
                "weights": {"type": "int8", "symmetric": True, "strategy": "channel"}
            },
            "algorithm": {MINMAX: {}},
        },
        AlgorithmRegistry,
    )
    lt, qtc = _build_layer_types_and_quant_type(qc, AlgorithmRegistry)
    assert "Linear" in lt
    assert qtc == "NOT_QUANTIZE int8"


def test_build_layer_types_multi_algo():
    qc = QuantConfig(
        {
            "batch_num": 1,
            "quant_cfg": {
                "weights": {"type": "int4", "symmetric": True, "strategy": "channel"}
            },
            "algorithm": {"awq": {"grids_num": 20}},
        },
        AlgorithmRegistry,
    )
    lt, qtc = _build_layer_types_and_quant_type(qc, AlgorithmRegistry)
    assert "Linear" in lt
    assert qtc == "NOT_QUANTIZE int4"


def test_build_layer_types_weight_none():
    qc = QuantConfig(
        {
            "batch_num": 1,
            "quant_cfg": {
                "weights": {"type": "int8", "symmetric": True, "strategy": "channel"},
                "inputs": {"enable_quant": False},
            },
            "algorithm": {MINMAX: {}},
        },
        AlgorithmRegistry,
    )
    lt, qtc = _build_layer_types_and_quant_type(qc, AlgorithmRegistry)
    assert qtc == "NOT_QUANTIZE int8"


# ---- _is_layer_supported ---------------------------------------------------


def test_is_layer_supported_linear():
    mod = nn.Linear(4, 4)
    lt = {"Linear": MINMAX}
    qc = QuantConfig(
        {
            "batch_num": 1,
            "quant_cfg": {
                "weights": {"type": "int8", "symmetric": True, "strategy": "channel"}
            },
            "algorithm": {MINMAX: {}},
        },
        AlgorithmRegistry,
    )
    assert _is_layer_supported(mod, "layer.0", lt, "NOT_QUANTIZE int8", qc) is True


def test_is_layer_supported_skip_layer():
    mod = nn.Linear(4, 4)
    lt = {"Linear": MINMAX}
    qc = QuantConfig(
        {
            "batch_num": 1,
            "quant_cfg": {
                "weights": {"type": "int8", "symmetric": True, "strategy": "channel"}
            },
            "algorithm": {MINMAX: {}},
            "skip_layers": ["layer.0"],
        },
        AlgorithmRegistry,
    )
    assert _is_layer_supported(mod, "layer.0", lt, "NOT_QUANTIZE int8", qc) is False


# ---- _check_layer_constraints ----------------------------------------------


def test_check_layer_constraints_no_weight():
    class NoWeightModule(nn.Module):
        pass

    mod = NoWeightModule()
    qc = QuantConfig(
        {
            "batch_num": 1,
            "quant_cfg": {
                "weights": {"type": "int8", "symmetric": True, "strategy": "channel"}
            },
            "algorithm": {MINMAX: {}},
        },
        AlgorithmRegistry,
    )
    assert (
        _check_layer_constraints(mod, "norm.0", MINMAX, "NOT_QUANTIZE int8", qc) is True
    )


# ---- get_supported_layers / parse_config -----------------------------------


class _MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Module(),
                nn.Module(),
            ]
        )
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
    qc = QuantConfig(
        {
            "batch_num": 1,
            "quant_cfg": {
                "weights": {"type": "int8", "symmetric": True, "strategy": "channel"}
            },
            "algorithm": {MINMAX: {}},
        },
        AlgorithmRegistry,
    )
    result = get_supported_layers(model, qc, AlgorithmRegistry)
    assert len(result) == 4


def test_parse_config_returns_detail():
    model = _MockModel()
    config = {
        "batch_num": 1,
        "quant_cfg": {
            "weights": {"type": "int8", "symmetric": True, "strategy": "channel"}
        },
        "algorithm": {MINMAX: {}},
    }
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
    qc = QuantConfig(
        {
            "batch_num": 1,
            "quant_cfg": {
                "*nonexistent.weights": {
                    "type": "int4",
                    "symmetric": True,
                    "strategy": "channel",
                }
            },
            "algorithm": {MINMAX: {}},
        },
        AlgorithmRegistry,
    )
    _check_fuzzy_config_warnings(["model.layers.0.self_attn.q_proj"], qc)


def test_build_layer_types_customized_algo(monkeypatch):
    AlgorithmRegistry.algo[MY_CUSTOM_PARSER] = {"Linear": object()}
    monkeypatch.setattr("amct_pytorch.common.config.parser.BUILT_IN_ALGORITHM", [])
    try:
        qc = QuantConfig(
            {
                "batch_num": 1,
                "quant_cfg": {
                    "weights": {
                        "type": "int8",
                        "symmetric": True,
                        "strategy": "channel",
                    }
                },
                "algorithm": {MY_CUSTOM_PARSER: {}},
            },
            AlgorithmRegistry,
        )
        _build_layer_types_and_quant_type(qc, AlgorithmRegistry)
    finally:
        del AlgorithmRegistry.algo[MY_CUSTOM_PARSER]


def test_get_supported_layers_no_weights():
    model = _MockModel()
    qc = QuantConfig(
        {
            "batch_num": 1,
            "quant_cfg": {
                "weights": {"type": "int4", "symmetric": True, "strategy": "channel"}
            },
            "algorithm": {MINMAX: {}},
        },
        AlgorithmRegistry,
    )
    get_supported_layers(model, qc, AlgorithmRegistry)


def test_get_supported_layers_constraint_skip():
    model = _MockModel()

    class BadWeightLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(64, 63).to(torch.float32))
            self.bias = None

    model.layers[0].self_attn.q_proj = BadWeightLinear()
    qc = _make_quant_config(
        weight_type="float4_e2m1",
        weight_strategy="group",
        group_size=64,
        enable_input=False,
    )
    result = get_supported_layers(model, qc, AlgorithmRegistry)
    assert "layers.0.self_attn.q_proj" not in result


def test_build_layer_types_and_quant_type_when_wts_type_is_none():
    from amct_pytorch.common.config.parser import _build_layer_types_and_quant_type

    quant_config = MagicMock()
    quant_config.quant_cfg.inputs_cfg.quant_input = False
    quant_config.quant_cfg.inputs_cfg.quant_type = "NOT_QUANTIZE"
    quant_config.quant_cfg.weights_cfg.quant_type = None
    registed_alg = MagicMock()
    registed_alg.algo = {"awq": {"Linear": {}}}
    layer_types, quant_type_comb = _build_layer_types_and_quant_type(
        quant_config, registed_alg
    )
    assert quant_type_comb is None


# ---- _check_quant_dtype_comb_rules: asymmetric + int8/int4 strategy rules ---


def _make_qc_for_comb_rules(wts_type, act_type, wts_sym, wts_strat, act_strat):
    qc = QuantConfig(
        {
            "batch_num": 1,
            "quant_cfg": {
                "weights": {
                    "type": wts_type,
                    "symmetric": wts_sym,
                    "strategy": wts_strat,
                }
            },
            "algorithm": {MINMAX: {}},
        },
        AlgorithmRegistry,
    )
    qc.quant_cfg.inputs_cfg.quant_type = act_type
    qc.quant_cfg.inputs_cfg.strategy = act_strat
    return qc


def test_comb_rules_raises_when_asymmetric_weight_for_tensor_comb():
    from amct_pytorch.common.config.parser import _check_quant_dtype_comb_rules

    # "int8 int8" is in ACT_GRANULARITY_SUPPORT_MAP["tensor"]; asymmetric weight -> error
    qc = _make_qc_for_comb_rules("int8", "int8", False, "channel", "tensor")
    with pytest.raises(ValueError, match="only support symmetric"):
        _check_quant_dtype_comb_rules("int8 int8", qc)


def test_comb_rules_raises_int8_int4_with_group_weight_strategy():
    from amct_pytorch.common.config.parser import _check_quant_dtype_comb_rules

    # Build with a valid weight strategy, then override to "group" post-construction
    # (the QuantConfig constructor rejects strategy="group" without a group_size).
    qc = _make_qc_for_comb_rules("int4", "int8", True, "tensor", "tensor")
    qc.quant_cfg.weights_cfg.strategy = "group"
    with pytest.raises(
        ValueError, match="only support weight quant strategy tensor or channel"
    ):
        _check_quant_dtype_comb_rules("int8 int4", qc)


def test_comb_rules_raises_int8_int4_with_channel_act_strategy():
    from amct_pytorch.common.config.parser import _check_quant_dtype_comb_rules

    qc = _make_qc_for_comb_rules("int4", "int8", True, "tensor", "channel")
    with pytest.raises(
        ValueError, match="only support activation quant strategy tensor"
    ):
        _check_quant_dtype_comb_rules("int8 int4", qc)


def test_comb_rules_passes_for_valid_int8_int4():
    from amct_pytorch.common.config.parser import _check_quant_dtype_comb_rules

    # symmetric weight, tensor weight strategy, tensor act strategy -> no raise
    qc = _make_qc_for_comb_rules("int4", "int8", True, "tensor", "tensor")
    _check_quant_dtype_comb_rules("int8 int4", qc)


# ---- check_kv_config --------------------------------------------------------


def test_check_kv_config_raises_unsupported_quant_type():
    from amct_pytorch.common.config.parser import check_kv_config

    qc = QuantConfig(
        {
            "batch_num": 1,
            "quant_cfg": {
                "weights": {"type": "int8", "symmetric": True, "strategy": "channel"}
            },
            "algorithm": {MINMAX: {}},
        },
        AlgorithmRegistry,
    )
    qc.quant_cfg.kvcache_cfg = MagicMock()
    qc.quant_cfg.kvcache_cfg.quant_type = "unsupported_type"
    with pytest.raises(ValueError, match="Do not support"):
        check_kv_config(qc, MINMAX)


def test_check_kv_config_raises_unsupported_algo():
    from amct_pytorch.common.config.parser import check_kv_config
    from amct_pytorch.common.utils.vars import ALGORITHM_SUPPORTED_QUANT_TYPE_KV_CACHE

    qc = QuantConfig(
        {
            "batch_num": 1,
            "quant_cfg": {
                "weights": {"type": "int8", "symmetric": True, "strategy": "channel"}
            },
            "algorithm": {MINMAX: {}},
        },
        AlgorithmRegistry,
    )
    qc.quant_cfg.kvcache_cfg = MagicMock()
    valid_type = next(iter(ALGORITHM_SUPPORTED_QUANT_TYPE_KV_CACHE))
    qc.quant_cfg.kvcache_cfg.quant_type = valid_type
    with pytest.raises(ValueError, match="do not support kvcache"):
        check_kv_config(qc, "nonexistent_algo")


# ---- _is_layer_supported: FP8Linear early return ---------------------------


def test_is_layer_supported_fp8linear_returns_false():
    from amct_pytorch.common.config.parser import _is_layer_supported

    class FP8Linear(nn.Module):
        pass

    mod = FP8Linear()
    qc = QuantConfig(
        {
            "batch_num": 1,
            "quant_cfg": {
                "weights": {"type": "int8", "symmetric": True, "strategy": "channel"}
            },
            "algorithm": {MINMAX: {}},
        },
        AlgorithmRegistry,
    )
    result = _is_layer_supported(mod, "fp8.0", {"FP8Linear": MINMAX}, "int8 int8", qc)
    assert result is False


# ---- per layer (fuzzy override) config validation ---------------------------


class _FuzzyModel(nn.Module):
    '''small: cin=64 (64 的倍数); wide: cin=96 (非 64 的倍数)'''

    def __init__(self):
        super().__init__()
        self.small = nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
        self.wide = nn.Linear(96, 64, bias=False, dtype=torch.bfloat16)


class _Fp32FuzzyModel(nn.Module):
    '''与 _FuzzyModel 同构，但权重为 fp32，用于触发 dtype 白名单跳层'''

    def __init__(self):
        super().__init__()
        self.small = nn.Linear(64, 64, bias=False, dtype=torch.float32)
        self.wide = nn.Linear(96, 64, bias=False, dtype=torch.float32)


def _make_fuzzy_quant_config(quant_cfg, algo=MINMAX):
    return QuantConfig(
        {"batch_num": 1, "quant_cfg": quant_cfg, "algorithm": {algo: {}}},
        AlgorithmRegistry,
    )


def test_resolve_layer_quant_type_comb_fuzzy_override():
    from amct_pytorch.common.config.parser import _resolve_layer_quant_type_comb

    layer_cfg = {
        "weights_cfg": {
            "quant_type": "float4_e2m1",
            "symmetric": True,
            "strategy": "group",
            "group_size": 32,
        },
        "inputs_cfg": {"enable_quant": False},
    }
    assert (
        _resolve_layer_quant_type_comb(layer_cfg, "NOT_QUANTIZE int8")
        == "NOT_QUANTIZE float4_e2m1"
    )


def test_resolve_layer_quant_type_comb_fuzzy_override_both_dtypes():
    from amct_pytorch.common.config.parser import _resolve_layer_quant_type_comb

    layer_cfg = {
        "weights_cfg": {"quant_type": "int4", "symmetric": True, "strategy": "channel"},
        "inputs_cfg": {
            "quant_type": "int8",
            "symmetric": True,
            "strategy": "tensor",
        },
    }
    assert _resolve_layer_quant_type_comb(layer_cfg, "int8 int8") == "int8 int4"


def test_resolve_layer_quant_type_comb_fallback_to_global():
    from amct_pytorch.common.config.parser import _resolve_layer_quant_type_comb

    assert _resolve_layer_quant_type_comb(None, "int8 int8") == "int8 int8"
    assert (
        _resolve_layer_quant_type_comb({"weights_cfg": None}, "int8 int8")
        == "int8 int8"
    )
    assert (
        _resolve_layer_quant_type_comb({"inputs_cfg": {"enable_quant": False}}, None)
        is None
    )


def test_resolve_layer_group_size_prefers_layer_config():
    from amct_pytorch.common.config.parser import _resolve_layer_group_size

    qc = _make_quant_config(weight_type="int4", weight_strategy="group", group_size=128)
    layer_cfg = {
        "weights_cfg": {
            "quant_type": "int4",
            "symmetric": True,
            "strategy": "group",
            "group_size": 32,
        }
    }
    assert _resolve_layer_group_size(layer_cfg, qc) == 32
    # 模糊配置未带 group_size 时不应回落到全局值（整体覆盖语义）
    assert (
        _resolve_layer_group_size(
            {"weights_cfg": {"quant_type": "int8", "strategy": "channel"}}, qc
        )
        is None
    )
    # 未传逐层配置时保持全局 group_size
    assert _resolve_layer_group_size(None, qc) == 128


def test_check_quant_op_constraint_uses_layer_group_size():
    mod = _make_mock_linear(in_features=64, out_features=64)
    qc = _make_quant_config(
        weight_type="int8", weight_strategy="channel", enable_input=False
    )
    layer_cfg = {
        "weights_cfg": {
            "quant_type": "int8",
            "symmetric": True,
            "strategy": "group",
            "group_size": 64,
        }
    }
    # group_size(64) >= cin(64) -> 逐层配置生效，跳过该层
    assert (
        check_quant_op_constraint(mod, "layer.0", "NOT_QUANTIZE int8", qc, layer_cfg)
        is False
    )
    layer_cfg["weights_cfg"]["group_size"] = 32
    assert (
        check_quant_op_constraint(mod, "layer.0", "NOT_QUANTIZE int8", qc, layer_cfg)
        is True
    )


def test_check_layer_config_none_comb_returns_early():
    from amct_pytorch.common.config.parser import check_layer_config

    check_layer_config(None, {"weights_cfg": {"strategy": "group"}}, MINMAX)


def test_check_layer_config_custom_algo_returns_early():
    from amct_pytorch.common.config.parser import check_layer_config

    check_layer_config("float64 float64", None, "my_custom_algo")


def test_check_layer_config_unsupported_comb():
    from amct_pytorch.common.config.parser import check_layer_config

    with pytest.raises(ValueError, match="Do not support combination"):
        check_layer_config("float64 float64", None, MINMAX)


def test_check_layer_config_algo_not_support_comb():
    from amct_pytorch.common.config.parser import check_layer_config

    layer_cfg = {
        "weights_cfg": {"quant_type": "hifloat8", "strategy": "channel"},
        "inputs_cfg": {"enable_quant": False},
    }
    with pytest.raises(ValueError, match="do not support act and weight quant dtype"):
        check_layer_config("NOT_QUANTIZE hifloat8", layer_cfg, MINMAX)


def test_check_layer_config_error_carries_layer_name():
    '''多个模糊 pattern 并存时，报错文案需指明是哪一层'''
    from amct_pytorch.common.config.parser import check_layer_config

    layer_cfg = {
        "weights_cfg": {"quant_type": "hifloat8", "strategy": "channel"},
        "inputs_cfg": {"enable_quant": False},
    }
    with pytest.raises(ValueError) as exc_info:
        check_layer_config(
            "NOT_QUANTIZE hifloat8", layer_cfg, MINMAX, "model.layers.0.mlp.down_proj"
        )
    assert str(exc_info.value).startswith("layer:model.layers.0.mlp.down_proj ")
    assert "do not support act and weight quant dtype" in str(exc_info.value)


def test_check_layer_config_no_layer_name_keeps_message():
    '''不传层名时保持与全局路径一致的原始文案'''
    from amct_pytorch.common.config.parser import check_layer_config

    with pytest.raises(ValueError) as exc_info:
        check_layer_config("float64 float64", None, MINMAX)
    assert str(exc_info.value) == (
        "Do not support combination float64 float64 of act and weight quant dtype."
    )


def test_check_layer_config_comb_rules_error_carries_layer_name():
    '''组合级规则（对称性 / 粒度）报错同样带层名'''
    from amct_pytorch.common.config.parser import check_layer_config

    layer_cfg = {
        "weights_cfg": {
            "quant_type": "int8",
            "symmetric": False,
            "strategy": "channel",
        },
        "inputs_cfg": {
            "quant_type": "int8",
            "symmetric": True,
            "strategy": "tensor",
        },
    }
    with pytest.raises(ValueError) as exc_info:
        check_layer_config("int8 int8", layer_cfg, MINMAX, "wide")
    assert str(exc_info.value) == (
        "layer:wide int8 int8 only support symmetric weight quantization"
    )


def test_check_layer_config_weight_strategy_not_supported():
    from amct_pytorch.common.config.parser import check_layer_config

    layer_cfg = {
        "weights_cfg": {
            "quant_type": "int8",
            "symmetric": True,
            "strategy": "group",
            "group_size": 32,
        },
        "inputs_cfg": {
            "quant_type": "int8",
            "symmetric": True,
            "strategy": "tensor",
        },
    }
    with pytest.raises(ValueError, match="do not support weight quant strategy"):
        check_layer_config("int8 int8", layer_cfg, MINMAX)


def test_check_layer_config_act_strategy_not_supported():
    from amct_pytorch.common.config.parser import check_layer_config

    layer_cfg = {
        "weights_cfg": {"quant_type": "int8", "strategy": "channel"},
        "inputs_cfg": {"quant_type": "int8", "strategy": "group"},
    }
    with pytest.raises(ValueError, match="do not support activation quant strategy"):
        check_layer_config("int8 int8", layer_cfg, MINMAX)


def test_check_layer_config_asymmetric_weight_raises():
    from amct_pytorch.common.config.parser import check_layer_config

    layer_cfg = {
        "weights_cfg": {
            "quant_type": "int8",
            "symmetric": False,
            "strategy": "channel",
        },
        "inputs_cfg": {
            "quant_type": "int8",
            "symmetric": True,
            "strategy": "tensor",
        },
    }
    with pytest.raises(ValueError, match="only support symmetric"):
        check_layer_config("int8 int8", layer_cfg, MINMAX)


def test_check_layer_config_group_size_invalid():
    from amct_pytorch.common.config.parser import check_layer_config

    layer_cfg = {
        "weights_cfg": {
            "quant_type": "int4",
            "symmetric": True,
            "strategy": "group",
            "group_size": 33,
        },
        "inputs_cfg": {"enable_quant": False},
    }
    with pytest.raises(ValueError, match="integer multiple of 32"):
        check_layer_config("NOT_QUANTIZE int4", layer_cfg, MINMAX)


def test_check_layer_config_valid():
    from amct_pytorch.common.config.parser import check_layer_config

    layer_cfg = {
        "weights_cfg": {
            "quant_type": "int4",
            "symmetric": True,
            "strategy": "group",
            "group_size": 64,
        },
        "inputs_cfg": {"enable_quant": False},
    }
    check_layer_config("NOT_QUANTIZE int4", layer_cfg, MINMAX)


def test_get_supported_layers_fuzzy_dtype_shape_constraint_skips_layer():
    '''模糊配置把某层改成 float4_e2m1，形状不满足时应跳过该层（与全局配置一致）'''
    model = _FuzzyModel()
    qc = _make_fuzzy_quant_config(
        {
            "weights": {"type": "int8", "symmetric": True, "strategy": "channel"},
            "*wide.weights": {
                "type": "float4_e2m1",
                "symmetric": True,
                "strategy": "group",
                "group_size": 32,
            },
        }
    )
    result = get_supported_layers(model, qc, AlgorithmRegistry)
    assert "small" in result
    assert "wide" not in result
    assert result["small"]["weights_cfg"]["quant_type"] == "int8"


def test_get_supported_layers_fuzzy_algo_comb_raises():
    '''模糊配置改出算法不支持的 dtype 组合时应立刻报错'''
    model = _FuzzyModel()
    qc = _make_fuzzy_quant_config(
        {
            "weights": {"type": "int8", "symmetric": True, "strategy": "channel"},
            "*wide.weights": {
                "type": "hifloat8",
                "symmetric": True,
                "strategy": "channel",
            },
        }
    )
    with pytest.raises(
        ValueError, match="layer:wide .*do not support act and weight quant dtype"
    ):
        get_supported_layers(model, qc, AlgorithmRegistry)


def test_get_supported_layers_fuzzy_weight_strategy_raises():
    '''模糊配置把不支持 per-group 的组合改成 group 粒度时应立刻报错'''
    model = _FuzzyModel()
    qc = _make_fuzzy_quant_config(
        {
            "weights": {"type": "int8", "symmetric": True, "strategy": "channel"},
            "inputs": {
                "type": "int8",
                "symmetric": True,
                "strategy": "tensor",
                "enable_quant": True,
            },
            "*wide.weights": {
                "type": "int8",
                "symmetric": True,
                "strategy": "group",
                "group_size": 32,
            },
        }
    )
    with pytest.raises(
        ValueError, match="layer:wide .*do not support weight quant strategy"
    ):
        get_supported_layers(model, qc, AlgorithmRegistry)


def test_get_supported_layers_fuzzy_only_group_size_skips_layer():
    '''仅配置模糊 pattern（无全局 weights）时，group_size >= cin 的层不应被静默量化'''
    model = _FuzzyModel()
    qc = _make_fuzzy_quant_config(
        {
            "*small.weights": {
                "type": "int8",
                "symmetric": True,
                "strategy": "group",
                "group_size": 64,
            },
        }
    )
    result = get_supported_layers(model, qc, AlgorithmRegistry)
    assert result == {}


def test_get_supported_layers_fuzzy_only_group_size_effective():
    '''仅配置模糊 pattern 时，逐层 group_size 应通过校验并生效'''
    model = _FuzzyModel()
    qc = _make_fuzzy_quant_config(
        {
            "*small.weights": {
                "type": "int8",
                "symmetric": True,
                "strategy": "group",
                "group_size": 32,
            },
            "*wide.weights": {
                "type": "int4",
                "symmetric": True,
                "strategy": "group",
                "group_size": 32,
            },
        }
    )
    result = get_supported_layers(model, qc, AlgorithmRegistry)
    assert set(result) == {"small", "wide"}
    assert result["small"]["weights_cfg"]["group_size"] == 32
    assert result["wide"]["weights_cfg"]["quant_type"] == "int4"


def test_get_supported_layers_fuzzy_group_size_overrides_global():
    '''模糊配置覆盖全局 group_size 时，约束检查应使用逐层值'''
    model = _FuzzyModel()
    qc = _make_fuzzy_quant_config(
        {
            "weights": {
                "type": "int8",
                "symmetric": True,
                "strategy": "group",
                "group_size": 32,
            },
            "*small.weights": {
                "type": "int8",
                "symmetric": True,
                "strategy": "group",
                "group_size": 64,
            },
        }
    )
    result = get_supported_layers(model, qc, AlgorithmRegistry)
    # small 的 cin=64，被覆盖成 group_size=64 后应跳过；wide 仍用全局 32
    assert "small" not in result
    assert "wide" in result
    assert result["wide"]["weights_cfg"]["group_size"] == 32
