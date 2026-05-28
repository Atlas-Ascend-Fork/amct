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

from amct_pytorch.algorithms import AlgorithmRegistry
from amct_pytorch.common.config.fields import (
    AlgorithmField,
    AwqField,
    BatchNumField,
    CastField,
    CustomAlgField,
    GptqField,
    InputsCfgField,
    MinmaxField,
    MxQuant,
    QuantCfgField,
    QuantConfig,
    SkipLayersField,
    SmoothQuantField,
    WeightsCfgField,
)

POSITIVE_INT = 'positive int'

# ---- BatchNumField ---------------------------------------------------------

ALGO_A = 'algo_a'
INT8 = 'int8'
MY_UNIQUE_ALGO = 'my_unique_algo'

ALGO_B = 'algo_b'


def test_batch_num_valid():
    f = BatchNumField(4)
    assert f.get_value() == {"batch_num": 4}


def test_batch_num_zero_raises():
    with pytest.raises(ValueError, match=POSITIVE_INT):
        BatchNumField(0)


def test_batch_num_negative_raises():
    with pytest.raises(ValueError, match=POSITIVE_INT):
        BatchNumField(-1)


def test_batch_num_non_int_raises():
    with pytest.raises(ValueError, match=POSITIVE_INT):
        BatchNumField("abc")


# ---- WeightsCfgField -------------------------------------------------------


def test_weights_cfg_valid():
    f = WeightsCfgField({"type": INT8, "symmetric": True, "strategy": "channel"})
    assert f.get_value() == {"quant_type": INT8, "symmetric": True, "strategy": "channel"}


def test_weights_cfg_with_group_size():
    f = WeightsCfgField({"type": "int4", "symmetric": True, "strategy": "group", "group_size": 128})
    v = f.get_value()
    assert v["group_size"] == 128


def test_weights_cfg_none_config():
    f = WeightsCfgField(None)
    assert f.get_value() is None


def test_weights_cfg_invalid_dtype():
    with pytest.raises(ValueError, match="Weights quant_dtype only support"):
        WeightsCfgField({"type": "float64", "symmetric": True, "strategy": "channel"})


def test_weights_cfg_invalid_symmetric():
    with pytest.raises(ValueError, match="symmetric only support bool"):
        WeightsCfgField({"type": INT8, "symmetric": "yes", "strategy": "channel"})


def test_weights_cfg_asymmetric_non_int():
    with pytest.raises(ValueError, match="symmetric only support to be True"):
        WeightsCfgField({"type": "float8_e4m3fn", "symmetric": False, "strategy": "channel"})


def test_weights_cfg_invalid_strategy():
    with pytest.raises(ValueError, match="strategy only support"):
        WeightsCfgField({"type": INT8, "symmetric": True, "strategy": "invalid"})


def test_weights_cfg_group_size_without_group_strategy():
    with pytest.raises(ValueError, match="group_size only support strategy group"):
        WeightsCfgField({"type": "int4", "symmetric": True, "strategy": "channel", "group_size": 64})


def test_weights_cfg_group_strategy_without_group_size():
    with pytest.raises(ValueError, match="group_size is necessary"):
        WeightsCfgField({"type": INT8, "symmetric": True, "strategy": "group"})


def test_weights_cfg_group_size_negative():
    with pytest.raises(ValueError, match=POSITIVE_INT):
        WeightsCfgField({"type": "int4", "symmetric": True, "strategy": "group", "group_size": -5})


def test_weights_cfg_group_size_not_int():
    with pytest.raises(ValueError, match=POSITIVE_INT):
        WeightsCfgField({"type": "int4", "symmetric": True, "strategy": "group", "group_size": "32"})


def test_weights_cfg_mxfp8_group_size_32_valid():
    f = WeightsCfgField({"type": "mxfp8_e4m3fn", "symmetric": True, "strategy": "group", "group_size": 32})
    assert f.get_value()["group_size"] == 32


def test_weights_cfg_mxfp8_group_size_64_invalid():
    with pytest.raises(ValueError, match="only support group_size value"):
        WeightsCfgField({"type": "mxfp8_e4m3fn", "symmetric": True, "strategy": "group", "group_size": 64})


# ---- InputsCfgField --------------------------------------------------------


def test_inputs_cfg_valid():
    f = InputsCfgField({"type": INT8, "symmetric": True, "strategy": "tensor"})
    assert f.get_value() == {"quant_type": INT8, "symmetric": True, "strategy": "tensor", "dynamic": None}


def test_inputs_cfg_disabled():
    f = InputsCfgField({"enable_quant": False})
    assert f.get_value() == {"enable_quant": False}


def test_inputs_cfg_invalid_dtype():
    with pytest.raises(ValueError, match="Inputs quant_dtype only support"):
        InputsCfgField({"type": "float64", "symmetric": True, "strategy": "tensor"})


def test_inputs_cfg_invalid_symmetric():
    with pytest.raises(ValueError, match="symmetric only support bool"):
        InputsCfgField({"type": INT8, "symmetric": "no", "strategy": "tensor"})


def test_inputs_cfg_asymmetric_non_int8_raises():
    with pytest.raises(ValueError, match="symmetric is unsupported to be False"):
        InputsCfgField({"type": "hifloat8", "symmetric": False, "strategy": "tensor"})


def test_inputs_cfg_token_with_asymmetric_raises():
    with pytest.raises(ValueError, match="token do not support asymmetric"):
        InputsCfgField({"type": INT8, "symmetric": False, "strategy": "token"})


def test_inputs_cfg_dynamic_without_token_strategy():
    with pytest.raises(ValueError, match="dynamic only support strategy"):
        InputsCfgField({"type": INT8, "symmetric": True, "strategy": "tensor", "dynamic": True})


def test_inputs_cfg_mxfp8_strategy_group():
    f = InputsCfgField({"type": "mxfp8_e4m3fn", "symmetric": True, "strategy": "group"})
    assert f.get_value()["strategy"] == "group"


# ---- QuantCfgField ---------------------------------------------------------


def test_quant_cfg_with_weights_and_inputs():
    cfg = {"weights": {"type": INT8, "symmetric": True, "strategy": "channel"},
           "inputs": {"type": INT8, "symmetric": True, "strategy": "tensor"}}
    f = QuantCfgField(cfg)
    v = f.get_value()
    assert v["weights_cfg"]["quant_type"] == INT8
    assert v["inputs_cfg"]["quant_type"] == INT8


def test_quant_cfg_with_fuzzy_patterns():
    cfg = {"*self_attn.q_proj.weights": {"type": "int4", "symmetric": True, "strategy": "channel"},
           "*self_attn.q_proj.inputs": {"type": INT8, "symmetric": True, "strategy": "tensor"}}
    f = QuantCfgField(cfg)
    assert len(f.fuzzy_configs["weights"]) == 1
    assert len(f.fuzzy_configs["inputs"]) == 1
    assert f.fuzzy_configs["weights"][0]["pattern"] == "*self_attn.q_proj.weights"
    assert f.fuzzy_configs["inputs"][0]["pattern"] == "*self_attn.q_proj.inputs"


def test_quant_cfg_inputs_only_with_default_kvcache():
    cfg = QuantCfgField({"inputs": {"type": INT8, "symmetric": True, "strategy": "tensor"}})
    assert cfg.kvcache_cfg.get_value() == {"enable_quant": False}


def test_quant_cfg_get_fuzzy_config_matches():
    cfg = {"*self_attn.q_proj.weights": {"type": "int4", "symmetric": True, "strategy": "channel"}}
    f = QuantCfgField(cfg)
    result = f.get_fuzzy_config("model.layers.0.self_attn.q_proj", "weights")
    assert result == {"type": "int4", "symmetric": True, "strategy": "channel"}


def test_quant_cfg_get_fuzzy_config_no_match():
    cfg = {"*self_attn.q_proj.weights": {"type": "int4", "symmetric": True, "strategy": "channel"}}
    f = QuantCfgField(cfg)
    assert f.get_fuzzy_config("model.layers.0.mlp.gate_proj", "weights") is None


# ---- AwqField --------------------------------------------------------------


def test_awq_field_valid():
    f = AwqField({"grids_num": 20})
    assert f.get_value() == {"awq": {"grids_num": 20}}


def test_awq_field_none_raises():
    with pytest.raises(ValueError, match="grids_num is necessary"):
        AwqField(None)


def test_awq_field_missing_grids_num_raises():
    with pytest.raises(ValueError, match="grids_num is necessary"):
        AwqField({})


def test_awq_field_negative_grids_num_raises():
    with pytest.raises(ValueError, match=POSITIVE_INT):
        AwqField({"grids_num": -1})


def test_awq_field_non_int_raises():
    with pytest.raises(ValueError, match=POSITIVE_INT):
        AwqField({"grids_num": "20"})


# ---- SmoothQuantField ------------------------------------------------------


def test_smooth_quant_valid():
    f = SmoothQuantField({"smooth_strength": 0.5})
    assert f.get_value() == {"smoothquant": {"smooth_strength": 0.5}}


def test_smooth_quant_none_raises():
    with pytest.raises(ValueError, match="smooth_strength is necessary"):
        SmoothQuantField(None)


def test_smooth_quant_strength_zero_raises():
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        SmoothQuantField({"smooth_strength": 0.0})


def test_smooth_quant_strength_one_raises():
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        SmoothQuantField({"smooth_strength": 1.0})


def test_smooth_quant_strength_non_float_raises():
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        SmoothQuantField({"smooth_strength": "0.5"})


# ---- GptqField -------------------------------------------------------------


def test_gptq_field():
    f = GptqField(None)
    assert f.get_value() == {"gptq"}


# ---- MinmaxField -----------------------------------------------------------


def test_minmax_field():
    f = MinmaxField(None)
    assert f.get_value() == {"minmax"}


# ---- MxQuant ---------------------------------------------------------------


def test_mxquant_field():
    f = MxQuant(None)
    assert f.get_value() == {"mxquant"}


# ---- CastField -------------------------------------------------------------


def test_cast_field():
    f = CastField(None)
    assert f.get_value() == {"cast"}


# ---- CustomAlgField --------------------------------------------------------


def test_custom_alg_field():
    f = CustomAlgField("myalgo", {"param": 1})
    assert f.get_value() == {"myalgo": {"param": 1}}


# ---- AlgorithmField --------------------------------------------------------


def test_algorithm_field_single_minmax():
    f = AlgorithmField({"minmax": {}}, AlgorithmRegistry)
    v = f.get_value()
    assert "minmax" in v["algorithm"]


def test_algorithm_field_multiple_algos(monkeypatch):
    fake_algo_a = type("AlgoA", (), {"keys": lambda self: ["Linear"]})()
    fake_algo_b = type("AlgoB", (), {"keys": lambda self: ["Conv2d"]})()
    fake_reg = type("FakeReg", (), {"algo": {ALGO_A: fake_algo_a, ALGO_B: fake_algo_b}})()
    monkeypatch.setattr(
        "amct_pytorch.common.config.fields.AlgorithmRegistry",
        fake_reg,
    )
    f = AlgorithmField({ALGO_A: {}, ALGO_B: {}}, fake_reg)
    v = f.get_value()
    assert ALGO_A in v["algorithm"]
    assert ALGO_B in v["algorithm"]


def test_algorithm_field_unknown_algo_raises():
    with pytest.raises(ValueError, match="Not support algorithm"):
        AlgorithmField({"nonexistent": {}}, AlgorithmRegistry)


def test_algorithm_field_str_input_accepted():
    f = AlgorithmField("minmax", AlgorithmRegistry)
    v = f.get_value()
    assert "minmax" in v["algorithm"]


def test_algorithm_field_gptq():
    f = AlgorithmField({"gptq"}, AlgorithmRegistry)
    v = f.get_value()
    assert "gptq" in v["algorithm"]


# ---- SkipLayersField -------------------------------------------------------


def test_skip_layers_valid():
    f = SkipLayersField(["lm_head", "embed"])
    assert f.get_value() == {"skip_layers": ["lm_head", "embed"]}


def test_skip_layers_empty():
    f = SkipLayersField([])
    assert f.get_value() == {"skip_layers": []}


def test_skip_layers_non_str_raises():
    with pytest.raises(ValueError, match="must be str"):
        SkipLayersField([123])


# ---- QuantConfig -----------------------------------------------------------


def test_quant_config_basic():
    cfg = {"batch_num": 2,
           "quant_cfg": {"weights": {"type": INT8, "symmetric": True, "strategy": "channel"}},
           "algorithm": {"minmax": {}},
           "skip_layers": ["lm_head"]}
    qc = QuantConfig(cfg, AlgorithmRegistry)
    assert qc.batch_num.get_value() == {"batch_num": 2}


def test_quant_config_get_layer_config_exact():
    cfg = {"batch_num": 1,
           "quant_cfg": {"weights": {"type": INT8, "symmetric": True, "strategy": "channel"}},
           "algorithm": {"minmax": {}}}
    qc = QuantConfig(cfg, AlgorithmRegistry)
    layer_cfg = qc.get_layer_config("layer.0")
    assert layer_cfg is not None
    assert layer_cfg["weights_cfg"]["quant_type"] == INT8


def test_quant_config_get_layer_config_fuzzy():
    cfg = {"batch_num": 1,
           "quant_cfg": {"*down_proj.weights": {"type": "int4", "symmetric": True, "strategy": "channel"}},
           "algorithm": {"minmax": {}}}
    qc = QuantConfig(cfg, AlgorithmRegistry)
    layer_cfg = qc.get_layer_config("model.layers.0.mlp.down_proj")
    assert layer_cfg["weights_cfg"]["quant_type"] == "int4"


def test_quant_config_get_layer_config_fuzzy_weights_override_inputs():
    cfg = {"batch_num": 1,
           "quant_cfg": {
               "*down_proj.weights": {"type": "int4", "symmetric": True, "strategy": "channel"},
               "*down_proj.inputs": {"type": INT8, "symmetric": True, "strategy": "tensor"}},
           "algorithm": {"minmax": {}}}
    qc = QuantConfig(cfg, AlgorithmRegistry)
    layer_cfg = qc.get_layer_config("model.layers.0.mlp.down_proj")
    assert layer_cfg["weights_cfg"]["quant_type"] == "int4"
    assert layer_cfg["inputs_cfg"]["quant_type"] == INT8


def test_quant_config_get_layer_config_no_match_returns_none():
    cfg = {"batch_num": 1,
           "quant_cfg": {"*down_proj.weights": {"type": "int4", "symmetric": True, "strategy": "channel"}},
           "algorithm": {"minmax": {}}}
    qc = QuantConfig(cfg, AlgorithmRegistry)
    assert qc.get_layer_config("model.layers.0.mlp.gate_proj") is None


def test_quant_config_get_layer_config_caching():
    cfg = {"batch_num": 1,
           "quant_cfg": {"weights": {"type": INT8, "symmetric": True, "strategy": "channel"}},
           "algorithm": {"minmax": {}}}
    qc = QuantConfig(cfg, AlgorithmRegistry)
    first = qc.get_layer_config("layer.0")
    second = qc.get_layer_config("layer.0")
    assert first is second


def test_quant_config_get_layer_config_fuzzy_inputs_fallback():
    cfg = {"batch_num": 1,
           "quant_cfg": {"*down_proj.weights": {"type": "int4", "symmetric": True, "strategy": "channel"}},
           "algorithm": {"minmax": {}}}
    qc = QuantConfig(cfg, AlgorithmRegistry)
    layer_cfg = qc.get_layer_config("model.layers.0.mlp.down_proj")
    assert layer_cfg["inputs_cfg"]["enable_quant"] is False


def test_quant_config_with_gptq():
    cfg = {"batch_num": 1,
           "quant_cfg": {"weights": {"type": "int4", "symmetric": True, "strategy": "channel"}},
           "algorithm": {"gptq"}}
    qc = QuantConfig(cfg, AlgorithmRegistry)
    assert qc.algorithm.names == ["gptq"]


def test_weights_cfg_asymmetric_int8():
    f = WeightsCfgField({"type": INT8, "symmetric": False, "strategy": "channel"})
    assert f.get_value()["symmetric"] is False


def test_weights_cfg_check_group_size_direct_call_unsupported_strategy():
    f = WeightsCfgField.__new__(WeightsCfgField)
    f.quant_type = "int4"
    f.strategy = "channel"
    f.group_size = 64
    with pytest.raises(ValueError, match="group_size only support strategy group"):
        f.check_group_size()


def test_weights_cfg_check_group_size_unsupported_dtype():
    f = WeightsCfgField.__new__(WeightsCfgField)
    f.quant_type = "float8_e4m3fn"
    f.strategy = "group"
    f.group_size = 64
    with pytest.raises(ValueError, match="group_size only support for"):
        f.check_group_size()


def test_inputs_cfg_quant_type_none_returns_none():
    f = InputsCfgField({"enable_quant": True})
    assert f.get_value() is None


def test_inputs_cfg_unquantized_returns_disable():
    f = InputsCfgField({"enable_quant": False})
    assert f.get_value() == {"enable_quant": False}


def test_inputs_cfg_invalid_strategy_not_mxfp8():
    with pytest.raises(ValueError, match="Inputs strategy only support"):
        InputsCfgField({"type": INT8, "symmetric": True, "strategy": "invalid"})


def test_algorithm_field_custom_algo():
    AlgorithmRegistry.algo[MY_UNIQUE_ALGO] = {"MatMul": object()}
    try:
        f = AlgorithmField({MY_UNIQUE_ALGO: {}}, AlgorithmRegistry)
        assert MY_UNIQUE_ALGO in f.names
    finally:
        del AlgorithmRegistry.algo[MY_UNIQUE_ALGO]


def test_algorithm_field_check_invalid_input():
    with pytest.raises(ValueError, match="Algorithm only support 1 str"):
        AlgorithmField.check({"a": "b", "c": "d"})


def test_quant_config_get_layer_config_inputs_fallback_enable_false():
    cfg = {"batch_num": 1,
           "quant_cfg": {"weights": {"type": INT8, "symmetric": True, "strategy": "channel"}},
           "algorithm": {"minmax": {}}}
    qc = QuantConfig(cfg, AlgorithmRegistry)
    layer_cfg = qc.get_layer_config("some_layer")
    assert layer_cfg["inputs_cfg"] == {"enable_quant": False}

