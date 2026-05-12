"""Tests for deploy_export pure-config functions.

`get_quant_ignore_linear_names` and `export_block_deploy` exercise QuantLinear
and the full pipeline; they are deferred to the tiny-config integration batch.
"""

import pytest

from amct_pytorch.common.models.llm.common.deploy_export import (
    generate_quant_config,
    generate_quant_group,
)


# ---- generate_quant_group ------------------------------------------------


def test_generate_quant_group_default_int():
    group = generate_quant_group()
    assert group["input_activations"]["num_bits"] == 8
    assert group["input_activations"]["strategy"] == "token"
    assert group["input_activations"]["dynamic"] is True
    assert group["weights"]["num_bits"] == 8
    assert group["weights"]["strategy"] == "channel"
    assert group["weights"]["dynamic"] is False
    assert group["activation_use_clip"] is False
    assert group["output_activations"] is None
    assert group["input_activations"]["type"] == "float"


def test_generate_quant_group_custom_bits_and_qtype():
    group = generate_quant_group(a_num_bits=4, w_num_bits=4, qtype="int", activation_use_clip=True)
    assert group["input_activations"]["num_bits"] == 4
    assert group["weights"]["num_bits"] == 4
    assert group["input_activations"]["type"] == "int"
    assert group["weights"]["type"] == "int"
    assert group["activation_use_clip"] is True


# ---- generate_quant_config -----------------------------------------------


def test_generate_quant_config_int_path_only_has_group_0():
    cfg = generate_quant_config(cache_scheme={}, ignores=["lm_head"])
    assert cfg["format"] == "int-quantized"
    assert cfg["ignore"] == ["lm_head"]
    assert cfg["quantization_status"] == "compressed"
    assert cfg["quant_method"] == "compressed-tensors"
    assert set(cfg["config_groups"]) == {"group_0"}
    assert cfg["config_groups"]["group_0"]["targets"] == ["Linear"]
    # int path: no weight_block_size key emitted.
    assert "weight_block_size" not in cfg


def test_generate_quant_config_mx_w8_adds_moegmm_group():
    cfg = generate_quant_config(cache_scheme={}, ignores=[], is_mx=True)
    assert cfg["format"] == "float-quantized"
    assert set(cfg["config_groups"]) == {"group_0", "group_1"}
    assert cfg["config_groups"]["group_1"]["targets"] == ["MoEGMM"]
    assert cfg["config_groups"]["group_1"]["weights"]["num_bits"] == 8
    assert cfg["weight_block_size"] == [1, 32]


def test_generate_quant_config_mx_w4a8_uses_4bit_weights_in_moegmm():
    cfg = generate_quant_config(cache_scheme={}, ignores=[], w4a8=True, is_mx=True)
    assert cfg["config_groups"]["group_1"]["weights"]["num_bits"] == 4
    assert cfg["config_groups"]["group_1"]["input_activations"]["num_bits"] == 8


def test_generate_quant_config_mx_w4a4_splits_up_gate_and_down():
    cfg = generate_quant_config(cache_scheme={}, ignores=[], w4a4=True, is_mx=True)
    assert set(cfg["config_groups"]) == {"group_0", "group_1", "group_2"}
    assert cfg["config_groups"]["group_1"]["targets"] == ["MoEGMMUpGate"]
    assert cfg["config_groups"]["group_1"]["weights"]["num_bits"] == 4
    assert cfg["config_groups"]["group_1"]["input_activations"]["num_bits"] == 4
    assert cfg["config_groups"]["group_2"]["targets"] == ["MoEGMMDown"]
    assert cfg["config_groups"]["group_2"]["weights"]["num_bits"] == 4
    assert cfg["config_groups"]["group_2"]["input_activations"]["num_bits"] == 8


def test_generate_quant_config_merges_cache_scheme_into_top_level():
    cache = {"kv_cache_scheme": {"strategy": "channel", "num_bits": 8}}
    cfg = generate_quant_config(cache_scheme=cache, ignores=[])
    assert cfg["kv_cache_scheme"] == cache["kv_cache_scheme"]


def test_generate_quant_config_has_global_compression_ratio_one():
    cfg = generate_quant_config(cache_scheme={}, ignores=[])
    assert cfg["global_compression_ratio"] == 1


import torch.nn as nn
from amct_pytorch.common.models.llm.common.deploy_export import (
    get_quant_ignore_linear_names,
    export_block_deploy,
)
from amct_pytorch.quantization.modules.quant_linear import QuantLinear


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
            nn.Module.__init__(self)
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
    pipeline = __import__("types").SimpleNamespace(
        args=__import__("types").SimpleNamespace(device="cpu"),
        build_quant_block=lambda idx: nn.Linear(4, 4),
        load_selected_layer_ptq_params=lambda idx, block, strict: None,
        get_layer_weight_prefix=lambda idx: "p.",
        iter_deploy_bindings=lambda idx, block: [("p.key.weight", fql)],
    )
    deploy_tensors, tensor_routes = export_block_deploy(pipeline, 0, [])
    assert deploy_tensors["p.key.weight"] == "qw_val"
    assert deploy_tensors["p.key.weight_scale"] == "ws_val"
    assert tensor_routes["p.key.weight_scale"] == "p.key.weight"


def test_export_block_deploy_skips_none_extra_tensor():
    class FakeQuantLinear:
        def export_deploy(self):
            return {"qweight": "qw", "extra": None}

    fql = FakeQuantLinear()
    pipeline = __import__("types").SimpleNamespace(
        args=__import__("types").SimpleNamespace(device="cpu"),
        build_quant_block=lambda idx: nn.Linear(4, 4),
        load_selected_layer_ptq_params=lambda idx, block, strict: None,
        get_layer_weight_prefix=lambda idx: "",
        iter_deploy_bindings=lambda idx, block: [("k.weight", fql)],
    )
    deploy_tensors, _ = export_block_deploy(pipeline, 0, [])
    assert "k.extra" not in deploy_tensors
