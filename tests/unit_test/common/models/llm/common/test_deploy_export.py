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
import torch
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
    assert group[INPUT_ACTIVATIONS]["strategy"] == "group"
    assert group[INPUT_ACTIVATIONS]["dynamic"] is True
    assert group[WEIGHTS][NUM_BITS] == 8
    assert group[WEIGHTS]["strategy"] == "group"
    assert group[WEIGHTS]["dynamic"] is False
    assert group["activation_use_clip"] is False
    assert group["output_activations"] is None
    assert group[INPUT_ACTIVATIONS]["type"] == "float"


def test_generate_quant_group_custom_bits_and_qtype():
    group = generate_quant_group(a_bits=4, w_bits=4, qtype=INT, activation_use_clip=True)
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
    assert set(cfg[CONFIG_GROUPS_KEY]) == {GROUP_0, GROUP_1}
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


def test_generate_quant_config_moegmm_w4a8_via_bits_scheme():
    scheme = [
        {"targets": ["Linear"], "w_bits": 8, "a_bits": 8},
        {"targets": ["MoEGMM"], "w_bits": 8, "a_bits": 8},
    ]
    cfg = generate_quant_config(cache_scheme={}, ignores=[], is_mx=True, bits_scheme=scheme)
    assert cfg[CONFIG_GROUPS_KEY][GROUP_1][WEIGHTS][NUM_BITS] == 8
    assert cfg[CONFIG_GROUPS_KEY][GROUP_1][INPUT_ACTIVATIONS][NUM_BITS] == 8


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


# ---- convert_state_dict (new in diff) ------------------------------------


def test_convert_state_dict_non_fp8_passthrough(tmp_path):
    """Non-FP8 weight (element_size > 1) is returned unchanged."""
    from amct_pytorch.common.models.llm.common.deploy_export import convert_state_dict

    weight = torch.randn(4, 4, dtype=torch.float32)
    result = convert_state_dict(
        weight, "layer.weight", "layer.weight_scale_inv",
        {}, tmp_path, {}, 32,
    )
    assert result is weight


def test_convert_state_dict_fp8_with_scale_inv_loaded(tmp_path):
    """FP8 int8 weight with scale_inv present gets dequantized."""
    from safetensors.torch import save_file as sf_save
    from amct_pytorch.common.models.llm.common.deploy_export import convert_state_dict

    # Create a safetensors file with scale_inv
    scale_inv = torch.ones(1, 2, dtype=torch.float32) * 0.5
    sf_save({"layer.weight_scale_inv": scale_inv}, str(tmp_path / "shard.safetensors"))

    weight = torch.ones(2, 4, dtype=torch.int8)
    weight_map = {"layer.weight_scale_inv": "shard.safetensors"}
    loaded_files = {}

    result = convert_state_dict(
        weight, "layer.weight", "layer.weight_scale_inv",
        weight_map, tmp_path, loaded_files, block_size=32,
    )
    # int8 packed dequant unpacks columns, so shape changes
    assert result.dtype != torch.int8
    assert result.shape[0] == weight.shape[0]


def test_convert_state_dict_fp8_missing_scale_inv_prints_warning(tmp_path, monkeypatch):
    """Missing scale_inv logs warning and returns original weight."""
    from amct_pytorch.common.models.llm.common import deploy_export as deploy_export_mod

    warnings = []
    monkeypatch.setattr(deploy_export_mod.logger, "warning", lambda message: warnings.append(message))

    weight = torch.ones(4, 4, dtype=torch.int8)
    weight_map = {}  # no scale_inv entry
    result = deploy_export_mod.convert_state_dict(
        weight, "layer.weight", "layer.weight_scale_inv",
        weight_map, tmp_path, {}, 32,
    )
    assert any("Missing scale_inv" in message for message in warnings)
    assert result is weight


def test_convert_state_dict_fp8_non_int8_dtype(tmp_path):
    """FP8 weight with non-int8 dtype uses non-packed dequant (block_size=1)."""
    from safetensors.torch import save_file as sf_save
    from amct_pytorch.common.models.llm.common.deploy_export import convert_state_dict

    scale_inv = torch.ones(2, 4, dtype=torch.float32)
    sf_save({"layer.weight_scale_inv": scale_inv}, str(tmp_path / "shard.safetensors"))

    # Use uint8 to simulate a non-int8 1-byte dtype (float8 not available on CPU)
    weight = torch.ones(2, 4, dtype=torch.uint8)
    weight_map = {"layer.weight_scale_inv": "shard.safetensors"}
    loaded_files = {}

    result = convert_state_dict(
        weight, "layer.weight", "layer.weight_scale_inv",
        weight_map, tmp_path, loaded_files, block_size=1,
    )
    assert result.dtype != torch.uint8


def test_convert_state_dict_reuses_loaded_file(tmp_path):
    """Already-loaded file is reused from loaded_files cache."""
    from safetensors.torch import save_file as sf_save
    from amct_pytorch.common.models.llm.common.deploy_export import convert_state_dict

    scale_inv = torch.ones(1, 2, dtype=torch.float32) * 2.0
    sf_save({"layer.weight_scale_inv": scale_inv}, str(tmp_path / "shard.safetensors"))

    # Pre-load the file
    from safetensors.torch import load_file as sf_load
    loaded_files = {"shard.safetensors": sf_load(str(tmp_path / "shard.safetensors"))}

    weight = torch.ones(2, 4, dtype=torch.int8)
    weight_map = {"layer.weight_scale_inv": "shard.safetensors"}

    result = convert_state_dict(
        weight, "layer.weight", "layer.weight_scale_inv",
        weight_map, tmp_path, loaded_files, block_size=32,
    )
    assert result.dtype != torch.int8


# ---- quant_payload (new in diff) -----------------------------------------


def test_quant_payload_basic():
    """quant_payload builds tensors dict from quant_cls.export_deploy."""
    from amct_pytorch.common.models.llm.common.deploy_export import quant_payload

    class FakeQuantCls:
        def __init__(self, bits):
            self.bits = bits

        def export_deploy(self, weight):
            return {
                "qweight": torch.ones(4, 4) * self.bits,
                "weight_scale": torch.ones(4, 1),
            }

    weight = torch.randn(4, 4)
    result = quant_payload(FakeQuantCls, "layer.weight", weight, bit=8)
    assert "layer.weight" in result
    assert torch.allclose(result["layer.weight"], torch.ones(4, 4) * 8)
    assert "layer.weight_scale" in result


def test_quant_payload_skips_none_extras():
    """quant_payload skips extra tensors that are None."""
    from amct_pytorch.common.models.llm.common.deploy_export import quant_payload

    class FakeQuantCls:
        def __init__(self, bits):
            pass

        def export_deploy(self, weight):
            _ = self
            return {"qweight": weight, "bias": None}

    weight = torch.randn(4, 4)
    result = quant_payload(FakeQuantCls, "layer.weight", weight, bit=4)
    assert "layer.weight" in result
    assert "layer.bias" not in result

