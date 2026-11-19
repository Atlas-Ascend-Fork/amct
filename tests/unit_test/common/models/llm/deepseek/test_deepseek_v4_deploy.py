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

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from amct_pytorch.common.models.llm.common.deployment.deploy_export import (
    make_description_from_plan,
)
from amct_pytorch.common.models.llm.common.deployment.deploy_selection import (
    DeploySelection,
    QuantLayerRule,
    QuantSpec,
)
from amct_pytorch.common.models.llm.deepseek import deepseek_deploy_common
from amct_pytorch.common.models.llm.deepseek.deepseek_v4.deepseekv4 import (
    DeepseekV4,
)

SHARD = "model.safetensors"
PACKED = "layers.0.ffn.experts.0.w1"
SHARED = "layers.0.ffn.shared_experts.w1"
MTP = "mtp.0.attn.wq_a"
WO_A = "layers.0.attn.wo_a"
PLAIN = "layers.0.attn.wq_a"


def _args(**overrides):
    values = {
        "model_name": "deepseek_v4",
        "model": "/fake/model",
        "output_dir": "/fake/output",
        "deploy_platform": "A3",
        "quant_dtype": None,
        "granularity": "tensor",
        "deploy_format": "ascend",
        "quant_layers_config": None,
        "deploy_selection": None,
        "algos": [],
        "attn_linear_param_dir": None,
        "attn_cache_param_dir": None,
        "moe_mlp_param_dir": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _bare_model(tmp_path, selection):
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "output"
    model_dir.mkdir()
    output_dir.mkdir()
    model = DeepseekV4.__new__(DeepseekV4)
    model.args = _args(
        model=str(model_dir),
        output_dir=str(output_dir),
        deploy_selection=selection,
        quant_dtype="int",
    )
    model.model_path = model_dir
    model.output_dir = output_dir
    model.safetensors_files = frozenset()
    model.processed_modules = set()
    model.config = {}
    return model


def _rule(pattern):
    return QuantLayerRule(pattern, QuantSpec("int", 8, 8))


def _hifp_rule(pattern):
    return QuantLayerRule(pattern, QuantSpec("hifp", 8, 8))


@pytest.fixture
def tensor_deploy_args(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "deepseek_v4"}), encoding="utf-8"
    )
    save_file(
        {f"{PLAIN}.weight": torch.ones(2, 4, dtype=torch.bfloat16)},
        str(model_dir / SHARD),
    )
    return _args(
        trust_remote_code=True,
        model=str(model_dir),
        output_dir=str(tmp_path / "output"),
        deploy_selection=DeploySelection((_rule(PLAIN),)),
    )


@pytest.mark.parametrize(
    "platform,dtype",
    [("A3", "int"), ("ascend950", "int"), ("ascend950", "hifp")],
)
@pytest.mark.parametrize("existing_output", [False, True])
def test_tensor_init_loads_checkpoint_metadata(
    tensor_deploy_args, tmp_path, platform, dtype, existing_output
):
    args = tensor_deploy_args
    args.deploy_platform = platform
    args.deploy_selection = DeploySelection(
        (QuantLayerRule(PLAIN, QuantSpec(dtype, 8, 8)),)
    )
    if existing_output:
        (tmp_path / "output").mkdir()

    model = DeepseekV4(args)

    assert model.config == {"model_type": "deepseek_v4"}
    assert model.quant_dtype == dtype
    assert model.model_path == tmp_path / "model"
    assert model.output_dir == tmp_path / "output"
    assert model.safetensors_files == frozenset(
        {(tmp_path / "model" / SHARD).resolve()}
    )
    assert model.processed_modules == set()
    assert model.output_tensor_bytes == 0


@pytest.mark.parametrize("output_path", ["model", "model/output", "."])
def test_tensor_init_rejects_overlapping_directories(
    tensor_deploy_args, tmp_path, output_path
):
    tensor_deploy_args.output_dir = str(tmp_path / output_path)

    with pytest.raises(ValueError, match="must not overlap"):
        DeepseekV4(tensor_deploy_args)


@pytest.mark.parametrize("output_is_file", [False, True])
def test_tensor_init_rejects_nonempty_or_file_output(
    tensor_deploy_args, tmp_path, output_is_file
):
    output = tmp_path / "output"
    if output_is_file:
        existing = output
    else:
        output.mkdir()
        existing = output / "existing.txt"
    existing.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="must be new or empty"):
        DeepseekV4(tensor_deploy_args)

    assert existing.read_text(encoding="utf-8") == "keep"


def test_validate_tensorwise_result_accepts_complete_conversion(tmp_path):
    model = _bare_model(tmp_path, DeploySelection((_rule(PLAIN),)))
    model.processed_modules = {PLAIN}
    plan = SimpleNamespace(selected_modules=frozenset({PLAIN}))

    assert model.validate_tensorwise_result(plan) == {PLAIN}


@pytest.mark.parametrize("processed", [frozenset(), frozenset({PLAIN, SHARED})])
def test_validate_tensorwise_result_rejects_missing_or_extra_modules(
    tmp_path, processed
):
    model = _bare_model(tmp_path, DeploySelection((_rule(PLAIN),)))
    model.processed_modules = set(processed)
    plan = SimpleNamespace(selected_modules=frozenset({PLAIN}))

    with pytest.raises(ValueError, match="differ from the deployment plan"):
        model.validate_tensorwise_result(plan)


@pytest.mark.parametrize("dtype", ["int", "hifp"])
def test_convert_shard_preserves_orphan_scale_and_unselected_float(tmp_path, dtype):
    model = _bare_model(tmp_path, DeploySelection((_rule(PLAIN),)))
    model.args.quant_dtype = dtype
    model.args.deploy_platform = "ascend950"
    tensors = {
        "orphan.scale": torch.tensor([1.25, 2.5], dtype=torch.float32),
        "norm.weight": torch.tensor([3.0, 4.0], dtype=torch.float32),
    }
    save_file(tensors, str(model.model_path / SHARD))
    model.safetensors_files = frozenset({(model.model_path / SHARD).resolve()})
    weight_map = dict.fromkeys(tensors, SHARD)

    routing = model.convert_tensorwise_shard(
        SHARD, model.model_path, weight_map, {}, {}
    )
    output = load_file(str(model.output_dir / SHARD))

    assert routing == weight_map
    assert output["orphan.scale"].dtype == torch.float32
    assert torch.equal(output["orphan.scale"], tensors["orphan.scale"])
    assert output["norm.weight"].dtype == torch.float32
    assert torch.equal(output["norm.weight"], tensors["norm.weight"])
    assert model.processed_modules == set()


@pytest.mark.parametrize(
    ("platform", "dtype"),
    [("A3", "int"), ("ascend950", "hifp")],
)
def test_convert_shard_rejects_fp8_weight_without_scale(tmp_path, platform, dtype):
    rule = _rule(PLAIN) if dtype == "int" else _hifp_rule(PLAIN)
    model = _bare_model(tmp_path, DeploySelection((rule,)))
    model.args.deploy_platform = platform
    model.args.quant_dtype = dtype
    tensors = {f"{PLAIN}.weight": torch.ones((2, 64), dtype=torch.float8_e4m3fn)}
    save_file(tensors, str(model.model_path / SHARD))
    model.safetensors_files = frozenset({(model.model_path / SHARD).resolve()})

    with pytest.raises(ValueError, match="Missing source scale"):
        model.convert_tensorwise_shard(
            SHARD,
            model.model_path,
            dict.fromkeys(tensors, SHARD),
            {PLAIN: 8},
            {},
        )


def test_validate_deploy_args_requires_a3_structured_int_json(tmp_path):
    path = tmp_path / "layers.json"
    path.write_text(
        json.dumps(
            {
                "quant_layers": [
                    {
                        "pattern": PACKED,
                        "quant_dtype": "int",
                        "w_bits": 8,
                        "a_bits": 8,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    args = _args(quant_layers_config=str(path))

    DeepseekV4.validate_deploy_args(args)

    assert args.quant_dtype == "int"
    assert args.deploy_selection.uses_structured_rules


def test_prepare_plan_allows_mtp_selection(tmp_path):
    selection = DeploySelection((_rule(PACKED), _rule(MTP)))
    model = _bare_model(tmp_path, selection)
    weight_map = {
        f"{PACKED}.weight": SHARD,
        f"{PACKED}.scale": SHARD,
        f"{MTP}.weight": SHARD,
        f"{MTP}.scale": SHARD,
    }
    save_file(
        {
            f"{PACKED}.weight": torch.ones((2, 2)),
            f"{PACKED}.scale": torch.ones((2, 1), dtype=torch.uint8),
            f"{MTP}.weight": torch.ones((2, 2)),
            f"{MTP}.scale": torch.ones((2, 1), dtype=torch.uint8),
        },
        str(model.model_path / SHARD),
    )
    model.safetensors_files = frozenset({(model.model_path / SHARD).resolve()})

    plan = model.prepare_tensor_deploy_plan(weight_map)

    assert plan.selected_modules == {PACKED, MTP}
    assert plan.mtp.selected == {MTP}


def test_convert_shard_decodes_packed_and_shared_source_weights(tmp_path, monkeypatch):
    selection = DeploySelection((_rule(PACKED),))
    model = _bare_model(tmp_path, selection)
    packed_scale = torch.full((2, 1), 127, dtype=torch.uint8)
    shared_scale = torch.tensor([[127, 127], [128, 128]], dtype=torch.uint8)
    tensors = {
        f"{PACKED}.weight": torch.tensor(
            [[0x11, 0x22, 0x33, 0x44], [0x55, 0x66, 0x77, -0x08]], dtype=torch.int8
        ),
        f"{PACKED}.scale": packed_scale,
        f"{SHARED}.weight": torch.ones((64, 64), dtype=torch.float8_e4m3fn),
        f"{SHARED}.scale": shared_scale,
    }
    save_file(tensors, str(model.model_path / SHARD))
    model.safetensors_files = frozenset({(model.model_path / SHARD).resolve()})
    weight_map = dict.fromkeys(tensors, SHARD)

    def fake_quant_payload(_, weight_name, weight, bit):
        assert bit == 8
        scale_name = weight_name.replace(".weight", ".weight_scale")
        return {
            weight_name: torch.zeros_like(weight, dtype=torch.int8),
            scale_name: torch.ones((weight.shape[0], 1)),
        }

    monkeypatch.setattr(
        deepseek_deploy_common,
        "quant_payload",
        fake_quant_payload,
    )

    model.convert_tensorwise_shard(SHARD, model.model_path, weight_map, {PACKED: 8}, {})
    output = load_file(str(model.output_dir / SHARD))

    assert output[f"{PACKED}.weight"].dtype == torch.int8
    assert output[f"{PACKED}.weight_offset"].dtype == torch.bfloat16
    assert output[f"{SHARED}.weight"].dtype == torch.bfloat16
    assert output[f"{SHARED}.weight"][0, 0] == 1
    assert output[f"{SHARED}.weight"][32, 0] == 2
    assert f"{PACKED}.scale" not in output
    assert f"{SHARED}.scale" not in output


@pytest.mark.parametrize(
    ("platform", "dtype"),
    [("A3", "int"), ("ascend950", "int"), ("ascend950", "hifp")],
)
def test_validate_deploy_source_accepts_dsv4_fp8_checkpoint(tmp_path, platform, dtype):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "quantization_config": {"quant_method": "fp8"},
            }
        ),
        encoding="utf-8",
    )

    args = _args(model=str(tmp_path), deploy_platform=platform, quant_dtype=dtype)
    assert DeepseekV4.validate_deploy_source(args)


@pytest.mark.parametrize(
    ("platform", "dtype"),
    [("A3", "int"), ("ascend950", "int"), ("ascend950", "hifp")],
)
def test_validate_deploy_args_accepts_dsv4_supported_platform_dtype(platform, dtype):
    selection = DeploySelection(
        (_rule(PACKED) if dtype == "int" else _hifp_rule(PACKED),)
    )
    args = _args(
        deploy_platform=platform,
        quant_dtype=dtype,
        deploy_selection=selection,
    )

    DeepseekV4.validate_deploy_args(args)

    assert args.quant_dtype == dtype
    assert args.deploy_platform == platform


def test_validate_deploy_args_requires_model_name():
    with pytest.raises(ValueError, match="--model_name is required"):
        DeepseekV4.validate_deploy_args(_args(model_name=""))


def test_convert_hifp_shard_writes_codes_and_grouped_wo_a_scale(tmp_path, monkeypatch):
    groups, rank, columns = 2, 32, 64
    selection = DeploySelection((_hifp_rule(PLAIN), _hifp_rule(WO_A)))
    model = _bare_model(tmp_path, selection)
    model.args.deploy_platform = "ascend950"
    model.args.quant_dtype = "hifp"
    model.config = {
        "text_config": {"o_groups": groups, "o_lora_rank": rank},
        "quantization_config": {"quant_method": "fp8"},
    }
    tensors = {}
    for name in (PLAIN, WO_A):
        tensors[f"{name}.weight"] = torch.ones(
            (groups * rank, columns), dtype=torch.float8_e4m3fn
        )
        tensors[f"{name}.scale"] = torch.full(
            (groups, columns // 32), 127, dtype=torch.uint8
        ).view(torch.float8_e8m0fnu)
    save_file(tensors, str(model.model_path / SHARD))
    model.safetensors_files = frozenset({(model.model_path / SHARD).resolve()})
    monkeypatch.setattr(
        deepseek_deploy_common,
        "hif8_pack",
        lambda weight: (
            torch.full(weight.shape, 7, dtype=torch.uint8),
            torch.arange(weight.shape[0], dtype=torch.float32).view(-1, 1),
        ),
    )

    model.convert_tensorwise_shard(
        SHARD,
        model.model_path,
        dict.fromkeys(tensors, SHARD),
        {PLAIN: 8, WO_A: 8},
        {},
    )
    output = load_file(str(model.output_dir / SHARD))

    assert output[f"{PLAIN}.weight"].dtype == torch.uint8
    assert output[f"{PLAIN}.weight_scale"].dtype == torch.float32
    assert output[f"{WO_A}.weight"].shape == (groups * rank, columns)
    assert output[f"{WO_A}.weight_scale"].shape == (groups * rank, 1)
    assert torch.equal(
        output[f"{WO_A}.weight_scale"].view(groups, rank),
        output[f"{WO_A}.weight_scale"].view(groups, rank)[0].expand(groups, rank),
    )
    assert f"{PLAIN}.scale" not in output
    assert f"{WO_A}.scale" not in output


def test_make_description_marks_hifp_modules():
    names = {f"{PLAIN}.weight", f"{PLAIN}.weight_scale", "norm.weight"}
    plan = SimpleNamespace(
        module_quant_specs={PLAIN: QuantSpec("hifp", 8, 8)},
        selected_modules=frozenset({PLAIN}),
    )

    description = make_description_from_plan(names, plan)

    assert description[PLAIN + ".weight"] == "W8A8_HIFP8"
    assert description[PLAIN + ".weight_scale"] == "W8A8_HIFP8"
    assert description["norm.weight"] == "FLOAT"


def test_refresh_config_removes_source_quantization_config(tmp_path):
    model = _bare_model(tmp_path, DeploySelection((_rule(PACKED),)))
    model.config = {
        "model_type": "deepseek_v4",
        "expert_dtype": "fp4",
        "quantization_config": {"quant_method": "fp8"},
    }

    model.refresh_config([])

    result = json.loads((model.output_dir / "config.json").read_text(encoding="utf-8"))
    assert result["expert_dtype"] == "fp4"
    assert "quantization_config" not in result


def test_make_description_marks_selected_mtp_modules_as_w8a8_dynamic(tmp_path):
    output_names = {
        f"{MTP}.weight",
        f"{MTP}.weight_scale",
        f"{MTP}.weight_offset",
        "mtp.0.attn.q_norm.weight",
    }
    plan = SimpleNamespace(
        module_quant_specs={MTP: QuantSpec("int", 8, 8)},
        selected_modules=frozenset({MTP}),
    )

    description = make_description_from_plan(output_names, plan)

    assert description[MTP + ".weight"] == "W8A8_DYNAMIC"
    assert description[MTP + ".weight_scale"] == "W8A8_DYNAMIC"
    assert description[MTP + ".weight_offset"] == "W8A8_DYNAMIC"
    assert description["mtp.0.attn.q_norm.weight"] == "FLOAT"
