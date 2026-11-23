# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0.

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm import register_llm_models
from amct_pytorch.common.models.llm.common.deployment.deploy_export import (
    make_description_from_plan,
)
from amct_pytorch.common.models.llm.common.deployment.deploy_selection import (
    DeploySelection,
    MtpSelectionReport,
    QuantLayerRule,
    QuantSpec,
    load_deploy_selection,
)
from amct_pytorch.common.models.llm.glm.glm5_3.glm5_3 import GLM5_3

LAYER = "model.language_model.layers.{}"
DENSE = LAYER.format(0) + ".mlp.gate_proj"
DENSE_UP = LAYER.format(0) + ".mlp.up_proj"
OTHER_FP8 = LAYER.format(0) + ".mlp.down_proj"
FULL_QA = LAYER.format(3) + ".self_attn.q_a_proj"
FULL_KVA = LAYER.format(3) + ".self_attn.kv_a_proj_with_mqa"
FULL_QB = LAYER.format(3) + ".self_attn.q_b_proj"
FULL_O = LAYER.format(3) + ".self_attn.o_proj"
EXPERT = LAYER.format(4) + ".mlp.experts.0.gate_proj"
SHARED = LAYER.format(4) + ".mlp.shared_experts.gate_proj"
MTP = LAYER.format(45) + ".mlp.experts.0.gate_proj"
REPO_ROOT = Path(__file__).resolve().parents[5]


def _config(*, quantized=True):
    config = {
        "architectures": ["Glm5NextForConditionalGeneration"],
        "model_type": "glm5_next",
        "text_config": {
            "model_type": "glm5_next_text",
            "num_hidden_layers": 45,
            "num_nextn_predict_layers": 1,
            "first_k_dense_replace": 3,
            "n_routed_experts": 288,
            "n_shared_experts": 1,
            "layer_types": ["linear_attention"] * 45,
            "linear_attn_config": {
                "full_attn_layers": [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]
            },
        },
    }
    config["text_config"]["layer_types"][3] = "deepseek_sparse_attention"
    if quantized:
        config["quantization_config"] = {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "weight_block_size": [128, 128],
        }
    return config


def _args(model, output, selection, **overrides):
    values = {
        "model": str(model),
        "output_dir": str(output),
        "granularity": "tensor",
        "deploy_platform": "ascend950",
        "quant_dtype": None,
        "device": "cpu",
        "algos": [],
        "attn_linear_param_dir": "",
        "attn_cache_param_dir": "",
        "moe_mlp_param_dir": "",
        "deploy_selection": selection,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _write_model(tmp_path, tensors, *, quantized=True):
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "output"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(_config(quantized=quantized)), encoding="utf-8"
    )
    save_file(tensors, str(model_dir / "model.safetensors"))
    return model_dir, output_dir


def _bare_model(tmp_path, tensors, selection, *, quantized=True, **overrides):
    model_dir, output_dir = _write_model(tmp_path, tensors, quantized=quantized)
    output_dir.mkdir()
    model = GLM5_3.__new__(GLM5_3)
    model.args = _args(model_dir, output_dir, selection, **overrides)
    model.model_path = model_dir
    model.output_dir = output_dir
    model.config = _config(quantized=quantized)
    model.safetensors_files = frozenset({(model_dir / "model.safetensors").resolve()})
    model.processed_modules = set()
    model._planned_specs = {}
    model.output_tensor_bytes = 0
    model.converted_names = set()
    return model


def _fp8_pair(shape=(2, 128)):
    dense = torch.ones(shape, dtype=torch.bfloat16)
    weight = dense.to(torch.float8_e4m3fn)
    scale = torch.ones((1, 1), dtype=torch.float8_e8m0fnu)
    return weight, scale


def test_glm53_is_registered_without_transformers_glm5_next():
    register_llm_models()
    assert MODEL_REGISTRY.get("glm5_3") is GLM5_3


def test_glm53_source_validator_accepts_only_official_fp8_source(tmp_path):
    model_dir, output_dir = _write_model(
        tmp_path, {"other.weight": torch.ones(1, 1, dtype=torch.bfloat16)}
    )
    args = _args(
        model_dir,
        output_dir,
        DeploySelection((QuantLayerRule(DENSE, QuantSpec("int", 8, 8)),)),
    )
    assert GLM5_3.validate_deploy_source(args)
    (model_dir / "quant_model_description.json").write_text("{}", encoding="utf-8")
    assert not GLM5_3.validate_deploy_source(args)


def test_glm53_candidates_follow_dense_moe_attention_and_exclude_mtp():
    adapter = GLM5_3.__new__(GLM5_3)
    adapter.config = _config()
    names = {
        DENSE,
        LAYER.format(0) + ".self_attn.q_proj",
        FULL_QA,
        FULL_QB,
        FULL_KVA,
        FULL_O,
        EXPERT,
        SHARED,
        MTP,
        LAYER.format(3) + ".self_attn.indexer.wq_b",
    }
    assert adapter.get_deploy_candidates(names) == {
        DENSE,
        FULL_QA,
        FULL_QB,
        FULL_KVA,
        FULL_O,
        EXPERT,
        SHARED,
    }


@pytest.mark.parametrize(
    "config_name, quant_dtype",
    [
        ("glm5_3_w8a8_int.json", "int"),
        ("glm5_3_w8a8_mxfp8.json", "mxfp"),
    ],
)
def test_glm53_deploy_configs_match_attention_and_mtp_rules(config_name, quant_dtype):
    selection = load_deploy_selection(
        REPO_ROOT / "amct_pytorch/configs/deploy/glm/glm5_3" / config_name
    )
    expected_full_attn_layers = _config()["text_config"]["linear_attn_config"][
        "full_attn_layers"
    ]
    o_proj_layers = sorted(
        int(rule.pattern.split(".layers.")[1].split(".")[0])
        for rule in selection.quant_layers
        if rule.pattern.endswith("self_attn.o_proj")
    )
    assert o_proj_layers == expected_full_attn_layers
    assert all(
        rule.quant_spec == QuantSpec(quant_dtype, 8, 8)
        for rule in selection.quant_layers
    )
    assert LAYER.format(45) + ".*" in selection.ignore_layers


def test_glm53_rejects_hifp_deployment(tmp_path):
    assert GLM5_3.get_deploy_capabilities() == {
        "A3": frozenset({"int"}),
        "ascend950": frozenset({"int", "mxfp"}),
    }
    args = _args(
        tmp_path / "model",
        tmp_path / "output",
        DeploySelection((QuantLayerRule(DENSE, QuantSpec("hifp", 8, 8)),)),
    )
    with pytest.raises(NotImplementedError, match="does not support hifp"):
        GLM5_3.validate_deploy_args(args)


def test_glm53_selection_requires_fused_attention_and_complete_experts(tmp_path):
    selection = DeploySelection((QuantLayerRule(FULL_QA, QuantSpec("int", 8, 8)),))
    model = GLM5_3(_args(*_write_model(tmp_path, {"x": torch.ones(1)}), selection))
    with pytest.raises(ValueError, match="Fused modules"):
        model.validate_deploy_selection({FULL_QA})

    with pytest.raises(ValueError, match="all experts"):
        model.validate_deploy_selection({EXPERT})

    partial_expert = {
        LAYER.format(4) + f".mlp.experts.{expert}.{projection}"
        for expert in (0,)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    with pytest.raises(ValueError, match="all experts"):
        model.validate_deploy_selection(partial_expert)


def test_glm53_prepare_marks_layer_45_as_mtp_and_selects_modules(tmp_path):
    tensors = {
        DENSE + ".weight": torch.ones(2, 128, dtype=torch.bfloat16),
        DENSE_UP + ".weight": torch.ones(2, 128, dtype=torch.bfloat16),
        MTP + ".weight": torch.ones(2, 128, dtype=torch.bfloat16),
    }
    selection = DeploySelection(
        (
            QuantLayerRule(DENSE, QuantSpec("int", 8, 8)),
            QuantLayerRule(DENSE_UP, QuantSpec("int", 8, 8)),
        )
    )
    model = _bare_model(tmp_path, tensors, selection, quantized=False)
    plan = model.prepare_tensor_deploy_plan(
        {name: "model.safetensors" for name in tensors}
    )
    assert plan.selected_modules == {DENSE, DENSE_UP}
    assert plan.mtp == MtpSelectionReport(frozenset({MTP}), frozenset(), frozenset())


def test_glm53_dequantizes_cross_shard_fp8_scale_and_exports_int(tmp_path):
    weight, scale = _fp8_pair()
    other_weight, other_scale = _fp8_pair()
    selection = DeploySelection((QuantLayerRule(DENSE, QuantSpec("int", 8, 8)),))
    model = _bare_model(
        tmp_path,
        {
            DENSE + ".weight": weight,
            DENSE + ".weight_scale_inv": scale,
            OTHER_FP8 + ".weight": other_weight,
            OTHER_FP8 + ".weight_scale_inv": other_scale,
        },
        selection,
    )
    model_dir = model.model_path
    output_dir = model.output_dir
    selection = DeploySelection((QuantLayerRule(DENSE, QuantSpec("int", 8, 8)),))
    scale_file = model_dir / "scale.safetensors"
    save_file(
        {
            DENSE + ".weight_scale_inv": scale,
            OTHER_FP8 + ".weight_scale_inv": other_scale,
        },
        str(scale_file),
    )
    save_file(
        {
            DENSE + ".weight": weight,
            OTHER_FP8 + ".weight": other_weight,
        },
        str(model_dir / "model.safetensors"),
    )
    model.safetensors_files = frozenset({model_dir / "model.safetensors", scale_file})
    model._planned_specs = {DENSE: QuantSpec("int", 8, 8)}
    loaded_files = {}
    routing = model.convert_tensorwise_shard(
        "model.safetensors",
        model.model_path,
        {
            DENSE + ".weight": "model.safetensors",
            DENSE + ".weight_scale_inv": "scale.safetensors",
            OTHER_FP8 + ".weight": "model.safetensors",
            OTHER_FP8 + ".weight_scale_inv": "scale.safetensors",
        },
        {DENSE: QuantSpec("int", 8, 8)},
        loaded_files,
    )
    assert set(routing) == {
        DENSE + ".weight",
        DENSE + ".weight_scale",
        DENSE + ".weight_offset",
        OTHER_FP8 + ".weight",
    }
    assert set(loaded_files) == {"scale.safetensors"}
    with safe_open(str(output_dir / "model.safetensors"), framework="pt") as shard:
        assert shard.get_slice(DENSE + ".weight").get_dtype() == "I8"
        assert shard.get_slice(DENSE + ".weight_scale").get_dtype() == "BF16"
        assert shard.get_slice(OTHER_FP8 + ".weight").get_dtype() == "BF16"


@pytest.mark.parametrize(
    "spec, weight_dtype, expected_dtype, expected_type",
    [
        (QuantSpec("mxfp", 8, 8), torch.float8_e4m3fn, "F8_E4M3", "W8A8_MXFP8"),
        (QuantSpec("mxfp", 4, 8), torch.uint8, "U8", "W4A8_MXFP"),
    ],
)
def test_glm53_exports_mxfp_w8a8_and_w4a8(
    tmp_path, spec, weight_dtype, expected_dtype, expected_type
):
    dense = torch.ones((2, 128), dtype=torch.bfloat16)
    model = _bare_model(
        tmp_path,
        {DENSE + ".weight": dense},
        DeploySelection((QuantLayerRule(DENSE, spec),)),
        quantized=False,
    )
    model._planned_specs = {DENSE: spec}
    model.convert_tensorwise_shard(
        "model.safetensors",
        model.model_path,
        {DENSE + ".weight": "model.safetensors"},
        {DENSE: spec},
        {},
    )
    with safe_open(
        str(model.output_dir / "model.safetensors"), framework="pt"
    ) as shard:
        assert shard.get_slice(DENSE + ".weight").get_dtype() == expected_dtype
        assert shard.get_slice(DENSE + ".weight_scale").get_dtype() == "U8"
        assert DENSE + ".weight_offset" not in set(shard.keys())
    plan = SimpleNamespace(
        selected_modules=frozenset({DENSE}),
        module_quant_specs={DENSE: spec},
    )
    description = make_description_from_plan(set(model.converted_names), plan)
    assert description[DENSE + ".weight"] == expected_type


def test_glm53_refresh_config_only_removes_source_quantization(tmp_path):
    original = _config()
    original["text_config"]["quantization_config"] = {
        "quant_method": "fp8",
        "fmt": "e4m3",
    }
    model = _bare_model(
        tmp_path,
        {DENSE + ".weight": torch.ones(2, 128)},
        DeploySelection((QuantLayerRule(DENSE, QuantSpec("int", 8, 8)),)),
    )
    model.config = copy.deepcopy(original)
    (model.model_path / "config.json").write_text(
        json.dumps(original), encoding="utf-8"
    )
    model.refresh_config([], tensor_plan=None)
    result = json.loads((model.output_dir / "config.json").read_text(encoding="utf-8"))
    expected = copy.deepcopy(original)
    expected.pop("quantization_config")
    expected["text_config"].pop("quantization_config")
    assert result == expected
    assert (
        json.loads((model.model_path / "config.json").read_text(encoding="utf-8"))
        == original
    )
