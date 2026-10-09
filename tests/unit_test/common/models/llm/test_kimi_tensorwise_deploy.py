# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from amct_pytorch.common.models.llm.kimi.kimi_k3.kimi_k3 import KimiK3

PREFIX = "language_model.model.layers.0."
PACKED = PREFIX + "block_sparse_moe.experts.0.w1.weight_packed"
SCALE = PACKED.replace("weight_packed", "weight_scale")


@pytest.mark.parametrize(
    "targets", [[], ["moe"], ["attn-linear"], ["moe", "attn-linear"]]
)
def test_ignore_categories_golden(tmp_path, targets):
    prefix = "language_model.model.layers."
    always = [
        prefix + "1.self_attn." + projection
        for projection in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "f_a_proj",
            "f_b_proj",
            "b_proj",
            "g_proj",
            "g_a_proj",
            "g_b_proj",
            "q_a_proj",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "kv_b_proj",
        )
    ]
    always += [PREFIX + "self_attn.kv_b_proj"]
    always += [
        prefix + "1." + group + "." + projection
        for group in ("mlp", "block_sparse_moe.shared_experts")
        for projection in ("gate_proj", "up_proj", "down_proj")
    ]
    always += [
        prefix + "1." + projection
        for projection in ("self_attention_res_proj", "mlp_res_proj")
    ]
    always += [
        "language_model.model.output_attn_res_proj",
        "language_model.lm_head",
        "mm_projector.proj.0",
        "vision_tower.encoder.blocks.0.wqkv",
        "vision_tower.encoder.blocks.0.wo",
        "vision_tower.encoder.blocks.0.mlp.fc0",
        "vision_tower.encoder.blocks.0.mlp.fc1",
    ]
    routed = prefix + "1.block_sparse_moe.experts.0.w1"
    latent = prefix + "1.block_sparse_moe.routed_expert_up_proj"
    mla = prefix + "0.self_attn.q_a_proj"
    unrelated = [
        prefix + "1.input_layernorm",
        "language_model.model.embed_tokens",
        "auxiliary",
    ]
    mapping = {
        name + ".weight": "a.safetensors" for name in [*always, latent, mla, *unrelated]
    }
    mapping[routed + ".weight_packed"] = "a.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": mapping})
    )
    model = adapter(tmp_path)
    model.quant_target = targets
    expected = set(always)
    if targets and "moe" not in targets:
        expected.update([routed, latent])
    if "attn-linear" not in targets:
        expected.add(mla)
    assert model.generate_tensorwise_ignore_layers() == sorted(
        name.removeprefix("language_model.") for name in expected
    )


def adapter(tmp_path, dtype="mxfp", config=None):
    output = tmp_path.with_name(tmp_path.name + "_out")
    output.mkdir()
    if config is None:
        config = {
            "text_config": {
                "num_hidden_layers": 3,
                "num_experts": 2,
                "linear_attn_config": {"full_attn_layers": [1]},
            }
        }
    (tmp_path / "config.json").write_text(json.dumps(config))
    model = KimiK3(
        SimpleNamespace(
            model=str(tmp_path),
            output_dir=str(output),
            granularity="tensor",
            quant_dtype=dtype,
            quant_target=["moe", "attn-linear"],
            deploy_format="legacy",
        )
    )
    (output / "config.json").write_text(json.dumps(config))
    return model


def test_cross_shard_passthrough_and_auxiliary(tmp_path):
    weight = torch.full((2, 16), 0x21, dtype=torch.uint8)
    scale = torch.full((2, 1), 127, dtype=torch.uint8)
    save_file(
        {PACKED: weight, "auxiliary": torch.tensor([1], dtype=torch.uint8)},
        str(tmp_path / "a.safetensors"),
    )
    save_file({SCALE: scale}, str(tmp_path / "b.safetensors"))
    mapping = {
        PACKED: "a.safetensors",
        "auxiliary": "a.safetensors",
        SCALE: "b.safetensors",
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": mapping})
    )
    model = adapter(tmp_path)
    routes = model.convert_tensorwise_shard(
        "a.safetensors", tmp_path, mapping, model.generate_tensorwise_quant_layers(), {}
    )
    result = load_file(
        str(tmp_path.with_name(tmp_path.name + "_out") / "a.safetensors")
    )
    assert torch.equal(result[PACKED], weight)
    assert torch.equal(result[SCALE], scale)
    assert "auxiliary" in routes


def test_bf16_decodes_uint8_packed_and_clears_nested_config(tmp_path):
    save_file(
        {
            PACKED: torch.full((2, 16), 0x21, dtype=torch.uint8),
            SCALE: torch.full((2, 1), 127, dtype=torch.uint8),
        },
        str(tmp_path / "a.safetensors"),
    )
    mapping = {PACKED: "a.safetensors", SCALE: "a.safetensors"}
    model = adapter(tmp_path, None)
    model.convert_tensorwise_shard("a.safetensors", tmp_path, mapping, {}, {})
    result = load_file(
        str(tmp_path.with_name(tmp_path.name + "_out") / "a.safetensors")
    )
    weight = result[PACKED.replace("weight_packed", "weight")]
    assert weight.dtype == torch.bfloat16
    assert weight.shape == (2, 32)
    assert torch.equal(
        weight[0, :4], torch.tensor([0.5, 1, 0.5, 1], dtype=torch.bfloat16)
    )
    assert SCALE not in result
    output_config = tmp_path.with_name(tmp_path.name + "_out") / "config.json"
    previous = json.loads(output_config.read_text())
    previous["quantization_config"] = {"old": True}
    previous["text_config"]["quantization_config"] = {"old": True}
    output_config.write_text(json.dumps(previous))
    model.refresh_config([])
    config = json.loads(
        (tmp_path.with_name(tmp_path.name + "_out") / "config.json").read_text()
    )
    assert "quantization_config" not in config
    assert "quantization_config" not in config["text_config"]


def test_missing_scale_is_error(tmp_path):
    save_file(
        {PACKED: torch.ones((2, 16), dtype=torch.uint8)},
        str(tmp_path / "a.safetensors"),
    )
    model = adapter(tmp_path)
    with pytest.raises(ValueError, match="scale"):
        model.convert_tensorwise_shard(
            "a.safetensors",
            tmp_path,
            {PACKED: "a.safetensors"},
            {PACKED.rsplit(".", 1)[0]: 4},
            {},
        )


def test_selection_excludes_shared_kda_and_nonfull_mla(tmp_path):
    names = [
        PACKED,
        PREFIX + "block_sparse_moe.shared_experts.up_proj.weight",
        PREFIX + "self_attn.q_a_proj.weight",
        PREFIX + "self_attn.q_proj.weight",
        "language_model.model.layers.1.self_attn.q_a_proj.weight",
        PREFIX + "block_sparse_moe.routed_expert_up_proj.weight",
    ]
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(names, "a.safetensors")})
    )
    model = adapter(tmp_path)
    selected = model.generate_tensorwise_quant_layers()
    assert set(selected.values()) == {4, 8}
    assert len(selected) == 3
    ignores = model.generate_tensorwise_ignore_layers()
    assert all(not name.startswith("language_model.") for name in ignores)
    assert "model.layers.0.self_attn.q_proj" in ignores


@pytest.mark.parametrize(
    "targets", [[], ["moe"], ["attn-linear"], ["moe", "attn-linear"]]
)
def test_tensorwise_selection_and_ignore_golden(tmp_path, targets):
    prefix = "language_model.model.layers."
    routed = [
        prefix + "1.block_sparse_moe.experts.0.w1",
        prefix + "1.block_sparse_moe.experts.1.w2",
    ]
    latent = prefix + "1.block_sparse_moe.routed_expert_down_proj"
    mla = prefix + "0.self_attn.q_a_proj"
    unselected = [
        prefix + "1.self_attn.q_b_proj",
        prefix + "1.self_attn.q_proj",
        prefix + "1.block_sparse_moe.shared_experts.up_proj",
    ]
    weight_map = {name + ".weight_packed": "a.safetensors" for name in routed}
    weight_map.update(
        {name + ".weight": "a.safetensors" for name in [latent, mla, *unselected]}
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    text = {
        "num_hidden_layers": 3,
        "num_experts": 2,
        "linear_attn_config": {"full_attn_layers": [1]},
    }
    model = adapter(tmp_path, config={"text_config": text})
    model.quant_target = targets
    expected = {}
    if not targets or "moe" in targets:
        expected.update(dict.fromkeys(routed, 4))
        expected[latent] = 8
    if "attn-linear" in targets:
        expected[mla] = 8
    assert model.generate_tensorwise_quant_layers() == expected
    assert model.generate_tensorwise_ignore_layers() == sorted(
        name.removeprefix("language_model.")
        for name in [*routed, latent, mla, *unselected]
        if name not in expected
    )
    assert model.bits_scheme()[1]["a_bits"] == 8


def test_mxfp8_requantization_and_nested_metadata(tmp_path):
    name = PREFIX + "self_attn.q_a_proj.weight"
    scale_name = name.replace(".weight", ".weight_scale")
    source = torch.ones((2, 32), dtype=torch.float8_e4m3fn)
    save_file(
        {name: source, scale_name: torch.full((2, 1), 127, dtype=torch.uint8)},
        str(tmp_path / "a.safetensors"),
    )
    model = adapter(tmp_path)
    mapping = {name: "a.safetensors", scale_name: "a.safetensors"}
    model.convert_tensorwise_shard(
        "a.safetensors", tmp_path, mapping, {name.removesuffix(".weight"): 8}, {}
    )
    result = load_file(
        str(tmp_path.with_name(tmp_path.name + "_out") / "a.safetensors")
    )
    assert result[name].dtype == torch.float8_e4m3fn
    assert result[scale_name].dtype == torch.uint8
    from amct_pytorch.quantization.dtypes.fp_impl import weight_dequant

    assert torch.equal(
        weight_dequant(result[name], result[scale_name], 32, is_mx=True),
        torch.ones((2, 32)),
    )
    model.refresh_config(["model.layers.0.self_attn.q_proj"])
    config = json.loads(
        (tmp_path.with_name(tmp_path.name + "_out") / "config.json").read_text()
    )
    assert config["quantization_config"] == config["text_config"]["quantization_config"]
    assert (
        config["quantization_config"]["config_groups"]["group_1"]["weights"]["num_bits"]
        == 4
    )


def test_malformed_scale_rejected_before_output(tmp_path):
    save_file(
        {
            PACKED: torch.ones((2, 16), dtype=torch.int8),
            SCALE: torch.ones((1, 1), dtype=torch.uint8),
        },
        str(tmp_path / "a.safetensors"),
    )
    model = adapter(tmp_path)
    with pytest.raises(ValueError, match="shape"):
        model.convert_tensorwise_shard(
            "a.safetensors",
            tmp_path,
            {PACKED: "a.safetensors", SCALE: "a.safetensors"},
            {},
            {},
        )
    assert not (tmp_path.with_name(tmp_path.name + "_out") / "a.safetensors").exists()


def test_duplicate_packed_and_plain_weight_rejected(tmp_path):
    save_file(
        {PACKED: torch.ones((2, 16), dtype=torch.uint8)},
        str(tmp_path / "a.safetensors"),
    )
    model = adapter(tmp_path)
    with pytest.raises(ValueError, match="Duplicate"):
        model.convert_tensorwise_shard(
            "a.safetensors",
            tmp_path,
            {
                PACKED: "a.safetensors",
                PACKED.replace("weight_packed", "weight"): "a.safetensors",
            },
            {},
            {},
        )


def test_attention_only_ignores_decoded_experts(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {PACKED: "a.safetensors"}})
    )
    model = adapter(tmp_path)
    model.quant_target = ["attn-linear"]
    assert (
        "model.layers.0.block_sparse_moe.experts.0.w1"
        in model.generate_tensorwise_ignore_layers()
    )


def test_actual_workflow_exports_index_and_nested_config(tmp_path):
    from amct_pytorch.workflows.llm_deploy import LlmDeployWorkflow

    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    config = {
        "text_config": {
            "num_hidden_layers": 1,
            "num_experts": 1,
            "linear_attn_config": {"full_attn_layers": [1]},
        }
    }
    (source / "config.json").write_text(json.dumps(config))
    shared = PREFIX + "block_sparse_moe.shared_experts.up_proj.weight"
    save_file(
        {
            PACKED: torch.full((2, 16), 0x21, dtype=torch.uint8),
            shared: torch.ones((2, 32), dtype=torch.bfloat16),
        },
        str(source / "a.safetensors"),
    )
    save_file(
        {SCALE: torch.full((2, 1), 127, dtype=torch.uint8)},
        str(source / "b.safetensors"),
    )
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    PACKED: "a.safetensors",
                    SCALE: "b.safetensors",
                    shared: "a.safetensors",
                }
            }
        )
    )
    args = SimpleNamespace(
        model=str(source),
        output_dir=str(output),
        model_name="kimi_k3",
        granularity="tensor",
        quant_dtype="mxfp",
        quant_target=["moe", "attn-linear"],
        deploy_format="legacy",
        seed=0,
        log_dir=str(tmp_path / "logs"),
    )
    LlmDeployWorkflow(args).run()
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert index["weight_map"][SCALE] == "a.safetensors"
    assert index["metadata"]["total_size"] > 0
    exported = json.loads((output / "config.json").read_text())
    assert (
        exported["quantization_config"]
        == exported["text_config"]["quantization_config"]
    )
    assert (
        shared.removeprefix("language_model.").removesuffix(".weight")
        in exported["quantization_config"]["ignore"]
    )
