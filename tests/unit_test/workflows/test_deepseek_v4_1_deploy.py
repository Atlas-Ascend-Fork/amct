# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deployment-only DeepSeek V4.1 tests using real FP8 safetensors on CPU."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from amct_pytorch.common.models import MODEL_REGISTRY
from amct_pytorch.common.models.llm.common.deploy_selection import (
    validate_tensor_deploy_args,
)
from amct_pytorch.workflows.llm_deploy import LlmDeployWorkflow


@pytest.fixture
def checkpoint(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    tensors = {
        "layers.0.attn.wq.weight": torch.arange(64 * 32)
        .reshape(64, 32)
        .remainder(16)
        .to(torch.float8_e4m3fn),
        "layers.0.attn.wq.scale": torch.tensor([[126], [129]], dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        ),
        "layers.0.ffn.experts.0.w1.weight": torch.arange(32, dtype=torch.uint8).reshape(
            4, 8
        ),
        "layers.0.ffn.experts.0.w1.scale": torch.full(
            (4, 1), 127, dtype=torch.uint8
        ).view(torch.float8_e8m0fnu),
        "engram.weight": torch.ones(32, 32).to(torch.float8_e4m3fn),
        "engram.scale": torch.full((32, 1), 127, dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        ),
        "head.weight": torch.ones(4, 4, dtype=torch.bfloat16),
    }
    save_file(tensors, source / "model.safetensors")
    config = {
        "model_type": "deepseek_v41",
        "quantization_config": {"quant_method": "fp8", "weight_block_size": [32, 32]},
        "text_config": {
            "num_hidden_layers": 1,
            "kv_source_layer_ids": [0],
            "index_source_layer_ids": [0],
        },
        "vision_config": {"hidden_size": 64},
    }
    (source / "config.json").write_text(json.dumps(config))
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"quant_layers": ["layers.*"]}))
    args = SimpleNamespace(
        model_name="deepseek_v4_1",
        seed=42,
        model=str(source),
        output_dir=str(tmp_path / "output"),
        granularity="tensor",
        deploy_format="legacy",
        deploy_platform="ascend950",
        quant_dtype="mxfp",
        quant_layers_config=str(selection),
        bit_config=None,
        quant_target=[],
    )
    return args, tensors, config


def adapter(args):
    # Import registers the lightweight adapter without building PTQ models.
    from amct_pytorch.common.models.llm.deepseek.deepseek_v4_1 import (
        deepseekv4_1,  # noqa: F401
    )

    return MODEL_REGISTRY.get("deepseek_v4_1")(args)


def run_deploy(args, monkeypatch):
    workflow = LlmDeployWorkflow(args)
    monkeypatch.setattr(workflow, "_register_components", lambda: adapter(args))
    return workflow.run()


def prepare_a3_source(args):
    source = Path(args.model) / "model.safetensors"
    tensors = load_file(str(source))
    tensors["layers.0.ffn.experts.0.w1.weight"] = torch.arange(
        64, dtype=torch.int8
    ).reshape(4, 16)
    save_file(tensors, str(source))


def test_950_arguments_accept_mxfp_without_bit_config(checkpoint):
    args, _, _ = checkpoint
    validate_tensor_deploy_args(args)
    assert args.deploy_selection.quant_layers == ("layers.*",)


def test_legacy_950_preserves_weights_and_expands_scale(checkpoint, monkeypatch):
    args, original, config = checkpoint
    result = run_deploy(args, monkeypatch)
    converted = load_file(str(Path(args.output_dir) / "model.safetensors"))
    assert converted.keys() == original.keys()
    for name, value in original.items():
        if name == "layers.0.attn.wq.scale":
            expected = torch.tensor([126] * 32 + [129] * 32, dtype=torch.uint8).reshape(
                64, 1
            )
            assert converted[name].dtype == torch.float8_e8m0fnu
            assert torch.equal(converted[name].view(torch.uint8), expected)
        else:
            assert converted[name].dtype == value.dtype
            assert torch.equal(
                converted[name].view(torch.uint8), value.view(torch.uint8)
            )
    output_config = json.loads((Path(args.output_dir) / "config.json").read_text())
    assert list(output_config) == [
        key for key in config if key != "quantization_config"
    ] + ["quantization_config"]
    assert output_config["text_config"] == config["text_config"]
    assert output_config["vision_config"] == config["vision_config"]
    quant = output_config["quantization_config"]
    assert quant["quant_method"] == "compressed-tensors"
    assert quant["weight_block_size"] == [1, 32]
    assert quant["config_groups"]["group_1"]["weights"]["num_bits"] == 4
    assert (
        quant["config_groups"]["group_0"]["input_activations"]["observer"]
        == "memoryless"
    )
    assert "layers.0.attn.compressor.wkv" in quant["ignore"]
    index = json.loads(Path(result["index_path"]).read_text())
    assert set(index["weight_map"]) == set(original)
    assert result["num_output_files"] == 1


def test_missing_fp8_selection_fails_before_output(checkpoint, monkeypatch):
    args, _, _ = checkpoint
    Path(args.quant_layers_config).write_text(
        json.dumps({"quant_layers": ["layers.*.ffn.*"]})
    )
    with pytest.raises(ValueError, match="FP8.*not selected"):
        run_deploy(args, monkeypatch)
    assert not Path(args.output_dir).exists()


@pytest.mark.parametrize(
    "platform,dtype,error",
    [
        (None, "mxfp", ValueError),
        ("A3", "mxfp", ValueError),
        ("ascend950", "int", ValueError),
    ],
)
def test_platform_contract(checkpoint, platform, dtype, error):
    args, _, _ = checkpoint
    args.deploy_platform, args.quant_dtype = platform, dtype
    with pytest.raises(error):
        validate_tensor_deploy_args(args)


def test_a3_ascend_contract_is_accepted(checkpoint):
    args, _, _ = checkpoint
    args.deploy_format = "ascend"
    args.deploy_platform, args.quant_dtype = "A3", "int"
    validate_tensor_deploy_args(args)


def test_a3_converts_mxfp8_and_mxfp4_before_int8_export(checkpoint):
    args, original, config = checkpoint
    args.deploy_format = "ascend"
    args.deploy_platform, args.quant_dtype = "A3", "int"
    prepare_a3_source(args)
    output = Path(args.output_dir)
    output.mkdir()
    model = adapter(args)
    weight_map = {name: "model.safetensors" for name in original}
    quant_layers = {
        "layers.0.attn.wq": 8,
        "layers.0.ffn.experts.0.w1": 8,
    }

    exported = model.convert_tensorwise_shard(
        "model.safetensors",
        Path(args.model),
        weight_map,
        quant_layers,
        {},
    )
    converted = load_file(str(output / "model.safetensors"))

    assert converted["layers.0.attn.wq.weight"].dtype == torch.int8
    assert converted["layers.0.attn.wq.weight_scale"].dtype == torch.bfloat16
    assert converted["layers.0.attn.wq.weight_offset"].dtype == torch.bfloat16
    assert converted["layers.0.ffn.experts.0.w1.weight"].dtype == torch.int8
    assert converted["layers.0.ffn.experts.0.w1.weight"].shape == (4, 32)
    assert converted["head.weight"].dtype == torch.bfloat16
    assert converted["engram.weight"].dtype == torch.bfloat16
    assert all(not name.endswith(".scale") for name in converted)
    assert set(exported) == set(converted)
    assert "quantization_config" in config


def test_a3_refreshes_config_and_description(checkpoint):
    args, _, _ = checkpoint
    args.deploy_format = "ascend"
    args.deploy_platform, args.quant_dtype = "A3", "int"
    prepare_a3_source(args)
    output = Path(args.output_dir)
    output.mkdir()
    model = adapter(args)
    model.refresh_config([], tensor_plan=None)
    refreshed = json.loads((output / "config.json").read_text())
    assert "quantization_config" not in refreshed
    assert refreshed["vision_config"] == {"hidden_size": 64}

    selected = frozenset(
        {"layers.0.attn.wq", "layers.0.ffn.experts.0.w1"}
    )
    output_names = {
        "layers.0.attn.wq.weight",
        "layers.0.attn.wq.weight_scale",
        "layers.0.attn.wq.weight_offset",
        "layers.0.ffn.experts.0.w1.weight",
        "layers.0.ffn.experts.0.w1.weight_scale",
        "layers.0.ffn.experts.0.w1.weight_offset",
        "head.weight",
    }
    description = model.make_ascend_description(output_names, selected)
    assert description["layers.0.attn.wq.weight"] == "W8A8_DYNAMIC"
    assert description["layers.0.attn.wq.weight_scale"] == "W8A8_DYNAMIC"
    assert description["head.weight"] == "FLOAT"


def test_a3_workflow_exports_mixed_checkpoint(checkpoint, monkeypatch):
    args, _, _ = checkpoint
    args.deploy_format = "ascend"
    args.deploy_platform, args.quant_dtype = "A3", "int"
    prepare_a3_source(args)
    result = run_deploy(args, monkeypatch)
    output = Path(args.output_dir)
    description = json.loads(
        (output / "quant_model_description.json").read_text()
    )
    config = json.loads((output / "config.json").read_text())
    tensors = load_file(str(output / "model.safetensors"))

    assert result["description_path"].endswith("quant_model_description.json")
    assert "quantization_config" not in config
    assert description["layers.0.attn.wq.weight"] == "W8A8_DYNAMIC"
    assert tensors["layers.0.ffn.experts.0.w1.weight"].dtype == torch.int8


@pytest.mark.parametrize("alias", ["short", "ids", "both"])
def test_config_aliases_preserve_nested_config(checkpoint, monkeypatch, alias):
    args, _, config = checkpoint
    fields = config["text_config"]
    if alias in ("short", "both"):
        for stem in ("kv", "index"):
            fields[f"{stem}_source_layers"] = fields[f"{stem}_source_layer_ids"]
            if alias == "short":
                del fields[f"{stem}_source_layer_ids"]
    (Path(args.model) / "config.json").write_text(json.dumps(config))
    run_deploy(args, monkeypatch)
    result = json.loads((Path(args.output_dir) / "config.json").read_text())
    assert result["text_config"] == fields
    assert result["quantization_config"]["ignore"] == [
        "layers.0.attn.indexer.weights_proj",
        "layers.0.attn.indexer.compressor.wgate",
        "layers.0.attn.indexer.compressor.wkv",
        "layers.0.attn.compressor.wgate",
        "layers.0.attn.compressor.wkv",
        "head",
    ]


def test_conflicting_aliases_fail_before_output(checkpoint, monkeypatch):
    args, _, config = checkpoint
    config["text_config"]["kv_source_layers"] = [1]
    (Path(args.model) / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Conflicting config fields"):
        run_deploy(args, monkeypatch)
    assert not Path(args.output_dir).exists()


def test_ignore_cannot_exclude_required_fp8(checkpoint, monkeypatch):
    args, _, _ = checkpoint
    Path(args.quant_layers_config).write_text(
        json.dumps(
            {
                "quant_layers": ["layers.*"],
                "ignore_layers": ["layers.*.attn.*"],
            }
        )
    )
    with pytest.raises(ValueError, match="FP8.*not selected"):
        run_deploy(args, monkeypatch)
    assert not Path(args.output_dir).exists()


def test_cross_shard_pairs_are_copied_without_expansion(checkpoint, monkeypatch):
    args, tensors, _ = checkpoint
    source = Path(args.model)
    scale_name = "layers.0.attn.wq.scale"
    scale = tensors.pop(scale_name)
    save_file(tensors, source / "model.safetensors")
    save_file({scale_name: scale}, source / "scale.safetensors")
    weight_map = {name: "model.safetensors" for name in tensors}
    weight_map[scale_name] = "scale.safetensors"
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    result = run_deploy(args, monkeypatch)
    actual = load_file(str(Path(args.output_dir) / "scale.safetensors"))[scale_name]
    assert actual.shape == scale.shape
    assert torch.equal(actual.view(torch.uint8), scale.view(torch.uint8))
    assert (
        json.loads(Path(result["index_path"]).read_text())["weight_map"] == weight_map
    )


def test_custom_conversion_and_config_never_fall_back(checkpoint, monkeypatch):
    args, _, _ = checkpoint

    def forbidden(*args, **kwargs):
        pytest.fail(
            "DeepSeek V4.1 must not call the generic converter/config refresher"
        )

    monkeypatch.setattr(LlmDeployWorkflow, "_convert_tensorwise_shard", forbidden)
    monkeypatch.setattr(LlmDeployWorkflow, "_refresh_config", forbidden)
    run_deploy(args, monkeypatch)


def test_custom_conversion_failure_does_not_fall_back(checkpoint, monkeypatch):
    args, _, _ = checkpoint
    model = adapter(args)

    def fail(*args, **kwargs):
        raise RuntimeError("custom conversion failed")

    monkeypatch.setattr(type(model), "convert_tensorwise_shard", fail)
    with pytest.raises(RuntimeError, match="custom conversion failed"):
        run_deploy(args, monkeypatch)


def test_950_description_remains_unimplemented(checkpoint):
    args, _, _ = checkpoint
    model = adapter(args)
    args.deploy_platform, args.quant_dtype = "ascend950", "mxfp"
    with pytest.raises(NotImplementedError, match="ascend description"):
        model.make_ascend_description(set(), set())


def test_cli_accepts_platform_and_selection_without_ptq(checkpoint, monkeypatch):
    from amct_pytorch.cli.llm.args import parser_gen

    args, _, _ = checkpoint
    monkeypatch.setattr(
        "sys.argv",
        [
            "deploy",
            "--model_name",
            "deepseek_v4_1",
            "--model",
            args.model,
            "--output_dir",
            args.output_dir,
            "--granularity",
            "tensor",
            "--deploy_platform",
            "ascend950",
            "--quant_dtype",
            "mxfp",
            "--quant_layers_config",
            args.quant_layers_config,
        ],
    )
    parsed = parser_gen("deploy")
    assert parsed.deploy_platform == "ascend950"
    assert parsed.deploy_selection.quant_layers == ("layers.*",)
