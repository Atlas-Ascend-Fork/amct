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
"""Logic tests for LlmDeployWorkflow.

The full `_run_blockwise` requires a real safetensors-backed model dir; we
cover the file-IO and helper logic in isolation.
"""

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file

from amct_pytorch.workflows.llm_deploy import LlmDeployWorkflow

CONFIG_JSON = "config.json"
GRANULARITY_BLOCK = "block"
SAFETENSORS_INDEX_JSON = "model.safetensors.index.json"
LAYER_WEIGHT = "layer.weight"
MODEL_SAFETENSORS = "model.safetensors"
TMP_DEPLOY_OUT = "/tmp/deploy_out"
FAKE_MODEL = "/fake/model"
MODEL_NAME_QWEN3 = "qwen3"
REST_00000 = "rest_00000.safetensors"
TMP_FAKE = "/tmp/fake"

BIG = 'big'
METADATA_KEY = 'metadata'
MODEL_LAYERS_0_MLP_UP_PROJ_WEIGHT = 'model.layers.0.mlp.up_proj.weight'
QUANTIZATION_CONFIG = 'quantization_config'
KEY_SHARD1_SAFETENSORS = 'shard1.safetensors'
KEY_SUBDIR = 'subdir'
KEY_UNKNOWN_WEIGHT = 'unknown.weight'


def _make_pipeline_mock(num_layers=2, **overrides):
    """Create a SimpleNamespace pipeline mock with required deploy methods."""
    defaults = {
        "num_layers": num_layers,
        "cache_scheme": lambda: {
            "kv_cache_scheme": {"num_bits": 8, "type": "float"},
            "li_cache_scheme": {"type": "float", "num_bits": 8},
        },
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_workflow(
    model_path=FAKE_MODEL, output_dir=TMP_DEPLOY_OUT, quant_dtype="int8"
):
    workflow = LlmDeployWorkflow.__new__(LlmDeployWorkflow)
    args = SimpleNamespace(
        granularity=GRANULARITY_BLOCK,
        model_name=MODEL_NAME_QWEN3,
        model=model_path,
        quant_dtype=quant_dtype,
        output_dir=output_dir,
    )
    workflow.args = args
    workflow.granularity = args.granularity
    workflow.model_name = args.model_name
    workflow.model_path = args.model
    workflow.quant_dtype = args.quant_dtype
    workflow.output_dir = args.output_dir
    workflow.is_mx = quant_dtype.startswith("mx")
    workflow.is_int = quant_dtype.startswith("int")
    workflow.is_hif = quant_dtype.startswith("hif")
    workflow.pipeline = None
    return workflow


def _tensor_json_args(
    workflow, tmp_path, quant=("model.layers.0.self_attn.o_proj",), ignore=()
):
    from amct_pytorch.common.models.llm.common.deployment.deploy_selection import (
        DeploySelection,
        QuantLayerRule,
        QuantSpec,
    )
    from amct_pytorch.quantization.bit_policy import BitPolicy

    workflow.granularity = workflow.args.granularity = "tensor"
    workflow.args.model_name = "glm5_2"
    workflow.args.deploy_selection = DeploySelection(
        tuple(QuantLayerRule(name, QuantSpec("int", 8, 8)) for name in quant),
        ignore,
    )
    workflow.args.bit_policy = BitPolicy(
        {group: {"w_bits": 8, "a_bits": 8} for group in ("attn-linear", "mlp", "moe")}
    )
    config = {
        "quant_layers": [
            {
                "pattern": name,
                "quant_dtype": "int",
                "w_bits": 8,
                "a_bits": 8,
            }
            for name in quant
        ],
        "ignore_layers": list(ignore),
    }
    config_path = Path(tmp_path) / "quant_layers_config.json"
    config_path.write_text(json.dumps(config))
    workflow.args.quant_layers_config = str(config_path)


def test_workflow_initializes_plural_safetensors_files_cache():
    args = SimpleNamespace(
        granularity=GRANULARITY_BLOCK,
        quant_target=["mlp"],
        model_name=MODEL_NAME_QWEN3,
        model=FAKE_MODEL,
        quant_dtype="int8",
        output_dir=TMP_DEPLOY_OUT,
        seed=0,
    )

    workflow = LlmDeployWorkflow(args)

    assert workflow.safetensors_files is None


def test_tensor_workflow_init_allows_quant_dtype_none():
    args = SimpleNamespace(
        granularity="tensor",
        quant_target=None,
        model_name="qwen3_6",
        model=FAKE_MODEL,
        quant_dtype=None,
        output_dir=TMP_DEPLOY_OUT,
        seed=0,
    )

    workflow = LlmDeployWorkflow(args)

    assert workflow.quant_dtype == ""
    assert not workflow.is_mx
    assert not workflow.is_int
    assert not hasattr(workflow, "safetensors_file")


# ---- dtype flag derivation ----------------------------------------------


@pytest.mark.parametrize(
    "dtype,is_mx,is_int,is_hif",
    [
        ("int8", False, True, False),
        ("mxfp8", True, False, False),
        ("hifp8", False, False, True),
    ],
)
def test_quant_dtype_flags_set_correctly(dtype, is_mx, is_int, is_hif):
    wf = _make_workflow(quant_dtype=dtype)
    assert wf.is_mx is is_mx
    assert wf.is_int is is_int
    assert wf.is_hif is is_hif


# ---- _is_weight_file -----------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        (SAFETENSORS_INDEX_JSON, True),
        ("model-00001-of-00002.safetensors", True),
        (CONFIG_JSON, False),
        ("tokenizer.model", False),
        ("README.md", False),
    ],
)
def test_is_weight_file_recognizes_safetensors_artifacts(name, expected):
    assert LlmDeployWorkflow._is_weight_file(Path(name)) is expected


# ---- _copy_support_files -------------------------------------------------


def test_copy_support_files_copies_non_weight_files_only(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / CONFIG_JSON).write_text("{}")
    (src / "tokenizer.model").write_text("tok")
    (src / MODEL_SAFETENSORS).write_text(BIG)
    (src / SAFETENSORS_INDEX_JSON).write_text("{}")
    (src / ".hidden").write_text("skip")
    (src / KEY_SUBDIR).mkdir()
    (src / KEY_SUBDIR / "more.txt").write_text("x")

    wf = _make_workflow(model_path=str(src), output_dir=str(dst))
    wf._copy_support_files()

    assert (dst / CONFIG_JSON).exists()
    assert (dst / "tokenizer.model").exists()
    assert (dst / KEY_SUBDIR / "more.txt").exists()
    # Weight files and hidden dotfiles are skipped.
    assert not (dst / MODEL_SAFETENSORS).exists()
    assert not (dst / SAFETENSORS_INDEX_JSON).exists()
    assert not (dst / ".hidden").exists()


def test_copy_support_files_skips_existing_destinations(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / CONFIG_JSON).write_text("new")
    (dst / CONFIG_JSON).write_text("old")

    wf = _make_workflow(model_path=str(src), output_dir=str(dst))
    wf._copy_support_files()
    assert (dst / CONFIG_JSON).read_text() == "old"


# ---- _load_weight_index --------------------------------------------------


def test_load_weight_index_reads_json(tmp_path):
    index = {
        "weight_map": {"a.weight": KEY_SHARD1_SAFETENSORS},
        METADATA_KEY: {"total_size": 999},
    }
    save_file({"a.weight": torch.ones(1)}, str(tmp_path / KEY_SHARD1_SAFETENSORS))
    (tmp_path / SAFETENSORS_INDEX_JSON).write_text(json.dumps(index))
    wf = _make_workflow(model_path=str(tmp_path))
    assert wf._load_weight_index() == index


def test_load_weight_index_rejects_shard_outside_model_directory(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    save_file({"allowed": torch.ones(1)}, str(model_dir / "allowed.safetensors"))
    outside_path = tmp_path / "outside.safetensors"
    save_file({"a.weight": torch.ones(1)}, str(outside_path))
    index = {"weight_map": {"a.weight": "../outside.safetensors"}}
    (model_dir / SAFETENSORS_INDEX_JSON).write_text(json.dumps(index))

    wf = _make_workflow(model_path=str(model_dir))
    with pytest.raises(ValueError, match="not in the model directory allowlist"):
        wf._load_weight_index()


# ---- _write_safetensor_file / _write_block_file ------------------------


def test_write_safetensor_file_creates_file_atomically(tmp_path):
    wf = _make_workflow(output_dir=str(tmp_path))
    wf._write_safetensor_file("layer.safetensors", {"w": torch.zeros(2, 3)})
    out = tmp_path / "layer.safetensors"
    assert out.exists()
    # The .tmp file should have been renamed away.
    assert not list(tmp_path.glob(".*tmp"))


def test_write_safetensor_file_no_op_for_empty_tensors(tmp_path):
    wf = _make_workflow(output_dir=str(tmp_path))
    wf._write_safetensor_file("layer.safetensors", {})
    assert not (tmp_path / "layer.safetensors").exists()


def test_write_block_file_uses_zero_padded_filename_and_returns_routing(tmp_path):
    wf = _make_workflow(output_dir=str(tmp_path))
    # Pretend the model has 12 layers so width = max(3, 2) = 3.
    wf.pipeline = SimpleNamespace(num_layers=12)
    routing = wf._write_block_file(
        layer_idx=4, layer_tensors={"a.weight": torch.zeros(2, 3)}
    )
    assert routing == {"a.weight": "layer_004.safetensors"}
    assert (tmp_path / "layer_004.safetensors").exists()


# ---- _collect_replaced_original_weights ---------------------------------


def test_collect_replaced_original_weights_uses_routing_to_resolve_base():
    wf = _make_workflow()
    layer_tensors = {
        "model.layers.0.mlp.up_proj.qweight": "irrelevant",
        "model.layers.0.mlp.up_proj.weight_scale": "irrelevant",
    }
    tensor_routes = {
        "model.layers.0.mlp.up_proj.qweight": MODEL_LAYERS_0_MLP_UP_PROJ_WEIGHT,
        "model.layers.0.mlp.up_proj.weight_scale": MODEL_LAYERS_0_MLP_UP_PROJ_WEIGHT,
    }
    original = {MODEL_LAYERS_0_MLP_UP_PROJ_WEIGHT: KEY_SHARD1_SAFETENSORS}
    out = wf._collect_replaced_original_weights(layer_tensors, tensor_routes, original)
    assert out == {MODEL_LAYERS_0_MLP_UP_PROJ_WEIGHT}


def test_collect_replaced_original_weights_returns_empty_when_unrelated():
    wf = _make_workflow()
    layer_tensors = {KEY_UNKNOWN_WEIGHT: "x"}
    routes = {KEY_UNKNOWN_WEIGHT: KEY_UNKNOWN_WEIGHT}
    original = {"different.weight": "shard.safetensors"}
    assert (
        wf._collect_replaced_original_weights(layer_tensors, routes, original) == set()
    )


def test_refresh_weight_index_writes_metadata_and_total_size(tmp_path):
    wf = _make_workflow(output_dir=str(tmp_path))
    # Create two shard files so total_size sums them.
    (tmp_path / REST_00000).write_bytes(b"x" * 100)
    (tmp_path / "layer_000.safetensors").write_bytes(b"y" * 50)

    original = {METADATA_KEY: {"foo": "bar"}, "weight_map": {}}
    weight_map = {
        "alpha": REST_00000,
        "beta": "layer_000.safetensors",
    }
    index_path = wf._refresh_weight_index(original, weight_map)
    saved = json.loads(Path(index_path).read_text())
    assert saved[METADATA_KEY]["foo"] == "bar"
    assert saved[METADATA_KEY]["total_size"] == 150
    assert saved["weight_map"] == weight_map


# ---- _refresh_config -----------------------------------------------------


def test_refresh_config_attaches_quantization_block(tmp_path):
    wf = _make_workflow(output_dir=str(tmp_path))
    wf.pipeline = SimpleNamespace(cache_scheme=dict, bits_scheme=lambda: None)
    # Original config that the workflow reads in.
    (tmp_path / CONFIG_JSON).write_text(json.dumps({"hidden_size": 4096}))

    wf._refresh_config(quant_ignore_layers=["lm_head"])
    refreshed = json.loads((tmp_path / CONFIG_JSON).read_text())
    assert refreshed["hidden_size"] == 4096
    assert QUANTIZATION_CONFIG in refreshed
    assert refreshed[QUANTIZATION_CONFIG]["ignore"] == ["lm_head"]
    # int dtype path -> int-quantized format.
    assert refreshed[QUANTIZATION_CONFIG]["format"] == "int-quantized"


def test_refresh_config_uses_float_format_for_mx_dtype(tmp_path):
    wf = _make_workflow(output_dir=str(tmp_path), quant_dtype="mxfp8")
    wf.pipeline = SimpleNamespace(cache_scheme=dict, bits_scheme=lambda: None)
    (tmp_path / CONFIG_JSON).write_text("{}")
    wf._refresh_config(quant_ignore_layers=[])
    refreshed = json.loads((tmp_path / CONFIG_JSON).read_text())
    assert refreshed[QUANTIZATION_CONFIG]["format"] == "float-quantized"


# ---- _write_remaining_original_weights ----------------------------------


def test_write_remaining_original_weights_skips_replaced_and_shards_rest(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    # Source shard with three small tensors.
    save_file(
        {"a": torch.zeros(2), "b": torch.ones(3), "c": torch.full((4,), 2.0)},
        str(src / KEY_SHARD1_SAFETENSORS),
    )
    wf = _make_workflow(model_path=str(src), output_dir=str(dst))

    weight_map = {
        "a": KEY_SHARD1_SAFETENSORS,
        "b": KEY_SHARD1_SAFETENSORS,
        "c": KEY_SHARD1_SAFETENSORS,
    }
    replaced = {"b"}  # b is replaced by a quant routine; should be skipped here.

    updated = wf._write_remaining_original_weights(weight_map, replaced)
    # a + c land in rest_00000 (small tensors easily fit one shard).
    assert set(updated) == {"a", "c"}
    assert (dst / REST_00000).exists()
    assert all(file_name.startswith("rest_") for file_name in updated.values())


def test_write_remaining_weights_rejects_shard_outside_model_directory(tmp_path):
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "output"
    model_dir.mkdir()
    output_dir.mkdir()
    save_file({"allowed": torch.ones(1)}, str(model_dir / "allowed.safetensors"))
    save_file({"a": torch.ones(1)}, str(tmp_path / "outside.safetensors"))

    wf = _make_workflow(model_path=str(model_dir), output_dir=str(output_dir))
    with pytest.raises(ValueError, match="not in the model directory allowlist"):
        wf._write_remaining_original_weights({"a": "../outside.safetensors"}, set())


def test_convert_tensorwise_shard_rejects_source_outside_model_directory(tmp_path):
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "output" / "deploy"
    model_dir.mkdir()
    output_dir.mkdir(parents=True)
    save_file({"allowed": torch.ones(1)}, str(model_dir / "allowed.safetensors"))
    save_file({"a.weight": torch.ones(1)}, str(tmp_path / "outside.safetensors"))

    wf = _make_workflow(
        model_path=str(model_dir), output_dir=str(output_dir), quant_dtype="bf16"
    )
    wf.pipeline = SimpleNamespace(
        get_scale_name=lambda name: ("_scale", "unused_scale"),
        block_size=lambda weight: 32,
    )

    with pytest.raises(ValueError, match="not in the model directory allowlist"):
        wf._convert_tensorwise_shard(
            "../outside.safetensors",
            model_dir,
            {"a.weight": "../outside.safetensors"},
            {},
            {},
        )

    assert not (output_dir.parent / "outside.safetensors").exists()


def test_llm_deploy_run_blockwise(monkeypatch):
    wf = _make_workflow()
    wf.granularity = GRANULARITY_BLOCK
    wf.args._deploy_args_prepared = True

    def setup():
        return "sink"

    wf.setup = setup

    def _run_blockwise():
        return {"index_path": "/out", "num_output_files": 1}

    wf._run_blockwise = _run_blockwise
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.logger",
        importlib.import_module("types").SimpleNamespace(remove=lambda h: None),
    )
    result = wf.run()
    assert result["index_path"] == "/out"


def test_llm_deploy_setup(monkeypatch):
    wf = _make_workflow()
    called = {}
    monkeypatch.setattr(
        wf, "_register_components", lambda: called.update({"reg": True})
    )
    monkeypatch.setattr(wf, "_build_pipeline", lambda: called.update({"pipe": True}))
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.setup_run_logging",
        lambda log_dir, name: ("sink", None),
    )
    monkeypatch.setattr("os.makedirs", lambda p, exist_ok: None)
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.ensure_log_dir", lambda d: None
    )
    wf.setup()
    assert called.get("reg") is True
    assert called.get("pipe") is True


# ---- Task 15: Additional deploy coverage ---------------------------------


def _make_deploy_workflow(**overrides):
    defaults = {
        "model": "/tmp/fake",
        "model_name": MODEL_NAME_QWEN3,
        "quant_dtype": "int4",
        "granularity": GRANULARITY_BLOCK,
        "output_dir": "/tmp/fake",
    }
    defaults.update(overrides)
    args = SimpleNamespace(**defaults)
    wf = LlmDeployWorkflow.__new__(LlmDeployWorkflow)
    wf.args = args
    wf.granularity = args.granularity
    wf.model_name = args.model_name
    wf.model_path = args.model
    wf.quant_dtype = args.quant_dtype
    wf.output_dir = args.output_dir
    wf.pipeline = None
    wf.is_mx = wf.quant_dtype.startswith("mx")
    wf.is_int = wf.quant_dtype.startswith("int")
    wf.is_hif = wf.quant_dtype.startswith("hif")
    return wf


def test_deploy_is_weight_file_safetensors():
    assert LlmDeployWorkflow._is_weight_file(Path(MODEL_SAFETENSORS)) is True
    assert LlmDeployWorkflow._is_weight_file(Path("layer_0.safetensors")) is True
    assert LlmDeployWorkflow._is_weight_file(Path(SAFETENSORS_INDEX_JSON)) is True
    assert LlmDeployWorkflow._is_weight_file(Path(CONFIG_JSON)) is False


def test_deploy_init_dtype_flags():
    wf_int = _make_deploy_workflow(quant_dtype="int4")
    assert wf_int.is_int is True
    assert wf_int.is_mx is False
    assert wf_int.is_hif is False

    wf_mx = _make_deploy_workflow(quant_dtype="mxfp4")
    assert wf_mx.is_mx is True
    assert wf_mx.is_int is False

    wf_hif = _make_deploy_workflow(quant_dtype="hifloat8")
    assert wf_hif.is_hif is True


def test_deploy_setup_creates_output_dir_and_registers(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.register_llm_models", lambda: None
    )
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.register_dtype", lambda: None
    )
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.register_algorithms", lambda: None
    )
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.ensure_log_dir", lambda d: None
    )
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.setup_run_logging",
        lambda log_dir, name: ("sink", None),
    )
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.MODEL_REGISTRY",
        SimpleNamespace(get=lambda k: type("FM", (), {"__init__": lambda s, a: None})),
    )
    out = tmp_path / "deploy_out"
    wf = _make_deploy_workflow(output_dir=str(out))
    wf.setup()
    assert out.exists()
    assert wf.pipeline is not None


def test_tensor_setup_validates_source_after_adapter_normalization(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.mkdir()
    (source / CONFIG_JSON).write_text(
        json.dumps({"quantization_config": {"quant_method": "fp8"}})
    )
    output = tmp_path / "output"
    args = SimpleNamespace(
        model=str(source),
        model_name="fake_deploy_model",
        quant_dtype="",
        granularity="tensor",
        output_dir=str(output),
    )
    workflow = LlmDeployWorkflow.__new__(LlmDeployWorkflow)
    workflow.args = args
    workflow.granularity = "tensor"
    workflow.model_name = args.model_name
    workflow.model_path = args.model
    workflow.output_dir = args.output_dir
    workflow.quant_dtype = ""
    workflow.is_mx = workflow.is_int = workflow.is_hif = False
    workflow.pipeline = None
    seen = []

    class Adapter:
        def __init__(self, adapter_args):
            adapter_args.quant_dtype = "int"

        @classmethod
        def validate_deploy_source(cls, adapter_args):
            seen.append(adapter_args.quant_dtype)
            return adapter_args.quant_dtype == "int"

    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.MODEL_REGISTRY",
        {args.model_name: Adapter},
    )
    monkeypatch.setattr(workflow, "_register_components", lambda: None)
    monkeypatch.setattr(workflow, "_build_pipeline", lambda: Adapter(args))
    monkeypatch.setattr(workflow, "_prepare_tensor_deploy_plan", lambda: None)
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.ensure_log_dir", lambda _: None
    )
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.setup_run_logging",
        lambda *_: (1, None),
    )

    workflow.setup()

    assert seen == ["int"]
    assert workflow.quant_dtype == "int"
    assert output.exists()


def test_deploy_run_unsupported_granularity(monkeypatch):
    monkeypatch.setattr("amct_pytorch.workflows.llm_deploy.logger", MagicMock())
    wf = _make_deploy_workflow(granularity="model")
    wf.args._deploy_args_prepared = True

    def setup():
        return "fake_sink"

    wf.setup = setup
    with pytest.raises(ValueError, match="Unsupported granularity"):
        wf.run()


# ---- _run_blockwise (mocked helpers) -------------------------------------


def test_deploy_run_blockwise_mocked_loop(monkeypatch, tmp_path):
    def _mock_export_block_deploy(
        pipeline, layer_idx, quant_ignore_layers, *, linear_bits=None
    ):
        linear_bits[f"model.layers.{layer_idx}.linear"] = (16, 8)
        return (
            {LAYER_WEIGHT: torch.zeros(2, 3)},
            {LAYER_WEIGHT: LAYER_WEIGHT},
        )

    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.export_block_deploy",
        _mock_export_block_deploy,
    )
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.logger",
        MagicMock(),
    )
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.tqdm",
        lambda iterable, desc="": iterable,
    )

    wf = _make_workflow(output_dir=str(tmp_path))
    wf.pipeline = _make_pipeline_mock(num_layers=2)
    wf._copy_support_files = MagicMock()
    wf._load_weight_index = MagicMock(
        return_value={
            "weight_map": {
                LAYER_WEIGHT: KEY_SHARD1_SAFETENSORS,
            }
        }
    )
    wf._write_block_file = MagicMock(
        return_value={LAYER_WEIGHT: "layer_000.safetensors"}
    )
    wf._collect_replaced_original_weights = MagicMock(return_value={LAYER_WEIGHT})
    wf._write_remaining_original_weights = MagicMock(
        return_value={
            "other.weight": REST_00000,
        }
    )
    wf._refresh_weight_index = MagicMock(
        return_value=str(tmp_path / SAFETENSORS_INDEX_JSON)
    )
    wf._refresh_config = MagicMock()

    result = wf._run_blockwise()
    assert "index_path" in result
    assert "num_output_files" in result
    assert wf._write_block_file.call_count == 2
    wf._refresh_config.assert_called_once_with(
        [],
        linear_bits={
            "model.layers.0.linear": (16, 8),
            "model.layers.1.linear": (16, 8),
        },
    )


# ---- __init__ via actual constructor -------------------------------------


def test_deploy_init_sets_all_attrs():
    args = SimpleNamespace(
        granularity=GRANULARITY_BLOCK,
        quant_target=["mlp"],
        model_name=MODEL_NAME_QWEN3,
        model=FAKE_MODEL,
        quant_dtype="int8",
        output_dir=TMP_DEPLOY_OUT,
        seed=0,
    )
    wf = LlmDeployWorkflow(args)
    assert wf.args is args
    assert wf.granularity == GRANULARITY_BLOCK
    assert wf.pipeline is None
    assert wf.model_name == MODEL_NAME_QWEN3
    assert wf.model_path == FAKE_MODEL
    assert wf.quant_dtype == "int8"
    assert wf.output_dir == TMP_DEPLOY_OUT
    assert wf.is_mx is False
    assert wf.is_int is True
    assert wf.is_hif is False


def test_deploy_init_rejects_unsupported_granularity():
    args = SimpleNamespace(
        granularity="model",
        model_name=MODEL_NAME_QWEN3,
        model=FAKE_MODEL,
        quant_dtype="int8",
        output_dir=TMP_DEPLOY_OUT,
        seed=0,
    )
    with pytest.raises(ValueError, match="granularity 'block' or 'tensor'"):
        LlmDeployWorkflow(args)


def test_deploy_init_requires_quant_target_for_block():
    args = SimpleNamespace(
        granularity=GRANULARITY_BLOCK,
        quant_target=[],
        model_name=MODEL_NAME_QWEN3,
        model=FAKE_MODEL,
        quant_dtype="int",
        output_dir=TMP_DEPLOY_OUT,
        seed=0,
    )
    with pytest.raises(ValueError, match="block requires: --quant_target"):
        LlmDeployWorkflow(args)


def test_deploy_init_requires_quant_dtype_for_block():
    args = SimpleNamespace(
        granularity=GRANULARITY_BLOCK,
        quant_target=["mlp"],
        model_name=MODEL_NAME_QWEN3,
        model=FAKE_MODEL,
        quant_dtype="",
        output_dir=TMP_DEPLOY_OUT,
        seed=0,
    )
    with pytest.raises(ValueError, match="block requires: --quant_dtype"):
        LlmDeployWorkflow(args)


def test_deploy_init_tensor_exempts_quant_args():
    args = SimpleNamespace(
        granularity="tensor",
        quant_target=[],
        model_name=MODEL_NAME_QWEN3,
        model=FAKE_MODEL,
        quant_dtype="",
        output_dir=TMP_DEPLOY_OUT,
        seed=0,
    )
    wf = LlmDeployWorkflow(args)
    assert wf.granularity == "tensor"


def test_deploy_init_mx_flag():
    wf = LlmDeployWorkflow(
        SimpleNamespace(
            granularity=GRANULARITY_BLOCK,
            quant_target=["mlp"],
            model_name="q",
            model="/m",
            quant_dtype="mxfp8",
            output_dir="/out",
            seed=0,
        )
    )
    assert wf.is_mx is True
    assert wf.is_int is False
    assert wf.is_hif is False


def test_deploy_init_hif_flag():
    wf = LlmDeployWorkflow(
        SimpleNamespace(
            granularity=GRANULARITY_BLOCK,
            quant_target=["mlp"],
            model_name="q",
            model="/m",
            quant_dtype="hifp8",
            output_dir="/out",
            seed=0,
        )
    )
    assert wf.is_hif is True


# ---- _write_remaining_original_weights: shard split -----------------------


def test_write_remaining_weights_splits_on_max_shard_size(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()

    n_elements = 3 * 1024 * 1024
    save_file(
        {BIG: torch.zeros(n_elements, dtype=torch.float32)},
        str(src / "shard.safetensors"),
    )
    wf = _make_workflow(model_path=str(src), output_dir=str(dst))
    weight_map = {BIG: "shard.safetensors"}
    updated = wf._write_remaining_original_weights(weight_map, set())
    assert BIG in updated
    assert (dst / REST_00000).exists()


# ---- _write_remaining_original_weights: empty chunk -----------------------


def test_write_remaining_weights_empty_input_returns_empty():
    wf = _make_workflow()
    updated = wf._write_remaining_original_weights({}, set())
    assert updated == {}


# ---- _run_blockwise: empty layer tensors ----------------------------------


def test_deploy_run_blockwise_empty_layer_tensors(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.export_block_deploy",
        lambda pipeline, layer_idx, quant_ignore_layers, *, linear_bits=None: ({}, {}),
    )
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.logger",
        MagicMock(),
    )
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.tqdm",
        lambda iterable, desc="": iterable,
    )

    wf = _make_workflow(output_dir=str(tmp_path))
    wf.pipeline = _make_pipeline_mock(num_layers=2)
    wf._copy_support_files = MagicMock()
    wf._load_weight_index = MagicMock(return_value={"weight_map": {}})
    wf._write_block_file = MagicMock()
    wf._write_remaining_original_weights = MagicMock(return_value={})
    wf._refresh_weight_index = MagicMock(return_value=str(tmp_path / "index.json"))
    wf._refresh_config = MagicMock()

    result = wf._run_blockwise()
    assert wf._write_block_file.call_count == 0
    assert "index_path" in result


def test_run_tensorwise_copies_and_rewrites_weight_index(monkeypatch, tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (dst / CONFIG_JSON).write_text("{}")
    save_file(
        {MODEL_LAYERS_0_MLP_UP_PROJ_WEIGHT: torch.ones(1)},
        str(src / KEY_SHARD1_SAFETENSORS),
    )

    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.load_file",
        lambda path, device="cpu": {
            MODEL_LAYERS_0_MLP_UP_PROJ_WEIGHT: torch.arange(
                6, dtype=torch.bfloat16
            ).reshape(2, 3),
        },
    )
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.tqdm",
        lambda iterable, desc="": iterable,
    )
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.logger",
        MagicMock(),
    )

    from amct_pytorch.quantization.dtypes import register_dtype

    register_dtype()
    wf = _make_workflow(model_path=str(src), output_dir=str(dst), quant_dtype="int")
    _tensor_json_args(wf, tmp_path, ("model.layers.0.mlp.up_proj",))
    wf._copy_support_files = MagicMock()
    original_index = {
        METADATA_KEY: {"foo": "bar"},
        "weight_map": {
            MODEL_LAYERS_0_MLP_UP_PROJ_WEIGHT: KEY_SHARD1_SAFETENSORS,
        },
    }
    wf._load_weight_index = MagicMock(return_value=original_index)
    wf.pipeline = MagicMock(
        spec=[
            "config",
            "cache_scheme",
            "bits_scheme",
            "get_deploy_candidates",
            "validate_deploy_selection",
        ]
    )
    wf.pipeline.config = SimpleNamespace()
    wf.pipeline.cache_scheme.return_value = {}
    wf.pipeline.bits_scheme.return_value = None
    wf.pipeline.get_deploy_candidates.side_effect = lambda names: names
    wf.pipeline.validate_deploy_selection.return_value = None

    result = wf._run_tensorwise()

    saved_index = json.loads((dst / SAFETENSORS_INDEX_JSON).read_text())
    refreshed = json.loads((dst / CONFIG_JSON).read_text())
    assert result["index_path"] == str(dst / SAFETENSORS_INDEX_JSON)
    assert result["num_output_files"] == 1
    assert saved_index[METADATA_KEY]["foo"] == "bar"
    assert saved_index["weight_map"] == {
        MODEL_LAYERS_0_MLP_UP_PROJ_WEIGHT: KEY_SHARD1_SAFETENSORS,
        "model.layers.0.mlp.up_proj.weight_scale": KEY_SHARD1_SAFETENSORS,
        "model.layers.0.mlp.up_proj.weight_offset": KEY_SHARD1_SAFETENSORS,
    }
    assert QUANTIZATION_CONFIG not in refreshed


# ---- Task 14: _convert_tensor ----------------------------------------------


def _make_bit_policy():
    """Build a minimal BitPolicy suitable for constructor tests."""
    from amct_pytorch.quantization.bit_policy import BitPolicy

    return BitPolicy(
        {
            "mlp": {"gate_proj": {"w_bits": 8, "a_bits": 8}},
            "attn-linear": {},
            "attn-cache": {"q": 8, "k": 8, "p": 8, "v": 8},
        }
    )


def test_convert_tensor_bf16():
    wf = _make_workflow(quant_dtype="bf16")
    t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    out = wf._convert_tensor("test.weight", t)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out.float(), t)


def test_convert_tensor_unsupported_raises():
    wf = _make_workflow(quant_dtype="int8")
    t = torch.zeros(2, 3)
    with pytest.raises(NotImplementedError, match="tensor granularity"):
        wf._convert_tensor("test.weight", t)


def test_refresh_config_bf16_uses_pipeline_quant_config(tmp_path):
    wf = _make_workflow(output_dir=str(tmp_path), quant_dtype="bf16")
    wf.pipeline = SimpleNamespace(cache_scheme=dict, bits_scheme=lambda: None)
    config = {"torch_dtype": "float32", "quantization_config": {"old": True}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    wf._refresh_config(quant_ignore_layers=[])
    refreshed = json.loads((tmp_path / "config.json").read_text())
    assert refreshed["torch_dtype"] == "float32"
    assert refreshed["quantization_config"]["format"] == "int-quantized"


# ---- _run_tensorwise int/mxfp quant branch (diff coverage) ---------------


def test_run_tensorwise_int_quant_path(monkeypatch, tmp_path):
    """When quant_dtype='int', weights in quant_layers go through quant_payload."""
    from amct_pytorch.quantization.dtypes import register_dtype

    register_dtype()

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()

    (src / CONFIG_JSON).write_text(json.dumps({"torch_dtype": "float32"}))
    save_file(
        {
            "model.layers.0.self_attn.o_proj.weight": torch.randn(
                4, 4, dtype=torch.bfloat16
            )
        },
        str(src / KEY_SHARD1_SAFETENSORS),
    )
    (src / SAFETENSORS_INDEX_JSON).write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "model.layers.0.self_attn.o_proj.weight": KEY_SHARD1_SAFETENSORS
                },
            }
        )
    )

    logger = MagicMock()
    monkeypatch.setattr("amct_pytorch.workflows.llm_deploy.logger", logger)
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.tqdm",
        lambda iterable, desc="": iterable,
    )

    wf = _make_workflow(model_path=str(src), output_dir=str(dst), quant_dtype="int")
    _tensor_json_args(wf, tmp_path=tmp_path)
    wf.pipeline = MagicMock(
        spec=[
            "config",
            "get_scale_name",
            "block_size",
            "cache_scheme",
            "bits_scheme",
            "get_deploy_candidates",
            "validate_deploy_selection",
        ]
    )
    wf.pipeline.get_scale_name = MagicMock(
        return_value=("_scale_inv", "missing_scale_inv")
    )
    wf.pipeline.config = SimpleNamespace()
    wf.pipeline.cache_scheme = MagicMock(return_value={})
    wf.pipeline.bits_scheme = MagicMock(return_value=None)
    wf.pipeline.block_size = MagicMock(return_value=32)
    wf.pipeline.get_deploy_candidates.side_effect = lambda names: names
    wf.pipeline.validate_deploy_selection.return_value = None
    wf.setup = MagicMock(return_value="sink")

    result = wf.run()

    # Should produce output with quantized weight keys
    refreshed_index = json.loads((dst / SAFETENSORS_INDEX_JSON).read_text())
    assert result["num_output_files"] == 1
    # quant_payload produces qweight + weight_scale + weight_bias
    assert "model.layers.0.self_attn.o_proj.weight" in refreshed_index["weight_map"]


def test_load_weight_index_single_shard_synthesizes(tmp_path):
    # Single model.safetensors (no index.json) -> synthesize equivalent index
    save_file(
        {"a.weight": torch.zeros(2), "b.weight": torch.ones(3)},
        str(tmp_path / MODEL_SAFETENSORS),
    )
    wf = _make_workflow(model_path=str(tmp_path))
    idx = wf._load_weight_index()  # pylint: disable=protected-access
    assert idx["weight_map"] == {
        "a.weight": MODEL_SAFETENSORS,
        "b.weight": MODEL_SAFETENSORS,
    }
    assert idx[METADATA_KEY]["total_size"] > 0


def test_load_weight_index_missing_raises(tmp_path):
    # Neither index.json nor model.safetensors -> FileNotFoundError
    wf = _make_workflow(model_path=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        wf._load_weight_index()  # pylint: disable=protected-access


# ---- seed wiring -----------------------------------------------------------


def test_init_applies_seed_from_args(monkeypatch):
    captured = []
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.seed_everything", captured.append
    )
    args = SimpleNamespace(
        granularity=GRANULARITY_BLOCK,
        quant_target=["mlp"],
        model_name=MODEL_NAME_QWEN3,
        model=FAKE_MODEL,
        quant_dtype="int",
        output_dir=TMP_DEPLOY_OUT,
        seed=13,
    )
    workflow = LlmDeployWorkflow(args)
    assert captured == [13]
    assert workflow.seed == 13


@pytest.mark.parametrize(
    "marker",
    [
        "config",
        "quant_model_description.json",
        "quant_model_weights.safetensors.index.json",
        "rot.safetensors",
    ],
)
def test_ascend_rejects_quantized_source_before_output(tmp_path, marker, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    config = (
        {"quantization_config": {"quant_method": "compressed-tensors"}}
        if marker == "config"
        else {}
    )
    (source / "config.json").write_text(json.dumps(config))
    if marker != "config":
        (source / marker).write_text("{}")
    workflow = _make_workflow(str(source), str(output), "int")
    _tensor_json_args(workflow, tmp_path=tmp_path)
    workflow.model_name = workflow.args.model_name = "qwen3_5"
    monkeypatch.setattr(workflow, "_build_pipeline", lambda: SimpleNamespace())
    with pytest.raises(ValueError, match="source"):
        workflow.run()
    assert not output.exists()


def test_shared_shard_export_serializes_payload_once(monkeypatch, tmp_path):
    from safetensors import safe_open
    from safetensors.torch import load_file

    from amct_pytorch.quantization.dtypes import register_dtype

    register_dtype()
    module = importlib.import_module("amct_pytorch.workflows.llm_deploy")
    original_quant = module.quant_payload
    calls = []

    def counted_quant(quant_cls, name, weight, bit, **kwargs):
        calls.append(name)
        return original_quant(quant_cls, name, weight, bit, **kwargs)

    monkeypatch.setattr(module, "quant_payload", counted_quant)
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    # Non-square matrix catches an accidental transpose; zero/constant rows
    # exercise the minimum scale and per-channel quantization.
    weight = torch.tensor([[0, 0, 0], [2, 2, 2], [-1, 0.25, 0.5]], dtype=torch.bfloat16)
    tensors = {"layer.weight": weight, "ignored.weight": torch.ones(2, 3)}
    tensors["layer.bias"] = torch.ones(3, dtype=torch.bfloat16)
    save_file(tensors, str(src / MODEL_SAFETENSORS))
    wf = _make_workflow(str(src), str(dst), "int")
    wf.pipeline = SimpleNamespace(
        get_scale_name=lambda name: ("_scale_inv", "missing_scale_inv"),
        block_size=lambda weight: 32,
    )
    weight_map = dict.fromkeys(tensors, MODEL_SAFETENSORS)
    exported = wf._convert_tensorwise_shard(
        MODEL_SAFETENSORS, src, weight_map, {"layer": 8}, {}
    )
    result = load_file(str(dst / MODEL_SAFETENSORS))
    assert calls == ["layer.weight"]
    assert set(exported) == set(result)
    assert set(result) == set(tensors) | {"layer.weight_scale", "layer.weight_offset"}
    assert result["layer.weight"].shape == weight.shape
    assert result["layer.weight"].dtype == torch.int8
    assert torch.equal(
        result["layer.weight"],
        torch.tensor([[0, 0, 0], [127, 127, 127], [-127, 32, 64]], dtype=torch.int8),
    )
    assert torch.equal(result["ignored.weight"], tensors["ignored.weight"])
    assert result["ignored.weight"].dtype == torch.float32
    scale = result["layer.weight_scale"]
    assert scale.shape == (3, 1)
    assert torch.isfinite(scale).all() and (scale > 0).all()
    # Includes INT rounding plus BF16 division/scale rounding.
    error = (result["layer.weight"].float() * scale.float() - weight.float()).abs()
    bound = (
        weight.float().abs().amax(dim=1, keepdim=True) / 127 / 2
        + weight.float().abs() / 128
        + 1e-8
    )
    assert (error <= bound).all()
    with safe_open(str(dst / MODEL_SAFETENSORS), framework="pt") as f:
        assert f.get_slice("layer.weight").get_dtype() == "I8"
        assert f.get_slice("layer.weight_scale").get_dtype() == ("BF16")
        assert f.get_slice("layer.weight_offset").get_dtype() == "BF16"
        assert torch.equal(result["layer.weight_offset"], torch.zeros_like(scale))
        assert torch.equal(result["layer.bias"], tensors["layer.bias"])


@pytest.mark.parametrize(
    "weight",
    [
        torch.ones(2, 3),
        torch.ones(3, dtype=torch.bfloat16),
        torch.ones(2, 3, dtype=torch.int8),
        torch.empty(0, 3, dtype=torch.bfloat16),
        torch.full((2, 3), float("nan"), dtype=torch.bfloat16),
        torch.full((2, 3), float("inf"), dtype=torch.bfloat16),
    ],
)
def test_ascend_rejects_invalid_weight_before_quantization(
    monkeypatch, tmp_path, weight
):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    save_file({"layer.weight": weight}, str(src / MODEL_SAFETENSORS))
    wf = _make_workflow(str(src), str(dst), "int")

    def unexpected_quant(*args):
        pytest.fail("Invalid source must be rejected before quantization")

    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.quant_payload", unexpected_quant
    )
    with pytest.raises(ValueError, match="layer.weight"):
        wf._convert_tensorwise_shard(
            MODEL_SAFETENSORS,
            src,
            {"layer.weight": MODEL_SAFETENSORS},
            {"layer": 8},
            {},
        )
    assert list(dst.iterdir()) == []


def test_tensor_payload_uses_ascend_adapter(monkeypatch):
    from amct_pytorch.quantization.dtypes import register_dtype

    register_dtype()
    wf = _make_workflow(quant_dtype="int")

    result = wf._export_tensor_payload(
        "layer.weight", torch.ones(2, 3, dtype=torch.bfloat16), 8
    )
    assert result["layer.weight_scale"].dtype == torch.bfloat16
    assert "layer.weight_offset" in result


@pytest.mark.parametrize(
    "quant_spec,global_dtype,expected_weight_dtype,expected_scale_dtype",
    [
        ("int", "mxfp", torch.int8, torch.bfloat16),
        ("mxfp", "int", torch.float8_e4m3fn, torch.uint8),
    ],
)
def test_structured_quant_spec_controls_ascend_payload(
    quant_spec, global_dtype, expected_weight_dtype, expected_scale_dtype
):
    from amct_pytorch.common.models.llm.common.deployment.deploy_selection import (
        QuantSpec,
    )
    from amct_pytorch.quantization.dtypes import register_dtype

    register_dtype()
    wf = _make_workflow(quant_dtype=global_dtype)

    result = wf._export_tensor_payload(
        "layer.weight",
        torch.ones(2, 64, dtype=torch.bfloat16),
        QuantSpec(quant_spec, 8, 8),
    )

    assert result["layer.weight"].dtype == expected_weight_dtype
    assert result["layer.weight_scale"].dtype == expected_scale_dtype
    assert ("layer.weight_offset" in result) is (quant_spec == "int")


@pytest.mark.parametrize(
    "quant_dtype,global_dtype,platform",
    [
        ("int", "mxfp", "A3"),
        ("int", "", "A3"),
        ("mxfp", "int", "ascend950"),
    ],
)
def test_qwen3_6_dense_structured_plan_exports_planned_dtype(
    tmp_path, quant_dtype, global_dtype, platform
):
    from safetensors.torch import load_file

    from amct_pytorch.common.models.llm.common.deployment.deploy_selection import (
        DeploySelection,
        QuantLayerRule,
        QuantSpec,
        build_tensor_deploy_plan,
    )
    from amct_pytorch.quantization.dtypes import register_dtype

    register_dtype()
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    name = "model.language_model.layers.0.mlp.gate_proj"
    tensors = {
        name + ".weight": torch.arange(128, dtype=torch.bfloat16).reshape(2, 64),
        "model.language_model.norm.weight": torch.ones(64, dtype=torch.bfloat16),
    }
    save_file(tensors, str(src / MODEL_SAFETENSORS))
    (src / CONFIG_JSON).write_text('{"num_hidden_layers": 1}')
    index = {"weight_map": dict.fromkeys(tensors, MODEL_SAFETENSORS)}
    selection = DeploySelection((QuantLayerRule(name, QuantSpec(quant_dtype, 8, 8)),))

    wf = _make_workflow(str(src), str(dst), global_dtype)
    wf.args.deploy_platform = platform
    wf.pipeline = SimpleNamespace()
    wf.deploy_plan = build_tensor_deploy_plan(index["weight_map"], selection)
    wf._load_weight_index = lambda: index

    result = wf._run_tensorwise()

    output = load_file(str(dst / MODEL_SAFETENSORS))
    if quant_dtype == "int":
        assert output[name + ".weight"].dtype == torch.int8
        assert output[name + ".weight_scale"].dtype == torch.bfloat16
        assert output[name + ".weight_offset"].dtype == torch.bfloat16
    else:
        assert output[name + ".weight"].dtype == torch.float8_e4m3fn
        assert output[name + ".weight_scale"].dtype == torch.uint8
        assert output[name + ".weight_scale"].shape == (2, 2)
        assert name + ".weight_offset" not in output
    assert output["model.language_model.norm.weight"].dtype == torch.bfloat16
    validation = json.loads(Path(result["validation_path"]).read_text())
    assert validation["status"] == "passed"


@pytest.mark.parametrize("relation", ["same", "child", "parent", "nonempty", "symlink"])
def test_ascend_rejects_unsafe_output_before_setup(tmp_path, relation, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    output = tmp_path / "output"
    if relation == "same":
        output = source
    elif relation == "child":
        output = source / "output"
    elif relation == "parent":
        output = tmp_path
    elif relation == "nonempty":
        output.mkdir()
        (output / "keep.txt").write_text("keep")
    else:
        try:
            output.symlink_to(source, target_is_directory=True)
        except OSError as exc:
            if sys.platform == "win32" and getattr(exc, "winerror", None) == 1314:
                pytest.skip(
                    "creating directory symlinks requires Windows developer mode"
                )
            raise
    wf = _make_workflow(str(source), str(output), "int")
    _tensor_json_args(wf, tmp_path=tmp_path)
    monkeypatch.setattr(
        wf, "setup", lambda: pytest.fail("Unsafe output must fail before setup")
    )
    with pytest.raises(ValueError, match="output"):
        wf.run()
    assert (source / "config.json").read_text() == "{}"


def test_ascend_shards_release_source_tensors(monkeypatch, tmp_path):
    import weakref

    from amct_pytorch.quantization.dtypes import register_dtype

    register_dtype()
    module = importlib.import_module("amct_pytorch.workflows.llm_deploy")
    original = module.load_file
    references = []

    def tracked_load(*args, **kwargs):
        assert all(ref() is None for ref in references)
        state = original(*args, **kwargs)
        references.extend(weakref.ref(tensor) for tensor in state.values())
        return state

    monkeypatch.setattr(module, "load_file", tracked_load)
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    for i in range(2):
        save_file(
            {
                f"layer{i}.weight": torch.ones(2, 3, dtype=torch.bfloat16),
                f"mtp{i}.weight": torch.ones(2, 3, dtype=torch.bfloat16),
            },
            str(src / f"{i}.safetensors"),
        )
    wf = _make_workflow(str(src), str(dst), "int")
    cache = {}
    for i in range(2):
        wf._convert_tensorwise_shard(
            f"{i}.safetensors", src, {}, {f"layer{i}": 8}, cache
        )
        assert cache == {}
        assert all(ref() is None for ref in references)


@pytest.mark.parametrize(
    "filename",
    ["/tmp/escape.safetensors", "../escape.safetensors", "nested/weights.safetensors"],
)
def test_ascend_plan_rejects_nonlocal_shard_names(filename):
    wf = _make_workflow(quant_dtype="int")
    with pytest.raises(ValueError, match="shard"):
        wf._prepare_tensor_deploy_plan_from_checkpoint(
            {"weight_map": {"x.weight": filename}}
        )


def test_prepare_ascend_structured_plan_does_not_require_bit_policy():
    from amct_pytorch.common.models.llm.common.deployment.deploy_selection import (
        DeploySelection,
        QuantLayerRule,
        QuantSpec,
    )

    name = "model.language_model.layers.0.mlp.gate_proj"
    wf = _make_workflow(quant_dtype="")
    wf.args.deploy_platform = "A3"
    wf.args.deploy_selection = DeploySelection(
        (QuantLayerRule(name, QuantSpec("int", 8, 8)),)
    )
    wf.pipeline = SimpleNamespace(
        args=wf.args,
        config=SimpleNamespace(num_hidden_layers=1, num_nextn_predict_layers=0),
        get_deploy_candidates=lambda names: names,
        validate_deploy_selection=lambda selected: None,
    )

    plan = wf._prepare_tensor_deploy_plan_from_checkpoint(
        {"weight_map": {name + ".weight": MODEL_SAFETENSORS}}
    )

    assert plan.module_quant_specs[name] == QuantSpec("int", 8, 8)


def test_json_tensor_export_two_shards_preserves_float_bias_and_metadata(
    tmp_path, monkeypatch
):
    from safetensors.torch import load_file

    from amct_pytorch.common.models.llm.common.deployment.deploy_selection import (
        DeploySelection,
        QuantLayerRule,
        QuantSpec,
    )
    from amct_pytorch.quantization.bit_policy import BitPolicy
    from amct_pytorch.quantization.dtypes import register_dtype

    register_dtype()
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    selected = "model.layers.0.self_attn.o_proj"
    ignored = "model.layers.1.self_attn.o_proj"
    unselected = "model.layers.0.mlp.up_proj"
    tensors = {
        selected + ".weight": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
        selected + ".bias": torch.tensor([0.5, -1.0, 2.0]),
        ignored + ".weight": torch.full((3, 4), 0.5),
        unselected + ".weight": torch.full((3, 4), 1.5),
        "model.norm.weight": torch.ones(4),
    }
    weight_map = {k: f"shard-{i % 2}.safetensors" for i, k in enumerate(tensors)}
    for filename in set(weight_map.values()):
        save_file(
            {k: v for k, v in tensors.items() if weight_map[k] == filename},
            str(src / filename),
        )
    (src / CONFIG_JSON).write_text(
        '{"num_hidden_layers":1,"num_nextn_predict_layers":1}'
    )
    (src / SAFETENSORS_INDEX_JSON).write_text(json.dumps({"weight_map": weight_map}))
    args = SimpleNamespace(
        model=str(src),
        model_name="glm5_2",
        seed=42,
        granularity="tensor",
        quant_dtype="int",
        output_dir=str(dst),
        quant_target=["moe"],
        quant_layers_config=str(tmp_path / "quant_layers_config.json"),
        deploy_selection=DeploySelection(
            (
                QuantLayerRule(
                    "model.layers.*.self_attn.o_proj", QuantSpec("int", 8, 8)
                ),
            ),
            (ignored,),
        ),
        bit_policy=BitPolicy({"attn-linear": {"w_bits": 8, "a_bits": 8}}),
    )
    Path(args.quant_layers_config).write_text(
        json.dumps(
            {
                "quant_layers": [
                    {
                        "pattern": "model.layers.*.self_attn.o_proj",
                        "quant_dtype": "int",
                        "w_bits": 8,
                        "a_bits": 8,
                    }
                ],
                "ignore_layers": [ignored],
            }
        )
    )
    wf = LlmDeployWorkflow(args)

    def forbidden(*args):
        pytest.fail("Tensor JSON path must not call model generators or bits_scheme")

    pipeline = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=1, num_nextn_predict_layers=1),
        get_scale_name=lambda k: ("_scale_inv", k + "_scale_inv"),
        block_size=lambda w: 32,
        get_deploy_candidates=lambda names: names,
        validate_deploy_selection=lambda selected: None,
        generate_tensorwise_quant_layers=forbidden,
        generate_tensorwise_ignore_layers=forbidden,
        bits_scheme=forbidden,
    )
    monkeypatch.setattr(wf, "_build_pipeline", lambda: pipeline)
    refresh = MagicMock(wraps=wf._refresh_config)
    monkeypatch.setattr(wf, "_refresh_config", refresh)
    result = wf.run()
    refresh.assert_not_called()
    actual = {}
    for file in dst.glob("*.safetensors"):
        actual.update(load_file(str(file)))
    assert set(actual) == set(tensors) | {
        selected + ".weight_scale",
        selected + ".weight_offset",
    }
    assert actual[selected + ".weight"].dtype == torch.int8
    ref_scale = tensors[selected + ".weight"].abs().amax(dim=1, keepdim=True) / 127
    expected_weight = torch.clamp(
        torch.round(tensors[selected + ".weight"] / ref_scale), -127, 127
    ).to(torch.int8)
    assert torch.equal(
        actual[selected + ".weight"],
        expected_weight,
    )
    for key in set(tensors) - {selected + ".weight"}:
        assert torch.equal(actual[key], tensors[key])
    cfg = json.loads((dst / CONFIG_JSON).read_text())
    assert "quantization_config" not in cfg
    assert set(json.loads(Path(result["index_path"]).read_text())["weight_map"]) == set(
        actual
    )
    assert wf.deploy_plan.mtp.status == "fully_ignored"


def test_tensor_invalid_bits_fails_before_setup(tmp_path, monkeypatch):
    from amct_pytorch.common.models.llm.common.deployment.deploy_selection import (
        DeploySelection,
        QuantLayerRule,
        QuantSpec,
    )

    args = SimpleNamespace(
        model="missing",
        model_name="glm5_2",
        seed=42,
        granularity="tensor",
        quant_dtype="int",
        output_dir=str(tmp_path / "out"),
        quant_layers_config=str(tmp_path / "quant_layers_config.json"),
        deploy_selection=DeploySelection(
            (
                QuantLayerRule(
                    "model.layers.*.mlp.experts.*.up_proj",
                    QuantSpec("int", 4, 8),
                ),
            )
        ),
    )
    Path(args.quant_layers_config).write_text(
        json.dumps(
            {
                "quant_layers": [
                    {
                        "pattern": "model.layers.*.mlp.experts.*.up_proj",
                        "quant_dtype": "int",
                        "w_bits": 4,
                        "a_bits": 8,
                    }
                ]
            }
        )
    )
    wf = LlmDeployWorkflow(args)
    monkeypatch.setattr(wf, "setup", lambda: pytest.fail("Must validate before setup"))
    with pytest.raises(ValueError, match="Unsupported deployment quantization scheme"):
        wf.run()
    assert not Path(args.output_dir).exists()


@pytest.mark.parametrize("custom_description", [False, True])
def test_ascend_tensor_run_preserves_source_config(
    tmp_path, monkeypatch, custom_description
):
    from amct_pytorch.common.models.llm.common.deployment.deploy_selection import (
        DeploySelection,
        QuantLayerRule,
        QuantSpec,
        build_tensor_deploy_plan,
    )
    from amct_pytorch.quantization.dtypes import register_dtype

    register_dtype()
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    source_config = '{\n  "model_type": "glm5", "custom_field": [1, 2]\n}\n'
    (src / CONFIG_JSON).write_text(source_config)
    name = "model.layers.0.self_attn.o_proj"
    save_file(
        {name + ".weight": torch.ones(2, 3, dtype=torch.bfloat16)},
        str(src / MODEL_SAFETENSORS),
    )
    index = {"weight_map": {name + ".weight": MODEL_SAFETENSORS}}
    wf = _make_workflow(str(src), str(dst), "int")
    wf.args.model_name = "hy_v3"
    wf.pipeline = SimpleNamespace(
        get_scale_name=lambda key: ("_scale_inv", "missing_scale_inv")
    )
    if custom_description:
        wf.pipeline.make_description = lambda names, selected: {
            "custom_tensors": sorted(names),
            "custom_selected": sorted(selected),
        }
    wf.deploy_plan = build_tensor_deploy_plan(
        index["weight_map"],
        DeploySelection((QuantLayerRule(name, QuantSpec("int", 8, 8)),), ()),
    )
    monkeypatch.setattr(wf, "_load_weight_index", lambda: index)
    refresh = MagicMock(
        side_effect=AssertionError("Deployment must not refresh config")
    )
    monkeypatch.setattr(wf, "_refresh_config", refresh)
    validation_report = {"status": "passed", "contract": {"quant_dtype": "int"}}
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.validate_artifacts",
        lambda output_dir, plan: validation_report,
    )
    wf._run_tensorwise()
    refresh.assert_not_called()
    assert (dst / CONFIG_JSON).read_bytes() == (src / CONFIG_JSON).read_bytes()
    assert (
        json.loads((dst / "deployment_validation.json").read_text())
        == validation_report
    )
    if custom_description:
        description = json.loads((dst / "quant_model_description.json").read_text())
        assert description["custom_selected"] == [name]
        assert set(description["custom_tensors"]) == {
            name + suffix for suffix in (".weight", ".weight_scale", ".weight_offset")
        }


def test_refresh_config_serializes_activation_only_linear_without_overlap(tmp_path):
    wf = _make_workflow(output_dir=str(tmp_path))
    wf.pipeline = SimpleNamespace(cache_scheme=lambda: {}, bits_scheme=lambda: None)
    (tmp_path / CONFIG_JSON).write_text(json.dumps({"hidden_size": 2}))
    bits = {
        "model.layers.0.mlp.gate_proj": (16, 8),
        "model.layers.0.mlp.up_proj": (8, 8),
        "model.layers.0.mlp.down_proj": (8, 16),
    }
    wf._refresh_config([], linear_bits=bits)
    saved = json.loads((tmp_path / CONFIG_JSON).read_text())
    config = saved[QUANTIZATION_CONFIG]
    groups = config["config_groups"].values()
    targets = [target for group in groups for target in group["targets"]]
    assert "Linear" not in targets
    assert all(targets.count(name) == 1 for name in bits)
    by_target = {target: group for group in groups for target in group["targets"]}
    assert by_target["model.layers.0.mlp.gate_proj"]["weights"] is None
    assert (
        by_target["model.layers.0.mlp.gate_proj"]["input_activations"]["num_bits"] == 8
    )
    assert by_target["model.layers.0.mlp.down_proj"]["input_activations"] is None
    assert by_target["model.layers.0.mlp.down_proj"]["weights"]["num_bits"] == 8
    assert saved["hidden_size"] == 2 and config["ignore"] == []


def test_blockwise_real_weight_files_keep_activation_only_metadata(tmp_path):
    from safetensors.torch import load_file
    from amct_pytorch.common.models.llm.common.base import BaseModel
    from amct_pytorch.quantization.modules.quant_linear import QuantLinear

    src, dst = tmp_path / "model", tmp_path / "deploy"
    src.mkdir()
    dst.mkdir()
    weight = torch.tensor([[1.0, -0.5], [0.25, 0.75]])
    other = torch.tensor([0.31, -0.97])
    key = "model.layers.0.linear.weight"
    save_file({key: weight, "other.weight": other}, str(src / MODEL_SAFETENSORS))
    (src / CONFIG_JSON).write_text(json.dumps({"hidden_size": 2}))
    args = SimpleNamespace(
        device="cpu", w_bits=16, a_bits=8, algos=[], quant_dtype="int"
    )
    linear = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        linear.weight.copy_(weight)
    block = torch.nn.ModuleDict(
        {"linear": QuantLinear(args, linear, w_bits=16, a_bits=8)}
    )

    class ModelLoadingBoundary:
        num_layers = 1
        iter_deploy_bindings = BaseModel.iter_deploy_bindings

        def __init__(self):
            self.args = args

        def build_quant_block(self, layer_idx):
            return block

        def load_selected_layer_ptq_params(self, *args, **kwargs):
            pass

        def get_layer_weight_prefix(self, layer_idx):
            return "model.layers.0."

    workflow = _make_workflow(
        model_path=str(src), output_dir=str(dst), quant_dtype="int8"
    )
    workflow.safetensors_files = None
    workflow.pipeline = ModelLoadingBoundary()
    result = workflow._run_blockwise()
    index = json.loads(Path(result["index_path"]).read_text())["weight_map"]
    restored = {}
    for name in set(index.values()):
        restored.update(load_file(str(dst / name)))
    assert torch.equal(restored[key], weight)
    assert torch.equal(restored["other.weight"], other)
    config = json.loads((dst / CONFIG_JSON).read_text())[QUANTIZATION_CONFIG]
    groups = [
        group
        for group in config["config_groups"].values()
        if "model.layers.0.linear" in group["targets"]
    ]
    assert len(groups) == 1 and groups[0]["weights"] is None
    assert groups[0]["input_activations"]["num_bits"] == 8
    assert config["ignore"] == []
