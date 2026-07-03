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
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import load_file as safe_load_file
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


def _make_workflow(model_path=FAKE_MODEL, output_dir=TMP_DEPLOY_OUT, quant_dtype="int8"):
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
    workflow.__dict__["_tensorwise_quant_layers"] = None
    return workflow

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
    index = {"weight_map": {"a.weight": KEY_SHARD1_SAFETENSORS}, METADATA_KEY: {"total_size": 999}}
    (tmp_path / SAFETENSORS_INDEX_JSON).write_text(json.dumps(index))
    wf = _make_workflow(model_path=str(tmp_path))
    assert wf._load_weight_index() == index

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
    routing = wf._write_block_file(layer_idx=4, layer_tensors={"a.weight": torch.zeros(2, 3)})
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


def test_llm_deploy_run_blockwise(monkeypatch):
    wf = _make_workflow()
    wf.granularity = GRANULARITY_BLOCK

    def setup():
        return "sink"
    wf.setup = setup

    def _run_blockwise():
        return {"index_path": "/out", "num_output_files": 1}
    wf._run_blockwise = _run_blockwise
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.logger",
        importlib.import_module("types").SimpleNamespace(remove=lambda h: None))
    result = wf.run()
    assert result["index_path"] == "/out"


def test_llm_deploy_setup(monkeypatch):
    wf = _make_workflow()
    called = {}
    monkeypatch.setattr(wf, "_register_components", lambda: called.update({"reg": True}))
    monkeypatch.setattr(wf, "_build_pipeline", lambda: called.update({"pipe": True}))
    monkeypatch.setattr("amct_pytorch.workflows.llm_deploy.setup_run_logging", lambda log_dir, name: ("sink", None))
    monkeypatch.setattr("os.makedirs", lambda p, exist_ok: None)
    monkeypatch.setattr("amct_pytorch.workflows.llm_deploy.ensure_log_dir", lambda d: None)
    wf.setup()
    assert called.get("reg") is True
    assert called.get("pipe") is True

# ---- Task 15: Additional deploy coverage ---------------------------------


def _make_deploy_workflow(**overrides):
    defaults = dict(
        model="/tmp/fake", model_name=MODEL_NAME_QWEN3, quant_dtype="int4",
        granularity=GRANULARITY_BLOCK, output_dir="/tmp/fake",
    )
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
    wf.__dict__["_tensorwise_quant_layers"] = None
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
        "amct_pytorch.workflows.llm_deploy.register_llm_models", lambda: None)
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.register_dtype", lambda: None)
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.register_algorithms", lambda: None)
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.ensure_log_dir", lambda d: None)
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.setup_run_logging", lambda log_dir, name: ("sink", None))
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.MODEL_REGISTRY",
        SimpleNamespace(get=lambda k: type("FM", (), {"__init__": lambda s, a: None})),
    )
    out = tmp_path / "deploy_out"
    wf = _make_deploy_workflow(output_dir=str(out))
    wf.setup()
    assert out.exists()
    assert wf.pipeline is not None


def test_deploy_run_unsupported_granularity():
    wf = _make_deploy_workflow(granularity="model")

    def setup():
        return "fake_sink"
    wf.setup = setup
    with pytest.raises(ValueError, match="Unsupported granularity"):
        wf.run()

# ---- _run_blockwise (mocked helpers) -------------------------------------


def test_deploy_run_blockwise_mocked_loop(monkeypatch, tmp_path):
    def _mock_export_block_deploy(pipeline, layer_idx, quant_ignore_layers):
        return (
            {LAYER_WEIGHT: torch.zeros(2, 3)},
            {LAYER_WEIGHT: LAYER_WEIGHT},
        )

    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.export_block_deploy",
        _mock_export_block_deploy,
    )
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.logger", MagicMock(),
    )
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.tqdm",
        lambda iterable, desc="": iterable,
    )

    wf = _make_workflow(output_dir=str(tmp_path))
    wf.pipeline = SimpleNamespace(num_layers=2)
    wf._copy_support_files = MagicMock()
    wf._load_weight_index = MagicMock(return_value={"weight_map": {
        LAYER_WEIGHT: KEY_SHARD1_SAFETENSORS,
    }})
    wf._write_block_file = MagicMock(return_value={LAYER_WEIGHT: "layer_000.safetensors"})
    wf._collect_replaced_original_weights = MagicMock(return_value={LAYER_WEIGHT})
    wf._write_remaining_original_weights = MagicMock(return_value={
        "other.weight": REST_00000,
    })
    wf._refresh_weight_index = MagicMock(return_value=str(tmp_path / SAFETENSORS_INDEX_JSON))
    wf._refresh_config = MagicMock()

    result = wf._run_blockwise()
    assert "index_path" in result
    assert "num_output_files" in result
    assert wf._write_block_file.call_count == 2
    wf._refresh_config.assert_called_once()

# ---- __init__ via actual constructor -------------------------------------


def test_deploy_init_sets_all_attrs():
    args = SimpleNamespace(
        granularity=GRANULARITY_BLOCK, model_name=MODEL_NAME_QWEN3, model=FAKE_MODEL,
        quant_dtype="int8", output_dir=TMP_DEPLOY_OUT,
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


def test_deploy_init_mx_flag():
    wf = LlmDeployWorkflow(SimpleNamespace(
        granularity=GRANULARITY_BLOCK, model_name="q", model="/m", quant_dtype="mxfp8",
        output_dir="/out",
    ))
    assert wf.is_mx is True
    assert wf.is_int is False
    assert wf.is_hif is False


def test_deploy_init_hif_flag():
    wf = LlmDeployWorkflow(SimpleNamespace(
        granularity=GRANULARITY_BLOCK, model_name="q", model="/m", quant_dtype="hifp8",
        output_dir="/out",
    ))
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
        lambda pipeline, layer_idx, quant_ignore_layers: ({}, {}),
    )
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.logger", MagicMock(),
    )
    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.tqdm",
        lambda iterable, desc="": iterable,
    )

    wf = _make_workflow(output_dir=str(tmp_path))
    wf.pipeline = SimpleNamespace(num_layers=2)
    wf._copy_support_files = MagicMock()
    wf._load_weight_index = MagicMock(return_value={"weight_map": {}})
    wf._write_block_file = MagicMock()
    wf._write_remaining_original_weights = MagicMock(return_value={})
    wf._refresh_weight_index = MagicMock(return_value=str(tmp_path / "index.json"))
    wf._refresh_config = MagicMock()

    result = wf._run_blockwise()
    assert wf._write_block_file.call_count == 0
    assert "index_path" in result


# ---- _run_tensorwise ------------------------------------------------------


def _write_tensorwise_source_model(src, source_file, tensors, weight_map):
    save_file(tensors, str(src / source_file))
    (src / SAFETENSORS_INDEX_JSON).write_text(json.dumps({
        "metadata": {"total_size": 1},
        "weight_map": weight_map,
    }))
    (src / CONFIG_JSON).write_text(json.dumps({"quantization_config": {"old": True}}))


def _run_tensorwise_workflow(wf):
    wf.granularity = "tensor"
    wf.setup = lambda: "fake_sink"
    return wf.run()


class _FakeWeightDequant:
    def __init__(self):
        self.calls = []

    def __call__(self, weight, scale, block_size=128, is_mx=False, is_packed=False):
        self.calls.append((weight.clone(), scale.clone(), block_size, is_mx, is_packed))
        return torch.full(weight.shape, 3.0, dtype=torch.float32)


class _FakeMxfpQuantObj:
    def __init__(self, bits):
        self.bits = bits

    def export_deploy(self, weight):
        assert self.bits == 8
        assert torch.equal(weight, torch.full((2, 2), 3.0))
        return {
            "qweight": torch.full(weight.shape, 7, dtype=torch.int8),
            "weight_scale": torch.ones(1, 1, dtype=torch.float32),
            "weight_bias": torch.full((2,), 2.0, dtype=torch.float32),
        }


class _FakeMxfpDtypeRegistry:
    @staticmethod
    def get(name):
        assert name == "mxfp"
        return _FakeMxfpQuantObj


class _FakeTensorwisePipeline:
    def __init__(self, quant_weight):
        self.quant_weight = quant_weight

    @staticmethod
    def get_scale_name(weight_name):
        return "_scale_inv", f"{weight_name}_scale_inv"

    def generate_tensorwise_quant_layers(self, config, weight_map):
        return {self.quant_weight.rsplit(".", 1)[0]: 8}


def _assert_tensorwise_mxfp_outputs(dst, source_file, quant_weight, plain_weight, scale_name, result):
    out_tensors = safe_load_file(dst / source_file, device="cpu")
    output_index = json.loads((dst / SAFETENSORS_INDEX_JSON).read_text())
    output_scale = quant_weight.replace(".weight", ".weight_scale")
    output_bias = quant_weight.replace(".weight", ".weight_bias")
    assert result["num_output_files"] == 1
    assert torch.equal(out_tensors[quant_weight], torch.full((2, 2), 7, dtype=torch.int8))
    assert torch.equal(out_tensors[output_scale], torch.ones(1, 1, dtype=torch.float32))
    assert torch.equal(out_tensors[output_bias], torch.full((2,), 2.0, dtype=torch.float32))
    assert torch.equal(out_tensors[plain_weight], torch.ones(2, 2, dtype=torch.bfloat16))
    assert scale_name not in out_tensors
    assert output_index["weight_map"] == {
        quant_weight: source_file,
        output_scale: source_file,
        output_bias: source_file,
        plain_weight: source_file,
    }


def test_deploy_run_tensorwise_uses_pipeline_tensorwise_info(monkeypatch, tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    source_file = "model-00001-of-00001.safetensors"
    quant_weight = "model.layers.1.mlp.experts.0.gate_proj.weight"
    scale_name = f"{quant_weight}_scale_inv"
    plain_weight = "lm_head.weight"

    _write_tensorwise_source_model(
        src,
        source_file,
        {
            quant_weight: torch.ones(2, 2, dtype=torch.uint8),
            scale_name: torch.ones(1, 1, dtype=torch.float32),
            plain_weight: torch.ones(2, 2, dtype=torch.bfloat16),
        },
        {
            quant_weight: source_file,
            scale_name: source_file,
            plain_weight: source_file,
        },
    )

    fake_weight_dequant = _FakeWeightDequant()
    monkeypatch.setattr("amct_pytorch.workflows.llm_deploy.weight_dequant", fake_weight_dequant)
    monkeypatch.setattr("amct_pytorch.workflows.llm_deploy.DTYPE_REGISTRY", _FakeMxfpDtypeRegistry())
    monkeypatch.setattr("amct_pytorch.workflows.llm_deploy.tqdm", lambda iterable, desc="": iterable)
    monkeypatch.setattr("amct_pytorch.workflows.llm_deploy.logger", MagicMock())

    wf = _make_workflow(model_path=str(src), output_dir=str(dst), quant_dtype="mxfp")
    wf.pipeline = _FakeTensorwisePipeline(quant_weight)

    result = _run_tensorwise_workflow(wf)

    _assert_tensorwise_mxfp_outputs(dst, source_file, quant_weight, plain_weight, scale_name, result)
    assert len(fake_weight_dequant.calls) == 1
    assert fake_weight_dequant.calls[0][2:] == (128, False, False)


def test_deploy_run_tensorwise_bf16_skips_quant_layers(monkeypatch, tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    source_file = "model-00001-of-00001.safetensors"
    quant_weight = "model.layers.1.mlp.experts.0.gate_proj.weight"
    scale_name = f"{quant_weight}_scale_inv"

    _write_tensorwise_source_model(
        src,
        source_file,
        {
            quant_weight: torch.ones(2, 2, dtype=torch.uint8),
            scale_name: torch.ones(1, 1, dtype=torch.float32),
        },
        {
            quant_weight: source_file,
            scale_name: source_file,
        },
    )

    class FakePipeline:
        @staticmethod
        def get_scale_name(weight_name):
            return "_scale_inv", f"{weight_name}_scale_inv"

        @staticmethod
        def generate_tensorwise_quant_layers(config, weight_map):
            raise AssertionError("bf16 tensorwise deploy must not generate quant layers")

    monkeypatch.setattr(
        "amct_pytorch.workflows.llm_deploy.weight_dequant",
        lambda weight, scale, **kwargs: torch.full(weight.shape, 5.0, dtype=torch.float32),
    )
    monkeypatch.setattr("amct_pytorch.workflows.llm_deploy.tqdm", lambda iterable, desc="": iterable)
    monkeypatch.setattr("amct_pytorch.workflows.llm_deploy.logger", MagicMock())

    wf = _make_workflow(model_path=str(src), output_dir=str(dst), quant_dtype="bf16")
    wf.pipeline = FakePipeline()

    _run_tensorwise_workflow(wf)

    out_tensors = safe_load_file(dst / source_file, device="cpu")
    refreshed_config = json.loads((dst / CONFIG_JSON).read_text())
    assert out_tensors[quant_weight].dtype == torch.bfloat16
    assert torch.equal(out_tensors[quant_weight].float(), torch.full((2, 2), 5.0))
    assert scale_name not in out_tensors
    assert refreshed_config["torch_dtype"] == "bfloat16"
    assert "quantization_config" not in refreshed_config


# ---- Task 14: _convert_tensor / _refresh_config_tensor --------------------


def _make_bit_policy():
    """Build a minimal BitPolicy suitable for constructor tests."""
    from amct_pytorch.quantization.bit_policy import BitPolicy
    return BitPolicy({
        "mlp": {"gate_proj": {"w_bits": 8, "a_bits": 8}},
        "attn-linear": {},
        "attn-cache": {"q": 8, "k": 8, "p": 8, "v": 8},
    })


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


def test_refresh_config_tensor_bf16(monkeypatch, tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    wf = _make_workflow(model_path=str(src), output_dir=str(dst), quant_dtype="bf16")
    config = {"torch_dtype": "float32", "quantization_config": {"old": True}}
    (src / "config.json").write_text(json.dumps(config))
    (src / SAFETENSORS_INDEX_JSON).write_text(json.dumps({"metadata": {}, "weight_map": {}}))
    monkeypatch.setattr("amct_pytorch.workflows.llm_deploy.tqdm", lambda iterable, desc="": iterable)
    monkeypatch.setattr("amct_pytorch.workflows.llm_deploy.logger", MagicMock())

    _run_tensorwise_workflow(wf)

    refreshed = json.loads((dst / "config.json").read_text())
    assert refreshed["torch_dtype"] == "bfloat16"
    assert "quantization_config" not in refreshed


def test_refresh_config_tensor_uses_pipeline_quant_config(monkeypatch, tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    wf = _make_workflow(model_path=str(src), output_dir=str(dst), quant_dtype="mxfp")
    config = {
        "model_type": "hy_v3",
        "num_hidden_layers": 80,
        "num_nextn_predict_layers": 1,
    }
    expected_quant_config = {
        "quant_method": "mxfp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
        "ignored_layers": ["lm_head", "model.layers.80"],
        "kv_cache_scheme": {"type": "float", "num_bits": 8},
    }
    wf.pipeline = SimpleNamespace(
        generate_tensorwise_quant_config=lambda config_dict, quant_layers: expected_quant_config
    )
    wf.__dict__["_tensorwise_quant_layers"] = {"model.layers.0.self_attn.q_proj": 8}
    (src / "config.json").write_text(json.dumps(config))
    (src / SAFETENSORS_INDEX_JSON).write_text(json.dumps({"metadata": {}, "weight_map": {}}))
    monkeypatch.setattr("amct_pytorch.workflows.llm_deploy.tqdm", lambda iterable, desc="": iterable)
    monkeypatch.setattr("amct_pytorch.workflows.llm_deploy.logger", MagicMock())

    _run_tensorwise_workflow(wf)

    refreshed = json.loads((dst / "config.json").read_text())
    assert refreshed["model_type"] == "hy_v3"
    assert refreshed["quantization_config"] == expected_quant_config


def test_hyv3_tensorwise_quant_config_uses_compressed_tensors_groups_and_weight_map_ignore():
    from amct_pytorch.common.models.llm.hyv3.hyv3 import HyV3
    from amct_pytorch.quantization.bit_policy import BitPolicy

    model = HyV3.__new__(HyV3)
    model.args = SimpleNamespace(
        quant_dtype="mxfp",
        quant_target=["moe", "attn-linear"],
        bit_policy=BitPolicy({
            "w_bits": 8,
            "a_bits": 8,
            "moe": {
                "routed": {"w_bits": 4, "a_bits": 8},
                "shared": {"w_bits": 8, "a_bits": 8},
            },
        }),
    )
    config = {
        "num_hidden_layers": 80,
        "num_nextn_predict_layers": 1,
        "first_k_dense_replace": 1,
        "num_experts": 1,
    }
    weight_map = {
        "model.embed_tokens.weight": "shard.safetensors",
        "model.layers.0.self_attn.q_proj.weight": "shard.safetensors",
        "model.layers.0.self_attn.q_norm.weight": "shard.safetensors",
        "model.layers.1.mlp.experts.0.down_proj.weight": "shard.safetensors",
        "model.layers.1.mlp.router.gate.weight": "shard.safetensors",
        "model.layers.1.mlp.experts.0.down_proj.weight_scale": "shard.safetensors",
        "model.layers.80.input_layernorm.weight": "shard.safetensors",
        "model.norm.weight": "shard.safetensors",
        "lm_head.weight": "shard.safetensors",
    }
    quant_layers = model.generate_tensorwise_quant_layers(config, weight_map)

    quant_config = model.generate_tensorwise_quant_config(
        config,
        quant_layers,
    )

    assert quant_config["quant_method"] == "compressed-tensors"
    assert quant_config["format"] == "float-quantized"
    assert quant_config["quantization_status"] == "compressed"
    assert quant_config["weight_block_size"] == [1, 32]
    assert quant_config["config_groups"]["group_0"]["targets"] == ["Linear"]
    assert quant_config["config_groups"]["group_0"]["weights"]["num_bits"] == 8
    assert quant_config["config_groups"]["group_0"]["input_activations"]["num_bits"] == 8
    assert quant_config["config_groups"]["group_0"]["weights"]["type"] == "float"
    assert quant_config["config_groups"]["group_1"]["targets"] == ["MoEGMM"]
    assert quant_config["config_groups"]["group_1"]["weights"]["num_bits"] == 4
    assert quant_config["config_groups"]["group_1"]["input_activations"]["num_bits"] == 8
    assert quant_config["config_groups"]["group_1"]["weights"]["type"] == "float"
    assert quant_config["kv_cache_scheme"] == {
        "num_bits": 8,
        "type": "float",
        "strategy": "token",
        "group_size": -1,
        "dynamic": "true",
        "symmetric": "true",
    }
    assert "model.layers.0.self_attn.q_proj" not in quant_config["ignore"]
    assert "model.layers.1.mlp.experts.0.down_proj" not in quant_config["ignore"]
    assert "model.layers.1.mlp.experts.0.down_proj.weight_scale" not in quant_config["ignore"]
    assert quant_config["ignore"] == [
        "model.embed_tokens",
        "model.layers.0.self_attn.q_norm",
        "model.layers.1.mlp.router.gate",
        "model.layers.80.input_layernorm",
        "model.norm",
        "lm_head",
    ]


def test_hyv3_tensorwise_quant_config_uses_int_type_for_int_quant():
    from amct_pytorch.common.models.llm.hyv3.hyv3 import HyV3
    from amct_pytorch.quantization.bit_policy import BitPolicy

    model = HyV3.__new__(HyV3)
    model.args = SimpleNamespace(
        quant_dtype="int",
        quant_target=["attn-linear"],
        bit_policy=BitPolicy({"w_bits": 8, "a_bits": 8}),
    )
    config = {"num_hidden_layers": 30}

    quant_config = model.generate_tensorwise_quant_config(
        config,
        {"model.layers.0.self_attn.q_proj": 8},
    )

    assert quant_config["format"] == "int-quantized"
    assert quant_config["config_groups"]["group_0"]["weights"]["type"] == "int"
    assert quant_config["config_groups"]["group_0"]["input_activations"]["type"] == "int"
    assert quant_config["config_groups"]["group_1"]["weights"]["type"] == "int"
    assert quant_config["kv_cache_scheme"]["type"] == "int"
    assert quant_config["ignore"] == ["lm_head", "model.layers.30"]


def test_hyv3_tensorwise_quant_layers_respect_moe_target():
    from amct_pytorch.common.models.llm.hyv3.hyv3 import HyV3
    from amct_pytorch.quantization.bit_policy import BitPolicy

    model = HyV3.__new__(HyV3)
    model.args = SimpleNamespace(
        quant_target=["moe"],
        bit_policy=BitPolicy({"w_bits": 8, "a_bits": 8}),
    )
    config = {
        "num_hidden_layers": 2,
        "num_nextn_predict_layers": 0,
        "first_k_dense_replace": 1,
        "num_experts": 1,
    }
    weight_map = {
        "model.layers.1.mlp.experts.0.gate_proj.weight": "shard.safetensors",
        "model.layers.1.mlp.shared_mlp.down_proj.weight": "shard.safetensors",
        "model.layers.1.self_attn.q_proj.weight": "shard.safetensors",
    }

    quant_layers = model.generate_tensorwise_quant_layers(config, weight_map)

    assert "model.layers.1.mlp.experts.0.gate_proj" in quant_layers
    assert "model.layers.1.mlp.shared_mlp.down_proj" in quant_layers
    assert "model.layers.1.self_attn.q_proj" not in quant_layers


def test_hyv3_tensorwise_quant_layers_respect_attn_linear_target():
    from amct_pytorch.common.models.llm.hyv3.hyv3 import HyV3
    from amct_pytorch.quantization.bit_policy import BitPolicy

    model = HyV3.__new__(HyV3)
    model.args = SimpleNamespace(
        quant_target=["attn-linear"],
        bit_policy=BitPolicy({"w_bits": 8, "a_bits": 8}),
    )
    config = {
        "num_hidden_layers": 2,
        "num_nextn_predict_layers": 0,
        "first_k_dense_replace": 1,
        "num_experts": 1,
    }
    weight_map = {
        "model.layers.1.mlp.experts.0.gate_proj.weight": "shard.safetensors",
        "model.layers.1.self_attn.q_proj.weight": "shard.safetensors",
    }

    quant_layers = model.generate_tensorwise_quant_layers(config, weight_map)

    assert "model.layers.1.self_attn.q_proj" in quant_layers
    assert "model.layers.1.mlp.experts.0.gate_proj" not in quant_layers

