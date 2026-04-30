"""Logic tests for LlmDeployWorkflow.

The full `_run_blockwise` requires a real safetensors-backed model dir; we
cover the file-IO and helper logic in isolation.
"""

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from amct_pytorch.workflows.llm_deploy import LlmDeployWorkflow


def _make_workflow(model_path="/fake/model", output_dir="/tmp/deploy_out", quant_dtype="int8"):
    workflow = LlmDeployWorkflow.__new__(LlmDeployWorkflow)
    args = SimpleNamespace(
        granularity="block",
        model_name="qwen3",
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
        ("model.safetensors.index.json", True),
        ("model-00001-of-00002.safetensors", True),
        ("config.json", False),
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
    (src / "config.json").write_text("{}")
    (src / "tokenizer.model").write_text("tok")
    (src / "model.safetensors").write_text("BIG")
    (src / "model.safetensors.index.json").write_text("{}")
    (src / ".hidden").write_text("skip")
    (src / "subdir").mkdir()
    (src / "subdir" / "more.txt").write_text("x")

    wf = _make_workflow(model_path=str(src), output_dir=str(dst))
    wf._copy_support_files()

    assert (dst / "config.json").exists()
    assert (dst / "tokenizer.model").exists()
    assert (dst / "subdir" / "more.txt").exists()
    # Weight files and hidden dotfiles are skipped.
    assert not (dst / "model.safetensors").exists()
    assert not (dst / "model.safetensors.index.json").exists()
    assert not (dst / ".hidden").exists()


def test_copy_support_files_skips_existing_destinations(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "config.json").write_text("new")
    (dst / "config.json").write_text("old")

    wf = _make_workflow(model_path=str(src), output_dir=str(dst))
    wf._copy_support_files()
    assert (dst / "config.json").read_text() == "old"


# ---- _load_weight_index --------------------------------------------------


def test_load_weight_index_reads_json(tmp_path):
    index = {"weight_map": {"a.weight": "shard1.safetensors"}, "metadata": {"total_size": 999}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
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
        "model.layers.0.mlp.up_proj.qweight": "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.up_proj.weight_scale": "model.layers.0.mlp.up_proj.weight",
    }
    original = {"model.layers.0.mlp.up_proj.weight": "shard1.safetensors"}
    out = wf._collect_replaced_original_weights(layer_tensors, tensor_routes, original)
    assert out == {"model.layers.0.mlp.up_proj.weight"}


def test_collect_replaced_original_weights_returns_empty_when_unrelated():
    wf = _make_workflow()
    layer_tensors = {"unknown.weight": "x"}
    routes = {"unknown.weight": "unknown.weight"}
    original = {"different.weight": "shard.safetensors"}
    assert (
        wf._collect_replaced_original_weights(layer_tensors, routes, original) == set()
    )


# ---- _refresh_weight_index ----------------------------------------------


def test_refresh_weight_index_writes_metadata_and_total_size(tmp_path):
    wf = _make_workflow(output_dir=str(tmp_path))
    # Create two shard files so total_size sums them.
    (tmp_path / "rest_00000.safetensors").write_bytes(b"x" * 100)
    (tmp_path / "layer_000.safetensors").write_bytes(b"y" * 50)

    original = {"metadata": {"foo": "bar"}, "weight_map": {}}
    weight_map = {
        "alpha": "rest_00000.safetensors",
        "beta": "layer_000.safetensors",
    }
    index_path = wf._refresh_weight_index(original, weight_map)
    saved = json.loads(Path(index_path).read_text())
    assert saved["metadata"]["foo"] == "bar"
    assert saved["metadata"]["total_size"] == 150
    assert saved["weight_map"] == weight_map


# ---- _refresh_config -----------------------------------------------------


def test_refresh_config_attaches_quantization_block(tmp_path):
    wf = _make_workflow(output_dir=str(tmp_path))
    # Original config that the workflow reads in.
    (tmp_path / "config.json").write_text(json.dumps({"hidden_size": 4096}))

    wf._refresh_config(quant_ignore_layers=["lm_head"])
    refreshed = json.loads((tmp_path / "config.json").read_text())
    assert refreshed["hidden_size"] == 4096
    assert "quantization_config" in refreshed
    assert refreshed["quantization_config"]["ignore"] == ["lm_head"]
    # int dtype path -> int-quantized format.
    assert refreshed["quantization_config"]["format"] == "int-quantized"


def test_refresh_config_uses_float_format_for_mx_dtype(tmp_path):
    wf = _make_workflow(output_dir=str(tmp_path), quant_dtype="mxfp8")
    (tmp_path / "config.json").write_text("{}")
    wf._refresh_config(quant_ignore_layers=[])
    refreshed = json.loads((tmp_path / "config.json").read_text())
    assert refreshed["quantization_config"]["format"] == "float-quantized"


# ---- _write_remaining_original_weights ----------------------------------


def test_write_remaining_original_weights_skips_replaced_and_shards_rest(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    # Source shard with three small tensors.
    save_file(
        {"a": torch.zeros(2), "b": torch.ones(3), "c": torch.full((4,), 2.0)},
        str(src / "shard1.safetensors"),
    )
    wf = _make_workflow(model_path=str(src), output_dir=str(dst))

    weight_map = {
        "a": "shard1.safetensors",
        "b": "shard1.safetensors",
        "c": "shard1.safetensors",
    }
    replaced = {"b"}  # b is replaced by a quant routine; should be skipped here.

    updated = wf._write_remaining_original_weights(weight_map, replaced)
    # a + c land in rest_00000 (small tensors easily fit one shard).
    assert set(updated) == {"a", "c"}
    assert (dst / "rest_00000.safetensors").exists()
    assert all(file_name.startswith("rest_") for file_name in updated.values())
