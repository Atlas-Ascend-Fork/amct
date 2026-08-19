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

import json
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from amct_pytorch.classic.quant_result import load_quant_manifest
from amct_pytorch.common.models.llm.common.deploy_export import export_block_deploy
from amct_pytorch.common.models.llm.common.ptq_params import PtqParamHandler
from amct_pytorch.common.models.llm.common.ptq_units import make_ptq_unit
from amct_pytorch.common.models.llm.common.quant_result import (
    LLM_ARTIFACT_KIND_DIRECT_CONFIG,
    LLM_ARTIFACT_KIND_PTQ_PARAMS,
    LLM_QUANT_MODULE_TYPE,
    LlmPtqUnitSpec,
    LlmQuantResultReader,
    LlmQuantResultSet,
    LlmQuantResultWriter,
    compute_checkpoint_fingerprint,
    discover_llm_ptq_units,
    flatten_ptq_payload,
    normalize_quant_result_dirs,
    unflatten_ptq_payload,
    validate_checkpoint_fingerprint,
)
from amct_pytorch.quantization.bit_policy import BitPolicy


MANIFEST_FIELDS = {
    "format",
    "schema_version",
    "artifact_id",
    "producer",
    "source_model",
    "units",
}
UNIT_FIELDS = {
    "layer_name",
    "file",
    "quant_module_type",
    "ori_module_type",
    "quant_config",
    "non_tensor_params",
    "tensors",
    "file_size",
    "sha256",
    "entry_digest",
}


class _Loadable(nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.register_buffer("loaded", torch.zeros(1))
        self.fail = fail

    def load_ptq_params(self, params):
        if self.fail:
            raise RuntimeError("forced load failure")
        self.loaded.copy_(params["value"])

    def export_deploy(self):
        return {"qweight": self.loaded.clone()}


class _TwoUnitBlock(nn.Module):
    def __init__(self, fail_second=False):
        super().__init__()
        self.first = _Loadable()
        self.second = _Loadable(fail=fail_second)


class _Pipeline:
    def __init__(self, fail_second=False):
        self.quant_target = ["mlp"]
        self.num_layers = 1
        self.ptq_param_handler = PtqParamHandler()
        self.fail_second = fail_second

    @staticmethod
    def get_layer_weight_prefix(layer_idx):
        return f"model.layers.{layer_idx}."

    @staticmethod
    def iter_ptq_units(layer_idx, block):
        yield make_ptq_unit("mlp", "first", layer_idx, block.first)
        yield make_ptq_unit("mlp", "second", layer_idx, block.second)


def _args(target="mlp"):
    return SimpleNamespace(
        model_name="tiny_llm",
        quant_target=[target],
        quant_dtype="int",
        bit_policy=BitPolicy({"w_bits": 8, "a_bits": 8}),
        algos=["lwc"],
        is_per_tensor=False,
        k_size=128,
    )


def _direct_args(target="mlp"):
    args = _args(target)
    args.algos = []
    return args


def _checkpoint(tmp_path, weight_value=1.0, model_type="tiny"):
    model_dir = tmp_path / f"model_{weight_value}_{model_type}"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": model_type, "hidden_size": 4}), encoding="utf-8"
    )
    save_file(
        {"model.layers.0.weight": torch.full((2, 2), weight_value)},
        str(model_dir / "model.safetensors"),
    )
    return model_dir


def _specs():
    return (
        LlmPtqUnitSpec("model.layers.0.first", 0, "mlp", "first", "_Loadable", {}),
        LlmPtqUnitSpec("model.layers.0.second", 0, "mlp", "second", "_Loadable", {}),
    )


def _write_artifact(tmp_path, model_dir, name="artifact", target="mlp"):
    artifact_dir = tmp_path / name
    writer = LlmQuantResultWriter.create(
        str(artifact_dir),
        _specs(),
        compute_checkpoint_fingerprint(str(model_dir)),
        _args(target),
        target,
    )
    block = _TwoUnitBlock()
    units = list(_Pipeline.iter_ptq_units(0, block))
    writer.write_unit(units[0], {"value": torch.tensor([3.0]), "step": 7})
    writer.write_unit(units[1], {"value": torch.tensor([5.0]), "step": 9})
    writer.publish()
    return artifact_dir


def _write_single_target_artifact(tmp_path, model_dir, name, target, kind, unit_name):
    artifact_dir = tmp_path / name
    module = _Loadable()
    spec = LlmPtqUnitSpec(
        f"model.layers.0.{unit_name}",
        0,
        kind,
        unit_name,
        "_Loadable",
        {},
    )
    writer = LlmQuantResultWriter.create(
        str(artifact_dir),
        [spec],
        compute_checkpoint_fingerprint(str(model_dir)),
        _args(target),
        target,
    )
    writer.write_unit(
        make_ptq_unit(kind, unit_name, 0, module),
        {"value": torch.tensor([4.0])},
    )
    writer.publish()
    return artifact_dir


def _write_direct_artifact(tmp_path, model_dir, name="direct_artifact"):
    artifact_dir = tmp_path / name
    writer = LlmQuantResultWriter.create(
        str(artifact_dir),
        _specs(),
        compute_checkpoint_fingerprint(str(model_dir)),
        _direct_args(),
        "mlp",
    )
    block = _TwoUnitBlock()
    for unit in _Pipeline.iter_ptq_units(0, block):
        writer.write_unit(unit, {})
    writer.publish()
    return artifact_dir


def test_flatten_round_trip_preserves_nested_types_and_tensors():
    payload = {
        "b": (torch.tensor([1, 2]), None, torch.float16),
        "a": [True, 3, 1.25, "value"],
    }
    tensors, structure = flatten_ptq_payload(payload)
    assert list(tensors) == ["tensor_000000"]
    restored = unflatten_ptq_payload(tensors, structure)
    assert torch.equal(restored["b"][0], payload["b"][0])
    assert restored["b"][1:] == (None, torch.float16)
    assert restored["a"] == payload["a"]


@pytest.mark.parametrize("bad_key", [1, ""])
def test_flatten_rejects_non_string_or_empty_mapping_keys(bad_key):
    with pytest.raises(TypeError, match="key"):
        flatten_ptq_payload({bad_key: torch.zeros(1)})


def test_flatten_rejects_arbitrary_objects():
    with pytest.raises(TypeError, match="unsupported object"):
        flatten_ptq_payload({"bad": object()})


def test_flatten_rejects_non_finite_float():
    with pytest.raises(ValueError, match="non-finite"):
        flatten_ptq_payload({"bad": float("inf")})


def test_unflatten_rejects_missing_tensor_reference():
    with pytest.raises(ValueError, match="missing tensor"):
        unflatten_ptq_payload({}, {"type": "tensor", "key": "tensor_000000"})


def test_unflatten_rejects_duplicate_tensor_reference():
    structure = {
        "type": "list",
        "items": [
            {"type": "tensor", "key": "tensor_000000"},
            {"type": "tensor", "key": "tensor_000000"},
        ],
    }
    with pytest.raises(ValueError, match="more than once"):
        unflatten_ptq_payload({"tensor_000000": torch.zeros(1)}, structure)


def test_unflatten_rejects_unreferenced_tensor():
    with pytest.raises(ValueError, match="unreferenced"):
        unflatten_ptq_payload({"tensor_000000": torch.zeros(1)}, {"type": "none"})


def test_checkpoint_fingerprint_is_stable_and_prefixed(tmp_path):
    model_dir = _checkpoint(tmp_path)
    first = compute_checkpoint_fingerprint(str(model_dir))
    second = compute_checkpoint_fingerprint(str(model_dir))
    assert first == second
    assert first["model_type"] == "tiny"
    assert first["config_sha256"].startswith("sha256:")
    assert first["state_dict_sha256"].startswith("sha256:")


def test_checkpoint_fingerprint_detects_weight_change(tmp_path):
    first = _checkpoint(tmp_path, weight_value=1.0)
    second = _checkpoint(tmp_path, weight_value=2.0)
    expected = compute_checkpoint_fingerprint(str(first))
    with pytest.raises(ValueError, match="state_dict_sha256"):
        validate_checkpoint_fingerprint(str(second), expected)


def test_checkpoint_fingerprint_detects_config_change(tmp_path):
    first = _checkpoint(tmp_path, weight_value=1.0, model_type="tiny_a")
    second = _checkpoint(tmp_path, weight_value=1.0, model_type="tiny_b")
    expected = compute_checkpoint_fingerprint(str(first))
    with pytest.raises(ValueError, match="model_type"):
        validate_checkpoint_fingerprint(str(second), expected)


def test_normalize_quant_result_dirs():
    assert normalize_quant_result_dirs(None) == ()
    assert normalize_quant_result_dirs("/a") == ("/a",)
    assert normalize_quant_result_dirs(["/a", "/b"]) == ("/a", "/b")


def test_normalize_quant_result_dirs_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        normalize_quant_result_dirs(["/a", "/a"])


def test_discover_llm_ptq_units_records_exact_module_paths():
    pipeline = _Pipeline()
    pipeline.build_quant_block = lambda layer_idx: _TwoUnitBlock()
    pipeline.num_layers = 2

    specs = discover_llm_ptq_units(pipeline, 0, 2)

    assert [spec.layer_name for spec in specs] == [
        "model.layers.0.first",
        "model.layers.0.second",
        "model.layers.1.first",
        "model.layers.1.second",
    ]
    assert [spec.key for spec in specs] == [
        (0, "mlp", "first"),
        (0, "mlp", "second"),
        (1, "mlp", "first"),
        (1, "mlp", "second"),
    ]


def test_writer_emits_frozen_v2_schema_and_round_trips(tmp_path):
    model_dir = _checkpoint(tmp_path)
    artifact_dir = _write_artifact(tmp_path, model_dir)
    manifest = load_quant_manifest(str(artifact_dir))
    assert set(manifest) == MANIFEST_FIELDS
    assert len(manifest["units"]) == 2
    for entry in manifest["units"]:
        assert set(entry) == UNIT_FIELDS
        assert entry["quant_module_type"] == LLM_QUANT_MODULE_TYPE
        assert entry["ori_module_type"] == "_Loadable"
        assert entry["quant_config"]["artifact_kind"] == LLM_ARTIFACT_KIND_PTQ_PARAMS
        assert entry["quant_config"]["bit_policy"] == {
            "w_bits": 8,
            "a_bits": 8,
        }
    result_set = LlmQuantResultSet.from_directories([str(artifact_dir)], str(model_dir))
    assert result_set.quant_targets == ("mlp",)
    assert torch.equal(
        result_set.readers[0].payloads[(0, "mlp", "first")]["value"],
        torch.tensor([3.0]),
    )


def test_direct_config_artifact_round_trips_without_calibrated_params(tmp_path):
    model_dir = _checkpoint(tmp_path)
    artifact_dir = _write_direct_artifact(tmp_path, model_dir)

    result_set = LlmQuantResultSet.from_directories([str(artifact_dir)], str(model_dir))

    reader = result_set.readers[0]
    assert reader.artifact_kind == LLM_ARTIFACT_KIND_DIRECT_CONFIG
    assert reader.base_config["algos"] == []
    assert reader.payloads
    assert all(payload == {} for payload in reader.payloads.values())


def test_writer_rejects_nonempty_direct_config_payload(tmp_path):
    model_dir = _checkpoint(tmp_path)
    writer = LlmQuantResultWriter.create(
        str(tmp_path / "direct"),
        _specs(),
        compute_checkpoint_fingerprint(str(model_dir)),
        _direct_args(),
        "mlp",
    )
    unit = make_ptq_unit("mlp", "first", 0, _Loadable())

    with pytest.raises(ValueError, match="must have an empty payload"):
        writer.write_unit(unit, {"value": torch.tensor([1.0])})


def test_writer_rejects_empty_ptq_params_payload(tmp_path):
    model_dir = _checkpoint(tmp_path)
    writer = LlmQuantResultWriter.create(
        str(tmp_path / "ptq"),
        _specs(),
        compute_checkpoint_fingerprint(str(model_dir)),
        _args(),
        "mlp",
    )
    unit = make_ptq_unit("mlp", "first", 0, _Loadable())

    with pytest.raises(ValueError, match="must contain calibrated parameters"):
        writer.write_unit(unit, {})


def test_reader_rejects_empty_payload_labeled_as_ptq_params(monkeypatch, tmp_path):
    model_dir = _checkpoint(tmp_path)
    artifact_dir = _write_direct_artifact(tmp_path, model_dir)
    manifest = load_quant_manifest(str(artifact_dir))
    entry = manifest["units"][0]
    entry["quant_config"]["artifact_kind"] = LLM_ARTIFACT_KIND_PTQ_PARAMS
    entry["quant_config"]["algos"] = ["lwc"]
    monkeypatch.setattr(
        "amct_pytorch.common.models.llm.common.quant_result.read_quant_unit",
        lambda *args, **kwargs: ({}, {}, entry),
    )

    with pytest.raises(ValueError, match="must contain calibrated parameters"):
        LlmQuantResultReader(str(artifact_dir), {"units": [entry]})


def test_direct_config_load_validates_units_but_skips_param_loading(tmp_path):
    model_dir = _checkpoint(tmp_path)
    artifact_dir = _write_direct_artifact(tmp_path, model_dir)
    result_set = LlmQuantResultSet.from_directories([str(artifact_dir)], str(model_dir))
    pipeline = _Pipeline()
    block = _TwoUnitBlock()

    reloaded = result_set.load_block(pipeline, 0, block)

    assert reloaded is not block
    assert torch.equal(reloaded.first.loaded, torch.zeros(1))
    assert torch.equal(reloaded.second.loaded, torch.zeros(1))


def test_interrupted_writer_has_no_manifest(tmp_path):
    model_dir = _checkpoint(tmp_path)
    artifact_dir = tmp_path / "interrupted"
    writer = LlmQuantResultWriter.create(
        str(artifact_dir),
        _specs(),
        compute_checkpoint_fingerprint(str(model_dir)),
        _args(),
        "mlp",
    )
    unit = make_ptq_unit("mlp", "first", 0, _Loadable())
    writer.write_unit(unit, {"value": torch.tensor([1.0])})
    assert not (artifact_dir / "quant_manifest.json").exists()
    with pytest.raises(RuntimeError, match="incomplete"):
        writer.publish()
    assert not (artifact_dir / "quant_manifest.json").exists()


def test_reader_rejects_missing_manifest(tmp_path):
    model_dir = _checkpoint(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="quant_manifest"):
        LlmQuantResultSet.from_directories([str(empty)], str(model_dir))


def test_reader_rejects_tampered_unit(tmp_path):
    model_dir = _checkpoint(tmp_path)
    artifact_dir = _write_artifact(tmp_path, model_dir)
    manifest = load_quant_manifest(str(artifact_dir))
    unit_path = artifact_dir / manifest["units"][0]["file"]
    with unit_path.open("r+b") as file_handle:
        file_handle.seek(-1, os.SEEK_END)
        byte = file_handle.read(1)
        file_handle.seek(-1, os.SEEK_END)
        file_handle.write(bytes([byte[0] ^ 1]))
    with pytest.raises(ValueError, match="SHA-256"):
        LlmQuantResultSet.from_directories([str(artifact_dir)], str(model_dir))


def test_reader_rejects_extra_file(tmp_path):
    model_dir = _checkpoint(tmp_path)
    artifact_dir = _write_artifact(tmp_path, model_dir)
    (artifact_dir / "extra.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="not listed"):
        LlmQuantResultSet.from_directories([str(artifact_dir)], str(model_dir))


def test_reader_rejects_different_source_checkpoint(tmp_path):
    source = _checkpoint(tmp_path, weight_value=1.0)
    other = _checkpoint(tmp_path, weight_value=2.0)
    artifact_dir = _write_artifact(tmp_path, source)
    with pytest.raises(ValueError, match="state_dict_sha256"):
        LlmQuantResultSet.from_directories([str(artifact_dir)], str(other))


def test_apply_to_args_uses_manifest_as_construction_source(tmp_path):
    model_dir = _checkpoint(tmp_path)
    artifact_dir = _write_artifact(tmp_path, model_dir)
    result_set = LlmQuantResultSet.from_directories([str(artifact_dir)], str(model_dir))
    args = SimpleNamespace(
        model_name="wrong",
        quant_target=[],
        quant_dtype="mxfp",
        bit_policy=BitPolicy(),
        algos=[],
        is_per_tensor=True,
        k_size=1,
    )
    result_set.apply_to_args(args)
    assert args.model_name == "tiny_llm"
    assert args.quant_target == ["mlp"]
    assert args.quant_dtype == "int"
    assert args.bit_policy.cfg == {"w_bits": 8, "a_bits": 8}
    assert args.algos == ["lwc"]
    assert args.is_per_tensor is False
    assert args.k_size == 128


def test_multiple_target_artifacts_form_one_deploy_configuration(tmp_path):
    model_dir = _checkpoint(tmp_path)
    mlp = _write_single_target_artifact(
        tmp_path, model_dir, "mlp_artifact", "mlp", "mlp", "mlp"
    )
    attn = _write_single_target_artifact(
        tmp_path,
        model_dir,
        "attn_artifact",
        "attn-linear",
        "attn",
        "self_attn",
    )

    result_set = LlmQuantResultSet.from_directories(
        [str(mlp), str(attn)], str(model_dir)
    )
    args = _args()
    result_set.apply_to_args(args)

    assert result_set.quant_targets == ("mlp", "attn-linear")
    assert args.quant_target == ["mlp", "attn-linear"]
    result_set.validate_for_pipeline(
        SimpleNamespace(quant_target=args.quant_target, num_layers=1)
    )


def test_multiple_artifacts_reject_duplicate_quant_target(tmp_path):
    model_dir = _checkpoint(tmp_path)
    first = _write_single_target_artifact(
        tmp_path, model_dir, "first_artifact", "mlp", "mlp", "mlp"
    )
    second = _write_single_target_artifact(
        tmp_path, model_dir, "second_artifact", "mlp", "mlp", "mlp"
    )
    with pytest.raises(ValueError, match="same quant_target"):
        LlmQuantResultSet.from_directories([str(first), str(second)], str(model_dir))


def test_load_block_is_transactional_on_second_unit_failure(tmp_path):
    model_dir = _checkpoint(tmp_path)
    artifact_dir = _write_artifact(tmp_path, model_dir)
    result_set = LlmQuantResultSet.from_directories([str(artifact_dir)], str(model_dir))
    pipeline = _Pipeline(fail_second=True)
    block = _TwoUnitBlock(fail_second=True)
    first_id = id(block.first)
    second_id = id(block.second)
    with pytest.raises(RuntimeError, match="forced load failure"):
        result_set.load_block(pipeline, 0, block)
    assert id(block.first) == first_id
    assert id(block.second) == second_id
    assert torch.equal(block.first.loaded, torch.zeros(1))
    assert torch.equal(block.second.loaded, torch.zeros(1))


def test_load_block_returns_loaded_copy_and_preserves_input(tmp_path):
    model_dir = _checkpoint(tmp_path)
    artifact_dir = _write_artifact(tmp_path, model_dir)
    result_set = LlmQuantResultSet.from_directories([str(artifact_dir)], str(model_dir))
    pipeline = _Pipeline()
    block = _TwoUnitBlock()
    staged = result_set.load_block(pipeline, 0, block)
    assert staged is not block
    assert torch.equal(staged.first.loaded, torch.tensor([3.0]))
    assert torch.equal(staged.second.loaded, torch.tensor([5.0]))
    assert torch.equal(block.first.loaded, torch.zeros(1))
    assert torch.equal(block.second.loaded, torch.zeros(1))


def test_artifact_and_legacy_paths_export_identical_deploy_tensors(tmp_path):
    model_dir = _checkpoint(tmp_path)
    artifact_dir = _write_artifact(tmp_path, model_dir)
    result_set = LlmQuantResultSet.from_directories([str(artifact_dir)], str(model_dir))

    class DeployPipeline(_Pipeline):
        def __init__(self):
            super().__init__()
            self.args = SimpleNamespace(device="cpu")

        @staticmethod
        def build_quant_block(layer_idx):
            return _TwoUnitBlock()

        @staticmethod
        def load_selected_layer_ptq_params(layer_idx, block, strict=False):
            block.first.load_ptq_params({"value": torch.tensor([3.0])})
            block.second.load_ptq_params({"value": torch.tensor([5.0])})

        @staticmethod
        def iter_deploy_bindings(layer_idx, block):
            yield "model.layers.0.first.weight", block.first
            yield "model.layers.0.second.weight", block.second

    pipeline = DeployPipeline()
    direct_tensors, direct_routes = export_block_deploy(pipeline, 0, [])
    artifact_tensors, artifact_routes = export_block_deploy(
        pipeline, 0, [], param_loader=result_set.load_block
    )

    assert direct_routes == artifact_routes
    assert set(direct_tensors) == set(artifact_tensors)
    for key in direct_tensors:
        assert torch.equal(direct_tensors[key], artifact_tensors[key])


def test_validate_full_coverage_rejects_partial_artifact(tmp_path):
    model_dir = _checkpoint(tmp_path)
    artifact_dir = _write_artifact(tmp_path, model_dir)
    result_set = LlmQuantResultSet.from_directories([str(artifact_dir)], str(model_dir))
    result_set.readers[0].validate_full_coverage(1)
    with pytest.raises(ValueError, match="missing layers"):
        result_set.readers[0].validate_full_coverage(2)
    (LlmQuantResultReader,)
