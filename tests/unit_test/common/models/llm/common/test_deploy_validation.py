# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

import importlib
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file


def validate(path):
    module = importlib.import_module(
        "amct_pytorch.common.models.llm.common.deploy_validation"
    )
    return module.validate_ascend_artifacts(
        path, SimpleNamespace(selected_modules=frozenset({"x"}))
    )


@pytest.fixture
def artifacts(tmp_path):
    tensors = {
        "x.weight": torch.ones(2, 3, dtype=torch.int8),
        "x.weight_scale": torch.ones(2, 1, dtype=torch.bfloat16),
        "x.weight_offset": torch.zeros(2, 1, dtype=torch.bfloat16),
        "model.layers.2.eh_proj.weight": torch.ones(2, 3, dtype=torch.bfloat16),
    }
    description = {
        "version": "1.0.0",
        "model_quant_type": "W8A8_DYNAMIC",
        "metadata": {},
        "group_size": 0,
        "is_rot_used": False,
    }
    description.update(
        {k: "W8A8_DYNAMIC" if k.startswith("x.") else "FLOAT" for k in tensors}
    )
    index = {
        "metadata": {"total_size": 26},
        "weight_map": dict.fromkeys(tensors, "one.safetensors"),
    }
    (tmp_path / "config.json").write_text(
        json.dumps({"num_hidden_layers": 2, "num_nextn_predict_layers": 1})
    )

    def write():
        save_file(tensors, str(tmp_path / "one.safetensors"))
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
        (tmp_path / "quant_model_description.json").write_text(json.dumps(description))

    write()
    return tmp_path, tensors, index, description, write


def test_valid_artifacts_report_header_bytes_and_counts(artifacts):
    path, *_ = artifacts
    report = validate(path)
    assert report["status"] == "passed"
    assert report["total_tensor_bytes"] == 26
    assert report["tensor_count"] == 4 and report["selected_modules"] == 1
    assert report["dynamic_tensor_count"] == 3 and report["float_tensor_count"] == 1
    assert all(value == 0 for value in report["error_counts"].values())


@pytest.mark.parametrize(
    "fault",
    [
        "missing_scale",
        "missing_offset",
        "nonzero_offset",
        "nan_scale",
        "zero_scale",
        "wrong_weight_dtype",
        "wrong_scale_dtype",
        "wrong_offset_shape",
        "index_missing",
        "index_wrong_file",
        "description_missing",
        "description_wrong",
        "duplicate",
        "total_size",
        "rotation",
        "mtp_quantized",
        "extra_tensor",
        "metadata",
        "static_parameter",
    ],
)
def test_corrupt_artifacts_fail_validation(artifacts, fault):
    path, tensors, index, description, write = artifacts
    if fault in {"missing_scale", "missing_offset"}:
        del tensors["x.weight_" + fault.removeprefix("missing_")]
    elif fault == "nonzero_offset":
        tensors["x.weight_offset"].fill_(1)
    elif fault in {"nan_scale", "zero_scale"}:
        tensors["x.weight_scale"].fill_(float("nan") if fault == "nan_scale" else 0)
    elif fault == "wrong_weight_dtype":
        tensors["x.weight"] = tensors["x.weight"].float()
    elif fault == "wrong_scale_dtype":
        tensors["x.weight_scale"] = tensors["x.weight_scale"].float()
    elif fault == "wrong_offset_shape":
        tensors["x.weight_offset"] = torch.zeros(2, dtype=torch.bfloat16)
    elif fault == "index_missing":
        del index["weight_map"]["x.weight"]
    elif fault == "index_wrong_file":
        index["weight_map"]["x.weight"] = "missing.safetensors"
    elif fault == "description_missing":
        del description["x.weight"]
    elif fault == "description_wrong":
        description["x.weight"] = "FLOAT"
    elif fault == "duplicate":
        save_file({"x.weight": tensors["x.weight"]}, str(path / "two.safetensors"))
    elif fault == "total_size":
        index["metadata"]["total_size"] += 8
    elif fault in {"rotation", "static_parameter", "extra_tensor"}:
        key = {
            "rotation": "rot.weight",
            "static_parameter": "x.input_scale",
            "extra_tensor": "extra.weight",
        }[fault]
        tensors[key] = torch.ones(1)
        if fault != "extra_tensor":
            index["weight_map"][key] = "one.safetensors"
            description[key] = "FLOAT"
            index["metadata"]["total_size"] += 4
    elif fault == "mtp_quantized":
        tensors["model.layers.2.eh_proj.weight"] = tensors[
            "model.layers.2.eh_proj.weight"
        ].to(torch.int8)
    elif fault == "metadata":
        description["is_rot_used"] = True
    write()
    report = validate(path)
    assert report["status"] == "failed"
    assert sum(report["error_counts"].values()) > 0
    assert report["errors"]


def test_validation_reads_only_auxiliary_tensor_values(artifacts, monkeypatch):
    path, *_ = artifacts
    module = importlib.import_module(
        "amct_pytorch.common.models.llm.common.deploy_validation"
    )
    original = module.safe_open
    reads = []

    class HeaderOnlyWeights:
        def __init__(self, *args, **kwargs):
            self.inner = original(*args, **kwargs)

        def __enter__(self):
            self.inner.__enter__()
            return self

        def __exit__(self, *args):
            return self.inner.__exit__(*args)

        def keys(self):
            return self.inner.keys()

        def get_slice(self, key):
            return self.inner.get_slice(key)

        def get_tensor(self, key):
            assert key in {"x.weight_scale", "x.weight_offset"}
            reads.append(key)
            return self.inner.get_tensor(key)

    monkeypatch.setattr(module, "safe_open", HeaderOnlyWeights)
    assert validate(path)["status"] == "passed"
    assert set(reads) == {"x.weight_scale", "x.weight_offset"}
