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

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[6]
    / "amct_pytorch/common/models/llm/common/weight_path_validation.py"
)
_SPEC = importlib.util.spec_from_file_location("weight_path_validation", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
resolve_safetensors_path = _MODULE.resolve_safetensors_path


def test_resolve_safetensors_path_accepts_plain_relative_file_name(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    shard_path = model_dir / "model-00001-of-00002.safetensors"
    shard_path.touch()

    assert (
        resolve_safetensors_path(model_dir, "model-00001-of-00002.safetensors")
        == shard_path.resolve()
    )


def test_resolve_safetensors_path_rejects_absolute_path_inside_model_directory(
    tmp_path,
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    shard_path = model_dir / "model.safetensors"
    shard_path.touch()

    with pytest.raises(ValueError, match="plain file name"):
        resolve_safetensors_path(model_dir, str(shard_path.resolve()))


@pytest.mark.parametrize(
    "file_name",
    [
        "../outside.safetensors",
        "subdir/shard.safetensors",
        "subdir\\shard.safetensors",
        ".",
        "..",
    ],
)
def test_resolve_safetensors_path_rejects_non_plain_file_name(tmp_path, file_name):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    with pytest.raises(ValueError, match="plain file name"):
        resolve_safetensors_path(model_dir, file_name)


def test_resolve_safetensors_path_rejects_non_safetensors_suffix(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.bin").touch()

    with pytest.raises(ValueError, match=r"\.safetensors suffix"):
        resolve_safetensors_path(model_dir, "model.bin")
