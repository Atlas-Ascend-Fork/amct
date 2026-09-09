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
import os
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


def collect_safetensors_files(model_path):
    return _MODULE.collect_safetensors_files(model_path)


def test_collect_safetensors_files_only_accepts_direct_regular_safetensors(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    shard_path = model_dir / "model.safetensors"
    shard_path.touch()
    (model_dir / "model.bin").touch()
    subdir = model_dir / "subdir"
    subdir.mkdir()
    (subdir / "nested.safetensors").touch()

    assert collect_safetensors_files(model_dir) == frozenset({shard_path.resolve()})


def test_collect_safetensors_files_rejects_directory_without_safetensors(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").touch()

    with pytest.raises(FileNotFoundError, match=r"No \.safetensors files found"):
        collect_safetensors_files(model_dir)


def test_collect_safetensors_files_rejects_non_directory(tmp_path):
    with pytest.raises(NotADirectoryError, match="Model path is not a directory"):
        collect_safetensors_files(tmp_path / "missing")


def test_resolve_safetensors_path_accepts_plain_relative_file_name(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    shard_path = model_dir / "model-00001-of-00002.safetensors"
    shard_path.touch()

    assert (
        resolve_safetensors_path(
            model_dir,
            "model-00001-of-00002.safetensors",
            collect_safetensors_files(model_dir),
        )
        == shard_path.resolve()
    )


def test_resolve_safetensors_path_accepts_absolute_path_inside_model_directory(
    tmp_path,
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    shard_path = model_dir / "model.safetensors"
    shard_path.touch()

    assert (
        resolve_safetensors_path(
            model_dir,
            str(shard_path.resolve()),
            collect_safetensors_files(model_dir),
        )
        == shard_path.resolve()
    )


@pytest.mark.skipif(os.name == "nt", reason="colon is reserved in Windows file names")
def test_resolve_safetensors_path_accepts_linux_file_name_with_colon(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    shard_path = model_dir / "C:shard.safetensors"
    shard_path.touch()

    assert (
        resolve_safetensors_path(
            model_dir,
            shard_path.name,
            collect_safetensors_files(model_dir),
        )
        == shard_path.resolve()
    )


def test_resolve_safetensors_path_rejects_absolute_path_outside_model_directory(
    tmp_path,
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "allowed.safetensors").touch()
    shard_path = tmp_path / "outside.safetensors"
    shard_path.touch()

    allowed_files = collect_safetensors_files(model_dir)
    with pytest.raises(ValueError, match="not in the model directory allowlist"):
        resolve_safetensors_path(model_dir, str(shard_path.resolve()), allowed_files)


def test_resolve_safetensors_path_rejects_absolute_path_in_subdirectory(tmp_path):
    model_dir = tmp_path / "model"
    shard_dir = model_dir / "subdir"
    shard_dir.mkdir(parents=True)
    (model_dir / "allowed.safetensors").touch()
    shard_path = shard_dir / "model.safetensors"
    shard_path.touch()

    allowed_files = collect_safetensors_files(model_dir)
    with pytest.raises(ValueError, match="not in the model directory allowlist"):
        resolve_safetensors_path(model_dir, str(shard_path.resolve()), allowed_files)


def test_resolve_safetensors_path_rejects_symlink_to_outside_model_directory(
    tmp_path,
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    outside_path = tmp_path / "outside.safetensors"
    outside_path.touch()
    shard_path = model_dir / "model.safetensors"
    try:
        shard_path.symlink_to(outside_path)
    except OSError as exc:
        pytest.skip(f"symlink creation is not available: {exc}")

    with pytest.raises(FileNotFoundError, match=r"No \.safetensors files found"):
        collect_safetensors_files(model_dir)


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
def test_resolve_safetensors_path_rejects_path_not_in_allowlist(tmp_path, file_name):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    with pytest.raises(ValueError):
        resolve_safetensors_path(model_dir, file_name, frozenset())


def test_resolve_safetensors_path_rejects_non_safetensors_suffix(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.bin").touch()

    with pytest.raises(ValueError, match=r"\.safetensors suffix"):
        resolve_safetensors_path(model_dir, "model.bin", frozenset())


@pytest.mark.parametrize("absolute", [False, True])
def test_resolve_safetensors_path_rejects_file_not_in_allowlist(tmp_path, absolute):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    shard_path = model_dir / "missing.safetensors"
    file_name = str(shard_path.resolve()) if absolute else shard_path.name

    with pytest.raises(ValueError, match="not in the model directory allowlist"):
        resolve_safetensors_path(model_dir, file_name, frozenset())
