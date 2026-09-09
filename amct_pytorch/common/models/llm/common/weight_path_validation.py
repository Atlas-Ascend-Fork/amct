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

import os
from pathlib import Path


def collect_safetensors_files(model_path):
    """Collect direct, regular safetensors files under model_path."""
    model_root = Path(model_path).resolve()
    if not model_root.is_dir():
        raise NotADirectoryError(f"Model path is not a directory: {model_path}")

    safetensors_files = set()
    with os.scandir(model_root) as entries:
        for entry in entries:
            if not entry.is_file(follow_symlinks=False):
                continue
            if Path(entry.name).suffix != ".safetensors":
                continue
            safetensors_files.add(Path(entry.path).resolve())

    if not safetensors_files:
        raise FileNotFoundError(
            f"No .safetensors files found in model directory: {model_path}"
        )
    return frozenset(safetensors_files)


def resolve_safetensors_path(model_path, file_name, safetensors_files):
    """Resolve a safetensors shard contained in the collected file allowlist."""
    if not isinstance(file_name, str) or not file_name:
        raise ValueError("Safetensors file name must be a non-empty string")
    if "\x00" in file_name:
        raise ValueError("Safetensors file name must not contain NUL")

    input_path = Path(file_name)
    if input_path.suffix != ".safetensors":
        raise ValueError("Weight file must use the .safetensors suffix")

    model_root = Path(model_path).resolve()
    weight_path = (
        input_path.resolve()
        if input_path.is_absolute()
        else (model_root / input_path).resolve()
    )

    if weight_path not in safetensors_files:
        raise ValueError(
            f"Safetensors file is not in the model directory allowlist: {file_name}"
        )
    return weight_path


def validate_weight_map(model_path, weight_map, safetensors_files):
    """Validate tensor-to-shard mappings loaded from a model index."""
    if not isinstance(weight_map, dict):
        raise ValueError("weight_map must be a dictionary")
    for weight_name, file_name in weight_map.items():
        if not isinstance(weight_name, str) or not weight_name:
            raise ValueError("weight_map keys must be non-empty strings")
        resolve_safetensors_path(model_path, file_name, safetensors_files)
