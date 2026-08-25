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

from pathlib import Path


def resolve_safetensors_path(model_path, file_name):
    """Resolve a safetensors shard that must be a file under model_path."""
    if not isinstance(file_name, str) or not file_name:
        raise ValueError("Safetensors file name must be a non-empty string")
    if "\x00" in file_name:
        raise ValueError("Safetensors file name must not contain NUL")

    model_root = Path(model_path).resolve()
    input_path = Path(file_name)
    if (
        input_path.is_absolute()
        or file_name in {".", ".."}
        or "/" in file_name
        or "\\" in file_name
    ):
        raise ValueError("Safetensors file name must be a plain file name")
    weight_path = (model_root / input_path).resolve()

    if weight_path.suffix != ".safetensors":
        raise ValueError("Weight file must use the .safetensors suffix")
    if weight_path.parent != model_root:
        raise ValueError("Safetensors file must stay inside the model directory")
    if not weight_path.is_file():
        raise FileNotFoundError(f"Safetensors file does not exist: {file_name}")
    return weight_path


def validate_weight_map(model_path, weight_map):
    """Validate tensor-to-shard mappings loaded from a model index."""
    if not isinstance(weight_map, dict):
        raise ValueError("weight_map must be a dictionary")
    for weight_name, file_name in weight_map.items():
        if not isinstance(weight_name, str) or not weight_name:
            raise ValueError("weight_map keys must be non-empty strings")
        resolve_safetensors_path(model_path, file_name)
