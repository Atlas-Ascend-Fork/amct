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
#
# The exported checkpoint follows AMD Quark release/0.13 format conventions.
# Reference:
# https://github.com/amd/Quark/blob/release/0.13/quark/torch/export/main_export/quant_config_parser.py
# Copyright (c) 2023 Advanced Micro Devices, Inc.
# Modifications Copyright (c) 2026 Modelbest and Huawei.
# See Third_Party_Open_Source_Software_Notice at the repository root for
# the upstream license text.

"""MXFP4 导出（RTN）所需的架构常量与工具函数。

供 ``export_rtn_mxfp4.py`` 在把 BF16 检查点转换为 Quark 风格 MXFP4 时使用。
"""

import fnmatch
import json

import torch

GROUP_SIZE = 32
NUM_ATTENTION_HEADS = 48
NUM_KEY_VALUE_HEADS = 8
DECODER_LAYERS_PATH = "model.layers"
ATTN_IMPLEMENTATION = "eager"

# 不做 MXFP4 量化、保持原 dtype 的模块（MoE 门控与 lm_head）。
EXCLUDE_PATTERNS = ["*block_sparse_moe.gate*", "*lm_head"]
IGNORE_STATE_DICT_KEYS = {"model.rotary_emb.inv_freq"}


def get_module(model, path: str):
    module = model
    for part in path.split("."):
        module = getattr(module, part)
    return module


def matches_exclude(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def load_device_map(path: str | None) -> str | dict[str, int]:
    if path is None:
        return "auto"
    with open(path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    return {key: int(value) for key, value in loaded.items()}


def get_available_accelerator() -> str:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
