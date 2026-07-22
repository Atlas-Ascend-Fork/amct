# Copyright (c) 2025 Advanced Micro Devices, Inc.
# Modifications Copyright (c) 2026 Modelbest and Huawei
#
# This file is adapted from upstream open-source code.
# Licensed under the MIT License.

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
