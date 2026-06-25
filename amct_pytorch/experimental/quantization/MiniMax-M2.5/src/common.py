# Copyright (c) 2025 Advanced Micro Devices, Inc.
# Modifications Copyright (c) 2026 Modelbest and Huawei
#
# This file is adapted from upstream open-source code.
# Licensed under the MIT License.

import fnmatch
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

ALPHA = 0.8
SCALE_CLAMP_MIN = 1e-3
GROUP_SIZE = 32
NUM_ATTENTION_HEADS = 48
NUM_KEY_VALUE_HEADS = 8
DECODER_LAYERS_PATH = "model.layers"
ATTN_IMPLEMENTATION = "eager"

SCALING_LAYERS = [
    {
        "prev_op": "input_layernorm",
        "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        "inp": "self_attn.q_proj",
        "module2inspect": "self_attn",
    },
    {
        "prev_op": "self_attn.v_proj",
        "layers": ["self_attn.o_proj"],
        "inp": "self_attn.o_proj",
    },
]

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


def get_primary_device(model) -> torch.device:
    if hasattr(model, "device"):
        device = model.device
        return device if isinstance(device, torch.device) else torch.device(device)
    return next(model.parameters()).device


def index_to_device(index: int) -> torch.device:
    accelerator = get_available_accelerator()
    if accelerator == "cpu":
        return torch.device("cpu")
    return torch.device(f"{accelerator}:{index}")


def load_index(model_dir: str) -> dict:
    with open(
        os.path.join(model_dir, "model.safetensors.index.json"), "r", encoding="utf-8"
    ) as f:
        return json.load(f)


def load_tensor(model_dir: str, name: str):
    index = load_index(model_dir)
    shard = index["weight_map"].get(name)
    if shard is None:
        return None
    with safe_open(os.path.join(model_dir, shard), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def iter_model_files(model_dir: str):
    root = Path(model_dir)
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path
