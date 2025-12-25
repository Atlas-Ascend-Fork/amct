# coding=utf-8
# Adapted from
# https://github.com/ruikangliu/FlatQuant/blob/main/flatquant/utils.py
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Copyright (c) 2024 ruikangliu. Licensed under MIT.
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

import random
import numpy as np
import torch
import transformers
from compressed_tensors.utils.safetensors_load import get_weight_mappings
from safetensors import safe_open

from accelerate import dispatch_model


def skip(*args, **kwargs):
    # This is a helper function to save time during the initialization!
    pass


def skip_initialization():
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip


def seed_everything(seed=0) -> None:
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    transformers.set_seed(seed)


def get_device_map(block, device, num_npus=1):
    if hasattr(block.mlp, 'experts'):
        experts_num = len(block.mlp.experts)
        device_map = {f'mlp.experts.{i}': f'npu:{i % num_npus}' for i in range(experts_num)}
        device_map['self_attn'] = device
        device_map['self_attn.q_a_proj'] = device
        device_map['mlp.gate'] = device
        device_map['mlp.shared_experts'] = device
        device_map['input_layernorm'] = device
        device_map['post_attention_layernorm'] = device
    else:
        device_map = {'': device}

    dispatch_model(block, device_map)
    return device_map


def get_layer_state_dict(model_path, layer_name):
    weight_mappings = get_weight_mappings(model_path)
    state_dict = {}
    for weight_name, file_with_weight_name in weight_mappings.items():
        if layer_name in weight_name:
            with safe_open(file_with_weight_name, framework="pt", device="cpu") as f:
                weight = f.get_tensor(weight_name)
                state_dict[weight_name.replace(layer_name, '')] = weight
    return state_dict


def load_embed_state_dict(model, model_path, strict=True):
    model.model.embed_tokens.to_empty(device="cpu")
    model.model.embed_tokens.load_state_dict(
        get_layer_state_dict(model_path, 'model.embed_tokens.'), strict=strict)

    model.model.norm.to_empty(device="cpu")
    model.model.norm.load_state_dict(
        get_layer_state_dict(model_path, 'model.norm.'), strict=strict)

    model.lm_head.to_empty(device="cpu")
    model.lm_head.load_state_dict(
        get_layer_state_dict(model_path, 'lm_head.'), strict=strict)


def load_layer_state_dict(layer, layer_idx, model_path, strict=True):
    layer.to_empty(device="cpu")
    state_dict = get_layer_state_dict(
        model_path, f'model.layers.{layer_idx}.')
    if "self_attn.rotary_emb.inv_freq" in state_dict:
        state_dict.pop("self_attn.rotary_emb.inv_freq")
    layer.load_state_dict(state_dict, strict=strict)
