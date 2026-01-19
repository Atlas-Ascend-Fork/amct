# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

import os
import json
from copy import deepcopy
import torch
from loguru import logger
from transformers import AutoConfig
from safetensors import safe_open
from cores.models.deepseek_v3_2.modeling_deepseek_v3_2 import DeepseekV3DecoderLayer, DeepseekV3MLP, \
    _prepare_4d_causal_attention_mask
from cores.quantization.node import ActivationQuantizer
from cores.utils.utils import skip_initialization
from cores.quantization.linear import QuantLinear


class QuantDeepseekV3MLP(torch.nn.Module):
    def __init__(self, args, module: DeepseekV3MLP):
        super().__init__()
        self.args = args
        self.hidden_size = module.hidden_size
        self.intermediate_size = module.intermediate_size
        self.act_fn = module.act_fn
        self.up_proj = QuantLinear(args, module.up_proj)
        self.gate_proj = QuantLinear(args, module.gate_proj)
        self.down_proj = QuantLinear(args, module.down_proj)

        self._ori_mode = False
        self.afq_x = ActivationQuantizer(bits=args.a_bits, \
                                         sym=not (args.a_asym), lac=True, groupsize=-1)

        self.afq_down = ActivationQuantizer(bits=args.a_bits, \
                                            sym=not (args.a_asym), lac=True, groupsize=-1)

    def _trans_forward(self, x):
        x_ts = x
        x_ts = self.afq_x(x_ts)
        up_states = self.up_proj(x_ts)
        gate_states = self.gate_proj(x_ts)

        x_ts_2 = self.act_fn(gate_states) * up_states

        x_ts_2 = self.afq_down(x_ts_2)
        down_states = self.down_proj(x_ts_2)
        return down_states

    def _ori_forward(self, x):
        x = self.act_fn(self.gate_proj._ori_forward(x)) * self.up_proj._ori_forward(x)
        down_states = self.down_proj._ori_forward(x)
        return down_states

    def forward(self, x):
        if self._ori_mode:
            return self._ori_forward(x)
        return self._trans_forward(x)


def apply_quant_to_moe(args, model, shared_expert_bits=None, routed_expert_bits=None):
    for name, mod in model.named_children():
        if name in ["mlp"] and hasattr(mod, "experts") is False:
            _new_args = deepcopy(args)
            _new_args.w_bits = 8
            _new_args.lwc = False
            _new_args.lac = False
            new_args = _new_args
            setattr(model, name, QuantDeepseekV3MLP(new_args, mod))
            break

        if name in ["experts"]:
            for i in range(len(mod)):
                new_args = args
                if routed_expert_bits is not None:
                    _new_args = deepcopy(args)
                    _new_args.w_bits = routed_expert_bits
                    new_args = _new_args
                mod[i] = QuantDeepseekV3MLP(new_args, mod[i])

        if name in ["shared_experts"]:
            logger.info(f"{name} - replace to QuantDeepseekV3MLP")
            if shared_expert_bits is not None:
                _new_args = deepcopy(args)
                _new_args.w_bits = shared_expert_bits
                new_args = _new_args
            setattr(model, name, QuantDeepseekV3MLP(new_args, mod))

        if len(list(mod.children())) > 0:
            apply_quant_to_moe(args, mod, shared_expert_bits=shared_expert_bits, routed_expert_bits=routed_expert_bits)

    return model


def apply_quant_to_mla(args, model, cls):
    skip_initialization()
    for name, mod in model.named_children():
        if name in ["self_attn"]:
            setattr(model, name, cls(args, mod))
        if len(list(mod.children())) > 0:
            apply_quant_to_mla(args, mod, cls)
    return model


def load_layer_weight(model_dir, rank, weight_map, layer_idx):
    prefix = "model.layers.%d." % layer_idx
    # find files need to load
    file_list = []
    for k, v in weight_map.items():
        if not k.startswith(prefix):
            continue
        file_list.append(v)
    file_list = list(set(file_list))
    #
    state_dict = {}
    for file in file_list:
        with safe_open(os.path.join(model_dir, file), framework="pt", device="cpu") as f:
            for k in f.keys():
                if not k.startswith(prefix):
                    continue
                v = f.get_tensor(k)
                state_dict[k] = v
    # remove prefix of layer:
    state_dict_new = {}
    for k, v in state_dict.items():
        state_dict_new[k[len(prefix):]] = v
    state_dict = state_dict_new
    return state_dict


def load_layer(model_dir, rank, config, weight_map, layer_idx, cls=DeepseekV3DecoderLayer):
    decoder_layer = cls(config, layer_idx, is_nextn=layer_idx == 61)

    state_dict = load_layer_weight(model_dir, rank, weight_map, layer_idx)
    try:
        state_dict.pop('self_attn.rotary_emb.inv_freq')
    except:
        logger.info("not find self_attn.rotary_emb.inv_freq in state_dict")

    decoder_layer.load_state_dict(state_dict)
    state_dict = None  # or it will be still in GPU mem
    decoder_layer.eval()
    return decoder_layer


def get_float_block(model_path, idx, dev, model_args=None):
    config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=True)
    config.model_args = model_args
    with open(f"{model_path}/model.safetensors.index.json") as f:
        weight_map = json.load(f)
    weight_map = weight_map["weight_map"]
    block = load_layer(model_path, dev, config, weight_map, idx)
    return block
