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
import functools
from collections import defaultdict
import torch

from cores.models.deepseek_v3_2.quant_utils import _prepare_4d_causal_attention_mask


def load_block_inps_outs(save_dir, layer_idx):
    total_bs = 20
    cached_inps = []
    cached_outs = []
    for bs in range(0, total_bs):
        inp_file = os.path.join(save_dir, f"layer_{layer_idx}_batch_{bs}_inp.pth")
        out_file = os.path.join(save_dir, f"layer_{layer_idx}_batch_{bs}_out.pth")
        if not os.path.exists(inp_file):
            raise ValueError(f"{inp_file} not found!")
        if not os.path.exists(out_file):
            raise ValueError(f"{out_file} not found!")

        inp = torch.load(inp_file)
        out = torch.load(out_file)

        cached_inps.append(inp)
        cached_outs.append(out)

    cached_inps = torch.cat(cached_inps)
    cached_outs = torch.cat(cached_outs)

    return cached_inps, cached_outs


def cache_input_hook(m, x, y, name, feat_dict):
    x = x[0]
    x = x.detach().cpu()
    feat_dict[name].append(x)


def cache_output_hook(m, x, y, name, feat_dict):
    y = y[0]
    y = y.detach().cpu()
    feat_dict[name].append(y)


def layer_forward(layer, inps, bs=1):
    seq_length = inps.shape[1]
    inps = inps.to(next(layer.parameters()).device)
    with torch.no_grad():
        for j in range(inps.shape[0] // bs):
            index = j * bs
            attention_mask = None
            attention_mask_batch = _prepare_4d_causal_attention_mask(
                attention_mask,
                (bs, seq_length),
                torch.randn_like(inps),
                0,
            )
            attention_mask_batch = attention_mask_batch.to(next(layer.parameters()).device)
            layer(inps[index:index + bs, ], attention_mask=attention_mask_batch)[0]


def get_self_attn_inps_outs(layer, inps):
    bs = 1
    outs = torch.zeros_like(inps)
    with torch.no_grad():
        for j in range(inps.shape[0] // bs):
            index = j * bs
            outs[index:index + bs, ] = layer.input_layernorm(
                inps[index:index + bs, ].to(layer.input_layernorm.weight.device))

    return outs


def get_linear_inps_outs(layer, inps):
    handles = []
    input_feat = defaultdict(list)
    for name, mod in layer.named_modules():
        if isinstance(mod, torch.nn.Linear):
            handles.append(
                mod.register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )

    layer_forward(layer, inps, bs=1)
    for h in handles:
        h.remove()
    return input_feat


def get_mla_moe_inputs(layer, fp_inps, dev="npu:0"):
    """
    hidden_states: block's input

    """
    layer.to(torch.bfloat16).to(dev)
    fp_inps = fp_inps.to(torch.bfloat16)
    fp_inps = fp_inps.to(dev)

    def pre_forward(hidden_states, attention_mask):
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        return hidden_states.to(dev)

    fp_outs = torch.zeros_like(fp_inps)
    cali_bsz = 1
    seq_length = fp_inps.shape[1]

    with torch.no_grad():
        for j in range(0, fp_inps.shape[0]):
            index = j * cali_bsz
            attention_mask = None
            attention_mask_batch = _prepare_4d_causal_attention_mask(
                attention_mask,
                (cali_bsz, seq_length),
                torch.randn_like(fp_inps),
                0,
            )

            fp_outs[index:index + cali_bsz, ] = pre_forward(fp_inps[index:index + cali_bsz, ],
                                                            attention_mask=attention_mask_batch)

    layer.to('cpu')
    torch.npu.empty_cache()
    return fp_outs
