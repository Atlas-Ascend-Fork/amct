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

import types
import os
from loguru import logger
import torch
from torch import nn
from tqdm import tqdm

from cores.models.deepseek_v3_2.quant_dsa import QuantDSA
from cores.utils.utils import get_device_map, load_embed_state_dict
from cores.quantization.utils import load_quant_params
from cores.models.deepseek_v3_2.quant_utils import apply_quant_to_moe, get_float_block, apply_quant_to_mla
from pp.forward.custom import model_causal_forward_only_pre, model_causal_forward_only_post, model_forward_only_post, \
    model_forward_only_pre


def load_file(param_dir, layer_idx):
    param_path = os.path.join(param_dir, f"quant_parameters_{layer_idx}.pth")
    if not os.path.exists(param_path):
        logger.warning(f'{param_path} not found.')
    quant_params = torch.load(param_path)
    return quant_params


def prepare_quant_params(args, layer_idx, param_dir=None):
    if args.cls == "c8":
        if args.train_mode == "block":
            quant_params = load_file(args.mla_param_dir, layer_idx)
            quant_params.update(load_file(args.moe_param_dir, layer_idx))
        else:
            quant_params = load_file(param_dir, layer_idx)
    else:
        quant_params = None
    return quant_params


def prepare_layer(args, layer, layer_idx, cls=QuantDSA):  # use for quantization
    if layer_idx < args.start_block_idx or layer_idx > args.end_block_idx:
        logger.warning \
            (f'Layer idx {layer_idx} not in the range {args.start_block_idx}-{args.end_block_idx}, not replace quant layer.')
        return layer
    if args.train_mode == "mla":
        layer = apply_quant_to_mla(args, layer, cls=cls)
        quant_params = prepare_quant_params(args, layer_idx, args.mla_param_dir)
    elif args.train_mode == "block":
        layer = apply_quant_to_mla(args, layer, cls=cls)
        layer = apply_quant_to_moe(args, layer)
        quant_params = prepare_quant_params(args, layer_idx)
    elif args.train_mode == "moe":
        layer = apply_quant_to_moe(args, layer)
        quant_params = prepare_quant_params(args, layer_idx, args.moe_param_dir)
    elif args.train_mode == "origin":
        logger.info(layer)
        layer.eval()
        return layer
    else:
        raise ValueError(f"Unknown train_mode {args.train_mode}")
    logger.info(layer)
    load_quant_params(layer, quant_params)
    layer.eval()
    return layer


def do_embedding_forward(args, model, layers, samples, output_dir, dtype):
    outs = []

    class Catcher(nn.Module):
        def __init__(self, module, dataset):
            super().__init__()
            self.module = module
            self.dataset = dataset
            self.index = 0
            self.attention_mask = None
            self.position_ids = None

        def forward(self, inp, **kwargs):
            self.dataset.append(inp.to('cpu'))
            self.index += 1
            if self.attention_mask is None:
                self.attention_mask = kwargs['attention_mask']
            if self.position_ids is None:
                self.position_ids = kwargs['position_ids']
            raise ValueError

    load_embed_state_dict(model, args.model)
    layers[0] = get_float_block(args.model, 0, 'cpu', model_args=args.model_args)
    layers[0] = layers[0].bfloat16()
    layers[0] = Catcher(layers[0], outs)
    with torch.no_grad():
        # Loop through each batch
        for bs, inputs in tqdm(enumerate(samples), desc=f"Layer Embedding Processing..."):
            try:
                model(inputs)
            except ValueError:
                pass
    position_ids = layers[0].position_ids
    attention_mask = layers[0].attention_mask

    attention_mask = attention_mask.to(
        dtype) if attention_mask is not None else None
    torch.save(position_ids, os.path.join(output_dir, 'position_ids.pkl'))
    torch.save(attention_mask, os.path.join(
        output_dir, 'attention_mask.pkl'))
    torch.save(outs, os.path.join(output_dir, 'layer_-1_out.pkl'))
    layers[0] = layers[0].module


def do_one_layer_forward(args, model, layers, layer_idxes, num_layers, output_dir, num_npus=1):
    for layer_idx in layer_idxes:
        layer = get_float_block(args.model, layer_idx, 'cpu', model_args=args.model_args)
        # replace quantization layer
        layer = prepare_layer(args, layer, layer_idx)
        logger.info(f'Dispatching {layer_idx} to all NPU.')
        layers[layer_idx] = layer.bfloat16()
        get_device_map(layer, f'npu:{layer_idx % num_npus}', num_npus=num_npus)

    device = torch.device(f'npu:{layer_idxes[0] % num_npus}')
    inps = torch.load(os.path.join(output_dir, f'layer_{layer_idxes[0] - 1}_out.pkl'),
                      weights_only=False, map_location=device)
    attention_mask = torch.load(
        os.path.join(output_dir, 'attention_mask.pkl'), weights_only=False, map_location=device)
    position_ids = torch.load(
        os.path.join(output_dir, 'position_ids.pkl'), weights_only=False, map_location=device)

    outs = [[] for _ in range(len(layer_idxes))]

    for i in tqdm(range(len(inps)), desc='obtain activation stat'):
        inp = inps[i]
        for layer_i, layer_idx in enumerate(layer_idxes):
            layer = layers[layer_idx]
            device = torch.device(f'npu:{layer_idx % num_npus}')
            if layer_idx == num_layers - 1:
                # load lm_head and norm
                load_embed_state_dict(model, args.model)
                model.model.forward_only_pre = types.MethodType(
                    model_forward_only_pre, model.model)
                model.model.forward_only_post = types.MethodType(
                    model_forward_only_post, model.model)
                model.forward_only_pre = types.MethodType(
                    model_causal_forward_only_pre, model)
                model.forward_only_post = types.MethodType(
                    model_causal_forward_only_post, model)
                model.model.norm.to(device)
                model.lm_head.to(device)
                os.makedirs(args.wikitext_final_out, exist_ok=True)

            out = layer(inp.to(device), position_ids=position_ids.to(device),
                        attention_mask=attention_mask.to(device) if attention_mask is not None else None)

            out = out[0]
            outs[layer_i].append(out.to('cpu'))
            inp = out

            if layer_idx == num_layers - 1:
                logits = model.forward_only_post(out)
                torch.save(logits, os.path.join(args.wikitext_final_out, f'{i}.pkl'))
    for layer_idx in layer_idxes:
        layers[layer_idx].to_empty(device="meta")
    for i, layer_idx in enumerate(layer_idxes):
        torch.save(outs[i], os.path.join(
            output_dir, f'layer_{layer_idx}_out.pkl'))
    torch.npu.empty_cache()
