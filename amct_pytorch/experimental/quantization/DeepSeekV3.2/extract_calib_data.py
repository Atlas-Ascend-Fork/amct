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

import shutil
import functools
import os
import json
import torch
from torch import nn
from tqdm import tqdm
from datasets import load_dataset
from loguru import logger
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights

from cores.utils import args_utils as args_utils
from cores.models.deepseek_v3_2.quant_utils import get_float_block
from cores.utils.utils import load_embed_state_dict
from pp.forward.infer import prepare_layer
from cores.models.deepseek_v3_2.indexer import ModelArgs


def pileval_awq(calib_dataset, tokenizer, n_samples, seq_len):
    dataset = calib_dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data['text']
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > seq_len:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    samples = torch.cat(samples, dim=1)
    n_split = samples.shape[1] // seq_len
    samples = [samples[:, i * seq_len: (i + 1) * seq_len] for i in range(n_split)]
    return samples


def get_pileval(tokenizer, n_samples, seq_len=512):
    testdata = load_dataset(
        'mit-han-lab/pile-val-backup', split='validation'
    )

    samples = pileval_awq(testdata, tokenizer, n_samples, seq_len)

    return samples


@torch.no_grad()
def get_act_stat(args, model, samples, tokenizer, layer_idx, output_dir, dtype=torch.bfloat16, nsamples=45, bs=1,
                 num_npus=1):
    model.eval()
    device = torch.device(f'npu:{layer_idx % num_npus}')
    act_stat = {}

    def stat_tensor(name, tensor, type):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        key_name = f"{name}.{type}"
        if key_name in act_stat:
            act_stat[key_name].append(tensor)
        else:
            act_stat[key_name] = [tensor]

    def stat_input_hook(m, x, y, name):
        stat_tensor(name, x, 'inp')
        stat_tensor(name, y, 'out')

    hooks = []
    for name, m in model.named_modules():
        if 'DeepseekV3DecoderLayer' in model.__class__.__name__:
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

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

    outs = []
    layers = model.model.layers

    if layer_idx == -1:

        load_embed_state_dict(model, args.model)
        layers[0] = get_float_block(args.model, 0, 'cpu', model_args=args.model_args)
        layers[0] = layers[0].bfloat16()

        # record attention and position_ids
        layers[0] = Catcher(layers[0], outs)

        with torch.no_grad():
            # Loop through each batch
            for i in tqdm(range(0, nsamples, bs)):
                logger.info(f'index : {(i + 1) // bs}/{nsamples // bs}')
                # Calculate end index
                j = min(i + bs, nsamples)
                inputs = samples[i].cpu()
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
    else:
        layer = get_float_block(args.model, layer_idx, 'cpu', model_args=args.model_args)
        layer = prepare_layer(args, layer, layer_idx)
        layer = layer.to(device)
        inps = torch.load(os.path.join(output_dir, f'layer_{layer_idx - 1}_out.pkl'),
                          weights_only=False, map_location=device)
        attention_mask = torch.load(
            os.path.join(output_dir, 'attention_mask.pkl'), weights_only=False, map_location=device)
        position_ids = torch.load(
            os.path.join(output_dir, 'position_ids.pkl'), weights_only=False, map_location=device)
        for i in tqdm(range(len(inps)), desc='obtain activation stat'):
            data = inps[i].to(device)
            out = layer(data.to(device), position_ids=position_ids,
                        attention_mask=attention_mask)
            out = out[0]
            outs.append(out.to('cpu'))
        torch.save(outs, os.path.join(
            output_dir, f'layer_{layer_idx}_out.pkl'))
        layers[layer_idx] = None
        layer.to('cpu')
        del layer
        torch.npu.empty_cache()
    for h in hooks:
        h.remove()
    torch.save(act_stat, os.path.join(
        output_dir, f'layer_{layer_idx}_act_stat.pkl'))
    return act_stat


def dump(input_dir, output_dir, num_layer):
    os.makedirs(output_dir, exist_ok=True)
    for layer_idx in range(num_layer):
        inp_tensors = torch.load(os.path.join(input_dir, f'layer_{layer_idx - 1}_out.pkl'))
        out_tensors = torch.load(os.path.join(input_dir, f'layer_{layer_idx}_out.pkl'))
        for batch_idx, (inp, out) in enumerate(zip(inp_tensors, out_tensors)):
            torch.save(inp, os.path.join(output_dir, f'layer_{layer_idx}_batch_{batch_idx}_inp.pth'))
            torch.save(out, os.path.join(output_dir, f'layer_{layer_idx}_batch_{batch_idx}_out.pth'))
    shutil.rmtree(input_dir)


if __name__ == '__main__':
    args = args_utils.parser_gen()
    num_npus = torch.npu.device_count()

    config = './cores/models/deepseek_v3_2/config_671B_v3.2.json'
    with open(config) as f:
        model_args = ModelArgs(**json.load(f))
    args.model_args = model_args

    config = AutoConfig.from_pretrained(
        args.model, trust_remote_code=True)
    # load empty model
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True)
    samples = get_pileval(tokenizer, args.nsamples, seq_len=args.seq_len)
    nsamples = len(samples)
    os.makedirs(args.output_dir, exist_ok=True)
    get_act_stat(args, model, samples, tokenizer, -1, args.exp_dir, nsamples=nsamples, num_npus=num_npus)
    for layer_idx in range(len(model.model.layers)):
        get_act_stat(args, model, samples, tokenizer, layer_idx, args.exp_dir, nsamples=nsamples, num_npus=num_npus)
    dump(args.exp_dir, args.output_dir, 61)
