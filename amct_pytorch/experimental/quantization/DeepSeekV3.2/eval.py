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
import torch
from loguru import logger
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights

from pp.run_pp_wiki import get_wiki_inputs, get_wikitext2, wikitext2_ppl
from cores.utils import args_utils as args_utils
from cores.models.deepseek_v3_2.indexer import ModelArgs
from pp.forward.infer import do_embedding_forward, do_one_layer_forward


@torch.no_grad()
def get_act_stat(args, model, samples, layer_idxes, output_dir, dtype=torch.bfloat16, num_npus=1):
    model.eval()
    act_stat = {}
    num_layers = len(model.model.layers)
    layers = model.model.layers
    if layer_idxes[0] == -1:
        do_embedding_forward(args, model, layers, samples, output_dir, dtype)
    else:
        do_one_layer_forward(args, model, layers, layer_idxes, num_layers, output_dir, num_npus=num_npus)
    return act_stat


if __name__ == '__main__':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    args = args_utils.parser_gen()

    config = './cores/models/deepseek_v3_2/config_671B_v3.2.json'
    with open(config) as f:
        model_args = ModelArgs(**json.load(f))
    args.model_args = model_args

    os.makedirs(args.wikitext_final_out, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    config = AutoConfig.from_pretrained(
        args.model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16)
    logger.info(model)

    num_npus = torch.npu.device_count()
    group = 8
    begin, end = 0, 60
    groups = [i for i in range(begin, end + 1)]
    indexes = [groups[i:i + group] for i in range(0, len(groups), group)]
    indexes.insert(0, [-1])
    indexes.append([61])
    for layer_idx in indexes:
        layer_idx = sorted(layer_idx)
        logger.info(layer_idx)

        if layer_idx[0] == -1:
            testenc = get_wikitext2(tokenizer)
            samples = get_wiki_inputs(testenc, seq_len=args.seq_len)
            nsamples = len(samples)
            get_act_stat(args, model, samples, layer_idx, args.output_dir, num_npus=num_npus)

        elif layer_idx[-1] < len(model.model.layers):
            get_act_stat(args, model, samples, layer_idx, args.output_dir, num_npus=num_npus)
        else:
            testenc = get_wikitext2(tokenizer)
            wikitext2_ppl(testenc, args.wikitext_final_out, seq_len=args.seq_len)
        torch.npu.empty_cache()
