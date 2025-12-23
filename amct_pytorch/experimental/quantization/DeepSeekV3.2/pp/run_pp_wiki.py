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
import torch.nn as nn
from loguru import logger
import torch
from tqdm import tqdm
from datasets import load_dataset, load_from_disk


def get_wikitext2(tokenizer):
    testdata = load_dataset(
        'wikitext', 'wikitext-2-raw-v1', split='test'
    )

    testenc = tokenizer(
        '\n\n'.join(testdata['text']), return_tensors='pt'
    )
    return testenc.input_ids


def get_wiki_inputs(testenc, seq_len=4096):
    wiki_inputs = []
    nsamples = testenc.numel() // seq_len
    bs = 1

    for i in tqdm(range(0, nsamples, bs), desc='Chunking wikitext2'):
        # Calculate end index
        j = min(i + bs, nsamples)
        # Prepare inputs and move to npu
        inputs = testenc[:, (i * seq_len): (j * seq_len)].cpu()
        inputs = inputs.reshape(j - i, seq_len)
        wiki_inputs.append(inputs)
    return wiki_inputs


def wikitext2_ppl(testenc, data_dir, seq_len=4096):
    nsamples = testenc.numel() // seq_len
    bs = 1
    nlls = []
    for i in tqdm(range(0, nsamples, bs), desc='Running wikitext2'):
        j = min(i + bs, nsamples)
        inputs = testenc[:, (i * seq_len): (j * seq_len)].cpu()
        inputs = inputs.reshape(j - i, seq_len)

        output = torch.load(os.path.join(
            data_dir, f'{i}.pkl'), weights_only=True)
        # Forward pass through the model
        lm_logits = output[0].cpu()

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )
        if i == 0:
            logger.info(f"iter {i}:loss {loss}")
        # Calculate negative log likelihood
        neg_Log_likelihood = loss.float() * seq_len * (j - i)
        # Append to list of negative log likelihoods
        nlls.append(neg_Log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seq_len))
    # Empty NPU cache to save memory
    testenc.cpu()
    torch.npu.empty_cache()
    logger.info(ppl)
    return ppl.item()
