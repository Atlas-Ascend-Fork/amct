# coding=utf-8
# Adapted from
# https://github.com/ModelTC/LightCompress/blob/main/llmc/data/dataset/specified_preproc.py
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

import torch
from tqdm import tqdm
from datasets import load_dataset, load_from_disk


def pileval_awq(calib_dataset, tokenizer, n_samples, seq_len):
    dataset = calib_dataset.shuffle(seed=42)
    samples = []
    total_tokens = 0
    target_tokens = n_samples * seq_len
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
        total_tokens += sample.shape[1]
        if total_tokens >= target_tokens:
            break
    if not samples:
        raise ValueError("No valid pileval samples were collected.")
    samples = torch.cat(samples, dim=1)
    n_split = samples.shape[1] // seq_len
    if n_split < n_samples:
        raise ValueError(
            f"Not enough pileval tokens to build {n_samples} samples with seq_len={seq_len}; "
            f"only got {n_split} samples."
        )
    samples = [samples[:, i * seq_len: (i + 1) * seq_len] for i in range(n_samples)]
    return samples


def get_pileval(tokenizer, n_samples, seq_len=512):
    testdata = load_dataset(
        'mit-han-lab/pile-val-backup', split='validation'
    )

    samples = pileval_awq(testdata, tokenizer, n_samples, seq_len)

    return samples


def get_wikitext2(tokenizer):
    testdata = load_dataset(
        'wikitext', 'wikitext-2-raw-v1', split='test'
    )

    testenc = tokenizer(
        '\n\n'.join(testdata['text']), return_tensors='pt'
    )
    return testenc


def get_wiki_inputs(tokenizer, seq_len=4096):
    testenc = get_wikitext2(tokenizer).input_ids
    wiki_inputs = []
    nsamples = testenc.numel() // seq_len
    bs = 1

    for i in tqdm(range(0, nsamples, bs), desc='Chunking wikitext2'):
        j = min(i + bs, nsamples)
        inputs = testenc[:, (i * seq_len): (j * seq_len)].cpu()
        inputs = inputs.reshape(j - i, seq_len)
        wiki_inputs.append(inputs)
    return wiki_inputs

