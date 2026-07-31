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
from datasets import load_dataset


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
    samples = [samples[:, i * seq_len : (i + 1) * seq_len] for i in range(n_samples)]
    return samples


def get_pileval(tokenizer, n_samples, seq_len=512, data_dir=None):
    # AMCT_LOCAL_PILEVAL_SUPPORT
    if data_dir:
        from pathlib import Path
        import random

        data_path = Path(data_dir)
        if data_path.is_dir():
            data_path = data_path / "pileval.txt"

        if not data_path.is_file():
            raise FileNotFoundError(
                f"Local calibration text not found: {data_path}"
            )

        dataset_text = data_path.read_text(encoding="utf-8")
        encoded = tokenizer(dataset_text, return_tensors="pt").input_ids

        token_count = encoded.shape[1]
        if token_count < seq_len:
            raise ValueError(
                f"Local calibration data has {token_count} tokens, "
                f"but seq_len={seq_len}"
            )

        rng = random.Random(0)
        max_start = token_count - seq_len
        samples = []

        for _ in range(n_samples):
            start = rng.randint(0, max_start) if max_start > 0 else 0
            samples.append(encoded[:, start:start + seq_len].cpu())

        return samples

    testdata = load_dataset('mit-han-lab/pile-val-backup', split='validation')

    samples = pileval_awq(testdata, tokenizer, n_samples, seq_len)

    return samples


def get_wikitext2(tokenizer, data_dir=None):
    if data_dir:
        from pathlib import Path

        data_path = Path(data_dir)
        if data_path.is_dir():
            data_path = data_path / "wikitext2_test.txt"
        if not data_path.is_file():
            raise FileNotFoundError(f"Local evaluation text not found: {data_path}")
        dataset_text = data_path.read_text(encoding="utf-8")
    else:
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        dataset_text = "\n\n".join(testdata["text"])

    return tokenizer(dataset_text, return_tensors="pt")

def get_wiki_inputs(tokenizer, seq_len=4096, data_dir=None):
    testenc = get_wikitext2(tokenizer, data_dir=data_dir).input_ids
    wiki_inputs = []
    nsamples = testenc.numel() // seq_len
    bs = 1

    for i in tqdm(range(0, nsamples, bs), desc='Chunking wikitext2'):
        j = min(i + bs, nsamples)
        inputs = testenc[:, (i * seq_len) : (j * seq_len)].cpu()
        inputs = inputs.reshape(j - i, seq_len)
        wiki_inputs.append(inputs)
    return wiki_inputs
