# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
import time
import torch
import tqdm
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM


def build_enc(model_path):
    enc = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, trust_remote_code=True
        )
    return enc


def get_llama2(model_path, seqlen=2048):
    print(f'Getting official pretrained {model_path}')

    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, offload_folder="offload/")

    model.seqlen = seqlen
    enc = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, trust_remote_code=True
    )
    return model, enc


def get_qwen(model_path, seqlen=2048):
    print(f'Getting official pretrained {model_path}')

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
    
    model.seqlen = seqlen
    enc = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, trust_remote_code=True
    )
    return model, enc


def get_test_dataset(enc, seqlen):
    print('Loading dataset: Wikitext2')
    testenc = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    testenc = enc("\n\n".join(testenc["text"]), return_tensors="pt")
    
    return testenc


def get_calib_dataset(tokenizer=None, n_samples=512, block_size=512):
    print('Loading dataset: pileval')
    dataset = load_dataset("mit-han-lab/pile-val-backup")
    dataset = dataset["validation"]
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ]


def infer_model(model, testenc):
    test_start_time = time.time()
    with torch.no_grad():
        model(testenc[:, :model.seqlen].to(next(model.parameters()).device))
    test_end_time = time.time()
    total_time = test_end_time - test_start_time
    print('Calibration time taken: ', total_time // 60, 'min ', total_time % 60, 's')


def test_ppl(model, testenc):
    nlls = []
    nsamples = testenc.numel() // model.seqlen
    test_start_time = time.time()
    for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
            model.device
        )
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    test_end_time = time.time()

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    total_time = test_end_time - test_start_time
    print('Test time taken: ', total_time // 60, 'min ', total_time % 60, 's')
    print('Score: ', ppl.item())