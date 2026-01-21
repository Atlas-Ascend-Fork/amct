# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache License for more details at
# http://www.apache.org/licenses/LICENSE-2.0
# ----------------------------------------------------------------------------

import random

import logging
import datasets
import torch
import transformers
import lm_eval
import numpy as np

from tqdm import tqdm
from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer


def seed_everything(seed=0) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    transformers.set_seed(seed)


def create_logger(dist_rank=0, name=''):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def build_enc(model_path):
    enc = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True
    )
    return enc


def get_llama(model_path, hf_token=None):
    config = transformers.LlamaConfig.from_pretrained(
        model_path, attn_implementation='eager')
    model = transformers.LlamaForCausalLM.from_pretrained(
        model_path, torch_dtype='auto', config=config,
        use_auth_token=hf_token, low_cpu_mem_usage=True)
    model.seqlen = 2048
    print(f'---> Loading {model_path} Model with seq_len: {model.seqlen}')
    return model


def get_wikitext2(nsamples, seed, seqlen, tokenizer, eval_mode=False):
    if eval_mode:
        testdata = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1')["test"] 
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
        return testenc
    else:
        traindata = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1')["train"]
        traindata = traindata.filter(lambda x: len(x) > 0)
        traindata = traindata.map(lambda x: {'text': x['text'].strip()})
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')    
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def test_acc(model, tokenizer, tasks, batch_size, logger):
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    logger.info(f"Evaluating {tasks}...")
    results = lm_eval.simple_evaluate(hflm, tasks=tasks)['results']

    for task_name in tasks:
        result = results[task_name]
        acc = round(result.get('acc_norm,none', result['acc,none']) * 100, 2)
        results[task_name] = acc
        logger.info(f"acc: {acc}%")
    
    metric_vals = {task: result for task, result in results.items()}
    metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 2)
    logger.info(f"ACC: {metric_vals}")


@torch.no_grad()
def test_ppl(model, testenc, dataset_name):
    print(f'Evaluating ppl for {dataset_name}')
    model.eval()

    max_length = 2048
    dev = next(model.parameters()).device

    testenc = testenc.input_ids
    testenc = testenc.to(dev)
    nsamples = testenc.numel() // max_length

    nlls = []
    for i in tqdm(range(nsamples)):
        batch = testenc[:, (i * max_length): ((i + 1) * max_length)]
        lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * max_length): ((i + 1) * max_length)][:, 1:].to(dev)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * max_length
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * max_length))
    return ppl.item()


def eval_total(model, tokenizer, ppl_eval_dataset, logger):
    # Evaluate ACC
    tasks = ["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "lambada_openai"]
    test_acc(model, tokenizer, tasks, 16, logger)

    # Evaluate PPL
    dataset_ppl = test_ppl(model, ppl_eval_dataset, "wikitext2")
    logger.info(f"PPL score: {dataset_ppl}")
    logger.info("All done!")