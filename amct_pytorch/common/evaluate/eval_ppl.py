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


def wikitext2_ppl(preds, samples, seq_len=4096):
    bs = 1
    nlls = []
    nsamples = len(samples)
    for i, sample in tqdm(enumerate(samples), desc="Evaluating"):
        shift_logits = preds[i]
        shift_labels = sample[:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        )
        if i == 0:
            logger.info(f"iter {i}:loss {loss}")
        neg_log_likelihood = loss.float() * seq_len
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seq_len))
    torch.npu.empty_cache()
    return ppl.item()
