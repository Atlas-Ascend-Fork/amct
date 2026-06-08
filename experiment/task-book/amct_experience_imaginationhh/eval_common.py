# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
# ----------------------------------------------------------------------------
"""HiFloat8 体验任务共用工具：离线友好的数据集/模型加载 + PPL 评估。

设计目标：在无法访问 huggingface.co 的环境中跑通 HiFloat8 量化全流程。
- wikitext2 测试集：优先读本地 parquet（WIKITEXT2_PARQUET 环境变量或默认路径），
  回退到 datasets.load_dataset。
- 模型：直接从本地路径加载。
"""

import os
import time

import torch
import torch.nn as nn
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_WIKITEXT2_PARQUET = "/home/developer/datasets/wikitext2/test.parquet"


def get_model(model_path, seqlen=2048, dtype=torch.float16):
    """加载本地预训练模型与分词器。"""
    print(f"Getting pretrained model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="cpu", torch_dtype=dtype
    )
    model.seqlen = seqlen
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True
    )
    return model, tokenizer


def get_wikitext2_test(tokenizer):
    """加载 wikitext2 测试集并编码为单条长序列（与官方样例一致）。"""
    parquet_path = os.environ.get(
        "WIKITEXT2_PARQUET", DEFAULT_WIKITEXT2_PARQUET
    )
    if os.path.isfile(parquet_path):
        print(f"Loading wikitext2 test set from local parquet: {parquet_path}")
        import pandas as pd

        texts = pd.read_parquet(parquet_path)["text"].tolist()
    else:
        print("Loading wikitext2 test set via datasets.load_dataset")
        from datasets import load_dataset

        texts = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")[
            "text"
        ]
    testenc = tokenizer("\n\n".join(texts), return_tensors="pt")
    return testenc


def get_calib_data(tokenizer, calib_path, device, batch_num=1, seqlen=2048):
    """构造校准数据（quantile/ofmr 算法需要）。

    优先读本地 parquet（calib_path 或 wikitext2 训练集），切分为 batch_num
    个长度为 seqlen 的序列。校准只需少量代表性数据。
    """
    path = calib_path or os.environ.get(
        "WIKITEXT2_PARQUET", DEFAULT_WIKITEXT2_PARQUET
    )
    if path and os.path.isfile(path):
        import pandas as pd

        texts = pd.read_parquet(path)["text"].tolist()
    else:
        from datasets import load_dataset

        texts = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")[
            "text"
        ]
    enc = tokenizer("\n\n".join(texts), return_tensors="pt").input_ids
    total = enc.numel() // seqlen
    n = min(batch_num, total)
    out = []
    for i in range(n):
        seg = enc[:, i * seqlen:(i + 1) * seqlen]
        out.append(seg.to(device))
    return out


@torch.no_grad()
def eval_ppl(model, testenc, device, max_samples=None):
    """逐段计算 wikitext2 困惑度（PPL）。

    Args:
        model: 待评估模型，需含 model.seqlen 属性。
        testenc: get_wikitext2_test 返回的编码结果（含 input_ids）。
        device: 推理设备（如 'npu' / 'cpu'）。
        max_samples: 限制评估段数（用于快速冒烟），None 表示全量。
    """
    input_ids = testenc.input_ids if hasattr(testenc, "input_ids") else testenc
    seqlen = model.seqlen
    nsamples = input_ids.numel() // seqlen
    if max_samples is not None:
        nsamples = min(nsamples, max_samples)

    nlls = []
    loss_fct = nn.CrossEntropyLoss()
    start = time.time()
    for i in tqdm.tqdm(range(nsamples), desc="evaluating ppl"):
        batch = input_ids[:, i * seqlen:(i + 1) * seqlen].to(device)
        logits = model(batch).logits
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = input_ids[:, i * seqlen:(i + 1) * seqlen][:, 1:].to(
            shift_logits.device
        )
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        nlls.append(loss.float() * seqlen)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    elapsed = time.time() - start
    print(
        f"PPL eval: {nsamples} samples, {elapsed:.1f}s, "
        f"score = {ppl.item():.6f}"
    )
    return ppl.item()
