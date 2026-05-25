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
"""剪枝示例公用工具：加载 HF 本地模型 / 准备校准数据 / 评估。

遵循 AMCT examples 约定：库本体不下载模型，调用方通过 ``--model_path`` 传入本地路径，
此处只负责 ``AutoModel*.from_pretrained(local_path)``。
"""

import time

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


def get_seqcls(model_path, num_labels=2):
    """加载 HF 本地序列分类模型 + tokenizer。"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=num_labels, torch_dtype=torch.float32,
    )
    return model, tokenizer


def get_causal_lm(model_path, dtype=torch.float16, seqlen=2048):
    """加载 HF 本地因果语言模型 + tokenizer（与 AMCT awq/utils.get_qwen 同型）。"""
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map="auto")
    model.seqlen = seqlen
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    return model, tokenizer


def get_boolq_calib(tokenizer, n_samples=128, max_length=256, split="train"):
    """构造 BoolQ 校准数据列表，可直接传给 amct.prune(..., data=...)."""
    dataset = load_dataset("boolq", split=split).shuffle(seed=42).select(range(n_samples))
    batches = []
    for item in dataset:
        enc = tokenizer(
            item["question"],
            item["passage"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batches.append({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        })
    return batches


def get_calib_dataset(tokenizer, n_samples=512, block_size=256, ds_name="mit-han-lab/pile-val-backup"):
    """通用 LM 校准数据集（沿用 AMCT awq/minmax utils 的做法）。"""
    dataset = load_dataset(ds_name)["validation"].shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        text = data["text"].strip()
        ids = tokenizer.encode(text)
        if len(ids) > 512:
            continue
        sample = torch.tensor([ids])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    cat = torch.cat(samples, dim=1)
    n_split = cat.shape[1] // block_size
    return [cat[:, i * block_size:(i + 1) * block_size] for i in range(n_split)]


def infer_model(model, testenc):
    test_start = time.time()
    with torch.no_grad():
        model(testenc[:, :model.seqlen].to(next(model.parameters()).device))
    total = time.time() - test_start
    print("Calibration time taken: ", total // 60, "min ", total % 60, "s")


def eval_seqcls_accuracy(model, batches, device="cpu"):
    """简单序列分类准确率评估。"""
    model = model.to(device).eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in batches:
            labels = batch.pop("labels", None)
            if labels is None:
                continue
            inputs = {k: v.to(device) for k, v in batch.items()}
            out = model(**inputs)
            logits = out.logits if hasattr(out, "logits") else out
            preds = torch.argmax(logits, dim=-1)
            correct += int((preds.cpu() == labels.cpu()).sum())
            total += int(labels.numel())
    return correct / max(total, 1)
