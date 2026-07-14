# -*- coding: UTF-8 -*-
"""模型加载、校准、WikiText2 PPL 评测。

要点：
- 单卡加载（不用 device_map="auto"），否则 8B 会被拆到 npu:0/npu:1，
  量化时报 "Expected all tensors on the same device"；
- test_ppl 与 ofmr 示例算法一致（滑窗、seqlen 分块、交叉熵累积 → exp 平均 NLL）。
"""
import time

import torch
import torch_npu  # noqa: F401  必须导入以启用 .npu()
import torch.nn as nn
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model(model_path, seqlen=2048, dtype=torch.float16, device="npu:0"):
    """单卡加载模型 + tokenizer。dtype 默认 float16（与已验证 baseline 一致）。"""
    print(f"Loading model: {model_path} (seqlen={seqlen}, dtype={dtype})")
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    model.seqlen = seqlen
    model = model.eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True
    )
    return model, tokenizer


@torch.no_grad()
def calibrate(model, calib_blocks):
    """喂一次校准前向：把校准块拼成一个 batch 过模型。

    对 GPTQ：该次前向内累积 Hessian 并当场做 Cholesky 误差补偿（batch_num=1）。
    calib_blocks: list[[1, block_size]] → 堆成 [n_blocks, block_size]。
    """
    calib = torch.cat(calib_blocks, dim=0)
    print(f"[calib] shape={tuple(calib.shape)}")
    t0 = time.time()
    model(calib[:, : model.seqlen].to(model.device))
    torch_npu.npu.empty_cache()
    print(f"[calib] done in {time.time() - t0:.1f}s")


@torch.no_grad()
def test_ppl(model, testenc):
    """WikiText2 PPL。testenc: [1, total_tokens] 的 input_ids（已在目标 device）。"""
    seqlen = model.seqlen
    nsamples = testenc.numel() // seqlen
    nlls = []
    t0 = time.time()
    for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, i * seqlen : (i + 1) * seqlen].to(model.device)
        logits = model(batch).logits
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[:, i * seqlen : (i + 1) * seqlen][:, 1:]
        loss = nn.CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        nlls.append(loss.float() * seqlen)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
    dt = time.time() - t0
    print(f"Test time: {dt // 60:.0f}min {dt % 60:.1f}s | nsamples={nsamples}")
    print(f"Score: {ppl.item()}")
    return ppl.item()
