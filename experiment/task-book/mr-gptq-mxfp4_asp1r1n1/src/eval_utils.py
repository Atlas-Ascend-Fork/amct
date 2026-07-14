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


def report_coverage_memory(model, skip_layers=("lm_head",),
                           quant_bits=4.25, bf16_bits=16):
    """统计量化层覆盖率与权重显存降低（在 amct.quantize 之前调用，此时 Linear 完整）。

    - 覆盖率 = 被量化的 nn.Linear 数 / 全部 nn.Linear 数（skip_layers 不量化）。
    - 显存：MXFP4 权重 4.25 bit/param（4bit 元素 + E8M0 8bit/32 组），对比 bf16 16 bit。
    """
    n_total = n_quant = 0
    w_all = w_quant = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            n_total += 1
            wp = m.weight.numel()
            w_all += wp
            if not any(s in name for s in skip_layers):
                n_quant += 1
                w_quant += wp
    cov = 100.0 * n_quant / max(n_total, 1)
    orig = w_all * bf16_bits
    new = (w_all - w_quant) * bf16_bits + w_quant * quant_bits
    red = 100.0 * (1 - new / orig)
    print(f"[coverage] 量化 nn.Linear: {n_quant}/{n_total} = {cov:.1f}%  (skip={list(skip_layers)})")
    print(f"[memory] Linear 权重 {bf16_bits}→{quant_bits} bit；整体权重显存降低 = {red:.1f}%")
    return cov, red


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
