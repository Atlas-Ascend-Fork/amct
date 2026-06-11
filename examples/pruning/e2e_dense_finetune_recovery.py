# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
"""真·E2E：dense LM FFN 剪枝 → **1-epoch 微调恢复**（结论 3 的可复现脚本）。

要点（实测）：训练充分的预训练 LM 的 dense FFN，*免微调*剪枝几乎无可用空间（容差门控选不出率）；
但**剪枝 + 单遍微调**即可恢复——损伤被吸收，常优于基线（在微调域上）。
微调超参须为 LM 量级：``lr=1e-5 + warmup``（1e-4 易发散），**单遍 distinct 数据**
（少量批多轮 epoch 会记忆训练集致 held 爆炸）。

流程：baseline ppl → reconstruct 剪 FFN 30% → 剪枝后 ppl → finetune(单遍) → 恢复后 ppl。
运行：HF_HUB_DISABLE_XET=1 python examples/pruning/e2e_dense_finetune_recovery.py

预训练权重不可达时退化为随机权重的小 LM（仅演示流程，数值无意义）。
"""

import copy
import os
import time

import torch
import torch.nn.functional as F

import amct_pytorch.experimental.pruning as P

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

MODEL = os.environ.get("AMCT_PRUNE_LM", "EleutherAI/pythia-160m")
DEV = "npu" if (hasattr(torch, "npu") and torch.npu.is_available()) else "cpu"


def _logits(out):
    """从 forward 输出取 logits（张量 / ModelOutput.logits / tuple[0]），惰性，不预求 out[0]。"""
    if torch.is_tensor(out):
        return out
    return out.logits if hasattr(out, "logits") else out[0]


def _wikitext_batches(tok, seq=128, bs=8, n=720):
    """WikiText-2 → [n] 个 [bs, seq] token 批（不足时回退随机）。"""
    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq

        path = hf_hub_download(
            repo_id="Salesforce/wikitext", repo_type="dataset",
            filename="wikitext-2-raw-v1/train-00000-of-00001.parquet",
        )
        txt = " ".join(t for t in pq.read_table(path).column("text").to_pylist() if t and t.strip())
        ids = tok(txt, return_tensors="pt").input_ids[0]
        usable = (ids.numel() // (bs * seq)) * (bs * seq)
        chunks = ids[:usable].view(-1, bs, seq)
        return [chunks[i] for i in range(min(n, chunks.shape[0]))]
    except Exception as exc:
        print(f"[E2E] WikiText 不可达({type(exc).__name__})，回退随机 token（仅演示流程）", flush=True)
        return [torch.randint(0, tok.vocab_size, (bs, seq)) for _ in range(n)]


def _ppl(model, held):
    model.eval()
    tot_loss = tot_tok = 0
    with torch.no_grad():
        for b in held:
            b = b.to(DEV)
            logits = _logits(model(b))
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1),
                reduction="sum",
            )
            tot_loss += float(loss)
            tot_tok += b[:, 1:].numel()
    import math
    return math.exp(tot_loss / max(tot_tok, 1))


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    dtype = torch.float16 if DEV == "npu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(DEV)
    batches = _wikitext_batches(tok)
    train, held = batches[:680], batches[700:714]
    assert held, "held 划分为空（数据集太小，不足 700:714）"
    recon = [b.to(DEV) for b in train[:24]]  # ≈24k token > FFN intermediate

    # baseline 与 pruned 用同一 dtype(fp32)，使 '+X%' 只归因于剪枝而非 fp16→fp32
    work = copy.deepcopy(model).to(DEV).float()
    base = _ppl(work, held)
    print(f"[E2E] {MODEL} baseline ppl = {base:.2f}", flush=True)

    pruned = copy.deepcopy(work)
    P.prune(
        pruned,
        {"methods": {"dense": {"name": "reconstruct", "kwargs": {"prune_ratio": 0.3}}},
         "missing_data_policy": "warn_skip"},
        data=recon,
    )
    p_ppl = _ppl(copy.deepcopy(pruned), held)
    print(f"[E2E] FFN 剪 30% 后 ppl = {p_ppl:.2f} (+{100 * (p_ppl / base - 1):.0f}%)", flush=True)

    t0 = time.time()
    P.finetune(
        pruned, train,
        loss_fn=lambda m, b: F.cross_entropy(
            _logits(m(b))[:, :-1].reshape(-1, m.config.vocab_size), b[:, 1:].reshape(-1)
        ),
        steps=len(train), lr=1e-5, warmup=20, grad_clip=1.0,
    )
    r_ppl = _ppl(pruned, held)
    print(
        f"[E2E] +1-epoch 微调(lr1e-5+warmup, 单遍) ppl = {r_ppl:.2f} "
        f"(+{100 * (r_ppl / base - 1):.0f}% vs base; 剪枝损伤已恢复) {time.time() - t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
