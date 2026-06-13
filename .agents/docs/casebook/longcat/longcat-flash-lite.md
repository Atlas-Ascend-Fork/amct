# LongCat-Flash-Lite

首个 LongCat 系列适配案例。adapter：`amct_pytorch/common/models/llm/longcat/longcat_lite/`。

## 适配重点

- 输入路径不是普通 token embedding，而是 `ngram_embeddings`；每个 decoder block = 双 self-attn 分支 + 双 dense MLP 分支 + 一条 packed MoE shortcut。
- 复用 `BaseModel` / `PtqUnit`(attn/mlp/moe) / `QuantLinear` / `QuantGatedMLP`；新增 `QuantLongcatMLA`、`LongcatPackedExperts`、`LongcatPackedExpertView` + LongCat 专属 embedding 加载与 expert 重组。
- 输入路径与 block 拓扑特殊，必须做源码公式重放 + 逐层 stepwise 校验。

## 关键陷阱（可复用）

- **int activation quant 在第一个 MLP block 就报错**：`weight_quant()` 收到 activation tensor 触发 shape 断言。根因：`ActivationQuantizer` 实例化 dtype 时漏传 `is_act=True`，且 `dynamic_per_token_quant()` 误用了只适用正数的 clamp。修复：传 `is_act=True` + 去掉 signed activation quant 里的正数 clamp。

## 量化结论

- 第二轮（2026-05-12）：量化范围由 `attn+mlp+moe` 收窄为 **`mlp+moe`** 直转 a8w8-int8 @ seq_len=4096，`ppl_bf16=14.6243 / ppl_quant=14.8006 / delta=0.1763`（≤0.2 达标，无需 PTQ）。

## 性能注意（可复用）

- prefill(M=4096)视角：**MoE 量化常为负收益**（per-expert 单 expert M≈384 太小，dynamic_quant + 头开销 > BF16 matmul，且每层 ×128 放大）；**MLP 是唯一明显正收益**；量化 attn 几乎无增益也无损益。
- 因此“attn 进不进量化”首要看**精度 delta 与下游 infer 路径覆盖**，不是 prefill 性能；MoE 性能须由 infer 侧 packed `MoEGMM` 实测确认（INT 部署若走 per-expert 则负收益）。decode(M=1) 未测，weight-bound 下可能翻正。

## 精度速查表

> ppl 口径 seq_len=4096（2048→4096 后精度反而变差）。 MXFP 双值为两次评测口径 a/b。

| 数据类型 | 量化配置 | 量化算法 | ppl |
| --- | --- | --- | --- |
| BF16 | 无 | 无 | 14.6133 |
| INT | A4w4: mlp, moe, attn-linear | 无 | 1053.832 |
| INT | A8w4: mlp, moe, attn-linear | 无 | 13.1681 |
| INT | A8w8: mlp, moe, attn-linear | 无 | 14.0334 |
| MXFP | A4w4: mlp, moe, attn-linear | 无 | 13.2715/12.283 |
| MXFP | A8w4: mlp, moe, attn-linear | 无 | 10.4672/10.061 |
| MXFP | A8w8: mlp, moe, attn-linear | 无 | 14.8785/15.091 |
| HiFP | A4w4: mlp, moe, attn-linear | 无 | 14.258 |
| HiFP | A8w4: mlp, moe, attn-linear | 无 |  |
| HiFP | A8w8: mlp, moe, attn-linear | 无 |  |
