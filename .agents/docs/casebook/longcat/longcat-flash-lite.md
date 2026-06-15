# LongCat-Flash-Lite · 系列 longcat · 结构 moe

> **速览**：首个 LongCat 适配。特殊输入 `ngram_embeddings`，block = 双 self-attn + 双 dense MLP + packed MoE shortcut。核心坑：int activation quant 的 `is_act` / signed clamp。

## 结构与适配要点

- **参考与差异**：参考 Qwen dense/MoE wrapper；差异：`ngram_embeddings` 输入路径 + 双 attn/双 MLP/packed MoE shortcut 特殊拓扑，需自有 attention/packed expert wrapper。
- 输入路径非普通 token embedding，而是 `ngram_embeddings`；每 block = 双 `self_attn` + 双 dense `mlp` + 一条 packed MoE shortcut。输入路径与拓扑特殊，必须**源码公式重放 + 逐层 stepwise 校验**。
- **复用**：`BaseModel`、`PtqUnit`（attn/mlp/moe）、`QuantLinear`、`QuantGatedMLP`。**新增**：`QuantLongcatMLA`、`LongcatPackedExperts`、`LongcatPackedExpertView` + LongCat 专属 embedding 加载与 expert 重组。

## 适配验证结论

- 三步闭环通过；`ngram_embeddings` 源码公式重放与逐层 blockwise 输出对齐 through 整个 decoder stack（含 `norm + lm_head` logits）。

## 关键陷阱（模型/框架内在，可复用）

- **int activation quant 在首个 MLP block 报错** —— 现象：`weight_quant()` 收到 activation tensor 触发 shape 断言。根因：`ActivationQuantizer` 实例化 dtype 漏传 `is_act=True`，且 `dynamic_per_token_quant()` 误用了只适用正数的 clamp。处理：传 `is_act=True` + 去掉 signed activation quant 的正数 clamp。

## 量化结论

- 第二轮（2026-05-12，业务收窄范围至 `mlp+moe`）：直转 a8w8-int8 @ seq_len=4096，`ppl_bf16=14.6243 / ppl_quant=14.8006 / delta=0.1763`（≤0.2 达标，无需 PTQ）。

## 性能注意（可复用）

- prefill(M=4096)视角：**MoE 量化常为负收益**（per-expert 单 expert M≈384 过小，dynamic_quant + 头开销 > BF16 matmul，每层 ×128 放大）；**MLP 唯一明显正收益**；量化 attn 几乎无增益无损益。
- “attn 进不进量化”首要看**精度 delta 与下游 infer 路径覆盖**，非 prefill 性能；MoE 性能须 infer 侧 packed `MoEGMM` 实测确认（INT 部署若 per-expert 则负收益）。decode(M=1) 未测，weight-bound 下可能翻正。

## 适配建议（下次同系列/同结构）

- 先参考：本案 + LongCat 系列 README。
- 先做什么：先验证自定义输入路径 → 逐层 block 等价 → 再直转。
- **不建议**：最简方案（近无损）前急上 PTQ。

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
