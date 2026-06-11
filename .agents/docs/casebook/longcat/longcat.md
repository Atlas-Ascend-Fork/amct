# LongCat-Flash-Lite / LongCat-Next · 系列 longcat · 结构 moe（自定义 modeling）

> **速览**：LongCat 特殊拓扑——`ngram_embeddings` 输入 + block = 双 self-attn + 双 dense MLP + packed MoE shortcut；接非 transformers 包内自定义 modeling。`LongcatNext(LongcatLite)` **仅 override 一个方法**（`empty_weights_model`：切 ngram text-only backbone + 多模态 lm_head 词表），其余逐方法复用 Lite。首推 `mlp+moe` a8w8-int8 直转（Lite 达标）。
> **触发信号**：`architectures` 不在 transformers 需 `trust_remote_code`（自定义 modeling）；`ngram_embeddings` 输入。读 [L2 · 接非 transformers 自定义 modeling + MoE 类](../structure-family-pitfalls.md) + [L1](../cross-model-pitfalls.md)。

## 结构与适配要点

- 输入路径非普通 token embedding，而是 `ngram_embeddings`；每 block = 双 `self_attn` + 双 dense `mlp` + 一条 packed MoE shortcut。输入路径与拓扑特殊，必须**源码公式重放 + 逐层 stepwise 校验**。`PtqUnit` 拆 `attn→self_attn_0/1`、`mlp→mlp_0/1`、`moe→expert_<i>`；checkpoint 逐 expert，运行时重组 packed。
- **参考与差异**：Lite 首条（参考 Qwen dense/MoE wrapper）；Next 拓扑同 Lite，唯一差异 = `empty_weights_model`。
- **复用**：`BaseModel`、`PtqUnit`（attn/mlp/moe）、`QuantLinear`、`QuantGatedMLP`。**新增**：`QuantLongcatMLA`、`LongcatPackedExperts`、`LongcatPackedExpertView` + LongCat 专属 embedding 加载与 expert 重组。

## 适配验证结论

### LongCat-Flash-Lite
- 三步闭环通过；`ngram_embeddings` 源码公式重放与逐层 blockwise 输出对齐 through 整个 decoder stack（含 `norm + lm_head` logits）。

### LongCat-Next（= LongcatLite，唯一 delta = `empty_weights_model`）
- 三步闭环通过；关闭量化 attn/mlp/moe 代表层 `max_abs=0` 完全对齐；`layer0/mlp_0` 最小 PTQ smoke（flatquant 1-epoch，回载 `max_abs=0`）打通。
- **唯一 delta（L3）**：adapter override `empty_weights_model()` → 实例化 `modeling_longcat_ngram.LongcatFlashNgramForCausalLM`（text-only backbone，避开顶层多模态 + `flash_attn` 依赖），`lm_head` 对齐 `text_vocab_plus_multimodal_special_token_size`；其余 forward/PTQ/deploy 与 Lite **逐方法相同**（显式 `super()` 透传）。

## 关键陷阱（L3 + 通用引 L2/L1）

- **int activation quant 的 `is_act` / signed clamp**（Lite 首遇） —— 现象：首个 MLP block `weight_quant()` 收到 activation tensor 触发 shape 断言。→ 见 [L1 · 量化通用](../cross-model-pitfalls.md)。
- **`trust_remote_code` 顶层走多模态、需切 text-only backbone**（Next 首遇） —— 现象：`from_config(..., trust_remote_code=True)` 被 `flash_attn` 阻塞。处理：adapter 内切 `modeling_longcat_ngram.LongcatFlashNgramForCausalLM` + 修 `lm_head`。→ 见 [L2 · 接非 transformers 自定义 modeling](../structure-family-pitfalls.md)。

## 量化结论（+ 性能注意）

- **Lite**：`mlp+moe` a8w8-int8 直转 `ppl_bf16=14.6243 / ppl_quant=14.8006 / delta=0.1763`（≤0.2 达标，无需 PTQ）。
- 性能注意：prefill MoE per-expert 常负收益、MLP 唯一明显正收益、attn 几乎无损益 → 见 [L2 · MoE 类](../structure-family-pitfalls.md)；MoE 性能须 infer 侧 packed `MoEGMM` 实测。

## 适配建议（下次同系列/同结构）

- 先参考：本案 + LongCat 系列 README。
- 先做什么：先验证自定义输入路径 → 逐层 block 等价 → 再直转；Next 先看能否复用 Lite block 拓扑，兼容收在 adapter。
- **不建议**：最简方案（近无损）前急上 PTQ；直接空载顶层多模态模型。

## 精度速查表

> ppl 口径 seq_len=4096。**口径敏感：`2048→4096` 后 ppl 反而变差，对比/复现须固定 `seq_len=4096`、同 transformers 版本。** MXFP 双值为两次评测口径 a/b。

### LongCat-Flash-Lite（BF16 14.6133）

| 数据类型 | 量化配置 | 量化算法 | ppl |
| --- | --- | --- | --- |
| BF16 | 无 | 无 | 14.6133 |
| INT | A4w4: mlp, moe, attn-linear | 无 | 1053.832 |
| INT | A8w4: mlp, moe, attn-linear | 无 | 13.1681 |
| INT | A8w8: mlp, moe, attn-linear | 无 | 14.0334 |
| MXFP | A4w4: mlp, moe, attn-linear | 无 | 13.2715/12.283 |
| MXFP | A8w4: mlp, moe, attn-linear | 无 | 10.4672/10.061 |
| MXFP | A8w8: mlp, moe, attn-linear | 无 | 14.8785/15.091 |

### LongCat-Next（BF16 17.518）

| 数据类型 | 量化配置 | 量化算法 | ppl |
| --- | --- | --- | --- |
| BF16 | 无 | 无 | 17.518 |
| INT | A4w4: mlp, moe, attn-linear | 无 | 3302.094 |
| INT | A8w4: mlp, moe, attn-linear | 无 | 24.8581 |
| INT | A8w8: mlp, moe, attn-linear | 无 | 15.718 |
| MXFP | A4w4: mlp, moe, attn-linear | 无 | 19.0125/18.266 |
| MXFP | A8w4: mlp, moe, attn-linear | 无 | 17.5067/16.899 |
| MXFP | A8w8: mlp, moe, attn-linear | 无 | 18.3902/19.2 |
