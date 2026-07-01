# Qwen3-Next-80B-A3B-Instruct · 系列 qwen · 结构 moe（混合 attention）

> **速览**：`Qwen3Next` —— block 内混合 `linear_attention`+`full_attention`、packed expert 重组。首推 `attn-linear + moe`（A8W8 INT MoE mmlu 85.5 ≈ BF16 85.4）。核心坑（均通用）：混合 attention causal mask 路由 + expert packed/展开。
> **触发信号**：`layer_types` 含 `linear_attention`+`full_attention`；config `num_experts>0`。读 [L2 · 混合 attention 类 + MoE 类](../structure-family-pitfalls.md) + [L1](../cross-model-pitfalls.md)。

## 结构与适配要点

- 48 层 decoder，block 内**混合 `linear_attention` 与 `full_attention`**；运行时 packed experts、磁盘展开（`transformers==5.3.0` 下 `Qwen3NextExperts` 运行/存储布局不同）。
- **参考与差异**：参考 `qwen3.5-3.6`（混合 attention 路由）+ `qwen3-moe`（MoE 重组）；差异：`transformers==5.3.0` 下 packed/展开布局差异。
- **复用**：`BaseModel` blockwise、`PtqUnit`（attn/moe/mlp）、`QuantLinear`。**新增**：`QuantQwen3NextAttn`、`QuantQwen3NextMLP`、`QuantPackedExperts`、`PackedExpertView` + causal mask 路由与 expert 重组。

## 适配验证结论

- 三步闭环通过；关闭量化 `attn`(3/23/47) 对齐、`moe`(0/23/47) 微漂；`layer0/shared_expert` 最小 PTQ smoke 打通。

## 关键陷阱（均为通用坑，已上抽到 L1/L2）

> 无 `Qwen3Next` 专属、不可迁移的 L3 坑。

- 混合 attention causal mask 丢失（layer0 `linear_attention` 捕到 `None` mask 误传后续 `full_attention`，BF16 PPL 低得离谱）→ [L2 · 混合 attention 类](../structure-family-pitfalls.md)。
- expert 布局 packed vs 展开（先确认 `transformers` 版本再定 expert 路径）→ [L2 · MoE 类](../structure-family-pitfalls.md)。

## 适配建议（下次同系列/同结构）

- 先参考：本案 + `qwen3.5-3.6` + Qwen 系列 README。
- 先做什么：先把 `transformers==5.3.0` 当基线，再判 expert 结构与 mask 路由。
- **不建议**：混合 attention 家族默认一套 mask 打到底。

## 精度速查表

> ppl 口径 seq_len=4096。MXFP 双值为两次评测口径 a/b。

| 数据类型 | 量化配置 | 量化算法 | ppl | mmlu |
| --- | --- | --- | --- | --- |
| BF16 | 无 | 无 | 5.4483 | 85.4 |
| INT | A4w4: moe, attn-linear | 无 | 561.4258 |  |
| INT | A8w4: moe, attn-linear | 无 | 6.0114 |  |
| INT | MOE A8W8 | 无 |  | 85.5 |
| MXFP | A4w4: moe, attn-linear | 无 | 6.123/5.901 |  |
| MXFP | A8w4: moe, attn-linear | 无 | 5.7337/5.609 |  |
| MXFP | A8w8: moe, attn-linear | 无 | 5.451 |  |
