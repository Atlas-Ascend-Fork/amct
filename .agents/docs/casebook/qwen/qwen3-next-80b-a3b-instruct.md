# Qwen3-Next-80B-A3B-Instruct · 系列 qwen · 结构 moe（混合 attention）

> **速览**：首条 `qwen3_next` 适配。block 内混合 `linear_attention`+`full_attention`、packed expert 重组。核心坑：混合 attention 的 causal mask 路由。

## 结构与适配要点

- 48 层 decoder，block 内**混合 `linear_attention` 与 `full_attention`**；运行时 packed experts，磁盘展开（`transformers==5.3.0` 下 `Qwen3NextExperts` 运行/存储布局不同）。
- **复用**：`BaseModel` blockwise、`PtqUnit`（attn/moe/mlp）、`QuantLinear`。**新增**：`QuantQwen3NextAttn`、`QuantQwen3NextMLP`、`QuantPackedExperts`、`PackedExpertView` + causal mask 路由与 expert 重组。参考 `qwen3_5_moe.py`（混合 attention）+ `qwen3_moe.py`（MoE 重组）。

## 适配验证结论

- 三步闭环通过；关闭量化 `attn`(3/23/47) 对齐、`moe`(0/23/47) 微漂；`layer0/shared_expert` 最小 PTQ smoke 打通。

## 关键陷阱（模型/框架内在，可复用）

- **混合 attention 的 causal mask 丢失** —— 现象：BF16 PPL 低得离谱（像“看到后文”）。根因：`do_embedding_forward()` 只捕 layer0 mask，而 layer0 是 `linear_attention` 捕到 `None`，后续 `full_attention` 复用即丢 causal mask。处理：adapter 内分别维护 `linear_attention_mask` 与 `full_attention_mask`（后者 `create_causal_mask(...)` 显式构造），wrapper 已显式传 mask 时保持 `is_causal=False`。**教训：混合 attention 家族 BF16 PPL 好得反常，先查 blockwise mask 传递链。**
- **expert 布局 packed vs 展开** —— 处理：adapter 内重组为 packed experts；先确认运行环境 `transformers` 版本再定 expert 路径（逐 expert / packed）。

## 精度速查表

> ppl 口径 seq_len=4096。 MXFP 双值为两次评测口径 a/b。

| 数据类型 | 量化配置 | 量化算法 | ppl | mmlu |
| --- | --- | --- | --- | --- |
| BF16 | 无 | 无 | 5.4483 | 85.4 |
| INT | A4w4: moe, attn-linear | 无 | 561.4258 |  |
| INT | A8w4: moe, attn-linear | 无 | 6.0114 |  |
| INT | A8w8: moe, attn-linear | 无 |  |  |
| INT | MOE A8W8 | 无 |  | 85.5 |
| INT | MOE A8W4 | 无 |  |  |
| INT | MOE A8W4 | LWC |  |  |
| MXFP | A4w4: moe, attn-linear | 无 | 6.123/5.901 |  |
| MXFP | A8w4: moe, attn-linear | 无 | 5.7337/5.609 |  |
| MXFP | A8w8: moe, attn-linear | 无 | 5.451 |  |
| HiFP | A4w4: moe, attn-linear | 无 | 5.6854 |  |
| HiFP | A8w4: moe, attn-linear | 无 |  |  |
| HiFP | A8w8: moe, attn-linear | 无 |  |  |
