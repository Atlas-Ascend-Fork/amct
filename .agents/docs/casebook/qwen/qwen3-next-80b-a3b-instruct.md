# Qwen3-Next-80B-A3B-Instruct

首条 `qwen3_next` 适配案例。adapter：`amct_pytorch/common/models/llm/qwen/qwen3_next/qwen3_next.py`（+ `llm/__init__.py` 注册）。

## 适配重点

- 48 层 decoder，block 内**混合 `linear_attention` 与 `full_attention`**；运行时 packed experts，磁盘是展开 expert 权重（`transformers==5.3.0` 下 `Qwen3NextExperts` 运行/存储布局不同，adapter 内重组为 packed）。
- 复用 `BaseModel` blockwise / `PtqUnit`(attn/moe/mlp 路由) / `QuantLinear`；新增 `QuantQwen3NextAttn`、`QuantQwen3NextMLP`、`QuantPackedExperts`、`PackedExpertView` + causal mask 路由与 expert 重组。参考 `qwen3_5_moe.py`（混合 attention）+ `qwen3_moe.py`（MoE 重组 / packed view）。

## 关键陷阱（可复用）

- **混合 attention 的 mask 路由**：`do_embedding_forward()` 只捕 layer0 的 `attention_mask`，而 layer0 是 `linear_attention` 捕到 `None`，后续 `full_attention` 层复用就丢了 causal mask → BF16 PPL 低得离谱。修复：adapter 内分别维护 `linear_attention_mask` 与 `full_attention_mask`（后者 `create_causal_mask(...)` 显式构造），wrapper 已显式传 mask 时保持 `is_causal=False`。**教训：混合 attention 家族 BF16 PPL 好得不正常，先查 blockwise mask 传递链。**

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
