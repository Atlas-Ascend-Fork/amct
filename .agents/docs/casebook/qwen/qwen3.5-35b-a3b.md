# Qwen3.5-35B-A3B

首条 `qwen3_5moe` 适配案例。adapter：`amct_pytorch/common/models/llm/qwen/qwen3_5/qwen3_5_moe.py`。

## 适配重点

- 40 层 decoder，`layer_types` 为重复的 `linear×3 + full` 混合 attention；checkpoint MoE 已是 **packed 布局**（无需像 235B 做 per-expert 重组）。
- 复用 `BaseModel` blockwise / `PtqUnit` / `QuantLinear` / `apply_quant_attn` / `apply_quant_moe_mlp`；新增 `qwen3_5moe` 的 `build_quant_block()` 路由、`QuantPackedExperts`、`PackedExpertView`。

## 关键陷阱（可复用）

- **别复用 DeepSeek 的 MLA 接口**：`qwen3_5moe` 继承自 `Qwen3_5`，旧路径误调不存在的 `apply_quant_mla()`，一进 quant block 构建就接口报错。修复：统一收回 Qwen 公共接口 `apply_quant_attn()` / `apply_quant_moe_mlp()`。**教训：Qwen 分支先看 dense adapter 的公共接口，不要直接用 DeepSeek 的 MLA 入口。**
- **混合 attention 分开处理**：`linear_attention` 与 `full_attention` 不能套同一套 wrapper；最小闭环阶段 `linear_attention` 暂留 BF16，真正量化时单独补专属 wrapper 与验证链。

## 精度速查表

> ppl 口径 seq_len=4096。 MXFP 双值为两次评测口径 a/b。

| 数据类型 | 量化配置 | 量化算法 | ppl |
| --- | --- | --- | --- |
| BF16 | 无 | 无 | 6.2486 |
| INT | A4w4: moe, attn-linear | 无 | 3839.2734 |
| INT | A8w4: moe, attn-linear | 无 | 7.1928 |
| INT | A8w8: moe, attn-linear | 无 | 6.3781 |
| MXFP | A4w4: moe, attn-linear | 无 | 6.9814/6.826 |
| MXFP | A8w4: moe, attn-linear | 无 | 6.508/6.436 |
| MXFP | A8w8: moe, attn-linear | 无 | 6.3168/6.285 |
| HiFP | A4w4: moe, attn-linear | 无 | 6.617 |
| HiFP | A8w4: moe, attn-linear | 无 |  |
| HiFP | A8w8: moe, attn-linear | 无 |  |
