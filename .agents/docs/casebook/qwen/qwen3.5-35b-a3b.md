# Qwen3.5-35B-A3B · 系列 qwen · 结构 moe（混合 attention）

> **速览**：首条 `qwen3_5moe` 适配。混合 `linear`+`full` attention；checkpoint MoE 已 packed（无需重组）。核心坑：误用 DeepSeek MLA 接口。

## 结构与适配要点

- **参考与差异**：参考 `qwen3_5` attention/blockwise + `qwen3_moe` MoE 路由；差异：checkpoint MoE 已 packed，无需 per-expert 重组。
- 40 层 decoder，`layer_types` 重复 `linear×3 + full`；checkpoint MoE 已是 packed 布局（无需像 235B 重组）。
- **复用**：`BaseModel` blockwise、`PtqUnit`、`QuantLinear`、`apply_quant_attn`、`apply_quant_moe_mlp`。**新增**：`qwen3_5moe` 的 `build_quant_block()` 路由、`QuantPackedExperts`、`PackedExpertView`。

## 适配验证结论

- 三步闭环通过；关闭量化 `attn`(3/19/39) 接近、`moe`(0/20/39) 对齐；`layer0/shared_expert` 最小 PTQ smoke 打通。

## 关键陷阱（模型/框架内在，可复用）

- **误用 DeepSeek 的 MLA 接口** —— 现象：进 quant block 构建即接口报错（调到不存在的 `apply_quant_mla()`）。根因：`qwen3_5moe` 继承 `Qwen3_5`，应复用 Qwen 公共接口而非 DeepSeek MLA 专用入口。处理：统一收回 `apply_quant_attn()`/`apply_quant_moe_mlp()`。**教训：Qwen 分支先看 dense adapter 公共接口，勿直接用 DeepSeek MLA 入口。**
- **混合 attention 分开处理** —— `linear_attention` 与 `full_attention` 不能套同一 wrapper；最小闭环阶段 `linear_attention` 暂留 BF16，真正量化时单独补专属 wrapper 与验证链。

## 适配建议（下次同系列/同结构）

- 先参考：本案 + Qwen 系列 README。
- 先做什么：先确认 experts 是 packed 还是展开。
- **不建议**：Qwen 分支误用 DeepSeek MLA 入口；`linear`/`full` attention 一把梭。

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
