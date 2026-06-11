# Qwen3-4B / Qwen3-8B · 系列 qwen · 结构 dense

> **速览**：`Qwen3` 标准 dense decoder（无 MoE），纯复用主链（不改 workflow/solver）。首推 `attn-linear + mlp` W8A8-int 直转（达标，可直接 deploy）。4B 是仓内 dense Qwen3 起点、8B 同类纯复用。
> **触发信号**：config 无 `num_experts`（dense）→ 无需 L2 家族；通用坑见 [L1](../cross-model-pitfalls.md)。

## 结构与适配要点

- 标准 Qwen3 decoder，attention 沿用官方 `Qwen3Attention`；quant wrapper 拆 `attn-linear`（q/k/v/o 投影）+ `attn-cache`（KV）。8B 关键 config：`hidden=4096`、`intermediate=12288`、`heads=32`、`kv_heads=8`、`head_dim=128`。
- **参考与差异**：4B 系列首条（无前置参考）；8B 参考 4B，无结构差异、纯复用。
- **复用**：`BaseModel` blockwise、`QuantQwen3Attn`、`QuantGatedMLP`、`QuantLinear`。**新增**：无（dense 纯复用）。
- **起步复用清单**（下一条 dense Qwen3）：从 `qwen3.py`、`quant_module.py`、`QuantGatedMLP` 起步，首轮 `attn-linear + mlp` W8A8-int。

## 适配验证结论

- **4B**：标准三步闭环（BF16 baseline → 关闭量化浮点等价 → 最小 PTQ smoke）全部通过。`attn-linear`/`mlp` 在 `bits=16` 与 BF16 **逐项对齐** → rebuilt wrapper 前向语义正确，量化误差来自数值精度而非结构；`block0/mlp` PTQ 链（提取→训练→导出→回载）打通。
- **8B**：三步闭环通过；关闭量化 `attn-linear+mlp` 与 BF16 对齐；`block0/mlp` 最小 PTQ smoke 打通。

## 关键陷阱（dense 起点首遇，均为通用坑，已上抽到 L1）

> 无 dense 专属、不可迁移的 L3 坑——以下都是后续任何网络复用的 L1 经验，4B 是首遇例。

- `lm_head` 与 `embed_tokens` tied → 见 [L1 · 入口与权重](../cross-model-pitfalls.md)。
- `Catcher` 未透传 `attention_type` → 见 [L1 · 入口与权重](../cross-model-pitfalls.md)。
- `attn-linear` wrapper 误替换官方 attention kernel（关闭量化等价漂移）→ 见 [L1 · 量化通用](../cross-model-pitfalls.md)。

## 量化结论

- **4B**：首推 `attn-linear + mlp` W8A8-int 直转，`delta ≤ 0.2`，可直接 deploy。
- **8B**：`attn-linear + mlp` W8A8-int 直转 `ppl_quant=8.8672`、`delta=-0.079`（≤0.2 达标；略优于 BF16 仅视为可接受，非质量提升证据）。

## 适配建议（下次同系列/同结构）

- 先参考：本案 + Qwen 系列 README。
- 先做什么：BF16 baseline → step8 拆 `mlp`/`attn-linear` 验等价 → 合并直转。
- **不建议**：dense Qwen3 误开 `quant_target=moe`。

## 精度速查表

> ppl 口径 seq_len=4096。MXFP 双值为两次评测口径 a/b。

### Qwen3-4B（BF16 12.377）

| 数据类型 | 量化配置 | 量化算法 | ppl |
| --- | --- | --- | --- |
| BF16 | 无 | 无 | 12.377 |
| INT | A4w4: mlp, attn-linear | 无 | 9942.9922 |
| INT | A8w4: mlp, attn-linear | 无 | 19.9143 |
| INT | A8w8: mlp, attn-linear | 无 | 12.4327 |
| MXFP | A4w4: mlp, attn-linear | 无 | 16.043/14.1 |
| MXFP | A8w4: mlp, attn-linear | 无 | 13.808/13.066 |
| MXFP | A8w8: mlp, attn-linear | 无 | 12.6976/12.365 |

### Qwen3-8B（BF16 8.9464）

| 数据类型 | 量化配置 | 量化算法 | ppl |
| --- | --- | --- | --- |
| BF16 | 无 | 无 | 8.9464 |
| INT | A8w8: mlp, attn-linear | 无 | 8.8672 |
