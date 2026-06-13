# Qwen3-8B · 系列 qwen · 结构 dense

> **速览**：`qwen3` dense 验证案例。adapter `qwen/qwen3/qwen3.py`，复用 `QuantQwen3Attn`/`QuantGatedMLP`，不改主链。W8A8 `attn-linear+mlp` 直转 `delta≈-0.08`（达标）。

## 结构与适配要点

- Dense Qwen3 decoder（无 MoE）。关键 config：`hidden_size=4096`、`intermediate_size=12288`、`heads=32`、`kv_heads=8`、`head_dim=128`。
- **复用**：`QuantQwen3Attn`、`QuantGatedMLP`，无需改 workflow/solver/quant 模块。**起步复用清单**：下一条 dense Qwen3 从 `qwen3.py`、`quant_module.py`、`QuantGatedMLP` 起步。

## 适配验证结论

- 三步闭环通过；关闭量化（bits=16）`attn-linear+mlp` 与 BF16 对齐；`block0/mlp` 最小 PTQ smoke 打通。

## 量化结论

- BF16 baseline（Wikitext PPL, seq_len=4096）`8.9464`；首轮直转 `attn-linear+mlp` W8A8-int（block，无 PTQ）`ppl_quant=8.8672`，`delta=-0.079`（≤0.2，达标；quant 略优于 BF16 仅视为可接受，非质量提升证据）。

> 注：dense 路径纯复用，未遇模型特有结构性陷阱。
