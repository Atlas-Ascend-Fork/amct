# Qwen3-8B

`qwen3` dense adapter 验证案例。adapter：`amct_pytorch/common/models/llm/qwen/qwen3/qwen3.py`。

## 适配重点

- Dense Qwen3 decoder（无 MoE）。关键 config：`hidden_size=4096`、`intermediate_size=12288`、`num_attention_heads=32`、`num_key_value_heads=8`、`head_dim=128`。
- 复用 `QuantQwen3Attn` 与 `QuantGatedMLP`，**无需改 workflow / solver / quantization 模块**。
- 复用：下一条 dense Qwen3 从 `qwen3.py`、`quant_module.py`、`QuantGatedMLP` 起步。

## 精度

- BF16 baseline：Wikitext PPL `seq_len=4096`，`ppl = 8.9464`
- 首轮直转 `attn-linear + mlp` W8A8 int（block，无 PTQ）：`ppl_quant = 8.8672`，`delta = -0.079`（≤0.2 接受；quant 略优于 BF16 仅视为可接受，非质量提升证据）
