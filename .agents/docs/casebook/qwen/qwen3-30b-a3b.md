# Qwen3-30B-A3B

`qwen3_moe` adapter 验证案例。adapter：`amct_pytorch/common/models/llm/qwen/qwen3/qwen3_moe.py`。

## 适配重点

- Qwen3 MoE decoder。关键 config：`hidden_size=2048`、`num_hidden_layers=48`、`num_experts=128`、`num_experts_per_tok=8`、`moe_intermediate_size=768`、`num_attention_heads=32`、`num_key_value_heads=4`。
- per-expert 权重由 `pack_gated_expert_weights` 打包；量化经 `QuantGatedExperts` 暴露 expert PTQ unit。
- 复用：下一条 Qwen3 MoE 从 `qwen3_moe.py`、`QuantGatedExperts`、`pack_gated_expert_weights` 起步。

## 关键陷阱（可复用）

- **MoE activation capture 命中 `mlp.gate` 报错**：宽泛的 `hook_name="mlp"` 会同时匹配 `mlp.gate`，而 Qwen3 MoE gate 返回 tuple → `AttributeError: 'tuple' object has no attribute 'detach'`。处理：用真实 Wikitext hidden states 过 gate、选实际命中的 expert 单元做 PTQ smoke，而非按名宽抓。

## 精度

- BF16 baseline：Wikitext PPL `seq_len=4096`，`ppl = 8.0444`
- 首轮直转 `attn-linear + moe` W8A8 int（block，无 PTQ）：`ppl_quant = 8.0280`，`delta = -0.016`（≤0.2 接受）

## 性能注意（可复用）

- MoE 走 **per-expert 动态量化**时，平均路由 `M_eff = 4096*8/128 ≈ 256` 偏小，单 expert 收益存疑甚至为负；attention projection 路径收益为正。MoE 性能须由 infer 侧 packed `MoEGMM` 实测确认，不能按 per-expert 直接下结论。
