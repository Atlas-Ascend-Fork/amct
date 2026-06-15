# Qwen3-30B-A3B · 系列 qwen · 结构 moe

> **速览**：`qwen3_moe` 验证案例。adapter `qwen/qwen3/qwen3_moe.py`，per-expert 打包经 `QuantGatedExperts`。W8A8 `attn-linear+moe` 直转 `delta≈-0.02`（达标）；MoE per-expert 动态量化性能存疑。

## 结构与适配要点

- **参考与差异**：参考 `qwen3-235b`（同 Qwen3 MoE packed-expert 思路）；差异：本案直接走 `qwen3_moe`、规模较小。
- Qwen3 MoE decoder。关键 config：`hidden_size=2048`、`num_hidden_layers=48`、`num_experts=128`、`num_experts_per_tok=8`、`moe_intermediate_size=768`、`heads=32`、`kv_heads=4`。
- **复用 + 新增**：per-expert checkpoint 权重由 `pack_gated_expert_weights` 打包，量化经 `QuantGatedExperts` 暴露 expert PTQ 单元。**起步复用清单**：下一条 Qwen3 MoE 从 `qwen3_moe.py`、`QuantGatedExperts`、`pack_gated_expert_weights` 起步。

## 适配验证结论

- 三步闭环通过；关闭量化 `attn+moe` 与 BF16 对齐（差 ≈ -0.005）；`block0/moe` 最小 PTQ smoke（128 expert 单元枚举、选实际命中 expert）打通。

## 关键陷阱（模型/框架内在，可复用）

- **MoE activation capture 误命中 `mlp.gate`** —— 现象：activation capture 报 `AttributeError: 'tuple' object has no attribute 'detach'`。根因：宽泛 `hook_name="mlp"` 同时匹配 `mlp.gate`，而 Qwen3 MoE gate 返回 tuple。处理：用真实 hidden states 过 gate、选实际命中的 expert 单元做 capture/PTQ，不按名宽抓。

## 量化结论

- BF16 baseline（seq_len=4096）`8.0444`；首轮直转 `attn-linear+moe` W8A8-int（block，无 PTQ）`ppl_quant=8.0280`，`delta=-0.016`（≤0.2，达标）。

## 性能注意（可复用）

- MoE 走 per-expert 动态量化时，平均路由 `M_eff = 4096×8/128 ≈ 256` 偏小，单 expert 收益存疑甚至为负；attention projection 路径为正。MoE 性能须由 infer 侧 packed `MoEGMM` 实测确认，不能按 per-expert 直接下结论。

## 适配建议（下次同系列/同结构）

- 先参考：`qwen3-235b` + 本案。
- 先做什么：BF16 → 关闭量化等价 → 用真实 hidden states 过 gate 选实际命中 expert 做 smoke。
- **不建议**：按名宽抓 capture（会撞 `mlp.gate`）；未确认 infer MoEGMM 路径前按 per-expert 给 MoE 性能结论。
