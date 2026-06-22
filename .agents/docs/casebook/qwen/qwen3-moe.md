# Qwen3-30B-A3B / Qwen3-235B-A22B · 系列 qwen · 结构 moe

> **速览**：`Qwen3Moe` MoE decoder，checkpoint 逐 expert 展开、运行时 packed（adapter 内经 `pack_gated_expert_weights`/`QuantGatedExperts` 重组）。首推 `attn-linear + moe` W8A8-int8 直转（达标）。30B 与 235B 同 `Qwen3Moe` 类，仅规模不同。
> **触发信号**：config `num_experts>0`；checkpoint `mlp.experts.<i>.*`。读 [L2 · MoE 类](../structure-family-pitfalls.md) + [L1](../cross-model-pitfalls.md)。

## 结构与适配要点

- Qwen3 MoE decoder。30B：48 层 / 128 expert / top-8 / `moe_intermediate=768` / heads 32 / kv 4。235B：94 层（**注意 config 谎报 5 层**）。两者 checkpoint 均逐 expert 展开，运行时 packed。
- **参考与差异**：235B 首条（参考 Qwen dense + LongCat packed-expert 思路）；30B 参考 235B，同 MoE packed-expert 思路、规模较小。
- **复用**：`BaseModel` blockwise、`PtqUnit`（attn/moe）、`QuantLinear`。**新增**：`QuantQwen3MoeAttn`、`QuantGatedExperts`(`QuantPackedExperts`)、`PackedExpertView` + adapter 内层数修正与 expert 重组（`pack_gated_expert_weights`）。
- **起步复用清单**（下一条 Qwen3 MoE）：从 `qwen3_moe.py`、`QuantGatedExperts`、`pack_gated_expert_weights` 起步，首轮 `attn+moe` a8w8-int8。

## 适配验证结论

- **30B**：三步闭环通过；关闭量化 `attn+moe` 与 BF16 对齐（≈-0.005）；`block0/moe` 最小 PTQ smoke（128 expert 枚举、选实际命中 expert）打通。
- **235B**：三步闭环通过；关闭量化 `attn`(0/47/93) 精确对齐、`moe` BF16 级微漂；`layer0/expert_0` 最小 PTQ smoke 打通。

## 关键陷阱（本系列遇到的均为通用坑，已上抽到 L1/L2）

> 无 `Qwen3Moe` 专属、不可迁移的 L3 坑——全部为同类网络复用经验。

- config 层数 ≠ checkpoint 真实层数（235B：config=5 / 真实 94）→ [L1 · checkpoint 是唯一事实源](../cross-model-pitfalls.md)。
- expert 磁盘展开 vs 运行时 packed → [L2 · MoE 类](../structure-family-pitfalls.md)。
- MoE activation capture 误命中 `mlp.gate`（30B：tuple 无 detach）→ [L2 · MoE 类](../structure-family-pitfalls.md)。
- 初始 BF16 PPL 离谱（235B：466，漏传 `attention_mask`，chunk0 loss 5.47→1.23）→ [L1 · BF16 PPL 异常](../cross-model-pitfalls.md)。
- 通用 PTQ provider tensor-only，不覆盖 attention unit → [L1 · 量化通用](../cross-model-pitfalls.md)。

## 量化结论（+ 性能注意）

- 首推 `attn+moe` W8A8-int8 直转：30B `delta=-0.016`、235B `delta=0.0363`（均 ≤0.2 无需 PTQ）。已落地粒度：Attention Linear A8W8 INT Per-Token；MoE Expert（不含 gate）A8W8 INT Per-Channel。
- 性能注意：MoE per-expert 动态量化在小 `M_eff`（30B ≈ 4096×8/128 ≈ 256）收益存疑甚至负，须 infer 侧 packed `MoEGMM` 实测确认；attention 路径正常 → 见 [L2 · MoE 类](../structure-family-pitfalls.md)。

## 适配建议（下次同系列/同结构）

- 先做什么：先验 checkpoint 真实深度 + expert 布局；BF16 离谱先查 blockwise `attention_mask`；首轮 `attn+moe` a8w8-int8。
- **不建议**：盲信 `config.num_hidden_layers`；写 expert loader 前不比对磁盘 key；按名宽抓 capture（撞 `mlp.gate`）；未确认 infer `MoEGMM` 前按 per-expert 给 MoE 性能结论。

## 精度速查表

> ppl 口径 seq_len=4096。MXFP 双值为两次评测口径 a/b。

### Qwen3-30B-A3B（BF16 8.0444）

| 数据类型 | 量化配置 | 量化算法 | ppl |
| --- | --- | --- | --- |
| BF16 | 无 | 无 | 8.0444 |
| INT | A8w8: moe, attn-linear | 无 | 8.0280 |

### Qwen3-235B-A22B（BF16 5.09）

| 数据类型 | 量化配置 | 量化算法 | ppl |
| --- | --- | --- | --- |
| BF16 | 无 | 无 | 5.09 |
| INT | A4w4: moe, attn-linear | 无 | 46.3021 |
| INT | A8w4: moe, attn-linear | 无 | 5.71 |
| INT | A8w8: moe, attn-linear | 无 | 5.1427 |
| MXFP | MoE Expert A8W8 | 无 | 5.115 |
| MXFP | Attn A8W8 | 无 | 5.12 |
| MXFP | MoE Expert A4W4 | 无 | 5.52 |
| MXFP | Attn A4W4 | 无 | 5.40 |
| MXFP | A4w4: moe, attn-linear | 无 | 5.82/5.656 |
| MXFP | A8w4: moe, attn-linear | 无 | 5.417/5.368 |
| MXFP | A8w8: moe, attn-linear | 无 | 5.1267/5.135 |
