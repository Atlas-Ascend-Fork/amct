# Qwen3-235B-A22B

首个 Qwen3 MoE 适配案例。**核心难点：模型 config 与真实 checkpoint 深度不一致；checkpoint 是展开的 per-expert 权重，运行时模块期待 packed expert 张量。**

## 适配重点

- 结构：`Qwen3MoeDecoderLayer` / `Qwen3MoeAttention` / `Qwen3MoeSparseMoeBlock`；运行时 packed experts，磁盘是展开 expert 权重。
- 复用仓内抽象：`BaseModel` blockwise 工作流、`PtqUnit`（attn/moe 路由）、`QuantLinear`；适配只动 `models/llm/...`，不改 workflow/solver。
- 新增模型专属实现：`QuantQwen3MoeAttn`、`QuantPackedExperts`、`PackedExpertView`，及 adapter 内层数修正与 expert 重组逻辑。

## 典型问题（可复用）

1. **config 层数 ≠ checkpoint 真实层数**：`config.json` 写 `num_hidden_layers=5`，但 shard index 是 94 层 → HF 只加载 5 层。修复：adapter 初始化按 weight index 推断真实层数覆盖 `num_hidden_layers`。教训：超大模型不要盲信 config 深度，先看 `model.safetensors.index.json`。
2. **expert 权重布局不一致**：checkpoint 暴露 `mlp.experts.<idx>.{gate,up,down}_proj.weight`，运行时期待 packed 张量。修复：adapter 内重组为 packed `gate_up_proj` / `down_proj`。教训：写 loader 前先比对磁盘 expert key 与运行时参数名。
3. **初始 BF16 PPL 离谱（466）**：根因 `do_block_forward` 未把 `attention_mask` 传给 block，blockwise 路径与真实 decoder 不一致。修复：显式传入捕获的 mask（修后 chunk0 loss 从 5.47 回到 1.23）。教训：BF16 PPL 离谱先查 blockwise `attention_mask` 传递。
4. **通用 PTQ provider 不覆盖 attention unit**：`materialize_gt()` 只转发 tensor batch，但 attention unit 还需 `position_embeddings` 和 `attention_mask`。当前处理：最小 PTQ smoke 先用 MoE expert unit 闭环；后续做 attention unit PTQ 需把 provider 泛化为支持额外上下文。

## 量化结论

- 首轮直转 `attn + moe` W8A8 int8 即较稳：`delta = 0.0363`（≤0.2，无需 PTQ）。
- 已落地粒度：Attention Linear A8W8 INT Per-Token；MoE Expert（不含 gate）A8W8 INT Per-Channel。

## 精度速查表

> ppl 口径 seq_len=4096。 MXFP 双值为两次评测口径 a/b。

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
| HiFP | A4w4: moe, attn-linear | 无 | 5.3359 |
| HiFP | A8w4: moe, attn-linear | 无 |  |
| HiFP | A8w8: moe, attn-linear | 无 |  |
