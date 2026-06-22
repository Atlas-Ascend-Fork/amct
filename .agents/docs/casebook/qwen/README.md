# Qwen Casebook

这一页是 **Qwen 系列总览页**，不是具体模型个案。通用坑先读 [L1 跨网络](../cross-model-pitfalls.md) + [L2 结构家族](../structure-family-pitfalls.md)；本页只留 Qwen 系列特有的取向。

## 系列概览

- 当前已覆盖模型：`qwen3`、`qwen3_moe`、`qwen3_5`、`qwen3_5moe`、`qwen3_6moe`、`qwen3_next`
- 整体结构特点：这一系列同时覆盖 dense 和多条 MoE 路径，attention 路径整体接近标准 Qwen，但不同检查点的层数配置、attention 形态、expert 布局和 rotary 上下文可能不同
- 当前已确认的 2 条 MoE 适配分支：
  - `qwen3_moe`：checkpoint 上是展开的 per-expert 权重，运行时期待 packed experts
  - `qwen3_5moe`：checkpoint 上已经是 packed experts，但 block 内部同时混有 `linear_attention` 和 `full_attention`
  - `qwen3_next`：当前 `transformers==5.3.0` 环境下，checkpoint 上是展开的 per-expert 权重，运行时 MoE 模块是 packed experts，block 内部也混有 `linear_attention` 和 `full_attention`
- 当前已确认的 dense 适配分支：
  - `qwen3`：标准 dense Qwen3 decoder，`lm_head` 与 `embed_tokens` tied；attention wrapper 在只做 `attn-linear` 时需要继续复用官方 `attention_interface`，不能提前切到自定义 attention kernel

## 默认参考路径

- 新模型进入时优先参考：[qwen3.py](../../../../amct_pytorch/common/models/llm/qwen/qwen3/qwen3.py)、[qwen3_moe.py](../../../../amct_pytorch/common/models/llm/qwen/qwen3/qwen3_moe.py)、[qwen3_5.py](../../../../amct_pytorch/common/models/llm/qwen/qwen3_5/qwen3_5.py)、[qwen3_5_moe.py](../../../../amct_pytorch/common/models/llm/qwen/qwen3_5/qwen3_5_moe.py)、[qwen3_next.py](../../../../amct_pytorch/common/models/llm/qwen/qwen3_next/qwen3_next.py)
- 适配优先顺序：先 BF16 blockwise baseline，再做关闭量化等价验证，再跑最小 PTQ smoke
- 默认优先复用的抽象：`BaseModel`、`PtqUnit`、`QuantLinear`、现有 quant-apply helper

## 通用适配经验

- 这一系列最容易需要模型专属 wrapper 的地方通常是 attention 和 MoE experts
- 默认不要先改 workflow、solver 和 quant module，优先把模型特有逻辑留在 `amct_pytorch/common/models/llm/qwen/...`
- 最常见的实现风险是：配置字段和真实 checkpoint 结构不一致、packed expert 布局与运行时模块不一致，或者 blockwise 路径漏传 `attention_mask`、`position_embeddings`、`position_ids`
- 对 `qwen3_5moe` 这一支，`linear_attention` 和 `full_attention` 不能混为同一类 attention 量化路径处理，适配时要先把两者的 block 行为对齐
- 对 `qwen3_next` 这一支，layer0 是 `linear_attention`，不能把它在 embedding 阶段捕获到的 `attention_mask` 直接复用给后续 `full_attention` 层；需要在 adapter 内显式维护 full-attn causal mask

## 通用量化经验

- BF16 baseline 一般先用 Wikitext PPL blockwise 口径拿，先看 chunk0 loss 是否合理，再看全量 PPL
- dense 模型的第一版直转量化可以先从 `mlp` 或 `attn + mlp` 开始；`qwen3_5moe` 和 `qwen3_moe` 这类模型可以优先从 `attn + moe` 的 `a8w8-int8` 开始
- `qwen3` 这一支在 `Qwen3-4B` 上已经验证过：`attn-linear + mlp` 的 `a8w8-int` 可以直接作为第一轮直转量化起点；前提是先做 step8，确认 attention wrapper 在 `bits=16` 时和 BF16 等价
- 更敏感的模块通常是：attention 路径的 mask / rotary 输入，以及 MoE 侧的 expert layout
- `qwen3_5moe` 当前的 `linear_attention` 仍应先保持 BF16，不要在模型适配阶段默认把它纳入 attention 量化闭环
- 只有在直转量化 `delta > 0.2` 或已经做过粗粒度定位后，才考虑升级到 PTQ

- `Qwen3-8B`：`attn-linear + mlp` W8A8 Int8 直转 @ `seq_len=4096`，精度达标（delta ≤ 0.2）。
- `Qwen3-30B-A3B`：`attn-linear + moe` W8A8 Int8 直转 @ `seq_len=4096`，精度达标。性能注意：MoE per-expert 路径在小 M_eff 下收益存疑，须 infer 侧 packed `MoEGMM` 实测确认；attention 路径正常。

## 常见问题

- 问题 1：BF16 PPL 明显异常
  - 现象：PPL 远高于预期，或者 quant 结果反而明显优于 BF16
  - 优先检查：blockwise `attention_mask`、`position_embeddings`、`position_ids` 是否和真实路径一致
- 问题 1.1：Qwen3-Next 的 BF16 PPL 低得离谱
  - 现象：blockwise BF16 PPL 异常地低，像是模型看到了后文
  - 优先检查：是否把 layer0 `linear_attention` 捕获到的 `None` mask 误传给了后续 `full_attention` 层；当前修复方式是在 adapter 内显式构造 `full_attention` 的 causal mask，并在 attention wrapper 中带显式 mask 时保持 `is_causal=False`
- 问题 2：MoE 权重加载不对
  - 现象：state dict key 不匹配，或者 expert 权重形状与运行时模块不一致
  - 优先检查：checkpoint 的真实层数、expert 是展开存储还是 packed 存储，以及 adapter 是否需要在加载时做 repack
- 问题 3：最小 PTQ smoke 选错 unit
  - 现象：unit 能枚举出来，但没有任何可训练 PTQ 参数，训练后也不会导出有效结果
  - 优先检查：当前 unit 是否真的暴露了 `trainable_params()` 或 `export_ptq_params()`；对于 packed expert 路径，必要时先用 `shared_expert` 打通 smoke

## 已有个案（同结构已合并）

- [Qwen3-dense](qwen3-dense.md) — `Qwen3` dense：Qwen3-4B（起点）+ Qwen3-8B（纯复用）
- [Qwen3-moe](qwen3-moe.md) — `Qwen3Moe`：Qwen3-30B-A3B + Qwen3-235B-A22B
- [Qwen3.5-3.6](qwen3.5-3.6.md) — `Qwen3_5Moe`：Qwen3.5-35B-A3B + Qwen3.6-35B（`Qwen3_6Moe(Qwen3_5Moe): pass`，零 delta）
- [Qwen3-Next-80B-A3B-Instruct](qwen3-next-80b-a3b-instruct.md) — `Qwen3Next`：MoE + 混合 attention
