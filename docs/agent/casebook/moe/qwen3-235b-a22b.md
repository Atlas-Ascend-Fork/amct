# Qwen3-235B-A22B

## 为什么这个案例重要

- 这是仓库里第一个 Qwen3 MoE 适配案例。
- 模型自带的配置文件与真实 checkpoint 深度不一致。
- checkpoint 在磁盘上是展开的 per-expert 权重，而运行时模块期待的是 packed expert 张量。

## 参考关系

- 所属系列：Qwen
- 复用的仓内抽象：`BaseModel`、`PtqUnit`、`QuantLinear`
- 最接近的参考思路：现有 Qwen dense 适配，以及 LongCat 的 packed-expert 处理方式

## 适配结论

- 结构判断：
  - `Qwen3MoeDecoderLayer`
  - `Qwen3MoeAttention`
  - `Qwen3MoeSparseMoeBlock`
  - 运行时是 packed experts，磁盘上是展开的 expert 权重
- 实际改动范围：
  - 只动了 `amct_pytorch/common/models/llm/...`
  - 模型适配本身没有改 workflow 或 solver
- 复用了这些抽象：
  - `BaseModel` 的 blockwise 工作流
  - `PtqUnit` 对 `attn` 和 `moe` 的路由
  - `QuantLinear`
- 新增的模型专属实现：
  - `QuantQwen3MoeAttn`
  - `QuantPackedExperts`
  - `PackedExpertView`
  - adapter 内的层数修正和 expert 重组逻辑
- 最终验证状态：
  - BF16 baseline：通过
  - 关闭量化后的浮点等价：通过
  - 最小 PTQ unit smoke：通过，使用 `layer 0 / expert_0`

## 精度与验证

- BF16 baseline：
  - Wikitext PPL，blockwise，`seq_len=2048`
  - `ppl_bf16 = 5.3579`
- 代表层关闭量化验证：
  - `attn` 在 `0 / 47 / 93` 层的测试样本上精确对齐
  - `moe` 在 `0 / 47 / 93` 层上只有很小的 BF16 级漂移
- 挂上 Quant wrapper 但关闭量化后的整链验证：
  - 全量 Wikitext PPL 为 `5.3518`
- 最小 PTQ unit smoke：
  - `quant_target = moe`
  - `layer 0 / expert_0`
  - 输入提取、GT 构造、1 epoch `flatquant`、参数导出和参数回载全部通过

## 量化结论

- 第一轮直转量化方案：
  - Wikitext PPL，blockwise，`seq_len=2048`
  - `quant_target = attn moe`
  - `quant_dtype = int`
  - `a_bits = 8`
  - `w_bits = 8`
- BF16 baseline 复用说明：
  - 模型、代码路径和评测口径都没有变化，所以直接复用适配阶段拿到的 BF16 baseline
- 直转量化结果：
  - `ppl_quant = 5.3942`
  - `delta = 0.0363`
- PTQ 决策：
  - 第一轮不需要进入 PTQ，因为 `delta <= 0.2`
- 最终采用方案：
  - `attn + moe` 的直转 `a8w8-int8`

## 典型问题

- 问题：
  - 配置层数和 checkpoint 真实层数不一致
- 现象：
  - HF 只加载出 5 层，但 checkpoint index 明确包含 `0..93` 共 94 层
- 根因：
  - `config.json` 里 `num_hidden_layers = 5`，而 shard index 反映的真实模型深度是 94
- 修复：
  - 在 adapter 初始化时，根据 weight index 推断真实层数并覆盖 `num_hidden_layers`
- 后续建议：
  - 超大模型不要盲信 config 里的深度字段，先检查 `model.safetensors.index.json`

- 问题：
  - expert 权重不能直接加载到运行时 MoE 模块
- 现象：
  - checkpoint 暴露的是 `mlp.experts.<idx>.gate_proj/up_proj/down_proj.weight`，但运行时模块期待 packed expert 张量
- 根因：
  - checkpoint 和运行时采用了不同的 expert 张量布局
- 修复：
  - 在 adapter 内把展开的 expert 权重重组为 packed `gate_up_proj` 和 `down_proj`
- 后续建议：
  - 动手写 loader 之前，先比对磁盘上的 expert key 和运行时参数名

- 问题：
  - 初始 BF16 PPL 明显错误
- 现象：
  - blockwise BF16 评测一开始得到 `ppl = 466.0364`
- 根因：
  - `do_block_forward` 里没有把 `attention_mask` 传给 block，导致 blockwise 路径与真实 decoder 计算不一致
- 修复：
  - 把捕获到的 `attention_mask` 显式传入每一层 block forward
- 后续建议：
  - 只要 BF16 PPL 明显离谱，先检查 chunk0 loss；这个模型在修完 mask 后，chunk0 loss 从 `5.46875` 回到 `1.2265625`

- 问题：
  - 当前通用 PTQ provider 还不能完整覆盖 attention unit
- 现象：
  - `materialize_gt()` 只会转发 tensor batch，但 attention unit 还需要 `position_embeddings` 和 `attention_mask`
- 根因：
  - 现在的 PTQ provider 接口是 tensor-only，而 Qwen attention unit forward 需要额外上下文
- 当前处理：
  - 最小 PTQ smoke 先使用 MoE expert unit 完成闭环
- 后续建议：
  - 如果后面要做 attention unit PTQ，需要把 PTQ provider 泛化为支持额外上下文输入

## 最终建议

- 下一个 Qwen3 MoE 模型，优先从这个案例和 Qwen 系列 README 开始。
- 先验证 checkpoint 深度和 expert 布局，再去排数值问题。
- 如果 BF16 PPL 看起来离谱，先查 blockwise `attention_mask` 传递，再往更深层定位。
- 对这个家族来说，第一轮量化直接从 `attn + moe` 的 `a8w8-int8` 开始已经是比较稳的方案。

## 已有个案

|  | 量化格式 | 粒度
|--------|-----|-----|
| Attention Linear | A8W8 INT| Per Token |
| MoE Expert（exclude gate） | A8W8 INT | Per Channel |