# Qwen3.5-35B-A3B

## 为什么这个案例重要

- 这是仓库里第一条稳定接通的 `qwen3_5moe` 适配案例。
- 同一个模型内部同时存在 `linear_attention` 和 `full_attention` 两种 attention 形态。
- checkpoint 上的 MoE 权重已经是 packed 布局，不需要像 `Qwen3-235B-A22B` 那样先做 per-expert 重组。

## 参考关系

- 所属系列：Qwen
- 复用的仓内抽象：`BaseModel`、`PtqUnit`、`QuantLinear`
- 最接近的参考思路：`qwen3_5.py` 的 attention / blockwise 路径，以及 `qwen3_moe.py` 的 MoE 路由思路

## 适配结论

- 结构判断：
  - 40 层 decoder
  - `layer_types` 是重复的 `linear_attention, linear_attention, linear_attention, full_attention`
  - 运行时是 MoE block，磁盘上的 expert 权重已经是 packed 张量
- 实际改动范围：
  - 只动了 `amct_pytorch/common/models/llm/qwen/qwen3_5/qwen3_5_moe.py`
  - 没有改 workflow、solver 和 quant module 主链
- 复用了这些抽象：
  - `BaseModel` 的 blockwise 工作流
  - `PtqUnit` 对 `attn` 和 `moe` 的路由
  - `QuantLinear`
  - `apply_quant_attn`
  - `apply_quant_moe_mlp`
- 新增或补齐的模型专属实现：
  - `qwen3_5moe` 的 `build_quant_block()` 路由
  - packed expert 的 `QuantPackedExperts`
  - 单 expert 视图 `PackedExpertView`
- 最终验证状态：
  - BF16 baseline：通过
  - 关闭量化后的浮点等价：通过
  - 最小 PTQ unit smoke：通过，使用 `layer 0 / shared_expert`

## 精度与验证

- BF16 baseline：
  - Wikitext PPL，blockwise，`seq_len=2048`
  - `ppl_bf16 = 6.5638`
- 代表层关闭量化验证：
  - `attn-only` 在 `3 / 19 / 39` 层上的测试样本与原始浮点 block 足够接近
  - `moe-only` 在 `0 / 20 / 39` 层上的测试样本精确对齐
- 挂上 Quant wrapper 但关闭量化后的整链验证：
  - 全量 Wikitext PPL 为 `6.5568`
- 最小 PTQ unit smoke：
  - `quant_target = moe`
  - `layer 0 / shared_expert`
  - 输入提取、GT 构造、1 epoch `flatquant`、参数导出和参数回载全部通过
  - 参数文件：`layer_0_shared_expert.pt`

## 典型问题

- 问题：
  - 旧代码路径仍在调用不存在的 `apply_quant_mla()`
- 现象：
  - 一进入 quant block 构建就会撞到接口错误，无法继续做 step8 或 PTQ
- 根因：
  - `qwen3_5moe` 继承自 `Qwen3_5`，实际应该复用的是 `apply_quant_attn()` 和 `apply_quant_moe_mlp()`，而不是 DeepSeek 风格的 MLA 专用接口
- 修复：
  - 把 `qwen3_5moe` 的 quant block 构建统一收回到现有 Qwen 公共接口
- 后续建议：
  - Qwen 分支先看当前 dense 适配器的公共接口，不要直接复用 DeepSeek 的 MLA 入口

- 问题：
  - `linear_attention` 和 `full_attention` 不能按同一条 attention 量化路径处理
- 现象：
  - 如果不区分 layer type，就会把两种 attention 结构混进同一套 wrapper 逻辑里
- 根因：
  - `qwen3_5moe` 的 block 内部本来就是混合 attention 结构
- 当前处理：
  - 模型适配阶段先保证 BF16 和关闭量化验证都走通
  - `linear_attention` 暂时保持 BF16，不纳入这次最小量化闭环
- 后续建议：
  - 真正做 attention 量化时，要单独给 `linear_attention` 补专属 wrapper 和验证链

- 问题：
  - packed expert 路径下，最小 PTQ smoke 不一定适合直接选 `expert_0`
- 现象：
  - unit 能枚举，但可能没有任何可训练 PTQ 参数
- 根因：
  - 当前 packed expert 这条实现更偏向量化执行包装，而不是结构算法训练入口
- 当前处理：
  - 最小 PTQ smoke 先用 `shared_expert` 打通
- 后续建议：
  - 真正要做 routed expert 的 PTQ 时，再单独整理 packed expert 的 trainable/export 语义

## 最终建议

- 下一个 `qwen3_5moe` 家族模型，优先从这个案例和 Qwen 系列 README 开始。
- 先确认 checkpoint 上的 experts 是 packed 还是展开，再决定是否需要重组 loader。
- 先把 BF16 baseline 和关闭量化等价验证做干净，再进入量化。
- 对这条模型分支，attention 侧要先把 `linear_attention` 和 `full_attention` 分开看，不要默认一把梭。
