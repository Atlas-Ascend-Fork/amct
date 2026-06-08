# Qwen3-Next-80B-A3B-Instruct

## 为什么这个案例重要

- 这是仓库里第一条稳定接通的 `qwen3_next` 适配案例。
- 同一个模型家族内部同时存在 `linear_attention` 和 `full_attention` 两种 attention 路径。
- 当前 `transformers==5.3.0` 环境下，checkpoint 在磁盘上是展开的 per-expert 权重，而运行时 MoE 模块使用 packed experts。

## 参考关系

- 所属系列：Qwen
- 复用的仓内抽象：`BaseModel`、`PtqUnit`、`QuantLinear`
- 最接近的参考思路：`qwen3_5_moe.py` 的混合 attention 路由，以及 `qwen3_moe.py` 的 MoE 权重重组和 packed expert 视图

## 适配结论

- 结构判断：
  - 48 层 decoder
  - block 内部混合 `linear_attention` 和 `full_attention`
  - 运行时是 packed experts，磁盘上是展开的 expert 权重
- 实际改动范围：
  - 动了 `amct_pytorch/common/models/llm/__init__.py`
  - 新增了 `amct_pytorch/common/models/llm/qwen/qwen3_next/qwen3_next.py`
  - 没有改 workflow、solver 和 quant module 主链
- 复用了这些抽象：
  - `BaseModel` 的 blockwise 工作流
  - `PtqUnit` 对 `attn`、`moe` 和 `mlp` 的路由
  - `QuantLinear`
  - 现有 quant-apply helper
- 新增或补齐的模型专属实现：
  - `QuantQwen3NextAttn`
  - `QuantQwen3NextMLP`
  - `QuantPackedExperts`
  - `PackedExpertView`
  - adapter 内的 causal mask 路由和 expert 重组逻辑
- 最终验证状态：
  - BF16 baseline：通过
  - 关闭量化后的浮点等价：通过
  - 最小 PTQ unit smoke：通过，使用 `layer 0 / shared_expert`

## 精度与验证

- BF16 baseline：
  - Wikitext PPL，blockwise，`seq_len=2048`
  - `ppl_bf16 = 5.7378`
- 代表层关闭量化验证：
  - `attn-only` 在 `3 / 23 / 47` 层的测试样本上精确对齐
  - `moe-only` 在 `0 / 23 / 47` 层上只有很小的 BF16 级漂移
- 挂上 Quant wrapper 但关闭量化后的整链验证：
  - 全量 Wikitext PPL 为 `5.7348`
- 最小 PTQ unit smoke：
  - `quant_target = mlp`
  - `layer 0 / shared_expert`
  - 输入提取、GT 构造、1 epoch `flatquant`、参数导出和参数回载全部通过
  - 参数文件：`layer_0_shared_expert.pt`

## 典型问题

- 问题：
  - 初始 BF16 PPL 明显不对，低得离谱
- 现象：
  - blockwise BF16 一开始得到的 PPL 接近模型“看到了后文”的量级，不符合预期的 `5.7` 左右
- 根因：
  - `BaseModel.do_embedding_forward()` 只会捕获 layer0 的 `attention_mask`
  - 而 `qwen3_next` 的 layer0 是 `linear_attention`，捕获到的是 `None`
  - 后续 `full_attention` 层如果继续复用这份 mask，就会丢掉 causal mask
- 修复：
  - 在 adapter 内分别维护 `linear_attention_mask` 和 `full_attention_mask`
  - `full_attention_mask` 通过 `create_causal_mask(...)` 显式构造
  - attention wrapper 在已经显式传 mask 时保持 `is_causal=False`
- 后续建议：
  - 对混合 attention 家族，只要 BF16 PPL 好得不正常，先查 blockwise 的 mask 传递链

- 问题：
  - checkpoint 和运行时 expert 结构不一致
- 现象：
  - checkpoint 暴露的是 `mlp.experts.<idx>.gate_proj/up_proj/down_proj.weight`
  - 运行时模块期待的是 packed `gate_up_proj` 和 `down_proj`
- 根因：
  - 当前 `transformers==5.3.0` 环境下，`Qwen3NextExperts` 的运行时参数布局和磁盘存储布局不同
- 修复：
  - 在 adapter 内把展开的 expert 权重重组为 packed experts
- 后续建议：
  - 先确认当前运行环境里的 `transformers` 版本，再决定 expert 路径是逐 expert 还是 packed

- 问题：
  - 最小 PTQ smoke 不适合直接选 routed expert
- 现象：
  - `expert_0 ... expert_511` 都能枚举，但当前最小 smoke 更适合先走 `shared_expert`
- 根因：
  - 当前 packed expert 路径更偏向量化执行包装；最小 PTQ 集成验证更需要一个稳定、可训练、可导出的 unit
- 当前处理：
  - 最小 PTQ smoke 先用 `shared_expert` 打通
- 后续建议：
  - 真正做 routed expert PTQ 时，再单独整理 packed expert 的 trainable/export 语义

## 最终建议

- 后续同家族模型优先参考这个案例和 Qwen 系列 README。
- 在当前环境里，先把 `transformers==5.3.0` 当成适配基线，再去判断 expert 结构。
- 只要模型内部混有 `linear_attention` 和 `full_attention`，就不要默认一套 mask 路径打到底。
- 先把 BF16 baseline、关闭量化验证和最小 PTQ smoke 做干净，再进入量化 skill。
