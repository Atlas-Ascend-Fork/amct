# Pangu Casebook

这一页是 **Pangu 系列总览页**，不是具体模型个案。

## 系列概览

- 当前已覆盖模型：`pangu-7b`
- 当前结构特点：这一系列目前接入的是 dense decoder-only 路径，attention 是标准 GQA，自定义 tokenizer 和模型实现通过 `trust_remote_code` 提供
- 当前最需要注意的不是 MoE 或特殊 block 拓扑，而是模型导出的 `transformers` 接口与远端运行环境版本是否兼容

## 默认参考路径

- 新模型进入时优先参考：[pangu_7b.py](../../../../amct_pytorch/common/models/llm/pangu/pangu_7b/pangu_7b.py)
- 适配优先顺序：先 BF16 blockwise baseline，再做关闭量化等价验证，再跑最小 PTQ smoke
- 默认优先复用的抽象：`BaseModel`、`PtqUnit`、`QuantLinear`、`QuantGatedMLP`、现有 quant-apply helper

## 通用适配经验

- 这条系列目前更像“自定义 dense 模型接入”而不是“模型结构复杂”
- 默认只改 `amct_pytorch/common/models/llm/pangu/...`，优先把兼容补丁和模型专属 wrapper 收在 adapter 内
- blockwise 路径最关键的输入是：
  - `attention_mask`
  - `position_ids`
  - `position_embeddings`
- 这条系列的 attention 和 MLP 都可以复用现有通用 quant wrapper 形态，不需要先改 workflow、solver 或 quant module

## 通用量化经验

- 第一轮先拿 BF16 blockwise baseline，确认数值稳定
- `pangu-7b` 的第一轮直转量化已经验证过 `a8w8-int8`
- 当前这条系列里，attention 路径比 MLP 路径更稳：
  - `attn-only` 会把 Wikitext PPL 从 `13.9502` 降到 `13.1908`
  - `mlp-only` 会把 Wikitext PPL 抬到 `14.4075`
  - `attn + mlp` 的整网结果是 `13.5534`
- 所以这一系列后续默认起点不再优先写成“先做 `mlp`”，而是优先考虑 `attn` 或 `attn + mlp`
- 如果模型自带的原生 NPU fused attention 路径在当前环境下不稳定，应先在 adapter 层固定回 eager 路径，再进入量化验证

## 常见问题

- 问题 1：`trust_remote_code` 模型导入失败
  - 现象：导入时缺少 `LossKwargs`、`auto_docstring`、`can_return_tuple` 或 `ROPE_INIT_FUNCTIONS['default']`
  - 优先检查：当前远端 `transformers` 版本是否低于模型导出时代码所依赖的接口版本
- 问题 2：BF16 PPL 明显异常
  - 现象：PPL 离谱偏高，或者原生整网 forward 出现 `nan`
  - 优先检查：模型自带的 NPU fused attention 路径是否和当前环境兼容；必要时在 adapter 层显式关闭并退回 eager attention

## 已有个案

- [Pangu-7B](pangu-7b.md)
