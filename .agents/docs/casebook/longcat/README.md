# LongCat Casebook

这一页是 **LongCat 系列总览页**，不是具体模型个案。通用坑先读 [L1 跨网络](../cross-model-pitfalls.md) + [L2 结构家族](../structure-family-pitfalls.md)。

## 系列概览

- 当前已覆盖模型：`LongCat-Flash-Lite`、`LongCat-Next`
- 典型结构特点：`ngram_embeddings` 输入路径、双 self-attn 分支、双 dense MLP 分支，以及一条 MoE shortcut 路径
- 在当前仓库里最接近的参考：Qwen 的 dense/MoE wrapper，以及公共的 `BaseModel` blockwise 工作流

## 默认参考路径

- 新模型进入时优先参考：[longcat_lite.py](../../../../amct_pytorch/common/models/llm/longcat/longcat_lite/longcat_lite.py)
- `LongCat-Next` 的 text-only 兼容入口在：[longcat_next.py](../../../../amct_pytorch/common/models/llm/longcat/longcat_next/longcat_next.py)
- 优先复用：`BaseModel`、`PtqUnit`、`QuantLinear`、`QuantGatedMLP`
- 除非模型结构无法在 `amct_pytorch/common/models/llm` 中表达，否则不要优先改 workflow 或 solver

## 通用适配经验

- 这个系列最需要单独处理的是输入路径，因为 LongCat 依赖 `ngram_embeddings`
- decoder block 不是简单的单 `attn + mlp` 结构，所以 PTQ unit 的拆分和 hook 路由都需要模型专属映射
- 当 `qk/v head dim` 不对称时，优先固定到更稳的 `eager` attention backend
- 如果顶层 `trust_remote_code` 模型额外依赖 `flash_attn` 等运行时，而 blockwise PTQ 只需要 text-only backbone，可以在 adapter 内直接实例化动态模块里的文本模型类
- 遇到 checkpoint 和默认 config 构造的 `lm_head` 形状不一致时，优先在 adapter 内做最小 patch，不要把兼容逻辑扩散到 workflow

## 通用量化经验

- 先拿 blockwise Wikitext PPL 的 BF16 baseline，并把它作为这个系列的唯一对比基线
- `LongCat-Flash-Lite` 的第一轮 `a8w8-int8` 直转量化在 Wikitext PPL 上已经接近无损
- `LongCat-Next` 目前只完成了 model-adapter 闭环，量化方案还没有正式归档
- 在判断精度之前，先确认 Ascend 运行时环境和 activation quant dtype 接线没有问题

## 常见问题

- 问题 2：int 直转量化一开始就报错，或者激活行为明显异常
  - 优先检查：`ActivationQuantizer` 是否传了 `is_act=True`，以及 activation quant 是否错误地把带符号值当成纯正数处理
- 问题 3：LongCat-Next 顶层模型不能直接空载
  - 优先检查：是否误走了带 `flash_attn` 依赖的顶层多模态模型，而没有切到 text-only 的 `modeling_longcat_ngram` 路径

## 已有个案（同结构已合并）

- [longcat](longcat.md) — `LongcatLite` + `LongcatNext`：LongCat-Flash-Lite（首个，ngram 输入）+ LongCat-Next（`LongcatNext(LongcatLite)`，唯一 delta = `empty_weights_model` 切 text-only backbone）
