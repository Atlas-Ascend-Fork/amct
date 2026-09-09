# 模型层开发与审查入口

本文件用于 `amct-coder` 对已有 LLM adapter/wrapper 做局部非结构修复和契约 review。新增模型注册、`PtqUnit` 拆分、quant block 或 attention/MLP/MoE wrapper 的结构性实现，必须加载并遵循 [model-adapter skill](../../quant-tools/model-adapter/SKILL.md)。本文件不重复其适配流程、实验流程和完成标准。

## 职责边界

- 模型专属逻辑放在 `amct_pytorch/common/models/llm/<family>/...`。
- workflow 只负责阶段编排和数据交接，不加入模型名、mask、position 或 wrapper 分支。
- solver 只负责优化和参数生命周期，不读取模型结构或离线输入文件。
- 通用 quant module 不判断模型身份；模型差异由 adapter 表达。

## 定位步骤

1. 从 `MODEL_REGISTRY` 和目标注册类定位 adapter，确认实际调用方。
2. 按问题搜索 `build_quant_block`、`iter_ptq_units`、embedding/block forward、`iter_deploy_bindings` 及对应测试。
3. 局部 bug 沿“捕获端 -> 离线文件 -> adapter 注入 -> block forward -> wrapper 消费”完整核验。
4. 若需要新增类、改变结构、注册链或 unit 拆分，切换到 `model-adapter` skill；若只改变下游接口，按契约 owner 追加对应 reference。

## 跨层契约

修改模型层或其消费者时，至少核对：

| 契约 | 必须一致的两端 |
| --- | --- |
| `PtqUnit.kind`、`save_name` | adapter、extract 文件名、PTQ 参数文件和回载逻辑 |
| hook 与 kwargs | adapter 的 embedding/block forward、`llm-data-workflow` 的保存与加载 |
| quant block 参数 | adapter、solver 的参数收集/finalize、`PtqParamStore` |
| deploy binding | adapter、deploy 导出的权重映射、ignore layers 和 dtype/algo 配置 |

文件、装饰器或 helper 存在不等于调用链已接通；必须同时找到注册、调用方和直接测试证据。

## Attention 规则

按真实 blockwise forward 确认 `attention_mask`、`position_ids`、`position_embeddings`、RoPE 和 cache 参数是否参与计算。当前路径不用的 decode/cache 分支不要保留；优先复用同族 wrapper 和 `attention_forward.py`，只替换量化目标内的投影层。确认 SDPA 的 `attn_mask` 与 `is_causal` 组合合法，混合 attention 使用对应 mask，MLA 按实际需求区分 `attn-linear` 与 `attn-cache`。dense/MoE 实现一致时共用一份 wrapper。关闭量化后必须与原始 float block 等价。

## Wrapper 规则

新增 wrapper 前比较同族实现的初始化、状态和 forward；三者一致就合并，只有模块结构、导出语义、`PtqUnit` 边界或量化路径真实不同才拆分。优先复用 `QuantLinear`、`QuantGatedMLP`、`apply_quant_to_attn()`、`apply_quant_to_moe_mlp()` 和 family helper。packed experts 要确认 PTQ 所需参数已 materialize；shared experts 要和 `build_no_algo_args(args)`、`iter_ptq_units()` 保持一致。`unit.kind` 必须对应 extract 文件，参数必须能保存和回载；loss 为 0 或 NaN 时立即停止检查输入、梯度和 unit 边界。

## 验证

局部修复至少运行目标 adapter 的直接单测；涉及输入捕获、unit 或回载时补对应 workflow mocked 测试。验证应分别说明源码调用链、单测覆盖、集成闭环和真实模型/NPU 验证，不能用 import 成功代替模型适配完成。
