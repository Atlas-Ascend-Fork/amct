# DeepSeek Casebook

这一页是 **DeepSeek 系列总览页**，不是具体模型个案。通用坑先读 [L1 跨网络](../cross-model-pitfalls.md) + [L2 结构家族（MLA / MoE / 自定义 modeling）](../structure-family-pitfalls.md)。

## 系列概览

- 当前已覆盖模型：`deepseek_v3_2`、`deepseek_v4`（`glm5` 亦继承 `DeepseekV32`）
- 整体结构特点：这一系列的 decoder block 同时包含 attention 侧和 MoE 侧，attention 路径内部还会分成不同分支，量化时通常需要把 `mla` 和 `moe` 分开看
- 与其他系列最接近的是：MoE 侧能复用通用 `PtqUnit` 和 `apply_quant_to_moe_mlp`，attention 侧则更依赖 DeepSeek 专属的 wrapper 与 target 路由

## 默认参考路径

- 新模型进入时优先参考：[deepseekv3_2.py](../../../../amct_pytorch/common/models/llm/deepseek/deepseek_v3_2/deepseekv3_2.py)、[quant_module.py](../../../../amct_pytorch/common/models/llm/deepseek/deepseek_v3_2/quant_module.py)；更复杂的 MLA+Compressor+Indexer+HC 见 [deepseekv4.py](../../../../amct_pytorch/common/models/llm/deepseek/deepseek_v4/deepseekv4.py)
- 适配优先顺序：先拿 BF16 blockwise baseline，确认 `position_ids` / `attention_mask` 路径一致，再分别打通 attention 侧和 MoE 侧的最小闭环
- 默认优先复用的抽象：`BaseModel`、`PtqUnit`、现有 quant-apply helper、`QuantDeepseekV3MLP`

## 通用适配经验

- 这一系列最容易需要单独 wrapper 的地方通常是 attention 侧分支和 MoE experts
- 默认不要先改 workflow、solver 和 quant module，优先把模型特有逻辑留在 `amct_pytorch/common/models/llm/deepseek/...`
- 适配时最常见的实现风险是：`layer_type` 分支判断不一致、target 路由和真实模块不一致，或者 blockwise 路径漏传 `position_ids` / `attention_mask`

## 通用量化经验

- BF16 baseline 一般先用 Wikitext PPL 的 blockwise 口径拿
- 第一版直转量化通常先分路径验证：attention 侧先看 `mla`，MoE 侧再单独看 `moe`，不要一开始把所有分支绑在一起
- 哪类模块通常更敏感：attention / rotary / cache 相关路径，以及 expert / shared expert 路由
- 什么时候通常要进入 PTQ：直转量化 `delta > 0.2`，或者已经做过一轮粗粒度误差定位之后

## 常见问题

- 问题 1：blockwise attention 路径和真实 decoder 对不齐
  - 现象：BF16 或 quant PPL 明显异常，或者单层 block 输出和整链行为不一致
  - 优先检查：`position_ids`、`attention_mask`、`layer_type` 分支，以及目标 target 是否走到了正确 wrapper
- 问题 2：MoE PTQ unit 枚举或输入匹配失败
  - 现象：expert 输入文件匹配不到、`iter_ptq_units()` 枚举结果异常，或者 routed/shared expert 混在一起
  - 优先检查：`iter_ptq_units()`、`_resolve_unit_input_files()` 里的命名约定，以及 expert metadata 是否和数据落盘命名一致

## 已有个案

- [DeepSeek-V3.2](deepseek-v3.2.md) — 待补（DeepseekV32 基类：MLA+MoE+DSA，moe-only，仅精度）
- [deepseekv4](deepseekv4.md) — 厚（MLA + Compressor + Indexer + HC + grouped wo + 自定义 forward；代码侧适配完成，等权重补 BF16/PTQ）
