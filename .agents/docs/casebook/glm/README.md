# GLM Casebook

这一页是 **GLM 系列总览页**，不是具体模型个案。通用坑先读 [L1 跨网络](../cross-model-pitfalls.md) + [L2 结构家族（MLA / MoE）](../structure-family-pitfalls.md)；GLM5 继承 DeepseekV32，结构经验参考 DeepSeek 系列。

## 系列概览

- 当前已覆盖模型：`glm5`
- 整体结构特点：`GLM5(DeepseekV32)` —— **直接继承 DeepSeek V3.2 适配**（MLA attention + MoE），量化复用 DeepSeek V3 的 attention/MLP wrapper。
- 与其他系列最接近的是：DeepSeek 系列（尤其 `deepseek-v3.2`，同为 `DeepseekV32` 路径）。

## 默认参考路径

- 新模型进入时优先参考：[glm5.py](../../../../amct_pytorch/common/models/llm/glm/glm5/glm5.py)（即 DeepseekV32 路径）+ [deepseek_v3_2/quant_module.py](../../../../amct_pytorch/common/models/llm/deepseek/deepseek_v3_2/quant_module.py)
- 默认优先复用的抽象：`BaseModel`、`apply_quant_to_attn` / `apply_quant_to_moe_mlp`、`QuantDeepseekV3Attention` / `QuantDeepseekV3MLP`

## 通用适配经验

- GLM5 当前是 `DeepseekV32` 的子类，结构性适配几乎全部复用 DeepSeek V3.2；**GLM5 适配器不含 DSA indexer 代码**（indexer 为 DSv3.2 专属，glm5 不走该路径，非缺陷）。
- 通用 MLA/MoE 陷阱见 DeepSeek 系列 README。

## 通用量化经验

- 待补（glm-5.1 适配经验待补；可参考 DeepSeek 系列的 MLA/MoE 量化经验）。

## 已有个案

- [GLM-5.1](glm-5.1.md) — 待补（继承 DeepseekV32，仅精度数据）
