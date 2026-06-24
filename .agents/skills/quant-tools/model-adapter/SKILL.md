---
name: model-adapter
description: 在 amct 中适配新模型族（注册模型 / 定义 PTQ unit / 接 quant block / 补模型专属 attention·MLP wrapper），保持 workflow·quantizer·algorithm·solver 边界稳定。触发场景：新增 Dense 或 MoE 适配器、为新结构补 wrapper。不适用于：重写主 PTQ 流程、跑量化实验、推荐方案（→ $quant-run / $scheme-recommendation）。
---

# amct 模型适配

## 概述

在 amct 中适配新模型族（注册模型 / 定义 `PtqUnit` / 接 quant block / 补模型专属 attention·MLP wrapper），保持 workflow·quantizer·algorithm·solver 边界稳定；**只做适配，不重写主 PTQ 流程**。

---

## 用户调用格式

- 用户明确点名调用，先返回本 skill 固定输入模板再执行。
- 不替用户改成主流程；仅当用户没指定 skill、只描述目标时，才建议是否回 `$quant-workflow`。
- 固定输入模板见 [../references/skill-input-template.md](../references/skill-input-template.md)。

---

## 先读哪些文件

1. [../../../docs/repo-map.md](../../../docs/repo-map.md)
2. [references/model-adapter.md](references/model-adapter.md)
3. [references/validation-checklist.md](references/validation-checklist.md)

## 评测口径强约束

- 当前模型适配阶段里凡是要落 `Wikitext PPL` 的验证，统一使用 `seq_len=4096`。
- 这是默认强约束，不要沿用旧的 `2048` 口径。
- 如果用户没有显式指定 `seq_len`，就按 `4096` 执行 BF16 baseline 和后续等价性验证。
- 如果历史 baseline 不是 `4096`，默认不能直接当作当前适配结论复用；除非明确说明只是参考，不是同口径结果。
- 只有用户明确要求其他 `seq_len` 时，才允许偏离默认口径，并且必须在结论里写清楚。

## 固定流程

1. 先读 `repo-map`，再抽查锚点文件，确认当前 map 仍有效。
2. 先查现有结果：
   - 仓库里是否已有同模型的适配结论
   - `.agents/docs/casebook/...` 是否已有可复用经验
   - `outputs/` 或日志里是否已有 BF16 baseline / wrapper 校验 / 最小 PTQ 集成 smoke 结果
3. 如果已有结果足够回答当前问题，先复用并说明口径；只有在代码、模型版本、评测口径或目标范围变化时，才重跑。
4. 解析新模型结构，至少明确：
   - block / attention / MLP class
   - dense 还是 MoE
   - experts 是显式模块还是 packed tensor
   - `PtqUnit` 准备怎么拆
5. 先确定 attention 的最小实现目标：
   - 当前框架默认目标是 `blockwise + PPL / PTQ / deploy`
   - 不要默认按上游源码完整保留 generate / decode / cache 分支
   - 如果当前任务不会用到 `past_key_values` 或 cache，就不要为了“以后可能会用到”而先保留复杂分支
   - 如果 dense / MoE 两个 attention wrapper 初始化和 `forward` 一样，优先共用一套实现，不要机械拆两份
6. 先定改动范围，默认只改 `amct_pytorch/common/models/llm/...`。
7. 先制定复用计划，优先复用：
   - `PtqUnit`
   - `QuantLinear`
   - `QuantGatedMLP`
   - 现有 quant-apply helper
8. 先打通 BF16 推理，拿到 baseline。
9. 再验证：关闭量化后，wrapper 前向与原始浮点模块一致或足够接近；这一步属于适配正确性验证，不属于量化效果判断。
10. 再打通一个最小 PTQ 集成 smoke：
   - 至少一个 quant block
   - 至少一个可枚举的 `PtqUnit`
   - 至少一次 unit 输入 / GT 准备
   - 至少一次参数导出与回载
11. 最后总结适配结果、剩余风险和未接通部分。
12. 结束前判断是否需要同步 `repo-map`、`casebook` 和 `Agent Docs`；触发则更新，不触发也要明确说明理由。

## Attention 适配原则

这个原则必须优先于“照搬源码”：

1. 先按当前框架真实目标做最小化实现。
   - 当前默认目标是 `blockwise` 路径上的 BF16 / PPL / PTQ / deploy。
   - 不是完整生成态推理。
2. 不要为了适配通路，强行保留并不参与当前路径的复杂步骤。
   - 例如 `past_key_values`
   - 例如 decode-only cache 分支
   - 例如当前始终为 `None` 的附加参数
3. `forward` 里的参数如果在当前路径下确实恒为 `None`，要先判断它是否真的参与计算。
   - 如果不参与，就不要为了“形式上对齐源码”而保留额外分支
   - 只保留当前路径真正需要的参数和步骤
4. attention wrapper 的实现目标是“足够正确且尽量简单”，不是“最大程度复刻上游源码的所有枝杈”。
5. 同系列模型如果 attention 结构和当前任务口径一致，优先共用实现。

## Wrapper 合并原则

除了 attention，MLP / MoE-MLP 也遵守同样的合并原则：

1. 如果同系列两个 wrapper 的初始化、状态和 `forward` 一样，就只保留一份实现。
2. 不要为了兼容旧类名、少改 import，长期保留行为完全重复的类。
3. 只有在下面这些情况出现时，才拆成两份 wrapper：
   - 底层模块结构真的不同
   - 导出语义不同
   - PTQ unit 边界不同
   - 量化路径不同
4. “一个是 dense 版本，一个是 moe 版本”本身不构成拆分类的理由；关键看实现是否真的不同。
5. 适配时要主动判断“能不能合并”，不要为了完全不动旧代码而保留重复实现。

## 边界规则

- 优先改 `amct_pytorch/common/models/llm/...`。
- 先查现有结果，再决定是否重跑；不要默认重复执行同口径实验。
- 优先复用 `PtqUnit`、`QuantLinear`、`QuantGatedMLP` 和现有 quant-apply helper。
- 不要把模型特有逻辑塞进 `amct_pytorch/workflows/...`。
- 除非模型真的需要新的优化范式，否则不要改 `amct_pytorch/common/optimization/...`。
- 除非现有接口无法表达该模型结构，否则不要改 `amct_pytorch/quantization/modules/...`。
- 默认不要抽新基类，除非同一模式已经在多个模型族中重复出现。
- `Agent Docs` 只在文档层级或职责边界变化时更新，不因普通模型适配而默认改动。
- 引用仓库内代码、脚本和文档路径时，一律使用仓库相对路径，例如 `amct_pytorch/common/models/llm/...`、`.agents/docs/...`；不要写绝对路径。
- 这里的最小 PTQ 集成 smoke 只用于验证 adapter 已接到 PTQ 主链，不在这里下 PPL、`delta`、PTQ 升级或量化方案优劣的结论；这些属于 `$quant-workflow`。
- 优先顺序是：
  - MLP
  - attention
  - MoE / packed experts

## 文档写回触发

适配完成后的文档写回（repo-map / 系列 casebook README / 个案 / Agent Docs，及默认不做）统一见 [../../../docs/README.md](../../../docs/README.md) 的「文档写回触发」。

## 完成标准

满足下面条件前，不要认为适配完成：

1. 至少一层的 quant wrapper 能正常构建。
2. BF16 baseline 已拿到，或明确记录为什么当前拿不到。
3. 已验证适配路径上“关闭量化后，wrapper 仍保持浮点等价”。
4. 至少一个 PTQ unit 能被枚举，并完成最小 PTQ 集成 smoke。
5. 已跑过验证清单。
6. 已完成文档写回判断；触发则已更新，未触发则已说明理由。

## 输出要求

结束时必须说明：

1. 新模型的结构判断结果。
2. 实际改动范围是否越界。
3. 复用了哪些现有抽象。
4. BF16 baseline 是否已跑通。
5. 浮点等价检查是否通过。
6. 最小 PTQ 集成 smoke 是否已打通。
7. 还剩哪些风险或未完成项。
8. 如果发生重跑，为什么旧结果不能直接复用。
9. attention 是否按“最小化实现原则”收敛；如果没有，为什么。
10. 是否更新了 `repo-map`；如果没有，为什么。
11. 是否更新了系列 `casebook README` 或个案；如果没有，为什么。
12. 是否需要更新 `Agent Docs`；如果不需要，为什么。
