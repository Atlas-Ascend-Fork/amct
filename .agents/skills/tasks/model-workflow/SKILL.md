---
name: model-workflow
description: 在 amct_llm 中以总控方式串联新模型适配、量化实验、deploy 导出与交付归档。适用于用户想从“新模型导入”一路推进到“量化验证、deploy 交付、文档归档”，或者需要先判断当前模型处于哪个阶段、下一步该调用哪个 skill。
---

# amct_llm 模型全流程

这个 skill 是模型生命周期的总控 skill。

它不替代已有子 skill，而是负责：
- 先查现状
- 判断当前阶段
- 串联已有 skill
- 优先复用已有结果
- 在 deploy 完成后，继续补齐交付文档

优先串联下面这些已有 skill：
- [../model-adapter/SKILL.md](../model-adapter/SKILL.md)
- [../quantization/SKILL.md](../quantization/SKILL.md)
- [../../quant-tools/deploy-export/SKILL.md](../../quant-tools/deploy-export/SKILL.md)

## 用户调用格式

- 如果用户明确点名调用这个总控 skill，先返回这个 skill 对应的固定输入模板，再开始执行。
- 不替用户改成别的 skill；只有当用户没有指定 skill、只是描述目标时，才由 agent 建议是否走这个总控 skill。
- 统一模板见：[../../reference/skill-input-template.md](../../reference/skill-input-template.md)

## 评测口径强约束

- 当前仓内所有通过 skill 触发的 `Wikitext PPL` 评测，统一使用 `seq_len=4096`。
- 这是默认强约束，不要再按旧口径使用 `2048` 或其他长度。
- 如果用户没有显式指定 `seq_len`，就必须按 `4096` 执行。
- 如果历史结果不是 `seq_len=4096`，默认不能直接和当前结果横向比较，也不能直接当作同口径结论复用。
- 只有用户明确要求其他 `seq_len` 时，才允许偏离该约束；一旦偏离，必须在结论里明确写出本次口径与默认口径不同。

## 先看哪些内容

按下面顺序判断：

1. 当前模型是否已经适配进框架
2. BF16 baseline 是否已经拿到
3. 关闭量化后的浮点等价是否已经验证
4. 最小 PTQ 集成 smoke 是否已经打通
5. 是否已经有量化结果
6. 是否已经进入 deploy 导出阶段
7. deploy 产物是否已经补齐对 infer 仓可直接使用的说明文档

## 阶段判断

把当前任务先归到下面 6 个阶段之一：

1. 未适配
   - 还没有模型注册、block wrapper、最小 PTQ 单元闭环
2. 已适配，未完成 BF16 闭环
   - 代码可能已接入，但 BF16 baseline、浮点等价或最小 PTQ smoke 还没完成
3. 已适配，准备量化
   - BF16 闭环已完成，当前准备直转 / 算法验证 / PTQ 升级
4. 已有量化结果，准备 deploy
   - 量化方案和精度结论已经接受，当前只差 deploy 导出
5. deploy 已完成，待补交付文档
   - 权重已经导出，但 infer 仓需要的 `deploy_quantization.md` 还没补齐
6. 已有完整结果，准备归档或复核
   - 已有结论足够回答问题，当前主要是整理、复用、归档或判断是否要重跑

## 固定流程

1. 先查现有结果：
   - `docs/agent/casebook/...` 是否已有同模型或同系列结论
   - `outputs/`、日志、脚本目录里是否已有 BF16 / quant / deploy 结果
   - 当前代码、模型版本、量化目标、评测口径是否与已有结果一致
2. 如果已有结果足够回答当前问题，先复用并说明口径；只有在代码、模型版本、目标范围或评测口径变化时，才重跑。
3. 先判断当前属于哪个阶段。
4. 如果属于“未适配”或“已适配但 BF16 闭环未完成”，调用 `$model-adapter`。
5. 如果属于“已适配，准备量化”，调用 `$quantization`。
6. 如果属于“已有量化结果，准备 deploy”，调用 `$deploy-export`。
7. 如果属于“deploy 已完成，待补交付文档”，继续调用 `$deploy-export`，但目标改为：
   - 复核 deploy 产物
   - 按模板生成 `deploy_quantization.md`
   - 确认该文档与当前导出目录一一对应
8. 如果属于“已有完整结果，准备归档或复核”，先整理已有结论，再决定是否还要调用上面的子 skill。
9. 如果当前任务跨多个阶段，就按顺序推进：
   - 先 `$model-adapter`
   - 再 `$quantization`
   - 再 `$deploy-export`
10. 任何一次进入 deploy 阶段，都不要默认认为“权重导出完成 = 全流程结束”。
    - 只有当导出目录下已经补齐 `deploy_quantization.md`，并且文档内容足以给 infer 仓使用时，deploy 阶段才算真正闭环。

## 交互节点

这个总控 skill 不是“全自动一路跑到底”，下面这些节点必须停下来和用户确认：

1. 从 `model-adapter` 切到 `quantization` 之前：
   - 先说明当前准备采用的量化方案
   - 如果用户没有明确锁定方案，必须询问是否沿用默认方案，还是要修改 `quant_target / bits / dtype / algos`
2. 如果查到了已有 BF16 / quant / deploy 结果：
   - 先汇报现有结果
   - 如果这些结果足够回答当前问题，不要默认重跑；先确认用户是否仍然要求重跑
3. 第一轮直转量化结果出来后：
   - 不要默认继续升级
   - 必须先确认是接受当前方案、修改量化方案重新测，还是继续升级到算法验证 / PTQ
4. 进入 deploy 阶段前：
   - 先确认用户已经接受当前量化方案
   - 不要把“默认继续导出”当成当然动作

## 分流规则

- 用户明确说“新模型导入 / 适配”，优先从 `$model-adapter` 开始。
- 用户明确说“量化实验 / 直转 / PTQ / 算法验证”，优先转 `$quantization`。
- 用户明确说“导出 deploy 权重”，优先转 `$deploy-export`。
- 用户说的是一整套目标，例如“把新模型接进来并完成第一轮量化，再走到 deploy 交付”，优先用这个总控 skill 统一编排。

## deploy 交付边界

需要明确下面这条边界：

- `amct_pytorch/cli/llm/deploy.py` 和 `amct_pytorch/workflows/llm_deploy.py` 负责导出权重产物。
- `deploy_quantization.md` 不要求由 deploy 代码自动生成。
- `deploy_quantization.md` 由 `$deploy-export` skill 按模板生成，并与本次导出目录绑定。

也就是说，deploy 阶段分成两步：

1. 代码导出权重目录
2. skill 补齐交付说明文档

不要把这两件事混在一起。

## 边界规则

- 这个 skill 自己不维护另一套模型适配、量化或导出规则。
- `model-adapter`、`quantization`、`deploy-export` 仍然是实际执行动作的子 skill。
- 这个 skill 的重点是“判断现在该走哪一步”，不是“重写每一步怎么做”。
- 默认先查现有结果，再决定是否重跑。
- 如果子 skill 已经足够回答当前问题，不要在这里重复展开其内部细则。
- 引用仓库内路径时，一律使用仓库相对路径。

## 输出要求

结束时至少说明：

1. 当前模型处于哪个阶段
2. 本轮复用了哪些已有结果
3. 本轮实际调用了哪些子 skill
4. 为什么当前要先走这一步
5. 当前还缺哪一环
6. 下一步应该调用哪个 skill
7. 如果已经进入 deploy 阶段：
   - deploy 权重是否已导出
   - `deploy_quantization.md` 是否已补齐
   - 若未补齐，为什么当前还不能算真正交付完成
