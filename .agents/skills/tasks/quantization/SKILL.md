---
name: quantization
description: 在 amct 中以总控方式推进已适配模型的量化实验与决策。适用于编排第一轮直转方案推荐、直转评测、算法推荐、算法验证，并在需要时继续做粗粒度误差定位、判断是否从直转升级到 PTQ、以及给出下一步决策。
---

# amct 量化实验

这个 skill 只用于**已适配模型的量化执行与升级决策**。

这个 skill 是量化阶段的**总控 skill**。
如果用户只需要量化流程里的某一段，优先分流到 `quant-tools/` 下对应的 skill，而不是默认走完整端到端流程。

按下面顺序阅读：

1. [references/metrics-and-thresholds.md](references/metrics-and-thresholds.md)
2. [references/direct-quant.md](references/direct-quant.md)
3. 如需升级到 PTQ，再读 [references/ptq-escalation.md](references/ptq-escalation.md)
4. 如网络结构特殊，再读 [references/network-strategies.md](references/network-strategies.md)

## 用户调用格式

- 如果用户明确点名调用这个 skill，先返回这个 skill 对应的固定输入模板，再开始执行。
- 不替用户改成主流程；只有当用户没有指定 skill、只是描述目标时，才建议是否回到 `$model-workflow`。
- 固定输入模板见：[../../reference/skill-input-template.md](../../reference/skill-input-template.md)

## 评测口径强约束

- 当前量化阶段的主判断指标仍然是 `Wikitext PPL`，但统一口径固定为 `seq_len=4096`。
- 这是默认强约束；只要 skill 内没有用户明确指定其他长度，就必须按 `4096` 跑 BF16 baseline、直转量化和 PTQ 后评测。
- 历史结果如果不是 `seq_len=4096`，默认不能直接与当前 `ppl_bf16 / ppl_quant / delta` 对比。
- 如果用户明确要求其他 `seq_len`，允许偏离；但必须在输出结论里明确标记本次不是默认口径。

## 分流入口

如果当前问题只覆盖量化实验与升级决策中的一段，优先转到下面这些子 skill：

- 第一轮直转量化方案推荐：
  - [../../quant-tools/scheme-recommendation/SKILL.md](../../quant-tools/scheme-recommendation/SKILL.md)
  - 适用于用户只想先拿方案参考，不想立刻跑实验
- 用户已指定直转方案，只想看 `ppl_bf16 / ppl_quant / delta`：
  - [../../quant-tools/direct-quant-eval/SKILL.md](../../quant-tools/direct-quant-eval/SKILL.md)
  - 适用于用户已有自己的量化方案，只想验证直转结果
- 用户已经完成量化方案与精度判断，只想把当前直转或 PTQ 结果导出成 deploy-ready 权重：
  - [../../quant-tools/deploy-export/SKILL.md](../../quant-tools/deploy-export/SKILL.md)
  - 这属于导出执行阶段；在这里不重新做量化评测、PTQ 或精度是否可接受的判断
- 已知直转掉点，想推荐下一步算法，或想先拿算法候选：
  - [../../quant-tools/algorithm-recommendation/SKILL.md](../../quant-tools/algorithm-recommendation/SKILL.md)
  - 适用于用户想根据结构、casebook 和已有掉点特征选算法
- 已经明确指定算法，想验证它是否比直转更好：
  - [../../quant-tools/algorithm-validation/SKILL.md](../../quant-tools/algorithm-validation/SKILL.md)
  - 适用于用户只想看某个算法的实际收益

## 固定流程

1. 先查现有结果：
   - `.agents/docs/casebook/...` 是否已有同模型、同口径结论
   - `outputs/`、日志或脚本目录里是否已有 BF16 baseline / quant 结果
   - 当前代码、参数口径、模型版本是否与已有结果一致
2. 如果已有结果足够回答当前问题，先整理并输出，再决定是否需要重跑；只有在代码、模型版本、评测口径或量化目标变化时，才重跑。
3. 先检查前置条件：
   - 模型已经接进框架
   - BF16 baseline 已拿到
   - 已验证关闭量化后保持浮点等价
4. 判断当前请求是否只覆盖量化流程中的一段：
   - 只要第一轮方案，转 `$scheme-recommendation`
   - 只评测指定直转方案，转 `$direct-quant-eval`
   - 已经进入导出执行阶段，只做 deploy 导出，转 `$deploy-export`
   - 只推荐下一步算法，转 `$algorithm-recommendation`
   - 只验证指定算法，转 `$algorithm-validation`
5. 如果当前要跑完整端到端量化流程，先明确本次量化目标：
   - 量化哪些模块
   - bit / dtype
   - 当前只看 Wikitext PPL
6. 如果当前还没有可执行的第一轮直转方案，先调用 `$scheme-recommendation` 产出“量化方案推荐卡”，再确定本轮起步方案。
7. 拿到直转方案后，调用 `$direct-quant-eval` 跑第一轮直转评测，拿到 `ppl_bf16`、`ppl_quant` 和 `delta`。
8. 如果 `delta <= 0.2`，当前方案可接受，可以先停止升级。
9. 如果 `delta > 0.2`，先判断当前需要的是“算法候选”还是“算法验证”：
   - 还没选算法时，先调用 `$algorithm-recommendation`
   - 已指定算法或已选好候选时，调用 `$algorithm-validation`
10. 如果直转和算法验证之后仍不能回答当前问题，再做一轮粗粒度误差定位。
11. 只有在粗定位之后，才允许在最小范围引入 PTQ。
12. 每轮实验结束后，必须输出下一步决策：
   - 停止
   - 补定位
   - 小幅升级
13. 结束前判断是否需要同步 `casebook`、`repo-map` 和 `Agent Docs`；触发则更新，不触发也要明确说明理由。

## 交互节点

这个量化总控 skill 不是“默认一路跑到底”。下面这些节点必须停下来和用户确认：

1. 在第一轮量化实验开始前：
   - 如果用户没有明确锁定量化方案，不能默认直接沿用推荐方案开跑。
   - 必须先说明当前准备采用的默认方案，并确认用户是否接受，还是要修改：
     - `quant_target`
     - `bits`
     - `quant_dtype`
     - `algos`
2. 如果查到了已有 BF16 / quant / deploy 结果：
   - 先汇报现有结果。
   - 如果这些结果已经足够回答当前问题，不要默认重跑；必须先确认用户是否仍然要求重跑。
3. 第一轮直转量化结果出来后：
   - 不要默认继续升级。
   - 必须先确认是：
     - 接受当前方案
     - 修改量化方案后重测
     - 继续升级到算法推荐 / 算法验证 / PTQ
4. 在从直转评测切到 PTQ 之前：
   - 必须先说明为什么当前结果不足以停在直转。
   - 不能默认把 PTQ 当成下一步；需要得到用户确认后再升级。
5. 在进入 deploy 阶段之前：
   - 必须先确认用户已经接受当前量化方案与结果。
   - 不能把“量化实验结束”自动等同于“继续 deploy”。

## 边界规则

- 这个 skill 不负责新模型适配；如果 BF16 baseline 还没拿到，回到 `$model-adapter`。
- 这个 skill 负责端到端量化实验与升级决策；如果用户只需要其中一个子任务，优先分流到 `$scheme-recommendation`、`$direct-quant-eval`、`$algorithm-recommendation` 或 `$algorithm-validation`。
- 如果用户已经接受当前量化方案，只想把现有量化结果导出成 deploy-ready 权重，说明任务已经进入导出执行阶段，优先转 `$deploy-export`。
- 方案推荐、直转评测、算法推荐、算法验证，优先复用 `quant-tools/` 下对应的 skill；不要在这里重复维护另一套独立规则。
- 先查现有结果，再决定是否重跑；不要默认重复执行同口径评测。
- 默认先做直转量化，再考虑 PTQ。
- 默认先从整网方案开始，再缩到 block / unit。
- 误差定位只做粗粒度，不追求一次找准根因。
- PTQ 先在最小范围实验，不一上来全模型全算法。
- 当前主判断指标是 Wikitext PPL 和 `delta <= 0.2`。
- 如果量化后 PPL 明显优于 BF16 很多，先检查链路，不直接当成功。
- 普通量化实验默认优先回写 `casebook`，不默认修改 `Agent Docs`。
- 引用仓库内代码、脚本和文档路径时，一律使用仓库相对路径，例如 `src/...`、`docs/...`、`outputs/...`；不要写绝对路径。

## 文档写回触发条件

### 更新系列 `casebook README`

满足下面任一条时，至少更新所属系列页：

- 某个模型首次拿到稳定的 BF16 baseline 和第一轮可复现的量化结论。
- 系列当前的默认量化起点、敏感模块、升级策略或常见问题需要补充或修正。
- 新结果推翻了系列页里的默认经验。

### 新增或更新个案文件

满足下面任一条时，新增或更新 `<series>/<case>.md`：

- 出现新的直转量化路径或新的 PTQ 升级路线。
- 出现值得复用的代表性问题、异常现象或排障结论。
- 结果明显偏离该系列已有经验。
- 已有个案需要按新结果修订。

### 更新 `repo-map`

只有在下面情况之一出现时，才更新 `.agents/docs/repo-map.md`：

- 任务实际改动了 workflow、solver、数据链路、导出加载语义或模型主注册链。
- 当前 `repo-map` 的“当前接通状态”已经和代码不一致。

### 更新 `Agent Docs`

只有在下面情况之一出现时，才更新 `.agents/docs/README.md`：

- 文档层级或职责边界发生变化。
- `Agent Docs` 记录的目录结构与真实目录不一致。

### 默认不做

- 单次失败实验、单次不稳定结果，不默认回写 `casebook` 结论。
- 如果 BF16 baseline 自身不稳定，不默认新增量化经验。
- 如果当前任务没有带来新的可复用信息，可以说明无需更新文档。

## 输出要求

结束时必须说明：

1. 本轮实验的量化方案是什么
2. `ppl_bf16`、`ppl_quant`、`delta` 分别是多少
3. 当前结果是否达标
4. 是否做过粗粒度误差定位
5. 是否已经进入 PTQ
6. 下一步决策是什么
7. 本轮调用了哪些 `quant-tools` 子 skill；如果没有调用，为什么
8. 如果发生重跑，为什么旧结果不能直接复用
9. 是否更新了系列 `casebook README` 或个案；如果没有，为什么
10. 是否需要更新 `repo-map`；如果不需要，为什么
11. 是否需要更新 `Agent Docs`；如果不需要，为什么
