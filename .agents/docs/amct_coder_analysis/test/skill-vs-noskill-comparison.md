# amct-coder 有 / 无 Skill 对比测评

## 1. 测评目的

本次对比关注的不是两份回答谁写得更长，而是 Agent 在面对同一组 AMCT 规划任务时，是否能够：

- 先识别正确的代码体系和职责边界；
- 根据活跃调用链而不是相似实现确定落点；
- 区分已确认事实、未知契约和条件式方案；
- 闭环 CLI、workflow、algorithm、solver、保存/回载及验证；
- 在题目跨出编码职责时正确路由到其它 Skill 和人工确认门。

对比材料：

- `../test-skill.md`：可发现并使用 `amct-coder` Skill 时的六题回答；第 487 行以后是后附 reviewer 审查，不属于被测 Agent 原始回答。
- `../test-noskill.md`：没有该 Skill 时的六题原始回答，未附同格式 reviewer 评分。

本文只整理已有答案和静态审查结论，没有重新执行 cold test，也没有运行模型、量化或 NPU/CANN 验证。

## 2. 总体结论

`amct-coder` 带来了明确但有限的增益：Agent 更容易进入 LLM new-path，能识别 registry、adapter、solver 等分层，也更愿意保留未知契约。六题相对结果为 **有 Skill 4 胜、1 平、1 负**。

但“相对更好”不等于“可以实施”。有 Skill 回答仍频繁把 helper 存在当成调用链闭环，跨阶段状态没有收口，且会使用不存在的 CLI 或提前固定模型方案。当前结论应表述为：**Skill 改善了定位和边界意识，但尚未把 Agent 稳定提升到准确、可执行的端到端规划水平。**

评分只能作为辅助信号：

| 版本 | 参考分数 | 口径 |
| --- | ---: | --- |
| 有 Skill | 约 3.5/10 | 来自 `test-skill.md` 后附 reviewer 的六题评分 |
| 无 Skill | 约 2.5/10 | 按相同关注点做的回顾性启发式估计，不是统一 rubric 实评分 |

两项分数来源不同，不能当作严格的定量 A/B 结果。

## 3. Agent 处理任务的方式

| 环节 | 无 Skill | 有 Skill | 实际影响 |
| --- | --- | --- | --- |
| 任务入口 | 从题目关键词和相似实现开始全仓搜索 | 先尝试判断职责、代码体系和主 playbook | 有 Skill 更少直接混用 classic、experimental 与 LLM new-path |
| 代码定位 | 看到功能相似就倾向复用 | 更关注 registry、adapter、solver 和层间契约 | 主落点通常更接近当前框架设计 |
| 事实与假设 | 很早确定类名、文件、参数和模型结构 | 会列待冻结契约和条件分支 | 有 Skill 降低了部分无依据设计，但仍会过早锁定候选落点 |
| 调用链核验 | 常以“类或函数存在”推断功能可用 | 已有活跃链意识，但执行不稳定 | 两版都会误判保存、回载、finalize 或 deploy 已接通 |
| 跨阶段闭环 | 侧重局部实现细节和改动清单 | 能列出更多职责层，但缺少统一状态机 | observe、solve、finalize、save/load、eval/deploy 仍容易断链 |
| 输出形态 | 内容长、方案和代码片段多，确定性偏强 | 更短、更结构化，边界说明更多 | 可读性改善，但压缩后仍可能掩盖关键事实错误 |

可概括为两种典型路径：

```text
无 Skill：关键词 -> 相似实现 -> 选定方案/文件 -> 展开大量细节 -> 补验证清单
有 Skill：职责路由 -> 分层/reference -> 源码锚点 -> 条件式方案 -> 验证矩阵
```

有 Skill 的预期路径更合理，但实际回答常在“源码锚点”之后过早结束核验，没有继续追到调用者、阶段产物和下游消费者。

## 4. 六题横向结果

| 测试 | 相对结果 | 无 Skill 的处理 | 有 Skill 的处理 | 仍未解决的问题 |
| --- | --- | --- | --- | --- |
| 1. BATQuant + MXFP4 | 有 Skill 胜 | 主要套用 classic `BaseQuantizeModule` / GPTQ 路径，并自行选择 Hessian、weight-only 等契约 | 进入 LLM new-path 的 `ALGO_REGISTRY`、structure transform 和 `BlockwiseSolver` | 可由源码确认的 transform 与参数收集被留作问题；保存链和 CLI 参数仍错误 |
| 2. HIF4 dtype | 有 Skill 胜 | 大量落点位于 classic 配置、quantize/deploy op，并自行定义 HIF4 语义 | 定位到 dtype registry，意识到编码、粒度、适用范围和 deploy payload 未冻结 | 漏掉 CLI、tensorwise deploy 和完整消费入口，仍提前确定文件和接口 |
| 3. 敏感层识别 | 有 Skill 胜 | 围绕 classic quantize/pass 新增全局 sensitivity analyzer | 将度量放在算法侧、决策放在 workflow 侧，更接近 new-path 边界 | 成对输出、Hessian 来源、score 聚合、跳层保存/回载及 eval/deploy 语义未闭环 |
| 4. GPTQ | 平 | 虽识别 new-path registry，但用较长设计描述直接移植 classic GPTQ | 更明确指出统计型算法与 trainable 算法生命周期不同 | 两版都未闭环 GT observe、无 trainable 参数时的数据遍历、一次性 finalize 和优化权重回载 |
| 5. Attention + MLP 联合校准 | 有 Skill 负 | 对 joint target、联合求解器和对比实验展开更完整 | 较早固定为 `attn+mlp` 整体 block unit，并误判递归保存已存在 | 两版都遗漏或误判 CLI choices、双 wrapper 注入、block 原始输入、kwargs、保存回载和公平基线 |
| 6. Kimi-K3 PTQ | 有 Skill 胜 | 直接假设复用 Qwen3，并转向未被当前主链证明的 experimental FlatQuant | 更接近 adapter 与 workflow 边界，也识别模型结构需要确认 | 未正确进入 `quant-workflow`；混淆 FP16/BF16、INT4 weight-only/W4A4，并发明模型落点 |

这里的“胜”只表示两份未达标答案中的相对表现，不表示该题通过。

## 5. Skill 的有效增益

从六题可以确认三类稳定改善：

1. **代码体系识别更准确。** BATQuant、HIF4、敏感层识别和 Kimi-K3 四题中，有 Skill 回答更少把 classic 或 experimental 的相似实现直接当成目标主链。
2. **分层意识更清楚。** Agent 开始主动讨论 registry、algorithm、workflow、solver、adapter 的职责，不再只给文件清单。
3. **未知契约意识增强。** 对新 dtype 和模型结构，Agent 能列出待确认项，而不是完全自行定义语义。

这些增益主要来自主 `SKILL.md` 的职责路由、事实门禁和 reference 导航，说明入口规则方向有效。

## 6. Skill 仍存在的漏洞

### 6.1 活跃调用链核验不稳定

Agent 仍会把 `export_ptq_params()`、递归遍历或其它 helper 的存在，直接解释成训练保存和 eval/deploy 回载已闭环。规划中缺少固定的反查动作：谁调用、何时调用、写出什么、谁消费、strict load 如何处理。

### 6.2 跨阶段生命周期没有统一收口

GPTQ 和敏感层题暴露出相同问题：局部算法接口看似合理，但 observe、GT、solve、finalize、保存、回载和部署不是一个完整状态机。现有规则能提示各阶段，却不能稳定强制 Agent 逐阶段出示证据。

### 6.3 事实门禁执行不到位

Agent 会把源码可查事实继续列为待确认项，也会使用当前不存在的 `--w_bits`、`--a_bits` 等 CLI 参数。这说明“先查源码”目前更像原则，尚未成为交付前的硬门禁。

### 6.4 复杂任务路由不完整

端到端 Kimi-K3 任务应由 `quant-workflow` 编排，并按阶段进入模型适配、方案、执行和判读能力。有 Skill 回答仍停留在 `amct-coder` 视角，人工确认门和职责交接没有真正发生。

### 6.5 局部规则正确不保证总体方案正确

测试 5 是最明显的反例：有 Skill 回答更早选中一个看似符合分层的 block unit 方案，却没有核对真实输入、wrapper 注入和保存链，结果反而弱于无 Skill 版本。这说明 Skill 需要强化“反证候选方案”，而不只是帮助 Agent 更快选层。

## 7. 共同问题

两种环境下都反复出现以下问题：

- 使用旧版或不存在的 CLI 参数，命令无法作为当前调用链证据；
- 混淆“源码存在”“单测覆盖”“集成闭环”“真实模型/硬件验证”；
- GPTQ 的统计、优化、finalize、保存和回载生命周期不完整；
- 联合校准没有闭环入口、数据、unit、solver、持久化和公平实验基线；
- 混淆 FP16 与 BF16、INT4 weight-only 与 W4A4；
- 缺少模型权重、数据集或硬件证据时仍给出过强结论。

## 8. 对后续测评的要求

下一轮不能继续用两种不同评分来源直接比较。建议固定同一 rubric，至少分别评分：

| 维度 | 核心问题 |
| --- | --- |
| 路由 | 是否进入正确 Skill、代码体系和主 playbook |
| 源码事实 | 文件、符号、参数和调用链是否与当前 commit 一致 |
| 计划闭环 | 入口、数据、状态、产物、消费者和验证是否完整 |
| 假设控制 | 是否区分事实、待确认契约和条件分支 |
| 证据等级 | 是否区分静态存在、单测、集成和真实环境结果 |

A/B 测试还应保持相同源码 commit、原始 prompt、模型配置、权限、超时和上下文；由不知道分组的 reviewer 独立评分。原始 prompt、Agent 输出和 reviewer 结果应分文件保存，避免 reviewer 内容被下一轮 Agent 读取。

## 9. 最终判断

本轮 A/B 结果证明 `amct-coder` **有方向性价值，但尚未达到可靠可用**。它主要改善了“去哪里找”和“哪些职责需要区分”，没有稳定解决“当前链路是否真的工作”以及“跨阶段方案是否完整”。后续改进重点应放在活跃调用链、状态生命周期和统一评分回归，而不是继续增加更多文件名或实现示例。
