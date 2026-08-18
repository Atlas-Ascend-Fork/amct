---
name: amct-coder
description: |
  AMCT 仓源代码编码、实现规划、行为诊断与代码 review 等辅助编码的入口：用于定位 workflow/CLI、solver、dtype、通用 quant module、LLM PTQ 算法、deploy 实现、classic graph、public API、tests、docs 和 examples 的修改落点，检查分层、越层、活跃调用链和上下游契约，并完成相应实现与最小验证。不适用于新增或结构性修改模型 adapter/wrapper/attention/MLP/MoE（转 $model-adapter）、Ascend C kernel/PyTorch C++ extension/schema/Meta/CMake（转 $amct-ops-dev）、实际运行量化或评测及生成 deploy 交付产物（转 $quant-workflow / $quant-run / $deploy-export）；评审已有文件或拟议落点是否越层仍使用本 skill。
---

# AMCT Coder

## 定位

把 `amct-coder` 作为 AMCT 编码、实现规划、行为诊断与代码 review 等辅助编码的入口。先判断任务职责归属，再选择操作手册，最后交付并核对证据。

不要把本 skill 当成量化实验或模型适配编排器。任务实质属于更专门 skill 时，实际加载并遵循它，由它形成职责范围内的实现或计划，不要模仿它写完模型结构、算子或实验方案后口头声称已经委派，也不要只回复“应使用某 skill”后结束。

默认使用中文。修改代码前，简短说明主落点层和允许改动范围。

## 固定流程

```text
用户请求
  -> 1. 按动作和当前已发现 skill 判断职责归属
  -> 2. 声明仓库边界，按 Reference 导航选择一个主场景手册
  -> 3. 读取源码锚点，完成编码、规划、诊断或 review
  -> 4. 执行或核对验证证据并汇总
```

## 1. 先判任务职责归属

先根据用户要执行的动作和当前会话已发现 skill 的 `name + description` 判断职责，不维护外部 skill 的完整副本。任务实质已被更专门 skill 覆盖时，不要为外部职责预读 amct-coder 的 repo-map 或场景手册，直接加载并遵循该 skill，后续加载和交付由它决定。

修改 eval/deploy 等**实现代码**仍属于 amct-coder；运行命令拿结果或生成交付产物属于操作流程。不要只看 `eval`、`deploy`、`attention`、`校准` 等关键词，必须结合动作、代码体系和被修改契约判断。用户点名 skill 时，只有不违反仓库边界和该 skill 的“不适用”条件才遵从。

混合任务按职责归属拆分：本 skill 处理辅助编码范围内的部分，其余部分进入对应 skill；全部完成后统一检查契约并汇总。

### 区分实现与落点 review

- 用户要求**新增或结构性实现**属于专门 skill 的职责时，第一步直接进入对应 skill。
- 用户要求**评审一个明确文件或拟议方案是否合理**时，由 amct-coder 审查当前落点：只读取该落点所需的 repo-map 切片、一个主操作手册和源码锚点，判断是否越层或错误复用；确认正确实现属于外部职责后，再进入对应 skill 继续。
- 用户报告局部 bug 时，先核验现象和根因。若最终需要新增模型结构、NPU kernel 或执行实验，再按核验结果路由；不要只因题面出现模型名、算子名或 eval 关键词提前改判。

### 事实门禁

形成具体修改计划前，区分三类信息：已由源码或用户输入确认的事实、尚待确认的契约、依赖这些契约的条件式方案。

按固定顺序消歧，不要把可自查的问题推回用户：先查源码、测试和调用方；查得到就据此判定或直接否决错误方案，并给出修改建议；只有查不到且不同答案会改变实现方案时，按「契约缺失时的处理形态」处理。

- 先查现有实现和测试，避免重复新增已经存在的类、注册或闭环。
- 输入输出、数值语义、兼容范围或用户控制方式未明确时，不自行发明类名、CLI 参数、schema、文件名或 payload 字段；先用现有实现、测试和调用方消歧，无法消歧按「契约缺失时的处理形态」处理。
- 不影响公开契约和交付产物的局部选择（变量名、日志措辞、等价实现取舍）自行决定并在最终说明里点明，不为此询问用户。默认值可能改变兼容性和用户可见行为，不属于当然的局部等价选择。
- 上述契约未冻结时，具体源码文件或函数只能列为待核验的候选锚点，不能写成已经确定的最终落点。
- 区分“函数存在”和“活跃调用链已接通”；没有核对调用方、注册或编排入口前，不把未调用 helper 当成当前运行路径。
- 用户描述与当前源码冲突时，以可引用的源码行为为准；可以反证题设并停止无依据修改。
- 没有排查完整数据链前，不使用“唯一根因”“已经完整覆盖”等强结论。

### 契约缺失时的处理形态

第一档能查到就查完：越层落点、错误复用和 BLOCKER 由源码、注册入口、活跃调用链和测试直接否决，不为已有答案的问题询问用户。

查不出来时先完成不依赖该答案的部分，不要让一个待确认项挂住整个任务。剩余部分按交付物区分：

- **只读规划**：条件分支本身就是交付物。列出各分支的前提、落点差异和验证差异，不必先冻结契约。
- **实际编码**：关键契约未冻结且不同答案产出不兼容实现时，直接询问用户，不写投机实现。按错误假设写出的完整版本返工成本高于提问，且容易把假设固化成事实。

询问遵循以下形态：

- 收敛成封闭选项并说明各选项影响面，不做开放式提问。只有源码惯例、兼容性或实现成本存在可引用证据时才给推荐项；没有依据时明确“无默认推荐”，不伪造偏好。
- 一次问完本轮全部阻塞项，不逐条追问。
- 一个待确认项不阻塞整个任务；其余不依赖该答案的部分继续做完。
- 拿不到答复时，只读规划保留条件分支；已有证据支持推荐项时可以沿推荐分支展开并标注未经确认，没有推荐依据时不替用户选择。实际编码停在契约确认前，只交付已完成部分和契约清单，不落盘投机代码。

## 2. 声明仓库边界

实际编码、实现规划、行为诊断、架构或代码行为 review 都要先声明边界；只检查代码风格、版权、贡献门槛和模板的纯规范合规 review 默认跳过本节，一旦结论涉及实现行为、架构落点、调用链或上下游契约，就不再是纯合规检查，必须回到本节补齐资料。

1. 读取 [repo-map.md](../../docs/repo-map.md) 的「分层边界」和「更新规则」。
2. 抽查候选落点直接相关的源码锚点，确认 map 与当前实现一致。
3. 在内部形成边界声明：主落点层、允许范围、禁止事项、必要辅助层、是否需要写回 repo-map。

repo-map 是稳定架构契约，源码是当前行为事实。repo-map 前置读取不代表默认写回，写回判据以已读的 repo-map「更新规则」为准。两者冲突时，先判断源码是否越层；源码代表预期的新边界或核心流程时，把 repo-map 修正纳入本次任务。

## 3. Reference 导航

下表列出从本入口直接进入的 reference。表格用于让资料可发现，不表示它们都是平级子能力，也不要求一次性全部读取。先选择一个主场景手册；其它文档只在表中条件成立时加载。实际操作 Issue/PR 的贡献流程由 `project-rules.md` 引出，不在本表重复。

| 类型 | Reference | 用途与加载时机 |
| --- | --- | --- |
| 项目规范 | [project-rules.md](references/project-rules.md) | AMCT 代码风格、版权、贡献门槛、模板和 GitCode 操作规范；修改代码前，或做规范合规 review 时读取 |
| 职责辅助 | [delegation.md](references/delegation.md) | 编码与专门流程的职责交界不清时读取；只解释稳定的高歧义交界，不是外部 skill 清单 |
| 定位辅助 | [repo-scenarios.md](references/repo-scenarios.md) | 无法确定代码体系或入口时读取 |
| 主场景 | [dtype-quant-module.md](references/dtype-quant-module.md) | 量化 dtype 和通用 quant module 相关开发与 review |
| 主场景 | [llm-algorithm.md](references/llm-algorithm.md) | LLM 量化算法相关开发与 review |
| 主场景 | [llm-model-unit.md](references/llm-model-unit.md) | LLM 模型层相关开发与 review |
| 主场景 | [llm-data-workflow.md](references/llm-data-workflow.md) | LLM 数据与 workflow 相关开发与 review |
| 主场景 | [ptq-solver-params.md](references/ptq-solver-params.md) | PTQ solver 和参数生命周期相关开发与 review |
| 主场景 | [deploy-export-code.md](references/deploy-export-code.md) | 量化模型导出代码相关开发与 review |
| 主场景 | [classic-graph.md](references/classic-graph.md) | classic 量化与 graph_based 相关开发与 review |
| 主场景 | [ops-boundary.md](references/ops-boundary.md) | `amct_pytorch` 调用 `amct_ops` 的 Python 侧边界开发与 review |
| 主场景 | [public-api-tests.md](references/public-api-tests.md) | 公共接口、打包、测试和用户资料相关开发与 review |
| 阶段辅助 | [stage-impact.md](references/stage-impact.md) | 修改 LLM workflow、阶段产物或跨阶段契约时读取 |
| 分层辅助 | [layer-rules.md](references/layer-rules.md) | 落点、越层或上下游边界仍不明确时读取 |
| 模型辅助 | [attention-rules.md](references/attention-rules.md) | attention 相关 review 或局部非结构修复时读取 |
| 模型辅助 | [wrapper-rules.md](references/wrapper-rules.md) | MLP、MoE 或 wrapper 相关 review 和局部非结构修复时读取 |
| 交付验证 | [validation.md](references/validation.md) | 选择静态检查、测试或构建；实际编码在修改后读取并执行匹配的最小验证，只做规划或 review 时按需读取相关部分 |

主场景描述只提供粗粒度导航。具体触发、源码锚点、执行步骤和 BLOCKER 以对应 reference 为准；不要在本表继续枚举算法、参数、文件或未来可能出现的全部需求。

多个主场景同时可能命中时，先根据用户动作、目标入口、被修改契约和活跃调用方确定主手册。仍无法唯一确定时，读取 `repo-scenarios.md` 定位代码体系；只有确认存在真实跨契约影响时，才把第二个主场景作为辅助手册加载。主手册选择仍依赖尚未冻结的关键契约时，按「契约缺失时的处理形态」处理。

命中时完整读取主 playbook 后再查目标源码和测试。只有实际改变上游契约或下游产物时，才把对应 playbook 作为辅助资料完整读取；不要仅因文件列表相邻而加载。

未命中现有手册时不要强行套入最相近项。先用搜索工具查公共入口、注册点、调用方和直接测试，以 repo-map 判断所属层，并根据相关契约形成任务规划。没有 playbook 兜底时落点判断只靠 repo-map 分层边界，因此实际编码前按「契约缺失时的处理形态」的提问形态与用户确认落点和改动范围再动手；只读规划直接交付该规划，不额外确认。若该场景形成稳定、重复出现的新代码体系，再更新现有手册或新增 reference。

各场景的详细执行步骤、候选锚点和 BLOCKER 由对应 playbook 定义，不在本文件重复。

## 4. 按需检查阶段影响

阶段只描述 LLM 数据流影响，不决定代码职责归属。仅在修改 LLM CLI/workflow、阶段产物或跨阶段契约时检查；classic、ops 和纯 public API/packaging 标记为“不适用”。

需要逐阶段核对时按 Reference 导航读取 `stage-impact.md`。通用能力可能同时影响多个阶段，不要强行选择单阶段；列出需要闭环的受影响阶段即可。

## 5. 渐进披露与读取预算

reference 是任务规则，源码列表只是候选锚点。不要把 playbook 中列出的源码路径理解为全文必读清单。按以下层级加载：

1. 只加载 §2 要求的 repo-map 切片和一个主 playbook；各 Reference 的用途和加载时机从 §Reference 导航查询，不另行推导。
2. 用搜索工具定位题目中的符号、调用方和直接测试；首轮默认选择 2～4 个最相关的源码/测试锚点，只读相关定义、调用链和断言切片。
3. 边界或契约仍不清楚时，先扩大同一调用链的源码切片，再按 §Reference 导航追加必要的辅助 reference；只有真实跨契约影响已经成立时才追加其它主 playbook。
4. 模型结构、checkpoint/config 或模型相关精度异常按 L1 -> L2 -> L3 读取 [casebook](../../docs/casebook/README.md)；主路径或 map/源码差异仍不清楚时，只补读 repo-map 对应 section。
5. 项目规范和验证规则按 §Reference 导航的加载时机读取，不把它们提前计入主场景资料。

达到以下条件时停止扩张上下文：仓库边界已确定；活跃调用链或未接通状态已有源码证据；已能区分确认事实、待确认契约和条件式方案；修改面与最小验证范围已经明确。不要为了穷举目录、填满 playbook 文件列表或证明无关分支而继续读取。

默认预算是 repo-map 最小切片 + 1 个主 playbook + 首轮 2～4 个源码/测试锚点；实际编码再加编码前规范。辅助 Reference 默认最多 1 份，项目规范和验证规则不计入主场景资料预算。真实跨契约任务按数据流分批加载，不要一次性把所有主 playbook、源码和测试都读完，每批重新检查是否已经满足停止条件。读取预算用于防止预读，不用于阻止完成任务；若必须超出，说明触发扩张的冲突或缺失证据。

同一会话处理连续任务时，可以复用已经读取且未变化的资料，但要重新按当前动作做职责判断，不能沿用前一任务的结论。记录本轮新增读取和复用项；不重复读取完整大文件，也不能用旧结论替代当前题设的事实核验。

## 全局 BLOCKER

遇到 BLOCKER 时只停止当前错误方案。替代路径明确、仍在用户目标和已有授权内时直接继续；只有缺少会改变实现方案的关键输入、契约或授权时才询问用户，并按「契约缺失时的处理形态」处理。信息缺失不等于 BLOCKER：先推进不依赖该信息的部分，不要把整个任务挂住。

- 不在 CLI、workflow、solver 或通用 quant module 中加入模型专属分支。
- 不在本 skill 内新增或结构性修改模型 adapter/wrapper/attention/MLP/MoE；路由 `model-adapter`，回来后只做契约 review。
- 不混用 LLM new-path `ALGO_REGISTRY` 与 classic `AlgorithmRegistry`。
- 新 dtype 要闭环数值语义、注册、支持位宽/粒度和实际消费入口；只有影响持久化或部署格式时才检查导出/回载。
- 新算法要闭环注册、`targets`、用户选择入口和实际调用链；只有产生可持久化参数时才检查保存/加载。
- 新 public API 要闭环定义、导出/懒加载、兼容性和用户可见测试；仅在安装或使用方式变化时更新 packaging/docs/examples。
- `PtqUnit.kind`、extract 文件命名、PTQ 参数保存/加载、eval/deploy `--algos` 和 deploy binding 必须保持一致。
- 新 NPU 算子实现路由 `amct-ops-dev`；上层调用不得绕过 schema/Meta/import 边界。

按层的详细检查项和落点判断见 [layer-rules.md](references/layer-rules.md)，本文件不复制。

## 交付与验证

以源码和测试为事实依据，优先复用现有抽象，限制改动在边界声明范围内。实际编码在修改前完成规范和边界读取并说明主落点，修改后执行匹配的静态检查、单测或构建；只做规划或 review 时核对既有和建议的验证证据。

最终说明只汇总本轮实际执行过的动作，不为未执行的步骤补写结果：

- 改了代码：主场景或未分类探查结果、改动层与文件、确认事实和待确认项、复用抽象、已跑与未跑验证、repo-map/casebook/docs/examples 是否需更新。
- 没改代码：分析边界与主场景、事实和待确认项、发现或方案、已核对与仍缺失的验证证据；不声称执行过修改后验证。
- 只做了纯规范合规检查：检查过的项目规则、发现和验证证据完整性；没有声明边界时不虚构主场景、改动层或 repo-map 结论。

所有情况都要区分“源码存在”“单测覆盖”“集成闭环”“真实模型或硬件验证”，不能用低一级证据替代高一级结论。缺 NPU、CANN、模型权重或数据集时明确写出，不能把未跑实验表述为已验证。
