---
name: amct-coder
description: |
  AMCT 仓源代码编码、调试与 review 入口：用于定位 workflow/CLI、solver、dtype、通用 quant module、LLM PTQ 算法、deploy 实现、classic graph、public API、tests、docs 和 examples 的修改落点，检查分层、越层、活跃调用链和上下游契约，并完成相应实现与最小验证。
---

# AMCT Coder

## 定位

把 `amct-coder` 作为 AMCT 编码与 review 的入口。先判断任务职责归属和仓库边界，再按修改场景选择操作手册，最后按实际改动做最小验证。

不要把本 skill 当成量化实验或模型适配编排器。命中外部 skill 时切换到对应流程并继续完成请求，不要只回复“应使用某 skill”后结束。BLOCKER 只阻止错误落点或错误方案；正确替代方案明确时继续执行。

“切换到对应流程”表示实际加载并遵循外部 skill，由它形成职责范围内的实现或计划；不能由 amct-coder 模仿外部 skill 写完模型结构、算子或实验方案后，口头声称已经委派。

默认使用中文。修改代码前，简短说明主落点层和允许改动范围。

## 固定流程

```text
用户请求
  -> 1. 按动作和当前已发现 skill 判断职责归属
  -> 2. amct-coder 承接后读取 repo-map 边界，声明落点
  -> 3. 匹配场景操作手册；未分类时先探查入口和调用链
  -> 4. 必要时检查阶段影响并加载辅助资料
  -> 5. 读取源码锚点，编码或 review
  -> 6. 有实际改动时读取 validation.md，执行最小验证
```

## 1. 先判任务职责归属

先根据用户要执行的动作和当前会话已发现 skill 的 `name + description` 判断职责，不维护外部 skill 的完整副本。任务实质已被更专门 skill 覆盖时，直接加载并遵循该 skill；不要为外部职责预读 amct-coder 的 repo-map 或场景手册，也不要只回复 skill 名后结束。

修改 eval/deploy 等**实现代码**仍属于 amct-coder；运行命令拿结果或生成交付产物属于操作流程。不要只看 `eval`、`deploy`、`attention`、`校准` 等关键词，必须结合动作、代码体系和被修改契约判断。用户点名 skill 时，只有不违反仓库边界和该 skill 的“不适用”条件才遵从。

混合任务按职责归属拆分：本 skill 处理边界、review 和非外部 skill 代码，其余部分进入对应 skill；全部完成后统一检查契约并汇总。

### 区分实现与错误落点 review

- 用户要求**新增或结构性实现**当前已发现专门 skill 的职责时，第一步直接进入对应 skill；不要先加载 amct-coder 的 repo-map 或场景操作手册。
- 用户要求**评审一个明确文件或拟议方案是否合理**时，由 amct-coder 审查当前落点。只读取该错误落点所需的 repo-map 切片、一个主操作手册和源码锚点；确认正确实现属于外部职责后，再进入对应 skill 继续。
- 用户报告局部 bug 时，先核验现象和根因。若最终需要新增模型结构、NPU kernel 或执行实验，再按核验结果路由；不要只因题面出现模型名、算子名或 eval 关键词提前改判。

### 事实门禁

形成具体修改计划前，区分三类信息：已由源码或用户输入确认的事实、尚待确认的契约、依赖这些契约的条件式方案。

按固定顺序消歧，不要把可自查的问题推回用户：先查源码、测试和调用方；查得到就据此判定或直接否决错误方案；只有查不到且不同答案会改变实现方案时，按「契约缺失时的处理形态」处理。

- 先查现有实现和测试，避免重复新增已经存在的类、注册或闭环。
- 输入输出、数值语义、兼容范围或用户控制方式未明确时，不自行发明类名、CLI 参数、schema、文件名或 payload 字段；先用现有实现、测试和调用方消歧，无法消歧按「契约缺失时的处理形态」处理。
- 不影响公开契约和交付产物的局部选择（变量名、日志措辞、默认值、等价实现取舍）自行决定并在最终说明里点明，不为此询问用户。
- 上述契约未冻结时，具体源码文件或函数只能列为待核验的候选锚点，不能写成已经确定的最终落点。
- 区分“函数存在”和“活跃调用链已接通”；没有核对调用方、注册或编排入口前，不把未调用 helper 当成当前运行路径。
- 用户描述与当前源码冲突时，以可引用的源码行为为准；可以反证题设并停止无依据修改。
- 没有排查完整数据链前，不使用“唯一根因”“已经完整覆盖”等强结论。
- 连续任务只能复用稳定的源码事实和未变化的 reference；当前题目中的“新算法”“某 helper”“新模型”等占位对象，不得继承上一题的具体名称、状态字段、CLI、文件名、payload 或设计决策。输出具体符号前，确认它来自当前用户输入或本轮源码证据。

### 契约缺失时的处理形态

第一档能查到就查完：越层落点、错误复用和 BLOCKER 由源码、注册入口、活跃调用链和测试直接否决，不为已有答案的问题询问用户。

查不出来时先完成不依赖该答案的部分，不要让一个待确认项挂住整个任务。剩余部分按交付物区分：

- **只读规划**：条件分支本身就是交付物。列出各分支的前提、落点差异和验证差异，不必先冻结契约。
- **实际编码**：关键契约未冻结且不同答案产出不兼容实现时，直接询问用户，不写投机实现。按错误假设写出的完整版本返工成本高于提问，且容易把假设固化成事实。

询问遵循以下形态：

- 收敛成封闭选项，给出推荐项及各选项影响面，不做开放式提问。
- 一次问完本轮全部阻塞项，不逐条追问。
- 一个待确认项不阻塞整个任务；其余部分继续做完。
- 拿不到答复时，只读规划按推荐项继续并标注未经确认；实际编码停在契约确认前，只交付已完成部分和契约清单，不落盘投机代码。

编码与专门流程的交界不清时读 [delegation.md](references/delegation.md)；该文件只解释稳定的高歧义交界，不作为外部 skill 清单。

## 2. 强制声明仓库边界

只有确认由 amct-coder 承接编码或 review 后，才执行本节；已经路由到外部 skill 的任务不要预读 repo-map。

1. 读取 [repo-map.md](../../docs/repo-map.md) 的「分层边界」和「更新规则」。
2. 抽查候选落点直接相关的源码锚点，确认 map 与当前实现一致。
3. 在内部形成边界声明：主落点层、允许范围、禁止事项、必要辅助层、是否需要写回 repo-map。

repo-map 是稳定架构契约，源码是当前行为事实。两者冲突时，先判断源码是否越层；源码代表预期的新边界或核心流程时，把 repo-map 修正纳入本次任务。

repo-map 前置读取不代表默认写回。写回判据以第 1 步已读的 repo-map「更新规则」为准，本文件不复制该清单。

## 3. 匹配场景操作手册

先判代码体系：LLM blockwise/new-path、classic graph、`amct_ops` Python 调用边界或 public API/packaging。无法定位体系或入口时读 [repo-scenarios.md](references/repo-scenarios.md)。

下表是已沉淀编码场景的操作手册索引，不是封闭的仓库分类，也不表示关键词的唯一归属。命中时选择**一个主操作手册**；公开面、测试或上下游闭环只作为辅助项。按动作和被修改契约判断，不按目录相邻或关键词堆叠判断。

| 场景操作手册 | 使用场景 | Playbook |
| --- | --- | --- |
| `dtype-quant-module` | dtype、bit policy、fake/real quant、quantizer、通用 quant module | [dtype-quant-module.md](references/dtype-quant-module.md) |
| `llm-algorithm` | LWC/LAC/AutoRound/OmniQuant/FlatQuant 类算法、`targets`、训练/统计参数导出 | [llm-algorithm.md](references/llm-algorithm.md) |
| `llm-model-unit` | 模型契约 review、契约对齐、局部非结构 adapter bug；结构实现转 `model-adapter` | [llm-model-unit.md](references/llm-model-unit.md) |
| `llm-data-workflow` | eval/extract/PTQ workflow、dataloader、离线输入、GT、PPL、逐层 forward | [llm-data-workflow.md](references/llm-data-workflow.md) |
| `ptq-solver-params` | solver、optimizer、scheduler、unit 调度、PTQ 参数保存/加载/回载 | [ptq-solver-params.md](references/ptq-solver-params.md) |
| `deploy-export-code` | deploy workflow、safetensors/index、`quantization_config`、导出实现 | [deploy-export-code.md](references/deploy-export-code.md) |
| `classic-graph` | classic quantize/convert/algorithm register、graph_based、QAT、剪枝、蒸馏 | [classic-graph.md](references/classic-graph.md) |
| `ops-boundary` | `amct_pytorch` 调用 `amct_ops` 的 import/schema/wheel/Python 侧边界 | [ops-boundary.md](references/ops-boundary.md) |
| `public-api-tests` | 顶层 API、`__all__`、懒加载、CLI choices、packaging、tests/docs/examples 闭环 | [public-api-tests.md](references/public-api-tests.md) |

`校准`、`评测`、`部署` 等词只描述阶段或业务目的，不能单独决定操作手册。以校准为例：classic 调用链内的校准属于 `classic-graph`；LLM 校准样本、离线输入和 GT 属于 `llm-data-workflow`；算法如何消费校准统计属于 `llm-algorithm`；通用 observe/min-max/scale 语义属于 `dtype-quant-module`。

命中时完整读取主 playbook 后再查目标源码和测试。只有实际改变上游契约或下游产物时，才把对应 playbook 作为辅助资料完整读取；不要仅因文件列表相邻而加载。

未命中现有手册时不要强行套入最相近项。先用 `rg` 查公共入口、注册点、调用方和直接测试，以 repo-map 判断所属层并完成当前任务；若该场景形成稳定、重复出现的新代码体系，再更新现有手册或新增 reference。

各场景的详细执行步骤、候选锚点和 BLOCKER 由对应 playbook 定义，不在本文件重复。

## 4. 按需检查阶段影响

阶段只描述 LLM 数据流影响，不决定代码职责归属。仅在修改 LLM CLI/workflow、阶段产物或跨阶段契约时检查；classic、ops 和纯 public API/packaging 标记为“不适用”。

需要逐阶段核对时读 [stage-impact.md](references/stage-impact.md)。通用 dtype、quant module 或算法可能同时影响多个阶段，不要强行选择单阶段；列出需要闭环的受影响阶段即可，按数据流顺序处理，不预读全部阶段 playbook。

## 5. 渐进披露与读取预算

reference 是任务规则，源码列表只是候选锚点。不要把 playbook 中列出的源码路径理解为全文必读清单。按以下层级加载：

1. 确认由 amct-coder 承接后，只读 repo-map 的「分层边界」「更新规则」；命中场景手册时再读一个主 playbook。
2. 用 `rg` 定位题目中的符号、调用方和直接测试；首轮默认选择 2～4 个最相关的源码/测试锚点，只读相关定义、调用链和断言切片。
3. 边界或契约仍不清楚时，再扩大同一链路的源码切片；只有真实跨契约影响已经成立时，才追加一个辅助 reference。
4. 实际修改代码后、准备选择验证命令时，读取 [validation.md](references/validation.md)。纯只读规划默认不加载；review 只有在判断测试证据等级或验证完整性时才加载。代码风格、版权头和 pre-commit 属于该文件的静态验证部分。
5. 需要创建 Issue/PR 或读写 GitCode 时，读取 [contribution-flow.md](references/contribution-flow.md)。只做本地改动时不加载。

以下风险出现时再追加资料：

- 落点或越层仍不清楚：读 [layer-rules.md](references/layer-rules.md)。
- attention/wrapper review 或局部模型 bug：读 [attention-rules.md](references/attention-rules.md) 或 [wrapper-rules.md](references/wrapper-rules.md)。
- 模型结构、checkpoint/config 或模型相关精度异常：按 L1 -> L2 -> L3 读取 [casebook](../../docs/casebook/README.md)。
- 主路径或 map/源码差异仍不清楚：只补读 repo-map 对应 section 并扩大源码锚点抽查。

达到以下条件时停止扩张上下文：仓库边界已确定；活跃调用链或未接通状态已有源码证据；已能区分确认事实、待确认契约和条件式方案；修改面与最小验证范围已经明确。不要为了穷举目录、填满 playbook 文件列表或证明无关分支而继续读取。

默认预算是 repo-map 最小切片 + 命中时 1 个主 playbook + 首轮 2～4 个源码/测试锚点；辅助 reference 默认最多 1 份。真实跨契约任务按数据流分批加载，每批重新检查是否已经满足停止条件。读取预算用于防止预读，不用于阻止完成任务；若必须超出，说明触发扩张的冲突或缺失证据。

同一会话处理连续任务时，可以复用已经读取且未变化的资料，但要重新按当前动作做职责判断，不能沿用前一任务的结论。记录本轮新增读取和复用项；不重复读取完整大文件，也不能用旧结论替代当前题设的事实核验。

## 全局 BLOCKER

遇到 BLOCKER 时只停止当前错误方案。替代路径明确、仍在用户目标和已有授权内时直接继续；只有缺少会改变实现方案的关键输入、契约或授权时才询问用户，并按「契约缺失时的处理形态」处理。信息缺失不等于 BLOCKER：先推进不依赖该信息的部分，不要把整个任务挂住。

- 不在 CLI、workflow、solver 或通用 quant module 中加入模型专属分支。
- 不在本 skill 内新增或结构性修改模型 adapter/wrapper/attention/MLP/MoE；路由 `model-adapter` 后做契约 review。
- 不混用 LLM new-path `ALGO_REGISTRY` 与 classic `AlgorithmRegistry`。
- 新 dtype 要闭环数值语义、注册、支持位宽/粒度和实际消费入口；只有影响持久化或部署格式时才检查导出/回载。
- 新算法要闭环注册、`targets`、用户选择入口和实际调用链；只有产生可持久化参数时才检查保存/加载。
- 新 public API 要闭环定义、导出/懒加载、兼容性和用户可见测试；仅在安装或使用方式变化时更新 packaging/docs/examples。
- `PtqUnit.kind`、extract 文件命名、PTQ 参数保存/加载、eval/deploy `--algos` 和 deploy binding 必须保持一致。
- 新 NPU 算子实现路由 `amct-ops-dev`；上层调用不得绕过 schema/Meta/import 边界。

按层的详细检查项和落点判断见 [layer-rules.md](references/layer-rules.md)，本文件不复制。

## 编码与验证

以源码和测试为事实依据，优先复用现有抽象，限制改动在边界声明范围内。修改前说明主落点；修改后读取 `validation.md`，只运行与实际改动匹配的静态检查、单测或构建。

最终说明：命中的场景操作手册（或未分类探查结果）和改动层、已确认事实与待确认项、复用的抽象、已跑验证、未跑验证及原因、repo-map/casebook/docs/examples 是否需要更新。区分“源码存在”“单测覆盖”“集成闭环”“真实模型或硬件验证”，不能用低一级证据替代高一级结论。缺 NPU、CANN、模型权重或数据集时明确写出，不能把未跑实验表述为已验证。
