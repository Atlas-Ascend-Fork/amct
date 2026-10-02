---
name: amct-coder
description: |
  AMCT 仓源码级编码辅助入口。用于实现规划、代码修改、源码行为诊断和代码 review，覆盖 LLM PTQ CLI 及相关 workflow/solver/data、量化模块（dtype/quant module）与 LLM PTQ 算法、deploy、public API 及其测试/文档的落点、调用链和契约检查。不用于量化实验执行、精度/算法收益判读、模型结构适配、classic/graph_based 或 NPU 算子实现。
---

# AMCT Coder

## 判动作

修改 eval/deploy 等**实现代码**属于 amct-coder；运行命令拿结果或生成交付产物属于操作流程。不要只看 `eval`、`deploy`、`attention`、`校准` 等关键词，必须结合动作、代码体系和被修改契约判断。

一次请求含多个动作时按动作拆开：源码级部分本 skill 处理，其余部分进入对应 skill；全部完成后统一检查契约并汇总，不重新接管其它 skill 的内部职责。

- **评审一个明确文件或拟议方案是否合理**：审查当前落点，判断是否越层或错误复用。
- **报告局部 bug**：先核验现象和根因，再定改动面。

## 声明边界

要不要声明边界看**边界是否已经确定**，不看任务叫什么名字：

- **改动面已被题面唯一钉死**——点名了文件或函数、只落一处、不触跨层契约、不波及共用架构层（定义见「交付格式」）——就地声明边界，不按 [conditional-rules「仓库边界」](references/conditional-rules.md#仓库边界) 的分档整读 repo-map，直接进入定位。典型形态是「把这个函数的报错文案改清楚」。
- **改动面还没钉死**——实际编码、实现规划、行为诊断、架构或代码行为 review，只要落点、调用链或影响面还需查证，都先按 conditional-rules「仓库边界」的分档和三步声明边界。
- **纯规范合规 review**（只检查代码风格、版权或贡献门槛，不落盘代码）默认跳过。

就地声明只在边界确实钉死时成立：一旦结论涉及其它落点、实现行为、架构落点、调用链或上下游契约，就回到 [conditional-rules「仓库边界」](references/conditional-rules.md#仓库边界) 按分档补齐资料，不拿最初那句就地声明当边界。

## 定主手册

主场景描述只提供粗粒度导航。具体触发、实现定位、执行步骤和禁止项以对应 reference 为准。

多个主场景同时可能命中时，先根据用户动作、目标入口、被修改契约和活跃调用方确定主手册。无法唯一确定时，先搜索公共入口、注册点、调用方和直接测试，再按源码所属层选择主手册；只有确认存在真实跨契约影响时，才把第二个主场景作为辅助手册加载。主手册选择仍依赖尚未冻结的关键契约时，按 [conditional-rules「契约缺失的处理形态」](references/conditional-rules.md#契约缺失的处理形态) 处理。

命中后按「必读公共节 + 命中场景节」读取主 playbook，不要求整份全文。**必读公共节**：职责边界/上下游依赖（本手册没有该节时读开头的链路图/组件表，判断改动是否越层）、禁止项（全局约束不随场景豁免）。**按需节**：执行步骤、契约状态表、快照节、自验等只在对应场景或动作命中时读取；诊断类任务可只读目标节（如走查示例的「参数回载诊断」），但该节引用了未读节的概念时先补读被引节。无论读到哪一节，手册都是当前任务的活动规则——不能以「只读了部分」为由豁免其中已适用的约束。实际编码、规划或只读 review 需要核对跨层契约时，可把对应 playbook 作为辅助资料按同样方式读取；不要仅因文件列表相邻而加载。

未命中现有手册时不要强行套入最相近项。先用搜索工具查公共入口、注册点、调用方和直接测试，以 repo-map 判断所属层，并根据相关契约形成任务规划。没有 playbook 兜底时，若源码证据已能确定落点和改动范围即可继续；仅当关键契约仍缺失且不同答案会改变实现时，才按「契约缺失的处理形态」询问用户。只读规划直接交付条件分支，不额外确认。

## 契约核对的触发条件

默认不含契约核对；命中以下任一条才追加「契约状态表逐项填写 + producer/consumer 两端核对」（按 [conditional-rules「比例原则」](references/conditional-rules.md#比例原则) 只填触及项）：

1. 命中门禁对象——新 dtype / 算法 / solver / granularity、复合 unit、skip/partial result 这几类，它们注册成功不等于真能用；
2. 跨契约——一端写、另一端读，比如保存/加载、CLI/registry、dtype/deploy；这类改动要同时核对 producer 和 consumer；
3. 触 public 契约。

## 定位、改、验、交付

- **定位**：rg 搜符号，读目标函数和它的直接调用方。首轮通常 2～4 个锚点就够，够了就停手。
- **改**：按 [coding-rules](references/coding-rules.md) 改，编码任务整份读——其中「贡献门槛对编码的约束」管住未获授权不 push、不建 Issue/PR。
- **验**：按 [validation.md](references/validation.md)「按变更类型选验证」挑最小的验证跑；跑失败按同文件「验证失败的处理」分类处置。
- **交付**：按「交付格式」写；验证状态、影响面、待确认项三项不许沉默。

## 读取预算与停止条件

reference 是任务规则。场景手册通过少量主要入口和关键符号说明实现定位，路径只是当前工作区的提示；失效时沿符号、注册和调用关系重新定位。测试路径和命令见 [validation.md](references/validation.md)；测试文件中的断言不能替代活跃调用链证据。按以下层级加载：

1. 只加载 [conditional-rules「仓库边界」](references/conditional-rules.md#仓库边界) 要求的 repo-map 范围和一个主 playbook；各 Reference 的用途和加载时机从「场景 Reference 导航」查询，不另行推导。
2. 用搜索工具定位题目中的符号、调用方和直接测试；首轮通常 2～4 个最相关的源码/测试锚点即够，只读相关定义、调用链和断言切片。这是典型规模提示，不是硬额度；每追加一个锚点前自问是否已满足下方停止条件。
3. 边界或契约仍不清楚时，先扩大同一调用链的源码切片，再按导航表追加必要的辅助 reference；只有真实跨契约影响已经成立时才追加其它主 playbook。
4. 模型结构、checkpoint/config 或模型相关精度异常按 L1 -> L2 -> L3 读取 [casebook](../../docs/casebook/README.md)；主路径或 map/源码差异仍不清楚时，只补读 repo-map 对应 section。
5. 项目规范和验证规则按导航表的加载时机读取，不把它们提前计入主场景资料。

达到以下条件时停止扩张上下文：仓库边界已确定；活跃调用链或未接通状态已有源码证据；已能区分确认事实、待确认契约和条件式方案；修改面与最小验证范围已经明确。不要为了穷举目录、填满 playbook 文件列表或证明无关分支而继续读取。

典型资料形态是 repo-map 最小切片 + 1 个主 playbook + 首轮那几个源码/测试锚点（编码再加规范）。这些数字只是防预读的规模信号，不是硬额度；硬性收口只有上段停止条件——不因接近数字而停。真实跨契约任务按数据流分批加载，每批重新检查停止条件；扩张时说明触发的冲突或缺失证据。

同一会话处理连续任务时，可以复用已经读取且未变化的资料，但要重新按当前动作做职责判断，不能沿用前一任务的结论。记录本轮新增读取和复用项；不重复读取完整大文件，也不能用旧结论替代当前题设的事实核验。

## 无条件约束

### 事实纪律（Q1/Q2/Q3）

每碰到一个待核实的契约，按 Q1 → Q2 → Q3 依次问，第一个答「是」就停在那一档：

- **Q1 源码、测试或调用方能查出来吗？**能就查完再下判断，不去问用户；先查，再建。
- **Q2 是不是不影响 public 契约的局部选择？**是就自己定，并把选择点明。（默认值一改就改变用户可见行为，那种不算局部选择。）
- **Q3 查不到，而且不同答案会产出不兼容的实现吗？**是就分情况：编码类先问用户（给封闭选项、一次问完、没有依据不伪造推荐）；规划类保留条件分支；其余情况按最合理的解释继续做，并标出这是假设。

### 全程标记纪律

- 契约还没冻结时只列候选锚点，不自己发明命名。
- 待确认项不能在后面的行文里被悄悄改写成已确认。
- 「函数存在」和「调用链接通」是两件事，别混着说。
- 源码和用户描述对不上时，以源码为准。
- 数据链没排查完整，不要写「唯一根因」（判定条件见 [validation.md「证据等级」](references/validation.md#证据等级)）。

### 验证证据分级

证据从弱到强：源码存在 < 单测覆盖 < 集成闭环 < 真实环境验证。不要拿低级证据去支撑高级结论；缺硬件或缺权重就把这一点明写出来。四级各自的判定细则见 [validation.md「证据等级」](references/validation.md#证据等级)。

### 交付格式

只汇总实际执行过的动作，没做的步骤不要补写。按任务类型给内容：

- **改了代码**：改动落在哪一层、改动文件、事实与待确认项、复用了哪些既有抽象、跑了/没跑哪些验证（没跑的写阻塞原因）、有没有波及共用架构层、repo-map 与 docs 要不要更新。
- **没改代码**：边界、事实、方案、已核对的证据与缺失的证据；开头一句话说清**核对到哪条链路、哪一环没验到**（例如「已核对训练保存链与 eval 回载入口，保存内容是否真被 eval 消费这一环没验到，需实跑一轮」）——这是结论的证据边界，用户靠它判断结论有多硬。报链路，不报读了哪些文件名的清单。
- **纯规范检查**：检查项、发现、证据。

**三项不许沉默**：跑了/没跑哪些验证、有没有波及共用架构层、有哪些待确认项——没有内容也要显式写一句；其余项不涉及就不写，不用 `n/a` 占行（契约状态表是固定行表格，仍按 [conditional-rules「比例原则」](references/conditional-rules.md#比例原则) 写 `n/a` + 原因，省略会藏住缺口）。

命中契约核对时写它的**结论**（兼容性影响、写入端和读取端是否一致），不写「本次叠加了契约核对」这个动作。

共用架构层就是 workflow、solver、quant module、public API、ops build 这五层——它们被多个场景共用，改一处会同时影响别的场景。一旦波及就单独点明，因为影响面已经超出本次落点。**交付里直接点层名**（写「波及 solver」，不写「波及共用架构层」）；这个统称只是判据用词，用户读不到它的含义。

### 禁止项

每个任务都要过这份清单。遇到禁止项时只停止当前错误方案：替代路径明确、仍在用户目标和已有授权内时直接继续；只有缺少会改变实现方案的关键输入、契约或授权时才询问用户，并按 [conditional-rules「契约缺失的处理形态」](references/conditional-rules.md#契约缺失的处理形态) 处理。信息缺失不算命中禁止项：先推进不依赖该信息的部分，不要把整个任务挂住。

- 不在 CLI、workflow、solver 或通用 quant module 中加入模型专属分支。
- 不在本 skill 内新增或结构性修改模型 adapter/wrapper/attention/MLP/MoE；路由 `model-adapter` skill，回来后只做契约 review。
- 不混用 LLM CLI path `ALGO_REGISTRY` 与 classic `AlgorithmRegistry`。
- 新 dtype 要闭环数值语义、注册、支持位宽/粒度和实际消费入口；只有影响持久化或部署格式时才检查导出/回载。
- 新算法要闭环注册、`targets`、用户选择入口和实际调用链；只有产生可持久化参数时才检查保存/加载。
- 新 public API 要闭环定义、导出/懒加载、兼容性和用户可见测试；仅在安装或使用方式变化时更新 packaging/docs/examples。
- `PtqUnit.kind`、extract 文件命名、PTQ 参数保存/加载、eval/deploy `--algos` 和 deploy binding 必须保持一致。
- 不在本 skill 内实现 NPU 算子的 kernel、注册与构建；只做 `amct_pytorch` 侧调用、可选 import、fallback 与报错时机，上层调用不得绕过 schema/Meta/import 边界。

落点、越层和上下游边界不明确时，先读 `repo-map.md`（按 [conditional-rules「仓库边界」](references/conditional-rules.md#仓库边界) 的分档范围），再按实际调用关系核对目标层的入口、注册点和直接测试。

## 条件激活的规则

下面这些节只在判据命中时才读，做法和查阅表都在 [conditional-rules](references/conditional-rules.md)：

| 命中条件 | 读哪一节 |
| --- | --- |
| 改动面还没钉死，要定 repo-map 读取范围并做边界声明 | [仓库边界](references/conditional-rules.md#仓库边界) |
| 命中「契约核对的触发条件」，要决定核对做到多重 | [契约核对的做法](references/conditional-rules.md#契约核对的做法)（比例原则、校准示例） |
| 契约查不到、要开口问用户（Q3） | [契约缺失的处理形态](references/conditional-rules.md#契约缺失的处理形态) |
| 要把手册快照当落点依据，或核对命令与断言不符 | [快照与核对命令纪律](references/conditional-rules.md#快照与核对命令纪律) |

## 场景 Reference 导航

| 类型 | Reference | 用途与加载时机 |
| --- | --- | --- |
| 项目规范 | [coding-rules.md](references/coding-rules.md) | 编码任务与规范合规 review 都整份读（含「贡献门槛对编码的约束」）；只做规划、诊断、代码行为 review 时不读 |
| 主场景 | [quant-dtype-module.md](references/quant-dtype-module.md) | 量化 dtype 与通用 quant module 开发/review |
| 主场景 | [llm-algorithm.md](references/llm-algorithm.md) | LLM 量化算法开发/review |
| 主场景 | [model-adapter-review.md](references/model-adapter-review.md) | 已有 adapter 局部非结构修复与契约 review |
| 主场景 | [llm-cli-dataflow.md](references/llm-cli-dataflow.md) | LLM CLI 架构与 eval/extract/ptq/deploy 数据流开发/review |
| 主场景 | [llm-solving-lifecycle.md](references/llm-solving-lifecycle.md) | PTQ 求解侧生命周期：solver、granularity/unit 门禁、参数保存回载 |
| 主场景 | [llm-deploy.md](references/llm-deploy.md) | deploy 实现代码：blockwise/tensorwise 导出、分片、index 与 config |
| 主场景 | [public-api-packaging.md](references/public-api-packaging.md) | public API 导出/懒加载、packaging 清单、experimental 开关与文档承诺的开发/review |
| 交付验证 | [validation.md](references/validation.md) | 改完读并执行最小验证（含验证失败的处理） |
| 走查 | [walkthrough.md](references/walkthrough.md) | 诊断/review/规划首次进入时读（示范什么时候停止扩张、开始交付）；编码任务不默认读 |
| 条件细则 | [conditional-rules.md](references/conditional-rules.md) | 命中「条件激活的规则」表内条件时读 |
