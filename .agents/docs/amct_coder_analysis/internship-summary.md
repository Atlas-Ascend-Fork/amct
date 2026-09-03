# amct-coder Skill 实习任务阶段总结

## 1. 任务定位

本任务的目标不是修改 AMCT 运行时代码，而是为 AMCT 建设一个仓库级编码辅助 Skill：帮助 Agent 判断任务属于哪条代码体系、定位合理的修改层、识别上下游契约，并在编码或规划后给出匹配的验证方式。

`amct-coder` 的实际定位为：

- AMCT 源码编码、实现规划、行为诊断和代码 review 的统一入口；
- 面向仓库边界、活跃调用链和跨模块契约的辅助能力；
- 不是量化实验执行器、模型训练基础设施，也不是 NPU kernel 开发器。

设计过程中参考了仓库级编码 Agent 的通用做法，将规则拆为统一入口、按需加载的子能力 Reference、源码/测试证据和独立评测材料，结合 AMCT 的代码结构进行了定制。

## 2. 已完成成果

### 2.1 完成 Skill 入口架构

形成了统一的任务处理流程：

```text
用户请求
  -> 判断任务职责和外部 Skill 路由
  -> 选择 AMCT 代码体系和主子能力
  -> 读取 repo-map、Reference 和最小源码锚点
  -> 区分事实、待确认契约和条件式方案
  -> 完成规划、编码、诊断或 review
  -> 执行或核对最小验证并汇总证据
```

主 `SKILL.md` 负责入口路由、全局边界、事实门禁、读取预算、BLOCKER 和最终说明；具体领域规则放在 `references/` 中，避免把所有 AMCT 知识一次性写入主 Skill。

### 2.2 完成子能力和 Reference 拆分

围绕 AMCT 主要代码体系，建立了以下子能力文档：

- dtype 和通用 quant module；
- LLM 量化算法；
- LLM 模型单元和 adapter 契约；
- 数据提取与 PTQ workflow；
- PTQ solver、optimizer 和参数保存回载；
- deploy 导出实现；
- classic graph；
- Python 与 NPU ops 的边界；
- public API、测试、验证和项目规范。

每份 Reference 都包含触发范围、源码锚点、上下游关系、执行步骤、BLOCKER 和最小自验方式。主入口只做导航，不重复维护各领域的全部细节。

### 2.3 建立代码体系和职责边界

针对 AMCT 中 LLM new-path、classic、experimental、模型适配和 NPU 算子容易混用的问题，增加了职责边界和负向路由约束：

- 模型结构性修改、模型注册和 wrapper 实现进入模型适配能力；
- Ascend C kernel、PyTorch C++ extension、schema、Meta 和构建调试进入 ops 能力；
- 量化命令执行、结果判读和 deploy-ready 产物生成进入对应操作型 Skill；
- `amct-coder` 负责 AMCT 源码落点判断、实现规划、行为诊断和代码 review；
- LLM new-path 使用 `ALGO_REGISTRY`，classic 使用独立的 `AlgorithmRegistry`，禁止混用。

### 2.4 建立事实门禁和契约检查

将 Agent 容易出错的地方转化为明确检查项：

- 区分“类或函数存在”和“已经有活跃 caller”；
- 区分注册成功、CLI 可达、quantizer 消费、deploy 消费和完整集成闭环；
- 输入、GT、unit、算法状态、solver、保存/回载和 deploy 产物保持一致；
- 仓库无法决定的参数、格式、命名和公开契约不由 Agent 臆造；
- 只读规划可以保留 `pending` 和条件分支，实际编码遇到不兼容契约时停止投机实现；
- 区分源码证据、单测证据、集成证据和真实模型/硬件证据。

### 2.5 完成 dtype 子能力专项迭代

针对 dtype 测试中出现的“未知契约在回答后半段被提前确定”问题，完成了两轮规则修订：

- 建立“当前仓库事实 / 目标契约”双维状态模型；
- 增加 `pending` 状态及状态转换条件；
- 强制检查 registry、CLI、bit policy、weight/activation quantizer、blockwise/tensorwise deploy 和算法兼容性；
- 将产品名称、CLI 值、registry key、Python 标识符和 ops schema 分开记录；
- 目标状态为 `pending` 时，不得把对应的 CLI、deploy、NPU 或 dtype 文件写成当前必改项；
- 增加既有语义复用门，避免仅因名称不同就重复创建等价 dtype。

有/无 Skill 的 dtype 测试已经从原始长回答整理为专项对比文档。

### 2.6 完成算法子能力规则修订

根据 BATQuant、敏感层识别和 GPTQ 测试结果，补充了算法生命周期和跨契约规则：

- 区分 `trainable`、`observe-only` 和 `observe-then-finalize` 三种执行形态；
- 明确 GT 生成、样本遍历、统计、finalize、fake/real quant、保存和 eval/deploy 回载之间的关系；
- 明确 helper 或 hook 没有 caller 时不能视为已接通；
- 按未解决契约选择 dtype、solver、workflow、deploy 或 public API 的 owner Reference；
- 默认只加载一份辅助 Reference，第二份资料需要先说明未解决问题、已查证据和扩张原因；
- 达到任务所需证据后停止扩张，减少无关源码、算法和测试目录的枚举。

### 2.7 完成设计迭代和测试材料整理

设计文档记录了从初版到当前版本的主要变化，包括：

- 单 Skill 与多子能力 Reference 的架构取舍；
- repo-map、casebook、Reference 和源码的读取顺序；
- 多场景任务的主手册裁决；
- 阶段影响与代码职责的区分；
- 活跃调用链、保存回载和 deploy 契约的检查方式；
- 有/无 Skill 对照测试、dtype 专项测试和后续回归计划。

当前主要设计材料位于 `amct_coder_analysis/design/`，测试材料位于 `amct_coder_analysis/test/`，行业方案参考位于 `amct_coder_analysis/other-skill-reference/`。

## 3. 已获得的测试结论

### 3.1 有/无 Skill 对比

基于六组 AMCT 复杂规划题，对 Agent 的任务处理路线进行了对比。主要观察维度包括：

- 是否进入正确的代码体系；
- 是否定位到合理的职责层；
- 是否根据活跃调用链而不是相似文件做判断；
- 是否区分事实、未知契约和条件式方案；
- 是否闭环 CLI、workflow、algorithm、solver、保存回载和 deploy。

已有结果显示，有 Skill 的 Agent 在以下方面有方向性改善：

- 更容易进入 LLM new-path，而不是直接套用 classic 或 experimental 实现；
- 更能识别 registry、adapter、workflow、solver 和 deploy 的职责边界；
- 对新 dtype 和模型结构更愿意保留未知契约；
- 在模型适配、dtype、敏感层和 NPU 边界问题上，越层和错误复用减少。

六题的相对结果为“有 Skill 4 胜、1 平、1 负”。这里的“胜”仅表示两份答案中相对更合理，不表示任务已经达到可实施标准。

### 3.2 仍暴露出的共性问题

两种环境下都出现过：

- 把 helper 存在误认为功能已经接通；
- 把局部算法接口误认为完整的 observe、solve、finalize、save/load 闭环；
- 使用当前仓库不存在或未经核对的 CLI 参数；
- 混淆 FP16/BF16、INT4 weight-only/W4A4 等用户目标；
- 过早确定模型结构、dtype 命名、deploy payload 或 NPU schema；
- 缺少模型、数据集或硬件证据时给出过强结论。

这些结果说明 Skill 主要改善了“去哪里找”和“如何划分职责”，尚未完全解决“当前调用链是否真的工作”和“方案是否具备端到端证据”。

## 4. 完成情况判断

| 工作项 | 当前状态 | 判断 |
| --- | --- | --- |
| Skill 入口和总体架构 | 已完成 | 已形成可执行的主入口、Reference 导航和处理流程 |
| 子能力 Reference 拆分 | 已完成 | 已覆盖主要 AMCT 编码场景，并包含规则、BLOCKER 和自验 |
| 代码体系和职责边界 | 阶段完成 | 已完成主要边界设计，仍需更多独立场景验证 |
| 事实门禁和契约检查 | 阶段完成 | 已落入主入口和多个 Reference，但行为稳定性还需 cold 回归 |
| dtype 子能力优化 | 规则已完成，行为未完全验收 | 已完成两轮规则修订，已有 warm 对照改善，缺少修改后 cold 证据 |
| 算法子能力优化 | 规则已完成，行为未完全验收 | 已补生命周期和读取收敛规则，尚未完成修改后的独立回归 |
| 有/无 Skill 测试整理 | 已完成阶段性整理 | 已完成六题横向对比和 dtype 专项整理 |
| 真实代码修改效果 | 未完成 | 本任务没有修改 AMCT 运行时代码，未验证生成代码能否直接合入 |
| 真实业务效率或质量提升 | 未完成 | 没有完整的固定任务集、统一评分、运行元数据和业务 reviewer 证据 |

综合判断：`amct-coder` 已完成从初版想法到可执行 Skill 原型和配套文档的建设，当前处于“文档和行为规则基本成型、业务验证尚未闭环”的阶段。

## 5. 尚未完成的工作

### 5.1 Skill 行为回归

需要在物理隔离、无历史答案泄露的新会话中验证规则是否真正改变 Agent 行为：

- dtype：补测信息充分的 `q6s`，并重跑 D1、D3、D4；
- algorithm：重跑 GPTQ observe-then-finalize、dtype real quant + deploy、敏感层自动跳层；
- 每个题至少多次运行，记录是否稳定保持 `pending`、是否正确识别 caller、是否在预算内停止。

目前已有的 dtype 对比属于 warm 对照，不能替代 cold 回归。

### 5.2 业务代码可用性验证

需要由熟悉 AMCT 的工程师提供标准答案或可接受结果，再验证 Agent 生成的实现规划或代码：

- 是否修改了正确的代码层；
- 是否接入真实 registry、CLI、workflow 和调用链；
- 是否通过对应的单元测试、集成测试和构建；
- 是否能在真实模型、数据集和 NPU 环境中运行；
- 是否影响量化精度、部署产物和已有功能。

这部分不能只依赖 Skill 作者或 Agent 自己判断，需要领域 reviewer 参与。

### 5.3 统一量化评测

如果需要对外宣称 Skill 带来质量或效率提升，仍需补齐：

- 固定任务集和任务难度分层；
- 同一源码 commit、模型配置、Agent 模型和权限；
- 有 Skill / 无 Skill 的独立运行记录；
- 统一评分维度，例如路由、源码事实、契约控制、闭环完整性、CLI 准确性和验证证据；
- token、运行时间、工具调用和中止原因；
- 由不知分组的业务 reviewer 进行评分。

现有“有 Skill 4 胜、1 平、1 负”适合作为阶段性行为观察，不足以证明真实业务质量提升。

### 5.4 扩展子能力覆盖

目前测试重点集中在 LLM new-path、dtype、算法、敏感层、联合校准和端到端模型任务。以下场景还需要独立正常题、边界题、未知契约题和伪活跃链题：

- classic graph；
- ops Python boundary；
- deploy export code；
- public API 和 packaging；
- solver 参数保存、回载和断点续跑；
- data/workflow 与 model adapter 的组合任务。

### 5.5 长期维护

后续维护时需要关注：

- Reference 规则与 AMCT 源码、CLI 和测试是否发生漂移；
- 新增外部 Skill 后，负向路由是否仍然对称；
- `SKILL.md` 是否重新膨胀为大而全知识库；
- Reference 是否出现规则重复或互相矛盾；
- 新增测试是否将原始 prompt、Agent 输出和 reviewer 结果分开保存。

## 6. 建议的后续处理顺序

1. 由 AMCT 业务工程师确认现有测试题的预期行为和关键事实。
2. 在隔离环境中完成 dtype 和算法 cold 回归，并记录读取轨迹。
3. 根据回归结果决定是否继续修改 Reference，而不是继续扩写主 `SKILL.md`。
4. 选取少量有明确标准答案的真实编码任务，验证 Agent 生成计划或代码的可用性。
5. 再决定是否建立正式的量化指标和扩大评测集。
6. 将 Skill 规则、测试材料和评分结果分开维护，避免测试结论泄露到被测上下文。

## 7. 交接时可使用的一句话总结

> 本阶段完成了面向 AMCT 的仓库级 `amct-coder` Skill 原型，建立了统一入口、子能力 Reference、代码体系路由、事实门禁和初步行为评测；有 Skill 的 Agent 在任务定位和工程边界判断上呈现方向性改善，但 dtype、算法和真实业务代码效果仍需要独立 cold 回归及 AMCT 领域工程师进一步验证。

