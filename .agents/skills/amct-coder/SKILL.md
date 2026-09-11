---
name: amct-coder
description: |
  AMCT 仓源码级编码辅助入口。用于实现规划、代码修改、源码行为诊断和代码 review，覆盖 LLM PTQ CLI 及相关 workflow/solver/data、量化模块（dtype/quant module）与 LLM PTQ 算法、deploy、public API 及其测试/文档的落点、调用链和契约检查。不用于量化实验执行、精度/算法收益判读、模型结构适配、classic/graph_based 或 NPU 算子实现。
---

# AMCT Coder

AMCT 源码级编码、实现规划、行为诊断与 review 的路由入口。默认中文；修改代码前简述主落点层和允许改动范围。

满足「已知道改哪个文件、改的是行为实现 bug、不新增 dtype/算法/solver、不动 CLI 参数和公开接口」的普通修复，直接按 [quick-path.md](references/quick-path.md) 的五步执行，无需先读分级表和 reference 细则。不满足条件（新功能、跨模块、动契约、诊断类、review 类）走下方完整流程。

## 核心流程（职责归属判定后、留在本 skill 的任务执行）

```text
0. 分级 -> 1. 判职责归属 -> 2. 边界（轻级跳过）+ 选主手册 -> 3. 锚点定位并执行 -> 4. 验证并按分级格式交付
```

### 规则执行层级

本 skill 的控制机制按执行时机分三层，轻级任务只承担第一层：

- **每个任务执行**：职责归属、分级与升级、落点定位（主手册公共节+命中节；引用快照事实前先跑核对命令）、事实纪律 Q1/Q2（查得到就查、局部选择自行决定）、最小验证、交付格式。
- **仅标准级及以上执行**：边界声明与 repo-map 读取（按领域分档）、契约状态模型/门禁表填写（重级）、跨 producer/consumer 双端核对（重级）。
- **仅异常路径执行**：Q3 与契约询问（查不到且答案影响实现才触发）、BLOCKER 处置（撞到才用）、快照冲突修正（核对命令与断言不符才改）。

正常路径下三层机制大多数不激活：简单 bug 修复的全部流程就是「定级 → 定位 → 修 → 验证 → 交付」，第一层之外的规则只在判据明确命中时才读。

### 分级

判据不能一眼判定时（需要先查调用链、要核对 public 契约范围、门禁对象归属不明），不要花时间论证——直接判标准级。分级判断本身超过一次源码定位的成本时就已选错级别。

| 级别 | 判据 | 流程 |
| --- | --- | --- |
| 轻 | ① 动手前能列出全部待改文件（诊断/review 类视为满足；**需先查调用链才能定位的不入轻级**）② 单一落点层 ③ 不触 public 契约 ④ 不命中门禁对象（新 dtype/算法/solver/granularity、复合 unit、skip/partial result） | 跳过边界声明；读一个主手册——公共节（职责边界/上下游依赖，没有该节读开头链路图；BLOCKER）+ 命中场景节——+ 编码按改动类型读 [coding-rules](references/coding-rules.md) 对应节（Python 改动读风格/日志/健壮性/测试节，C++ 读 C++ 节；不整读）→ 改 → 最小验证 |
| 标准 | 默认 | 完整流程：职责归属 → 边界声明 → 主手册与锚点定位 → 修改/诊断 → 验证与交付（细则见 [process-rules.md](references/process-rules.md)） |
| 重 | ① 命中轻级判据④的门禁对象清单 ② 跨契约（保存/加载、CLI/registry、dtype/deploy 等两端）③ 触 public 契约 | 完整流程 + 门禁表/状态模型逐项填写 + 双端验证 |

单向升级：轻级执行中越界即升标准级补齐（边界声明回 process-rules「仓库边界」，已写代码按新级别的验证要求重验），资料复用。分级写进最终说明。

分级示例（校准用；新任务仿照判据路径，不套用结论）：

| 请求 | 级别 | 判据路径 |
| --- | --- | --- |
| 「修一下这个函数的报错日志措辞」 | 轻 | 文件可列出、单一落点、不改行为 |
| 「把某 CLI 参数默认值从 A 改成 B」 | 重 | 默认值改变用户可见行为 → 触 public 契约（也命中事实纪律 Q2 的例外） |
| 「新增一个 dtype 注册到 `--quant_dtype`」 | 重 | 命中门禁对象（新 dtype）+ CLI choices 是 public 契约 |
| 「已有 adapter 的 forward 有个 bug，输出和 float 不等价」 | 标准 | 文件可列出但行为正确性需沿捕获→forward→wrapper 链核验，前置资料不足轻级 |
| 「只检查这次改动是否符合代码风格/版权」 | 轻 | 不落盘代码、无行为面；读 coding-rules 的相关规范节即可，无需主手册 |

**职责归属**（优先级从高到低）：① 用户明确指定的 skill 按用户目标执行；② 否则按当前会话已发现 skill 的 `name + description` 判断，任务实质已被更专门 skill 覆盖时直接加载并遵循它，不预读本 skill 的 repo-map/手册，也不要只说「应使用某 skill」就结束；③ 列表疑似不完整（插件未加载、skill 未注册）时，以仓库 `.agents/skills/` 各 skill 目录的 frontmatter description 补充核对；④ 同名冲突或仍无法确认时按本 skill 流程处理并在交付说明点明可能的职责重叠。实现代码 vs 运行命令按动作判断，不按关键词；混合任务按职责拆分，本 skill 收口源码级部分和共享契约。

**事实纪律**（对每个待核实契约按序，第一个「是」停）：

- Q1 能由源码/测试/调用方查明？→ 查完再判定，不问用户；先查再建。
- Q2 是不影响公开契约的局部选择？→ 自行决定并点明（默认值改变用户可见行为不属此类）。
- Q3 查不到时不同答案产出不兼容实现？→ 编码类按「契约缺失形态」先问用户（封闭选项、一次问完、无依据不伪造推荐）；规划类保留条件分支；否则按最合理解释继续并标注假设。

全程标记纪律：未冻结契约只列候选锚点、不发明命名；前后文不得静默改写待确认项；区分「函数存在」与「调用链接通」；源码与用户描述冲突时以源码为准；未排查完整数据链不用「唯一根因」。

**验证证据分级**：源码存在 < 单测覆盖 < 集成闭环 < 真实环境验证；不得用低级证据替代高级结论，缺硬件/权重时明确写出。改完按 [validation.md](references/validation.md) 选最小验证；只读任务核对既有证据、不改快照（差异写进交付说明）。

**交付格式**：开头一句话标分级；只汇总实际执行过的动作，不补写未执行步骤——改了代码（主场景、改动文件、事实与待确认项、复用抽象、已跑/未跑验证、repo-map/docs 是否需更新）/ 没改代码（边界、事实、方案、已核对与缺失证据）/ 纯规范检查（检查项、发现、证据）。

**BLOCKER 只停错误方案不停任务**：替代路径明确且在用户目标内就直接继续；信息缺失先推进不依赖它的部分。

## 场景 Reference 导航

| 类型 | Reference | 用途与加载时机 |
| --- | --- | --- |
| 项目规范 | [coding-rules.md](references/coding-rules.md) | 编码任务的固定必读文件，但按改动类型只读对应节：Python 改动读「代码风格与版权」「日志与错误处理」「健壮性边界」「测试编写」相关项，C/C++ 读对应项，改依赖再读「依赖与落点」；规范合规 review 必读 |
| 职责辅助 | [delegation.md](references/delegation.md) | 职责交界不清时读取 |
| 主场景 | [quant-dtype-module.md](references/quant-dtype-module.md) | 量化 dtype 与通用 quant module 开发/review |
| 主场景 | [llm-algorithm.md](references/llm-algorithm.md) | LLM 量化算法开发/review |
| 主场景 | [model-adapter-review.md](references/model-adapter-review.md) | 已有 adapter 局部非结构修复与契约 review |
| 主场景 | [llm-cli-dataflow.md](references/llm-cli-dataflow.md) | LLM CLI 架构与 eval/extract/ptq/deploy 数据流开发/review |
| 主场景 | [llm-solving-lifecycle.md](references/llm-solving-lifecycle.md) | PTQ 求解侧生命周期：solver、granularity/unit 门禁、参数保存回载 |
| 主场景 | [llm-deploy.md](references/llm-deploy.md) | deploy 实现代码：blockwise/tensorwise 导出、分片、index 与 config |
| 主场景 | [public-api-packaging.md](references/public-api-packaging.md) | public API 导出/懒加载、packaging 清单、experimental 开关与文档承诺的开发/review |
| 交付验证 | [validation.md](references/validation.md) | 修改后读并执行最小验证（含验证失败的处理）；skill 自身修改的自验也在此 |
| 快速路径 | [quick-path.md](references/quick-path.md) | 普通局部 bug 修复五步直达（入口条件见文件头）；命中时无需读分级表与 reference 细则 |
| 走查 | [walkthrough.md](references/walkthrough.md) | 诊断/review/规划首次进入时读（收束形态示范）；编码任务不默认读 |
| 流程细则 | [process-rules.md](references/process-rules.md) | 标准级及以上的完整流程纪律：边界声明、契约询问形态、读取层级与停止条件、全局 BLOCKER 清单 |

## 完整规则（轻级无需读，标准级及以上按需读对应节）

轻级任务读完核心流程和主手册即可开工。以下细则在对应情形触发时读取 [process-rules.md](references/process-rules.md)：

| 触发情形 | 读哪节 |
| --- | --- |
| 标准级及以上任务的边界声明（repo-map 读取、写回判据） | 「仓库边界」 |
| Q3 触发、需要向用户询问契约（封闭选项、推荐项依据、拿不到答复时） | 「契约缺失的处理形态」 |
| 选择/切换主手册、按节读取规则、快照核对命令的使用 | 「Reference 读取」 |
| 读取预算与停止条件、连续任务复用资料 | 「渐进披露与读取预算」 |
| 全局 BLOCKER 清单（分层、算法系统、跨链一致性） | 「全局 BLOCKER」 |
| 实现与落点 review 的区分、混合任务拆分细则 | 「职责归属细则」 |
