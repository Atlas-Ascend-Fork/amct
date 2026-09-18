---
name: amct-coder
description: |
  AMCT 仓源码级编码辅助入口。用于实现规划、代码修改、源码行为诊断和代码 review，覆盖 LLM PTQ CLI 及相关 workflow/solver/data、量化模块（dtype/quant module）与 LLM PTQ 算法、deploy、public API 及其测试/文档的落点、调用链和契约检查。不用于量化实验执行、精度/算法收益判读、模型结构适配、classic/graph_based 或 NPU 算子实现。
---

# AMCT Coder

AMCT 源码级编码、实现规划、行为诊断与 review 的路由入口。默认中文；修改代码前简述改动落在哪一架构层和允许改动范围。

## 轻级四步直达

**①~④ 全中才走快速路径**（此处是四条判据的唯一出处，[process-rules「任务分级」](references/process-rules.md#任务分级) 的分级表引用它）：① 动手前能列出全部待改文件（诊断/review 类视为满足；**需先查调用链才能定位的不入轻级**）② 只落在单一架构层（错层是本仓最主要的错误来源） ③ 不触 public 契约 ④ 不命中门禁对象（新 dtype / 算法 / solver / granularity、复合 unit、skip/partial result）。
**全中 + 普通修复（行为实现 bug）** → 就地四步直达，**不必读 process-rules（含「任务分级」的分级表）、repo-map、walkthrough、主手册（＝下方导航表标「主场景」的 reference，一个任务只认一个）**：定位（rg 符号 → 读目标函数与直接调用方，通常 ≤3 个锚点即停）→ 改（按 [coding-rules](references/coding-rules.md) 改动类型对应节，不整读）→ 验（按 [validation.md](references/validation.md)「按变更类型选验证」选最小验证，失败按其「验证失败的处理」分类处置）→ 交付（按下方「交付格式」，不适用项写 `n/a`）。**全中 + 诊断/review/规划** → 仍判轻级，读一个主手册的公共节+命中节。**纯规范合规检查**（不落盘代码、无行为面）→ 免读主手册，只读 coding-rules 相关节。**任一条不中，或属新功能/跨模块/动契约** → 标准级或重级，按导航表 process-rules 行读「任务分级」与各节走完整流程。
判据有疑（要先查才能定）→ 直接判标准级；分级判断超过一次源码定位的成本时就已选错级别。豁免只免读取、不免约束：①~④、命中手册的禁止项与全局禁止项照常适用；执行中越界即按「单向升级」升标准级补齐，资料复用，分级写进交付说明。四步没覆盖的情形按「场景 Reference 导航」查单个 reference，不整读导航表。本 skill 不设术语表，自造词都在使用点就地解释。

## 核心流程（职责归属判定后、留在本 skill 的任务执行）

```text
0. 分级 -> 1. 判职责归属 -> 2. 边界（轻级跳过）+ 选主手册 -> 3. 锚点定位并执行 -> 4. 验证并按分级格式交付
```

### 各级都要做

**职责归属**（优先级从高到低）：① 用户明确指定的 skill 按用户目标执行；② 否则按当前会话已发现 skill 的 `name + description` 判断，任务实质已被更专门 skill 覆盖时直接加载并遵循它，不预读本 skill 的 repo-map/手册，也不要只说「应使用某 skill」就结束；③ 列表疑似不完整（插件未加载、skill 未注册）时，以仓库 `.agents/skills/` 各 skill 目录的 frontmatter description 补充核对；④ 同名冲突或仍无法确认时按本 skill 流程处理并在交付说明点明可能的职责重叠。实现代码 vs 运行命令按动作判断，不按关键词；混合任务按职责拆分，本 skill 收口源码级部分和共享契约。

**事实纪律**（对每个待核实契约按序，第一个「是」停）：

- Q1 能由源码/测试/调用方查明？→ 查完再判定，不问用户；先查再建。
- Q2 是不影响公开契约的局部选择？→ 自行决定并点明（默认值改变用户可见行为不属此类）。
- Q3 查不到时不同答案产出不兼容实现？→ 编码类先问用户（封闭选项、一次问完、无依据不伪造推荐）；规划类保留条件分支；否则按最合理解释继续并标注假设。

全程标记纪律：未冻结契约只列候选锚点、不发明命名；前后文不得静默改写待确认项；区分「函数存在」与「调用链接通」；源码与用户描述冲突时以源码为准；未排查完整数据链不用「唯一根因」。

**单向升级**：轻级执行中越界即升标准级补齐（边界声明回 process-rules「仓库边界」，已写代码按新级别的验证要求重验），资料复用；分级写进最终说明。升级即离开四步直达路径，且需回读 [process-rules](references/process-rules.md#任务分级)「任务分级」。

**验证证据分级**：源码存在 < 单测覆盖 < 集成闭环 < 真实环境验证；不得用低级证据替代高级结论，缺硬件/权重时明确写出。改完按 [validation.md](references/validation.md) 选最小验证；只读任务核对既有证据、不改手册的快照断言（差异写进交付说明）。

**交付格式**（唯一出处；任何 reference 只可引用、不得复制此清单）：开头一句话标分级；只汇总实际执行过的动作，不补写未执行步骤——改了代码（主场景、改动文件、事实与待确认项、复用抽象、已跑/未跑验证及阻塞原因、是否波及共用架构层、repo-map/docs 是否需更新）/ 没改代码（边界、事实、方案、已核对与缺失证据）/ 纯规范检查（检查项、发现、证据）。共用架构层指 workflow、solver、quant module、public API、ops build 这些被多场景共用的软件架构层，波及即单独点明（影响面超出本次落点）。不适用的项写 `n/a`，不整项省略。

**禁止项只停错误方案不停任务**：替代路径明确且在用户目标内就直接继续；信息缺失先推进不依赖它的部分。轻级不读 process-rules，故此处内联其「全局禁止项」里**轻级真能违反的三条**：① 不在 CLI、workflow、solver 或通用 quant module 中加模型专属分支；② 不混用 LLM CLI 的 `ALGO_REGISTRY` 与 classic `AlgorithmRegistry`；③ `PtqUnit.kind`、extract 文件命名、PTQ 参数保存/加载、`--algos` 与 deploy binding 保持一致。其余各条已被判据①~④ 排除（命中即非轻级）；完整清单以 [process-rules「全局禁止项」](references/process-rules.md#全局禁止项) 为准。

## 场景 Reference 导航

| 类型 | Reference | 用途与加载时机 |
| --- | --- | --- |
| 项目规范 | [coding-rules.md](references/coding-rules.md) | 编码任务的固定必读文件，但按改动类型只读对应节：Python 改动读「代码风格与版权」「日志与错误处理」「健壮性边界」「测试编写」相关项，C/C++ 改动加读「代码风格与版权」内的 C/C++ 条（本仓不单设 C++ 节），改依赖再读「依赖与落点」；规范合规 review 必读 |
| 职责辅助 | [delegation.md](references/delegation.md) | 职责交界不清时读取 |
| 主场景 | [quant-dtype-module.md](references/quant-dtype-module.md) | 量化 dtype 与通用 quant module 开发/review |
| 主场景 | [llm-algorithm.md](references/llm-algorithm.md) | LLM 量化算法开发/review |
| 主场景 | [model-adapter-review.md](references/model-adapter-review.md) | 已有 adapter 局部非结构修复与契约 review |
| 主场景 | [llm-cli-dataflow.md](references/llm-cli-dataflow.md) | LLM CLI 架构与 eval/extract/ptq/deploy 数据流开发/review |
| 主场景 | [llm-solving-lifecycle.md](references/llm-solving-lifecycle.md) | PTQ 求解侧生命周期：solver、granularity/unit 门禁、参数保存回载 |
| 主场景 | [llm-deploy.md](references/llm-deploy.md) | deploy 实现代码：blockwise/tensorwise 导出、分片、index 与 config |
| 主场景 | [public-api-packaging.md](references/public-api-packaging.md) | public API 导出/懒加载、packaging 清单、experimental 开关与文档承诺的开发/review |
| 交付验证 | [validation.md](references/validation.md) | 修改后读并执行最小验证（含验证失败的处理）；skill 自身修改的自验也在此；引入自造词前读「自造词准入」（含已废弃旧词清单） |
| 走查 | [walkthrough.md](references/walkthrough.md) | 诊断/review/规划首次进入时读（示范何时停止扩张并开始交付）；编码任务不默认读 |
| 流程细则 | [process-rules.md](references/process-rules.md) | 标准级及以上按需读对应节（轻级不读）：判级/升级回读、标准与重级追加判据、比例原则与分级示例 →「任务分级」；边界声明与 repo-map 分档 →「仓库边界」；Q3 需向用户询问契约（封闭选项、推荐项依据、拿不到答复时）→「契约缺失的处理形态」；选/换主手册、按节读取、快照核对命令 →「Reference 读取」；读取预算与停止条件、连续任务复用资料 →「渐进披露与读取预算」；全局禁止项完整清单 →「全局禁止项」；实现与落点 review 的区分、混合任务拆分 →「职责归属细则」 |
