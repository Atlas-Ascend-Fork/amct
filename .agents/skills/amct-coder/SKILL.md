---
name: amct-coder
description: |
  AMCT 仓源码级编码辅助入口。用于实现规划、代码修改、源码行为诊断和代码 review，覆盖 LLM PTQ CLI 及相关 workflow/solver/data、量化模块（dtype/quant module）与 LLM PTQ 算法、deploy、public API 及其测试/文档的落点、调用链和契约检查。不用于量化实验执行、精度/算法收益判读、模型结构适配、classic/graph_based 或 NPU 算子实现。
---

# AMCT Coder

## 执行顺序（留在本 skill 的任务）

```text
1. 声明边界 + 定主手册 -> 2. 定位并改 -> 3. 验证并交付
```

每步细则、读哪些节、什么时候停止扩张，都在 [process-rules](references/process-rules.md)（节索引在该文件开头）；要不要叠加契约核对按其「契约核对的触发条件」定。

第 2~3 步的做法：

- **定位**：rg 搜符号，读目标函数和它的直接调用方。一般 3 个锚点以内就够，够了就停手。
- **改**：按 [coding-rules](references/coding-rules.md) 里对应改动类型的那一节改，不整份读。
- **验**：按 [validation.md](references/validation.md)「按变更类型选验证」挑最小的验证跑；跑失败按同文件「验证失败的处理」分类处置。
- **交付**：按「交付格式」写，不适用的项写 `n/a`。

## 无条件约束（任何任务都适用）

### 不归本 skill 的部分

选哪个 skill 是 agent 读到本文件之前就按 frontmatter description 判完的事，本文件不重复路由规则。进来之后只剩两条：

- **读源码后发现活的实质归别处**（跑命令拿结果、精度/收益判读、新增模型结构、NPU kernel 实现），就实际加载对应 skill 并遵循它，不要只丢一句「应使用某 skill」就结束。交界表与转出目标见 [delegation](references/delegation.md)；会话里找不到对应 skill 时，去仓库 `.agents/skills/` 各目录的 frontmatter description 补查。
- **一个请求里混了两类活**，按动作拆开，不看 `eval`、`deploy`、`attention`、`校准` 这类关键词：改实现代码留在这里，跑命令拿结果转出去；源码级部分和共享契约由本 skill 收口。

名字撞车、或查完仍确认不了，就按本 skill 流程做，并在交付说明里点明可能存在职责重叠。

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
- 数据链没排查完整，不要写「唯一根因」。

### 验证证据分级

证据从弱到强：源码存在 < 单测覆盖 < 集成闭环 < 真实环境验证。不要拿低级证据去支撑高级结论；缺硬件或缺权重就把这一点明写出来。改完按 [validation.md](references/validation.md) 挑最小验证跑。只读任务只核对既有证据、不改手册的快照断言，发现的差异写进交付说明。

### 交付格式

本节是交付清单的唯一出处；任何 reference 只可引用、不得复制。

开头一句话标清实际读了哪些资料、有没有叠加契约核对。只汇总实际执行过的动作，没做的步骤不要补写。按任务类型给内容：

- **改了代码**：主场景、改动文件、事实与待确认项、复用了哪些既有抽象、跑了/没跑哪些验证（没跑的写阻塞原因）、有没有波及共用架构层、repo-map 与 docs 要不要更新。
- **没改代码**：边界、事实、方案、已核对的证据与缺失的证据。
- **纯规范检查**：检查项、发现、证据。

共用架构层就是 workflow、solver、quant module、public API、ops build 这五层——它们被多个场景共用，改一处会同时影响别的场景。一旦波及就单独点明，因为影响面已经超出本次落点。不适用的项写 `n/a`，不整项省略。

### 禁止项

撞到禁止项只停掉这个错误方案，不停任务：替代路径明确、又还在用户目标内，就直接接着做；信息缺失也先把不依赖它的部分推进完。清单的唯一出处是 [process-rules「全局禁止项」](references/process-rules.md#全局禁止项)，本文件不抄副本。

## 场景 Reference 导航

| 类型 | Reference | 用途与加载时机 |
| --- | --- | --- |
| 项目规范 | [coding-rules.md](references/coding-rules.md) | 编码任务固定要读，但按改动类型只读对应节：改代码读「代码风格与版权」「日志与错误处理」「健壮性边界」「测试编写」里相关的项，改依赖或落点再加读「依赖与落点」；规范合规 review 整份读 |
| 职责辅助 | [delegation.md](references/delegation.md) | 职责交界不清时读 |
| 主场景 | [quant-dtype-module.md](references/quant-dtype-module.md) | 量化 dtype 与通用 quant module 开发/review |
| 主场景 | [llm-algorithm.md](references/llm-algorithm.md) | LLM 量化算法开发/review |
| 主场景 | [model-adapter-review.md](references/model-adapter-review.md) | 已有 adapter 局部非结构修复与契约 review |
| 主场景 | [llm-cli-dataflow.md](references/llm-cli-dataflow.md) | LLM CLI 架构与 eval/extract/ptq/deploy 数据流开发/review |
| 主场景 | [llm-solving-lifecycle.md](references/llm-solving-lifecycle.md) | PTQ 求解侧生命周期：solver、granularity/unit 门禁、参数保存回载 |
| 主场景 | [llm-deploy.md](references/llm-deploy.md) | deploy 实现代码：blockwise/tensorwise 导出、分片、index 与 config |
| 主场景 | [public-api-packaging.md](references/public-api-packaging.md) | public API 导出/懒加载、packaging 清单、experimental 开关与文档承诺的开发/review |
| 交付验证 | [validation.md](references/validation.md) | 改完读并执行最小验证（含验证失败的处理）；改 skill 自身的自验也在这里；要引入自造词前先读「自造词准入」（含已废弃旧词清单） |
| 走查 | [walkthrough.md](references/walkthrough.md) | 诊断/review/规划首次进入时读（示范什么时候停止扩张、开始交付）；编码任务不默认读 |
| 流程细则 | [process-rules.md](references/process-rules.md) | 按需读对应节；节索引在该文件开头 |
