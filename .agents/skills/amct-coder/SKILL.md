---
name: amct-coder
description: |
  AMCT 仓源码级编码辅助入口。用于实现规划、代码修改、源码行为诊断和代码 review，覆盖 LLM PTQ CLI 及相关 workflow/solver/data、量化模块（dtype/quant module）与 LLM PTQ 算法、deploy、public API 及其测试/文档的落点、调用链和契约检查。不用于量化实验执行、精度/算法收益判读、模型结构适配、classic/graph_based 或 NPU 算子实现。
---

# AMCT Coder

## 快速路径

拿到任务先判路径：下面四条判据（后文统称判据①~④，编号即条号）**全部满足**才走快速路径。判路径本身不该比一次源码定位更贵——要翻源码才能确定某条满不满足，就说明路已经选错了。

1. **不用读主手册就能把要改的文件全列出来。**主手册＝下方「场景 Reference 导航」表里标「主场景」的那本 reference，一个任务只认一个。诊断、review、规划类任务默认算满足；但要先查调用链才知道改哪儿的，不算满足。
2. **改动只落在一个架构层，而且这一层不是共用架构层。**共用架构层就是 workflow、solver、quant module、public API、ops build 这五层——它们被多个场景共用，改一处会同时影响别的场景。落错层、波及共用架构层，是本仓错误的主要来源。
3. **不碰 public 契约，也不动跨契约的两端。**跨契约就是一端写、另一端读，比如保存/加载、registry 注册与取用；这类改动要同时核对 producer 和 consumer。
4. **不涉及门禁对象。**门禁对象是新 dtype / 算法 / solver / granularity、复合 unit、skip/partial result 这几类——它们注册成功不等于真能用。

**任一条不满足、判据有疑，或者任务属于新功能／跨模块／动契约 → 走完整流程**，按「场景 Reference 导航」表 process-rules 行读对应节（含要不要叠加契约核对）。下面那张读取表只列快速路径的三种形态。

四条都满足，就按四步做完：

- **定位**：rg 搜符号，读目标函数和它的直接调用方。一般 3 个锚点以内就够，够了就停手。
- **改**：按 [coding-rules](references/coding-rules.md) 里对应改动类型的那一节改，不整份读。
- **验**：按 [validation.md](references/validation.md)「按变更类型选验证」挑最小的验证跑；跑失败按同文件「验证失败的处理」分类处置。
- **交付**：按「交付格式」写，不适用的项写 `n/a`。

**走快速路径时读哪些文件**：

| 任务形态 | 读什么 |
| --- | --- |
| 普通修复（行为实现 bug） | 四步就地做完，**不读** process-rules、repo-map、walkthrough，也不读主手册 |
| 诊断 / review / 规划 | 四步照走，另外读一个主手册的公共节 + 命中节；本会话第一次进本 skill 还要读 walkthrough |
| 纯规范合规检查（不落盘代码、无行为面） | 不读主手册，只读 coding-rules 的相关节 |

**免掉的只是读文件，规则一条不免**：判据①~④、命中的主手册里的禁止项、全局禁止项照常适用。执行中发现越界，就按「单向升级」转完整流程补齐，已经读过的资料接着用，最后说明里写清走了哪条路。快速路径没覆盖的情形，按「场景 Reference 导航」查单个 reference，不整读导航表。

## 执行顺序（职责归属判定后、留在本 skill 的任务）

```text
0. 判路径（判据①~④ 全中 -> 快速路径；否则完整流程，并按 process-rules「契约核对的触发条件」定要不要叠加契约核对）
1. 判职责归属 -> 2. 声明边界（快速路径跳过）+ 选主手册 -> 3. 锚点定位并执行 -> 4. 验证并交付
```

同一流程的两个视图：上面「快速路径」里的四步就是这里的第 3~4 步（定位与改＝第 3 步，验与交付＝第 4 步）；第 2 步里只有声明边界被快速路径跳过，选不选主手册按上面那张读取表定。

## 无条件约束（快速路径与完整流程都适用）

### 职责归属

从高到低按这个顺序判：

1. 用户明确指定了 skill，就按用户的目标执行。
2. 没指定，就拿当前会话已发现 skill 的 `name + description` 对一遍：任务实质已经被更专门的 skill 覆盖时，直接加载它并遵循它，不要先读本 skill 的 repo-map 或手册，也不要只丢一句「应使用某 skill」就结束。
3. 怀疑列表不全（插件没加载、skill 没注册）时，再去仓库 `.agents/skills/` 各 skill 目录的 frontmatter description 补查一遍。
4. 名字撞车、或查完还是确认不了，就按本 skill 流程做，并在交付说明里点明可能存在职责重叠。

分清「实现代码」还是「运行命令」要看动作，不看关键词。一个请求里混了两类活就按职责拆开，源码级的部分和共享契约由本 skill 收口。

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

### 单向升级

快速路径执行中发现越界，就转完整流程把缺的补齐：边界声明回 process-rules「仓库边界」补，已经写的代码按完整流程的验证要求重验，读过的资料继续用；最后说明里写清走了哪条路。升级是单向的，离开快速路径后要回读 [process-rules](references/process-rules.md#完整流程与契约核对)「完整流程与契约核对」。

### 验证证据分级

证据从弱到强：源码存在 < 单测覆盖 < 集成闭环 < 真实环境验证。不要拿低级证据去支撑高级结论；缺硬件或缺权重就把这一点明写出来。改完按 [validation.md](references/validation.md) 挑最小验证跑。只读任务只核对既有证据、不改手册的快照断言，发现的差异写进交付说明。

### 交付格式

本节是交付清单的唯一出处；任何 reference 只可引用、不得复制。

开头一句话标清走的路径（快速路径／完整流程，及有没有含契约核对）。只汇总实际执行过的动作，没做的步骤不要补写。按任务类型给内容：

- **改了代码**：主场景、改动文件、事实与待确认项、复用了哪些既有抽象、跑了/没跑哪些验证（没跑的写阻塞原因）、有没有波及共用架构层、repo-map 与 docs 要不要更新。
- **没改代码**：边界、事实、方案、已核对的证据与缺失的证据。
- **纯规范检查**：检查项、发现、证据。

共用架构层（判据② 已枚举那五层，此处不重述）一旦波及就单独点明，因为影响面已经超出本次落点。不适用的项写 `n/a`，不整项省略。

### 禁止项

撞到禁止项只停掉这个错误方案，不停任务：替代路径明确、又还在用户目标内，就直接接着做；信息缺失也先把不依赖它的部分推进完。快速路径不读 process-rules，所以这里内联它「全局禁止项」里**快速路径真有可能违反的三条**：

1. 不在 CLI、workflow、solver 或通用 quant module 中加模型专属分支。
2. 不混用 LLM CLI 的 `ALGO_REGISTRY` 与 classic `AlgorithmRegistry`。
3. `PtqUnit.kind`、extract 文件命名、PTQ 参数保存/加载、`--algos` 与 deploy binding 保持一致。

其余各条已被判据①~④ 排除在外（命中就进不了快速路径）；完整清单以 [process-rules「全局禁止项」](references/process-rules.md#全局禁止项) 为准。

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
| 流程细则 | [process-rules.md](references/process-rules.md) | 完整流程按需读对应节（快速路径不读）；节索引在该文件开头 |
