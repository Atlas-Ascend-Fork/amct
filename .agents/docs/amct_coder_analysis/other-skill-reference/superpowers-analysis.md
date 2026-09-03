# superpowers 分析与 amct-coder 的迁移学习

> 用途：说明 amct-coder 在设计上参考了业内哪些成熟实践、哪些是我按 AMCT 领域改造后落地的、哪些是我判断不该学的。
> 日期：2026-09-01。
> 材料来源：本机克隆并逐文件阅读源码的 `obra/superpowers`（14 个 skill + hooks）与 `anthropics/skills`（官方 `skill-creator`）；amct-coder 侧为 `.agents/skills/amct-coder/SKILL.md`（162 行）+ `references/`（17 份）+ `.agents/docs/amct_coder_analysis/skill-vs-noskill-comparison.md`（六题有/无 skill 对照）。
> 引用的第三方评测数据我没有独立复现，正文均已标注。

---

## 1. 为什么把它作为主要参考对象

`obra/superpowers`（作者 Jesse Vincent）是目前装机量最高的**流程约束型** skill 库，单个 skill 20 万级安装。选它做主要参考不是因为领域接近——它完全不涉及量化或深度学习——而是因为**形态同类**：两者都不教具体 API，而是约束 agent 的工作方式，先规划、先找根因、先验证，再动手。

同时它是被公开批评最多的 skill 库。"限制模型能力""费 token""很多人卸载了"这些说法都指向它。这个双重身份让它成为很合适的参考对象：**正面提供已经被大规模验证过的设计模式，反面提供已经暴露出代价的强制手段**，两边都有实测材料可依。

本机克隆路径 `/tmp/arch-superpowers`（临时目录，重启失效；复现用 `git clone --depth 1 https://github.com/obra/superpowers.git`）。

---

## 2. 它是什么：结构与强制机制

不是单个 skill，而是一个 Claude Code plugin：

```
superpowers/
├── skills/           # 14 个 skill，扁平命名空间
├── hooks/
│   ├── hooks.json    # SessionStart 钩子
│   ├── session-start/
│   └── run-hook.cmd
├── AGENTS.md / CLAUDE.md / GEMINI.md   # 多 runtime 适配
└── gemini-extension.json
```

14 个 skill 分三类职责：**元/入口**（`using-superpowers`、`writing-skills`）、**流程**（`brainstorming`、`writing-plans`、`executing-plans`、`subagent-driven-development`、`dispatching-parallel-agents`）、**纪律**（`systematic-debugging`、`test-driven-development`、`verification-before-completion`、`receiving-code-review`、`requesting-code-review`）、**工程**（`using-git-worktrees`、`finishing-a-development-branch`）。

一个贯穿全库的规定值得先说：**所有 skill 的 description 都只写"何时用 + 在什么动作之前"，没有一个在描述"这个 skill 做什么"**。这是刻意的，理由见 §4.3。

它的强制机制集中在入口 skill `using-superpowers`，正文开头是：

```
<EXTREMELY-IMPORTANT>
If you think there is even a 1% chance a skill might apply to what you are doing,
you ABSOLUTELY MUST invoke the skill.

IF A SKILL APPLIES TO YOUR TASK, YOU DO NOT HAVE A CHOICE. YOU MUST USE IT.

This is not negotiable. You cannot rationalize your way out of this.
</EXTREMELY-IMPORTANT>
```

规则是**在任何回应或动作之前**先调用相关 skill，包括澄清性提问、探索代码库、查看文件。后面跟一张 12 行的 Red Flags 表（`using-superpowers:41-56`），逐条否决 agent 可能想到的每个例外——"这只是个简单问题"→"问题就是任务，检查 skill"；"我需要先了解上下文"→"skill 检查在澄清提问之前"。再配 `hooks/hooks.json` 的 `SessionStart` 钩子，在 startup / clear / compact 三个时机自动注入，也就是**每个新会话、每次清空、每次上下文压缩之后这套规则都重新加载一遍**。

这套机制是 §5 要讲的"需要避免"的部分，先记在这里。

---

## 3. 迁移学习的判断方法

我不是逐条照抄，先按三个维度判断一条设计能不能迁移：

| 判断维度 | 说明 |
| --- | --- |
| 形态是否同类 | 两者都是"约束工作方式"而非"提供 API 知识"，所以流程与纪律层面的设计可迁移 |
| 领域是否同类 | 不同类。superpowers 面向通用编码，amct-coder 面向 AMCT 单仓的架构约束——**它的具体规则内容一律不可搬，只能搬结构** |
| 约束对象是否同类 | 不同类。它约束**做事顺序**（先根因、先验证），amct-coder 需要约束**架构不变量**（不越层、不混用两套 registry）。这决定了同一个"硬禁止"模式落到两边内容完全不同 |

所以迁移动作分三种：**结构直接沿用**（如 Iron Law 的写法、description 的规定）、**模式沿用但内容重写**（如证据门禁，从两级扩到四级并绑定昇腾环境）、**明确不采纳**（如 hook 强制，见 §5）。

---

## 4. 已迁移的设计：原样、改造、以及改造的理由

### 4.1 Iron Law → BLOCKER 清单（结构沿用，约束对象改变）

superpowers 把最重要的规则写成不带例外条款的一行大写文字。`systematic-debugging:17`：

```
NO FIXES WITHOUT ROOT CAUSE INVESTIGATION FIRST
```

紧跟一句 `If you haven't completed Phase 1, you cannot propose fixes.`。`verification-before-completion:25` 是同样的写法：

```
NO COMPLETION CLAIMS WITHOUT FRESH VERIFICATION EVIDENCE
```

两个 skill 都在 Overview 补同一句：**"Violating the letter of this process is violating the spirit of debugging."**——预先堵掉"我理解了精神所以可以跳过步骤"这类解释。

设计意图很清楚：LLM 擅长为跳过步骤找理由，一条不带例外的硬规则比一段讲道理的说明更难被绕过。

**迁移结果**：amct-coder 的全局 BLOCKER 清单（9 条）沿用了这个写法，但**约束对象换成架构不变量**——不在 CLI/workflow/solver/通用 quant module 里加模型专属分支、不混用 LLM new-path `ALGO_REGISTRY` 与 classic `AlgorithmRegistry`、新增 dtype 必须闭环数值语义 + 注册 + 位宽粒度 + 消费入口。

**为什么必须换**：superpowers 的 Iron Law 管的是时序，违反了顶多是效率问题，重跑一次能补救；AMCT 这边违反架构不变量会产出**看起来能跑、但污染通用层**的代码，review 时不易发现，后续维护成本转移给别人。所以对 AMCT 来说，值得写成硬禁止的不是"先做什么"，而是"哪些落点绝对不能改"。这是同一个模式在不同领域的必然分化。

### 4.2 证据门禁 → 四级证据分级（模式沿用，粒度加深）

`verification-before-completion:29-44` 的 Gate Function：

```
BEFORE claiming any status or expressing satisfaction:
1. IDENTIFY: What command proves this claim?
2. RUN: Execute the FULL command (fresh, complete)
3. READ: Full output, check exit code, count failures
4. VERIFY: Does output confirm the claim?
5. ONLY THEN: Make the claim

Skip any step = lying, not verifying
```

配一张对照表（`:52-58`），关键在第三列：

| Claim | Requires | Not Sufficient |
| --- | --- | --- |
| Tests pass | 测试命令输出：0 failures | 上次跑的结果、"应该能过" |
| Linter clean | linter 输出：0 errors | 部分检查、外推 |
| Build succeeds | build 命令 exit 0 | linter 过了、日志看着正常 |
| Bug fixed | 测原始症状：通过 | 代码改了、假定修好了 |

"Not Sufficient"这一列是精髓——它没有只说该做什么，而是**逐条列出常见的偷懒替代**，把漏洞先堵上。

**迁移结果**：amct-coder 把它从两级（声称 vs 命令输出）扩成四级——**源码存在 / 单测覆盖 / 集成闭环 / 真实模型或硬件验证**，并规定低级证据不能替代高级证据；缺 NPU、CANN 版本、模型权重或数据集时必须显式写出缺什么，不能把未跑的实验说成已验证。

**为什么要加深**：superpowers 假设的是"命令能跑"的通用开发环境，两级够用。AMCT 跑在昇腾上，最常见的情况恰恰是**跑不了**——没卡、CANN 版本不对、权重没下载。这时如果只有两级，agent 只能在"声称通过"和"什么都不说"之间选，实际表现是倾向前者。四级的作用是给"我改完了代码但没跑真机"这种真实状态一个**合法的表达位置**，它才不会去伪装成更高一级。

### 4.3 description 只写触发条件（规定原样沿用）

`writing-skills:112-118` 的硬要求：第三人称、只描述触发条件、以 "Use when..." 开头、列具体症状与场景、**NEVER summarize the skill's process or workflow**、尽量 500 字符内。

理由在同文件的 SDO（Skill Discovery Optimization）章节：agent 靠读 description 决定"我现在要不要加载这个 skill"。**如果 description 概述了流程，agent 会以为自己已经知道内容，从而不去读正文。**

**迁移结果**：amct-coder 的 description 写的是触发场景与定位用途（定位 workflow/CLI、solver、dtype、LLM PTQ 算法等的修改落点），不概述内部流程。这条我基本原样沿用，因为它的失效机制与领域无关。

### 4.4 把写 skill 当 TDD 做（方法论沿用，机制化尚未完成）

`writing-skills:30-45` 给了完整映射：

| TDD 概念 | 写 Skill 对应 |
| --- | --- |
| 测试用例 | pressure scenario（给子 agent 的压力场景） |
| 生产代码 | SKILL.md |
| 测试失败（RED） | 不给 skill 时 agent 违反规则（baseline） |
| 测试通过（GREEN） | 给了 skill 后 agent 遵守 |
| 先写测试 | 写 skill **之前**先跑 baseline |
| 看它失败 | 记录 agent 用了哪些具体借口（rationalizations） |
| 最小代码 | 只针对这些具体违规写规则 |
| 重构循环 | 发现新借口 → 堵 → 重新验证 |

核心主张（`:16`）：**"If you didn't watch an agent fail without the skill, you don't know if the skill teaches the right thing."**

还有一条独立判据（`:59`）值得单记：**能用 regex 或校验机械强制的，就别写进文档；文档只留给判断题。**

**迁移结果**：做了六题有/无 skill 对照（`skill-vs-noskill-comparison.md`），这是按这套方法论做的，但**只做到一次性人工对照，没有做成可重复跑的 eval**。对比业内：官方 `skill-creator` 配了 `run_eval.py`、`aggregate_benchmark.py` 和 grader/analyzer/comparator 三个子 agent；`addyosmani/agent-skills` 有 `evals/cases` 和 `fixtures`。这是当前最明确的短板，见 §7。

---

## 5. 明确不采纳的部分，以及依据

这一节比 §4 更重要。superpowers 的问题不是设计粗糙，而是**它为了保证纪律被执行，选择了剥夺模型判断权**，而这个选择有实测代价。

### 5.1 三条独立来源指向同一个问题

以下数据来自公开评测文章，**我没有独立复现，文章标注的发表日期也无法独立核实**。三条来源互不相关但结论一致：

**（1）强制 + 费 token，小任务上不成比例。** [Glean 转载的 justinyan.me 文章](https://glean.smartcoder.ai/en/a/switching-from-superpowers-to-mattpocock-skills-less-token-w-y7bggx) 承认它能自主跑一小时，但缺点是"**强制，且费 token**"，会"轻易把简单问题复杂化"。作者的处理方式很激进：让 coding agent **删掉 superpowers 全部 hooks**，改成只在手动调用时生效。

**（2）上下文膨胀导致体感变慢，出现卸载潮。** [Nahornyi AI Lab](https://nahornyi.ai/en/news/claude-code-slowdown-superpowers-overhead) 报告装了 superpowers 后 Claude Code 变慢的普遍反馈，归因于 skill 链式加载（brainstorming → writing-plans → subagent-driven-development）造成的上下文膨胀，有用户报告小任务在新会话上磨了近一小时。一句判断值得记：**如果移除一个 skill 包能明显加速，问题就在 prompt 编排架构，而不在模型。** 作者用"token 肥胖"解释：脚手架越多，模型读自己的输出而不是读代码。

**（3）12 组对照实验：收益完全取决于任务规模。** [Mejba Ahmed 的实测](https://www.mejba.me/blog/superpowers-plugin-claude-code-review)，6 组有 / 6 组无，$2 上限，不人工干预：

| 任务规模 | 结果 |
| --- | --- |
| 简单任务 | token **更多**——"在 50 行脚本上做规划纯粹是税" |
| 中等任务 | 大致持平，输出结构更好 |
| 复杂任务 | 收益集中在这里——便宜的文字规划避免了"建错东西"这种昂贵失败 |

同一实验里最有说服力的一条正面数据：**token 方差比 baseline 收窄 2–3 倍**，成本从抽签变成可预测。

### 5.2 逐条对照：amct-coder 的取舍

| superpowers 的做法 | 代价 | amct-coder 的选择 |
| --- | --- | --- |
| 入口剥夺判断权（"1% 可能就必须用""你没有选择权"） | 小任务上开销不成比例 | 入口第一步是**判职责归属**，判完可以路由到别的 skill，也可以判定本 skill 不适用 |
| SessionStart hook 在 startup/clear/compact 三处强制注入 | 规则无法随会话推进而失效，用户被迫接受 | **不使用 hook**，靠 description 匹配触发 |
| 链式加载多个 skill | 上下文膨胀，"token 肥胖" | 显式**读取预算**：repo-map 最小切片 + 1 个主 playbook + 首轮 2～4 个源码/测试锚点，辅助 reference 默认最多 1 份，配四条停止条件 |
| 12 行 Red Flags 表逐条否决 agent 的每个例外 | 连"先看一下代码"都被禁止 | 保留"禁止用可自查的问题推回用户"这类**方向性约束**，但不否决 agent 的探索动作 |

**核心判断：superpowers 的"纪律"值得学，"强制"不值得学。**

理由不只是 token 成本。amct-coder 面对的是**多 skill 协作**的场景（amct-coder / amct-ops-dev / quant-workflow / quant-tools 下 7 个子 skill / gitcode-pr / gitcode-issue）。如果每个 skill 都照 superpowers 那样声明"我适用时你没有选择权"，多个 skill 会互相争夺入口，agent 无法路由。所以 amct-coder 入口的第一件事必须是**判职责归属并允许路由出去**，这与 superpowers 的强制模型在架构上不兼容——不是我不想学，是学了会坏。

### 5.3 一个我们同样存在、尚未解决的风险

上表第三行我写了"有读取预算"，但要如实说明：**预算本身不随任务规模缩放**。superpowers 在小任务上是纯税这个问题，amct-coder 同样存在——目前只做到"纯规范合规 review 可以跳过部分步骤"，没有做到按任务规模自动降档。这条在 §7 待办里。

---

## 6. 迁移效果：六题 A/B 对照的实际结论

`.agents/docs/amct_coder_analysis/skill-vs-noskill-comparison.md` 的结论摘要：

- 六题相对结果：**有 skill 4 胜、1 平、1 负**
- 参考分数：有 skill 约 3.5/10，无 skill 约 2.5/10
- 但文档明确声明：两个分数来源不同（前者是 reviewer 实评，后者是回顾性启发式估计），**不能当严格定量 A/B**
- 总体判断：**"Skill 改善了定位和边界意识，但尚未把 Agent 稳定提升到准确、可执行的端到端规划水平"**

**三类确认的增益**：代码体系识别更准（更少把 classic/experimental 的相似实现当目标主链）、分层意识更清楚（主动讨论 registry/algorithm/workflow/solver/adapter 职责）、未知契约意识增强（列待确认项而非自行定义语义）。

**五类仍存在的漏洞**：活跃调用链核验不稳定、跨阶段生命周期没有统一收口、事实门禁执行不到位（会用不存在的 `--w_bits`/`--a_bits`）、复杂任务路由不完整（Kimi-K3 该进 `quant-workflow` 但停在 amct-coder 视角）、局部规则正确不保证总体方案正确。

**最需要说明的是那个"1 负"（测试 5，attention + MLP 联合校准）。** 有 skill 版本更早选中一个看似符合分层的 block unit 方案，却没核对真实输入、wrapper 注入和保存链，结果**弱于**无 skill 版本。

这个结果的意义在于：**skill 不只是"有没有效"的问题，它会引入新的失败模式**——通过帮 agent 更快选定落点，反而让它更快收敛到一个错误方案，跳过了本该有的怀疑。这恰好是 `writing-skills` 讲的 refactor 循环该处理的情况：发现新的 rationalization，然后堵。对应的改法不是继续往 reference 里加文件名，而是加"反证候选方案"的要求。

---

## 7. 结论与后续

**已完成的迁移**（4 项）：Iron Law 结构 → BLOCKER 清单（约束对象改为架构不变量）；证据门禁 → 四级证据分级（绑定昇腾环境的缺失声明）；description 只写触发条件（原样沿用）；baseline-first 方法论 → 六题人工对照。

**明确不采纳**（2 项）：入口强制与判断权剥夺、SessionStart hook 注入。依据是三条独立实测来源显示的小任务开销问题，以及多 skill 协作场景下强制模型在架构上不兼容。

**后续待办**：

| 优先级 | 事项 | 依据 |
| --- | --- | --- |
| 中 | 把 A/B 对照做成可重复 eval（固定同一 rubric、盲评、原始 prompt 与 reviewer 结果分文件保存） | `writing-skills:16`、官方 `skill-creator` 的 `run_eval.py`、A/B 文档第 8 节自提 |
| 中 | 加"反证候选方案"规则（针对测试 5 的失败模式） | A/B 文档 §6.5 |
| 中 | 读取预算按任务规模降档，避免小任务上成为纯税 | §5.1 三条批评、§5.3 |
| 低 | 逐条复查 BLOCKER，把"能机械强制却写成文档"的项改成可执行检查命令 | `writing-skills:59` |

**相关文档**：
- `.agents/docs/superpowers-reference-analysis.md` —— superpowers 的完整逐节分析（含源码行号引用）
- `.agents/docs/other-skills-reference-analysis.md` —— 厂商/仓库专属型样板（Cloudflare、Convex、cuda-kernels 等），与 amct-coder **同类**的参考，含另一批待办
- `.agents/docs/amct_coder_analysis/skill-vs-noskill-comparison.md` —— 六题 A/B 对照原始记录


