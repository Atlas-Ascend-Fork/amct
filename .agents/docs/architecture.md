# amct 量化 Agent —— 架构与设计

> 本文档说明 amct 大模型量化 agent（`.agents/`）的整体架构、分层职责与设计依据，供评审、维护与二次开发参考。

## 1. 总览

面向「输入模型 → 完成量化适配 → 产出可部署权重」的端到端研发，采用业界主流的 **单编排入口 + 专职子代理 + 叶子技能** 分层：

```text
AGENTS.md / .agents/README.md          入口说明与路由
        │
quant-workflow（唯一编排入口）          判阶段 → 串联/分流 → 复用 casebook → 汇总
        │  调度
   ┌────┴───────────────┐
quant-analyzer  quant-implementer  quant-reviewer    三类专职子代理（分析 / 实施 / 审查）
        │  各自 skills: 挂载
quant-tools/*（含 model-adapter 等叶子技能，真正干活的最小单元）
        │  消费
docs/casebook/  +  docs/repo-map.md     知识层（适配经验 / 仓内导航）
```

## 2. 分层职责

| 层 | 组件 | 职责 |
|---|---|---|
| 入口 | `AGENTS.md` / `.agents/README.md` | 说明与路由：端到端 → quant-workflow；单点需求 → 对应叶子 |
| 编排 | `skills/quant-workflow/` | **唯一编排入口**：阶段判断、串联/分流、复用已有结果、deploy 交付边界、文档写回规则——薄，不重写叶子逻辑 |
| 工作者 | `agents/quant-{analyzer,implementer,reviewer}.md` | 按 **分析 / 实施 / 审查** 隔离：analyzer 只读+设计方案、implementer 跑 CLI（统一 quant-run）+改 adapter、reviewer 只读结果判读·不跑评测/ptq |
| 叶子 | `skills/quant-tools/*`（含 `model-adapter`）| 最小可执行技能：方案推荐 / 算法推荐 / 统一执行（quant-run）/ 直转判读 / 算法收益判读 / 部署导出、新模型适配 |
| 知识 | `docs/casebook/`、`docs/repo-map.md` | 模型适配经验（hard bug/陷阱/精度）与仓内导航，供方案推荐复用 |
| 共享 | `skills/quant-tools/references/skill-input-template.md` | 4 个 skill 共享的用户输入模板 |
| 协作 | `skills/gitcode-pr`、`gitcode-issue`、`default-skills` | 通用协作（PR/issue/按需装 skill），非量化功能 |

## 3. 单一源投影机制（Single Source of Truth）

`.agents/` 是**唯一源**（git tracked）；`scripts/init-agent.sh` 把它投影成两套 runtime 视图，分两种方式：

- **符号链接**（源改即时生效）：`skills` / `agents` / `hooks` / `docs`，及 `CLAUDE.md → AGENTS.md`。
- **拷贝**（源改后须重跑 `init-agent.sh` 才同步）：`settings.json → .claude/`、`opencode.json → .opencode/`。

视图分布：`.claude/{skills,agents,docs,settings.json}` + `CLAUDE.md`；`.opencode/{skills,agents,docs,opencode.json}`。两端视图 **全 gitignored、全量生成**，clone 后跑 `bash scripts/init-agent.sh` 生成。一处源、双端一致，不提交生成物、不双份维护。

## 4. 设计原则

1. **单一编排入口**：端到端只走 `quant-workflow`，杜绝多入口路由歧义。
2. **编排薄、逻辑下沉**：编排只做拆解/路由/汇总，领域逻辑全在叶子技能。
3. **职责隔离**：子代理按 plan(analyzer) / execute(implementer) / review(reviewer) 硬边界，analyzer 不改代码、reviewer 不改方案。
4. **单一事实源**：口径/阈值/交互门/写回规则只在编排入口写一处，避免多处漂移。
5. **casebook 驱动**：方案推荐优先复用同系列实测经验；casebook 只沉淀「后续其他网络会复用、且不易解」的 hard bug，不记过程流水。
6. **human-in-the-loop**：编排明列 方案选择 / 升级 PTQ / 算法选择 / 是否导出 四个强制确认门，非全自动。

## 5. 为什么单编排入口（依据）

采用业界主流的「单编排 / 路由 + 工作者 + 工具」范式：Anthropic *orchestrator-workers / routing*、OpenAI Agents SDK *triage + handoffs*、LangGraph *supervisor*、CrewAI *hierarchical manager*、Claude Code *主 agent + subagents* 均如此；Anthropic 并强调“按需才分层，不为分层而分层”。本 agent 的「单 `quant-workflow` + 三子代理 + 叶子技能」即此形态。

## 6. casebook 设计

- **目的**：让 agent 适配新模型能力更强——是手工/自动适配过程沉淀下来的可复用经验。
- **组织**：按 vendor 系列、对齐源码 `amct_pytorch/common/models/llm/<vendor>/`（qwen / deepseek / glm / longcat）；`<vendor>/README.md` 系列总览 + `<vendor>/<model>.md` 个案。
- **模板**（`case-template.md`）：速览 → 结构与适配要点（参考与差异 / 复用·新增 / 起步复用清单）→ 适配验证结论 → 关键陷阱（现象→根因→处理→教训）→ 量化结论 → 适配建议（先做 / 不建议）→ 精度速查表。
- **收录原则**：只放典型/关键、模型或框架内在、可迁移的 bug/陷阱 + 适配重点 + 精度；agent 环境配置/过度推断类、一次性过程流水不入库。

## 7. 运行与验证

- 启动：`bash scripts/init-agent.sh` 生成视图后，在 Claude Code 或 OpenCode 中启动。
- 已实测（OpenCode + glm-5）：执行流（W8A8 直转）、分析流（方案推荐）、PTQ 流（extract → ptq → eval）三类端到端均由 agent 自主跑通且结果核实准确。

## 8. 面向 agent 集成（黑盒契约）

上游 / 多 agent 集成**只对接唯一入口 `quant-workflow`**，不直接调子代理或叶子技能（封装；对应 orchestrator-workers / handoff 范式）。最小契约：

- **能力 + 触发**：见 `quant-workflow` frontmatter `description`（机读接口；README 面向人）。
- **输入**：模型路径 + 工作目录；可选 `quant_target / bits / device`。
- **输出 / 状态面**：`progress.md` 顶部机读状态块（`STAGE / STATUS / DELTA / ARTIFACTS / BLOCKED`）可轮询；deploy 产物为终态交付物。同框架委派则另取子代理返回摘要。
- **交互模式**：human-in-the-loop——方案选择 / PTQ 升级 / 算法选择 / 是否导出 四个强制确认门，非全自动。
- **前置**：`amct_pytorch` 可导入、NPU device 由调用方 `--device` 提供、模型与数据可达；不满足则 fail-fast 写 `BLOCKED`（环境是调用方责任，agent 不自行排障）。
- **重入安全**：重复对接先复用 `progress.md` 已有结果，不重跑。
