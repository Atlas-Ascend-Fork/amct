# amct-coder 设计迭代记录

本文记录 `amct-coder` 从初版设计审视到主 `SKILL.md` 重构的讨论过程，供后续维护者理解关键取舍。具体运行规则以 `.agents/skills/amct-coder/SKILL.md` 和命中的 reference 为准，完整设计以 `amct-coder-final-design-v0.md` 为准。

## 1. 初始目标

初版把 `amct-coder` 定位为 AMCT 仓编码总入口，主要负责：

- 识别代码体系和修改落点。
- 拦截 CLI、workflow、solver、通用 quant module 中的模型专属逻辑。
- 路由 dtype、算法、workflow、solver、deploy、classic、ops boundary、public API 等子能力。
- 修改结束后选择最小验证。

初版已经具备 9 个子能力 playbook，但主文件同时包含依赖链、常见映射、越层示例、全局边界和验证说明，运行时入口偏重。

## 2. 第一轮审视：先修职责和继续执行语义

讨论首先发现三个关键问题：

1. 冲突裁决写成“用户要求优先”，可能让用户指定的错误实现路径覆盖仓库边界。
2. BLOCKER 容易被理解为终止整个任务，而不是只阻止错误方案。
3. “委派到其它 skill”容易退化为只回复路由建议，不继续完成用户请求。

形成的结论：

- 当前作用域 `AGENTS.md`、独立 skill 职责、repo-map 分层边界和公共契约优先于用户指定的代码落点。
- 保留用户业务目标，但拒绝越层实现；正确替代方案明确时直接继续。
- 命中外部 skill 后切换其流程并继续执行；混合任务按职责拆分，最后统一做契约对齐和汇总。
- 区分三类 deploy 请求：修改 deploy 实现归 `deploy-export-code`，实际导出交付归 `deploy-export`，eval/extract/PTQ 跑数归 `quant-run`。

## 3. 第二轮审视：阶段不是代码职责

要求 LLM new-path 任务必须先判 eval / extract / ptq / deploy 阶段。讨论后认为，阶段适合描述运行数据流，不适合作为编码任务的首要分类。

原因：

- dtype、quant module、algorithm 是横向能力，可能同时影响 eval、ptq 和 deploy。
- 参数保存、unit 命名、deploy payload 等是跨阶段契约，应表达为“主阶段 + 受影响阶段”。
- classic、ops、public API/packaging 不属于 LLM 四阶段。
- `deploy`、`eval` 等关键词不能区分“修改实现”与“运行命令”。

因此把“必须判阶段”改为“按需检查阶段影响”：

- 先确定任务职责、仓库层和主子能力。
- 修改 LLM CLI/workflow、阶段产物或跨阶段契约时，再检查主阶段和受影响阶段。
- 通用能力列出需要闭环的阶段，不强行选择单阶段。
- `quant-workflow` 仍以生命周期阶段为核心；`amct-coder` 以代码职责为核心。

## 4. 第三轮审视：渐进披露与编码前必读

初版一方面要求渐进披露，另一方面要求所有 LLM/cross-layer 任务固定通读 repo-map、casebook 和 `AGENTS.md`，两者存在冲突。

讨论形成两层读取模型：

- 基础读取：完成当前编码职责所必需的稳定边界、主 playbook、目标源码/测试和 validation。
- 条件读取：repo-map 其它章节、casebook、attention/wrapper 规则、上下游 playbook。

`AGENTS.md` 是当前作用域已经生效的约束，不要求为了摘抄规范重复通读。casebook 只在模型结构、checkpoint/config、模型契约或相关精度异常时加载。

## 5. 第四轮审视：repo-map 边界读取前置

随后进一步明确：既然 `amct-coder` 的核心职责是分层和越层判断，就不应先凭关键词猜落点，判断困难后才补读 repo-map。

最终采用两级入口：

1. 先按用户动作和独立 skill 职责排除跑数、判读、模型结构实现、NPU kernel、deploy 交付、GitCode 操作等外部任务。
2. 确认由 `amct-coder` 承接编码或 review 后，强制读取 repo-map 的「分层边界」「更新规则」，抽查候选源码锚点并形成边界声明。

边界声明包含：主落点层、允许范围、禁止事项、必要辅助层、repo-map 是否需要写回。

repo-map 读取前置不表示默认写回。只有框架边界、核心流程、注册集合、公共契约变化，或 map 已确认落后于预期源码事实时更新；局部 bug、普通算法文件和不改变复用边界的模型细节默认不更新。

## 6. 复杂度控制

讨论确认设计文档可以保留理由和裁决过程，但运行时 `SKILL.md` 不能照搬完整设计。信息按用途分布：

| 内容 | 位置 |
| --- | --- |
| 运行时必须执行的判断和硬规则 | `SKILL.md` |
| 子能力执行步骤、BLOCKER、验证方法 | `references/*.md` |
| 设计理由、方案比较、冲突分析 | 本目录设计文档 |
| 稳定仓库分层和架构事实 | `repo-map.md` |
| 模型经验和已踩坑 | `casebook/` |

复杂度控制规则：

- 每次先确定一个主子能力，其它只作为辅助闭环。
- 固定读取 repo-map 最小边界切片、一个主 playbook 和 validation。
- 默认再追加最多一份辅助资料；真实跨契约任务分阶段加载。
- 不在主文件重复维护完整数据流、模型名单、详细测试命令和大段示例。

## 7. 当前主流程

最终将 `SKILL.md` 收敛为以下流程：

```text
动作与任务职责归属
  -> 外部 skill 路由并继续执行
  -> amct-coder 承接后读取 repo-map，声明边界
  -> 判定代码体系，选择一个主子能力
  -> 按需检查阶段影响并加载辅助资料
  -> 读取源码锚点，编码或 review
  -> 读取 validation.md，执行最小验证
```
