# AMCT Coder 贡献流程

本文件只在需要创建 Issue/PR 或读写 GitCode 时读取。代码风格、版权头和 pre-commit 属于改动后的静态验证，见 [validation.md](validation.md)。

## Issue 与 PR 门槛

- 简单 bug 修复可以直接创建 PR。
- 新功能或接口变更必须先创建 Issue 讨论并达成共识，再进入 PR；不要用 PR 描述替代未经讨论的接口决策。

## 模板

- 创建 PR 前按 `.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md` 组织内容。
- 创建 Issue 时按类型选择 `.gitcode/ISSUE_TEMPLATE/` 下的模板：`bug_report.yaml`、`feature_requeset.yaml`、`documentaion.yaml`、`question.yaml`。

## PR 内容要求

- 必须关联对应 Issue。
- 包含特性代码本身，以及受影响的文档和 README 更新；文档改动范围以本次特性实际影响为准，不做无关文档 churn。

## GitCode 操作

实际读取、创建或操作 GitCode Issue/PR 时，加载并遵循仓内 `gitcode-issue` 或 `gitcode-pr` skill。当前会话未发现对应 skill 时，加载 `default-skills` 并按其流程补装缺失项。不要绕过这些 skill 直接调用 API 或拼 URL。
