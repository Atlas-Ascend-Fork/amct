# AMCT 项目规范

修改代码前读取本文件，做规范合规 review 时也读取；只做实现规划、行为诊断或代码行为 review 时不读。这里定义 AMCT 修改前和规范合规 review 的约束；检查命令和证据要求见 [validation.md](validation.md)。

## 代码风格与版权

- Python 遵循 PEP8，使用 Ruff 检查和格式化。
- C/C++ 遵循 Google 代码规范，使用仓库 `.clang-format`：4 空格缩进，行宽 120 字符。
- 提交前安装并运行 pre-commit，只检查本次改动文件，避免全仓存量问题掩盖新增问题。
- 新增代码保留仓库既有的 Apache 2.0 版权声明；修改现有文件时不得删除或破坏版权头。

读取本文件只建立约束，不表示立即安装工具。准备提交且环境未安装 pre-commit 时再安装；具体检查命令按 `validation.md` 执行。

## Issue 与 PR 门槛

- 简单 bug 修复可以直接提交 PR。
- 新功能或接口变更必须先通过 Issue 讨论并达成共识，再提交 PR。
- PR 必须关联对应 Issue，并包含本次变更的代码、优化文档和 README 更新；不做无关文档改动。

开始实现前先判断当前任务是否属于新功能或接口变更。需要先讨论但用户只授权本地编码时，说明贡献门槛，不擅自创建 Issue、推送代码或创建 PR。

## Issue 与 PR 模板

- PR 使用 `.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md`。
- Issue 从 `.gitcode/ISSUE_TEMPLATE/` 按类型选择：
  - `bug_report.yaml`
  - `feature_requeset.yaml`
  - `documentaion.yaml`
  - `question.yaml`

## GitCode 操作

- 涉及读取或操作 GitCode Issue 时，加载并遵循仓内 `gitcode-issue` skill。
- 涉及创建或操作 GitCode PR、推送代码时，加载并遵循仓内 `gitcode-pr` skill。
- 对应 skill 缺失时，加载 `default-skills`，按其流程补装缺失项；已有版本不因缺失检查而覆盖。
- 不绕过上述 skill 直接拼接 GitCode URL 或自行调用 API。

只做本地代码修改时不加载 GitCode skill。实际进入 Issue/PR 操作前，再读取 [contribution-flow.md](contribution-flow.md) 并遵守用户授权范围。
