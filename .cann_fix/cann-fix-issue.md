# CANN Issue 读取报告

## Issue 信息
- **URL**: https://gitcode.com/cann/amct/issues/146
- **仓库**: cann/amct
- **Issue**: #146
- **标题**: [Bug-Report|缺陷反馈]: pre-commit ruff-check 失败 364 处（F841/E741/F811），静态检查门禁为红
- **状态**: open
- **作者**: gothizzz
- **Assignee**: fujun19
- **标签**: bug

## Issue 正文

### 问题描述
仓库已配置 `.pre-commit-config.yaml`（含 `clang-format`、`ruff-check`、`ruff-format` 三个 hook），但当前 master 分支执行 `pre-commit run --all-files` 时，`ruff-check` 失败，共报告 364 处问题。

### 期望
- 对 F841：确认是否为遗漏的断言，补全断言或显式删除无用赋值
- 对 E741：将变量名 `l` 重命名为更具描述性的名字
- 对 F811：移除重复定义或重命名
- 使 `pre-commit run --all-files` 全量通过

## 评论 (1)
### yaoguangxiu - 2026-06-25T18:59:27+08:00
感谢你的发现，我们将check下test pre-commit规则
