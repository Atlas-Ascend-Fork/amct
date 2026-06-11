---
name: default-skills
description: |
  **必须触发此 default-skills 的场景**：① gitcode-pr 或 gitcode-issue skill 缺失/找不到时（补装缺失，不覆盖已有版本）；② 用户明确要求更新/同步上游 skill 时（使用 --force 覆盖）；
---

## 执行步骤

**第一步（必须）**：读取 `.claude/skills/default-skills/scripts/install-default-skills.sh` 获取 `DEFAULT_SKILLS` 列表

**场景一：补装缺失（默认）**
1. 执行 `.claude/skills/default-skills/scripts/install-default-skills.sh`（已存在的 skill 自动跳过）
2. 检查 `.claude/skills/` 目录是否已有所有 `DEFAULT_SKILLS`，有则结束；若脚本执行失败，执行步骤 3
3. 兜底（脚本失败时手动复刻，与脚本逻辑等价）：`git clone --depth 1 https://gitcode.com/cann-agent/skills.git` 到临时目录，将缺失的 `DEFAULT_SKILLS` 拷贝到 `.claude/skills/`

**场景二：强制更新（用户明确要求更新/同步上游时）**
1. 执行 `.claude/skills/default-skills/scripts/install-default-skills.sh --force`（从上游覆盖所有 skill）
2. 若脚本失败，同场景一步骤 3 手动兜底（不做存在性检查，直接覆盖）
3. 提示用户：本地对这些 skill 的定制改动已被上游版本覆盖

## 默认skills使用场景
1. **必须触发 gitcode-issue 的场景**（用户提到以下任何内容时使用）：
   - 查看/读取 issue：查看issue、看看issue、读取issue、打开issue、issue详情、issue是什么
   - GitCode URL：gitcode.com/**/issues/**、cann/amct/issues、issue链接
   - 直接说编号：issue 123、#123、问题123
   - 查看评论：issue评论、评论内容

2. **必须触发此 gitcode-pr 的场景**（用户提到以下任何内容时使用）：
   - 创建/提交 PR：创建PR、提个PR、发PR、做个PR、帮我PR、生成PR、需要PR、pull request、merge request
   - 推送代码到远程：push代码、推代码、把代码推上去、提交到远程、推送到gitcode、提交代码到GitCode
   - 合并请求：合并请求、代码合入请求、请求合并、merge request
   - PR模板/描述：PR模板、PR描述、PR格式
   - 关联issue创建PR：关issue的PR、关联issue创建PR
   - 获取PR改动：查看PR变更、PR文件列表、PR改了什么、看PRdiff、获取PR文件
   - **获取 PR 评论**：查看PR评论、PR评论、获取评论、read comments
   - **查看 PR 讨论**：PR discussions、查看讨论、discussions
   - **删除 PR 评论**：删除评论、删除PR评论、移除评论、delete comment、移除这条评论
