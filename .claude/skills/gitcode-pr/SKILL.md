---
name: gitcode-pr
description: |
  使用 GitCode API 创建 Pull Request 和获取 PR 评论。当用户需要**创建** GitCode PR、将代码**推送并创建合并请求**、**获取 PR 评论**、**查看 PR 讨论**、**查看 PR 改动**或**删除 PR 评论**时使用此 skill。支持读取普通评论和行内(diff)评论，包括评论内容、文件路径、代码行号等详细信息。
  **必须触发此 skill 的场景**（用户提到以下任何内容时使用）：
  - 创建/提交 PR：创建PR、提个PR、发PR、做个PR、帮我PR、生成PR、需要PR、pull request、merge request
  - 推送代码到远程：push代码、推代码、把代码推上去、提交到远程、推送到gitcode、提交代码到GitCode
  - 合并请求：合并请求、代码合入请求、请求合并、merge request
  - PR模板/描述：PR模板、PR描述、PR格式
  - 关联issue创建PR：关issue的PR、关联issue创建PR
  - 获取PR改动：查看PR变更、PR文件列表、PR改了什么、看PRdiff、获取PR文件
  - **获取 PR 评论**：查看PR评论、PR评论、获取评论、read comments
  - **查看 PR 讨论**：PR discussions、查看讨论、discussions
  - **删除 PR 评论**：删除评论、删除PR评论、移除评论、delete comment、移除这条评论
---

# GitCode PR Skill

创建 GitCode Pull Request 和获取 PR 评论的标准化流程。

## 工作流程

### 1. 获取访问令牌（第一步必须）

**询问用户**："请提供您的 GitCode 访问令牌（Access Token）"

检查环境变量：
```bash
echo $GITCODE_API_TOKEN
```

如果不存在，提示用户获取令牌：
1. 登录 [GitCode](https://gitcode.com)
2. 点击头像 → 设置 → 访问令牌
3. 创建新令牌，选择 `read_repository`、`write_repository` 和 `read_api` 权限
4. 复制令牌，建议保存到 `~/.bashrc`：`export GITCODE_API_TOKEN="your_token_here"`

### 2. 识别PR的目标仓库

#### 2.1 查询远程仓库

```bash
git remote -v
```

根据远程仓库 URL 确定目标仓库：
- **目标仓库**：PR 要合并到的仓库（通常是 origin 或 upstream）
- **源仓库**：当前工作分支所在的仓库

#### 2.2 提取仓库 owner/repo 信息

**关键**：所有 API 调用都需要使用当前仓库的 owner/repo，而非硬编码。

```bash
# 从远程 URL 提取 owner 和 repo
repo_url=$(git remote get-url origin)

# 处理不同 URL 格式
# SSH 格式: git@gitcode.com:owner/repo.git
# HTTPS 格式: https://gitcode.com/owner/repo.git
if [[ $repo_url == git@* ]]; then
  owner=$(echo $repo_url | sed 's|.*:\([^/]*\)/\([^/]*\)\.git$|\1|')
  repo=$(echo $repo_url | sed 's|.*:\([^/]*\)/\([^/]*\)\.git$|\2|')
else
  owner=$(echo $repo_url | sed 's|.*gitcode\.com/\([^/]*\)/\([^/]*\)\.git$|\1|')
  repo=$(echo $repo_url | sed 's|.*gitcode\.com/\([^/]*\)/\([^/]*\)\.git$|\2|')
fi

# URL 编码（用于 API 路径）
encoded_repo=$(printf '%s' "${owner}/${repo}" | jq -sRr @uri)

echo "Owner: $owner"
echo "Repo: $repo"
echo "Encoded: $encoded_repo"  # 例如: cann%2Fge
```

**后续所有 API 调用都应使用这些变量**：
- GitLab API v4 格式：`/projects/${encoded_repo}/...`
- GitHub API v5 格式：`/repos/${owner}/${repo}/...`

#### 2.3 查询 Fork 的原仓库（当当前仓库是 fork 时）

当当前仓库是 fork 仓库时，需要查询其 fork 的原仓库作为 PR 目标仓库：

```bash
# 查询仓库信息获取 fork 来源
curl -s -H "Authorization: Bearer $GITCODE_API_TOKEN" \
  "https://api.gitcode.com/api/v5/repos/${owner}/${repo}" | \
  jq -r '
    if .fork then
      "原仓库: " + .parent.full_name + "\n" +
      "原仓库URL: " + .parent.html_url + "\n" +
      "目标分支: " + .parent.default_branch
    else
      "这不是一个 fork 仓库，直接使用当前仓库"
    end
  '
```

**响应关键字段**：
| 字段 | 说明 |
|------|------|
| `fork` | 是否为 fork 仓库（`true`/`false`） |
| `parent.full_name` | 原仓库完整名称（格式：`owner/repo`） |
| `parent.html_url` | 原仓库网页地址 |
| `parent.default_branch` | 原仓库默认分支 |

**示例输出**：
```json
{
  "fork": true,
  "parent": {
    "full_name": "cann-agent/skills",
    "html_url": "https://gitcode.com/cann-agent/skills",
    "default_branch": "main"
  }
}
```

### 3. 获取 PR 评论和讨论

#### 获取 PR 讨论列表（包含行内评论）

GitCode 使用 GitLab API v4 格式获取 PR 讨论和评论：

```bash
# 获取 PR 的所有讨论（包括行内评论）
curl -s -H "PRIVATE-TOKEN: $GITCODE_API_TOKEN" \
  "https://api.gitcode.com/api/v4/projects/${encoded_repo}/merge_requests/<PR_NUMBER>/discussions"
```

**关键说明**：
- 认证头使用 `PRIVATE-TOKEN:`（不是 `Authorization: Bearer`）
- 项目路径需要 URL 编码：`${encoded_repo}` = `owner%2Frepo`
- 使用 `merge_requests` 而不是 `pulls`
- 使用 GitLab API v4: `/api/v4/projects/`

#### 解析讨论数据

返回的讨论数据包含以下关键字段：

| 字段 | 说明 |
|------|------|
| `notes[].type` | 评论类型：`DiffNote` 表示行内评论，`DiscussionNote` 表示普通讨论 |
| `notes[].body` | 评论内容 |
| `notes[].author` | 评论作者信息 |
| `notes[].position` | 行内评论的位置信息 |
| `notes[].diff_file` | 被评论的文件路径 |
| `notes[].new_line` | 新代码行号 |
| `notes[].content` | 被评论的具体代码行内容 |
| `notes[].resolved` | 评论是否已解决 |
| `notes[].created_at` | 评论创建时间 |

**行内评论的 position 字段结构**：
```json
{
  "base_sha": "base提交SHA",
  "start_sha": "start提交SHA",
  "head_sha": "head提交SHA",
  "old_path": "旧文件路径",
  "new_path": "新文件路径",
  "old_line": null,
  "new_line": 46,
  "diff_id": 5664724
}
```

### 4. 获取 PR 文件变更

```bash
# 基本查询
curl -s -H "PRIVATE-TOKEN: $GITCODE_API_TOKEN" \
  "https://api.gitcode.com/api/v4/projects/${encoded_repo}/merge_requests/<PR_NUMBER>/changes"

# 每页 100 条
curl -s -H "PRIVATE-TOKEN: $GITCODE_API_TOKEN" \
  "https://api.gitcode.com/api/v4/projects/${encoded_repo}/merge_requests/<PR_NUMBER>/changes?per_page=100"
```

### 5. 提交行内评论（支持多行选择）

#### 创建新的 Discussion（推荐）

```bash
curl -s -X POST \
  -H "PRIVATE-TOKEN: $GITCODE_API_TOKEN" \
  -H "Content-Type: application/json" \
  "https://api.gitcode.com/api/v4/projects/${encoded_repo}/merge_requests/<PR_NUMBER>/discussions" \
  -d '{
    "repoId": "'"${encoded_repo}"'",
    "iid": <PR_NUMBER>,
    "body": "评论内容",
    "line_types": "new",
    "position": {
      "base_sha": "<base_commit_sha>",
      "start_sha": "<start_commit_sha>",
      "head_sha": "<head_commit_sha>",
      "position_type": "text",
      "old_path": "文件路径",
      "new_path": "文件路径",
      "old_line": null,
      "new_line": <结束行号>,
      "start_old_line": null,
      "start_new_line": <起始行号>,
      "ignore_whitespace_change": false
    },
    "assignee_id": <用户ID>,
    "proposer_id": <用户ID>,
    "severity": "suggestion"
  }'
```

**多行选择说明**：
- `start_new_line`: 选中的起始行号（多行选择时设置）
- `new_line`: 选中的结束行号
- 单行评论时，`start_new_line` 和 `new_line` 设置为相同值

**参数说明**：

| 参数 | 说明 | 必需 |
|------|------|------|
| `body` | 评论内容 | ✅ |
| `line_types` | `"new"` 选择新代码（右侧），`"old"` 选择旧代码（左侧） | ✅ |
| `position.base_sha` | base 提交 SHA | ✅ |
| `position.start_sha` | start 提交 SHA | ✅ |
| `position.head_sha` | head 提交 SHA | ✅ |
| `position.new_path` | 文件相对路径 | ✅ |
| `position.new_line` | 结束行号 | ✅ |
| `position.start_new_line` | 起始行号（多行选择） | 多行时 |
| `severity` | 严重程度：`suggestion`、`warning` | ❌ |

#### 回复已有 Discussion

```bash
curl -s -X POST \
  -H "PRIVATE-TOKEN: $GITCODE_API_TOKEN" \
  -H "Content-Type: application/json" \
  "https://api.gitcode.com/api/v4/projects/${encoded_repo}/merge_requests/<PR_NUMBER>/discussions/<DISCUSSION_ID>/notes" \
  -d '{
    "body": "回复内容"
  }'
```

### 6. 提交普通评论

```bash
curl -s -X POST \
  -H "PRIVATE-TOKEN: $GITCODE_API_TOKEN" \
  -H "Content-Type: application/json" \
  "https://api.gitcode.com/api/v4/projects/${encoded_repo}/merge_requests/<PR_NUMBER>/notes" \
  -d '{
    "body": "评论内容"
  }'
```

### 7. 删除 PR 评论

当用户需要删除 PR 中的评论时使用此功能。

**重要**：只能删除自己创建的评论，或具有仓库管理权限。

```bash
# 获取评论 ID
curl -s -H "PRIVATE-TOKEN: $GITCODE_API_TOKEN" \
  "https://api.gitcode.com/api/v4/projects/${encoded_repo}/merge_requests/<PR_NUMBER>/notes" | \
  jq '.[] | {id: .id, author: .author.username, body: .body}'

# 删除评论
curl -s -X DELETE \
  "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/comments/<COMMENT_ID>?access_token=$GITCODE_API_TOKEN"
```

详细 API 参数、响应码和权限说明请参考 `references/gitcode_api.md` 的「删除 PR 评论」章节。

### 8. 创建 PR 的正确流程

**关键**：源分支必须基于目标分支，确保PR只包含期望的变更。

```bash
# 1. 拉取目标分支
git fetch origin <目标分支>

# 2. 基于目标分支创建新分支
git checkout -b <新分支名> origin/<目标分支>

# 3. Cherry-pick需要的commit
git cherry-pick <commit-sha>  # 单个commit
# 或
git cherry-pick <sha1> <sha2>  # 多个commit

# 4. 推送新分支
git push <个人远程仓库名> <新分支名> -u
```

**示例：往 origin/9.0.0 提交单个commit**
```bash
git fetch origin 9.0.0
git checkout -b fix/gcc13-link-error origin/9.0.0
git cherry-pick 4fbf3b183
git push hgjupstream fix/gcc13-link-error -u
```

### 9. 创建 PR

**目标分支策略**：
- **默认合入 `develop` 分支**（日常开发、新功能、bugfix）
- 仅当用户明确要求时，才合入其他分支（如 `master`、`release/x.y`）

使用 GitCode API 创建 PR：

```bash
curl -s -X POST "https://gitcode.com/api/v5/repos/${owner}/${repo}/pulls" \
  -H "Authorization: Bearer $GITCODE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "docs: 优化文档描述(#32)",
    "head": "username:fix/issue-32-description",
    "base": "develop",
    "body": "PR描述内容（见下方模板）"
  }'
```

**参数说明**：
- `title`: PR 标题（必填）
- `head`: 源分支，格式 `username:branch`（必填）
- `base`: 目标分支，**默认为 `develop`**，除非用户明确要求合入其他分支
- `body`: PR 描述（Markdown 格式）

### 10. PR URL 格式

创建 PR 后，PR 的访问地址格式为（注意是 `/pull/` 而非 `/pulls/`）：

```
https://gitcode.com/${owner}/${repo}/pull/<PR_NUMBER>
```

**示例**：`https://gitcode.com/cann/ge/pull/1807`

### 11. 后续跟进

- 检查 CI/CD 运行结果
- 回复审查者意见
- 根据反馈更新代码

---

## PR 代码审查

当用户说"检视 PR"、"审查 PR"、"review PR"、"给 PR 提意见"时触发。

### 执行步骤

1. **读取 commands/review.md** - 使用 Read 工具获取完整审查流程

2. **执行 review.md 中的审查流程**：
   - 步骤 1: 前置检查（PR 状态、草稿、是否已审查）
   - 步骤 2: 获取项目规范上下文
   - 步骤 3: 获取 PR 变更摘要
   - 步骤 4: 代码审查（Bug 扫描、规范合规性）
   - 步骤 5: 验证问题
   - 步骤 6: 过滤问题
   - 步骤 7: 输出审查摘要
   - 步骤 8: 准备评论列表（仅当提供 `--comment` 时）
   - 步骤 9: 发布行内评论（仅当提供 `--comment` 时）

**注意**：review.md 中包含每个步骤的详细说明和 API 示例。

---

## 输出格式

### 评论列表格式

获取 PR 评论后，使用以下格式输出：

```markdown
# PR 评论摘要

## 总体统计
- 讨论数量: X
- 行内评论: Y
- 已解决评论: Z

## 行内评论详情

### 1. 文件路径 - 第 N 行

**评论内容**: ...
**评论者**: @username
**时间**: YYYY-MM-DD HH:mm
**状态**: 未解决/已解决

**代码片段**:
```cpp
// 被评论的代码行
```

---

## PR 标题格式

遵循 Conventional Commits 规范：

```
<type>: <描述>(#issue_id)
```

**类型**：

| 类型 | 说明 |
|------|------|
| `feat` | 新功能 |
| `fix` | Bug 修复 |
| `docs` | 文档更新 |
| `style` | 代码风格 |
| `refactor` | 重构 |
| `perf` | 性能优化 |
| `test` | 测试相关 |
| `chore` | 构建/工具 |

**示例**：
- `docs: 优化docs/api/README.md中的ge命名空间描述(#32)`
- `fix: 修复三库json下载安装问题`
- `feat: 添加operator注册V2接口支持(#45)`

---

## PR 描述模板

创建 PR 时使用以下模板格式化描述。

**重要**：
- **变更类型**：根据实际变更内容，将对应选项的 `[ ]` 改为 `[x]` 勾选
- **核对清单**：提交 PR 前所有项都应满足，默认全部勾选 `[x]`

```markdown
# Pull Request

## 描述
<!-- 根据代码变更内容填写描述 -->

## 变更类型
请选择本次引入的变更类型（勾选对应项）：
- [ ] 🐛 Bug 修复
- [ ] ✨ 新功能
- [ ] 💄 代码风格更新（格式化，局部变量）
- [ ] ♻️ 重构（既不修复错误也不增加功能的代码变动）
- [ ] 📦 构建过程或辅助工具的变动
- [ ] 📝 文档内容更新

## 关联的Issue
Closes #<issue_number>

## 如何测试
描述测试此变更的步骤和前提条件：

## 核对清单
- [ ] 我的代码遵循了项目的代码风格
- [ ] 我已对代码进行了自测
- [ ] 我已更新了相关的文档
- [ ] 我在标题中使用了合适的类型标签（如：`feat:`, `fix:`）
- [ ] 我已经详细阅读了贡献指南（CONTRIBUTING.md）

## 其他信息
在此添加任何其他关于本次 PR 的说明。
```

---

## Resources

### references/gitcode_api.md

GitCode API 完整参考文档，用于：
- 了解 API 参数格式
- 查看响应结构
- 排查 API 调用问题
