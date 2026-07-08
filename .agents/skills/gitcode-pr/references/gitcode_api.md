# GitCode PR API 参考

## 目录

1. [认证方式](#认证方式)
2. [获取仓库信息（查询 Fork 来源）](#获取仓库信息查询-fork-来源)
3. [获取 PR 信息](#获取-pr-信息)
4. [获取 PR 讨论（包含行内评论）](#获取-pr-讨论包含行内评论)
5. [提交行内评论（支持多行选择）](#提交行内评论支持多行选择)
6. [提交普通评论](#提交普通评论)
7. [删除 PR 评论](#删除-pr-评论)
8. [获取 PR 文件变更](#获取-pr-文件变更)

---

## 认证方式

GitCode 使用两种 API 风格：

| API 风格 | 认证头 | 端点格式 |
|---------|-------|---------|
| **GitLab API v4**（推荐） | `PRIVATE-TOKEN: <token>` | `/api/v4/projects/<encoded_path>/...` |
| GitHub 兼容 | `Authorization: Bearer <token>` | `/api/v5/repos/<path>/...` |

**项目路径编码**：`owner/repo` → `owner%2Frepo`（使用 `jq -sRr @uri` 或 `printf '%s' "owner/repo" | jq -sRr @uri`）

---

## 获取仓库信息（查询 Fork 来源）

### API 端点

查询仓库详情，判断是否为 fork 仓库并获取其原仓库信息。

```bash
GET https://api.gitcode.com/api/v5/repos/:owner/:repo
```

### 路径参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `owner` | string | ✅ | 仓库所属空间地址（组织或个人的地址 path） |
| `repo` | string | ✅ | 仓库路径（path） |

### 请求示例

```bash
# 查询 stevenaw0/cann-agent 仓库信息
curl -s -H "Authorization: Bearer $GITCODE_API_TOKEN" \
  "https://api.gitcode.com/api/v5/repos/stevenaw0/cann-agent"
```

### 响应字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | integer | 仓库 ID |
| `name` | string | 仓库名称 |
| `full_name` | string | 仓库完整名称（格式：`owner/repo`） |
| `fork` | boolean | 是否为 fork 仓库 |
| `parent` | object | 如果是 fork，包含原仓库信息 |
| `default_branch` | string | 默认分支 |

### Fork 相关字段

当 `fork` 为 `true` 时，`parent` 对象包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | integer | 原仓库 ID |
| `full_name` | string | 原仓库完整名称 |
| `ssh_url_to_repo` | string | 原仓库 SSH 地址 |
| `http_url_to_repo` | string | 原仓库 HTTP 地址 |
| `html_url` | string | 原仓库网页 URL |
| `default_branch` | string | 原仓库默认分支 |

### 示例响应

```json
{
  "id": 9451994,
  "name": "cann-agent",
  "full_name": "stevenaw0/cann-agent",
  "fork": true,
  "default_branch": "main",
  "parent": {
    "id": 9445803,
    "full_name": "cann-agent/skills",
    "ssh_url_to_repo": "git@gitcode.com:cann-agent/skills.git",
    "http_url_to_repo": "https://gitcode.com/cann-agent/skills.git",
    "html_url": "https://gitcode.com/cann-agent/skills",
    "default_branch": "main"
  }
}
```

### 使用脚本自动获取原仓库信息

```bash
#!/bin/bash

# 获取当前远程仓库名称
repo_url=$(git remote get-url origin)
if [ -z "$repo_url" ]; then
  echo "错误：无法获取远程仓库 URL"
  exit 1
fi

# 提取 owner 和 repo
owner=$(echo "$repo_url" | sed 's|.*:\([^/]*\)/\([^/]*\)\.git$|\\1|')
repo=$(echo "$repo_url" | sed 's|.*:\([^/]*\)/\([^/]*\)\.git$|\\2|')

# 查询仓库信息
response=$(curl -s -H "Authorization: Bearer $GITCODE_API_TOKEN" \
  "https://api.gitcode.com/api/v5/repos/${owner}/${repo}")

# 使用 jq 解析结果
if echo "$response" | jq -e '.fork' > /dev/null; then
  is_fork=$(echo "$response" | jq -r '.fork')
  if [ "$is_fork" = "true" ]; then
    parent_name=$(echo "$response" | jq -r '.parent.full_name')
    parent_branch=$(echo "$response" | jq -r '.parent.default_branch')
    echo "这是一个 fork 仓库"
    echo "原仓库: $parent_name"
    echo "原仓库默认分支: $parent_branch"
    echo "原仓库 URL: $(echo "$response" | jq -r '.parent.html_url')"
  else
    echo "这不是一个 fork 仓库"
    echo "仓库: $(echo "$response" | jq -r '.full_name')"
  fi
else
  echo "API 调用失败"
fi
```

---

## 获取 PR 信息

```bash
curl -s -H "PRIVATE-TOKEN: $GITCODE_API_TOKEN" \
  "https://api.gitcode.com/api/v4/projects/${encoded_repo}/merge_requests/<PR_NUMBER>"
```

---

## 获取 PR 讨论（包含行内评论）

```bash
curl -s -H "PRIVATE-TOKEN: $GITCODE_API_TOKEN" \
  "https://api.gitcode.com/api/v4/projects/${encoded_repo}/merge_requests/<PR_NUMBER>/discussions"
```

**返回字段说明**：

| 字段 | 说明 |
|------|------|
| `notes[].type` | `DiffNote` = 行内评论，`DiscussionNote` = 普通讨论 |
| `notes[].body` | 评论内容 |
| `notes[].diff_file` | 被评论的文件路径 |
| `notes[].new_line` | 代码行号（结束行） |
| `notes[].position.start_new_line` | 起始行号（多行选择） |
| `notes[].resolved` | 是否已解决 |

---

## 提交行内评论（支持多行选择）

### 创建新的 Discussion（推荐）

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
    "severity": "suggestion"
  }'
```

### 参数说明

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

### 多行选择说明

- **单行评论**：`start_new_line` 和 `new_line` 设置为相同值
- **多行评论**：`start_new_line` = 起始行，`new_line` = 结束行

### 回复已有 Discussion

```bash
curl -s -X POST \
  -H "PRIVATE-TOKEN: $GITCODE_API_TOKEN" \
  -H "Content-Type: application/json" \
  "https://api.gitcode.com/api/v4/projects/${encoded_repo}/merge_requests/<PR_NUMBER>/discussions/<DISCUSSION_ID>/notes" \
  -d '{
    "body": "回复内容"
  }'
```

---

## 提交普通评论

```bash
curl -s -X POST \
  -H "PRIVATE-TOKEN: $GITCODE_API_TOKEN" \
  -H "Content-Type: application/json" \
  "https://api.gitcode.com/api/v4/projects/${encoded_repo}/merge_requests/<PR_NUMBER>/notes" \
  -d '{
    "body": "评论内容"
  }'
```

---

## 删除 PR 评论

### API 端点

```
DELETE https://api.gitcode.com/api/v5/repos/:owner/:repo/pulls/comments/:id
```

### 路径参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `owner` | string | ✅ | 仓库所属空间地址（组织或个人的地址 path） |
| `repo` | string | ✅ | 仓库路径（path） |
| `id` | integer | ✅ | 评论的 ID |

### 查询参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `access_token` | string | ✅ | 用户授权码 |

### 请求示例

```bash
curl -s -X DELETE \
  "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/comments/<COMMENT_ID>?access_token=$GITCODE_API_TOKEN"
```

### 响应

| 状态码 | 说明 |
|--------|------|
| `200` | 成功删除，返回空对象 `{}` |
| `401` | 未授权 |
| `403` | 无权限删除此评论 |
| `404` | 评论不存在 |

### 获取评论 ID

```bash
# 获取 PR 的所有讨论（包含行内评论）
curl -s -H "PRIVATE-TOKEN: $GITCODE_API_TOKEN" \
  "https://api.gitcode.com/api/v4/projects/${encoded_repo}/merge_requests/<PR_NUMBER>/discussions" | \
  jq '.[].notes[] | {id: .id, type: .type, author: .author.username, body: .body}'

# 获取 PR 的所有普通评论
curl -s -H "PRIVATE-TOKEN: $GITCODE_API_TOKEN" \
  "https://api.gitcode.com/api/v4/projects/${encoded_repo}/merge_requests/<PR_NUMBER>/notes" | \
  jq '.[] | {id: .id, author: .author.username, body: .body}'
```

### 权限说明

- 只能删除自己创建的评论
- 仓库管理员可以删除任何评论
- 删除操作不可恢复

---

## 获取 PR 文件变更

```bash
# 基本查询
curl -s -H "PRIVATE-TOKEN: $GITCODE_API_TOKEN" \
  "https://api.gitcode.com/api/v4/projects/${encoded_repo}/merge_requests/<PR_NUMBER>/changes"

# 每页 100 条
curl -s -H "PRIVATE-TOKEN: $GITCODE_API_TOKEN" \
  "https://api.gitcode.com/api/v4/projects/${encoded_repo}/merge_requests/<PR_NUMBER>/changes?per_page=100"
```

---

## 获取文件原始内容

```bash
curl -s "https://raw.gitcode.com/${owner}/${repo}/raw/{commit_sha}/文件路径"
```

---

## 注意事项

1. **Token 配置**：使用环境变量 `GITCODE_API_TOKEN`
2. **项目路径**：动态从当前仓库获取 `owner` 和 `repo`，并 URL 编码为 `${encoded_repo}`
3. **认证头**：GitLab API v4 使用 `PRIVATE-TOKEN:`，不是 `Authorization: Bearer`
4. **多行选择**：必须同时设置 `start_new_line` 和 `new_line`
