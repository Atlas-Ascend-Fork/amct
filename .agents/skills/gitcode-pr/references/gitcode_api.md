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
| **GitCode API v5**（推荐，读写均可） | `?access_token=<token>` 或 `Authorization: Bearer <token>` | `/api/v5/repos/<owner>/<repo>/...` |
| GitLab 风格 v4（**仅读，写已禁用**） | `PRIVATE-TOKEN: <token>` | `/api/v4/projects/<encoded_path>/...` |

**项目路径编码**：`owner/repo` → `owner%2Frepo`（使用 `jq -sRr @uri` 或 `printf '%s' "owner/repo" | jq -sRr @uri`）

> ⚠️ **v4 写接口已全部禁用（HTTP 403）**
>
> 所有 `POST`/`PUT`/`DELETE` 的 v4 调用都会返回：
>
> ```json
> {"message": "当前 /api/v4 接口已禁用，请使用官方文档中的 /api/v5 接口"}
> ```
>
> 因此**任何写操作（发评论、发行内评论、回复讨论）必须走 v5**；
> v4 仅可用于部分**读**接口（`merge_requests/<n>`、`.../changes`、`.../discussions`）。
> 此外 `GET /api/v4/projects/<encoded_repo>/merge_requests/<n>/notes` 未发布，返回
> `404 {"error_code":"APIG.0101"}`，读取评论请统一用 v5。

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

> v4 读接口仍可用（200）。若需与写操作保持一致的认证方式，也可用 v5：
> `GET /api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>?access_token=$GITCODE_API_TOKEN`

---

## 获取 PR 讨论（包含行内评论）

### 推荐：v5 评论列表

```bash
# 直接返回数组，无需再解析包装对象
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/comments?access_token=$GITCODE_API_TOKEN" | \
  jq '.[] | {id, discussion_id, body, author: .user.login, comment_type}'
```

### 备选：v4 discussions（只读）

```bash
curl -s -H "PRIVATE-TOKEN: $GITCODE_API_TOKEN" \
  "https://api.gitcode.com/api/v4/projects/${encoded_repo}/merge_requests/<PR_NUMBER>/discussions"
```

> ⚠️ **返回结构不是数组，而是包装对象**，且 `data[]` 中混有系统事件（其 `notes` 为 `null`）：
>
> ```json
> {
>   "end_id": null,
>   "end_system_id": null,
>   "data": [
>     { "id": 1, "action": "create merge request...", "notes": null },
>     { "id": 2, "discussion_id": "b4ecb001...", "notes": [ { "id": 171409022, "body": "..." } ] }
>   ]
> }
> ```
>
> 因此 **`jq '.[].notes[]'` 会报错** `Cannot index number with string "notes"`
> （`end_id`/`end_system_id` 是数字/`null`）。正确写法必须先取 `.data[]` 并过滤掉系统事件：
>
> ```bash
> curl -s -H "PRIVATE-TOKEN: $GITCODE_API_TOKEN" \
>   "https://api.gitcode.com/api/v4/projects/${encoded_repo}/merge_requests/<PR_NUMBER>/discussions" | \
>   jq '.data[] | select(.notes != null) | .notes[] | {id, type, body, author: .author.username}'
> ```
>
> 另注：`GET .../discussions/<DISCUSSION_ID>/notes`（单个讨论的 notes）**未发布，返回 404**，
> 请直接使用上面的列表接口。

**返回字段说明**：

| 字段 | 说明 |
|------|------|
| `data[].notes[].type` | `DiffNote` = 行内评论，`DiscussionNote` = 普通讨论 |
| `data[].notes[].body` | 评论内容 |
| `data[].notes[].diff_file` | 被评论的文件路径 |
| `data[].notes[].new_line` | 代码行号（结束行） |
| `data[].notes[].position.start_new_line` | 起始行号（多行选择） |
| `data[].notes[].resolved` | 是否已解决 |
| `data[].action` | 系统事件描述（此时 `notes` 为 `null`，需过滤） |

v5 评论列表对应字段见 `../SKILL.md`「解析评论数据」章节（`comment_type`、`discussion_id`、
`user.login`；行内位置信息需再查单条接口 `/pulls/comments/:id`）。

---

## 提交行内评论（支持多行选择）

> ⚠️ **写操作必须使用 v5**。v4 的 `POST .../merge_requests/<n>/discussions`
> 已禁用，返回 `403 {"message":"当前 /api/v4 接口已禁用，请使用官方文档中的 /api/v5 接口"}`。

### 使用 v5 API（推荐）

```bash
curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/comments?access_token=$GITCODE_API_TOKEN" \
  -d '{
    "body": "评论内容",
    "path": "文件相对路径",
    "position": <结束行号>,
    "start_position": <起始行号>
  }'
```

### 参数说明

| 参数 | 说明 | 必需 |
|------|------|------|
| `body` | 评论内容 | ✅ |
| `path` | 文件相对路径 | ✅ |
| `position` | 结束行号 | ✅ |
| `start_position` | 起始行号（多行选择，未文档化但实测有效） | 多行时 |
| `access_token`（query） | 用户授权码 | ✅ |

### 多行选择说明

- **单行评论**：只传 `position`（或 `position` 与 `start_position` 设为相同值）
- **多行评论**：`start_position` = 起始行，`position` = 结束行

> 行内评论所需的 `base_sha`/`head_sha` 不需要手工传入，v5 会自行解析。
> 如确需 SHA，可从 `GET /api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/files.json`
> 的 `diff_refs` 字段获取。

**返回值注意**：POST 立即返回的 `id` 是**哈希字符串**（即 `discussion_id`），
不是数字 ID。需要数字 ID（如后续删除）时，请回查评论列表精确定位，
详见 `../SKILL.md`「创建评论的返回值问题」。

### 回复已有 Discussion

```bash
curl -s -X POST \
  -H "Content-Type: application/json" \
  "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/discussions/<DISCUSSION_ID>/comments?access_token=$GITCODE_API_TOKEN" \
  -d '{
    "body": "回复内容"
  }'
```

> `<DISCUSSION_ID>` 取自评论列表的 `discussion_id` 字段（哈希串）。
> 回复**不会**出现在 `/pulls/<PR_NUMBER>/comments` 列表中，需用单条接口
> `/pulls/comments/<REPLY_ID>` 验证。
> 旧写法 `POST /api/v4/.../discussions/<id>/notes` 已随 v4 写接口一并禁用（403）。

---

## 提交普通评论

```bash
curl -s -X POST \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/comments?access_token=$GITCODE_API_TOKEN" \
  -d '{
    "body": "评论内容"
  }'
```

> 不传 `path`/`position` 即为普通评论（`comment_type: "pr_comment"`）。
> 旧写法 `POST /api/v4/.../merge_requests/<n>/notes` 已禁用（403）。

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
# 获取 PR 的所有评论（普通评论 + 行内评论），v5 直接返回数组
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/comments?access_token=$GITCODE_API_TOKEN" | \
  jq '.[] | {id, discussion_id, comment_type, author: .user.login, body}'

# 若必须使用 v4 discussions（只读），注意返回是包装对象且含系统事件：
curl -s -H "PRIVATE-TOKEN: $GITCODE_API_TOKEN" \
  "https://api.gitcode.com/api/v4/projects/${encoded_repo}/merge_requests/<PR_NUMBER>/discussions" | \
  jq '.data[] | select(.notes != null) | .notes[] | {id: .id, type: .type, author: .author.username, body: .body}'
```

> ⚠️ 不要用 `jq '.[].notes[]'` 解析 v4 discussions——顶层不是数组，会报
> `Cannot index number with string "notes"`。
> 也不要使用 `GET /api/v4/.../merge_requests/<n>/notes`（未发布，返回 404），
> 普通评论统一从 v5 `/pulls/<PR_NUMBER>/comments` 读取。

### 权限说明

- 只能删除自己创建的评论
- 仓库管理员可以删除任何评论
- 删除操作不可恢复

---

## 获取 PR 文件变更

```bash
# 推荐：v5 files.json（含 diff_refs，可直接拿 base/head/start SHA）
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/files.json?access_token=$GITCODE_API_TOKEN" | \
  jq '{count, diff_refs, paths: [.diffs[].statistic.new_path]}'

# 备选：v4 changes（只读，仍可用）
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
2. **项目路径**：动态从当前仓库获取 `owner` 和 `repo`；v5 直接用 `${owner}/${repo}`，
   v4 需 URL 编码为 `${encoded_repo}`
3. **认证方式**：v5 用 `?access_token=<token>`（或 `Authorization: Bearer <token>`）；
   v4 只读接口才用 `PRIVATE-TOKEN:` 头
4. **读写分工**：**所有写操作（发评论 / 行内评论 / 回复 / 创建 PR）一律走 v5**；
   v4 写接口已禁用并返回 403
5. **多行选择**：v5 用 `start_position`（起始行）+ `position`（结束行）
6. **v4 discussions 解析**：返回 `{end_id, end_system_id, data:[...]}`，
   必须 `.data[] | select(.notes != null) | .notes[]`
