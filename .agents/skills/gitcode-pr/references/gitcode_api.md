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

官方文档（<https://docs.gitcode.com/docs/apis/>）明确「当前可用的版本为 `/api/v5`」，
读写一律使用 v5：

| API 版本 | 认证方式 | 端点格式 |
|---------|---------|---------|
| **GitCode API v5** | `?access_token=<token>`（或 `Authorization: Bearer <token>`、`PRIVATE-TOKEN: <token>`） | `/api/v5/repos/<owner>/<repo>/...` |

> ⚠️ **不要使用 `/api/v4`**：v4 未收录于官方文档，写接口已被禁用，调用返回
> `403 {"message":"当前 /api/v4 接口已禁用，请使用官方文档中的 /api/v5 接口"}`；
> 部分 v4 读接口从未发布，返回 `404 {"error_code":"APIG.0101"}`。

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
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>?access_token=$GITCODE_API_TOKEN"
```

---

## 获取 PR 讨论（包含行内评论）

顶层直接返回数组，行内评论与普通评论混在同一列表，用 `comment_type` 区分：

```bash
# 单页（评论数 ≤ 100 时够用）
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/comments?access_token=$GITCODE_API_TOKEN&per_page=100" | \
  jq '.[] | {id, discussion_id, comment_type, author: .user.login, resolved, lines: .diff_position, body}'

# 翻页取完所有评论（评论数可能 > 100，必须循环，否则会漏读）
page=1
while :; do
  resp=$(curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/comments?access_token=$GITCODE_API_TOKEN&per_page=100&page=${page}")
  # 响应必须是数组：空响应会让 `[ "" -lt 100 ]` 报错不 break（死循环），
  # 错误对象（{"error_code":...}）的 `jq 'length'` 是键数，会被当成「本页 N 条」
  # 静默提前终止。两者都必须显式报错。
  if [ "$(echo "$resp" | jq -r 'type' 2>/dev/null)" != "array" ]; then
    echo "ERROR: 第 ${page} 页评论请求失败或响应非数组，已取结果不完整：$resp" >&2
    exit 1   # 取不全即失败：空列表 + exit 0 会被调用方误读成「取全了且没有评论」
  fi
  echo "$resp" | jq -c '.[] | {id, discussion_id, comment_type, author: .user.login, resolved, body}'
  count=$(echo "$resp" | jq 'length')
  [ "$count" -lt 100 ] && break   # 本页不足 100 条即为最后一页
  page=$((page + 1))
done
```

> **失败必须显式**：无效 token（`{"message":"401 Unauthorized"}`）、不可解析域名（空响应）
> 都会走进上面的非数组分支。若那里用 `break`，脚本仍以 `exit 0` 结束、stdout 是空列表，
> 调用方无法区分「取全了但没有评论」与「取失败了」，据空列表判断「尚未评论过」就会重复发评论。
> 因此失败分支用 `exit 1`；嵌在更大脚本里不能直接退出时，改置 `incomplete=1` 标志再 `break`，
> 由调用方检查该标志。调用方无论如何都要检查退出码。

> **翻页终止的另一种判据**：响应头带 `total_count`（评论总数）与 `total_page`（总页数），
> 实测 `per_page=5&page=1` → `total_count: 10`、`total_page: 2`；`per_page=100` → `total_page: 1`
> （数值随评论增长变化，此处为测量当时的值）。
> 该接口**不返回** `Link` header，不要按 GitHub 的 `rel="next"` 判断。

**查询参数**：

| 参数 | 说明 |
|------|------|
| `page` / `per_page` | 分页；`per_page` 默认 **20**、最大 **100**，评论较多时必须显式传；评论数 > 100 时还必须按 `page` 递增翻页直到本页返回条数 < 100，否则第 2 页之后的评论会被漏读 |
| `direction` | `asc` / `desc` |
| `comment_type` | 筛选评论类型：`diff_comment`（行内）/ `pr_comment`（普通） |

**返回字段说明**：

| 字段 | 说明 |
|------|------|
| `id` | 评论 ID（整数），删除评论用它 |
| `discussion_id` | 讨论 ID（哈希串），回复该讨论用它 |
| `comment_type` | `diff_comment` = 行内评论，`pr_comment` = 普通评论 |
| `diff_position.start_new_line` / `end_new_line` | 行内评论的起止行号 |
| `diff_position.position_type` | `text` = 代码行评论，`binary` = 文件级评论 |
| `resolved` / `resolved_by` | 检视意见是否已解决、解决人 |
| `reply[]` | 该讨论下的回复，嵌套在父评论里，不单独出现在顶层 |
| `user.login` | 评论者 |

> ⚠️ 列表接口**不返回被评论的文件路径**（`path` 恒为 `null`）。需要路径时改查单条
> `GET /api/v5/repos/${owner}/${repo}/pulls/comments/<COMMENT_ID>`，其 `position` 字段含
> `old_path` / `new_path` / `new_line` / `base_sha` / `head_sha`。

> ⚠️ **两个接口的 `comment_type` 取值不是同一套**，按上表的口径去解析单条响应会拿到预期外的值：
>
> | 评论种类 | 列表接口 `GET /pulls/<n>/comments` | 单条接口 `GET /pulls/comments/<id>` |
> |----------|-----------------------------------|--------------------------------------|
> | 行内评论 | `diff_comment` | `DiffNote` |
> | 普通评论 | `pr_comment` | `null` |
>
> 实测（cann/amct PR!455）：同一条行内评论 `194208411` 在列表里是 `diff_comment`、
> 在单条接口里是 `DiffNote`；普通评论 `194141205`/`194176110` 在列表里是 `pr_comment`、
> 在单条接口里是 `null`（见不到 `DiscussionNote`）。
> 因此**判断行内/普通只用列表接口的值**；单条接口的 `comment_type` 只能当「非 null 即行内」用。

---

## 提交行内评论（支持多行选择）

```bash
curl -s -X POST \
  -H "Content-Type: application/json" \
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
| `start_position` | 起始行号（多行选择） | 多行时 |
| `position_type` | `text`（默认，代码行评论）/ `binary`（文件级评论，此时 `position` 不生效） | 否 |
| `access_token`（query） | 用户授权码 | ✅ |

**成功响应（HTTP 201）**：

```json
{"id": "<discussion_id 哈希串>", "body": "...", "note_id": <评论 ID 整数>}
```

> ⚠️ 响应里的 `id` 是**讨论 ID**（哈希串），不是评论 ID。删除评论必须用 `note_id`
> （即评论列表里的整数 `id`）；把哈希串传给 `DELETE` 会得到
> `400 {"error_code_name":"UN_KNOW","error_message":"Orchestration error: Invalid path parameter: note_id, number required"}`。

### 多行选择说明

- **单行评论**：只传 `position`（或与 `start_position` 设为相同值）
- **多行评论**：`start_position` = 起始行，`position` = 结束行

> `start_position` 未出现在官方文档的请求体 schema 中（官方只列 `body`/`path`/`position`/`position_type`），
> 但实测有效：传 `start_position=158` + `position=160` 后读回
> `diff_position = {"start_new_line":158,"end_new_line":160,"position_type":"text"}`。

> 行内评论无需手工传 `base_sha`/`head_sha`；如确需 SHA，可从
> `GET /api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/files.json` 的 `diff_refs` 获取。

### 回复已有 Discussion

```bash
curl -s -X POST \
  -H "Content-Type: application/json" \
  "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/discussions/<DISCUSSION_ID>/comments?access_token=$GITCODE_API_TOKEN" \
  -d '{
    "body": "回复内容"
  }'
```

> `<DISCUSSION_ID>` 取自评论列表的 `discussion_id` 字段（哈希串），
> 或发表评论时响应里的 `id`。

**成功响应（HTTP 201）**：`{"id":"<discussion_id>","body":"...","note_id":<评论 ID 整数>}`。
回复不会作为独立条目出现在评论列表顶层，而是嵌进父评论的 `reply[]`。

---

## 提交普通评论

```bash
curl -s -X POST \
  -H "Content-Type: application/json" \
  "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/comments?access_token=$GITCODE_API_TOKEN" \
  -d '{
    "body": "评论内容"
  }'
```

> 不传 `path`/`position` 即为普通评论（`comment_type: "pr_comment"`）。

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
| `200` | 成功删除，返回空 body |
| `400` | `id` 传成了 `discussion_id` 哈希串（要求整数） |
| `401` | 未授权 |
| `403` | 无权限删除此评论 |
| `404` | 评论不存在 |

### 获取评论 ID

```bash
# 顶层直接是数组；per_page 默认 20、最大 100，评论多时必须显式传 100 并翻页，否则 ID 会漏
page=1
while :; do
  resp=$(curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/comments?access_token=$GITCODE_API_TOKEN&per_page=100&page=${page}")
  # 空响应会让 `[ "" -lt 100 ]` 报错不 break（死循环）；错误对象的 `jq 'length'`
  # 是键数，会被当成「本页 N 条」静默提前终止、漏掉后续 ID。两者都必须显式报错。
  if [ "$(echo "$resp" | jq -r 'type' 2>/dev/null)" != "array" ]; then
    echo "ERROR: 第 ${page} 页评论请求失败或响应非数组，已取 ID 不完整：$resp" >&2
    exit 1   # 取不全即失败：ID 列表不完整时继续删会漏删，且 exit 0 会被读成「已清理干净」
  fi
  echo "$resp" | jq -c '.[] | {id, discussion_id, comment_type, author: .user.login, body}'
  count=$(echo "$resp" | jq 'length')
  [ "$count" -lt 100 ] && break
  page=$((page + 1))
done
```

> 这里的整数 `id` 就是 `DELETE` 要用的评论 ID（发表评论时响应里的 `note_id`）。
> 该循环失败分支同样必须 `exit 1`（理由见上「失败必须显式」）：ID 没取全就继续删会**漏删**，
> 而 `break` + `exit 0` 会让调用方把「取失败了」读成「已清理干净」。

### 权限说明

- 只能删除自己创建的评论
- 仓库管理员可以删除任何评论
- 删除操作不可恢复

---

## 获取 PR 文件变更

```bash
# 文件级变更列表，每项含 patch.diff（unified diff，带 @@ hunk header）
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/files?access_token=$GITCODE_API_TOKEN"

# 需要整体统计与 diff_refs（base_sha / start_sha / head_sha）时用 files.json
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/files.json?access_token=$GITCODE_API_TOKEN" | \
  jq '{count, added_lines, remove_lines, diff_refs}'
```

**`files` 返回字段说明**：

| 字段 | 说明 |
|------|------|
| `filename` / `sha` | 文件路径与 blob sha |
| `additions` / `deletions` | 增删行数 |
| `patch.diff` | unified diff 文本，行内评论定位行号从这里取 |
| `patch.old_path` / `new_path` | 重命名前后路径 |
| `patch.new_file` / `renamed_file` / `deleted_file` / `too_large` | 变更类型标记 |

---

## 获取文件原始内容

```bash
curl -s "https://raw.gitcode.com/${owner}/${repo}/raw/{commit_sha}/文件路径"
```

---

## 注意事项

1. **Token 配置**：使用环境变量 `GITCODE_API_TOKEN`
2. **项目路径**：动态从当前仓库获取 `owner` 和 `repo`，直接拼进
   `/api/v5/repos/${owner}/${repo}/...`，无需 URL 编码
3. **认证方式**：`?access_token=<token>`（或 `Authorization: Bearer <token>`）
4. **API 版本**：只用 v5；`/api/v4` 已停用（写接口 403、部分读接口 404），不要使用
5. **多行选择**：写入用 `start_position`（起始行）+ `position`（结束行），
   读取看 `diff_position.start_new_line` / `end_new_line`
6. **分页**：`per_page` 默认 20、最大 100，评论较多时必须显式传 `per_page=100`；
   评论数 > 100 时还必须从 `page=1` 起递增翻页，直到本页返回条数 < 100 为止，
   否则会漏读第 2 页之后的评论
7. **两种 ID 不要混用**：`discussion_id`（哈希串，用于回复）vs 评论 `id`（整数，用于删除）
