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

**基址（host）**：文档统一写 `https://api.gitcode.com`。实测 `https://gitcode.com/api/v5/...`
指向同一套 v5 接口（同一 token 下 GET 返回同一份 JSON；对不存在仓库 POST 也返回同样的
`400 head或base不能为空`），两者都可用，但**不要在同一份脚本里混用**，便于排查与替换。

**认证方式（实测均可用，返回 200）**：`?access_token=` query 参数、`Authorization: Bearer <token>`
请求头、`PRIVATE-TOKEN: <token>` 请求头。token 错误或缺失时返回 `401 {"message":"401 Unauthorized"}`
（不是 403），**且 `curl -s` 的退出码仍是 0**——判成功要看 HTTP 状态码或校验响应里有没有期望的键。

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
| `web_url` | string | 原仓库网页 URL（实测键名；**没有** `html_url` 这个键，取它会静默得到空串） |
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
    "web_url": "https://gitcode.com/cann-agent/skills",
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
# ⚠️ SSH 与 HTTPS 两种 remote 格式都要处理：只用 `.*:` 这套 SSH 模式时，
#    HTTPS 地址（https://gitcode.com/owner/repo.git）匹配不上，sed 会**原样输出整串 URL**，
#    owner 和 repo 都变成那条 URL，后续 API 请求直接 404（实测）
# ⚠️ 尾部 `.git` 是**可选**的：用 `\.git$` 锚定时，`https://gitcode.com/owner/repo`
#    这种 remote 同样匹配不上、同样退化成整串 URL 而 404（实测本仓 origin 就是无 `.git` 形式），
#    故先去掉可选后缀，再按「无后缀」模式提取。
repo_url=${repo_url%.git}
if [[ $repo_url == git@* ]]; then
  owner=$(echo "$repo_url" | sed 's|.*:\([^/]*\)/\([^/]*\)$|\1|')
  repo=$(echo "$repo_url" | sed 's|.*:\([^/]*\)/\([^/]*\)$|\2|')
else
  owner=$(echo "$repo_url" | sed 's|.*gitcode\.com/\([^/]*\)/\([^/]*\)$|\1|')
  repo=$(echo "$repo_url" | sed 's|.*gitcode\.com/\([^/]*\)/\([^/]*\)$|\2|')
fi
# ⚠️ sed 替换里必须写 `\1`/`\2`，写成 `\\1` 得到的是**字面量** `\1`（实测）
if [ -z "$owner" ] || [ "$owner" = "$repo_url" ]; then
  echo "错误：无法从 $repo_url 解析 owner/repo"
  exit 1
fi

# 查询仓库信息
response=$(curl -s -H "Authorization: Bearer $GITCODE_API_TOKEN" \
  "https://api.gitcode.com/api/v5/repos/${owner}/${repo}")

# 使用 jq 解析结果
# ⚠️ 不能用 `jq -e '.fork'` 判断「API 是否调用成功」：`jq -e` 对 `false`/`null` 同样返回
#    退出码 1，于是**正常的非 fork 仓库**会被误报成「API 调用失败」（实测）。
#    要判响应有效性，得看它是不是带 fork 键的对象。
if [ "$(echo "$response" | jq -r 'if (type == "object") and (has("fork")) then "ok" else "bad" end' 2>/dev/null)" != "ok" ]; then
  echo "API 调用失败或响应异常：$response"
  exit 1
fi
if [ "$(echo "$response" | jq -r '.fork')" = "true" ]; then
  echo "这是一个 fork 仓库"
  echo "原仓库: $(echo "$response" | jq -r '.parent.full_name')"
  echo "原仓库默认分支: $(echo "$response" | jq -r '.parent.default_branch')"
  # ⚠️ 网页地址的键是 web_url，不是 html_url（后者不存在，取它只会输出 "null"）
  echo "原仓库 URL: $(echo "$response" | jq -r '.parent.web_url')"
else
  echo "这不是一个 fork 仓库"
  echo "仓库: $(echo "$response" | jq -r '.full_name')"
fi
```

---

## 获取 PR 信息

```bash
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>?access_token=$GITCODE_API_TOKEN"
```

---

## 获取 PR 讨论（包含行内评论）

顶层直接返回数组，行内评论与普通评论混在同一列表，用 `comment_type` 区分。

> ⚠️ **顶层数组不含回复**：回复嵌在父评论的 `reply[]` 里（实测 cann/amct PR!455：
> 回复的 ID 在顶层数组一个都查不到）。
> `reply[]` 里的对象**只有** `id`/`body`/`user`/`created_at`/`updated_at`，
> **没有** `discussion_id`/`comment_type`/`resolved`/`path`/`diff_position`，需要时从父评论带下来。
>
> 带下来的 `comment_type` 是**列表接口那一套取值**（`diff_comment`/`pr_comment`），
> 与「跟随父讨论种类」是两回事：同一条回复用单条接口查会得到**另一套**取值
> （实测回复 `194233994`：列表父评论 `diff_comment`，单条接口 `DiffNote`），
> 两套不要混用，对照见下「两个接口的 `comment_type` 取值不是同一套」。
> `resolved` 是讨论级属性、以父评论为准，**不要给回复伪造取值**。
> 回复的完整字段仍需用 `note_id` 查单条接口 `GET /pulls/comments/<id>`。
>
> **展开后的行形状与父行不同**：回复行没有 `resolved`/`lines`（`diff_position`）键，
> 按这两个字段过滤只会命中父行——这正是想要的（讨论级属性只在父行）；
> 需要区分时用 `reply_to`（`null` = 父评论，非 null = 回复，值为父评论 ID）。
>
> **分页 `length` 与响应头 `total_count` 数的都是父评论条数**，回复不计入
> `per_page`/`total_count`（实测 cann/amct PR!455：响应头 `total_count` 的值等于
> 顶层数组长度，`reply[]` 内的回复不计入）。因此校验「是否取全」要拿**父评论**条数去比，
> **不要拿展开后的输出条数去比**（展开后必然偏大，会被误判成「多取了/取重了」）。

```bash
# 单页（评论数 ≤ 100 时够用）
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/comments?access_token=$GITCODE_API_TOKEN&per_page=100" | \
  jq '.[] | {id, discussion_id, comment_type, author: .user.login, resolved, lines: .diff_position, body} as $c
      | $c, (.reply[]? | {id, discussion_id: $c.discussion_id, comment_type: $c.comment_type, author: .user.login, body, reply_to: $c.id})'

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
  # 父评论原样输出，再把 reply[] 展开成同形对象（回复没有 discussion_id/comment_type，从父评论带下来）
  echo "$resp" | jq -c '.[] | {id, discussion_id, comment_type, author: .user.login, resolved, body} as $c | $c, (.reply[]? | {id, discussion_id: $c.discussion_id, comment_type: $c.comment_type, author: .user.login, body, reply_to: $c.id})'
  count=$(echo "$resp" | jq 'length')
  [ "$count" -lt 100 ] && break   # count 数的是**父评论**条数（回复不计入分页），不足 100 即最后一页
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
| `page` / `per_page` | 分页；`per_page` 默认 **20**（实测），最大 **100**（官方文档值，未实测——抽样中没有父评论 > 100 的 PR，在 38 条的 PR 上传 `per_page=150/200` 都返回全部 38 条，既不能证明也不能推翻该上限；按 100 翻页在上限更大时只是多翻几页、结果仍正确）。评论较多时必须显式传；评论数 > 100 时还必须按 `page` 递增翻页直到本页返回条数 < 100，否则第 2 页之后的评论会被漏读 |
| `direction` | `asc` / `desc` |
| `comment_type` | 筛选评论类型：`diff_comment`（行内）/ `pr_comment`（普通） |

**返回字段说明**：

| 字段 | 说明 |
|------|------|
| `id` | 评论 ID（整数），删除评论用它 |
| `discussion_id` | 讨论 ID（哈希串），回复该讨论用它 |
| `comment_type` | `diff_comment` = 行内评论，`pr_comment` = 普通评论 |
| `diff_position.start_new_line` / `end_new_line` | 行内评论的起止行号 |
| `diff_position.position_type` | `text` = 代码行评论（实测）；`binary` = 文件级评论（官方文档值，**未实测**） |
| `resolved` / `resolved_by` | 检视意见是否已解决、解决人 |
| `reply[]` | 该讨论下的回复，嵌套在父评论里，不单独出现在顶层 |
| `user.login` | 评论者 |

> ⚠️ 列表接口**不返回被评论的文件路径**——注意是**没有 `path` / `position` 这两个键**
> （实测 cann/amct PR!455 全部父评论：`has("path")` 与 `has("position")` 均为 `false`，
> 不是「键存在但值为 `null`」）。用 `jq '.path'` 取值会得到 `null`，但**不要按这两个字段过滤**，
> 永远命中不了。需要路径时改查单条
> `GET /api/v5/repos/${owner}/${repo}/pulls/comments/<COMMENT_ID>`，其 `position` 字段含
> `old_path` / `new_path` / `new_line` / `base_sha` / `head_sha`。
>
> ⚠️ **行号不需要查单条接口**：列表接口的 `diff_position.start_new_line` / `end_new_line`
> 就是起止行（实测 PR!455 全部 5 条 `diff_comment` 都带此键，单行评论起止同值）。
> 但**行号数值易变**：按 PR head **当前**指向的 commit 重算，PR 每推新 commit 就可能整体位移
> （实测同一条 `194143085`：head 为 `ae35b167…` 时是 `60`，head 推到 `a4f4aec…` 后变成 `81`），
> 故此处只固化「哪些键存在」这类结构事实，不要把具体行号当稳定结论引用。
> 另，**两个接口的位置键名不是同一套**，别串用（实测 cann/amct PR!447 多行行内评论 `193754581`：
> 列表 `diff_position = {start_new_line:29, end_new_line:31}`，
> 单条 `position = {new_line:31, start_new_line:29, new_path:"…"}`）：
>
> | 含义 | 列表接口 `diff_position` | 单条接口 `position` |
> |------|--------------------------|---------------------|
> | 起始行 | `start_new_line` | `start_new_line`（**单行评论时该键不存在**） |
> | 结束行 | `end_new_line` | `new_line`（**不叫 `end_new_line`**） |
> | 文件路径 | **无** | `new_path` / `old_path` |
> | base/head SHA | **无** | `base_sha` / `start_sha` / `head_sha` |

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
| `position_type` | `text`（默认，代码行评论）/ `binary`（文件级评论，此时 `position` 不生效）——**`binary` 取自官方文档、未实测** | 否 |
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
# ⚠️ curl -s 不加 --fail 时，400/401/403/404/405 的退出码仍是 0，只看 $? 会把删失败当成删成功，
#    必须用 -w '%{http_code}' 校验（成功为 200）
curl -s -o /dev/null -w '%{http_code}\n' -X DELETE \
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
| `405` | **该评论下还有回复**：`{"error_code":405,"error_message":"Can not delete a discussion that has replies."}`，必须先把 `reply[]` 里的回复全部删掉，再删父评论 |

> ⚠️ **删除有顺序要求**：父评论带着回复时删不掉（`405`）。批量清理时必须**先删回复、再删父评论**。
> 且当筛选条件只命中父评论、漏掉它的某条回复时，这个 `405` 是**必然**发生而非偶发——
> 应在删除前就用 `reply[]` 预检出来并直接失败（见下文批量清理脚本第 2 步），
> 否则会把 N 次注定失败的 `DELETE` 全发一遍，只留下 N 行看不出根因的 ERROR。
> 另：`curl -s` 不加 `-f`/`--fail` 时，`405` 这类 HTTP 错误的**退出码仍是 0**，
> 只看 `$?` 会把「删失败」当成「删成功」——必须用 `-w '%{http_code}'` 显式校验状态码。
>
> **证据边界（别把两类验证混为一谈）**：
> - **真实 API 实测**（cann/amct PR!455 自测探针）：父 `194259915` 带存活回复 `194259921` 时
>   删父 → `405 Can not delete a discussion that has replies.`；先删回复（`200`）再删父 → `200`。
> - **「N 个父评论全部 405、脚本 exit 1、清理一条没完成」的批量场景是本地假 API 模拟得到的**
>   （100 父 + 100 回复、标记只覆盖父评论），不是对真实 API 的批量实测——
>   真实 API 上不会为验证这一点去批量制造/删除上百条评论。
>   由单点实测（父带存活回复删不掉）+ 筛选条件没覆盖回复，可**推得**批量场景必然逐条 `405`。

### 获取评论 ID

```bash
# 顶层直接是数组；per_page 默认 20（实测）、「最大 100」为官方文档值未实测（按 100 翻页在
# 上限更大时也只是多翻几页、结果仍正确），评论多时必须显式传 100 并翻页，否则 ID 会漏。
# ⚠️ 顶层**不含回复**：自己以「回复」形式发的评论嵌在父评论 reply[] 里，不展开就永远收不到
#    它的 id，DELETE 时会漏删（实测 cann/amct PR!455：回复的 ID 在顶层一个都查不到）。
page=1
while :; do
  resp=$(curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/comments?access_token=$GITCODE_API_TOKEN&per_page=100&page=${page}")
  # 空响应会让 `[ "" -lt 100 ]` 报错不 break（死循环）；错误对象的 `jq 'length'`
  # 是键数，会被当成「本页 N 条」静默提前终止、漏掉后续 ID。两者都必须显式报错。
  if [ "$(echo "$resp" | jq -r 'type' 2>/dev/null)" != "array" ]; then
    echo "ERROR: 第 ${page} 页评论请求失败或响应非数组，已取 ID 不完整：$resp" >&2
    exit 1   # 取不全即失败：ID 列表不完整时继续删会漏删，且 exit 0 会被读成「已清理干净」
  fi
  # 父评论 + 其 reply[] 一并输出（回复没有 discussion_id/comment_type，从父评论带下来）
  echo "$resp" | jq -c '.[] | {id, discussion_id, comment_type, author: .user.login, body} as $c | $c, (.reply[]? | {id, discussion_id: $c.discussion_id, comment_type: $c.comment_type, author: .user.login, body, reply_to: $c.id})'
  count=$(echo "$resp" | jq 'length')
  [ "$count" -lt 100 ] && break   # count 数的是**父评论**条数（回复不计入分页）
  page=$((page + 1))
done
```

> 这里的整数 `id` 就是 `DELETE` 要用的评论 ID（发表评论时响应里的 `note_id`）。
> 该循环失败分支同样必须 `exit 1`（理由见上「失败必须显式」）：ID 没取全就继续删会**漏删**，
> 而 `break` + `exit 0` 会让调用方把「取失败了」读成「已清理干净」。
> 同理，**不展开 `reply[]` 也会漏删**：以回复形式留下的评论不在顶层，
> 只投影顶层字段时它的 `id` 永远收不到——这类漏删还是静默的（脚本 `exit 0`）。

⚠️ **展开后还必须调整删除顺序**：上面的投影是「父在前、回复在后」，照这个顺序直接删，
父评论会因「还有回复」返回 `405 Can not delete a discussion that has replies.` 而留下，
回复反倒被删掉——清理只完成一半，且 `curl -s` 的退出码仍是 `0`，看起来像成功。
实测（cann/amct PR!455 自测探针）：父 `194259915` 带回复 `194259921` 时删父 → `405`；
先删回复（`200`）再删父 → `200`。
因此按 `reply_to` 排序（**非 null 的回复先删**），并逐个校验 HTTP 状态码：

⚠️ **另外两个同样会「少删却 exit 0」的坑**：

1. **不能边翻页边删**。删除会让列表变短、后续条目前移，此时再取「下一页」拿到的已不是
   原来那批数据，中间被跳过的条目**永远删不到**。必须**先把所有页的 ID 取全（只读不删），
   再统一删**；若坚持边翻边删，则必须用 `total_page` **倒序**翻页（从最后一页往前删，
   前面的偏移才不受影响）。
2. **不能在循环里覆盖同一个变量、循环外再用它删**。`resp=$(...)` 每翻一页都会覆盖上一页，
   循环结束后只剩**最后一页**——评论超过 100 条时前面几页的 ID 全丢，且脚本仍 `exit 0`。
   必须在循环内把 ID **追加**到累积容器（临时文件/变量），循环结束后统一删。

此外 `select(.body | startswith(...))` 里的 `.body` **可能为 `null`**（行内 diff 评论、
系统生成的评论等），对 `null` 调 `startswith` 会让 jq **中途报错退出**：崩之前已输出的 ID
算数、其后的 ID 全部丢失，而外层 `for` 不检查 jq 退出码，于是拿着残缺名单照常跑完并 `exit 0`。
必须写成 `(.body // "")`，并对 jq 的退出码显式判断。

```bash
# ---- 第 1 步：翻页把所有待删 ID 取全（只读不删），追加进临时文件 ----
ids=$(mktemp)
blocked=$(mktemp)
trap 'rm -f "$ids" "$blocked"' EXIT
page=1
while :; do
  resp=$(curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/comments?access_token=$GITCODE_API_TOKEN&per_page=100&page=${page}")
  if [ "$(echo "$resp" | jq -r 'type' 2>/dev/null)" != "array" ]; then
    echo "ERROR: 第 ${page} 页评论请求失败或响应非数组，ID 未取全：$resp" >&2
    exit 1
  fi
  # reply_to 非 null（回复）标记为 1，父评论为 0；(.body // "") 兜住 null，否则 jq 会中途死掉丢 ID
  echo "$resp" | jq -r '
      .[] | . as $c
      | ({id, body, reply_to: null}, ((.reply // [])[] | {id, body, reply_to: $c.id}))
      | select((.body // "") | startswith("<唯一标记>"))
      | "\(if .reply_to then 1 else 0 end)\t\(.id)"' >> "$ids" \
    || { echo "ERROR: 第 ${page} 页 jq 解析失败，ID 未取全" >&2; exit 1; }
  # 405 预检：被选中的父评论若还有**未被选中的存活回复**，删它必然 405，翻页阶段就能算出来
  echo "$resp" | jq -r '
      .[] | select((.body // "") | startswith("<唯一标记>")) as $p
      | ($p.reply // [])[]
      | select(((.body // "") | startswith("<唯一标记>")) | not)
      | "父 \($p.id) 的回复 \(.id) 未被筛选条件命中"' >> "$blocked" \
    || { echo "ERROR: 第 ${page} 页 405 预检 jq 解析失败" >&2; exit 1; }
  count=$(echo "$resp" | jq 'length')
  [ "$count" -lt 100 ] && break
  page=$((page + 1))
done

# ---- 第 2 步：删除前的 405 预检，有阻塞项就直接失败、一次 DELETE 都不发 ----
if [ -s "$blocked" ]; then
  echo "ERROR: 以下父评论带「未被筛选条件命中的存活回复」，直接删必然 405：" >&2
  sort -u "$blocked" >&2
  echo "       处置：改筛选条件让它同时命中父评论与其回复，或先单独删掉上面列出的回复 ID。" >&2
  exit 1
fi

# ---- 第 3 步：全部取完后再统一删除（sort -r 使「回复在前、父评论在后」）----
fail=0
for id in $(sort -r "$ids" | cut -f2); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -X DELETE \
    "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/comments/${id}?access_token=$GITCODE_API_TOKEN")
  [ "$code" = "200" ] || { echo "ERROR: 删除评论 ${id} 失败（http=${code}），清理不完整" >&2; fail=1; }
done
[ "$fail" -eq 0 ] || exit 1
```

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
| `filename` | 文件路径 |
| `sha` | **提交 sha**（实测等于 `diff_refs.head_sha`），**不是 blob sha**；blob sha 在 `blob_id` |
| `blob_id` | blob sha（实测等于 `git rev-parse <commit>:<path>`），可拼成稳定的 `…/blobs/<blob_id>/<path>` 取内容地址 |
| `blob_url` | 网页版文件页（`gitcode.com/<owner>/<repo>/blob/<commit_sha>/<path>`），不是取原始内容的地址 |
| `raw_url` | 原始内容地址，但形式是 `…/raw/<commit_sha>/<path>`，**实测不稳定**（见下文「获取文件原始内容」），别当唯一手段 |
| `additions` / `deletions` | 增删行数 |
| `patch.diff` | unified diff 文本，行内评论定位行号从这里取 |
| `patch.old_path` / `new_path` | 重命名前后路径 |
| `patch.added_lines` / `removed_lines` | 该文件的增删行数（与顶层 `additions`/`deletions` 实测一致） |
| `patch.new_file` / `renamed_file` / `deleted_file` / `too_large` | 变更类型标记 |
| `source_branch` / `target_branch` | PR 的源 / 目标分支名 |
| `source_project` / `target_project` | 源 / 目标仓库对象（含 `id`/`full_name`/`human_name`/`path`）。**跨 fork PR 取 head 仓库用 `source_project.full_name`**，别去猜 |
| `patch.a_mode` / `b_mode` | 变更前后的文件权限位（如 `100644`） |

> ⚠️ **接口给的增删行数与本地 `git diff --numstat` 可能差 1**（实测 cann/amct PR!455：
> `references/gitcode_api.md` 本地 numstat `358/90`，`files` 与 `files.json` 两个接口都给 `357/89`，
> 其余 3 个文件完全一致；两接口之间从未出现分歧）。差值不影响判断，但**不要拿它做「diff 是否一致」的
> 精确对账**，也不要把这类数字写进 PR 描述（易变）。

---

## 获取文件原始内容

```bash
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/contents/{文件路径}?ref={commit_sha}&access_token=$GITCODE_API_TOKEN" | \
  jq -r '.content' | base64 -d
```

**返回字段**（实测）：`type` / `encoding` / `size` / `name` / `path` / `content`（base64）/
`sha`（**blob sha**）/ `url` / `html_url` / `download_url` / `_links`。

`download_url` 是 `https://raw.gitcode.com/${owner}/${repo}/blobs/{blob_sha}/{文件路径}` 形式，
直接返回明文，实测稳定 `200`，适合管道给 `grep -n` 定位行号。

> ⚠️ **`…/raw/{commit_sha}/{文件路径}` 这个形式实测不稳定，不要当唯一手段**：同一 URL、同一 sha，
> 在某一段时间内对 `.agents/skills/gitcode-pr/` 下的 `commands/review.md`、
> `references/gitcode_api.md`、`evals/evals.json` 连续 3 次返回 `403`（body「暂不支持预览」），
> 同一时刻同 sha 的 `SKILL.md`/`README.md`/`build.sh` 却返回 `200`；几分钟后同一批 URL
> 又连续 5 次全部 `200`；后续复查再次复现，且**命中哪些文件也在变**（同一 sha 下 `SKILL.md` 返回
> `403`、`review.md`/`README.md`/`build.sh` 返回 `200`）。`/pulls/<n>/files` 响应里的 `raw_url`、
> `files.json` 里 `diffs[].head.url` 都是该形式，同样不可靠。

**`ref` 参数取值（实测 cann/amct）**：commit sha（40 位全 sha 或短 sha 如 `65dbe33`）与**分支名都可用**——
`ref=master`、`ref=9.2.0`、`ref=feature/community-tasks`（带斜杠，原样或 `%2F` 编码均可）都返回 `200`，
且分支名取到的内容与本地 `git show cann/<branch>:<path>` 的 md5 一致；不传 `ref` 时用默认分支。
跨 fork PR 的 head commit 在**目标仓库**路径下也取得到（实测 PR!447 head `59bd5789…`，
用 `cann/amct` 与源仓库 `m0_45199755/amct` 两条路径都 `200`、blob sha 相同）。

> ⚠️ 审查 PR 时**仍应传 head commit sha**，不要用分支名：分支名指向分支**当前**位置，
> PR 提交后分支若又推了新 commit，取到的就不是被审查的那一版，`grep -n` 得到的行号会与 diff 对不上。
>
> ⚠️ **`404` 区分不了「ref 不存在」和「文件在该 ref 上不存在」**：两种情况响应体完全相同，
> 都是 `{"error_code":404,"error_code_name":"NOT_FOUND","error_message":"项目或文件不存在","trace_id":"…"}`
> （实测三种组合返回一模一样：`ref=nonexist-branch-xyz` + 存在的 `README.md`；
> `ref=feature/community-tasks` + 该分支上没有的 `.agents/…/SKILL.md`；`ref=master` + 不存在的文件）。
> 遇到 `404` 要两种可能都排查。**没有** `{"message":"404 Commit Not Found"}` 这种响应体。
>
> ⚠️ `curl -s` 对 `403`/`404` 的退出码仍是 `0`，取完必须校验状态码 / 内容非空，
> 否则会把错误页当成文件内容、`grep -n` 静默输出空结果。

---

## 注意事项

1. **Token 配置**：使用环境变量 `GITCODE_API_TOKEN`
2. **项目路径**：动态从当前仓库获取 `owner` 和 `repo`，直接拼进
   `/api/v5/repos/${owner}/${repo}/...`，无需 URL 编码
3. **认证方式**：`?access_token=<token>`（或 `Authorization: Bearer <token>`）
4. **API 版本**：只用 v5；`/api/v4` 已停用（写接口 403、部分读接口 404），不要使用
5. **多行选择**：写入用 `start_position`（起始行）+ `position`（结束行），
   读取看 `diff_position.start_new_line` / `end_new_line`
6. **分页**：`per_page` 默认 20（实测）；「最大 100」取自官方文档、未实测（抽样中没有
   父评论 > 100 的 PR，在 38 条的 PR 上传 `per_page=150/200` 都返回全部 38 条）。
   按 100 翻页在上限更大时也只是多翻几页、结果仍正确，故评论较多时必须显式传 `per_page=100`；
   评论数 > 100 时还必须从 `page=1` 起递增翻页，直到本页返回条数 < 100 为止，
   否则会漏读第 2 页之后的评论
7. **两种 ID 不要混用**：`discussion_id`（哈希串，用于回复）vs 评论 `id`（整数，用于删除）
