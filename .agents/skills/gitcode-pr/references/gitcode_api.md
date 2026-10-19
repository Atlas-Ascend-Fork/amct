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

**认证方式（三种实测均可用）**：`?access_token=` query 参数、`Authorization: Bearer <token>`
请求头、`PRIVATE-TOKEN: <token>` 请求头。token 错误或缺失时返回 `401 {"message":"401 Unauthorized"}`
（不是 403），**且 `curl -s` 的退出码仍是 0**——判成功要看 HTTP 状态码或校验响应里有没有期望的键。

> ⚠️ **`401` 不一定是 token 的问题**：`${owner}`/`${repo}` 未赋值时 URL 会拼成
> `…/repos///pulls/<n>`，若此时 token 也缺失 / 无效，返回的正是上面那个 `401`，与「路径正确、
> token 失效」**响应体一字不差**。实测四格（只读的 `GET /pulls/<n>`）：
>
> | URL | token | 实测响应 |
> |---|---|---|
> | `…/repos///pulls/<n>`（变量为空） | 有效 | `404`，Spring 体：`{"timestamp":…,"status":404,"error":"Not Found","path":"/api/v5/repos///pulls/<n>"}` |
> | `…/repos///pulls/<n>`（变量为空） | 缺失 / 无效 | `401 {"message":"401 Unauthorized"}` |
> | 路径正确 | 有效 / **缺失** | `200`（该接口公开可读，无 token 也返回完整 JSON） |
> | 路径正确 | 无效 | `401 {"message":"401 Unauthorized"}` |
>
> 判别：`404` 的 `path` 字段会**原样回显请求路径**，连续斜杠（`repos///pulls`）一眼可见 →
> 变量没填上；`401` 则两种原因都可能（路径正确时无 token 也能 `200`，故 `401` 必是 token 无效，
> 或变量为空叠加 token 无效）。**先 `echo "$owner/$repo"` 确认变量非空，再怀疑 token。**

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

# ${owner}/${repo} 的解析脚本见 SKILL.md「提取仓库 owner/repo 信息」（含 SSH 与 HTTPS 两种
# remote 格式都要处理、尾部 `.git` 可选、sed 里 `\1` 不能写成 `\\1` 这三条实测坑），此处不重复。
# 下面假定这两个变量已按该节设好。

# 查询仓库信息
response=$(curl -s -H "Authorization: Bearer $GITCODE_API_TOKEN" \
  "https://api.gitcode.com/api/v5/repos/${owner}/${repo}")

# 使用 jq 解析结果
# ⚠️ 不能用 `jq -e '.fork'` 判断「API 是否调用成功」：`jq -e` 对 `false`/`null` 同样返回
#    退出码 1，于是**正常的非 fork 仓库**会被误报成「API 调用失败」（实测）。
#    要判响应有效性，得看它是不是带 fork 键的对象。
if [ "$(echo "$response" | jq -r 'if (type == "object") and (has("fork")) then "ok" else "bad" end' 2>/dev/null)" != "ok" ]; then
  echo "API 调用失败或响应异常：$response" >&2
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

> ⚠️ **顶层数组不含回复**：回复嵌在父评论的 `reply[]` 里（实测：任一带回复的 PR，
> 回复的 ID 在顶层数组一个都查不到）。
> `reply[]` 里的对象**只有** `id`/`body`/`user`/`created_at`/`updated_at`，
> **没有** `discussion_id`/`comment_type`/`resolved`/`path`/`diff_position`，需要时从父评论带下来。
>
> 带下来的 `comment_type` 是**列表接口那一套取值**（`diff_comment`/`pr_comment`），
> 与「跟随父讨论种类」是两回事：同一条回复用单条接口查会得到**另一套**取值
> （实测：行内讨论下的回复，列表父评论 `diff_comment`，单条接口 `DiffNote`），
> 两套不要混用，对照见下「两个接口的 `comment_type` 取值不是同一套」。
> `resolved` 是讨论级属性、以父评论为准，**不要给回复伪造取值**。
> 回复的完整字段仍需用 `note_id` 查单条接口 `GET /pulls/comments/<id>`。
>
> **展开后的行形状与父行不同**：回复行没有 `resolved`/`lines`（`diff_position`）键，
> 按这两个字段过滤只会命中父行——这正是想要的（讨论级属性只在父行）；
> 需要区分时用 `reply_to`（`null` = 父评论，非 null = 回复，值为父评论 ID）。
>
> **分页 `length` 与响应头 `total_count` 数的都是父评论条数**，回复不计入
> `per_page`/`total_count`（实测：响应头 `total_count` 的值等于顶层数组长度，
> `reply[]` 内的回复不计入）。因此校验「是否取全」要拿**父评论**条数去比，
> **不要拿展开后的输出条数去比**（展开后必然偏大，会被误判成「多取了/取重了」）。

⚠️ **取评论列表的通用翻页循环全仓只有一份**，在 `SKILL.md` →「获取 PR 讨论列表（包含行内评论）」，
此处不重抄。评论数可能 > 100，必须从 `page=1` 起翻页取完，终止判据只用响应头 `total_page`
（实测依据见本节下方「翻页终止以响应头 `total_page` 为准」）。本文件下方「批量清理脚本」第 1 步
内联的那份循环是它的**删除场景变体**（多带唯一标记筛选与 `405` 预检），不算第二份通用循环。

本节只给**字段投影的全集版**（含 `resolved` / `diff_position`；`SKILL.md` 那份按用途裁掉了
这两个字段，差异是有意的、不是漏抄）。下面这条**只取第一页**，用于看单页字段形状，
不要拿它当「取全」的实现：

```bash
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/comments?access_token=$GITCODE_API_TOKEN&per_page=100" | \
  jq '.[] | {id, discussion_id, comment_type, author: .user.login, resolved, lines: .diff_position, body} as $c
      | $c, (.reply[]? | {id, discussion_id: $c.discussion_id, comment_type: $c.comment_type, author: .user.login, body, reply_to: $c.id})'
```

> **失败必须显式**：无效 token（`{"message":"401 Unauthorized"}`）与不可解析域名（空响应）
> 都会让响应不是数组，该分支必须 `exit 1`（完整理由与 `incomplete=1` 兜底写法见
> `commands/review.md` →「六条硬规则」第 3 条）。

> **翻页终止以响应头 `total_page` 为准**（上面的循环就是这么写的）。该接口在响应头返回
> `total_count`（父评论总数）与 `total_page`（总页数），`total_page = ceil(total_count / 生效页长)`。
>
> **为什么不能用「本页条数 < 请求的 `per_page`」判终止**：请求值超过上限会被服务端**静默夹回 100**，
> 而 `total_page` 按**生效**页长算、不受夹影响。实测一个父评论远超一页的 PR：
> 请求 `per_page=150` 只返回 100 条，若终止判据写成 `count < 150`，第一页就 `break`，
> 第一页之后的评论**全部静默丢掉**；同一请求的 `total_page` 仍按生效页长 100 计算，按它翻页不会漏。
>
> **读头的三个坑**：
> - `curl -s` 会丢掉响应头，必须用 `-D <文件>`（或 `-sD -` 混流后自行分离）才读得到；
> - 头是 **CRLF** 结尾，取值前必须 `tr -d '\r'`，否则值带回车、`[ "$page" -ge "$total_page" ]`
>   直接报 `integer expression expected`；
> - 头名是全小写自定义头（`total_count`/`total_page`），HTTP 头大小写不敏感但 `awk`/`grep` 敏感，
>   用 `tolower($1)` 或 `grep -i` 匹配。
>
> **`total_page` 是快照**：它是本次请求时刻算出来的。翻页途中有人新发评论会让总页数变大，
> 因此**每页都重读** `total_page`（上面的循环就是这样），用最新值判终止，可自纠正。
>
> **越界 `page` 的行为**（兜底判据的依据）：实测 `page=2`/`99`/`100000` 都返回**空数组 + HTTP 200**，
> 不会回退返回第一页，所以 `total_page` 缺失时退回「本页为空」判据不会死循环；
> `page=0`/负数/非数字返回 **HTTP 400 + 对象体**，由上面的非数组守卫 `exit 1` 覆盖。
>
> 该接口**不返回** `Link` header，不要按 GitHub 的 `rel="next"` 判断翻页。
> 单条资源接口（如 `GET /pulls/<n>`）**没有**这两个分页头，只有列表接口才有。

**查询参数**：

| 参数 | 说明 |
|------|------|
| `page` / `per_page` | 分页；`per_page` 默认 **20**、最大 **100**（超出会被**静默夹回 100**）。评论较多时必须显式传，超过一页时必须从 `page=1` 起递增翻页取完，终止判据用响应头 `total_page`（**不要**用「本页条数 < 请求的 `per_page`」），否则第 2 页之后会被漏读。夹回行为与判据的实测依据见上「翻页终止以响应头 `total_page` 为准」 |
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
> （实测抽样中全部父评论：`has("path")` 与 `has("position")` 均为 `false`，
> 不是「键存在但值为 `null`」）。用 `jq '.path'` 取值会得到 `null`，但**不要按这两个字段过滤**，
> 永远命中不了。需要路径时改查单条
> `GET /api/v5/repos/${owner}/${repo}/pulls/comments/<COMMENT_ID>`，其 `position` 字段含
> `old_path` / `new_path` / `new_line` / `base_sha` / `head_sha`。
>
> ⚠️ **行号不需要查单条接口**：列表接口的 `diff_position.start_new_line` / `end_new_line`
> 就是起止行（实测抽样中全部 `diff_comment` 都带此键，单行评论起止同值）。
> 但**行号数值易变**：接口按 PR head **当前**指向的 commit 重算行号，锚定的仍是同一行源码，
> 行号却随该文件上方的增删位移（实测同一条行内评论：head 推进前 `60`，
> 推进后同一行变 `81`，该行内容一字未变），
> 故此处只固化「哪些键存在」这类结构事实，不要把具体行号当稳定结论引用。
> 另，**两个接口的位置键名不是同一套**，别串用（实测一条多行行内评论：
> 列表 `diff_position = {start_new_line:29, end_new_line:31}`，
> 单条 `position = {new_line:31, start_new_line:29, new_path:"<被评论文件路径>"}`）：
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
> 实测：同一条行内评论在列表里是 `diff_comment`、在单条接口里是 `DiffNote`；
> 同一条普通评论在列表里是 `pr_comment`、在单条接口里是 `null`（见不到 `DiscussionNote`）。
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
>
> **证据边界**：`405` 与「先删回复再删父 → `200`」是真实 API 单点实测；
> 「N 个父评论全部 `405`、脚本 `exit 1`、一条没删掉」的批量场景由本地假 API 模拟
> （100 父 + 100 回复、标记只覆盖父评论）+ 单点实测**推得**，未在真实 API 上批量制造评论验证。

### 获取评论 ID

本节不再单列一份通用翻页循环：清理场景直接用下面「批量清理脚本」的第 1 步
（它已把翻页、`reply[]` 展开、唯一标记筛选、405 预检合在一起）；要另写筛选条件，
就用 `SKILL.md` →「获取 PR 讨论列表（包含行内评论）」那份循环（分页规则与实测依据见上）。

⚠️ **四条会「少删却 `exit 0`」的坑**（批量清理脚本已逐条落地，改写时别丢）：

1. **不能边翻页边删**：删除会让列表前移，被跳过的条目永远删不到。必须**先取全 ID（只读不删）
   再统一删**；坚持边翻边删就得按 `total_page` **倒序**翻页。
2. **不能在循环里覆盖同一个变量、循环外再删**：那样只剩最后一页的 ID，评论超过 100 条时
   前面几页全丢且脚本仍 `exit 0`。必须在循环内**追加**到累积容器（临时文件/变量）。
3. **`.body` 可能为 `null`**（行内 diff 评论、系统评论）：对 `null` 调 `startswith` 会让 jq
   **中途报错退出**——崩之前输出的 ID 算数、其后的全丢，外层 `for` 又不检查 jq 退出码，
   于是拿着残缺名单跑完并 `exit 0`。必须写 `(.body // "")` 并显式判断 jq 退出码。
4. **排序键只放 `reply_to` 标志 + id，不要掺 `.body`**：`body` 常带换行而 `sort` 按行排序，
   掺进去会把一条评论拆成多行重排，「回复在前」的顺序读不出来（实测同一批 43 条评论：
   掺 `body` 输出 475 行，只留标志 + id 输出 43 行）。要看正文另起一条 `jq -c` 命令。

⚠️ **展开后还必须调整删除顺序**：投影是「父在前、回复在后」，照这个顺序删只会删掉回复、
父评论因「还有回复」返回 `405` 而留下，清理只完成一半且 `curl -s` 退出码仍是 `0`。
规则：按 `reply_to` 排序（**非 null 的回复先删**），并逐个校验 HTTP 状态码。

### 批量清理脚本（六条不变式全部落在代码里）

```bash
# ---- 第 1 步：翻页把所有待删 ID 取全（只读不删），追加进临时文件 ----
ids=$(mktemp)
blocked=$(mktemp)
hdr=$(mktemp)
trap 'rm -f "$ids" "$blocked" "$hdr"' EXIT
MAX_PAGES=50   # 兜底上界（50 × 100 = 5000 条父评论），正常永不触发
page=1
while :; do
  [ "$page" -gt "$MAX_PAGES" ] && { echo "ERROR: 翻页超过 ${MAX_PAGES} 页兜底上界，ID 未取全" >&2; exit 1; }
  # -D 把响应头落到文件；头是 CRLF 结尾，取值前必须 tr -d '\r'，否则数值比较会失败
  resp=$(curl -s -D "$hdr" "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/comments?access_token=$GITCODE_API_TOKEN&per_page=100&page=${page}")
  total_page=$(tr -d '\r' < "$hdr" | awk -F': *' 'tolower($1)=="total_page"{print $2}')
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
  [ "$count" -eq 0 ] && break           # total_page 缺失时的兜底；count 数的是**父评论**条数
  [ -n "$total_page" ] && [ "$page" -ge "$total_page" ] && break
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

> 六条不变式对应关系：① 先取全再统一删（第 1 步 / 第 3 步分离）② 响应非数组即 `exit 1`
③ 展开 `reply[]`（第 1 步的 jq 投影）④ 405 预检（第 2 步，有阻塞项一次 `DELETE` 都不发）
⑤ 回复在前、父评论在后（`sort -r`）⑥ 逐条校验 `-w '%{http_code}'`。
> 想先看清单再删（不自动执行）：只跑第 1 步，把第 3 步换成打印 `sort -r "$ids"`。
> `SKILL.md` 只保留这六条不变式的清单与指针，脚本全仓以此处为唯一一份，别在两处各写一遍。

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

> ⚠️ **接口给的增删行数与本地 `git diff --numstat` 可能差 1**（实测：同一文件本地 numstat
> 比 `files` 与 `files.json` 给的值各多 1，其余文件完全一致；两接口之间从未出现分歧）。
> 差值不影响判断，但**不要拿它做「diff 是否一致」的精确对账**，也不要把这类数字写进 PR 描述（易变）。

---

## 获取文件原始内容

```bash
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/contents/<文件路径>?ref=<commit_sha>&access_token=$GITCODE_API_TOKEN" | \
  jq -r '.content' | base64 -d
```

**返回字段**（实测）：`type` / `encoding` / `size` / `name` / `path` / `content`（base64）/
`sha`（**blob sha**）/ `url` / `html_url` / `download_url` / `_links`。

`download_url` 是 `https://raw.gitcode.com/${owner}/${repo}/blobs/<blob_sha>/<文件路径>` 形式，
直接返回明文，实测稳定 `200`（连测 3 次全 `200`，且下载内容与本地文件 md5 一致），
适合管道给 `grep -n` 定位行号。

> ⚠️ **`…/raw/<commit_sha>/<文件路径>` 这个形式实测不稳定，不要当唯一手段**：同一 URL、同一 sha，
> 在某一段时间内对 `.agents/skills/gitcode-pr/` 下的 `commands/review.md`、
> `references/gitcode_api.md`、`evals/evals.json` 连续 3 次返回 `403`（body「暂不支持预览」），
> 而同一时刻同 sha 的 `SKILL.md`/`README.md`/`build.sh` 返回 `200`；几分钟后同一批 URL
> 又连续 5 次全部 `200`；后续复查再次复现，且换成 `SKILL.md` 返回 `403`、
> `review.md`/`README.md`/`build.sh` 返回 `200`——**命中哪些文件也在变**。
> `/pulls/<n>/files` 响应里的 `raw_url`、
> `files.json` 里 `diffs[].head.url` 都是该形式，同样不可靠。

**`ref` 参数取值（实测）**：commit sha（40 位全 sha 或短 sha）与**分支名都可用**——
`ref=<默认分支名>`、`ref=<带点的版本分支名>`、`ref=<带斜杠的特性分支名>`（原样或 `%2F` 编码均可）
都返回 `200`，且分支名取到的内容与本地 `git show <remote>/<branch>:<path>` 的 md5 一致；
不传 `ref` 时用默认分支。跨 fork PR 的 head commit 在**目标仓库**路径下也取得到
（实测用目标仓库与源仓库两条路径都 `200`、blob sha 相同）。

> ⚠️ 审查 PR 时**仍应传 head commit sha**，不要用分支名：分支名指向分支**当前**位置，
> PR 提交后分支若又推了新 commit，取到的就不是被审查的那一版，`grep -n` 得到的行号会与 diff 对不上。
>
> ⚠️ **`404` 区分不了「ref 不存在」和「文件在该 ref 上不存在」**：两种情况响应体完全相同，
> 都是 `{"error_code":404,"error_code_name":"NOT_FOUND","error_message":"项目或文件不存在","trace_id":"…"}`
> （实测三种组合返回一模一样：「不存在的 ref + 存在的文件」；「存在的 ref + 该 ref 上没有的文件」；
> 「存在的 ref + 不存在的文件」）。
> 遇到 `404` 要两种可能都排查。**没有** `{"message":"404 Commit Not Found"}` 这种响应体。
>
> ⚠️ `curl -s` 对 `403`/`404` 的退出码仍是 `0`，取完必须校验状态码 / 内容非空，
> 否则会把错误页当成文件内容、`grep -n` 静默输出空结果。

---

## 注意事项

1. **Token 配置**：使用环境变量 `GITCODE_API_TOKEN`
2. **多行选择**：写入用 `start_position`（起始行）+ `position`（结束行），
   读取看 `diff_position.start_new_line` / `end_new_line`
3. **分页**：必须显式传 `per_page=100`、从 `page=1` 起递增翻页取完；⚠️ **终止判据用响应头
   `total_page`**，不要用「本页条数 < 请求的 `per_page`」。上限静默夹回、`total_page` 读头要点、
   越界 `page` 行为与全部实测依据见「获取 PR 讨论（包含行内评论）」
4. **两种 ID 不要混用**：`discussion_id`（哈希串，用于回复）vs 评论 `id`（整数，用于删除）
