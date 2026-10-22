# GitCode PR API 参考

## 目录

1. [官方文档已知缺陷（先读）](#官方文档已知缺陷先读)
2. [认证方式](#认证方式)
3. [获取当前用户（token 对应的账号）](#获取当前用户token-对应的账号)
4. [获取仓库信息（查询 Fork 来源）](#获取仓库信息查询-fork-来源)
5. [获取 PR 信息](#获取-pr-信息)
6. [获取 PR 讨论（包含行内评论）](#获取-pr-讨论包含行内评论)
7. [提交行内评论（支持多行选择）](#提交行内评论支持多行选择)
8. [提交普通评论](#提交普通评论)
9. [删除 PR 评论](#删除-pr-评论)
10. [获取 PR 文件变更](#获取-pr-文件变更)
11. [获取文件原始内容](#获取文件原始内容)
12. [注意事项](#注意事项)

---

## 官方文档已知缺陷（先读）

官方文档（docs.gitcode.com）在本 skill 依赖的几个点上缺失、滞后或有硬错误，按字面执行会出错：

| # | 文档缺陷 |
|---|------|
| 1 | **几乎不文档化错误响应**：绝大多数接口页的 `Responses` 段只列 2xx → 照文档写的解析代码没有失败分支 |
| 2 | **`access_token` 被标 `required`（query 参数）**，几乎全部接口页都这么标；且**全站不提任何请求头认证写法** |
| 3 | **`access_token` 被描述成「用户授权码」**；而 OAuth 页里「授权码」是另一个东西（`code`，一次性、需换 token）→ 按字面执行会拿错凭证 |
| 4 | **流水线页示范的是另一种认证头**：`docs/help/home/org_project/pipeline/security-permissions/token-permissions` 给的是 `Authorization: token $ATOMGIT_TOKEN`，示例域名指向 `atomgit.com`；该页从未说明这种写法是否适用于 v5 REST（不适用，见「认证方式」） |
| 5 | **响应 Schema 滞后**：`get-api-v-5-user` 页面自称 `Example (auto)`，其 Schema 段列出的字段少于实际响应 |

---

## 认证方式

官方文档（<https://docs.gitcode.com/docs/apis/>）明确「当前可用的版本为 `/api/v5`」，
读写一律使用 v5：

| API 版本 | 认证方式 | 端点格式 |
|---------|---------|---------|
| **GitCode API v5** | `?access_token=<token>`（或 `Authorization: Bearer <token>`、`PRIVATE-TOKEN: <token>`） | `/api/v5/repos/<owner>/<repo>/...` |

### 基址

文档统一写 `https://api.gitcode.com`。`https://gitcode.com/api/v5/...` 与
`https://atomgit.com/api/v5/...` 指向**同一套 v5 后端**，同一枚 PAT 通用
（`scripts/create_pr.py` 用的就是 `gitcode.com/api/v5`）。本 skill 一律用 `api.gitcode.com`；
**同一份脚本不要混用域名**。

### 认证写法

`?access_token=` query 参数、`Authorization: Bearer <token>`、`PRIVATE-TOKEN: <token>`
三种均可用。官方流水线页示范的 `Authorization: token <token>` 在 v5 REST 上返回 `401`
——那是流水线内部注入令牌的专用格式，不适用 v5 REST（见「官方文档已知缺陷」#4）。

> ⚠️ 官方文档把 `access_token`（query 参数）标为 `required`、且完全不提请求头写法
> （见「官方文档已知缺陷」#2）。query 形式会把长期凭证带进 shell history、CI 日志、服务端
> access log 与代理日志；**共享环境或 CI 里请优先用请求头**。本 skill 下文示例两种写法都有，
> 沿用即可，但**同一份脚本不要混用两种写法**。

> ⚠️ **`401` 不一定是 token 的问题**：先 `echo "$owner/$repo"` 确认仓库路径变量非空、
> `echo "${#GITCODE_API_TOKEN}"` 确认凭证非空，再怀疑 token 本身。

---

## 获取当前用户（token 对应的账号）

### API 端点

取「当前 token 对应的账号」，用于把评论的 `user.login` 与「本 agent」比对
（`../SKILL.md` →「回读校验」的「作者 + 唯一标记」备选判据）。

```bash
GET https://api.gitcode.com/api/v5/user
```

> ⚠️ **不带路径参数**：`/api/v5/user` 是「当前认证用户」，`/api/v5/users/<login>` 才是按 login 查任意用户。
> 本 skill 只用前者。

### 请求示例：先探测，再取字段

> ⚠️ **不要写成 `curl -s … | jq -r '.login // empty'`**（官方文档只给了 `200` 的 Schema，照它写
> 就会漏掉失败分支，见「官方文档已知缺陷」#1）：本端点可能返回 `401`/`403`，jq 对错误体
> 取 `.login` 得空串——「权限不够」「token 失效」「变量没导出」「网络不通」
> 四种故障会退化成同一个空输出。必须把 **HTTP 状态码与响应体**一起留住。

```bash
# -w 用字面量标记 __HTTP__ 分隔 body 与状态码，${var%…}/${var##…} 是 POSIX 参数展开，bash 与 dash 均可
resp=$(curl -s -w '__HTTP__%{http_code}' \
  "https://api.gitcode.com/api/v5/user?access_token=$GITCODE_API_TOKEN")
body=${resp%__HTTP__*}      # HTTP 响应体
status=${resp##*__HTTP__}   # HTTP 状态码
MY_LOGIN=$(printf '%s' "$body" | jq -r '.login // empty')
# 失败判据要覆盖两种「没拿到 login」：非 200，以及 **200 但 .login 为空**
if [ "$status" != "200" ] || [ -z "$MY_LOGIN" ]; then
  echo "WARN: GET /api/v5/user 未取到 login（HTTP ${status}）" >&2
  # 响应体自带换行，直接 echo 会多打一个空行，故先 tr 掉
  echo "      响应体：$(printf '%s' "$body" | tr -d '\r\n')" >&2
fi
[ -n "$MY_LOGIN" ] || { echo "ERROR: 取不到当前账号 login（HTTP ${status}）" >&2; exit 1; }
echo "MY_LOGIN=$MY_LOGIN"
```

> 拿到 `MY_LOGIN` 后与评论列表的 `author` 比对即可：
> `<翻页脚本输出> | jq -c --arg me "$MY_LOGIN" 'select(.author == $me) | {id, reply_to, body}'`。
> ⚠️ 过滤器别带 `.[]`：翻页脚本输出的是**逐行对象流**，带 `.[]` 是对每个对象的**值**再迭代，
> 会报 `Cannot index number with string "author"`。

### 响应字段

顶层是**对象**（不是数组）。本 skill **只依赖 `login` 这一个字段**（与评论的 `user.login` 同口径）。
官方文档的 Schema 少列了字段（见「官方文档已知缺陷」#5），所以**别按文档 Schema 做严格解析**，
完整键集以你自己那次响应为准。

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
| `web_url` | string | 原仓库网页 URL |
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
# ${owner}/${repo} 的解析见 ../SKILL.md「提取仓库 owner/repo 信息」，此处假定这两个变量已设好。
response=$(curl -s -H "Authorization: Bearer $GITCODE_API_TOKEN" \
  "https://api.gitcode.com/api/v5/repos/${owner}/${repo}")
# ⚠️ 不能用 `jq -e '.fork'` 判「调用是否成功」：`jq -e` 对 false/null 同样返回退出码 1，
#    正常的非 fork 仓库会被误报成「API 调用失败」；判有效性要看它是不是带 fork 键的对象。
if [ "$(echo "$response" | jq -r 'if (type == "object") and (has("fork")) then "ok" else "bad" end' 2>/dev/null)" != "ok" ]; then
  echo "API 调用失败或响应异常：$response" >&2; exit 1
fi
if [ "$(echo "$response" | jq -r '.fork')" = "true" ]; then
  echo "这是一个 fork 仓库"
  echo "原仓库: $(echo "$response" | jq -r '.parent.full_name')"
  echo "原仓库默认分支: $(echo "$response" | jq -r '.parent.default_branch')"
  echo "原仓库 URL: $(echo "$response" | jq -r '.parent.web_url')"   # 键是 web_url，不是 html_url
else
  echo "这不是一个 fork 仓库：$(echo "$response" | jq -r '.full_name')"
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

> ⚠️ **顶层数组不含回复**：回复嵌在父评论的 `reply[]` 里，顶层数组查不到任何回复的 ID。
> `reply[]` 里的对象**只有** `id`/`body`/`user`/`created_at`/`updated_at`，
> **没有** `discussion_id`/`comment_type`/`resolved`/`path`/`diff_position`，需要时从父评论带下来；
> `resolved` 是讨论级属性、以父评论为准，**不要给回复伪造取值**。带下来的 `comment_type` 是
> **列表接口那一套取值**（`diff_comment`/`pr_comment`），单条接口是另一套，对照见下
> 「两个接口的 `comment_type` 取值不是同一套」。回复的完整字段仍需用 `note_id` 查单条接口
> `GET /pulls/comments/<id>`。
>
> **展开后的行形状与父行不同**：回复行没有 `resolved`/`lines`（`diff_position`）键，
> 按这两个字段过滤只会命中父行；需要区分时用 `reply_to`（`null` = 父评论，非 null = 回复，值为父评论 ID）。
>
> 分页 `length` 与 `total_count` 都只数父评论；Σ 校验口径见 `../SKILL.md` →「翻页规则」。

本节给**字段投影的全集版**（含 `resolved` / `diff_position`）。下面这条**只取第一页**，
用于看单页字段形状，不要拿它当「取全」的实现：

```bash
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/comments?access_token=$GITCODE_API_TOKEN&per_page=100" | \
  jq '.[] | {id, discussion_id, comment_type, author: .user.login, resolved, lines: .diff_position, body} as $c
      | $c, (.reply[]? | {id, discussion_id: $c.discussion_id, comment_type: $c.comment_type, author: .user.login, body, reply_to: $c.id})'
```

> **失败必须显式**：响应不是数组时该分支必须 `exit 1`，完整规则见
> `../SKILL.md` →「翻页规则」。
>
> **响应头带 `total_page` / `total_count`，翻页终止以 `total_page` 为准**。

**查询参数**：

| 参数 | 说明 |
|------|------|
| `page` / `per_page` | 分页参数；`per_page` 默认 **20**、最大 **100**（超出会被**静默夹回 100**）。翻页规则见 `../SKILL.md` →「翻页规则」 |
| `direction` | `asc` / `desc` |
| `comment_type` | 筛选评论类型：`diff_comment`（行内）/ `pr_comment`（普通） |

**返回字段说明**：

| 字段 | 说明 |
|------|------|
| `id` | 评论 ID（整数），删除评论用它 |
| `discussion_id` | 讨论 ID（哈希串），回复该讨论用它 |
| `comment_type` | `diff_comment` = 行内评论，`pr_comment` = 普通评论 |
| `diff_position.start_new_line` / `end_new_line` | 行内评论的起止行号 |
| `resolved` / `resolved_by` | 检视意见是否已解决、解决人（**仅行内评论有此键**） |
| `reply[]` | 该讨论下的回复，嵌套在父评论里，不单独出现在顶层 |
| `user.login` | 评论者 |

> ⚠️ 列表接口**没有 `path` / `position` 这两个键**，**按这两个字段过滤永远命中不了**。
> 需要路径时改查单条
> `GET /api/v5/repos/${owner}/${repo}/pulls/comments/<COMMENT_ID>`，其 `position` 字段含
> `old_path` / `new_path` / `new_line` / `base_sha` / `head_sha`。
>
> ⚠️ **行号不需要查单条接口**：列表接口的 `diff_position.start_new_line` / `end_new_line`
> 就是起止行（单行评论起止同值）。但**行号数值易变**：接口按 PR head **当前**指向的 commit
> 重算行号，锚定的仍是同一行源码，行号却随该文件上方的增删位移——所以只把「哪些键存在」
> 当结构事实用，**不要把具体行号当稳定结论引用**。
> 另，**两个接口的位置键名不是同一套**，别串用：
>
> | 含义 | 列表接口 `diff_position` | 单条接口 `position` |
> |------|--------------------------|---------------------|
> | 起始行 | `start_new_line` | `start_new_line`（**单行评论时该键不存在**） |
> | 结束行 | `end_new_line` | `new_line` |
> | 文件路径 | **无** | `new_path` / `old_path` |
> | base/head SHA | **无** | `base_sha` / `start_sha` / `head_sha` |
> | 旧代码行 | **无** | **无 `old_line` 键**（旧行号取不到，不要用它判断或过滤） |

> ⚠️ **两个接口的 `comment_type` 取值不是同一套**，按上表口径解析单条响应会拿到预期外的值：
>
> | 评论种类 | 列表接口 `GET /pulls/<n>/comments` | 单条接口 `GET /pulls/comments/<id>` |
> |----------|-----------------------------------|--------------------------------------|
> | 行内评论 | `diff_comment` | `DiffNote` |
> | 普通评论 | `pr_comment` | `null` |
>
> **判断行内/普通只用列表接口的值**；单条接口的 `comment_type` 只能当「非 null 即行内」用。

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
| `access_token`（query） | 三种认证写法之一，见「认证方式」 | 否 |

**成功响应（HTTP 201）**：

```json
{"id": "<discussion_id 哈希串>", "body": "...", "note_id": 123456789}
```

> ⚠️ 响应里的 `id` 是**讨论 ID**（哈希串），不是评论 ID。删除评论必须用 `note_id`
> （即评论列表里的整数 `id`）；把哈希串传给 `DELETE` 会得到 `400`。

### 多行选择说明

- **单行评论**：只传 `position`（或与 `start_position` 设为相同值）
- **多行评论**：`start_position` = 起始行，`position` = 结束行

> `start_position` 未出现在官方文档的请求体 schema 中（官方只列 `body`/`path`/`position`/`position_type`），
> 但实际有效：传了之后读回的 `diff_position` 会带上这两个行号（`start_new_line` / `end_new_line`）。

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

**成功响应（HTTP 201）**：`{"id":"<discussion_id>","body":"...","note_id":123456789}`。
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
| `access_token` | string | 否 | 三种认证写法之一，见「认证方式」 |

### 请求示例

```bash
# 用 -w '%{http_code}' 校验，成功为 200（非 200 的含义见下表）
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
| `405` | **该评论下还有回复**，删不掉 |

> ⚠️ **删除有顺序要求**：父评论带着回复时删不掉（`405`），必须**按 `reply_to` 排序**
> （非 null 的回复先删、父评论最后删），别照「父在前、回复在后」的投影顺序删。
> 筛选条件漏掉回复时这个 `405` 是**必然**发生而非偶发——应在删除前就用 `reply[]` 预检出来并直接失败。

### 获取评论 ID

用 `../SKILL.md` →「获取 PR 讨论列表（包含行内评论）」那份翻页循环收集 `id`。

⚠️ **三条会「少删却 `exit 0`」的坑**：

1. **不能边翻页边删**：删除会让列表前移，被跳过的条目永远删不到。必须**先取全 ID（只读不删）
   再统一删**，且在循环内**追加**到累积容器（覆盖同一个变量只会剩最后一页的 ID）。
2. **`.body` 可能为 `null`**（行内 diff 评论、系统评论）：对 `null` 调 `startswith` 会让 jq
   **中途报错退出**——崩之前输出的 ID 算数、其后的全丢。必须写 `(.body // "")` 并显式判断 jq 退出码。
3. **排序键只放 `reply_to` 标志 + id，不要掺 `.body`**：`body` 常带换行而 `sort` 按行排序，
   掺进去会把一条评论拆成多行重排。

> ⚠️ 删除阶段同样逐个校验 HTTP 状态码（成功为 `200`）。
>
> ⚠️ 批量删除前先「只看清单不执行」跑一遍（把 `DELETE` 换成打印 ID），确认后再真删。

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
| `sha` | **提交 sha**（等于 `diff_refs.head_sha`），**不是 blob sha**；blob sha 在 `blob_id` |
| `blob_id` | blob sha（等于 `git rev-parse <commit>:<path>`），可拼成稳定的 `…/blobs/<blob_id>/<path>` 取内容地址 |
| `blob_url` | 网页版文件页（`gitcode.com/<owner>/<repo>/blob/<commit_sha>/<path>`），不是取原始内容的地址 |
| `raw_url` | 形式是 `…/raw/<commit_sha>/<path>`，**不要用**；取内容用上面的 `blob_id` 拼 `…/blobs/<blob_id>/<path>` |
| `additions` / `deletions` | 增删行数 |
| `patch.diff` | unified diff 文本，行内评论定位行号从这里取 |
| `patch.old_path` / `new_path` | 重命名前后路径 |
| `patch.added_lines` / `removed_lines` | 该文件的增删行数 |
| `patch.new_file` / `renamed_file` / `deleted_file` / `too_large` | 变更类型标记 |
| `source_branch` / `target_branch` | PR 的源 / 目标分支名 |
| `source_project` / `target_project` | 源 / 目标仓库对象（含 `id`/`full_name`/`human_name`/`path`）。**跨 fork PR 取 head 仓库用 `source_project.full_name`**，别去猜 |
| `patch.a_mode` / `b_mode` | 变更前后的文件权限位（如 `100644`） |

> ⚠️ **接口给的增删行数与本地 `git diff --numstat` 可能对不上**，不影响判断，
> 但别拿它做「diff 是否一致」的精确对账，也别把这类数字写进 PR 描述。

---

## 获取文件原始内容

```bash
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/contents/<文件路径>?ref=<commit_sha>&access_token=$GITCODE_API_TOKEN" | \
  jq -r '.content' | base64 -d
```

**返回字段**：`type` / `encoding` / `size` / `name` / `path` / `content`（base64）/
`sha`（**blob sha**）/ `url` / `html_url` / `download_url` / `_links`。

`download_url` 是 `https://raw.gitcode.com/${owner}/${repo}/blobs/<blob_sha>/<文件路径>` 形式，
直接返回明文，适合管道给 `grep -n` 定位行号。

**`ref` 参数取值**：commit sha 与分支名都可用，不传 `ref` 时用默认分支。

> ⚠️ 审查 PR 时**仍应传 head commit sha**，不要用分支名：分支名指向分支**当前**位置，
> PR 提交后分支若又推了新 commit，取到的就不是被审查的那一版，`grep -n` 得到的行号会与 diff 对不上。
>
> ⚠️ 取完必须校验状态码 / 内容非空，否则会把错误页当成文件内容、`grep -n` 静默输出空结果。

---

## 注意事项

1. **Token 配置**：使用环境变量 `GITCODE_API_TOKEN`
2. **多行选择**：写入用 `start_position`（起始行）+ `position`（结束行），
   读取看 `diff_position.start_new_line` / `end_new_line`
3. **分页**：规则见 `../SKILL.md` →「翻页规则」；查询参数与返回字段见「获取 PR 讨论（包含行内评论）」
4. **两种 ID 不要混用**：`discussion_id`（哈希串，用于回复）vs 评论 `id`（整数，用于删除）
