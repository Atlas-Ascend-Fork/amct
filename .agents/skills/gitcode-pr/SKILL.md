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

### 获取访问令牌（第一步必须）

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

### 识别PR的目标仓库

#### 查询远程仓库

```bash
git remote -v
```

根据远程仓库 URL 确定目标仓库：

- **目标仓库**：PR 要合并到的仓库（通常是 origin 或 upstream）
- **源仓库**：当前工作分支所在的仓库

#### 提取仓库 owner/repo 信息

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

echo "Owner: $owner"
echo "Repo: $repo"
```

**后续所有 API 调用都应使用这些变量**：

- GitHub API v5 格式：`/repos/${owner}/${repo}/...`

#### 查询 Fork 的原仓库（当当前仓库是 fork 时）

当当前仓库是 fork 仓库时，需要查询其 fork 的原仓库作为 PR 目标仓库：

```bash
# 查询仓库信息获取 fork 来源
curl -s -H "Authorization: Bearer $GITCODE_API_TOKEN" \
  "https://api.gitcode.com/api/v5/repos/${owner}/${repo}" | \
  jq -r '
    if (type != "object") or (has("fork") | not) then
      "ERROR: 仓库信息响应异常（token 失效 / 仓库不存在？）：" + tostring
    elif .fork then
      "原仓库: " + (.parent.full_name // "<缺失>") + "\n" +
      "原仓库URL: " + (.parent.web_url // "<缺失>") + "\n" +
      "目标分支: " + (.parent.default_branch // "<缺失>")
    else
      "这不是一个 fork 仓库，直接使用当前仓库"
    end
  '
```

> ⚠️ 必须显式区分「响应异常」和「不是 fork」：`if .fork then … else …` 里 `null` 也是假值，
> token 失效时返回的 `{"message":"401 Unauthorized"}` 会走进 else，被读成「这不是 fork 仓库」，
> 于是 PR 直接建到自己的 fork 上而不是上游仓库。
> ⚠️ 原仓库网页地址的键名是 **`web_url`**，不是 `html_url`——v5 的 `repos` 响应里**没有**
> `html_url` 键（实测 `yaoguangxiu/amct`：`parent` 只有 `url`/`ssh_url_to_repo`/
> `http_url_to_repo`/`web_url`/`readme_url`），取错键不报错，只会静默输出空串。

**响应关键字段**：

| 字段 | 说明 |
| ------ | ------ |
| `fork` | 是否为 fork 仓库（`true`/`false`） |
| `parent.full_name` | 原仓库完整名称（格式：`owner/repo`） |
| `parent.web_url` | 原仓库网页地址（实测键名；**不是** `html_url`，该键不存在） |
| `parent.default_branch` | 原仓库默认分支 |

**示例响应（实测 `GET /repos/yaoguangxiu/amct`，仅保留相关键）**：

```json
{
  "fork": true,
  "parent": {
    "full_name": "cann/amct",
    "web_url": "https://gitcode.com/cann/amct",
    "http_url_to_repo": "https://gitcode.com/cann/amct.git",
    "default_branch": "master"
  }
}
```

### 获取 PR 评论和讨论

#### 获取 PR 讨论列表（包含行内评论）

GitCode 使用 v5 API 获取 PR 评论：

```bash
# 获取 PR 的所有评论（包括行内评论）
# ⚠️ 必须显式传 per_page=100 并从 page=1 起翻页取完：per_page 默认只有 20（实测：
#    cann/amct PR!431 共 38 条父评论、PR!417 共 45 条，不带 per_page 都只返回 20 条）。
#    「最大 100」取自官方文档，未实测（抽样中没有父评论 > 100 的 PR；在 38 条的 PR 上传
#    per_page=150/200 都返回全部 38 条，既不能证明也不能推翻上限）。按 100 翻页在
#    上限更大时也只是多翻几页、结果仍正确。
# ⚠️ 顶层数组**不含回复**：回复嵌在父评论的 reply[] 里（实测 cann/amct PR!455：
#    回复 ID 在顶层一个都查不到），必须一并展开；
#    reply 对象只有 id/body/user/时间戳，discussion_id 与 comment_type 要从父评论带下来
#    （带下来的是**列表接口**那套取值，单条接口查同一条回复会得到另一套，别混用）。
page=1
while :; do
  resp=$(curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/comments?access_token=$GITCODE_API_TOKEN&per_page=100&page=${page}")
  # 响应必须是数组：无效 token 返回 {"message":"401 Unauthorized"}、网络失败返回空响应，
  # 两者都要显式失败退出，不能 break 后以 0 退出——空列表 + exit 0 会被读成「取全了且没有评论」。
  if [ "$(echo "$resp" | jq -r 'type' 2>/dev/null)" != "array" ]; then
    echo "ERROR: 第 ${page} 页评论请求失败或响应非数组，结果不完整：$resp" >&2
    exit 1
  fi
  # 每条父评论先原样输出，再把它 reply[] 里的回复展开成同形对象（带 reply_to 标记来源）
  echo "$resp" | jq -c '.[] | {id, discussion_id, comment_type, author: .user.login, body} as $c | $c, (.reply[]? | {id, discussion_id: $c.discussion_id, comment_type: $c.comment_type, author: .user.login, body, reply_to: $c.id})'
  count=$(echo "$resp" | jq 'length')
  [ "$count" -lt 100 ] && break   # count 数的是**父评论**条数（回复不计入分页），不足 100 即最后一页
  page=$((page + 1))
done
```

> 评论列表**取不全时必须停止**依赖它的判断（如「是否已评论过」「该删哪条」），
> 不得据空列表推断「没有评论」。完整翻页写法与响应头校验见
> `references/gitcode_api.md` → 「获取 PR 讨论（包含行内评论）」。
>
> **回复必须展开**：只投影顶层字段时，「只在既有讨论里回复过」的活动完全不可见——
> 判断「是否已评论过」会漏判并重复发意见，给 `DELETE` 收集 ID 会漏删自己发过的回复。
> 回复对象**没有** `discussion_id`/`comment_type`/`resolved`，前两者要从父评论带下来；
> `resolved` 是讨论级属性、以父评论为准，不要给回复伪造取值。
>
> ⚠️ 带下来的 `comment_type` 是**列表接口**那套取值（`diff_comment`/`pr_comment`），
> 不等于「和单条接口一致」：同一条回复用单条接口查是另一套
> （实测回复 `194233994`：父评论列表值 `diff_comment`，单条接口 `DiffNote`），
> 两套取值对照见 `references/gitcode_api.md` →「两个接口的 `comment_type` 取值不是同一套」。
> 回复的完整位置信息（文件路径 `position.new_path`、行号 `position.new_line`）
> 仍需用 `note_id` 查单条接口——列表接口既不给回复也不给父评论文件路径。
>
> ⚠️ **展开后的行形状与父行不同**（回复行没有 `resolved` 等父行才有的键），
> 用 `reply_to` 区分：`null` = 父评论，非 null = 回复（值为父评论 ID）。
> **校验「是否取全」要拿父评论条数比**：`per_page`/`total_count` 都只数父评论
> （实测 cann/amct PR!455：响应头 `total_count` 的值等于顶层数组长度，`reply[]` 内的回复不计入），
> 拿展开后的输出条数去比 `total_count` 必然偏大、会误判成取重了。

## GitCode API v5 说明

**认证头**：

- 推荐使用 query 参数：`?access_token=$GITCODE_API_TOKEN`
- 也可使用 Header：`PRIVATE-TOKEN: $GITCODE_API_TOKEN`

#### 解析评论数据

v5 API 返回**直接数组**，不是包装对象。评论数据包含以下关键字段：

| 字段 | 说明 |
| ------ | ------ |
| `id` | 评论 ID |
| `discussion_id` | 讨论 ID（用于回复） |
| `body` | 评论内容 |
| `user.login` | 评论作者用户名 |
| `created_at` | 评论创建时间 |
| `updated_at` | 评论更新时间 |
| `comment_type` | 评论类型，只有两种取值：`diff_comment` = 行内评论，`pr_comment` = 普通评论 |
| `diff_position.start_new_line` / `end_new_line` | 行内评论的起止行号（仅行内评论有此键） |
| `diff_position.position_type` | `text` = 代码行评论（实测）；`binary` = 文件级评论（官方文档值，**未实测**） |
| `resolved` / `resolved_by` | 检视意见是否已解决、解决人 |
| `reply[]` | 该讨论下的回复，**嵌在父评论里**，不单独出现在顶层数组 |

**示例返回结构**：

```json
[
  {
    "id": 171409022,
    "discussion_id": "b4ecb001ec8af59d02c3ffeffeb83fb7a8226c84",
    "body": "评论内容",
    "user": {
      "login": "username",
      "name": "显示名称"
    },
    "created_at": "2026-05-14T15:25:32+08:00",
    "comment_type": "pr_comment"
  }
]
```

**行内评论示例**：

```json
{
  "id": 123456,
  "discussion_id": "a8d4cef18a01d322fa540cfcc2eccb94e9d544d0",
  "body": "这段代码需要优化",
  "comment_type": "diff_comment",
  "resolved": false,
  "diff_position": {
    "start_new_line": 60,
    "end_new_line": 60,
    "position_type": "text"
  },
  "reply": []
}
```

**重要说明（实测 cann/amct PR!455 的全部父评论，条数随评论增长变化，不在此固化）**：

- 列表接口**没有** `path` / `position` 这两个键（`has("path")` 与 `has("position")` 均为 `false`，
  不是「值为 `null`」）。用 `jq '.path'` 取值会得到 `null`，但**不要按这两个字段过滤**，永远命中不了。
- **行号列表接口就能拿到**：在 `diff_position.start_new_line` / `end_new_line` 里
  （当时全部 `diff_comment` 都带此键，如 `194143085` → `60/60`）。
  不需要为了行号再逐条调单条接口。
- **只有「被评论的文件路径」列表接口拿不到**，需要查单条评论接口的 `position.new_path`。
- `comment_type` 只有 `diff_comment`（行内）/ `pr_comment`（普通）两种取值。

#### 获取单条评论详情（包含完整位置信息）

**端点**：

```
GET /repos/:owner/:repo/pulls/comments/:id
```

**示例**：

```bash
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/comments/<COMMENT_ID>?access_token=$GITCODE_API_TOKEN"
```

**返回完整字段**：

```json
{
  "id": 171419900,
  "body": "行内评论内容",
  "path": null,
  "position": {
    "base_sha": "d8eb570...",
    "start_sha": "d8eb570...",
    "head_sha": "e54ef3...",
    "old_path": "runtime/v1/executor.cc",
    "new_path": "runtime/v1/executor.cc",
    "position_type": "text",
    "old_line": 212,
    "new_line": 212
  },
  "comment_type": null,
  "user": "username",
  "created_at": "2026-05-14T16:06:24+08:00"
}
```

> ⚠️ **两个接口的 `comment_type` 不是同一套取值**（实测 cann/amct PR!455）：
>
> | 评论种类 | 列表接口 `/pulls/<n>/comments` | 单条接口 `/pulls/comments/<id>` |
> |----------|-------------------------------|----------------------------------|
> | 行内评论 | `diff_comment` | `DiffNote` |
> | 普通评论 | `pr_comment` | `null` |
>
> 同一条行内评论 `194208411`：列表里是 `diff_comment`，单条里是 `DiffNote`；
> 普通评论 `194141205`/`194176110`：列表里是 `pr_comment`，单条里是 `null`
> （v5 见不到 `DiscussionNote` 这个取值）。
> 所以**判断行内/普通一律用列表接口的值**；单条接口只能当「`comment_type` 非 `null` 即行内」用，
> 不要拿列表接口的取值去匹配单条响应（反之亦然），否则永远匹配不上。

**关键位置字段（单条接口 `/pulls/comments/:id` 专有）**：

| 字段 | 说明 |
| ------ | ------ |
| `position.new_path` | 文件路径 |
| `position.new_line` | 行号（新代码） |
| `position.old_line` | 行号（旧代码） |
| `position.base_sha` | Base 提交 SHA |
| `position.head_sha` | Head 提交 SHA |

> ⚠️ 上表的 `position.*` 是**单条接口**的字段名，与列表接口的 `diff_position.*` **不是同一套键名**，
> 且**结束行的字段名在两个接口里不同**（实测 cann/amct PR!447 多行行内评论 `193754581`，
> 列表 `diff_position = {start_new_line:29, end_new_line:31}`，单条
> `position = {new_line:31, start_new_line:29, new_path:"…/deploy_selection.py", …}`）：
>
> | 含义 | 列表接口 `diff_position` | 单条接口 `position` |
> |------|--------------------------|---------------------|
> | 起始行 | `start_new_line` | `start_new_line`（**单行评论时该键不存在**） |
> | 结束行 | `end_new_line` | `new_line`（**不叫 `end_new_line`**） |
> | 文件路径 | **无** | `new_path` / `old_path` |
> | base/head SHA | **无** | `base_sha` / `start_sha` / `head_sha` |
>
> 所以：**行号两个接口都能拿到**（键名不同，别拿 `end_new_line` 去查单条响应）；
> **文件路径与 SHA 只有单条接口有**。

**接口对比**：

| 接口 | 位置信息 | comment_type | 用途 |
| ------ | --------- | ------------- | ------ |
| `/pulls/:number/comments` | `diff_position.start_new_line`/`end_new_line`（**有行号、无文件路径**；无 `path`/`position` 键） | `diff_comment`（行内）/ `pr_comment`（普通） | 获取评论列表 |
| `/pulls/comments/:id` | 完整 `position` 对象（**含 `new_path` 文件路径**） | `DiffNote`（行内）/ `null`（普通） | 获取单条评论详情 |

**注意**：

- 要按「行内 / 普通」区分评论，请使用列表接口的 `comment_type`；
  单条评论接口对普通评论返回 `null`、对行内评论返回 `DiffNote`，取值形态与列表接口不同。
- 要**行号**用列表接口的 `diff_position` 即可，不必逐条调单条接口；
  只有要**文件路径**时才需要单条接口的 `position.new_path`。

### 获取 PR 文件变更

```bash
# 使用 v5 API（files.json）
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/files.json?access_token=$GITCODE_API_TOKEN"

# 解析第一个文件路径
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/files.json?access_token=$GITCODE_API_TOKEN" | \
  jq -r '.diffs[0].statistic.new_path'
```

**响应字段说明**：

- `diffs[].statistic.new_path` - 新文件路径
- `diffs[].statistic.old_path` - 旧文件路径
- `diff_refs.base_sha` - Base SHA
- `diff_refs.head_sha` - Head SHA
- `diff_refs.start_sha` - Start SHA

### 提交行内评论（支持多行选择）

#### 使用 v5 API（推荐）

```bash
curl -X POST \
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

**参数说明**：

| 参数 | 说明 | 必需 |
| ------ | ------ | ------ |
| `body` | 评论内容 | ✅ |
| `path` | 文件相对路径 | ✅ |
| `position` | 结束行号 | ✅ |
| `start_position` | 起始行号（多行选择） | 多行时 |

**单行评论示例**：

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/comments?access_token=$GITCODE_API_TOKEN" \
  -d '{
    "body": "第70行需要重构",
    "path": "runtime/v1/opskernel_executor/ops_kernel_executor_manager.cc",
    "position": 70
  }'
```

**多行评论示例**：

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/comments?access_token=$GITCODE_API_TOKEN" \
  -d '{
    "body": "第70-75行需要优化",
    "path": "runtime/v1/opskernel_executor/ops_kernel_executor_manager.cc",
    "position": 75,
    "start_position": 70
  }'
```

**注意**：`start_position` 参数是未文档化的参数，但测试证明可能有效。

#### 创建评论的返回值问题

**重要**：POST 创建评论 API 立即返回的是**哈希字符串格式 ID**，不是数字 ID。

**返回示例**：

```json
{
  "id": "b227720346ef5a37e410d17e76a1c1e1f60d5d80",  // 哈希字符串
  "discussion_id": null,
  "body": "评论内容",
  "created_at": null
}
```

**获取真正的数字 ID**：
创建评论后，需要查询评论列表获取数字 ID（用于后续删除等操作）。
⚠️ 不要用 `contains(.body)` 做宽匹配定位——若 PR 中已有包含相同文本的评论，
会匹配到错误的评论 ID，导致误删/误操作他人评论。应满足以下任一精确条件：

- 在评论正文里嵌入一个**唯一标记**（如随机串/时间戳），用 `==` 或 `contains(唯一标记)` 精确匹配；
- 或结合**当前用户**（`.user.login`）、**创建时间**（`.created_at`）、POST 返回的
  **`discussion_id`** 共同过滤，缩小到唯一一条。

```bash
# 推荐：用 POST 返回的 discussion_id 定位，再用 reply_to 收敛到唯一一条
DISCUSSION_ID="b227720346ef5a37e410d17e76a1c1e1f60d5d80"   # POST 返回的哈希串
# ⚠️ 同样必须 per_page=100 **并翻页**：目标评论在第 20 条之后时不带 per_page 会查不到、
#    在第 100 条之后时不翻页会查不到，两种都会把「创建成功」误判成「创建失败」。
#    下面按页查找，命中即停。
# 顶层不含回复：若目标是「回复」而非新建评论，必须把 reply[] 一并展开才 select 得到
# ⚠️ 展开后 discussion_id **不再唯一**：同一讨论的父评论与其全部回复共享同一个 discussion_id
#    （实测 PR!455 讨论 d21d994a… 命中 3 条：父 194208411 + 回复 194229227/194233994），
#    所以必须再加一层收敛条件：
#      · 目标是刚新建的顶层评论 → 追加 select(.reply_to == null)
#      · 目标是刚发出的回复     → 用回复响应里的 note_id 直接比 .id，或用唯一标记比 .body
page=1
found=""
while :; do
  resp=$(curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/comments?access_token=$GITCODE_API_TOKEN&per_page=100&page=${page}")
  if [ "$(echo "$resp" | jq -r 'type' 2>/dev/null)" != "array" ]; then
    echo "ERROR: 第 ${page} 页评论请求失败或响应非数组，无法确认评论是否创建成功：$resp" >&2
    exit 1
  fi
  found=$(echo "$resp" | jq -c --arg did "$DISCUSSION_ID" --arg me "$MY_LOGIN" '
      .[] | {id, discussion_id, comment_type, author: .user.login, body} as $c
      | $c, (.reply[]? | {id, discussion_id: $c.discussion_id, comment_type: $c.comment_type, author: .user.login, body, reply_to: $c.id})
      | select(.discussion_id == $did or (.author == $me and .body == "唯一标记内容"))
      | select(.reply_to == null)          # 目标是新建的顶层评论；找回复则改用 .id == <note_id>
      | {id, discussion_id, comment_type, reply_to}')
  [ -n "$found" ] && break
  count=$(echo "$resp" | jq 'length')
  [ "$count" -lt 100 ] && break   # count 数的是**父评论**条数（回复不计入分页）
  page=$((page + 1))
done
# ⚠️ 找不到必须显式失败：空结果 + exit 0 会被读成「查过了、确实没有」，
#    与「请求失败/漏页导致查不到」无法区分——这正是上面要 exit 1 的同一个理由
[ -n "$found" ] || { echo "ERROR: 翻完所有页仍未找到该评论，创建可能真的失败了" >&2; exit 1; }
echo "$found" | jq .

# 返回：
# {
#   "id": 171431239,  // 数字 ID
#   "discussion_id": "b227720346ef5a37e410d17e76a1c1e1f60d5d80",
#   "comment_type": "diff_comment",
#   "reply_to": null        // 非 null 表示这条是回复，值为父评论 ID
# }
```

### 提交普通评论

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/comments?access_token=$GITCODE_API_TOKEN" \
  -d '{
    "body": "评论内容"
  }'
```

### 回复已有评论

v5 API 回复评论需要使用 `discussions` 端点：

```
POST /repos/:owner/:repo/pulls/:number/discussions/:discussion_id/comments
```

**步骤**：

1. **获取 discussion_id**：

```bash
# per_page 默认 20，必须显式传 100；评论数 > 100 时按 page 递增翻页取完（见「获取 PR 讨论列表」）
# 此处只需父评论：回复不能单独被回复，reply[] 里的条目共享父评论的 discussion_id
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/comments?access_token=$GITCODE_API_TOKEN&per_page=100" | \
  jq '.[] | {id: .id, discussion_id: .discussion_id}'
```

2. **回复评论**：

    ```bash
    curl -s -X POST \
      -H "Content-Type: application/json" \
      "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/discussions/<DISCUSSION_ID>/comments?access_token=$GITCODE_API_TOKEN" \
    -d '{
       "body": "回复内容"
     }'
   ```

3. **验证回复成功**：

    ```bash
    # 回复接口（HTTP 201）只返回 {"id":"<discussion_id>","body":"...","note_id":<评论 ID 整数>}，
    # 不含 comment_type；用 note_id 查单条接口，比对 discussion_id 是否等于父讨论的 discussion_id
    curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/comments/<NOTE_ID>?access_token=$GITCODE_API_TOKEN" | \
    jq '{discussion_id: .discussion_id, comment_type: .comment_type}'
    ```

**参数**：

- `discussion_id`（URL 中）：要回复的讨论 ID
- `body`：回复内容

**注意**：

- 回复不作为独立条目出现在 `/pulls/:number/comments` 列表**顶层**，而是嵌进父评论的 `reply[]`；
  要验证就用回复响应里的 `note_id` 查单条接口
- 单条接口查回复时，`comment_type` 跟随**父讨论**的种类：父讨论是行内评论 → `DiffNote`
  （实测 PR!455 的回复 `194209148`、`194148955`），父讨论是普通评论 → `null`
  （实测 PR!431 的回复 `193411043`/`193411048`/`193706419`）。v5 **没有** `DiscussionNote` 这个取值
- 因此校验「回复是否落在目标讨论」请用 `discussion_id` 是否相等，**不要**用 `comment_type` 匹配

### 删除 PR 评论

当用户需要删除 PR 中的评论时使用此功能。

**重要**：只能删除自己创建的评论，或具有仓库管理权限。

**只删一条**（已知数字 `id`）：

```bash
# ⚠️ curl -s 不加 --fail 时，405/403/404 的退出码仍是 0，只看 $? 会把删失败当成删成功，
#    必须用 -w '%{http_code}' 校验（成功为 200）
# ⚠️ 父评论下还有回复时会返回 405，必须先删掉它的回复
curl -s -o /dev/null -w '%{http_code}\n' -X DELETE \
  "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/comments/<COMMENT_ID>?access_token=$GITCODE_API_TOKEN"
```

**批量清理**（按唯一标记删掉自己发过的测试评论）：必须**先翻完所有页把 ID 取全、再统一删**。
下面这段可直接复制执行，六条不变式（翻页取全 / 非数组守卫 / 展开 `reply[]` / **405 预检** /
回复在前 / 校验状态码）已全部落在代码里，不靠注释提醒：

```bash
# ---- 第 1 步：翻页把所有待删 ID 取全（只读不删），追加进临时文件 ----
# ⚠️ 不能边翻页边删：删除会让列表变短、后续条目**前移**，再取「下一页」拿到的已是挪位后的
#    数据，中间被跳过的条目**永远删不到**（若坚持边翻边删，必须按 total_page **倒序**翻页）
# ⚠️ 不能在循环里覆盖同一个变量、循环外再删：resp 每页都被覆盖，结束后只剩**最后一页**，
#    评论超过 100 条时前面几页的 ID 全丢，而脚本仍 exit 0
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
  # ⚠️ 必须展开 reply[]：自己以「回复」形式发的评论不在顶层，只投影顶层字段永远收不到它的 id
  # ⚠️ (.body // "") 兜住 null（行内 diff 评论、系统评论的 body 可能为 null），
  #    否则 jq 在 startswith 处中途报错退出：崩之前输出的 ID 算数、其后全丢，
  #    而外层循环不检查 jq 退出码，会拿着残缺名单跑完并 exit 0
  # ⚠️ 排序键**只放 reply_to 标志 + id，不要掺 .body**：body 常带换行，sort 是按行排序的，
  #    掺进 body 会把一条评论拆成多行重排——记录数被撑大、带 id 的行被正文碎片淹没，
  #    「回复在前」的顺序根本读不出来（实测同一批 43 条评论：掺 body 输出 475 行，
  #    只留标志+id 输出 43 行）。要看正文另起一条命令用 jq -c 输出（一行一个 JSON 对象）
  echo "$resp" | jq -r '
      .[] | . as $c
      | ({id, body, reply_to: null}, ((.reply // [])[] | {id, body, reply_to: $c.id}))
      | select((.body // "") | startswith("<唯一标记>"))
      | "\(if .reply_to then 1 else 0 end)\t\(.id)"' >> "$ids" \
    || { echo "ERROR: 第 ${page} 页 jq 解析失败，ID 未取全" >&2; exit 1; }
  # ⚠️ 405 预检：**被选中的父评论**若还有**未被选中的存活回复**，删它必然 405。
  #    这类失败在翻页阶段就能算出来，没必要先发 N 次注定失败的 DELETE、再靠 405 反推
  #    （实测：标记只覆盖父评论时，100 个父评论 DELETE **全部** 405、脚本 exit 1，
  #     清理一条都没完成，且 100 行 ERROR 里看不出根因是「筛选条件没覆盖回复」）。
  echo "$resp" | jq -r '
      .[] | select((.body // "") | startswith("<唯一标记>")) as $p
      | ($p.reply // [])[]
      | select(((.body // "") | startswith("<唯一标记>")) | not)
      | "父 \($p.id) 的回复 \(.id) 未被筛选条件命中"' >> "$blocked" \
    || { echo "ERROR: 第 ${page} 页 405 预检 jq 解析失败" >&2; exit 1; }
  count=$(echo "$resp" | jq 'length')
  [ "$count" -lt 100 ] && break   # count 数的是**父评论**条数（回复不计入分页）
  page=$((page + 1))
done

# ---- 第 2 步：删除前的 405 预检，有阻塞项就直接失败、一次 DELETE 都不发 ----
if [ -s "$blocked" ]; then
  echo "ERROR: 以下父评论带「未被筛选条件命中的存活回复」，直接删必然 405：" >&2
  sort -u "$blocked" >&2
  echo "       处置：改筛选条件让它同时命中父评论与其回复，或先单独删掉上面列出的回复 ID。" >&2
  exit 1
fi

# ---- 第 3 步：全部取完后再统一删（sort -r 使「回复在前、父评论在后」）----
# ⚠️ 必须先删回复再删父评论：父评论带回复时 DELETE 返回
#    405 {"error_message":"Can not delete a discussion that has replies."}（实测），
#    照投影原顺序（父在前）删会只删掉回复、留下父评论，清理只完成一半且退出码仍是 0
fail=0
for id in $(sort -r "$ids" | cut -f2); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -X DELETE \
    "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/comments/${id}?access_token=$GITCODE_API_TOKEN")
  [ "$code" = "200" ] || { echo "ERROR: 删除评论 ${id} 失败（http=${code}），清理不完整" >&2; fail=1; }
done
[ "$fail" -eq 0 ] || exit 1
```

> 想先看清单再删（不自动执行）：把上面第 3 步换成 `sort -r "$ids"` 打印即可。
> 完整的坑位说明与响应码表见 `references/gitcode_api.md` →「删除 PR 评论」。

**注意**：DELETE 接口的 `<COMMENT_ID>` 必须是评论列表（`/pulls/<PR_NUMBER>/comments`）
返回的**数字 `id`**（如 `171431239`），`references/gitcode_api.md` 中该 `id` 字段类型为 integer。
而 POST 创建评论时立即返回的**哈希字符串**（如 `b227720346...`）是 `discussion_id`/临时返回标识，
**不能**直接作为删除接口的 `COMMENT_ID`——需先按上文「获取真正的数字 ID」查列表拿到数字 `id`。

**删除顺序（实测）**：父评论下还有回复时删父评论会 `405 Can not delete a discussion that has replies.`；
先删回复（`200`）再删父评论（`200`）才干净。批量清理必须按 `reply_to` 排序（回复在前），
并逐条校验 HTTP 状态码，否则「删了一半」会以退出码 0 静默通过。

详细 API 参数、响应码和权限说明请参考 `references/gitcode_api.md` 的「删除 PR 评论」章节。

### 创建 PR 的正确流程

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

### 创建 PR

**目标分支策略**：

- **默认合入 `develop` 分支**（日常开发、新功能、bugfix）
- 仅当用户明确要求时，才合入其他分支（如 `master`、`release/x.y`）

#### 读取 PR 模板

**重要**：PR 描述必须严格遵循仓库中的模板文件，不要使用硬编码模板。

按以下优先级查找仓库中存在的 PR 模板（取第一个命中的文件）：

```bash
# 按候选路径顺序查找 PR 模板，使用第一个存在的
for tpl in \
  .gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md \
  .gitcode/PULL_REQUEST_TEMPLATE.en-US.md \
  .gitcode/PULL_REQUEST_TEMPLATE.md \
  .gitcode/pull_request_template.md \
  .github/PULL_REQUEST_TEMPLATE.md \
  .github/pull_request_template.md \
  docs/PULL_REQUEST_TEMPLATE.md; do
  if [ -f "$tpl" ]; then echo "使用模板: $tpl"; cat "$tpl"; break; fi
done
```

根据命中的模板结构填充各字段内容。多语言模板共存时（如同时有 zh-CN 与 en-US），
优先使用与 PR 描述语言一致的模板（本仓优先 `zh-CN`）。若仓库无任何模板，
再回退到本 skill 末尾「PR 描述模板」一节的通用结构。

#### 使用 API 创建 PR

```bash
# 注意：endpoint 的 ${owner}/${repo} 必须是 PR 要合入的【目标仓库】；
# 跨 fork 时目标仓库是上游仓库，而非你的 fork
curl -s -X POST "https://api.gitcode.com/api/v5/repos/${target_owner}/${target_repo}/pulls" \
  -H "Authorization: Bearer $GITCODE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "docs: 优化文档描述(#32)",
    "head": "source_owner/source_repo:fix/issue-32-description",
    "base": "develop",
    "body": "按模板填充的PR描述"
  }'
```

**参数说明**：

- `title`: PR 标题（必填）
- `head`: 源分支（必填）。需按场景区分填写：
  - **同仓库 PR**（源分支与目标在同一仓库）：直接用分支名，如 `fix/issue-32-description`。
  - **跨 fork PR**（从 fork 向上游提）：必须用**完整源仓库路径**前缀，
    格式 `source_owner/source_repo:branch`。
    ⚠️ 仅用 `username:branch` 经实测会返回 `400 BAD_REQUEST`；只用纯分支名时
    API 会去目标仓库找该分支并报 `404 Can not find the branch`。
  - endpoint 路径里的 owner/repo 始终用**目标仓库**（跨 fork 时为上游仓库）。
- `base`: 目标分支，**默认为 `develop`**，除非用户明确要求合入其他分支
- `body`: PR 描述（Markdown 格式，按[读取 PR 模板](#读取-pr-模板)查找到的仓库 PR 模板填充）

### PR URL 格式

创建 PR 后，PR 的访问地址格式为（注意是 `/pull/` 而非 `/pulls/`）：

```
https://gitcode.com/${owner}/${repo}/pull/<PR_NUMBER>
```

**示例**：`https://gitcode.com/${owner}/${repo}/pull/1807`

### 后续跟进

- 检查 CI/CD 运行结果
- 回复审查者意见
- 根据反馈更新代码

---

## PR 标题格式

遵循 Conventional Commits 规范：

```
<type>: <描述>(#issue_id)
```

**类型**：

| 类型 | 说明 |
| ------ | ------ |
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

创建 PR 时**必须读取并使用**仓库中的 PR 模板（按[读取 PR 模板](#读取-pr-模板)的候选路径查找）。

**步骤**：

1. 按[读取 PR 模板](#读取-pr-模板)的候选路径查找并读取仓库中存在的 PR 模板文件
2. 根据实际变更填充模板中的各字段（描述、测试项、测试结果等）
3. 将对应选项的 `[ ]` 改为 `[x]` 勾选
4. Checklist 默认全部勾选 `[x]`
5. 将填充后的模板内容作为 `body` 参数传入创建 PR 的 API

**不要使用硬编码的模板内容**，始终以仓库中实际存在的模板文件为准；
若仓库无任何模板，再回退到通用结构。

---

## PR 代码审查

当用户说"检视 PR"、"审查 PR"、"review PR"、"给 PR 提意见"时触发。

### 执行步骤

1. **读取 `commands/review.md`** —— 使用 Read 工具获取完整审查流程（本 skill 目录下的 `commands/review.md`）。

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

**注意**：

- `commands/review.md` 中包含每个步骤的详细说明和 API 示例。
- 其中的评论获取/发布 API 须遵循本文档前述的 GitCode v5 约定
  （`/repos/${owner}/${repo}/...` + `access_token`；评论定位用数字 `id` 等）。

---

## Resources

### references/gitcode_api.md

GitCode API 完整参考文档，用于：

- 了解 API 参数格式
- 查看响应结构
- 排查 API 调用问题
