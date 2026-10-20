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

检查环境变量（**别用裸 `echo $GITCODE_API_TOKEN`**：为空时只打印一个空行、退出码仍是 `0`，
看不出「未设置」与「已设置」的区别；也不要打印值本身，长期凭证会进 shell history / CI 日志）：

```bash
# 只看长度，不看值
echo "GITCODE_API_TOKEN 长度 = ${#GITCODE_API_TOKEN}"
# 长度为 0 不等于必须停止：读 PR 详情 / 读评论列表匿名也能跑，但查仓库信息（fork 判定）、
# 取 MY_LOGIN、发评论 / 删评论匿名必然失败。所以别在入口加「无 token 就 exit 1」的硬守卫；
# 该守的是：GET /repos/<owner>/<repo> 的响应形状、GET /user 的状态码、写操作的状态码。
# 匿名调用共用服务端额度桶（会撞 429，何时恢复不由你决定），「能匿名跑」≠「能连着匿名跑」。
# 各步匿名时的实际状态码 / 响应体见 references/gitcode_api.md →「匿名（无 token / 空 token）时各步骤的真实行为」。
```

如果长度为 0 且你要执行写操作（或步骤 1 判断「是否已评论过」要先取到自己的 login），提示用户获取令牌：

1. 登录 [GitCode](https://gitcode.com)
2. 点击头像 → 设置 → 访问令牌
3. 创建新令牌，勾上**仓库读写 + PR/评论读写 + 用户信息读取**相关的权限项
   > ⚠️ **权限项的确切名字以令牌页实际显示的勾选项为准，不要照抄本行**：官方文档从不枚举 PAT 的
   > scope 名，文档里并存两套互不映射的词汇，服务端报错串用的又是第三种（`no scopes:<name>`）。
   > 缺陷清单见 [references/gitcode_api.md](references/gitcode_api.md) →「官方文档已知缺陷（先读）」。
   >
   > ⚠️ **建好令牌后先探测一次 `GET /api/v5/user`**（取当前 token 对应的账号 login，用于判断
   > 「本 agent 是否已评论过」）：它的 scope 前提官方文档不记载，权限不够只在运行时暴露，
   > 且**只影响这一步**（仓库/评论读写接口仍正常）。探测法与失败分支见
   > [references/gitcode_api.md](references/gitcode_api.md) →「获取当前用户（token 对应的账号）」；
   > 失败时按响应体里的 `no scopes:<name>` 去补权限（名字以响应体为准，不要背）；**无法获取权限：
   > 停止审查，不得提供绕过开关**，也不要用环境变量伪造 login（伪造值会让「是否已评论过」的判断
   > 失效，可能重复刷屏评论）。
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

- **API 版本纪律：一律使用 v5**（官方文档《OpenAPI 使用简介》：「当前可用的版本为 `/api/v5`」），
  本 skill 的全部读写调用只用 `/api/v5/...`，下文所有端点示例均为 v5。
- GitCode API v5 端点格式：`https://api.gitcode.com/api/v5/repos/${owner}/${repo}/...`

#### 查询 Fork 的原仓库（当当前仓库是 fork 时）

当当前仓库是 fork 仓库时，需要查询其 fork 的原仓库作为 PR 目标仓库：

```bash
# 查询仓库信息获取 fork 来源
# ⚠️ 这个端点**匿名必然失败**（无 token / token 失效都走进下面的 ERROR 分支）。ERROR 与空输出都必须
#    exit 1、不能只打印：jq 管道自身的退出码是 0，只打印就等于把「不知道是不是 fork」
#    伪装成「查过了、不是 fork」。
fork_info=$(curl -s --connect-timeout 10 --max-time 60 -H "Authorization: Bearer $GITCODE_API_TOKEN" \
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
  ')
echo "$fork_info"
# `""` 也要一起当失败：curl 超时 / DNS 不通 / 代理返回 HTML 时 jq 拿到空输入或非 JSON，
# 什么都不输出且退出码仍是 0，只匹配 `ERROR:*` 的写法会静默穿过去。
case $fork_info in
  ERROR:*|"") echo "错误：无法判定当前仓库是否为 fork（${fork_info:-响应为空：超时 / DNS / 非 JSON}），停止该步（把原样响应报给用户，别猜）" >&2; exit 1 ;;
esac
```

> ⚠️ 必须显式区分「响应异常」和「不是 fork」：`if .fork then … else …` 里 `null` 也是假值，
> token 失效时返回的 `{"message":"401 Unauthorized"}` 会走进 else，被读成「这不是 fork 仓库」，
> 于是 PR 直接建到自己的 fork 上而不是上游仓库。
> ⚠️ 原仓库网页地址的键名是 **`web_url`**，不是 `html_url`——v5 的 `repos` 响应里**没有**
> `html_url` 这个键，取错键不报错，只会静默输出空串。

**响应关键字段**：

| 字段 | 说明 |
| ------ | ------ |
| `fork` | 是否为 fork 仓库（`true`/`false`） |
| `parent.full_name` | 原仓库完整名称（格式：`owner/repo`） |
| `parent.web_url` | 原仓库网页地址（**不是** `html_url`，该键不存在） |
| `parent.default_branch` | 原仓库默认分支 |

### 获取 PR 评论和讨论

#### 获取 PR 讨论列表（包含行内评论）

GitCode 使用 v5 API 获取 PR 评论：

```bash
# 翻页取全所有评论（含行内评论），并展开每条父评论的 reply[]
hdr=$(mktemp) || exit 1
trap 'rm -f "$hdr"' EXIT   # 嵌进更大脚本时，把这个 rm 合并进已有的 trap，别覆盖它
MAX_PAGES=50   # 空转上界（50×100=5000 条父评论，远超真实 PR）；撞上界 exit 1，不 break 后以 0 退出
page=1; sum=0
while :; do
  [ "$page" -gt "$MAX_PAGES" ] && { echo "ERROR: 翻页超过 ${MAX_PAGES} 页上界，结果不可信" >&2; exit 1; }
  # -D 落盘才读得到响应头（curl -s 会丢掉）；头是 CRLF 结尾，取值前必须 tr -d '\r'
  resp=$(curl -s --connect-timeout 10 --max-time 60 -D "$hdr" "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/comments?access_token=$GITCODE_API_TOKEN&per_page=100&page=${page}")
  total_page=$(tr -d '\r' < "$hdr" | awk -F': *' 'tolower($1)=="total_page"{print $2}')
  total_count=$(tr -d '\r' < "$hdr" | awk -F': *' 'tolower($1)=="total_count"{print $2}')
  http_status=$(head -n1 "$hdr" | awk '{print $2}' | tr -d '\r'); http_status=${http_status:-000}
  # 失败一律 exit 1，并把状态码与响应体一起打（只打响应体分不清 401/403/404/429/空）
  [ "$(echo "$resp" | jq -r 'type' 2>/dev/null)" = "array" ] || {
    echo "ERROR: 第 ${page} 页评论请求失败或响应非数组（http=${http_status}），结果不完整：$resp" >&2; exit 1; }
  [ -n "$total_page" ] && [ -n "$total_count" ] || {
    echo "ERROR: 第 ${page} 页响应缺 total_page/total_count 头（http=${http_status}），无法判断是否取全" >&2; exit 1; }
  # 每条父评论原样输出，再把它 reply[] 里的回复展开成同形对象（reply_to 标记来源）
  echo "$resp" | jq -c '.[] | {id, discussion_id, comment_type, author: .user.login, body} as $c
    | $c, (.reply[]? | {id, discussion_id: $c.discussion_id, comment_type: $c.comment_type,
                        author: .user.login, body, reply_to: $c.id})'
  count=$(echo "$resp" | jq 'length')   # 父评论条数；回复不计入分页，也不计入 total_count
  [ "$count" -eq 0 ] && [ "$page" -lt "$total_page" ] && {
    echo "ERROR: 第 ${page} 页为空但 total_page=${total_page}（http=${http_status}），响应自相矛盾，评论未取全" >&2; exit 1; }
  sum=$((sum + count))
  [ "$page" -ge "$total_page" ] && break
  page=$((page + 1))
done
# Σ父评论条数必须等于 total_count，对不上就是没取全；不自愈（不自动重跑、不打 WARN 后继续）
[ "$sum" -eq "$total_count" ] || {
  echo "ERROR: 评论未取全 —— Σ父评论条数=${sum} ≠ total_count=${total_count}（翻了 ${page} 页），不要据本次列表做任何判断" >&2
  echo "       Σ 偏小 = 服务端漏数据，或翻页期间有人新增了评论；Σ 偏大 = 翻页期间有评论被删、列表前移导致同一条被读两次" >&2
  echo "       重跑一次：仍不等就把上面数字原样上报，别猜" >&2
  exit 1; }
```

> **翻页规则（少一条就会静默少读）**：
> - `per_page` 默认 **20**、最大 **100**，请求值超上限会被**静默夹回**，必须显式传 `100`；
> - 终止判据**只用响应头 `total_page`**，且每页重读（它是本次请求的快照）；**不要**用
>   「本页条数 < 请求的 `per_page`」——夹回后该判据第一页就 `break`，后面的评论全部静默丢掉。
>   越界 `page` 返回**空数组 + `200`**，所以合法的「本页为空」必然伴随 `page >= total_page`；
> - 该接口**不返回** `Link` 头，别按 GitHub 的 `rel="next"` 翻页；单条资源接口（`GET /pulls/<n>`）
>   没有这两个分页头，只有列表接口才有；
> - `curl -s` 对 4xx / 网络失败的退出码**仍是 `0`**：响应非数组（无效 token 返回
>   `{"message":"401 Unauthorized"}`、网络失败返回空）、缺分页头、体空但 `page < total_page`、
>   Σ ≠ `total_count` 这四种都必须 `exit 1`，不要退回「本页为空就算翻完」这类猜测判据；
>
> **顶层数组不含回复**：回复嵌在父评论的 `reply[]` 里，必须一并展开——否则「只在既有讨论里回复过」
> 完全不可见，判断「是否已评论过」会漏判并重复发意见，给 `DELETE` 收集 ID 会漏删。回复对象只有
> `id`/`body`/`user`/时间戳，`discussion_id` 与 `comment_type` 从父评论带下来（带下来的是**列表接口**
> 那套取值，单条接口查同一条回复会得到另一套，别混用）；`resolved` 是讨论级属性、以父评论为准，
> 不要给回复伪造取值。展开后用 `reply_to` 区分：`null` = 父评论，非 null = 回复（值为父评论 ID）。
> 校验「是否取全」拿**父评论**条数比，拿展开后的输出条数比必然偏大、会误判成取重了。
>
> 评论列表**获取不全：停止审查，不得判断为「未评论过」**（也不得据此决定「该删哪条」）。
> 单页字段形状、`comment_type` 两套取值的对照表见
> `references/gitcode_api.md` →「获取 PR 讨论（包含行内评论）」。

## GitCode API v5 说明

- **认证头**：`?access_token=`、`Authorization: Bearer`、`PRIVATE-TOKEN` 三种写法均可用；
  `Authorization: token`（官方流水线页的示范写法）不适用 v5 REST，两个域名上都返回 `401`。
  共享环境或 CI 里优先用请求头——query 形式会把长期凭证带进 shell history / CI 日志 / 服务端 access log。
- **域名**：本 skill 一律用 `https://api.gitcode.com`（`scripts/create_pr.py` 里的
  `https://gitcode.com/api/v5` 是同一套 v5 后端，不是错误）。`atomgit.com` 也是同一套后端，
  但**同一份脚本不要混用域名**：两者在需要凭证的端点上错误响应形状不同，混用会把「凭证没送达」
  误判成「缺权限」。
- **超时**：所有 `curl` 必须带 `--connect-timeout 10 --max-time 60`（小请求可 30），
  否则网络挂起会无限等待；超时后 `%{http_code}` 是 `000`，走既有失败分支，**不要**当成权限问题。
- **判成功只看 HTTP 状态码或校验响应里有没有期望的键**：`curl -s` 的退出码不作数（恒为 `0`）。

域名对比表、匿名时各步骤的真实行为、认证写法的自证矩阵、官方文档在认证上的缺陷清单见
[references/gitcode_api.md](references/gitcode_api.md) →「认证方式」「官方文档已知缺陷（先读）」。

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
| `reply[]` | 该讨论下的回复，**嵌在父评论里**，不单独出现在顶层数组；**无回复时是 `null` 而不是 `[]`**，投影必须写 `.reply[]?` |

**示例返回结构**（下方 ID / 哈希为示意值，别当契约）：

```json
[
  {
    "id": 123456789,
    "discussion_id": "<40 位十六进制哈希串>",
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
  "discussion_id": "<40 位十六进制哈希串>",
  "body": "这段代码需要优化",
  "comment_type": "diff_comment",
  "resolved": false,
  "diff_position": {
    "start_new_line": 60,
    "end_new_line": 60,
    "position_type": "text"
  },
  "reply": null
}
```

**重要说明**（条数随评论增长变化，不在此固化）：

- 列表接口**没有** `path` / `position` 这两个键（`has("path")` 与 `has("position")` 均为 `false`，
  不是「值为 `null`」）。用 `jq '.path'` 取值会得到 `null`，但**不要按这两个字段过滤**，永远命中不了。
- **行号列表接口就能拿到**：在 `diff_position.start_new_line` / `end_new_line` 里
  （`diff_comment` 都带此键，单行评论两个值相同），不必逐条调单条接口。
  ⚠️ 但**行号数值本身易变**：接口按 PR head **当前**指向的 commit 重算，锚定的还是同一行源码，
  行号却会随该文件上方的增删位移。所以这里只固化「哪些键存在」这类结构事实，
  **不要把具体行号当稳定结论引用或写进 PR 描述**，要行号就现场查
  （漂移机理见 `references/gitcode_api.md` →「获取 PR 讨论（包含行内评论）」）。
- **只有「被评论的文件路径」列表接口拿不到**，需要查单条评论接口的 `position.new_path`。
- `comment_type` 只有 `diff_comment`（行内）/ `pr_comment`（普通）两种取值。

#### 获取单条评论详情（包含完整位置信息）

**端点**：

```
GET /repos/:owner/:repo/pulls/comments/:id
```

**示例**：

```bash
curl -s --connect-timeout 10 --max-time 60 "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/comments/<COMMENT_ID>?access_token=$GITCODE_API_TOKEN"
```

**返回字段**（下方 ID / 哈希 / 行号为示意值，别当契约）：

> 下面 JSON 是**某一时刻的快照**：`position.head_sha` / `new_line` 会随 PR head 推进而漂移
> （见下方 ⚠️），别把里面的具体行号、head sha 当稳定事实引用；**键名与键的存在性**才是要看的。

行内评论（`comment_type` = `DiffNote`）：

```json
{
  "id": 123456789,
  "discussion_id": "<40 位十六进制哈希串>",
  "body": "评论内容",
  "comment_type": "DiffNote",
  "user": { "id": 1, "object_id": "…", "login": "username", "name": "显示名称", "type": "…" },
  "created_at": "2026-05-14T16:06:24+08:00",
  "updated_at": "2026-05-14T16:06:24+08:00",
  "is_outdated": false,
  "position": {
    "base_sha": "<40 位 commit sha>",
    "start_sha": "<40 位 commit sha>",
    "head_sha": "<40 位 commit sha>",
    "old_path": "<被评论文件路径>",
    "new_path": "<被评论文件路径>",
    "position_type": "text",
    "new_line": 60
  }
}
```

普通评论：

```json
{
  "id": 123456789,
  "discussion_id": "…",
  "body": "评论内容",
  "user": { "id": 1, "object_id": "…", "login": "username", "name": "显示名称", "type": "…" },
  "created_at": "2026-05-14T15:25:32+08:00",
  "updated_at": "2026-05-14T15:25:32+08:00"
}
```

> ⚠️ **单条接口的响应形状（容易踩坑）**：
>
> - `user` 是**对象**（含 `login`/`name`/`id`/`object_id`/`type`），**不是用户名字符串**；
>   取用户名用 `.user.login`。
> - 单条接口**没有** `path` 键（`has("path")` 为 `false`），文件路径只在 `position.new_path` 里。
> - 普通评论的响应里**根本没有** `comment_type` / `position` / `is_outdated` 这三个键
>   （`jq '.comment_type'` 得到 `null` 是因为键不存在，不是值为 `null`）；
>   行内评论三个键都在。
> - 单条接口**没有**列表接口才有的 `resolved` / `resolved_by` / `diff_position` / `reply` 键；
>   要看检视意见是否已解决、要拿讨论下的回复，必须走列表接口。
> - `position.head_sha` / `new_line` 是**按 PR head 当前指向的 commit 重算的快照值**，head 推进、
>   该文件上方有增删时行号就会漂移；`base_sha` / `start_sha` 才是稳定的。所以上面 JSON 里的
>   行号、head sha 只作形状示范，**不要当稳定事实引用**，要行号就现场查
>   （漂移机理见 `references/gitcode_api.md` →「获取 PR 讨论（包含行内评论）」）。

> ⚠️ **两个接口的 `comment_type` 不是同一套取值**：列表接口给 `diff_comment`/`pr_comment`，
> 单条接口给 `DiffNote`/`null`。**判断行内/普通一律用列表接口的值**，单条接口只能当
> 「`comment_type` 非 `null` 即行内」用，别拿一套的取值去匹配另一套的响应。
> 对照表见 `references/gitcode_api.md` →「两个接口的 `comment_type` 取值不是同一套」。

**关键位置字段（单条接口 `/pulls/comments/:id` 专有）**：

| 字段 | 说明 |
| ------ | ------ |
| `position.new_path` | 文件路径 |
| `position.new_line` | **结束行**行号（新代码） |
| `position.start_new_line` | **起始行**行号（新代码）；**单行评论没有此键** |
| `position.position_type` | `text` = 代码行评论（实测）；`binary` = 文件级评论（官方文档值，**未实测**） |
| `position.old_path` | 文件路径（重命名前） |
| `position.base_sha` | Base 提交 SHA |
| `position.head_sha` | Head 提交 SHA |

> ⚠️ 行内评论的 `position` 键**只有**
> `base_sha`/`start_sha`/`head_sha`/`old_path`/`new_path`/`position_type`/`new_line`，
> **没有 `old_line`**（多行评论额外多一个 `start_new_line`）。所以旧代码行号取不到，
> 不要用 `position.old_line` 做判断或过滤。

> ⚠️ 上表的 `position.*` 是**单条接口**的字段名，与列表接口的 `diff_position.*` **不是同一套键名**：
> **行号两个接口都能拿到**（列表叫 `end_new_line`、单条叫 `new_line`，别串用），
> **文件路径与 SHA 只有单条接口有**。两接口键名逐项对照表与样例见
> `references/gitcode_api.md` →「获取 PR 讨论（包含行内评论）」。

**接口对比**：

| 接口 | 位置信息 | comment_type | 用途 |
| ------ | --------- | ------------- | ------ |
| `/pulls/:number/comments` | `diff_position.start_new_line`/`end_new_line`（**有行号、无文件路径**；无 `path`/`position` 键） | `diff_comment`（行内）/ `pr_comment`（普通） | 获取评论列表 |
| `/pulls/comments/:id` | 完整 `position` 对象（**含 `new_path` 文件路径**） | `DiffNote`（行内）/ `null`（普通） | 获取单条评论详情 |

### 获取 PR 文件变更

```bash
# 使用 v5 API（files.json）
curl -s --connect-timeout 10 --max-time 60 "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/files.json?access_token=$GITCODE_API_TOKEN"

# 解析第一个文件路径
curl -s --connect-timeout 10 --max-time 60 "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/files.json?access_token=$GITCODE_API_TOKEN" | \
  jq -r '.diffs[0].statistic.new_path'
```

**响应字段说明**：

顶层是**对象**（不是数组），键为 `code` / `added_lines` / `remove_lines` / `count` / `diff_refs` / `diffs`：

- `code` - 状态码，成功为 `0`
- `count` - 变更文件数
- `added_lines` / `remove_lines` - 整个 PR 的新增 / 删除行数合计
- `diff_refs.base_sha` / `start_sha` / `head_sha` - 三个基准 SHA（发行内评论要用）
- `diffs[]` - 每个变更文件一项，键为 `new_blob_id` / `statistic` / `head` / `added_lines` / `remove_lines` / `content`
- `diffs[].statistic.new_path` / `old_path` - 新 / 旧文件路径（**同一路径时两者相同**）
- `diffs[].statistic.path` - 文件路径（与 `new_path` 同值，取哪个都行）
- `diffs[].statistic.type` - 变更类型，如 `text_type`
- `diffs[].statistic.view` - 布尔值（文本文件为 `false`），用途未文档化，不要依赖
- `diffs[].new_blob_id` - 变更后文件的 blob sha（等于 contents API 的 `sha`，
  可拼成稳定的 `…/blobs/<new_blob_id>/<path>` 取内容地址）
- `diffs[].head` - **对象**，键为 `commit_id`（= `diff_refs.head_sha`）与 `url`；
  ⚠️ `url` 是 `…/raw/<commit_sha>/<path>` 形式，**不稳定**（间歇 `403`「暂不支持预览」），
  别拿它当取内容的地址，改用 contents API 或 `…/blobs/<new_blob_id>/<path>`
- `diffs[].content` - ⚠️ **不是 diff 文本，而是对象** `{"text":[…]}`；`text[]` 每项形如
  `{"line_content":"<diff 行原文>","old_line":"…","new_line":"…","type":"match"}`
  （diff 较大的文件即为此形状；**项数随该文件 diff 大小变化，不在此固化**）。
  要拿可直接 `grep` 的 unified diff 文本，
  用 `/pulls/<n>/files` 的 `patch.diff`；非要从本接口拼，得
  `jq -r '.diffs[0].content.text[].line_content'`

> 另有 `/pulls/<n>/files`（**返回数组**，每项含 `filename`/`additions`/`deletions`/`blob_id`/`patch.diff` 等）；
> 要**整体统计与 `diff_refs`** 用 `files.json`，要**逐文件 patch / blob sha** 用 `files`。
> 两者字段名不同（如 `diffs[].content` vs `patch.diff`），别把一套 jq 表达式套到另一个接口上。
>
> ⚠️ 两个接口的增删行数与本地 `git diff --numstat` **可能差 1**。不影响判断，但别拿它做精确对账，
> 也别把这类数字写进 PR 描述（易变）。细节见 `references/gitcode_api.md` →「获取 PR 文件变更」。

### 提交行内评论（支持多行选择）

#### 使用 v5 API

```bash
curl -s --connect-timeout 10 --max-time 60 -X POST \
  -H "Content-Type: application/json" \
  "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/comments?access_token=$GITCODE_API_TOKEN" \
  -d '{
    "body": "评论内容",
    "path": "文件相对路径",
    "position": <结束行号>,
    "start_position": <起始行号>
  }'
```

> `body`/`path`/`position` 必填，多行时再加 `start_position`（= 起始行，`position` = 结束行）。
> 完整参数表（另有 `position_type`、query 里的 `access_token`）与成功响应形态见
> `references/gitcode_api.md` →「提交行内评论（支持多行选择）」。

**单行评论示例**：

```bash
curl -s --connect-timeout 10 --max-time 60 -X POST \
  -H "Content-Type: application/json" \
  "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/comments?access_token=$GITCODE_API_TOKEN" \
  -d '{
    "body": "第70行需要重构",
    "path": "runtime/v1/opskernel_executor/ops_kernel_executor_manager.cc",
    "position": 70
  }'
```

**多行评论示例**：

```bash
curl -s --connect-timeout 10 --max-time 60 -X POST \
  -H "Content-Type: application/json" \
  "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/comments?access_token=$GITCODE_API_TOKEN" \
  -d '{
    "body": "第70-75行需要优化",
    "path": "runtime/v1/opskernel_executor/ops_kernel_executor_manager.cc",
    "position": 75,
    "start_position": 70
  }'
```

**注意**：`start_position` **未出现在官方文档的请求体 schema 里**，但**实际有效**。
取值与读回结果见 `references/gitcode_api.md` →「多行选择说明」。

#### 创建评论的返回值

**重要**：POST 创建评论返回 **HTTP 201**，响应体是（与「回复 discussion」接口同形）：

```json
{"id": "<discussion_id 哈希串>", "body": "评论内容", "note_id": 123456789}
```

- `id` 是**讨论 ID**（哈希串），用于回复该讨论，**不能**用于删除；
- `note_id` 就是**数字评论 ID**，`DELETE /pulls/comments/<note_id>` 直接用它，
  **不必先翻评论列表反查**；
- 拿哈希串去 DELETE 会得到
  `400 {"error_code_name":"UN_KNOW","error_message":"Orchestration error: Invalid path parameter: note_id, number required"}`；
- 响应里**没有** `comment_type` / `created_at` / `discussion_id` 这些键，别按它们校验。

**回读校验**（`201` 不代表落点正确：行号错位 / `path` 写错会静默落到别处）：用列表接口按
`note_id` 精确比对 `.id`。**不要**用 `contains(.body)` 做宽匹配——PR 中已有含相同文本的评论时
会匹配到错误的 ID，导致误操作他人评论；也**不要**拿 POST 响应里的哈希串 `id`（那是
`discussion_id`）去反查：`reply[]` 展开后父评论与其回复继承同一个 `discussion_id`，一个哈希会命中多条。

```bash
# 回读校验：复用上面「获取 PR 讨论列表」的翻页循环，只把每页的 jq 换成下面这段「命中即停」
NOTE_ID="<NOTE_ID>"   # POST 响应里的 note_id（数字评论 ID）
# NOTE_ID 必须是整数：占位符没替换时 jq --argjson 会以 exit 2 报错，被循环吞掉后只剩
# 「翻完所有页仍未找到」，看不出根因，所以先自己挡一道（MY_LOGIN 同理）。
case "$NOTE_ID" in ''|*[!0-9]*) echo "ERROR: NOTE_ID 必须是整数，当前=\"$NOTE_ID\"（占位符未替换？）" >&2; exit 1;; esac
#   found=$(echo "$resp" | jq -c --argjson nid "$NOTE_ID" --arg me "$MY_LOGIN" '
#       .[] | {id, discussion_id, comment_type, author: .user.login, body} as $c
#       | $c, (.reply[]? | {id, discussion_id: $c.discussion_id, comment_type: $c.comment_type,
#                           author: .user.login, body, reply_to: $c.id})
#       | select(.id == $nid or (.author == $me and .body == "<唯一标记内容>"))
#       | {id, discussion_id, comment_type, reply_to}')
#   [ -n "$found" ] && break
#   [ -n "$found" ] || { echo "ERROR: 翻完所有页仍未找到该评论，创建可能真的失败了" >&2; exit 1; }
```

> - **同样必须 `per_page=100` 并翻页**：目标在第 20 条之后时不带 `per_page` 查不到、在第 100 条之后
>   时不翻页查不到，两种都会把「创建成功」误判成「创建失败」。
> - **这是「找到目标」不是「取全列表」，Σ == `total_count` 那条校验不适用**：命中即停时 Σ 必然偏小，
>   照搬会把「找到了」误报成「评论未取全」。其余守卫（非数组、缺分页头、体空但 `page < total_page`、
>   `MAX_PAGES`）都要保留——「没找到」这个结论的可靠性由它们保证。
> - **`select` 里的两支是并集（`or`）**，不是「任选其一」：只填你手上真有的那一支，
>   有 `note_id` 就别再填标记文本。
> - **走「作者 + 唯一标记」这条备选判据时才用到 `MY_LOGIN`**（备选 ≠ 兜底，它不掩盖失败）：
>   未定义时 `--arg me ""` 不会报错，只会让 `.author == $me` 永远不成立、判据静默失效。
>   取法见 `references/gitcode_api.md` →「获取当前用户（token 对应的账号）」，取不到就 `exit 1` 原样上报。
> - **找不到必须显式失败**：空结果 + `exit 0` 会被读成「查过了、确实没有」。
>
> 命中输出形如 `{"id":<NOTE_ID>,"discussion_id":"<哈希串>","comment_type":"diff_comment","reply_to":null}`；
> `reply_to` 非 null 表示命中的是回复（值为父评论 ID），找回复也走同一段代码，不必再分支。

### 提交普通评论

```bash
curl -s --connect-timeout 10 --max-time 60 -X POST \
  -H "Content-Type: application/json" \
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
# per_page 默认 20、最大 100（请求值超过 100 会被静默夹回 100），必须显式传；
# 评论数 > 100 时按 page 递增翻页取完、以响应头 total_page 为终止判据（见「获取 PR 讨论列表」）
# 此处只需父评论：回复不能单独被回复，reply[] 里的条目共享父评论的 discussion_id
curl -s --connect-timeout 10 --max-time 60 "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/comments?access_token=$GITCODE_API_TOKEN&per_page=100" | \
  jq '.[] | {id: .id, discussion_id: .discussion_id}'
```

2. **回复评论**：

    ```bash
    curl -s --connect-timeout 10 --max-time 60 -X POST \
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
    curl -s --connect-timeout 10 --max-time 60 "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/comments/<NOTE_ID>?access_token=$GITCODE_API_TOKEN" | \
    jq '{discussion_id: .discussion_id, comment_type: .comment_type}'
    ```

**参数**：

- `discussion_id`（URL 中）：要回复的讨论 ID
- `body`：回复内容

**注意**：

- 回复不作为独立条目出现在 `/pulls/:number/comments` 列表**顶层**，而是嵌进父评论的 `reply[]`；
  要验证就用回复响应里的 `note_id` 查单条接口
- 校验「回复是否落在目标讨论」请用 `discussion_id` 是否相等，**不要**用 `comment_type` 匹配
  （单条接口查回复时 `comment_type` 跟随父讨论种类、与列表接口不是同一套取值，对照表见
  `references/gitcode_api.md` →「两个接口的 `comment_type` 取值不是同一套」）

### 删除 PR 评论

当用户需要删除 PR 中的评论时使用此功能。

**重要**：只能删除自己创建的评论，或具有仓库管理权限。

**只删一条**（已知数字 `id`）：

```bash
# ⚠️ 必须用 -w '%{http_code}' 校验（成功为 200）：curl -s 的退出码对 4xx/5xx 不作数
# ⚠️ 父评论下还有回复时会返回 405，必须先删掉它的回复
curl -s --connect-timeout 10 --max-time 60 -o /dev/null -w '%{http_code}\n' -X DELETE \
  "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/comments/<COMMENT_ID>?access_token=$GITCODE_API_TOKEN"
```

**注意**：DELETE 接口的 `<COMMENT_ID>` 必须是**数字评论 ID**（整数，如 `123456789`）；
两个来源等价、任选其一：POST 响应里的 **`note_id`**（拿到即可直接删，不必再翻列表反查），
或评论列表 `/pulls/<PR_NUMBER>/comments` 里的 **`id`**。
POST 响应里的**哈希串 `id`** 是 `discussion_id`，**不能**用来删——两种 ID 的区别与 `400`
响应原文见上文[创建评论的返回值](#创建评论的返回值)。

详细 API 参数、响应码原文和权限说明请参考 `references/gitcode_api.md` 的「删除 PR 评论」章节。

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

- **默认合入目标仓库的 `default_branch`**，不要硬编码某个分支名。该值从仓库信息接口读：

  ```bash
  # 跨 fork 时 target_owner/target_repo 是上游仓库（由 parent.full_name 得到）
  curl -s --connect-timeout 10 --max-time 60 "https://api.gitcode.com/api/v5/repos/${target_owner}/${target_repo}?access_token=$GITCODE_API_TOKEN" | \
    jq -r '.default_branch'
  ```

  > ⚠️ **各仓库的默认分支并不统一**，写死分支名会让建 PR 直接失败：`base` 传了不存在的分支
  > 会得到 `404 Can not find the branch`。一律用上面的接口读，不要硬编码。
- 仅当用户明确要求时，才合入其他分支（如 `release/x.y`、某特性分支）

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
优先使用与 PR 描述语言一致的模板（本仓优先 `zh-CN`）。填充要点：按实际变更写描述 /
测试项 / 测试结果，选项 `[ ]` 改为 `[x]`（Checklist 默认全部勾选），填充结果作为
创建 PR 的 `body` 传入。若仓库无任何模板，再回退到通用结构
（变更说明 / 测试项 / 测试结果 / Checklist）。

#### 使用 API 创建 PR

```bash
# 注意：endpoint 的 ${owner}/${repo} 必须是 PR 要合入的【目标仓库】；
# 跨 fork 时目标仓库是上游仓库，而非你的 fork
curl -s --connect-timeout 10 --max-time 60 -X POST "https://api.gitcode.com/api/v5/repos/${target_owner}/${target_repo}/pulls" \
  -H "Authorization: Bearer $GITCODE_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "docs: 优化文档描述(#32)",
    "head": "source_owner/source_repo:fix/issue-32-description",
    "base": "<目标仓库的 default_branch，按上文读取>",
    "body": "按模板填充的PR描述"
  }'
```

**参数说明**：

- `title`: PR 标题（必填）
- `head`: 源分支（必填）。需按场景区分填写：
  - **同仓库 PR**（源分支与目标在同一仓库）：直接用分支名，如 `fix/issue-32-description`。
  - **跨 fork PR**（从 fork 向上游提）：必须用**完整源仓库路径**前缀，
    格式 `source_owner/source_repo:branch`。
    ⚠️ 仅用 `username:branch` 会返回 `400 BAD_REQUEST`；只用纯分支名时
    API 会去目标仓库找该分支并报 `404 Can not find the branch`。
  - endpoint 路径里的 owner/repo 始终用**目标仓库**（跨 fork 时为上游仓库）。
- `base`: 目标分支，**默认取目标仓库的 `default_branch`**（见[目标分支策略](#创建-pr)，用接口读、
  不要硬编码），除非用户明确要求合入其他分支
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

## PR 代码审查

当用户说"检视 PR"、"审查 PR"、"review PR"、"给 PR 提意见"时触发。

### 执行步骤

1. **读取 `commands/review.md`** —— 使用 Read 工具获取完整审查流程（本 skill 目录下的 `commands/review.md`）。

2. **严格按 `commands/review.md` 的步骤编号顺序执行**：该文件的步骤编号即执行顺序
   （含 `3.1` 这类子步骤），是否发布评论由 `--comment` 参数决定（见其步骤 7）。

**注意**：`commands/review.md` 中的评论获取/发布 API 须遵循本文档前述的 GitCode v5 约定
（`/repos/${owner}/${repo}/...` + `access_token`；评论定位用数字 `id` 等）。
