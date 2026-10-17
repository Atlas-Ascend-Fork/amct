---
description: 审查 GitCode Pull Request 的代码变更
# ⚠️ allowed-tools 必须覆盖正文里所有会被复制执行的命令：步骤 3.1「方法 2」用 contents API 取
#    文件内容时要管道给 base64 -d（解码）或 xargs -r curl（跟 download_url），漏授权会让
#    推荐路径被工具白名单拦住、agent 只能退回去用不可靠的 raw/<commit>/<path> 形式
allowed-tools: Bash(curl:*), Bash(jq:*), Bash(grep:*), Bash(base64:*), Bash(xargs:*), Bash(git diff:*), Bash(git log:*), Bash(git show:*)
---

# GitCode PR 代码审查

对指定的 Pull Request 进行代码审查。

**Agent 假设（适用于所有 agent 和子 agent）：**
- 所有工具都能正常工作，不会出错。不要测试工具或进行探索性调用。确保每个启动的子 agent 都清楚这一点。
- 只有在完成任务需要时才调用工具。每次工具调用都应有明确目的。

## 审查流程

请严格按照以下步骤执行：

### 步骤 1: 前置检查

检查以下条件是否成立：
- PR 已关闭
- PR 是草稿状态
- PR 不需要代码审查（如：自动化 PR、明显正确的微小变更）
- **本 agent 自己**已经在此 PR 上评论过（通过 GitCode API 检查 PR 评论，**必须翻完所有分页**，
  见「GitCode API 参考 → 基本格式」，否则会漏读第 100 条之后的评论而误判为未评论过；
  且**必须展开父评论的 `reply[]`**——回复不出现在列表顶层，只在既有讨论里回复过的活动
  不展开就完全看不到，同样会被误判为「未评论过」而重复发布审查意见）

> ⚠️ **「是否已评论过」按 `user.login` 比对，不要按 agent 品牌名比对**：执行本 skill 的可能是
> Codex、amct-coder、Claude 等任意 agent，正文里不写死品牌名。判断方法是拿当前 token 对应的
> 账号 login（`GET /api/v5/user` 的 `.login`）去比评论的 `author`，命中即视为「已评论过」。
> 按品牌名比对会两种错法都犯：别的 agent 发过的审查意见被漏判（重复发布），
> 同账号换 agent 跑时被误判（该发的没发）。

下面这段可直接复制执行（放在引用块外，避免每行带 `> ` 前缀而不能直接跑）：

```bash
# 取当前 token 对应的账号 login（实测 200；无 token 返回 401 {"message":"401 Unauthorized"}）
MY_LOGIN=$(curl -s "https://api.gitcode.com/api/v5/user?access_token=$GITCODE_API_TOKEN" | jq -r '.login // empty')
[ -n "$MY_LOGIN" ] || { echo "ERROR: 取不到当前账号 login，无法判断是否已评论过" >&2; exit 1; }
echo "MY_LOGIN=$MY_LOGIN"
# 再用「基本格式」的翻页输出比 author（回复也要一起比，故先展开 reply[]）：
#   … | jq -c --arg me "$MY_LOGIN" '.[] | select(.author == $me) | {id, reply_to, body}'
# 有输出 = 已评论过（含回复），按步骤 1 停止；无输出 = 未评论过
```

如果以上任一条件成立，停止执行，不继续后续步骤。

> ⚠️ **评论列表取不全时必须停止审查，不得据此判断「未评论过」**：
> 「基本格式」的翻页脚本非 0 退出、或往 stderr 打了 `ERROR`，都说明列表不完整。
> 此时拿到的空列表既可能是「确实没人评论过」，也可能是「请求失败/漏页」，
> 两者无法区分——按后者误判成前者就会重复发布审查意见。
> 必须先修好请求（token、网络、端点、分页）把评论取全，再继续步骤 2。

**注意**：仍然需要审查由 AI agent 生成的 PR（包括本 agent 自己生成的）。

### 步骤 2: 获取项目规范上下文

返回所有相关规范文件的文件路径列表（不含内容）：
- 根目录的规范文件（如 CONTRIBUTING.md、CODE_STYLE.md 等）
- PR 修改文件所在目录中的规范文件

### 步骤 3: 获取 PR 变更摘要

查看 PR 并返回变更摘要：

```bash
# 获取 PR 信息
curl -s "https://api.gitcode.com/api/v5/repos/{owner}/{repo}/pulls/{pr_number}" \
  -H "Authorization: Bearer $GITCODE_API_TOKEN"

# 获取 PR 变更文件列表（每项含 patch.diff）
curl -s "https://api.gitcode.com/api/v5/repos/{owner}/{repo}/pulls/{pr_number}/files" \
  -H "Authorization: Bearer $GITCODE_API_TOKEN"

# 获取 PR diff：v5 没有 diff=true 参数，diff 在 files 的 patch.diff 字段里
curl -s "https://api.gitcode.com/api/v5/repos/{owner}/{repo}/pulls/{pr_number}/files" \
  -H "Authorization: Bearer $GITCODE_API_TOKEN" | jq -r '.[] | .patch.diff'
```

### 步骤 3.1: 确定问题代码的准确行号

发布行内评论需要准确的行号。通过以下方法确定：

#### 方法 1: 从 PR Diff 获取

```bash
# 获取 PR 变更文件列表（包含 diff 信息）
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/files" \
  -H "Authorization: Bearer $GITCODE_API_TOKEN"
```

返回的 `patch.diff` 字段包含 hunk header，格式为：
```
@@ -old_start,old_count +new_start,new_count @@
```

- `-old_start,old_count`: 原文件的起始行和变更行数
- `+new_start,new_count`: 新文件的起始行和新增行数

**注意**：hunk header 只提供了起始行号，需要根据 diff 内容逐行计算实际行号。

#### 方法 2: 取文件原始内容验证（推荐）

直接读取 PR head commit 上的完整文件，用 `grep -n` 拿准确行号。**首选 v5 contents API**
（`content` 是 base64，需解码）：

```bash
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/contents/path/to/file.cc?ref=<HEAD_SHA>&access_token=$GITCODE_API_TOKEN" \
  | jq -r '.content' | base64 -d | grep -n "问题代码模式"
```

或用同一响应的 `download_url`（`…/blobs/<blob_sha>/<path>` 形式，返回明文，无需解码）：

```bash
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/contents/path/to/file.cc?ref=<HEAD_SHA>&access_token=$GITCODE_API_TOKEN" \
  | jq -r '.download_url' | xargs -r curl -s | grep -n "问题代码模式"
```

> ⚠️ **`https://raw.gitcode.com/${owner}/${repo}/raw/<commit_sha>/<path>` 这个形式实测不稳定，
> 不要当唯一手段**：同一 URL、同一 sha，在某一段时间内对 `.agents/skills/gitcode-pr/` 下的
> `commands/review.md`、`references/gitcode_api.md`、`evals/evals.json` 连续 3 次都返回
> `403`（body 为「暂不支持预览」），而同一时刻同 sha 的 `SKILL.md`/`README.md`/`build.sh`
> 返回 `200`；几分钟后同一批 URL 又连续 5 次全部 `200`（复查时再次复现：同一 sha 下
> `SKILL.md` 返回 `403`、`review.md`/`README.md`/`build.sh` 返回 `200`——**命中哪些文件也在变**）。
> `/pulls/<n>/files` 响应里的 `raw_url`、`files.json` 里 `diffs[].head.url` 都是这个形式，同样不可靠。
>
> 因此：
> - **取内容用 contents API 或它的 `download_url`（blobs 形式）**，两者实测稳定 `200`
>   （blobs 形式连测 3 次全 200，且下载内容与本地文件 md5 一致）；
> - `ref` **用 commit sha 或分支名都可以**（实测 cann/amct：`ref=master`、`ref=9.2.0`、
>   `ref=feature/community-tasks`（带斜杠，原样或 `%2F` 编码均可）、`ref=65dbe33`（短 sha）、
>   `ref=<40 位全 sha>` 全部 `200`，分支名取到的内容与本地 `git show cann/<branch>:<path>` md5 一致；
>   跨 fork PR 的 head commit 在**目标仓库**路径下也取得到，实测 PR!447 head `59bd5789…`
>   用 `cann/amct` 与源仓库 `m0_45199755/amct` 两条路径都 `200`、blob sha 相同）。
>   **审查 PR 时仍应传 head commit sha**：分支名指向的是分支**当前**位置，PR 提交后分支若又推了新
>   commit，用分支名取到的就不是被审查的那一版，行号会与 diff 对不上；
> - `404` **不能用来区分「ref 不存在」和「文件在该 ref 上不存在」**：两种情况响应体完全相同，
>   都是 `{"error_code":404,"error_code_name":"NOT_FOUND","error_message":"项目或文件不存在","trace_id":"…"}`
>   （实测；`ref=nonexist-branch-xyz` + 存在的 `README.md`、`ref=feature/community-tasks` + 该分支上
>   没有的 `.agents/…/SKILL.md`、`ref=master` + 不存在的文件，三者返回一模一样）。
>   遇到 404 要两种可能都排查，别一口咬定是 ref 写错；
> - 无论用哪种，都要**校验 HTTP 状态码 / 内容非空**再 `grep`——`curl -s` 对 403/404 的退出码
>   仍是 `0`，不校验就会把错误页当成文件内容，`grep -n` 静默输出空结果，行号定位随之失败。

#### 最佳实践

1. **先用 diff 定位大致范围**：通过 PR diff 找到变更的代码块
2. **再取文件原始内容确认精确行号**：避免手动计算出错（用方法 2 的 contents API，
   不要依赖 `…/raw/<commit>/<path>`——该形式实测会间歇 `403`）
3. **发布评论前验证**：确保行号对应的是真正的问题代码

### 步骤 4: 代码审查

从以下方面全面审查变更，返回问题列表，每个问题包含描述和标记原因（如"规范合规性"、"Bug"）：

**规范合规性审查**
检查变更是否符合项目规范。注意：评估文件的规范合规性时，只考虑该文件路径或父目录中的规范文件。

**Bug 扫描**
扫描明显的 Bug。只关注 diff 本身，不读取额外上下文。只标记显著的 Bug；忽略细微问题和可能的误报。不要标记需要查看 diff 之外上下文才能验证的问题。

**代码问题检查**
查找引入代码中的问题。可能是安全问题、逻辑错误等。只查找变更代码范围内的问题。

**关键原则：只需要高信号问题。** 标记以下类型的问题：
- 代码无法编译或解析（语法错误、类型错误、缺少导入、未解析引用）
- 无论输入如何，代码肯定会产生错误结果（明显的逻辑错误）
- 明确、无歧义的规范违反，可以引用被违反的具体规则

**不要标记：**
- 代码风格或质量问题
- 依赖于特定输入或状态的潜在问题
- 主观建议或改进意见

**如果你不确定某个问题是否真实存在，不要标记它。** 误报会损害信任并浪费审查者时间。

此外，每个子 agent 都应被告知 PR 标题和描述。这有助于理解作者的意图。

### 步骤 5: 验证问题

对于步骤 4 发现的每个问题，进行验证。验证工作应获取 PR 标题和描述以及问题描述，确信地验证所述问题确实是真实问题。

例如，如果标记了"变量未定义"这样的问题，验证代码中确实如此。另一个例子是规范合规问题，验证被违反的规范规则确实适用于此文件且确实被违反。

### 步骤 6: 过滤问题

过滤掉步骤 5 中未验证通过的问题。这一步将得到我们的高信号问题列表。

### 步骤 7: 输出审查摘要

在终端输出审查结果摘要：
- 如果发现问题：列出每个问题及简要描述
- 如果未发现问题：输出"未发现问题。已检查 Bug 和规范合规性。"

如果 **未提供** `--comment` 参数，在此停止。不要发布任何 GitCode 评论。

如果 **提供了** `--comment` 参数且 **未发现问题**，使用 GitCode API 发布摘要评论并停止。

如果 **提供了** `--comment` 参数且 **发现问题**，继续步骤 8。

### 步骤 8: 准备评论列表

创建计划发布的所有评论列表。这仅供你自己确认对评论内容满意。不要在任何地方发布此列表。

### 步骤 9: 发布行内评论

使用 GitCode API 为每个问题发布行内评论。

> ⚠️ 统一使用 **v5** API（官方文档：「当前可用的版本为 `/api/v5`」）。
> 不要使用 `/api/v4`，其写接口已被禁用，返回
> `403 {"message":"当前 /api/v4 接口已禁用，请使用官方文档中的 /api/v5 接口"}`。

#### 创建行内评论

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

**多行选择说明**：
- `start_position`: 选中的起始行号（多行选择时设置）
- `position`: 选中的结束行号
- 单行评论时只传 `position`（或与 `start_position` 设为相同值）

**参数说明**：

| 参数 | 说明 | 必需 |
|------|------|------|
| `body` | 评论内容 | ✅ |
| `path` | 文件相对路径 | ✅ |
| `position` | 结束行号 | ✅ |
| `start_position` | 起始行号（多行选择） | 多行时 |
| `access_token`（query） | 用户授权码 | ✅ |

对于每个评论：
- 提供问题的简要描述
- 对于小型、自包含的修复，包含可提交的建议代码块
- 对于较大的修复（6+ 行、结构性变更或跨多个位置的变更），描述问题和建议修复方式，不包含建议代码块
- 永远不要发布可提交建议，除非提交该建议能完全修复问题。如果需要后续步骤，不要留下可提交建议。

**重要：每个唯一问题只发布一条行内评论。不要发布重复评论。**

## 误报列表

在步骤 4 和 5 评估问题时使用此列表（这些是误报，不要标记）：

- 已存在的问题
- 看起来是 Bug 但实际上是正确的代码
- 高级工程师不会标记的吹毛求疵
- Linter 会捕获的问题（不要运行 Linter 验证）
- 一般代码质量问题（如缺乏测试覆盖、一般安全问题），除非规范中明确要求
- 规范中提到但在代码中明确静默的问题（如通过 lint ignore 注释）

## 注意事项

- 使用 GitCode API 与 GitCode 交互（如获取 Pull Request、创建评论）。不要使用网页抓取。
- 开始前创建待办列表。
- 必须在行内评论中引用并链接每个问题（如引用规范文件时，包含指向它的链接）。
- 如果未发现问题且提供了 `--comment` 参数，发布以下格式的评论：

---

## 代码审查

未发现问题。已检查 Bug 和规范合规性。

---

- 在行内评论中链接代码时，严格遵循以下格式，否则 Markdown 预览无法正确渲染：
  `https://gitcode.com/{owner}/{repo}/blob/{full_git_sha}/path/to/file#L{start}-L{end}`
  - 需要完整的 git sha
  - 必须提供完整 sha。类似 `https://gitcode.com/owner/repo/blob/$(git rev-parse HEAD)/foo/bar` 的命令不起作用，因为评论会直接在 Markdown 中渲染
  - 仓库名必须与你正在代码审查的仓库匹配
  - 文件名后使用 # 号
  - 行范围格式为 L[start]-L[end]
  - 在你评论的行前后至少提供 1 行上下文（如评论第 5-6 行，应链接到 L4-L7）

## GitCode API 参考

**重要**：读写**统一使用 v5 API**，认证方式为 query 参数
`?access_token=$GITCODE_API_TOKEN`（或 `Authorization: Bearer $GITCODE_API_TOKEN`）。
不要使用 `/api/v4`：官方文档只收录 v5，v4 写接口已被禁用（403），部分读接口从未发布（404）。

### 快速参考

| 操作 | API 端点 |
|------|---------|
| 获取 PR 信息 | `GET /api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>` |
| 获取 PR 变更 | `GET /api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/files` |
| 获取 PR 评论 | `GET /api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/comments?per_page=100&page=<N>`（`per_page` 默认 20（实测）、最大 100（文档值，未实测），**必须翻页取完**，见「基本格式」） |
| 发布普通评论 | `POST /api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/comments`（只传 `body`） |
| 发布行内评论 | `POST /api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/comments`（传 `body`+`path`+`position`） |
| 回复评论 | `POST /api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/discussions/<DISCUSSION_ID>/comments` |
| 删除评论 | `DELETE /api/v5/repos/${owner}/${repo}/pulls/comments/<COMMENT_ID>`（用整数评论 ID，非 `discussion_id`） |

### 基本格式

```bash
# 读取 PR 评论（顶层直接是数组；comment_type 区分 diff_comment / pr_comment）
# ⚠️ per_page 默认 20（实测）；「最大 100」取自官方文档、未实测（抽样中没有父评论 > 100 的
#    PR，在 38 条的 PR 上传 per_page=150/200 都返回全部 38 条，既不能证明也不能推翻上限）。
#    按 100 翻页在上限更大时也只是多翻几页、结果仍正确，故评论数可能超过 100 的 PR
#    必须按 page 递增翻页取完，
#    否则步骤 1「本 agent 是否已评论过」会漏读后面的评论而误判，导致重复发布审查意见。
# ⚠️ 顶层数组**不含回复**：回复嵌在父评论的 reply[] 里（实测 cann/amct PR!455：
#    回复 ID 在顶层一个都查不到），必须一并展开，
#    否则「只在既有讨论里回复过」的活动看不到，步骤 1 同样会误判为未评论过。
page=1
while :; do
  resp=$(curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/comments?access_token=$GITCODE_API_TOKEN&per_page=100&page=${page}")
  # 响应必须是数组才继续。两类失败都要显式报错、不能静默 break：
  #   ① 请求失败/空响应 —— `jq 'length'` 得空值，`[ "" -lt 100 ]` 报错返回非 0，
  #      不触发 break，循环永不终止；
  #   ② 返回错误对象（如 {"error_code":"APIG.0101"}）—— `jq 'length'` 得键数，
  #      会被当成「本页 N 条」而提前静默终止，漏读后续评论却毫无提示。
  if [ "$(echo "$resp" | jq -r 'type' 2>/dev/null)" != "array" ]; then
    echo "ERROR: 第 ${page} 页评论请求失败或响应非数组，已取结果不完整：$resp" >&2
    exit 1   # 取不全就是失败，必须非 0 退出（见下方「失败必须显式」）
  fi
  # 父评论原样输出，再把 reply[] 展开成同形对象；reply 对象只有 id/body/user/时间戳，
  # discussion_id 与 comment_type 从父评论带下来（带下来的是**列表接口**那套取值：
  # diff_comment / pr_comment；单条接口查同一条回复是另一套，如 DiffNote，别混用）
  echo "$resp" | jq -c '.[] | {id, discussion_id, comment_type, author: .user.login, body} as $c | $c, (.reply[]? | {id, discussion_id: $c.discussion_id, comment_type: $c.comment_type, author: .user.login, body, reply_to: $c.id})'
  count=$(echo "$resp" | jq 'length')
  [ "$count" -lt 100 ] && break   # count 数的是**父评论**条数（回复不计入分页），不足 100 即最后一页
  page=$((page + 1))
done
```

> **失败必须显式（不能 `break` 后以 0 退出）**：`break` 版本在两类失败下都只往 stderr
> 打一行 `ERROR`，脚本仍 `exit 0`、stdout 是空列表——实测把 token 换成无效值返回
> `{"message":"401 Unauthorized"}`、把域名换成不可解析主机返回空响应，两种情况
> `echo $?` 都是 `0`。步骤 1 正是靠这份列表判断「本 agent 是否已评论过」（按 user.login 比对，见步骤 1），
> 空列表 + `exit 0` 会被读成「取全了且没人评论过」，于是重复发布审查意见，
> 也就是上一轮「必须翻页取完」要防的后果。故失败分支用 `exit 1`；
> 若这段循环嵌在更大的脚本里不能直接退出，则改成置标志（`incomplete=1`）后 `break`，
> 并由调用方检查该标志，非 0/为真时停止审查。调用方无论如何都要检查退出码。

> **是否取全可用响应头校验**：该接口在响应头返回 `total_count`（评论总数）与
> `total_page`（总页数）——实测 `per_page=5&page=1` 返回 `total_count: 10`、`total_page: 2`，
> `per_page=100` 返回 `total_page: 1`（数值随评论增长变化，此处为测量当时的值）。
> 用 `curl -sD -` 取头即可核对是否已取全。
> ⚠️ `total_count` 只数**父评论**，回复不计入（实测 cann/amct PR!455：响应头 `total_count`
> 的值等于顶层数组长度，`reply[]` 内的回复不计入）。核对时要拿**父评论**条数比，
> 别拿展开 `reply[]` 后的输出条数比。
> ⚠️ 该接口**不返回** GitHub 风格的 `Link` header，不要按 `rel="next"` 判断翻页。
>
> 终止条件用「本页条数 < per_page」而非「本页为空」：评论总数恰为 `per_page` 整数倍时
> 会多发一次请求拿到空数组才停（实测 10 条 / `per_page=5` → 3 次请求），行为正确、无副作用；
> 想省掉这次请求可改读 `total_page` 头。
>
> 自检分页是否生效：把 `per_page` 临时调小（如 `per_page=5`）后分别请求
> `page=1` 与 `page=2`，确认第 2 页返回的是剩余的评论而非重复内容。

**完整 API 格式（包括多行行内评论）详见**：`../references/gitcode_api.md`
