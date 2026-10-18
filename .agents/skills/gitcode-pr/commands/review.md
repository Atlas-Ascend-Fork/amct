---
description: 审查 GitCode Pull Request 的代码变更
# ⚠️ allowed-tools 必须覆盖正文里所有会被复制执行的命令：步骤 3.1「方法 2」用 contents API 取
#    文件内容时要管道给 base64 -d（解码）或 xargs -r curl（跟 download_url），漏授权会让
#    推荐路径被工具白名单拦住、agent 只能退回去用不可靠的 raw/<commit>/<path> 形式
#    步骤 3 起的 URL 用 ${owner}/${repo} 这两个 shell 变量，取值脚本在 SKILL.md
#    「提取仓库 owner/repo 信息」，用到 git remote get-url 与 sed；步骤 1 的自检块用到 echo。
#    这三条不授权，变量就填不上（后果见步骤 3 的 ⚠️）
allowed-tools: Bash(curl:*), Bash(jq:*), Bash(grep:*), Bash(base64:*), Bash(xargs:*), Bash(sed:*), Bash(echo:*), Bash(git remote get-url:*), Bash(git diff:*), Bash(git log:*), Bash(git show:*)
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
- **本 agent 自己**已经在此 PR 上评论过（通过 GitCode API 检查 PR 评论：**必须翻完所有分页**，
  只取第一页会漏读后面的评论；**必须展开父评论的 `reply[]`**，回复不在列表顶层、不展开就看不到。
  实现见「GitCode API 参考 → 基本格式」，判断方法见下方小节）

如果以上任一条件成立，停止执行，不继续后续步骤。

**注意**：仍然需要审查由 AI agent 生成的 PR（包括本 agent 自己生成的）。

#### 判断「本 agent 是否已评论过」

> ⚠️ **按 `user.login` 比对，不要按 agent 品牌名比对**：执行本 skill 的可能是
> Codex、Claude Code、Cursor 等任意 agent/客户端，正文里不写死品牌名。判断方法是拿当前 token 对应的
> 账号 login（`GET /api/v5/user` 的 `.login`）去比评论的 `author`，命中即视为「已评论过」（含回复）。
>
> ⚠️ **评论列表取不全时必须停止审查，不得据此判断「未评论过」**：翻页脚本非 0 退出、或往 stderr 打了
> `ERROR`，都说明列表不完整——此时空列表无法区分「确实没人评论过」与「请求失败/漏页」。必须先修好请求
> （token、网络、端点、分页）把评论取全再继续步骤 2，理由见「基本格式」的「六条硬规则」第 3 条。

下面这段可直接复制执行：

```bash
# 取当前 token 对应的账号 login（实测 200；无 token 返回 401 {"message":"401 Unauthorized"}）
MY_LOGIN=$(curl -s "https://api.gitcode.com/api/v5/user?access_token=$GITCODE_API_TOKEN" | jq -r '.login // empty')
[ -n "$MY_LOGIN" ] || { echo "ERROR: 取不到当前账号 login，无法判断是否已评论过" >&2; exit 1; }
echo "MY_LOGIN=$MY_LOGIN"
# 再用「基本格式」的翻页输出比 author（回复也要一起比，故先展开 reply[]）：
#   … | jq -c --arg me "$MY_LOGIN" '.[] | select(.author == $me) | {id, reply_to, body}'
# 有输出 = 已评论过（含回复），按步骤 1 停止；无输出 = 未评论过
```

### 步骤 2: 获取项目规范上下文

返回所有相关规范文件的文件路径列表（不含内容）：
- 根目录的规范文件（如 CONTRIBUTING.md、CODE_STYLE.md 等）
- PR 修改文件所在目录中的规范文件

### 步骤 3: 获取 PR 变更摘要

查看 PR 并返回变更摘要。**先按 SKILL.md「提取仓库 owner/repo 信息」把 `${owner}`/`${repo}`
设好**（本文件不重复那段脚本），再执行下面的命令：

```bash
# 获取 PR 信息
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>" \
  -H "Authorization: Bearer $GITCODE_API_TOKEN"

# 获取 PR 变更文件列表（每项含 patch.diff）
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/files" \
  -H "Authorization: Bearer $GITCODE_API_TOKEN"

# 获取 PR diff：v5 没有 diff=true 参数，diff 在 files 的 patch.diff 字段里
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/files" \
  -H "Authorization: Bearer $GITCODE_API_TOKEN" | jq -r '.[] | .patch.diff'
```

> ⚠️ 占位符统一写 `${owner}` / `${repo}` / `<PR_NUMBER>`（与本文件步骤 3.1、下方「快速参考」表一致）：
> `owner`/`repo` 按 SKILL.md「提取仓库 owner/repo 信息」从 `git remote get-url` 解析得到，
> **不能**照抄字面量 `{owner}`——bash 的花括号展开需要逗号或 `..`，`{owner}` 不构成展开，
> 会原样拼进 URL。实测照抄后：`…/repos/{owner}/{repo}/pulls/{pr_number}` 返回
> `400 {"error_code":1000,"error_code_name":"PARAMETER_ERROR","error_message":"参数 pr_number 类型错误"}`；
> 只把 owner/repo 写成字面量、PR 号给数字时返回
> `404 {"error_code_name":"UN_KNOW","error_message":"Project not found:owner/repo"}`
> （同一时刻正确形式 `…/repos/<owner>/<repo>/pulls/<PR_NUMBER>` 返回 `200`）。
> 两种失败 `curl -s` 的退出码都仍是 `0`，不校验状态码就会拿着错误响应当 PR 信息往下走。
>
> 同理，**`${owner}`/`${repo}` 未赋值时拼出的是 `…/repos///pulls/<PR_NUMBER>`**，实测返回
> `401 {"message":"401 Unauthorized"}`，与「token 失效」的响应体**一字不差**
> （同一时刻正确路径无 token 返回 `200`，带坏 token 返回同样的
> `401 {"message":"401 Unauthorized"}`）。看到这个 401 先 `echo "$owner/$repo"` 确认变量非空，
> 别一上来就怀疑 token。

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

> ⚠️ **不要用 `…/raw/<commit_sha>/<path>` 取内容**（`/pulls/<n>/files` 的 `raw_url`、
> `files.json` 的 `diffs[].head.url` 都是这个形式）：实测间歇 `403`「暂不支持预览」，
> 同一 sha 下命中哪些文件、什么时候恢复都在变。取内容用上面方法 2 的 contents API 或它的
> `download_url`（`…/blobs/<blob_sha>/<path>`），两者实测稳定 `200`。
> 403 的完整实测记录与 `ref` 取值枚举见 `../references/gitcode_api.md` →「获取文件原始内容」。
>
> **两条硬纪律（与用哪个地址无关；上面的代码示例本身不含这两道防护，得自己加）：**
> - `ref` **必须传 PR head commit sha**，不要用分支名——分支名指向分支**当前**位置，
>   PR 提交后分支若又推了新 commit，取到的就不是被审查的那一版，`grep -n` 的行号会与 diff 对不上；
> - 取完**必须校验 HTTP 状态码 / 内容非空**再 `grep`：`curl -s` 对 `403`/`404` 的退出码仍是 `0`，
>   不校验就会把错误页当成文件内容，`grep -n` 静默输出空结果，行号定位随之失败。遇到 `404`
>   别一口咬定是 ref 写错——「ref 不存在」与「文件在该 ref 上不存在」的响应体完全相同。

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

> 完整参数表（另有 `position_type`）见 `../references/gitcode_api.md` →「提交行内评论（支持多行选择）」。

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

### 快速参考

| 操作 | API 端点 |
|------|---------|
| 获取 PR 信息 | `GET /api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>` |
| 获取 PR 变更 | `GET /api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/files` |
| 获取 PR 评论 | `GET /api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/comments?per_page=100&page=<N>`（**必须翻页取完**，`per_page` 上限与终止判据见「基本格式」） |
| 发布普通评论 | `POST /api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/comments`（只传 `body`） |
| 发布行内评论 | `POST /api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/comments`（传 `body`+`path`+`position`） |
| 回复评论 | `POST /api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/discussions/<DISCUSSION_ID>/comments` |
| 删除评论 | `DELETE /api/v5/repos/${owner}/${repo}/pulls/comments/<COMMENT_ID>`（用整数评论 ID，非 `discussion_id`） |

### 基本格式

```bash
# 读取 PR 评论（顶层直接是数组；comment_type 区分 diff_comment / pr_comment）
# ⚠️ per_page 默认 20、最大 100（请求值超过 100 会被服务端**静默夹回 100**），必须显式传 100
#    并从 page=1 起按 page 递增翻页取完，否则步骤 1「本 agent 是否已评论过」会漏读后面的评论而误判。
# ⚠️ 顶层数组**不含回复**：回复嵌在父评论的 reply[] 里，必须一并展开，
#    否则「只在既有讨论里回复过」的活动看不到，步骤 1 同样会误判为未评论过。
MAX_PAGES=50   # 兜底上界（50 × 100 = 5000 条父评论，远超真实 PR 的评论量），正常永不触发，
               # 只防服务端分页异常导致空转；撞上界必须 exit 1，不能 break 后以 0 退出
hdr=$(mktemp) || exit 1
trap 'rm -f "$hdr"' EXIT   # 嵌进更大脚本时，把这个 rm 合并进已有的 trap，别覆盖它
page=1
while :; do
  [ "$page" -gt "$MAX_PAGES" ] && { echo "ERROR: 翻页超过 ${MAX_PAGES} 页兜底上界，结果不可信" >&2; exit 1; }
  # -D 把响应头落到文件：curl -s 会丢掉响应头，total_page 就读不到了
  resp=$(curl -s -D "$hdr" "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/comments?access_token=$GITCODE_API_TOKEN&per_page=100&page=${page}")
  # 响应头是 CRLF 结尾，必须 tr -d '\r'，否则值带回车、数值比较报 `integer expression expected`
  total_page=$(tr -d '\r' < "$hdr" | awk -F': *' 'tolower($1)=="total_page"{print $2}')
  # 响应必须是数组才继续：空响应/错误对象的 `jq 'length'` 会被当成「本页 N 条」而静默提前终止
  if [ "$(echo "$resp" | jq -r 'type' 2>/dev/null)" != "array" ]; then
    echo "ERROR: 第 ${page} 页评论请求失败或响应非数组，已取结果不完整：$resp" >&2
    exit 1   # 取不全就是失败，必须非 0 退出（见下方「六条硬规则」第 3 条「失败必须显式」）
  fi
  # 父评论原样输出，再把 reply[] 展开成同形对象；reply 对象只有 id/body/user/时间戳，
  # discussion_id 与 comment_type 从父评论带下来（带下来的是**列表接口**那套取值：
  # diff_comment / pr_comment；单条接口查同一条回复是另一套，如 DiffNote，别混用）
  # 投影字段**按用途裁剪**：步骤 1 只比 author，故不带 resolved / diff_position；
  # 要按「检视意见是否已解决」或行号过滤，自己把这两个字段加进两处投影（父行与 reply 行）
  echo "$resp" | jq -c '.[] | {id, discussion_id, comment_type, author: .user.login, body} as $c | $c, (.reply[]? | {id, discussion_id: $c.discussion_id, comment_type: $c.comment_type, author: .user.login, body, reply_to: $c.id})'
  # count 数的是**父评论**条数（回复不计入分页，也不计入 total_page/total_count）
  count=$(echo "$resp" | jq 'length')
  [ "$count" -eq 0 ] && break   # total_page 缺失时的兜底；越界 page 实测返回空数组+200，不会死循环
  [ -n "$total_page" ] && [ "$page" -ge "$total_page" ] && break
  page=$((page + 1))
done
```

> **六条硬规则（改写这段循环、或把它嵌进更大脚本时同样适用）**：
> 1. 终止判据**只用响应头 `total_page`**（它按服务端**生效**页长算），**不要**用
>    「本页条数 < 请求的 `per_page`」——理由见上面代码里的 `per_page` 注释；
> 2. `total_page` 必须**每页重读**、用最新值判终止（它是本次请求时刻的快照，
>    翻页途中新增评论会让它变大，每页重读才能自纠正）；
> 3. **失败必须显式**：失败分支用 `exit 1`，**不能 `break` 后以 0 退出**——空列表 + `exit 0`
>    会被读成「取全了且没人评论过」，步骤 1 就会重复发布审查意见。嵌在更大脚本里不能直接退出时，
>    改置 `incomplete=1` 标志后 `break`，由调用方检查该标志；调用方无论如何都要检查退出码；
> 4. 读响应头必须 `-D <文件>`（`curl -s` 会丢掉响应头）、取值前必须 `tr -d '\r'`（头是 CRLF）、
>    匹配头名必须 `tolower($1)` 或 `grep -i`（头名是全小写自定义头，`awk`/`grep` 大小写敏感）；
> 5. `total_count` 只数**父评论**，回复不计入——核对「是否取全」要拿父评论条数比，
>    别拿展开 `reply[]` 后的输出条数比；
> 6. 该接口**不返回** GitHub 风格的 `Link` header，不要按 `rel="next"` 判断翻页；
>    单条资源接口（如 `GET /pulls/<n>`）也没有这两个分页头，只有列表接口才有。
>
> 每条规则的实测依据（`per_page=150` 只返回 100 条、无效 token 与不可解析域名的响应体、
> 越界 `page` 返回空数组 + `200`、`page=0`/负数返回 `400`、`total_count` 等于顶层数组长度等）
> 见 `../references/gitcode_api.md` →「获取 PR 讨论（包含行内评论）」。
>
> 自检分页是否生效：把 `per_page` 临时调小（如 `per_page=5`）后分别请求 `page=1` 与 `page=2`，
> 确认第 2 页返回的是剩余的评论而非重复内容，且 `total_page` 等于 `ceil(total_count / 5)`。

**完整 API 格式（包括多行行内评论）详见**：`../references/gitcode_api.md`
