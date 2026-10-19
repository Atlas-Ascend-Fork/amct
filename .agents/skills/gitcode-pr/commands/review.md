---
description: 审查 GitCode Pull Request 的代码变更
# ⚠️ allowed-tools 三项来源，删任何一项前先对号：
#    ① 正文里会被复制执行的命令: curl / jq / grep / echo / base64 / xargs
#    ② ${owner}/${repo} 靠 SKILL.md「提取仓库 owner/repo 信息」填，用到 git remote get-url 与 sed
#    ③ 审查时查本地提交的预留授权: git diff / git log / git show
allowed-tools: Bash(curl:*), Bash(jq:*), Bash(grep:*), Bash(base64:*), Bash(xargs:*), Bash(sed:*), Bash(echo:*), Bash(git remote get-url:*), Bash(git diff:*), Bash(git log:*), Bash(git show:*)
---

# GitCode PR 代码审查

对指定的 Pull Request 进行代码审查。

## 审查流程

请严格按照以下步骤执行：

### 步骤 1: 前置检查

检查以下条件是否成立：
- PR 已关闭
- PR 是草稿状态
- PR 不需要代码审查（如：自动化 PR、明显正确的微小变更）
- **本 agent 自己**已经在此 PR 上评论过（通过 GitCode API 检查 PR 评论：**必须翻完所有分页**，
  只取第一页会漏读后面的评论；**必须展开父评论的 `reply[]`**，回复不在列表顶层、不展开就看不到）

如果以上任一条件成立，停止执行，不继续后续步骤。

**注意**：仍然需要审查由 AI agent 生成的 PR（包括本 agent 自己生成的）。

#### 判断「本 agent 是否已评论过」

> ⚠️ **按 `user.login` 比对**。判断方法是拿当前 token 对应的账号 login（`GET /api/v5/user` 的 `.login`）去比评论的 `author`，命中即视为「已评论过」（含回复）。
>
> ⚠️ **评论列表取不全时必须停止审查，不得据此判断「未评论过」**：翻页脚本非 0 退出、或往 stderr 打了 `ERROR`，都说明列表不完整——此时空列表无法区分「确实没人评论过」与「请求失败/漏页」。必须先修好请求（token、网络、端点、分页）把评论取全再继续步骤 2，理由见「取评论：翻页取全（规范实现）」的「六条硬规则」第 3 条。

下面这段可直接复制执行：

```bash
# 取当前 token 对应的账号 login（token 无效/缺失时取不到，由下一行守卫 exit 1）
MY_LOGIN=$(curl -s "https://api.gitcode.com/api/v5/user?access_token=$GITCODE_API_TOKEN" | jq -r '.login // empty')
[ -n "$MY_LOGIN" ] || { echo "ERROR: 取不到当前账号 login，无法判断是否已评论过" >&2; exit 1; }
echo "MY_LOGIN=$MY_LOGIN"
# 再拿「取评论：翻页取全（规范实现）」的翻页输出比 author（该脚本已展开 reply[]，回复一起比）：
#   <翻页脚本输出> | jq -c --arg me "$MY_LOGIN" 'select(.author == $me) | {id, reply_to, body}'
#   有输出 = 已评论过（含回复），按步骤 1 停止；无输出 = 未评论过
# ⚠️ 过滤器别带 `.[]`：该脚本输出的是**逐行对象流**，带 `.[]` 是对每个对象的**值**再迭代，
#    实测报 `Cannot index number with string "author"`（要先聚成数组再 `.[]` 得加 `-s`）
```

### 步骤 2: 获取项目规范上下文

返回所有相关规范文件的文件路径列表（不含内容）：
- 根目录的规范文件（如 CONTRIBUTING.md、CODE_STYLE.md 等）
- PR 修改文件所在目录中的规范文件

### 步骤 3: 获取 PR 变更摘要

查看 PR 并返回变更摘要。**先按 SKILL.md「提取仓库 owner/repo 信息」把 `${owner}`/`${repo}` 设好**，再执行下面的命令：

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

> ⚠️ 本文件所有 bash 命令 / API 请求 URL 里的占位符统一写 `${owner}` / `${repo}` / `<PR_NUMBER>`。
> `{owner}` 这种花括号写法在 bash 里不展开，会原样拼进 URL，且报错会赖在 `pr_number` 上。
> `curl -s` 对 4xx 的退出码仍是 `0`，必须校验状态码 / 响应是数组，否则错误响应会被当成 PR 信息往下走。
> `${owner}`/`${repo}` 未赋值时的 `401`/`404` 鉴别方法见 `../references/gitcode_api.md` →「认证方式」。

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

> ⚠️ **不要用 `…/raw/<commit_sha>/<path>` 取内容**（`/pulls/<n>/files` 的 `raw_url`、`files.json` 的 `diffs[].head.url` 都是这个形式）：实测间歇 `403`，改用上面方法 2 的两条路径。
> 403 的完整实测记录与 `ref` 取值枚举见 `../references/gitcode_api.md` →「获取文件原始内容」。
>
> **两条硬纪律（与用哪个地址无关；上面的代码示例本身不含这两道防护，得自己加）：**
> - `ref` **必须传 PR head commit sha**，不要用分支名——分支名指向分支**当前**位置，PR 提交后分支若又推了新 commit，取到的就不是被审查的那一版，`grep -n` 的行号会与 diff 对不上；
> - 取完**必须校验 HTTP 状态码 / 内容非空**再 `grep`：`curl -s` 对 `403`/`404` 的退出码仍是 `0`，不校验就会把错误页当成文件内容，`grep -n` 静默输出空结果，行号定位随之失败。遇到 `404` 别一口咬定是 ref 写错——「ref 不存在」与「文件在该 ref 上不存在」的响应体完全相同。

#### 最佳实践

1. **先用 diff 定位大致范围**：通过 PR diff 找到变更的代码块
2. **再取文件原始内容确认精确行号**：避免手动计算出错
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

**参数与多行选择**：`body`（内容）、`path`（文件相对路径）、`position`（**结束**行号）必填；
多行选择再加 `start_position`（**起始**行号），单行评论只传 `position`（或与 `start_position` 同值）。
完整参数表（另有 `position_type`、query 里的 `access_token`）与实测读回结果见
`../references/gitcode_api.md` →「提交行内评论（支持多行选择）」。

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
端点清单、参数表与响应字段见 `SKILL.md` 各节及 `../references/gitcode_api.md`，本文件不重抄。

### 取评论：翻页取全（规范实现）

**可执行的翻页循环全仓只有一份**：`SKILL.md` →「获取 PR 讨论列表（包含行内评论）」。
步骤 1 直接用它（输出已展开 `reply[]`，回复一起比 `author`），本文件不重抄。
⚠️ 该接口的 `per_page` 默认 **20**、最大 **100**（超出被服务端**静默夹回**），
不显式传 100 并从 `page=1` 翻页取完，步骤 1「本 agent 是否已评论过」就会漏读后面的评论而误判。
那份循环的投影字段**按用途裁剪**：步骤 1 只比 `author`，故不带 `resolved` / `diff_position`；
要按「检视意见是否已解决」或行号过滤，把这两个字段加进**两处**投影（父行与 `reply` 行）。

> **六条硬规则（改写这段循环、或把它嵌进更大脚本时同样适用）**：
> 1. 终止判据**只用响应头 `total_page`**（它按服务端**生效**页长算），**不要**用「本页条数 < 请求的 `per_page`」——请求值超过上限会被静默夹回，该判据第一页就 `break`、后面评论全丢；
> 2. `total_page` 必须**每页重读**、用最新值判终止（它是本次请求时刻的快照，翻页途中新增评论会让它变大，每页重读才能自纠正）；
> 3. **失败必须显式**：失败分支用 `exit 1`，**不能 `break` 后以 0 退出**——空列表 + `exit 0` 会被读成「取全了且没人评论过」，步骤 1 就会重复发布审查意见。嵌在更大脚本里不能直接退出时，改置 `incomplete=1` 标志后 `break`，由调用方检查该标志；调用方无论如何都要检查退出码；
> 4. 读响应头必须 `-D <文件>`（`curl -s` 会丢掉响应头）、取值前必须 `tr -d '\r'`（头是 CRLF）、匹配头名必须 `tolower($1)` 或 `grep -i`（头名是全小写自定义头，`awk`/`grep` 大小写敏感）；
> 5. `total_count` 只数**父评论**，回复不计入——核对「是否取全」要拿父评论条数比，别拿展开 `reply[]` 后的输出条数比；
> 6. 该接口**不返回** GitHub 风格的 `Link` header，不要按 `rel="next"` 判断翻页；单条资源接口（如 `GET /pulls/<n>`）也没有这两个分页头，只有列表接口才有。
>
> 每条规则的实测依据（`per_page=150` 只返回 100 条、无效 token 与不可解析域名的响应体、越界 `page` 返回空数组 + `200`、`page=0`/负数返回 `400`、`total_count` 等于顶层数组长度等）见 `../references/gitcode_api.md` →「获取 PR 讨论（包含行内评论）」。
>
> 自检分页是否生效：把 `per_page` 临时调小（如 `per_page=5`）后分别请求 `page=1` 与 `page=2`，确认第 2 页返回的是剩余的评论而非重复内容，且 `total_page` 等于 `ceil(total_count / 5)`。

**完整 API 格式（包括多行行内评论）详见**：`../references/gitcode_api.md`
