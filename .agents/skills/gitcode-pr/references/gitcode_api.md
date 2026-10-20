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

> ⚠️ **本节只描述「文档文本可核的事实」，不描述服务端行为。** 官方文档（docs.gitcode.com）在本
> skill 依赖的几个点上有缺失、滞后和硬错误，所以「以官方文档为准」在这些问题上无从执行。
> 每条都附**复核方法**：请在**你自己的环境**里跑一遍验证或推翻。
> **状态码、响应体、字段清单、scope 名一律不进本表**——那些是运行时观测量，只能探测，
> 分别落在「认证方式」、「获取当前用户」→「响应字段」、「scope 前提与失败分支」三节。

| # | 文档缺陷（可核事实） | 复核方法 |
|---|------|----------|
| 1 | **PAT 的 scope 名全站不枚举**：`docs/help/home/user_center/security_management/user_pat` 只说「根据需要选择不同的范围，如项目、Issue 等」，不列任何勾选项名。全站仅 OAuth 2.0 两页出现 scope 名，且那是 **OAuth 的词汇**（`all_user all_key all_groups all_projects all_pr all_issue all_note all_hook all_repository`），文档从未说明它与 PAT 勾选项有何对应关系 | 抓 `https://docs.gitcode.com/sitemap.xml` 取全部 URL，逐页抓正文 grep `read_user\|read_api\|read_repository\|write_repository\|all_user`。**务必 gzip 解压并剥掉 `script/style/svg`**：内联 SVG 图标的 path 坐标（如 `0.403`）会让 `\b40[13]\b` 这类正则大面积误报 |
| 2 | **接口页不写 scope 前提**：`docs/apis/get-api-v-5-user` 通篇无 `scope` / `read_user` / `all_user` 字样 | 单页抓正文 grep |
| 3 | **几乎不文档化错误响应**：368 个 API 页中 315 页有 `Responses` 段，其中 **312 页只有 2xx**；正文真出现 4xx/5xx 的仅 5 页（含 1 个索引页清单、1 个 release note）→ 照文档写的解析代码没有失败分支 | 全量抓 API 页，剥标签后 grep `\b(4[0-9]{2}\|5[0-9]{2})\b` |
| 4 | **`access_token` 被标 `required`（query 参数）**，共 193 页这么标；且**全站不提任何请求头认证写法** | 全量抓 API 页：grep `access_token` 与 `required` 同现的页数；再 grep `Bearer\|PRIVATE-TOKEN`，应为 0 命中 |
| 5 | **`access_token` 被描述成「用户授权码」**，共 278 页；而 OAuth 页里「授权码」是另一个东西（`code`，一次性、需换 token）→ 按字面执行会拿错凭证 | 对照任一接口页的参数描述与 `docs/apis/oauth` |
| 6 | **流水线页示范的是另一种认证头**：`docs/help/home/org_project/pipeline/security-permissions/token-permissions` 给的是 `Authorization: token $ATOMGIT_TOKEN`，示例域名指向 `atomgit.com`；该页从未说明这种写法是否适用于 v5 REST（不适用，见「认证方式」） | 打开该页看示例；再跑「认证方式」自证矩阵第 4 行 |
| 7 | **两套权限词汇表并存且无映射**：OAuth 页用 `all_*`；流水线页用 `project/pr/issue/note/repository/action` × `read/write/none`。文档既不给两者对照，也不说明 PAT 该按哪套勾 | 对照两页；再比对你自己 token 的 `X-OAuth-Scopes` 响应头属于哪套 |
| 8 | **响应 Schema 滞后**：`get-api-v-5-user` 页面自称 `Example (auto)`，其 Schema 段只列 **15 个字段**（实测响应比它多；多出的字段清单属运行时观测，见「响应字段」节） | 文档页 Schema 段 vs 你自己的 `GET /api/v5/user` 响应，比对键集 |
| 9 | **整合期品牌/域名并存**：docs.gitcode.com 的页面标题已是「AtomGit 帮助文档」、示例域名打 `atomgit.com`，而接口文档的基址仍写 `api.gitcode.com`；**文档没有一处说明这几个域名之间的关系**（为同一套后端、同一 PAT 通用，见「认证方式」→「基址」） | 看任一帮助页标题与示例域名；再用同一 token 分别打两个域名的 `GET /api/v5/user`，比对 `id` 是否相同 |

### 应对原则（由上表推出，非缺陷本身；本 skill 全篇适用）

1. **官方文档只作线索，不作契约**。文档没写的（scope 前提、错误响应形态）一律按「未知」处理，
   在运行时探测后再分支。
2. **本文件里出现的响应体、状态码、scope 名、字段清单都只是「某次观测样本」**，写出来只为让你
   认得形状。服务端可能改，你的环境也可能不同（不同 token 的 scope 集就不一样）——**读你自己
   那次请求的原始输出，别拿样本去套**。因此本文件只写**规则、可识别形状、消歧动作与复核方法**，
   不写观测日期、复现轮数、复验记录与个案侦破过程：那些改变不了你的下一步动作，
   而你也无法凭「作者测过」去采信一个自己验不了的断言。
3. **失败时原样上报并停止**：把实际状态码、响应体、`X-OAuth-Scopes` 打给用户，未知状态码就走
   「原样上报 + 停止依赖它的判断」。**不要为绕过失败而加旁路开关**——绕过是用户/运维的决定，
   不是 skill 替他们做的决定。
4. **需要断言权限时，用服务端回显而不是文档**：凭证**被服务端认下来**时，`X-OAuth-Scopes` 响应头在
   v5 端点上都返回（`/user`、`/repos/<owner>/<repo>` 等），是零成本的权威诊断；但**匿名 /
   凭证不被认**时（`401`、`429`）该头**不返回**。所以「没看到这个头」只能说明「服务端没认下你的
   凭证」，**不能**反推「scope 集为空」。错误响应体里的 `no scopes:<name>` 则是服务端自己说出的 scope 名。
5. **兜底准入（防止本 skill 无限膨胀）**：新增任何旁路/兜底前，必须同时满足
   (a) **正解不可达**（用户确实改不了）(b) **兜底路径自身不引入新失败模式**
   (c) **语义不与主路径冲突**（不允许出现「`exit 0` 但不代表成功」这类状态）。
   三条有一条不满足就不加，改为「原样上报 + 停止」。同一事实只在一处写全，别处引用。

---

## 认证方式

官方文档（<https://docs.gitcode.com/docs/apis/>）明确「当前可用的版本为 `/api/v5`」，
读写一律使用 v5：

| API 版本 | 认证方式 | 端点格式 |
|---------|---------|---------|
| **GitCode API v5** | `?access_token=<token>`（或 `Authorization: Bearer <token>`、`PRIVATE-TOKEN: <token>`） | `/api/v5/repos/<owner>/<repo>/...` |

### 基址（host）：`api.gitcode.com` 与 `atomgit.com` 是同一套后端

文档统一写 `https://api.gitcode.com`。`https://gitcode.com/api/v5/...` 指向同一套 v5 接口
（同一 token 下 GET 返回同一份 JSON；对不存在仓库 POST 也返回同样的 `400 head或base不能为空`）。

`atomgit.com` **也是同一套后端**——这是「官方文档已知缺陷」#9 的结论
（复核方法见该行第三列：用同一枚 PAT 打两个域名的 `GET /api/v5/user` 比对 `id`）：

| 对比项 | `api.gitcode.com` | `atomgit.com` |
|---|---|---|
| `?access_token=<同一 PAT>` | `200` | `200` |
| 响应里的 `id` / `login` / `external_uid` | 一组值 | **与左侧完全相同** |
| `x-uid` / `X-OAuth-Scopes` / `Server` 响应头 | 一组值 | **与左侧完全相同** |
| 响应体里回显的 URL 域名 | `api.gitcode.com` / `gitcode.com` | `api.atomgit.com` / `atomgit.com` |
| **无凭证**（不带认证信息 / `?access_token=` 空值；端点 `GET /api/v5/user`） | `401 {"message":"401 Unauthorized"}` | `403 {"error_code":403,…,"error_message":"no scopes:read_user"}` |
| **凭证已送达但不被识别**（无效 token；`?access_token=` / `Bearer` / `PRIVATE-TOKEN` / `Authorization: token` 四种写法） | `401 {"message":"401 Unauthorized"}`（**与上一行同形**） | `401 {"error_code":401,…,"error_message":"401, token not found"}` |
| 匿名打 `GET /repos/<owner>/<repo>` | `401 {"message":"401 Unauthorized"}` | `400 {"error_code":400,…,"Orchestration error: Invalid header parameter: private-token,…"}` |

> ⚠️ **「无凭证」还有第三种写法：认证头值为空**（`Authorization: Bearer ` 后面没东西——
> `GITCODE_API_TOKEN` 没导出时，本 skill 的 `-H "Authorization: Bearer $GITCODE_API_TOKEN"`
> 就是这个形状）。它在 `api.gitcode.com` 上仍与无效 token 同形（`401`），在 `atomgit.com` 上
> 却是 `401 …token not found`，**不是**上表那行的 `403 no scopes:`。所以上表「无凭证」那行
> 只覆盖前两种写法；**两个域名都先 `echo "${#GITCODE_API_TOKEN}"` 确认非空，再怀疑服务端。**

背景（官方口径）：GitCode 由重庆开源共创科技有限公司运营（依托 CSDN，2023-09-22 与华为云
CodeArts 联合推出）；AtomGit 属开放原子开源基金会。**2025-10-28 双方宣布深度整合升级**，
服务条款（生效 2025-11-21）载明平台整合与账户统一；`docs.atomgit.com` 的 sitemap 里 URL
全部指回 `docs.gitcode.com`。所以现在是「**一套后端、两个门面**」，不是两个独立产品——
不要因此分别建 token、分别适配。

> ⚠️ **上表倒数第二、三行是本 skill 最容易踩的坑，且两个域名的判别规则相反**
> （复核方法：每种认证状态 × 每个域名各发一次只读请求）：
>
> - **`api.gitcode.com`**：无凭证、空值、无效 token、错误认证写法**全部**返回
>   `401 {"message":"401 Unauthorized"}`，**从不**返回 `no scopes:`。所以在该域名上看到
>   `403 no scopes:<name>`，就说明**凭证已被识别且有效、确实缺 `<name>` 这个权限**，
>   直接按 `<name>` 去补，**不必**再怀疑「凭证没送达」。
> - **`atomgit.com`**：无凭证 / 空值 → `403 …no scopes:read_user`；带了凭证但不被识别 →
>   `401 …token not found`。所以在该域名上看到 `403 no scopes:`，含义是「**服务端眼里你是匿名的**」，
>   第一嫌疑是凭证没送达（域名或认证写法导致 token 未生效），**不是**缺某个 scope。
>
> ⚠️ **`no scopes:<name>` 这句话本身不是假话**：匿名主体持有的 scope 集合确实是空的，
> 「缺 `<name>` 权限」在字面上成立。问题在于它对「压根没带凭证」与「有效 token 但缺该 scope」
> **同形**，而两者的修复动作**相反**——前者去令牌页加权限**永远修不好**（凭证根本没送达），
> 后者加权限**正好修好**。所以必须先用**域名 + 状态码**消歧，再决定动作，别读到 `no scopes:`
> 就去加权限。判别表见「scope 前提与失败分支」。
>
> ⚠️ 本 skill 一律用 `api.gitcode.com`；**同一份脚本不要混用域名**，便于排查与替换。

### 认证写法：先自证，再采信

`?access_token=` query 参数、`Authorization: Bearer <token>` 请求头、
`PRIVATE-TOKEN: <token>` 请求头三种均可用；而**官方流水线页示范的 `Authorization: token <token>`
在两个域名上都返回 `401`**——那是流水线内部注入令牌的专用格式，不适用 v5 REST
（文档缺陷见 #6）。**这四种写法在你的环境里是否同样成立，请自己跑一遍下面的矩阵确认**：

```bash
# 认证方式自证矩阵：4 种写法 × 2 个域名的真实状态码（只读端点，无副作用）
T="$GITCODE_API_TOKEN"
for H in api.gitcode.com atomgit.com; do
  U="https://$H/api/v5/user"
  echo "--- $H"
  printf '%-26s %s\n' "?access_token="        "$(curl -s --connect-timeout 10 --max-time 30 -o /dev/null -w '%{http_code}' "$U?access_token=$T")"
  printf '%-26s %s\n' "Authorization: Bearer" "$(curl -s --connect-timeout 10 --max-time 30 -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $T" "$U")"
  printf '%-26s %s\n' "PRIVATE-TOKEN"         "$(curl -s --connect-timeout 10 --max-time 30 -o /dev/null -w '%{http_code}' -H "PRIVATE-TOKEN: $T" "$U")"
  printf '%-26s %s\n' "Authorization: token"  "$(curl -s --connect-timeout 10 --max-time 30 -o /dev/null -w '%{http_code}' -H "Authorization: token $T" "$U")"
done
```

> ⚠️ **官方文档把 `access_token`（query 参数）标为 `required`、且完全不提请求头写法**
> （见「官方文档已知缺陷」#4）。query 形式会把长期凭证带进 shell history、CI 日志、服务端
> access log 与代理日志；**共享环境或 CI 里请优先用请求头**。本 skill 下文示例为简洁仍写 query
> 形式，这是**已知取舍**，不是推荐做法。
>
> ⚠️ token 错误或缺失时，`api.gitcode.com` 返回 `401 {"message":"401 Unauthorized"}`
> （不是 `403`；换到 `atomgit.com` 时形状不同，见上面的域名对比表），
> **且 `curl -s` 的退出码仍是 0**——判成功要看 HTTP 状态码或校验响应里有没有期望的键。
> 该响应体同样是观测样本，你的环境以你自己那次请求的输出为准。

> ⚠️ **`401` 不一定是 token 的问题**：`${owner}`/`${repo}` 未赋值时 URL 会拼成
> `…/repos///pulls/<n>`，若此时 token 也缺失 / 无效，返回的正是上面那个 `401`，与「路径正确、
> token 失效」**响应体一字不差**。四格（只读的 `GET /pulls/<n>`）：
>
> | URL | token | 响应 |
> |---|---|---|
> | `…/repos///pulls/<n>`（变量为空） | 有效 | `404`，Spring 体：`{"timestamp":…,"status":404,"error":"Not Found","path":"/api/v5/repos///pulls/<n>"}` |
> | `…/repos///pulls/<n>`（变量为空） | 缺失 / 无效 | `401 {"message":"401 Unauthorized"}` |
> | 路径正确 | 有效 / **缺失** | `200`（该接口公开可读，无 token 也返回完整 JSON） |
> | 路径正确 | 无效 | `401 {"message":"401 Unauthorized"}` |
>
> 判别：`404` 的 `path` 字段会**原样回显请求路径**，连续斜杠（`repos///pulls`）一眼可见 →
> 变量没填上；`401` 则两种原因都可能（路径正确时无 token 也能 `200`，故 `401` 必是 token 无效，
> 或变量为空叠加 token 无效）。**先 `echo "$owner/$repo"` 确认变量非空，再怀疑 token。**

### 匿名（无 token / 空 token）时各步骤的真实行为

> ⚠️ **本节全是运行时观测样本，不是契约**（应对原则第 2 条）：请在自己的环境按上面「认证写法」
> 的自证矩阵复跑一遍再采信。`SKILL.md`「获取访问令牌（第一步必须）」只写**规则结论**
> （哪几步匿名能跑、哪几步必然失败、该守哪三处），状态码 / 响应体 / `429` 细节
> **只落在本节**，两处不要各写一份（应对原则第 5 条末句）。

下表是 `api.gitcode.com` 上的观测样本：

| 步骤 | 无 token / `access_token=`（空） | 结论 |
|---|---|---|
| 读 PR 详情 `GET /pulls/<n>`、读评论列表（含分页头 `total_page`/`total_count`） | `200`，返回完整数据 | **公开可读**（但匿名共用额度桶，见下面 ⚠️） |
| 读仓库信息 `GET /repos/<owner>/<repo>`（`SKILL.md`「查询 Fork 的原仓库」那步用它） | `401 {"message":"401 Unauthorized"}` | **需要凭证**：fork 判定这步匿名跑不了。该步的 `jq` 守卫会把这个对象判成「响应异常」，走 ERROR 分支并 `exit 1` |
| `GET /api/v5/user`（取 `MY_LOGIN`） | `401 {"message":"401 Unauthorized"}` | 「请求示例：先探测，再取字段」的脚本按守卫 `exit 1`，「是否已评论过」无法判断 |
| `POST` 发评论 / 回复、`DELETE` 删评论 | `401 {"message":"401 Unauthorized"}` | **必须校验 HTTP 状态码**：`curl -s` 退出码仍是 `0`，只看 `$?` 会把「没发出去」当成「发成功了」 |

> ⚠️ **匿名调用共用服务端账号的额度桶**：匿名打只读端点会撞上 `429`，其 `error_message`
> 自报被限的账号与门槛，形状是
> `Threshold: 10W times per user per hour. Limited account:<某个不是你的账号>`——限的是匿名调用
> 共用的那个服务端账号，**不是你**。所以**何时恢复不由你决定**：别承诺「等 N 秒重试」，
> 也别把匿名当可用路径，「能匿名跑」≠「能连着匿名跑」。`429` 的响应体是 **JSON object**、
> 不带 `total_page`/`total_count`，翻页循环的非数组守卫（或缺头时的守卫①）会把它拦成 `exit 1`
> ——**别读成「PR 不存在 / 没有评论」**。
>
> ⚠️ 上表的匿名 `401` 与「无效 token」的 `401` **同形**（该域名不区分，见「scope 前提与失败分支」）；
> 所以看到 `401` 先 `echo "${#GITCODE_API_TOKEN}"` 确认凭证真的非空，再怀疑 token 本身失效。

---

## 获取当前用户（token 对应的账号）

### API 端点

取「当前 token 对应的账号」，用于把评论的 `user.login` 与「本 agent」比对
（`commands/review.md` 步骤 1 判断是否已评论过、`SKILL.md` 回读校验里「按作者 + 唯一标记匹配」
这条备选判据；备选 ≠ 兜底/旁路，它不掩盖失败，只是首选 `note_id` 比 `.id` 之外的另一种匹配方式）。

```bash
GET https://api.gitcode.com/api/v5/user
```

> ⚠️ **不带路径参数**：`/api/v5/user` 是「当前认证用户」，`/api/v5/users/<login>` 才是按 login 查任意用户。
> 本 skill 只用前者。

### 请求示例：先探测，再取字段

> ⚠️ **不要写成 `curl -s … | jq -r '.login // empty'`**（官方文档只给了 `200` 的 Schema，照它写
> 就会漏掉失败分支，见「官方文档已知缺陷」#3）：本端点可能返回 `401`/`403`，而 `curl -s` 的退出码
> 恒为 `0`、jq 对错误体取 `.login` 得空串——「权限不够」「token 失效」「变量没导出」「网络不通」
> 四种故障会退化成同一个空输出。必须把 **HTTP 状态码、响应体、`X-OAuth-Scopes` 响应头**一起留住。

```bash
# 一次请求同时留住三样：HTTP 状态码 + 响应体 + 服务端回显的「本 token 实际持有的 scope」
# ⚠️ -w 用字面量标记 `__HTTP__` 分隔 body 与状态码，`${var%…}`/`${var##…}` 是 POSIX 参数展开，
#    bash 与 dash 下都能跑；换到别的 shell 前自己验一次。
#    若改用 `-w '\n%{http_code}'` + `${var%$'\n'*}` 切分则是 **bash 专有**（`$'…'` 为 ANSI-C quoting）：
#    dash 下 `$'\n'` 被当字面量、切分失效，`${resp%$'\n'*}` 原样返回「响应体+状态码」整串，
#    jq 报 `Cannot index number with string "login"`，状态码变量里是整个响应体。
# ⚠️ --connect-timeout/--max-time 必带：不带的话网络挂起会无限等待；超时后状态码是 000，走失败分支。
hdr=$(mktemp) || exit 1
trap 'rm -f "$hdr"' EXIT   # 嵌进更大脚本时，把这个 rm 合并进已有的 trap，别覆盖它
# -D 把响应头落到文件：curl -s 会丢掉响应头，X-OAuth-Scopes 就读不到了
resp=$(curl -s --connect-timeout 10 --max-time 30 -D "$hdr" -w '__HTTP__%{http_code}' \
  "https://api.gitcode.com/api/v5/user?access_token=$GITCODE_API_TOKEN")
body=${resp%__HTTP__*}      # HTTP 响应体
status=${resp##*__HTTP__}   # HTTP 状态码
# 响应头是 CRLF 结尾，必须 tr -d '\r'
scopes=$(tr -d '\r' < "$hdr" | awk -F': *' 'tolower($1)=="x-oauth-scopes"{print $2}')
MY_LOGIN=$(printf '%s' "$body" | jq -r '.login // empty')
# 失败判据要覆盖两种「没拿到 login」：非 200，以及 **200 但 .login 为空**。只按状态码分支会漏掉后者。
if [ "$status" != "200" ] || [ -z "$MY_LOGIN" ]; then
  echo "WARN: GET /api/v5/user 未取到 login（HTTP ${status}）" >&2
  # 响应体自带换行，直接 echo 会多打一个空行，故先 tr 掉（POSIX 安全，bash/dash 均可）
  echo "      响应体：$(printf '%s' "$body" | tr -d '\r\n')" >&2
  echo "      本 token 的 X-OAuth-Scopes：${scopes:-<无此响应头>}" >&2
  # 各状态码的含义与处置**不在此重抄**：拿这三件套去查下面「scope 前提与失败分支」的判别表。
fi
[ -n "$MY_LOGIN" ] || { echo "ERROR: 取不到当前账号 login，无法判断是否已评论过（HTTP ${status}）" >&2; exit 1; }
echo "MY_LOGIN=$MY_LOGIN"
echo "token scopes: ${scopes:-<无此响应头>}"
```

> **`MY_LOGIN` 的取法全仓只此一份**，`SKILL.md` 回读校验的「作者 + 唯一标记」备选判据照抄这段。
> 拿到后与评论列表的 `author` 比对即可判断「本 agent 是否已评论过」（含回复）：
> `<翻页脚本输出> | jq -c --arg me "$MY_LOGIN" 'select(.author == $me) | {id, reply_to, body}'`
> ——有输出 = 已评论过，无输出 = 未评论过。
> ⚠️ 过滤器别带 `.[]`：翻页脚本输出的是**逐行对象流**，带 `.[]` 是对每个对象的**值**再迭代，
> 会报 `Cannot index number with string "author"`（要先聚成数组再 `.[]` 得加 `-s`）。
> ⚠️ 「无输出」只有在评论列表**翻页取全并展开 `reply[]`** 后才可信，取不全时不得判断「未评论过」：
> 见 `commands/review.md` →「取评论：翻页取全（规范实现）」。取不到 login 时按上面 `exit 1`
> 原样上报三件套，**不要绕过**。

> **`X-OAuth-Scopes` 是服务端回显，比任何文档都权威**，凭证被认下来时实测过的 v5 端点都返回它
> （零成本诊断）；**匿名 / 凭证不被认时不返回**（`401`、`429`），故「没这个头」= 服务端没认你的
> 凭证，不等于「scope 集为空」（详见「应对原则」第 4 条）。
> 想知道「我的 token 到底有哪些权限」，看这个头，别去猜令牌页的勾选项叫什么名字。

### 响应字段（`200`；**一次观测样本，非契约**）

顶层是**对象**（不是数组）。本 skill **只依赖 `login` 这一个字段**——这是唯一需要依赖的字段。
下表其余条目仅为某次观测的记录：官方文档的 Schema 少列了若干字段
（见「官方文档已知缺陷」#8），所以**别按文档 Schema 做严格解析**，也别把下表当完整清单。

| 字段 | 类型 | 说明 |
|------|------|------|
| `login` | string | **账号 login**，与评论的 `user.login`、`/api/v5/user` 的 `html_url` 尾段一致 |
| `name` | string | 显示名称（实测与 `login` 同值，不保证） |
| `id` | string | 用户 ID（**字符串**，形如 `6822e242096b0e7b44aa67dc`；不是数字）。两个域名下**同一账号取值相同**，可用来验证「同后端」 |
| `external_uid` | number | 外部 UID（数字） |
| `type` | string | 实测 `User` |
| `html_url` | string | 用户主页 `https://gitcode.com/<login>`（打 `atomgit.com` 时回显为 `https://atomgit.com/<login>`） |
| `url` | string | 实测为 `https://api.gitcode.com/api/v5/user/<login>` |
| `avatar_url` / `followers_url` | string | 头像、粉丝列表地址 |
| `email` / `bio` / `blog` / `company` / `location` / `website` / `description` / `github_account` | string | 资料字段，未填写时为空串 |
| `followers` / `following` | number | 粉丝数 / 关注数 |
| `top_languages` | array | 主要语言列表 |
| `created_at` / `updated_at` | string | 时间戳（`+08:00` 带时区） |

### scope 前提与失败分支：**运行时自证，不要背样本**

**官方文档不记载本端点的任何 scope 前提**（见「官方文档已知缺陷」#1、#2），所以本 skill
**不断言「必须有某个 scope」**。正确做法是：在**你自己的环境**里跑一次上面的探测，按
**实际观测到的状态码**分支——下表是「观测到过哪些形状」，不是「服务端承诺的行为」：

| 你观测到的 | 含义 | 处置 |
|------|------|------|
| `200` 且 `.login` 非空 | 权限够 | 用它 |
| `200` 但 `.login` 为空 | 响应形状与文档 Schema 不符（缺陷 #8 的另一面：Schema 只列 15 个字段，实际响应键集可能不同） | 原样上报响应体；**别拿 `.name` / `.id` 等别的字段凑一个 login 出来**——凑出来的值会让「是否已评论过」的判断失效 |
| `403` + `no scopes:<name>`，域名是 **`api.gitcode.com`** | **凭证已被识别且有效，确实缺 `<name>` 权限**。依据：该域名对「无凭证 / 空值 / 无效 token / 错误认证写法」**一律**返回 `401 {"message":"401 Unauthorized"}`、**从不**返回 `no scopes:`（见「认证方式」→「基址（host）」的域名对比表） | 按**响应体里的 `<name>`** 去令牌页补权限；令牌页找不到同名勾选项时属文档缺陷 #1，把 `<name>` 与令牌页实际选项原样上报，别硬猜 |
| `403` + `no scopes:<name>`，域名是 **`atomgit.com`** | **服务端眼里你是匿名的**。依据：该域名对「带了凭证但不被识别」返回的是 `401 …token not found`，只有「压根没带 / 空值」才返回这个 `403`（见「认证方式」→「基址（host）」的域名对比表） | 先确认域名与认证写法、`GITCODE_API_TOKEN` 是否真的非空（`echo ${#GITCODE_API_TOKEN}`）；换 `api.gitcode.com` 重试一次再判。**此时去令牌页加权限是无效的** |
| `401`（`api.gitcode.com`） | 无凭证 / 空值 / 无效 token / 错误认证写法——**四种情形同形，该域名不区分** | 先 `echo ${#GITCODE_API_TOKEN}` 确认非空，再确认用的是本 skill 三种写法之一，最后才怀疑 token 本身失效 |
| `401`（`atomgit.com`，响应体含 `token not found`） | 凭证送到了但不被识别 | 换有效 token；确认没有多空格 / 换行（`tr -d '[:space:]'`） |
| `000` | 超时 / DNS / 网络不通（`--max-time` 生效） | 检查网络与代理；**不要**据此判断权限 |
| 其它状态码 / 响应体非预期 | 未知 | **原样把状态码、响应体、`X-OAuth-Scopes` 报给用户**；停止依赖 login 的判断（如「是否已评论过」），别猜、别套本表样本 |

> ⚠️ **为什么不能「读到 `no scopes:` 就去加权限」**：这句话字面上不是假话——匿名主体持有的
> scope 集合确实是空的。但它对「压根没带凭证」与「有效 token 缺该 scope」**同形**，而两者的
> 修复动作**相反**（前者加权限永远修不好，后者加权限正好修好）。**消歧只靠域名 + 状态码**，
> 见上表两行 `403`。

> ⚠️ **`<name>` 以你自己那次请求的响应体为准，不要背名字**：见过的取值是 `read_user`，
> 但服务端接受的名字**不止一个、也可能随时间变化**；拿它去核对令牌页很可能对不上号（缺陷 #1）。
>
> ⚠️ **权限类 `403` 只影响这一步**：同一 token 请求 `GET /api/v5/repos/<owner>/<repo>`、
> `GET /api/v5/repos/<owner>/<repo>/pulls/<n>` 与评论读写接口仍是 `200`，即「评论能正常读写、
> 只有取 login 失败」。所以别把这条 `403` 误判成 token 整体失效而去换 token——**但这条同样要在
> 你的环境里验一遍**（用同一个 token 打一次仓库读接口即可），不要直接采信。
>
> ⚠️ **只 `jq -r '.login // empty'` 会把状态码、响应体、scope 回显全丢掉**：`401`、`403`、
> 网络失败三种情形都退化成「取不到 login」，无法区分是权限不够还是 token 失效，
> 下游会带着空 login 继续跑（正是「官方文档已知缺陷」#3 的直接后果）。
>
> **补不上权限怎么办**：原样上报状态码 + 响应体 + `X-OAuth-Scopes`，然后**停止依赖 login 的判断**
> （如「是否已评论过」），不得据空 login 判断「未评论过」。本 skill **不提供绕过开关**——理由见
> 「应对原则」第 3、5 条（旁路开关会引入「`exit 0` 但不代表成功」的语义冲突，且自身在离线场景
> 仍会卡在网络请求上）。

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
curl -s --connect-timeout 10 --max-time 60 -H "Authorization: Bearer $GITCODE_API_TOKEN" \
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
response=$(curl -s --connect-timeout 10 --max-time 60 -H "Authorization: Bearer $GITCODE_API_TOKEN" \
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
curl -s --connect-timeout 10 --max-time 60 "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>?access_token=$GITCODE_API_TOKEN"
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
（依据见本节下方「翻页终止以响应头 `total_page` 为准」）。本文件下方「批量清理脚本」第 1 步
内联的那份循环是它的**删除场景变体**（多带唯一标记筛选与 `405` 预检），不算第二份通用循环。

本节只给**字段投影的全集版**（含 `resolved` / `diff_position`；`SKILL.md` 那份按用途裁掉了
这两个字段，差异是有意的、不是漏抄）。下面这条**只取第一页**，用于看单页字段形状，
不要拿它当「取全」的实现：

```bash
curl -s --connect-timeout 10 --max-time 60 "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/comments?access_token=$GITCODE_API_TOKEN&per_page=100" | \
  jq '.[] | {id, discussion_id, comment_type, author: .user.login, resolved, lines: .diff_position, body} as $c
      | $c, (.reply[]? | {id, discussion_id: $c.discussion_id, comment_type: $c.comment_type, author: .user.login, body, reply_to: $c.id})'
```

> **失败必须显式**：无效 token（`{"message":"401 Unauthorized"}`）与不可解析域名（空响应）
> 都会让响应不是数组，该分支必须 `exit 1`（完整理由，以及「嵌进更大脚本不能直接退出时置
> `incomplete=1` 标志」的写法与调用方义务——**必须自行以非 0 退出**，不是兜底——见
> `commands/review.md` →「六条硬规则」第 3 条）。

> **翻页终止以响应头 `total_page` 为准**（上面的循环就是这么写的）。该接口在响应头返回
> `total_count`（父评论总数）与 `total_page`（总页数），`total_page = ceil(total_count / 生效页长)`。
>
> **为什么不能用「本页条数 < 请求的 `per_page`」判终止**：请求值超过上限会被服务端**静默夹回 100**，
> 而 `total_page` 按**生效**页长算、不受夹影响。夹回后若终止判据写成 `count < 150`（请求
> `per_page=150`），第一页只有 100 条 → 第一页就 `break`，后面的评论**全部静默丢掉**；同一响应的
> `total_page` 仍按生效页长 100 算，按它翻页不会漏。
>
> **这条判据不依赖夹回是否真发生**：`total_page` 夹不夹都对，而「本页条数 < 请求的 `per_page`」
> 在夹回时第一页就停（漏读），不夹回时也只是碰巧对（依赖「服务端不夹」这个未承诺行为）。
> 夹回是否发生、发生在哪些端点，用**列表端点**（`GET /repos/<owner>/<repo>/pulls`，
> `total_count` 通常上百）自查最快：请求 `per_page=150`，看返回条数是不是 100、
> `total_page` 是不是按 100 算（`= ceil(total_count / 100)`）。
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
> **越界 `page` 的行为**（「体空 ⟹ 必已越界」这条守卫的依据）：越界页返回**空数组 + HTTP 200**，
> 不会回退返回第一页。
> ⚠️ 别把这句读成「`page=2` 一定返回空」：**页号在 `total_page` 范围内时正常返回数据**——
> 把 `per_page` 调小让 `total_page` 变大，范围内的每一页都有数据；「空数组」只发生在
> `page > total_page` 时。
> 且**响应头与响应体出自同一次响应**：请求越界页时体是 0 条、头仍照常写
> `total_count`/`total_page`。
> 推论：**合法的「本页为空」必然伴随 `page >= total_page`**，所以「体空但 `page < total_page`」
> 只能是服务端自相矛盾，据此 `exit 1` 的误报面≈0（零评论 PR 的 `total_page` 是 `1`，`page=1` 时
> `1 < 1` 不成立，不会被误伤）。
> `page=0`/负数/非数字返回 **HTTP 400 + 对象体**，由上面的非数组守卫 `exit 1` 覆盖。
> 复核方法：任取一个多评论 PR，先用 `per_page=5` 请求「范围内页」与「越界页」各一次，比对体与头。
>
> ⚠️ **`total_page` 缺失时不要退回任何猜测判据**（如「本页为空就算翻完」「本页不满 `per_page`
> 就算最后一页」）：那会把「不知道取全没有」伪装成「取全了」，正是本 skill 最忌讳的静默假成功。
> 缺头就 `exit 1` 原样上报（应对原则第 3 条）。
>
> 该接口**不返回** `Link` header，不要按 GitHub 的 `rel="next"` 判断翻页。
> 单条资源接口（如 `GET /pulls/<n>`）**没有**这两个分页头，只有列表接口才有。

**查询参数**：

| 参数 | 说明 |
|------|------|
| `page` / `per_page` | 分页；`per_page` 默认 **20**、最大 **100**（超出会被**静默夹回 100**）。评论较多时必须显式传，超过一页时必须从 `page=1` 起递增翻页取完，终止判据用响应头 `total_page`（**不要**用「本页条数 < 请求的 `per_page`」），否则第 2 页之后会被漏读。夹回行为与判据的依据见上「翻页终止以响应头 `total_page` 为准」 |
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
curl -s --connect-timeout 10 --max-time 60 -X POST \
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
curl -s --connect-timeout 10 --max-time 60 -X POST \
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
curl -s --connect-timeout 10 --max-time 60 -o /dev/null -w '%{http_code}\n' -X DELETE \
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
> ⚠️ **批量删除的失败形态未经真实验证**：先「只看清单不执行」跑一遍——保留第 1、2 步
> （取全 ID + `405` 预检），只把第 3 步的 `DELETE` 换成打印（跑法见下「批量清理脚本」末尾那条），
> 确认待删 ID 与预检结果符合预期后再真删。

### 获取评论 ID

本节不再单列一份通用翻页循环：清理场景直接用下面「批量清理脚本」的第 1 步
（它已把翻页、`reply[]` 展开、唯一标记筛选、405 预检合在一起）；要另写筛选条件，
就用 `SKILL.md` →「获取 PR 讨论列表（包含行内评论）」那份循环（分页规则与依据见上）。

⚠️ **四条会「少删却 `exit 0`」的坑**（批量清理脚本已逐条落地，改写时别丢）：

1. **不能边翻页边删**：删除会让列表前移，被跳过的条目永远删不到。必须**先取全 ID（只读不删）
   再统一删**；坚持边翻边删就得按 `total_page` **倒序**翻页。
2. **不能在循环里覆盖同一个变量、循环外再删**：那样只剩最后一页的 ID，评论超过 100 条时
   前面几页全丢且脚本仍 `exit 0`。必须在循环内**追加**到累积容器（临时文件/变量）。
3. **`.body` 可能为 `null`**（行内 diff 评论、系统评论）：对 `null` 调 `startswith` 会让 jq
   **中途报错退出**——崩之前输出的 ID 算数、其后的全丢，外层 `for` 又不检查 jq 退出码，
   于是拿着残缺名单跑完并 `exit 0`。必须写 `(.body // "")` 并显式判断 jq 退出码。
4. **排序键只放 `reply_to` 标志 + id，不要掺 `.body`**：`body` 常带换行而 `sort` 按行排序，
   掺进去会把一条评论拆成多行重排，「回复在前」的顺序读不出来（复核：比对输出行数与评论条数，
   相等才说明没被拆开）。要看正文另起一条 `jq -c` 命令。

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
MAX_PAGES=50   # 上界（50 × 100 = 5000 条父评论），正常永不触发；
               # 只防服务端分页异常导致空转，撞上界必须 exit 1，不能 break 后以 0 退出。
               # 「响亮失败的上界」，不是「应对原则」第 5 条禁止的绕过式兜底（见 SKILL.md 同一处注释）
page=1
sum=0; tc_first=""; tc_last=""   # 守卫③的总账：Σ父评论条数、首页与末页 total_count
while :; do
  [ "$page" -gt "$MAX_PAGES" ] && { echo "ERROR: 翻页超过 ${MAX_PAGES} 页上界（末页 http=${http_status:-未取到}），ID 未取全" >&2; exit 1; }
  # -D 把响应头落到文件；头是 CRLF 结尾，取值前必须 tr -d '\r'，否则数值比较会失败
  resp=$(curl -s --connect-timeout 10 --max-time 60 -D "$hdr" "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/comments?access_token=$GITCODE_API_TOKEN&per_page=100&page=${page}")
  # 状态码从 -D 落盘的头文件**第一行状态行**取，不必为此多发一次请求；
  # 连接失败 / 超时时头文件是空的（curl 的 %{http_code} 此时为 000），故取不到就用固定文案兜住。
  http_status=$(head -n1 "$hdr" | awk '{print $2}' | tr -d '\r')
  http_status=${http_status:-000(无响应头：连接失败或超时)}
  total_page=$(tr -d '\r' < "$hdr" | awk -F': *' 'tolower($1)=="total_page"{print $2}')
  total_count=$(tr -d '\r' < "$hdr" | awk -F': *' 'tolower($1)=="total_count"{print $2}')
  if [ "$(echo "$resp" | jq -r 'type' 2>/dev/null)" != "array" ]; then
    echo "ERROR: 第 ${page} 页评论请求失败或响应非数组（http=${http_status}），ID 未取全：$resp" >&2
    exit 1
  fi
  # 守卫①：缺分页头就无从判断 ID 是否取全 —— 必须失败，**不**退回「本页为空」等猜测判据
  #        （删除场景漏 ID 的后果是「少删却 exit 0」，比只读场景更严重）
  [ -n "$total_page" ] && [ -n "$total_count" ] || {
    echo "ERROR: 第 ${page} 页响应缺 total_page/total_count 头（http=${http_status}），无法判断 ID 是否取全" >&2; exit 1; }
  # reply_to 非 null（回复）标记为 1，父评论为 0；(.body // "") 兜住 null，否则 jq 会中途死掉丢 ID
  echo "$resp" | jq -r '
      .[] | . as $c
      | ({id, body, reply_to: null}, ((.reply // [])[] | {id, body, reply_to: $c.id}))
      | select((.body // "") | startswith("<唯一标记>"))
      | "\(if .reply_to then 1 else 0 end)\t\(.id)"' >> "$ids" \
    || { echo "ERROR: 第 ${page} 页 jq 解析失败（http=${http_status}），ID 未取全" >&2; exit 1; }
  # 405 预检：被选中的父评论若还有**未被选中的存活回复**，删它必然 405，翻页阶段就能算出来
  echo "$resp" | jq -r '
      .[] | select((.body // "") | startswith("<唯一标记>")) as $p
      | ($p.reply // [])[]
      | select(((.body // "") | startswith("<唯一标记>")) | not)
      | "父 \($p.id) 的回复 \(.id) 未被筛选条件命中"' >> "$blocked" \
    || { echo "ERROR: 第 ${page} 页 405 预检 jq 解析失败（http=${http_status}）" >&2; exit 1; }
  # count 数的是**父评论**条数（回复不计入分页，也不计入 total_page/total_count）
  count=$(echo "$resp" | jq 'length')
  # 守卫②：头与体出自**同一次响应**，「体空但 page < total_page」是服务端自相矛盾，不是「翻完了」
  #        （判据推导、零评论 PR 不会误伤的分析与形状样本，全仓只写一份：见本文件
  #          「翻页终止以响应头 `total_page` 为准」→「越界 `page` 的行为」）
  [ "$count" -eq 0 ] && [ "$page" -lt "$total_page" ] && {
    echo "ERROR: 第 ${page} 页为空但 total_page=${total_page}（http=${http_status}），响应自相矛盾，ID 未取全" >&2; exit 1; }
  [ "$page" -eq 1 ] && tc_first=$total_count
  sum=$((sum + count)); tc_last=$total_count
  [ "$page" -ge "$total_page" ] && break
  page=$((page + 1))
done
# 守卫③：Σ父评论条数必须等于 total_count，对不上就是 ID 没取全——此时**一次 DELETE 都不能发**。
# **不自愈**：不自动重跑、不打 WARN 后继续，重跑是人的决定。
# 下面几行 echo 是脚本必须自带的**运行时输出**，不是第二份说明；判读规则的**说明**全仓只一份，
# 连同「哪些场景不适用守卫③」见 SKILL.md 的守卫③注释。
[ "$sum" -eq "$tc_last" ] || {
  echo "ERROR: ID 未取全 —— Σ父评论条数=${sum} ≠ total_count=${tc_last}（首页 total_count=${tc_first}，翻了 ${page} 页）" >&2
  echo "       首末 total_count 不同 → 翻页期间有人增删评论（移动靶），重跑即可；" >&2
  echo "       首末相同但 Σ 与 total_count 不等 → 请把上面数字原样上报，不要执行任何删除：" >&2
  echo "         Σ 偏小 = 服务端漏数据；Σ 偏大 = 翻页期间有评论被删、列表前移导致同一条被读了两次。" >&2
  exit 1; }

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
  code=$(curl -s --connect-timeout 10 --max-time 60 -o /dev/null -w '%{http_code}' -X DELETE \
    "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/comments/${id}?access_token=$GITCODE_API_TOKEN")
  [ "$code" = "200" ] || { echo "ERROR: 删除评论 ${id} 失败（http=${code}），清理不完整" >&2; fail=1; }
done
[ "$fail" -eq 0 ] || exit 1
```

> 六条不变式对应关系：① 先取全再统一删（第 1 步 / 第 3 步分离）② 响应非数组即 `exit 1`
③ 展开 `reply[]`（第 1 步的 jq 投影）④ 405 预检（第 2 步，有阻塞项一次 `DELETE` 都不发）
⑤ 回复在前、父评论在后（`sort -r`）⑥ 逐条校验 `-w '%{http_code}'`。
> 想先看清单再删（不自动执行）：保留第 1、2 步（预检有阻塞项时它会照常 `exit 1` 报出来），
> 把第 3 步的 `DELETE` 换成打印 `sort -r "$ids"`。
> `SKILL.md` 只保留这六条不变式的清单与指针，脚本全仓以此处为唯一一份，别在两处各写一遍。

### 权限说明

- 只能删除自己创建的评论
- 仓库管理员可以删除任何评论
- 删除操作不可恢复

---

## 获取 PR 文件变更

```bash
# 文件级变更列表，每项含 patch.diff（unified diff，带 @@ hunk header）
curl -s --connect-timeout 10 --max-time 60 "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/files?access_token=$GITCODE_API_TOKEN"

# 需要整体统计与 diff_refs（base_sha / start_sha / head_sha）时用 files.json
curl -s --connect-timeout 10 --max-time 60 "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/pulls/<PR_NUMBER>/files.json?access_token=$GITCODE_API_TOKEN" | \
  jq '{count, added_lines, remove_lines, diff_refs}'
```

**`files` 返回字段说明**：

| 字段 | 说明 |
|------|------|
| `filename` | 文件路径 |
| `sha` | **提交 sha**（实测等于 `diff_refs.head_sha`），**不是 blob sha**；blob sha 在 `blob_id` |
| `blob_id` | blob sha（实测等于 `git rev-parse <commit>:<path>`），可拼成稳定的 `…/blobs/<blob_id>/<path>` 取内容地址 |
| `blob_url` | 网页版文件页（`gitcode.com/<owner>/<repo>/blob/<commit_sha>/<path>`），不是取原始内容的地址 |
| `raw_url` | 原始内容地址，但形式是 `…/raw/<commit_sha>/<path>`，**不稳定**（见下文「获取文件原始内容」），别当唯一手段 |
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
curl -s --connect-timeout 10 --max-time 60 "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/contents/<文件路径>?ref=<commit_sha>&access_token=$GITCODE_API_TOKEN" | \
  jq -r '.content' | base64 -d
```

**返回字段**（实测）：`type` / `encoding` / `size` / `name` / `path` / `content`（base64）/
`sha`（**blob sha**）/ `url` / `html_url` / `download_url` / `_links`。

`download_url` 是 `https://raw.gitcode.com/${owner}/${repo}/blobs/<blob_sha>/<文件路径>` 形式，
直接返回明文、稳定 `200`（复核：下载内容与本地同一文件 md5 比对），
适合管道给 `grep -n` 定位行号。

> ⚠️ **`…/raw/<commit_sha>/<文件路径>` 这个形式不稳定，不要当唯一手段**：同一 URL、同一 sha
> 可能间歇返回 `403`（body「暂不支持预览」），过一会儿又恢复 `200`，且**命中哪些文件也会变**，
> 与文件内容无关。取内容请用上面的 `download_url`（`…/blobs/<blob_sha>/<path>`）。
> `/pulls/<n>/files` 响应里的 `raw_url`、`files.json` 里 `diffs[].head.url` 都是该形式，同样不可靠。

**`ref` 参数取值（实测）**：commit sha（40 位全 sha 或短 sha）与**分支名都可用**——
`ref=<默认分支名>`、`ref=<带点的版本分支名>`、`ref=<带斜杠的特性分支名>`（原样或 `%2F` 编码均可）
都返回 `200`，且分支名取到的内容与本地 `git show <remote>/<branch>:<path>` 的 md5 一致；
不传 `ref` 时用默认分支。跨 fork PR 的 head commit 在**目标仓库**路径下也取得到
（复核：目标仓库与源仓库两条路径都 `200`、blob sha 相同）。

> ⚠️ 审查 PR 时**仍应传 head commit sha**，不要用分支名：分支名指向分支**当前**位置，
> PR 提交后分支若又推了新 commit，取到的就不是被审查的那一版，`grep -n` 得到的行号会与 diff 对不上。
>
> ⚠️ **`404` 区分不了「ref 不存在」和「文件在该 ref 上不存在」**：两种情况响应体完全相同，
> 都是 `{"error_code":404,"error_code_name":"NOT_FOUND","error_message":"项目或文件不存在","trace_id":"…"}`
> （三种组合返回一模一样：「不存在的 ref + 存在的文件」；「存在的 ref + 该 ref 上没有的文件」；
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
   越界 `page` 行为与相关依据见「获取 PR 讨论（包含行内评论）」
4. **两种 ID 不要混用**：`discussion_id`（哈希串，用于回复）vs 评论 `id`（整数，用于删除）
