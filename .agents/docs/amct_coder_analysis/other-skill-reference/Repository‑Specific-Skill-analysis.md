# 业内 Skill 参考（superpowers 以外）与 amct-coder 对比

> 整理日期：2026-09-01。
> 姊妹文档：`superpowers-reference-analysis.md`（`obra/superpowers` 单独成文，含 amct-coder 的 A/B 实测）。

---

## 1. 分类：两种 skill，amct-coder 属于后者

业内 skill 分两类，形态和价值主张完全不同：

| | 通用流程型 | 厂商/仓库专属型 |
| --- | --- | --- |
| 教什么 | 约束 agent 的工作流程（先规划、先找根因、先验证） | 特定代码库/构建体系/硬件的硬约束 |
| 代表 | `obra/superpowers`、`mattpocock/skills`、`addyosmani/agent-skills` | `cloudflare/skills`、`get-convex/agent-skills`、`huggingface/kernels@cuda-kernels` |
| 核心价值 | 补充纪律 | **否决默认知识**（见 §2.2） |
| 维护方 | 第三方作者 | 项目方自己 |

`amct-coder` 的真实分类是**仓库专属型**。superpowers 只在"形态"上相似（都不教具体 API 而是约束流程），所以它单独成文；本文集中处理和 amct-coder **同类**的参考

---

## 2. 主要参考：两个商业厂商样板，取舍正好相反

`cloudflare/skills` 和 `get-convex/agent-skills` 都是项目方自己维护的仓库专属 skill 库，但切分策略相反。这个对比本身就是最好的答案：**这件事没有唯一解，关键是切分依据要和领域结构对齐。**


### 2.1 两种相反的切分策略

| | `cloudflare/skills` | `get-convex/agent-skills` |
| --- | --- | --- |
| skill 数 | 13 | 33（全部 `convex-*`） |
| 切分依据 | 按**产品线** | 按**单个能力动作** |
| 入口 | 决策树 | 纯路由表 |


Cloudflare 用大文件 + 决策树，Convex 用极薄文件 + 路由器。**Convex**，一个 skill 只干一件事，靠入口分发。

amct-coder 目前是中间形态：1 个 162 行入口 + 17 个 reference 切片，更接近 Cloudflare 的"厚入口"；但子能力是采用的类似Convex，子能力分发，这个混合是有理由的——AMCT 的**层次结构**稳定，按层切比按动作切更贴合；而不同任务下编码本身就是独立动作

### 2.2 三家独立收敛到同一条：retrieval over pre-training

三个互不相关的厂商用几乎相同的措辞写了同一条规则：

- `cuda-kernels:16`：那些在通用 PyTorch 教程里标准的写法在 ABI3 下是硬失败，"即使 PyTorch 文档或你的训练数据这么建议，也不要用"
- `cloudflare/wrangler:8`："Your knowledge of Wrangler CLI flags, config fields, and subcommands **may be outdated**. Prefer retrieval over pre-training for any Wrangler task."
- `convex:31`："**Generic model knowledge of Convex is often stale**; `convex-expert` has the current object-form function syntax..."

总结共同强调的核心要点：

1. 软件、框架、工具会持续迭代，**大模型预训练固化的静态知识存在时效性缺陷，会滞后于现实版本变更**；
2. 领域内的文档、旧教程、网上流传的标准写法，不等于当前环境可用，不能直接拿来就用；
3. 处理这类工具 / 框架实操任务，**优先外部检索、调用最新的专用专家源，拒绝直接使用模型内置的通用记忆知识**，避免输出在真实环境直接报错的结果。

**对 amct-coder 的直接可借鉴项**：AMCT 的对应检索源是明确的、而且是本地可 grep 的——solver 参数看 `ptq-solver-params.md` 背后的源码定义、CLI 参数看 argparse 定义、dtype 看枚举定义。写明冲突时**以源码为准**。

### 2.3 Convex：业内领先的机制

**(1) skill 是构建产物，不是手写文件。** 每个文件顶部都有生成标记：

```
<!-- GENERATED from convex-agents content/capabilities/convex-verify.json — do not edit by hand. -->
```

配 `scripts/skillsManifest.ts`（199 行）+ `skillsManifest.test.ts`（127 行），用 `node:test` 断言目录哈希稳定（"stable across file creation order"）。**业内已经把 skill 当代码来测了**，不只是当文档。

**(2) 时效性外置到线上 catalog。** `convex-add` 不内置能力清单，而是运行时拉 `https://basic-anteater-667.convex.site/capabilities.json`，规则写明：

> "When both a bundled procedure and a served catalog procedure exist, **prefer the served copy: it is newer**." + "If a served URL is unreachable, fall back to the bundled skill's own procedure; **never hard-fail on a catalog miss**."

新能力上线无需用户重装 skill。这解决的是 §2.2 那个问题的下一层——**skill 文件本身也会过时**。AMCT 是内网离线场景，这条不能直接照搬，但"版本对应关系外置成一份可单独更新的文件、且缺失时降级不硬失败"的形态可以。

**(3) findings 结构化总线，skill 之间靠证据组合。** `convex-verify` 结尾要求把发现按 schema 发到总线上：

> "Emit findings on the bus (`specs/finding.schema.json`, class authz/correctness, evidence kind probe-result with the exact failing call) so a composite pass or self-heal can pick it up."

`convex-launch-readiness` 就是消费端——把 authz / reviewer / advisor / insights 四个审计聚成一份去重打分报告（自称 "Lighthouse for your backend"）。这比 amct-coder 当前"人读多个 skill 的输出再自己汇总"要强。

### 2.4 `convex-verify`：证据门禁的更强版本

这个 skill 只有 34 行，它的门禁不是"跑了测试"，而是**必须包含否证**，必须可溯源：

> "A green typecheck proves the code parses; it does not prove a non-owner is actually denied." —— 要求 seed → drive → assert 三步，且 "every verification includes at least one NEGATIVE assertion... **the happy path alone is not proof**"。

另两条同样值得抄：

- **禁止改测试凑绿**："Never weaken an assertion to make it pass: a failing negative test is a real defect → hand the fix to convex-authz, don't edit the test until it's green."
- **优先级有实证来源**：它说 authz 缺陷是 "the #1 real bug"，依据是 "the 30-app corpus"——**规则的优先级来自真实语料统计，不是作者直觉**。

---

## 3. 硬件/构建体系参考：`huggingface/kernels@cuda-kernels`

### 3.1 它的架构

```
cuda-kernels/
├── SKILL.md            # 664 行
├── scripts/            # 5 个可执行脚本（benchmark、injection example）
└── references/
    ├── h100-optimization-guide.md   # 按硬件切
    ├── a100-optimization-guide.md
    ├── t4-optimization-guide.md
    ├── diffusers-integration.md     # 按上游框架切
    ├── transformers-integration.md
    ├── huggingface-kernels-integration.md
    ├── kernel-templates.md
    └── troubleshooting.md
```

### 3.2 最值得借鉴的一点：禁止表写成"错什么 / 为什么错 / 用什么代替"

`SKILL.md:18-30` 是一张 10 行的 Disallowed patterns 表，三列结构：

| ❌ 绝不使用 | 为什么失败 | ✅ 改用 |
| --- | --- | --- |
| 任何形式的 pybind11（`#include <torch/extension.h>`、`PYBIND11_MODULE`、任何 `py::` 符号） | pybind11 与 limited API(ABI3) 不兼容，编译直接失败 | `torch-ext/torch_binding.cpp` 里用 `TORCH_LIBRARY_EXPAND`。注意 `torch/extension.h` 会间接引入 pybind11——改 include `torch/torch.h` + `torch/library.h` |
| 手写 `setup.py` / `pyproject.toml` 调 `torch.utils.cpp_extension` | setuptools extension 不是 ABI3，绕过 `build.toml` | `build.toml` + `nix run .#build-and-copy -L` |
| `TORCH_LIBRARY(my_kernel, m)` 硬编码命名空间 | kernel-builder 会给 op namespace 加 per-build hash（如 `_my_kernel_a1b2c3d`），硬编码永远解析不到 | `TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops)` |
| Python 里硬编码 `torch.ops.my_kernel.fn(...)` | 同样的 namespace mangling，名字只在 build 时才知道 | `from ._ops import ops` 然后 `ops.fn(...)` |
| `torch-ext/` 内绝对导入自己的包 | 从 Hub 加载时包目录会被改名 | 只用相对导入 |

关键的一句在表格前面（`SKILL.md:16`）：**"Several patterns that are standard in generic PyTorch-extension tutorials are therefore hard build failures here. Do not use them, even if PyTorch documentation or your training data suggests them."**

这直接对抗模型的训练数据先验——明确告诉 agent"你记得的那套通用做法在这里是错的"。这正是仓库专属型 skill 存在的根本理由：**通用知识在特定构建体系下是负资产。**

### 3.3 第二点：pre-flight checklist 用可执行的 grep

`SKILL.md:69-77`，7 条交付前检查，前三条是能直接跑的命令：

```bash
grep -rn "pybind11\|PYBIND11\|torch/extension.h\|py::" torch-ext/    # 必须无输出
grep -rn "TORCH_LIBRARY(\|TORCH_LIBRARY_FRAGMENT\|PyInit" torch-ext/  # 必须无输出
# 3. 除 kernel-builder 生成的以外，不存在 setup.py
# 4. kernel-builder check-config 通过
# 5. kernel 目录是 git 仓库且文件已提交（Nix 拒绝非 git 构建）
# 6. nix run .#build-and-copy -L 成功
# 7. kernel-builder check-abi 通过
```

这恰好实践了 superpowers `writing-skills:59` 那条判据——**能机械强制的就别写成文档规则**。禁止 pybind11 是可 grep 的，所以给 grep 命令；而"kernel 该拆多细"是判断题，才留给文档。


### 3.4 可以补进 amct-coder / amct-ops-dev 的三条

1. **在 BLOCKER 里补"为什么错 + 用什么代替"两列。** 当前 9 条 BLOCKER 是单列禁止（"不在 CLI/workflow/solver 中加入模型专属分支"）。三列结构让 agent 不只知道禁止，还知道正确落点，减少禁止后原地卡住。
2. **把可 grep 的 BLOCKER 写成命令。** 比如"不混用 LLM new-path `ALGO_REGISTRY` 与 classic `AlgorithmRegistry`"就是可检查的，给出 grep 命令比留给判断更可靠。

---

## 4

**Q：写这个 skill 时参考了业内哪些实践？**

分两类看。**仓库专属型**——也就是 amct-coder 的真实分类——参考 `cloudflare/skills`（13 skill / 3010 行，厚入口 + 决策树）、`get-convex/agent-skills`（33 skill / 1320 行，薄切分 + 生成式）、`huggingface/kernels@cuda-kernels`（硬件/构建体系约束）。**通用流程型**参考 `obra/superpowers`、Anthropic 官方 `skill-creator`（渐进披露的三层定义）、`mattpocock/skills`（受控术语表）。具体借鉴：Iron Law 式硬禁止、证据门禁、description 只写触发条件、按场景切 reference、检索源优先于训练数据。

**Q：仓库专属型的 skill，业内成熟的做法是什么？（本类主问题）**

我读了源码的两个商业样板取舍正好相反，说明这件事没有唯一解，关键是切分依据要和领域结构对齐。Cloudflare 按**产品线**切 13 个，单个 wrangler 就 922 行，入口 248 行全是决策树；Convex 按**单个能力动作**切 33 个，30 个都在 40 行以内，入口只有 69 行纯路由。我的选择更靠 Cloudflare（1 个 162 行入口 + 17 个 reference 切片），因为 AMCT 的层次结构本身稳定、按层切比按动作切更贴合；但 quant-tools 下面 7 个子 skill 是 Convex 式薄切分，因为那些是独立的操作动作。

最强的一条交叉验证是三家独立收敛到同一句：知识可能过时，**检索优先于训练数据**。Cloudflare 写 "Prefer retrieval over pre-training"，Convex 写 "Generic model knowledge of Convex is often stale"，cuda-kernels 写"即使 PyTorch 文档这么建议也不要用"。三个互不相关的厂商措辞几乎一样，说明这一类 skill 的本质价值不是补充知识，而是**否决默认知识**。

**Q：特定场景的 skill 和通用编码 skill 有什么本质区别？**

通用 skill 约束流程，特定场景 skill 要对抗模型的训练数据先验。`cuda-kernels` 那句话最直白：那些在通用 PyTorch extension 教程里标准的写法，在 ABI3 构建体系下是硬编译失败，"即使 PyTorch 文档或你的训练数据这么建议，也不要用"。这和我遇到的问题一样——agent 会反复套用 classic `BaseQuantizeModule` 路径、发明不存在的 CLI 参数，因为那些在别的量化框架里是对的。

**Q：这些样板里有哪些是你还没做到的？**

三个，都出自 Convex，都是把 skill 当工程产物而不是文档。一是 **skill 是构建产物**：每个文件带 `<!-- GENERATED from ... do not edit by hand -->`，配 199 行 manifest 脚本和 127 行 `node:test`，断言目录哈希跨文件创建顺序稳定。二是**时效性外置**：`convex-add` 运行时拉线上 catalog，规则写明"线上版更新则优先线上，但 catalog 拉不到时降级到内置、绝不硬失败"——AMCT 是内网离线场景不能照搬，但"版本对应关系单独成文件 + 缺失时降级"的形态可以。三是 **findings 结构化总线**：`convex-verify` 按 schema 发现问题，`convex-launch-readiness` 消费四个审计的输出聚成一份去重打分报告，skill 之间靠证据组合而不是靠人读。

同时我会说明我不是照抄：Convex 那个 `specs/finding.schema.json` 被 4 个 skill 引用，仓库里根本搜不到该文件——这种量级的厂商 skill 也有悬空引用，所以 reference 的可达性要自己核。

**Q：Cloudflare 和 Convex 有没有哪条比你现在的做法更好，你准备改？**

两条。Cloudflare 把检索源写成**可机器核对的表格**（config schema 在 `node_modules/wrangler/config-schema.json`、类型查 `npm pack`），并给了 tie-break 规则"reference 和文档冲突时信文档"，点名适用于数值上限、定价、类型签名、配置项。我的 BLOCKER 现在只说"不要这么写"，没说"去哪查权威答案"——AMCT 的对应源是本地可 grep 的（solver 参数看源码定义、CLI 看 argparse、dtype 看枚举），应该列成表；但 tie-break 方向要反过来写成"以源码为准"，因为离线内网场景下文档比源码更容易旧。

Convex 的 `convex-verify` 则给了比我更狠的证据门禁：**必须包含否证**。"typecheck 绿了只证明代码能解析，不证明非 owner 真的被拒绝"，且 "the happy path alone is not proof"。它还禁止改测试凑绿——"failing negative test 是真缺陷，交给 convex-authz 修，不要改断言"。我的四级证据分的是验证**强度**，它分的是验证**方向**，两者正交可叠加。顺带一提它的规则优先级有实证来源（authz 是 "#1 real bug"，依据是 "the 30-app corpus"），不是作者直觉——这比我目前六题人工对照更硬。


---

## 6. 后续amct-coder的改进事项

| 优先级 | 事项 | 依据 |
| --- | --- | --- |
| 高 | BLOCKER 补"为什么错 / 用什么代替"两列 | `cuda-kernels:18-30`、`workers-best-practices:86-100`（两列即可） |
| 高 | 可 grep 的 BLOCKER 改写成可执行检查命令 | `cuda-kernels:69-77` |
| 高 | 加一句显式对抗训练数据先验的声明 | `cuda-kernels:16`、`wrangler:8`、`convex:31` |
| 高 | 加"检索源表"+ tie-break（solver 参数 / CLI argparse / dtype 枚举各自的权威位置，冲突时以源码为准） | `cloudflare/wrangler:12-18`、`cloudflare:20-30` |
| 高 | 证据门禁增加"否证方向"：验证必须含至少一条反例断言，且禁止改测试凑绿 | `convex-verify` 全文 |
| 中 | 评估把 A/B 语料统计化，让 BLOCKER 优先级有实证来源 | `convex-verify` 的 "30-app corpus" |
| 中 | 评估 skill 自动化校验（reference 可达性、frontmatter 合规、目录哈希） | `convex/scripts/skillsManifest.test.ts` |
| 低 | 版本对应关系外置成单独文件，缺失时降级不硬失败 | `convex:36`（catalog 降级规则） |
| 低 | 考虑受控术语表，标注该避免的近义词 | `mattpocock/skills` 的 `codebase-design` |
| 低 | 评估 `allowed-tools` / `argument-hint` / `references` 等 frontmatter 字段 | `cuda-kernels` frontmatter、`cloudflare` 的 `references:` |
---