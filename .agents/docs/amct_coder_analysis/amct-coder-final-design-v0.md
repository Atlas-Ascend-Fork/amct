# amct-coder 最终版设计文档

本文档整合 `SKILL.md`、`amct-coder-module-analysis.md`、`data_flow_design_v2.md` 和各子能力 reference，作为 `amct-coder` 后续演进的设计基准。

如果本文与旧分析文档冲突，以本文为准；如果本文与 `SKILL.md` 的实际执行规则冲突，先按本文修订 `SKILL.md`，再同步 reference。本文不替代具体 playbook，具体编码规则仍在 `.agents/skills/amct-coder/references/` 中维护。

## 设计目标

`amct-coder` 是 AMCT 仓的编码入口，不是大而全知识库。它要解决的是：

1. 先判断用户需求是否越界。
2. 再判断需求属于哪条代码体系。
3. 对 AMCT LLM CLI 需求，按命令行任务阶段识别 eval / extract / ptq / deploy。
4. 根据阶段数据流判断命中的子能力和读取顺序。
5. 编码后按命中子能力做最小自验。

核心原则：**边界优先，任务流次之，子能力最后落点。**
执行入口还要遵守 `skill-creator` 的渐进披露原则：先做轻量判断，只有命中对应体系后才加载 repo-map、casebook 和子能力 reference，避免把无关上下文提前读进来。

## 文档分工

| 文档 | 定位 | 是否执行时读取 |
| --- | --- | --- |
| `.agents/skills/amct-coder/SKILL.md` | agent 执行入口：边界、路由、读取预算、全局 BLOCKER | 是 |
| `references/*.md` | 子能力 playbook：执行步骤、BLOCKER、自验 | 按需 |
| `amct-coder-final-design.md` | 维护者设计基准：流程、分工、冲突裁决 | 否，改 skill 结构时读 |
| `amct-coder-module-analysis.md` | 架构分层分析：目录、OWN/NEED/PRODUCE/RIPPLE | 否，分析材料 |
| `data_flow_design_v2.md` | 阶段任务流分析：eval/extract/ptq/deploy 数据流 | 否，分析材料 |
| `.agents/docs/repo-map.md` | 仓库结构原则、分层职责、模型适配原则 | 条件读 |
| `.agents/docs/casebook/` | 已踩坑和模型个案 | 条件读 |

维护约束：同一条执行规则只保留一个正文位置。`SKILL.md` 负责路由和全局规则，reference 负责动作细节，analysis 文档只解释设计。

## 用户请求处理主流程

用户请求进入 `amct-coder` 后，按下面顺序处理。

```text
用户请求
  -> 1. 边界与委派判断
  -> 2. 轻量体系判定
  -> 3. 编码前必读（按体系裁剪）
  -> 4. AMCT LLM CLI 阶段任务流判定
  -> 5. 子能力路由与读取
  -> 6. 编码或 review
  -> 7. 最小自验与最终说明
```

### 1. 边界与委派判断

先看需求是否触发全局 BLOCKER 或委派条件。能直接判断越界或命中独立 skill 时，不进入 amct-coder 子能力。

必须优先处理的边界：

- 新增模型 adapter、wrapper、attention/MLP/MoE 结构、模型注册：转 `model-adapter`。
- NPU kernel、Ascend C、PyTorch C++ extension、Meta backend：转 `amct-ops-dev`。
- 跑 eval / extract / PTQ / deploy 实验命令：转 `quant-run`。
- 判读 ppl / delta / 算法收益：转 `direct-quant-eval` 或 `algorithm-validation`。
- 模型专属逻辑写进 CLI、workflow、solver、通用 quant module：报越层 BLOCKER。
- 混用 LLM new-path `ALGO_REGISTRY` 与 classic `AlgorithmRegistry`：报 BLOCKER。
- 新 dtype / 新算法 / 新 public API 只改局部不闭环：按对应全局 BLOCKER 处理。

边界判断使用 `SKILL.md` 全局 BLOCKER、`references/delegation.md` 和必要时的 `references/layer-rules.md`。如果需要回到 repo-map 校验，读取 `.agents/docs/repo-map.md` 的「分层边界」表、表内「默认规则」列，以及「更新规则」节中“哪些情况不要重写 repo-map”的清单；不要把 repo-map 中不存在的「改 vs 不改规则」当成独立章节名。

委派发生在编码前必读之前。当前需求通过边界判断后，如果命中已有独立 skill 的职责，直接委派，不继续进入 amct-coder 子能力 playbook。

| 用户需求 | 委派目标 | 依据 |
| --- | --- | --- |
| 端到端大模型量化，或单段生命周期任务需要自动分流：适配 / 方案 / 评测 / 导出 / 归档 | `quant-workflow` | 编排入口，只做阶段判断、复用、串联和确认 |
| 新增模型族、注册模型、定义 `PtqUnit`、接 quant block、补 attention / MLP / MoE wrapper、模型 adapter 结构实现 | `model-adapter` | 模型层结构实现统一由该 skill 承接 |
| NPU 自定义算子、Ascend C kernel、PyTorch C++ extension、`torch.ops.amct.*` schema、Meta backend、`amct_ops` 构建调试 | `amct-ops-dev` | 硬件算子层独立于 `amct_pytorch` 算法/流程 |
| 按已确认方案实际跑 eval / extract_ptq_data / ptq / PTQ 结果评测，产出 PPL 或参数文件 | `quant-run` | implementer，只执行和记录，不判达标 |
| 已适配模型，想先拿第一轮量化方案推荐卡 | `scheme-recommendation` | 只推荐起步方案，不跑实验、不下最终最优结论 |
| 已知直转掉点，或无直转结果但想按结构和 casebook 选下一步算法 | `algorithm-recommendation` | 只推荐 `lwc / lac / omniquant / autoround` 候选 |
| 直转 PPL 已出，需要判 `ppl_bf16 / ppl_quant / delta` 是否达标 | `direct-quant-eval` | reviewer，只读直转结果，不跑评测 |
| PTQ 算法结果已出，需要判相对直转是否改善 | `algorithm-validation` | reviewer，只读算法结果，不跑 PTQ |
| deploy-ready 量化模型目录导出、补齐 `deploy_quantization.md` 交付文档 | `deploy-export` | deploy 交付独立闭环 |
| 创建 PR、推送代码、查看 PR 改动 / 评论 / discussion、删除 PR 评论 | `gitcode-pr` | GitCode PR 操作必须走 API skill |
| 查看 GitCode issue 详情、issue URL / 编号、issue 评论 | `gitcode-issue` | GitCode issue 内容通过 API 获取 |
| `gitcode-pr` / `gitcode-issue` 缺失，或用户明确要求同步上游 skill | `default-skills` | 只补装缺失或按用户要求强制同步 |

委派规则：

1. 用户明确点名某个现有 skill，且不违反边界时，尊重用户点名。
2. 用户未点名，但需求命中上表任一独立 skill 时，直接委派。
3. 混合需求先拆边界：amct-coder 只处理边界检查、review、非委派代码部分；执行、判读、模型结构实现等交对应 skill。
4. 新模型 adapter / wrapper 结构实现统一交 `model-adapter`；amct-coder 只保留委派前落点判断、模型层契约 review、局部非结构 bug 判断。
5. 量化命令执行和结果判读严格分离：跑数交 `quant-run`，直转判读交 `direct-quant-eval`，PTQ 收益判读交 `algorithm-validation`。

### 2. 轻量体系判定

边界合法且未委派后，先做轻量体系判定。轻量判定只依赖用户请求、路径、命令名和少量入口信号，不在这一步预读全部 repo-map、casebook 或子能力 reference。

这里的 `LLM new-path` 不是泛指“大模型相关代码”，而是指 AMCT LLM CLI 命令行工具链：用户通过 `python -m amct_pytorch.eval`、`extract_ptq_data`、`ptq`、`deploy` 等 CLI 入口触发 eval / extract / ptq / deploy 阶段，后端再进入 workflow、adapter、quant module、algorithm、solver、deploy 等分层。因此更准确的表述是 **AMCT LLM CLI / LLM new-path**。

| 体系 | 判断信号 | 主落点 |
| --- | --- | --- |
| AMCT LLM CLI / LLM new-path | `python -m amct_pytorch.eval/extract_ptq_data/ptq/deploy`，`workflows/llm_*`，`common/models/llm`，`quantization/dtypes`，`algorithms/quant` | 继续做阶段任务流判定；模型结构实现先委派 `model-adapter` |
| classic graph | `quantize()`、`convert()`、`algorithm_register()`、`classic/`、`graph_based/`、QAT、剪枝、蒸馏、auto calibration | `classic-graph` |
| ops boundary | `amct_ops`、`torch.ops.amct.*`、NPU kernel、Meta backend、wheel/schema/import 边界 | 上层调用用 `ops-boundary`，kernel 实现转 `amct-ops-dev` |
| public API / packaging | `__all__`、CLI choices、setup/build、docs/examples、tests 入口 | `public-api-tests`，必要时补主语义子能力 |

独立体系直接落子能力，不强行套 eval/extract/ptq/deploy 阶段。

### 3. 编码前必读（按体系裁剪）

只有未命中外部独立 skill、确定由 amct-coder 承接时，才进入编码前必读。必读内容按体系裁剪，核心目标是满足边界判断和最小自验，不把无关文档提前加载。

裁剪规则：

- AMCT LLM CLI / LLM new-path 或跨层编码任务：执行下面的完整四步读取。
- classic graph：读 repo-map 中 classic / public API 相关边界和 `AGENTS.md`；只有涉及 LLM 量化经验复用时才读 casebook。
- public API / packaging / docs / examples：读 repo-map 中 public API / packaging 边界和 `AGENTS.md`；通常不读 casebook。
- ops boundary：只处理 `amct_pytorch` 调用 `amct_ops` 的上层边界；真正 kernel / C++ / Meta 实现已经在第 1 步委派给 `amct-ops-dev`。
- GitCode Issue / PR 协作任务：已在第 1 步委派给 `gitcode-issue` / `gitcode-pr`，不进入本节。

AMCT LLM CLI / 跨层编码任务的完整读取顺序：

1. 读 `.agents/docs/repo-map.md` 的「分层边界」表。
   - 目标：判断本次改动落在哪一层：CLI / Workflow / Model adapter / Quant modules / Algorithms / Optimization / Data。
   - 输出：一句“本次主要改动层 = X，允许改动范围 = Y”。
2. 读 `.agents/docs/repo-map.md` 的「分层边界」表内「默认规则」列，以及「更新规则」节中“哪些情况不要重写 repo-map”的清单。
   - 目标：确认当前层允许改什么、不允许改什么，以及本次是否需要更新 repo-map。
   - 注意：不要写成读取 repo-map 的「改 vs 不改规则」章节；当前 repo-map 没有这个独立标题。
3. 读 casebook，顺序为：
   - L1：`.agents/docs/casebook/cross-model-pitfalls.md`
   - L2：按 config / checkpoint / 结构信号命中 `.agents/docs/casebook/structure-family-pitfalls.md`
   - L3：命中模型族个案；没有个案则说明未命中。
   - 目标：先吸收跨网络坑，再叠加结构家族坑，最后看模型个案。
4. 读 `AGENTS.md` 的「开发规范」段。
   - 目标：确认代码风格、构建命令、pre-commit、PR/Issue 规范。

AMCT 项目规范必须纳入执行检查：

- Python 遵循 PEP8，使用 `ruff` 检查和格式化。
- C/C++ 遵循 Google 代码规范，使用 `.clang-format`：4 空格缩进、120 字符行宽。
- 提交前安装并运行 `pre-commit` 检查。
- 新增代码须保留 Apache 2.0 版权声明。
- 简单 bug 修复可直接提 PR；新功能或接口变更需先通过 Issue 讨论。
- PR 应关联对应 Issue，并包含代码、优化文档和 README 更新。
- 创建 Issue / PR 必须遵循仓库模板：
  - PR：`.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md`
  - Issue：`.gitcode/ISSUE_TEMPLATE/` 下按类型选择模板。
- 涉及 GitCode Issue 或 PR 时，使用仓内 `gitcode-issue`、`gitcode-pr` skill；缺失时按 `default-skills` 安装。

完成以上四步后，再按任务范围追加读取：

- 涉及 attention / wrapper review 或局部非结构 bug 时，读 `.agents/docs/repo-map.md` 的「Attention 适配原则」和「Wrapper 合并原则」。
- 涉及 C/C++ 时，读 `.clang-format`，确认 Google style、行宽 120、4 空格缩进。
- 涉及模型 adapter 结构实现时，不继续读 amct-coder 子能力，直接委派 `model-adapter`。
- 涉及具体子能力实现时，读 `.agents/skills/amct-coder/references/<sub-capability>.md`。

### 4. AMCT LLM CLI 阶段任务流判定

AMCT LLM CLI / LLM new-path 需求再按阶段判定。阶段用于回答“这次命令行任务在数据流里处于哪一步”，不是替代子能力边界。

| 阶段 | 主责子能力 | 关键数据 | 常见需求 |
| --- | --- | --- | --- |
| eval | `llm-data-workflow` | Wikitext 输入、PPL、bit policy、adapter forward | PPL、bf16/quant eval、seq_len、评测数据、eval_mode |
| extract_ptq_data | `llm-data-workflow` | Pileval 输入、`hook_name`、`block_<layer>_<target>_in.pkl`、kwargs | 校准数据、离线输入、hook 捕获、nsamples |
| ptq | `ptq-solver-params` + `llm-algorithm` + `llm-data-workflow` | `PtqUnit`、离线输入、GT、solver、`layer_*.pt` | solver、算法训练、参数保存/加载、GT |
| deploy | `deploy-export-code` | `layer_*.pt`、deploy binding、dtype payload、safetensors、config | safetensors、index、quantization_config、导出字段 |

跨阶段需求按数据流顺序拆解，不要一开始全量读取所有 reference。

### 5. 子能力路由与读取

子能力按正交契约划分，不按目录机械划分。子能力只在未命中外部独立 skill 时使用。

| 子能力 | 核心职责 | 不负责 |
| --- | --- | --- |
| `dtype-quant-module` | dtype、bit policy、quantizer、通用 quant module、deploy payload | 模型结构、算法训练策略 |
| `llm-algorithm` | LWC/LAC/OmniQuant/AutoRound/FlatQuant 类算法、可训练参数、统计导出 | solver 调度、模型结构 |
| `llm-model-unit` | 模型层契约检查、review、局部非结构 adapter bug；确认 `PtqUnit.kind`、adapter forward、extract 命名、deploy binding | 新增模型层结构实现，统一交 `model-adapter` |
| `llm-data-workflow` | eval/extract/PTQ workflow、dataloader、离线输入、GT、PPL | 模型结构、dtype 语义、算法参数优化 |
| `ptq-solver-params` | solver、optimizer、scheduler、unit 调度、PTQ 参数保存/加载/回载 | 算法定义、数据提取 |
| `deploy-export-code` | deploy workflow、safetensors、weight index、quantization_config | 重新训练、评测、定义 dtype |
| `classic-graph` | classic eager/module 替换、graph_based、QAT、校准、剪枝、蒸馏 | AMCT LLM CLI / LLM new-path 算法注册 |
| `ops-boundary` | `amct_pytorch` 调用 `amct_ops` 的 Python 侧边界 | NPU kernel/C++ 实现 |
| `public-api-tests` | 公开 API、CLI choices、packaging、tests、docs/examples 同步 | 业务语义本身 |

默认读取顺序：

1. 主命中的子能力 playbook。
2. 只有涉及跨层或越层风险时，读 `layer-rules.md` 或 `delegation.md`。
3. 只有触到 wrapper/attention 细节时，读 `wrapper-rules.md` 或 `attention-rules.md`。
4. 只有上游契约或下游产物变化时，补读相关 playbook 的「上下游依赖」和「BLOCKER」。
5. 编码结束前读 `validation.md`。

默认软预算仍是 3 个 reference：主 playbook + 最多 1 个辅助规则 + validation。跨 3 个以上子能力时按阶段分批读取。


## 四个阶段的数据流

### eval

```text
CLI args
  -> LlmEvalWorkflow
  -> get_wiki_inputs()
  -> adapter do_embedding_forward()
  -> adapter do_block_forward(use_quant_block, enable_quant)
  -> adapter do_head_forward()
  -> wikitext2_ppl()
  -> PPL
```

主责：`llm-data-workflow`。

`llm-model-unit` 只保证 adapter 分层 forward 等价、mask/position/input_ids 等 block kwargs 传递正确。

### extract_ptq_data

```text
CLI args
  -> LlmExtractPtqDataWorkflow
  -> get_pileval()
  -> 选择 hook_name
  -> adapter do_embedding_forward(..., hook_name=...)
  -> adapter do_block_forward(..., hook_name=...)
  -> save_ptq_inps()
  -> save_ptq_kwargs()
  -> block_<layer>_<target>_in.pkl + kwargs pkl
```

主责：`llm-data-workflow`。

`llm-model-unit` 保证 hook 名、adapter 捕获位置、`PtqUnit.kind` 与文件名一致。

### ptq

```text
CLI args
  -> LlmPtqWorkflow
  -> adapter build_quant_block()
  -> adapter iter_ptq_units()
  -> data_provider load_unit_inputs()
  -> materialize_gt()
  -> algorithm trainable_params()
  -> solver solve/finalize
  -> layer_<idx>_<unit.save_name>.pt
```

主责由多个子能力分摊：

- `llm-model-unit`：quant block 与 PtqUnit 契约。
- `llm-data-workflow`：离线输入和 GT。
- `llm-algorithm`：可训练参数与统计导出。
- `ptq-solver-params`：训练调度、保存、回载。

### deploy

```text
CLI args
  -> LlmDeployWorkflow
  -> PtqParamStore load_layer()
  -> adapter iter_deploy_bindings()
  -> dtype export_deploy()
  -> algorithm export_deploy()
  -> safetensors + index + config
```

主责：`deploy-export-code`。

`llm-model-unit` 保证 deploy binding 能把模型层权重与量化参数对齐。

## 冲突裁决规则

后续维护中遇到冲突，按下面优先级处理：

1. 用户本次明确要求优先。
2. 现有独立 skill 的职责优先于 amct-coder 子能力；命中即委派。
3. 全局边界和 BLOCKER 优先于任务阶段流转。
4. 轻量体系判定优先于 AMCT LLM CLI 阶段判定；classic / ops / public API 不强行套 LLM 四阶段。
5. 阶段任务流只用于判断协作链，不改变子能力 OWN 边界。
6. 子能力职责以正交契约为准，不以目录相邻为准。
7. `llm-model-unit` 不承接模型层结构实现；结构实现转 `model-adapter`。
8. eval 中 `llm-model-unit` 参与角色叫 adapter forward，不叫 hook；extract 中才使用 hook_name 捕获契约。
9. 修改公开参数、用户产物或文件格式时，必须同步 `public-api-tests` 或对应下游文档。
10. 无法判断是否越界时，先读取 `delegation.md` / `layer-rules.md`，仍不清楚再询问用户。

## 外部 Skill 委派表

`amct-coder` 的委派表必须从现有 skill 的 `description`、`不适用于`、`边界规则` 抽取，不按经验脑补。当前需要覆盖：

| Skill | amct-coder 中的委派口径 |
| --- | --- |
| `quant-workflow` | 端到端量化生命周期和需要自动分流的单段量化任务 |
| `model-adapter` | 新模型族适配、模型注册、`PtqUnit`、quant block、attention / MLP / MoE wrapper 结构实现 |
| `amct-ops-dev` | `amct_ops` 下 NPU kernel、C++ extension、schema、Meta backend、构建和调试 |
| `quant-run` | 已确认方案后的实际命令执行：bf16 / quant eval、extract、ptq、PTQ eval |
| `scheme-recommendation` | 第一轮量化方案推荐卡 |
| `algorithm-recommendation` | 下一步 PTQ 算法候选推荐，仅限已注册 `lwc / lac / omniquant / autoround` |
| `direct-quant-eval` | 直转结果判读，只读 `ppl_bf16 / ppl_quant / delta` |
| `algorithm-validation` | PTQ 算法收益判读，只读 `ppl_algo` 与直转参考 |
| `deploy-export` | deploy-ready 目录导出和 `deploy_quantization.md` 交付文档 |
| `gitcode-pr` | GitCode PR 创建、推送、评论、diff、discussion、删除评论 |
| `gitcode-issue` | GitCode issue 详情、评论、URL、编号读取 |
| `default-skills` | GitCode PR/Issue skill 缺失补装，或用户明确要求同步上游 skill |

## 对 `SKILL.md` 的目标形态

最终 `SKILL.md` 应保持短入口，结构建议如下：

1. 定位：`amct-coder` 是边界判断 + 子能力路由 + 最小自验入口。
2. 执行流程：边界与委派判断 -> 轻量体系判定 -> 编码前按体系裁剪读取 -> AMCT LLM CLI 阶段任务流 -> 子能力读取 -> 编码自验。
3. 委派和全局 BLOCKER：集中在主文件，必要时指向 `delegation.md`。
4. 体系判定表：AMCT LLM CLI / LLM new-path / classic / ops / public API。
5. AMCT LLM CLI 阶段表：eval / extract / ptq / deploy 的主责、数据、相关子能力。
6. 子能力路由表：9 个子能力及 playbook；说明子能力只在未命中外部 skill 时使用。
7. 读取预算：默认 3 个 reference，跨阶段分批读取。
8. 验证要求：编码结束前读 `validation.md`。

避免在 `SKILL.md` 中塞入长篇架构解释；长解释留在本文和分析文档。

## 维护检查清单

改 `amct-coder` 结构时检查：

1. 是否先写清了边界判断，而不是先给子能力分类。
2. 是否把现有独立 skill 的委派规则放在子能力路由之前。
3. 是否先做轻量体系判定，再按体系裁剪 repo-map / casebook / reference 的读取范围。
4. 是否把 independent 体系从 AMCT LLM CLI 四阶段中剥离出来。
5. 是否把 eval / extract 的 `llm-model-unit` 角色区分清楚。
6. 是否保持子能力职责正交，没有按目录重复切分。
7. 是否保留读取预算，避免一开始全量读取 reference。
8. 是否明确哪些是执行规则，哪些只是分析说明。
9. 是否同步更新存在误导的旧分析文字。
10. 是否需要更新 `repo-map.md` 或 casebook；只有边界或核心流程语义变化时才更新。
