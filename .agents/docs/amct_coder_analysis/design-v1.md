# amct-coder 最终版设计文档

本文档整合 `SKILL.md`、`amct-coder-module-analysis.md`、`data_flow_design_v2.md` 和各子能力 reference，作为 `amct-coder` 后续演进的设计基准。

如果本文与旧分析文档冲突，以本文为准；如果本文与 `SKILL.md` 的实际执行规则冲突，先按本文修订 `SKILL.md`，再同步 reference。本文不替代具体 playbook，具体编码规则仍在 `.agents/skills/amct-coder/references/` 中维护。

## 设计目标

`amct-coder` 是 AMCT 仓的编码入口，不是大而全知识库。它要解决的是：

1. 先按仓库边界和独立 skill 职责判断用户提出的动作与落点是否越界。
2. 再判断需求属于哪条代码体系和哪个主子能力。
3. 只有任务明确涉及 LLM CLI、workflow、阶段产物或跨阶段契约时，才检查 eval / extract / ptq / deploy 的阶段影响。
4. 确认由 amct-coder 承接编码/review 后，先读取 repo-map 的边界契约，再根据主子能力和真实风险渐进加载其它资料与源码锚点。
5. 编码后按命中子能力做最小自验。

核心原则：**仓库边界优先，动作归属次之，子能力确定主落点，阶段影响按需补充。**
执行入口还要遵守 `skill-creator` 的渐进披露原则：amct-coder 承接编码/review 后固定读取 repo-map 的最小边界切片；casebook、repo-map 其它章节和辅助 reference 仍按任务风险加载，避免把无关上下文提前读进来。

## 文档分工

| 文档 | 定位 | 是否执行时读取 |
| --- | --- | --- |
| `.agents/skills/amct-coder/SKILL.md` | agent 执行入口：边界、路由、读取预算、全局 BLOCKER | 是 |
| `references/*.md` | 子能力 playbook：执行步骤、BLOCKER、自验 | 按需 |
| `amct-coder-final-design-v0.md` | 维护者设计基准：流程、分工、冲突裁决 | 否，改 skill 结构时读 |
| `amct-coder-module-analysis.md` | 架构分层分析：目录、OWN/NEED/PRODUCE/RIPPLE | 否，分析材料 |
| `data_flow_design_v2.md` | 阶段任务流分析：eval/extract/ptq/deploy 数据流 | 否，分析材料 |
| `.agents/docs/repo-map.md` | 仓库结构原则、分层职责、模型适配原则 | amct-coder 编码/review 必读「分层边界」「更新规则」，其余条件读 |
| `.agents/docs/casebook/` | 已踩坑和模型个案 | 条件读 |

维护约束：同一条执行规则只保留一个正文位置。`SKILL.md` 负责路由和全局规则，reference 负责动作细节，analysis 文档只解释设计。

## 用户请求处理主流程

用户请求进入 `amct-coder` 后，按下面顺序处理。

```text
用户请求
  -> 1. 按动作和职责归属做外部 skill 路由
  -> 2. 确认由 amct-coder 承接后，读取 repo-map 边界并声明落点
  -> 3. 轻量体系判定
  -> 4. 命中主子能力
  -> 5. 必要时检查 LLM CLI 主阶段与受影响阶段
  -> 6. 读取基础资料，并按风险追加资料
  -> 7. 编码或 review
  -> 8. 最小自验与最终说明
```

### 1. 仓库边界与外部 skill 路由

先按用户动作和已加载的独立 skill metadata 判断任务职责归属。跑数、结果判读、deploy 交付、模型结构实现、NPU kernel 和 GitCode 操作等明确命中外部 skill 的任务直接路由，不为这些任务预读 repo-map。只有确认任务实质是由 amct-coder 承接的编码或 review，才进入仓库分层边界判断。

仓库 `AGENTS.md`、repo-map 分层边界和已有公共契约高于用户指定的实现路径；用户目标应尽量完成，但用户提出的错误落点不能覆盖仓库边界。

必须优先处理的边界：

- 新增模型 adapter、wrapper、attention/MLP/MoE 结构、模型注册：转 `model-adapter`。
- NPU kernel、Ascend C、PyTorch C++ extension、Meta backend：转 `amct-ops-dev`。
- 跑 bf16/quant eval、extract、PTQ、PTQ 结果评测命令：转 `quant-run`。
- 实际导出 deploy-ready 模型目录和交付文档：转 `deploy-export`；修改 deploy 实现代码仍由 `amct-coder/deploy-export-code` 承接。
- 判读 ppl / delta / 算法收益：转 `direct-quant-eval` 或 `algorithm-validation`。
- 模型专属逻辑写进 CLI、workflow、solver、通用 quant module：阻止该错误落点，给出正确落点；正确方案明确时继续路由和执行。
- 混用 LLM new-path `ALGO_REGISTRY` 与 classic `AlgorithmRegistry`：阻止混用方案，回到对应体系的注册链继续处理。
- 新 dtype / 新算法 / 新 public API 只改局部不闭环：按对应全局 BLOCKER 处理。

这里的 BLOCKER 阻止的是错误实现方案，不是默认终止用户任务。处理顺序是：指出违反的边界 -> 给出正确落点或替代方案 -> 在无需新增产品决策时切换到正确 skill/子能力继续执行。只有正确方案仍存在行为、兼容性或范围选择，且无法从仓库现状判断时，才停下询问用户。

外部 skill 路由发生在 amct-coder 子能力读取之前。命中独立 skill 后，应切换到该 skill 的流程并继续完成当前请求，不能只回复“应使用某 skill”后结束。运行环境明确支持且允许 subagent 时可以委派给对应 agent；否则由当前 agent 加载并执行对应 skill。混合任务按职责拆分，最后回到主任务做契约对齐和结果汇总。

| 用户需求 | 路由目标 | 依据 |
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

路由与继续执行规则：

1. 用户明确点名某个现有 skill，且不违反仓库边界和该 skill 的“不适用”条件时，尊重用户点名。
2. 用户未点名，但需求命中上表任一独立 skill 时，切换到对应 skill 并继续执行，不把路由说明当作任务终点。
3. 用户指定了错误实现层时，保留其业务目标，拒绝错误落点；正确落点明确则直接按正确方案继续，存在关键决策才询问。
4. 混合需求先按职责拆分：amct-coder 处理边界检查、review 和非外部 skill 代码；执行、判读、模型结构实现等走对应 skill；最后统一做契约对齐和汇总。
5. 新模型 adapter / wrapper 结构实现统一转 `model-adapter`；amct-coder 只保留路由前落点判断、模型层契约 review、局部非结构 bug 判断。
6. 量化命令执行和结果判读严格分离：eval/extract/PTQ 跑数交 `quant-run`，deploy 产物交 `deploy-export`，直转判读交 `direct-quant-eval`，PTQ 收益判读交 `algorithm-validation`。

#### amct-coder 承接后的强制边界读取

确认由 amct-coder 承接编码或 review 后，在轻量体系和主子能力判定之前执行：

1. 读取 `.agents/docs/repo-map.md` 的「分层边界」表，确定 CLI / Workflow / Model adapter / Quant modules / Algorithms / Optimization / Data 等层的职责归属和默认禁止事项。
2. 读取 repo-map 的「更新规则」，提前判断本次改动是否可能需要写回 repo-map。
3. 抽查与候选落点直接相关的源码锚点，确认 repo-map 仍与当前实现一致；repo-map 是稳定架构说明，源码是当前行为事实。
4. 形成边界声明：主落点层、允许改动范围、禁止事项、必要的辅助层、repo-map 是否需要写回。

边界声明用于约束后续体系、子能力、阶段影响和验证选择，不要求每次都把完整声明展示给用户。repo-map 与源码不一致时不能静默忽略：先判断源码是否越层；若源码代表预期的新边界或核心流程，则将 repo-map 修正纳入本次任务。

repo-map 的读取前置不表示每次都写回。只有框架边界、核心流程、实际注册集合或公共契约变化，或者 repo-map 已确认落后于预期源码事实时，才更新受影响 section。普通算法文件、局部 bug、未接主链目录和不改变复用边界的模型专属细节默认不更新。

### 2. 轻量体系判定

完成 repo-map 最小边界读取和边界声明后，再做轻量体系判定。轻量判定只依赖用户请求、路径、命令名、边界声明和少量入口信号，不在这一步预读全部 casebook、repo-map 其它章节或子能力 reference。

这里的 `LLM new-path` 指当前 blockwise LLM PTQ 主路径，覆盖 CLI、workflow、model adapter、quant module、algorithm、solver、data provider 和 deploy。它是一条代码体系，不等同于 eval / extract / ptq / deploy 四个运行阶段；四阶段只是观察这条体系中数据与产物如何流转的辅助视角。

| 体系 | 判断信号 | 主落点 |
| --- | --- | --- |
| LLM blockwise / LLM new-path | `cli/llm`、`workflows/llm_*`、`common/models/llm`、`quantization/dtypes`、`quantization/modules`、`algorithms/quant`、`common/optimization`、`common/datasets` | 先命中主子能力；出现阶段信号时再检查四阶段影响 |
| classic graph | `quantize()`、`convert()`、`algorithm_register()`、`classic/`、`graph_based/`、QAT、剪枝、蒸馏、auto calibration | `classic-graph` |
| ops boundary | `amct_ops`、`torch.ops.amct.*`、NPU kernel、Meta backend、wheel/schema/import 边界 | 上层调用用 `ops-boundary`，kernel 实现转 `amct-ops-dev` |
| public API / packaging | `__all__`、CLI choices、setup/build、docs/examples、tests 入口 | `public-api-tests`，必要时补主语义子能力 |

独立体系直接落子能力，不强行套 eval/extract/ptq/deploy 阶段。

### 3. 主子能力路由

子能力按正交契约划分，不按目录机械划分。先根据用户动作、目标文件和要改变的契约选择一个主子能力；公开面、测试或上下游闭环可以作为辅助子能力。子能力只在未命中外部独立 skill 时使用。

| 子能力 | 核心职责 | 不负责 |
| --- | --- | --- |
| `dtype-quant-module` | dtype、bit policy、quantizer、通用 quant module、deploy payload | 模型结构、算法训练策略 |
| `llm-algorithm` | LWC/LAC/OmniQuant/AutoRound/FlatQuant 类算法、可训练参数、统计导出 | solver 调度、模型结构 |
| `llm-model-unit` | 模型层契约检查、review、局部非结构 adapter bug；确认 `PtqUnit.kind`、adapter forward、extract 命名、deploy binding | 新增模型层结构实现，统一交 `model-adapter` |
| `llm-data-workflow` | eval/extract/PTQ workflow、dataloader、离线输入、GT、PPL | 模型结构、dtype 语义、算法参数优化 |
| `ptq-solver-params` | solver、optimizer、scheduler、unit 调度、PTQ 参数保存/加载/回载 | 算法定义、数据提取 |
| `deploy-export-code` | deploy workflow、safetensors、weight index、quantization_config | 实际执行 deploy 交付、重新训练、评测、定义 dtype |
| `classic-graph` | classic eager/module 替换、graph_based、QAT、校准、剪枝、蒸馏 | LLM new-path 算法注册 |
| `ops-boundary` | `amct_pytorch` 调用 `amct_ops` 的 Python 侧边界 | NPU kernel/C++ 实现 |
| `public-api-tests` | 公开 API、CLI choices、packaging、tests、docs/examples 同步 | 业务语义本身 |

主子能力判断以动作语义优先，不能只看关键词。例如：

- “修改 deploy index 写入逻辑”命中 `deploy-export-code`；“把模型导出成 deploy-ready 目录”路由 `deploy-export`。
- “修 eval workflow 代码”命中 `llm-data-workflow`；“运行量化 eval 拿 PPL”路由 `quant-run`。
- “修已有 wrapper 的 kwargs 漏传”可命中 `llm-model-unit`；“新增或结构性修改 wrapper”路由 `model-adapter`。

### 4. 按需进行阶段影响检查

阶段影响检查只回答“改动处在运行数据流的哪一段、会影响哪些上下游产物”，不回答“代码归谁负责”。代码职责归属已由体系和主子能力确定，因此阶段不是每个 LLM new-path 编码任务都必须产生的单一分类。

不强制所有任务先判阶段，原因如下：

1. dtype、quant module、algorithm 等是横向能力，同一改动可能同时影响 eval、ptq 和 deploy，强行选择单阶段会丢失下游闭环。
2. solver 参数保存、unit 命名、dtype payload 等属于跨阶段契约，正确表达应是“主阶段 + 受影响阶段”，而不是单标签。
3. classic、ops、public API 等独立体系不属于 LLM 四阶段，强套阶段会制造错误路由。
4. 用户已经给出明确文件和代码动作时，主子能力可以直接确定；先判阶段不会增加信息，反而可能诱导加载无关 workflow reference。

按以下条件决定是否执行阶段影响检查：

| 请求信号 | 阶段处理 |
| --- | --- |
| 明确修改某个 LLM CLI、workflow、dataloader 或阶段产物 | 标记一个主阶段，并检查直接上下游 |
| 修改参数保存/加载、unit 输入命名、deploy payload 等跨阶段契约 | 标记主阶段和所有受影响阶段，按数据流顺序处理 |
| 修改通用 dtype、quant module、算法定义，未触及具体 workflow | 不要求先选单阶段；主子能力确定后，只列出需要闭环的受影响阶段 |
| classic、ops、纯 public API/packaging | 阶段记为“不适用” |

阶段影响矩阵如下：

| 阶段 | 主责子能力 | 关键数据 | 常见需求 |
| --- | --- | --- | --- |
| eval | `llm-data-workflow` | Wikitext 输入、PPL、bit policy、adapter forward | PPL、bf16/quant eval、seq_len、评测数据、eval_mode |
| extract_ptq_data | `llm-data-workflow` | Pileval 输入、`hook_name`、`block_<layer>_<target>_in.pkl`、kwargs | 校准数据、离线输入、hook 捕获、nsamples |
| ptq | `ptq-solver-params` + `llm-algorithm` + `llm-data-workflow` | `PtqUnit`、离线输入、GT、solver、`layer_*.pt` | solver、算法训练、参数保存/加载、GT |
| deploy | `deploy-export-code` | `layer_*.pt`、deploy binding、dtype payload、safetensors、config | safetensors、index、quantization_config、导出字段 |

示例：修改 `extract_ptq_data` 的 hook 捕获，以 extract 为主阶段；修改 `PtqParamStore` 的保存格式，以 ptq 为主阶段、eval/deploy 为受影响阶段；新增 dtype 时主分类仍是 `dtype-quant-module`，只检查 eval/ptq/deploy 的接入闭环，不需要先把它归入某一个阶段。

跨阶段需求按数据流顺序拆解，但不要一开始全量读取所有阶段 reference。

### 5. 渐进披露与编码前定位

“编码前必读”不能解释为“每次编码前固定通读全部架构和经验文档”。repo-map 的「分层边界」「更新规则」是 amct-coder 完成核心职责所需的最小稳定契约，确认由本 skill 承接后固定读取；casebook、repo-map 其它章节和辅助 reference 仍按风险加载。这样既避免凭关键词猜落点，也避免模型个案覆盖当前代码证据。

每个由 amct-coder 承接的任务都执行以下基础步骤：

1. 遵守当前作用域已生效的 `AGENTS.md`，不要求为了摘抄规范重复通读。
2. 读取 repo-map 的「分层边界」「更新规则」，抽查候选落点锚点并形成边界声明。
3. 读取主命中子能力 playbook。
4. 用 `rg` 和目标源码/测试锚点核对真实实现；源码是当前行为的事实依据。
5. 修改结束前读取 `validation.md`，选择与改动匹配的最小验证。

其余资料按风险升级加载：

| 条件 | 追加读取 | 不需要读取的典型场景 |
| --- | --- | --- |
| 基础边界声明仍不足以解释主路径、跨层契约或 repo-map 与源码差异 | repo-map 的 PTQ 主路径、当前接通状态、数据与导出链路等相关 section，并扩大源码锚点抽查 | 路径和职责都明确的单层局部修复 |
| 模型结构、adapter 契约、checkpoint/config 信号、模型相关精度异常 | casebook L1；按结构信号读 L2；命中时读 L3 | 通用 dtype、算法注册、packaging、classic 修复 |
| attention/wrapper review 或局部非结构模型 bug | repo-map 对应原则，或 `attention-rules.md` / `wrapper-rules.md` | 不涉及模型 wrapper 的 workflow/solver 修复 |
| 上游契约或下游产物确实变化 | 相关 playbook 的「上下游依赖」和「BLOCKER」小节 | 只改内部实现且契约不变 |
| C/C++ 代码 | `.clang-format` 和对应构建入口 | 纯 Python 任务 |

模型结构实现已经在第 1 步路由给 `model-adapter`，由该 skill 按自己的固定流程读取 repo-map 和 casebook；amct-coder 不应在外部 skill 职责归属明确前替它预读。GitCode、实验执行、结果判读和 deploy 交付同理，进入外部 skill 后遵守目标 skill 的读取规则。

读取预算统计所有设计资料，不只统计 `references/`。默认基础预算是：repo-map 的「分层边界」「更新规则」最小切片 + 主 playbook + `validation.md`；再按风险追加最多 1 份资料或辅助规则。repo-map 其它 section 和 casebook 都占追加预算，并优先只读命中小节。跨多个契约的真实大改动可以分阶段加载；预算用于防止预读，不用于阻止完成任务。

项目规范由适用的 `AGENTS.md` 持续约束。Python 使用仓库既有 ruff/pre-commit 配置，C/C++ 遵循 `.clang-format`；只对本次改动运行有针对性的检查。工具或依赖未安装时报告未执行原因，不在普通编码任务中擅自安装。涉及 PR/Issue 操作时再进入对应 GitCode skill 和模板流程。

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

1. 当前作用域的 `AGENTS.md`、现有独立 skill 职责归属和全局 BLOCKER 先确定任务由谁承接；命中外部 skill 后路由并继续执行。
2. 确认由 amct-coder 承接编码/review 后，repo-map 分层边界和已有公共契约优先，用户指定的实现路径不能覆盖这些约束。
3. 在不违反前两项的前提下，优先满足用户本次的业务目标、行为要求和明确范围；用户点名 skill 时还要满足该 skill 的适用边界。
4. BLOCKER 只阻止错误落点或错误方案；正确替代方案明确时继续处理，只有存在无法从仓库判断的关键决策才询问用户。
5. 轻量体系和主子能力判定优先于阶段影响检查；classic / ops / public API 不强行套 LLM 四阶段。
6. 阶段影响检查只用于识别数据流位置和上下游影响，不改变子能力 OWN 边界。
7. 子能力职责以正交契约为准，不以目录相邻为准。
8. `llm-model-unit` 不承接模型层结构实现；结构实现路由 `model-adapter`，完成后由 amct-coder 做契约 review。
9. eval 中 `llm-model-unit` 参与角色叫 adapter forward，不叫 hook；extract 中才使用 hook_name 捕获契约。
10. 修改公开参数、用户产物或文件格式时，必须同步 `public-api-tests` 或对应下游文档。
11. 无法判断是否越界时，先读取 `delegation.md` / `layer-rules.md`；仍不清楚且影响实现决策时再询问用户。

## 外部 Skill 路由表

`amct-coder` 的路由表必须从现有 skill 的 `description`、`不适用于`、`边界规则` 抽取，不按经验脑补。路由表示职责切换，不表示任务结束；命中后继续执行目标 skill 的流程。当前需要覆盖：

| Skill | amct-coder 中的路由口径 |
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
2. 执行流程：外部 skill 职责归属 -> amct-coder 承接后读取 repo-map 边界并声明落点 -> 轻量体系 -> 主子能力 -> 按需阶段影响检查 -> 渐进加载资料 -> 编码自验。
3. 外部 skill 路由和全局 BLOCKER：集中在主文件，明确 BLOCKER 不默认终止任务，必要时指向 `delegation.md`。
4. 体系判定表：LLM blockwise / LLM new-path / classic / ops / public API。
5. 子能力路由表：9 个子能力及 playbook；说明子能力只在未命中外部 skill 时使用。
6. 阶段影响矩阵：eval / extract / ptq / deploy 的主责、数据和影响范围；只在出现阶段信号时使用。
7. 渐进披露：repo-map 最小边界切片固定读取，其它 section/casebook/辅助规则按条件追加，并纳入总读取预算。
8. 验证要求：编码结束前读 `validation.md`。

避免在 `SKILL.md` 中塞入长篇架构解释；长解释留在本文和分析文档。

## 维护检查清单

改 `amct-coder` 结构时检查：

1. 是否先按动作和独立 skill 职责归属排除不由 amct-coder 承接的任务，并明确路由后继续执行。
2. 确认由 amct-coder 承接编码/review 后，是否强制读取 repo-map 的「分层边界」「更新规则」并形成边界声明。
3. 是否抽查候选落点源码锚点；repo-map 与源码冲突时是否判断源码越层还是 map 过时。
4. 是否先做轻量体系和主子能力判定，再决定是否需要阶段影响检查。
5. 是否只在 CLI/workflow/阶段产物/跨阶段契约场景使用阶段矩阵，并把 classic / ops / public API 剥离出来。
6. 是否把 eval / extract 的 `llm-model-unit` 角色区分清楚。
7. repo-map 最小边界切片之外，是否再按风险加载其它 section、casebook 和辅助规则。
8. 是否保持子能力职责正交，没有按目录重复切分。
9. 读取预算是否覆盖所有设计资料，而不是只统计 `references/`。
10. 是否明确 repo-map 的读取前置不代表默认写回，写回仍受「更新规则」限制。
11. 是否明确哪些是执行规则，哪些只是分析说明。
12. 是否同步更新存在误导的旧分析文字。
