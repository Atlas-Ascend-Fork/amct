# amct-coder 最终版设计文档

本文档整合 `SKILL.md`、`amct-coder-module-analysis.md`、`data_flow_design_v2.md` 和各子能力 reference，记录 `amct-coder` 的 v1 设计基线及后续当前设计。

从“设计目标”到“维护检查清单”保留原 v1 设计；文末“当前设计（v1 后续演进）”记录最新结构。两者冲突时以文末当前设计为准。`SKILL.md` 与 `.agents/skills/amct-coder/references/` 是 Agent 实际执行的运行时事实源；设计记录与运行时规则冲突时，先以运行时文件确认当前行为，再同步本文，不能用旧设计静默回退已经生效的规则。

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

## 当前设计（v1 后续演进）

本节记录经过多轮 forward-test、自审和评审反馈后形成的当前设计。它沿用 v1 的结构，但不复制易过期的外部 Skill 清单、算法名、参数名和源码路径；这些动态事实由当前 Skill metadata、Reference 和源码现场确认。

### 设计目标

`amct-coder` 当前定位为 AMCT 仓源代码编码、实现规划、行为诊断与代码 review 等辅助编码的入口，负责把通用 Agent 能力约束到仓库真实边界和活跃调用链。它要解决的是：

1. 先根据用户动作和当前已发现 Skill 的职责判断任务归属，不与模型适配、NPU 算子、量化执行、结果判读或 deploy 交付等专门流程竞争。
2. 每份 Reference 在主 `SKILL.md` 的 Reference 导航表中就近声明自己的加载时机，不设中间层的任务类型矩阵；`project-rules.md` 与 `validation.md` 在各自文件开头也直接写出加载条件。
3. 除只检查风格、版权、贡献门槛和模板的纯规范合规 review 外，都先读取 repo-map 最小边界切片并抽查源码锚点，区分稳定架构契约与当前行为事实。
4. 从完整 Reference 索引中选择一个粗粒度主场景；多个场景冲突时按动作、入口、被修改契约和活跃调用方裁决。
5. 形成计划前区分确认事实、待确认契约和条件式方案，不把“函数存在”误写成“活跃调用链已接通”。
6. 只有修改 LLM CLI/workflow、阶段产物或跨阶段契约时才检查 eval、extract、ptq、deploy 的阶段影响。
7. 按真实风险渐进加载 Reference、repo-map 其它章节、casebook 和源码测试锚点，达到停止条件后不继续扩张上下文。
8. 实际修改后执行最小验证，其它路径按需核对验证方案或既有证据，并明确区分源码存在、单测覆盖、集成闭环和真实环境验证。

核心原则：**职责归属先行，加载时机就近声明，仓库边界约束行为落点，Reference 负责导航，源码与测试确认事实，阶段只描述影响，验证等级不互相替代。**

### 文档分工

| 文档 | 当前定位 | 是否执行时读取 |
| --- | --- | --- |
| `.agents/skills/amct-coder/SKILL.md` | 运行时入口：职责判断、事实门禁、Reference 导航、读取预算、全局 BLOCKER 和最终说明 | 是 |
| `.agents/skills/amct-coder/references/*.md` | 项目规范、主场景 playbook、辅助边界、验证和贡献操作规则 | 按主 `SKILL.md` Reference 导航表标注的加载时机读取 |
| `design-v1.md` 原 v1 正文 | 初始设计基线，用于理解演进起点 | 否，维护 Skill 结构时参考 |
| `design-v1.md` 文末当前设计 | 当前结构设计快照和维护约束 | 否，维护 Skill 结构时参考 |
| `design-iteration.md` | 每轮问题、裁决、实施范围、测试限制和验证证据 | 否，设计迭代时读取 |
| `.agents/docs/repo-map.md` | AMCT 稳定分层边界和更新规则 | 除纯规范合规 review 外按主 `SKILL.md` §2 读取最小切片 |
| `.agents/docs/casebook/` | 模型结构信号、个案和已踩坑经验 | 模型或精度问题按 L1 -> L2 -> L3 条件读取 |

维护约束：运行时规则只在 `SKILL.md` 或对应 Reference 中保留一份正文。设计文档解释规则为什么存在并记录当前结构，不复制外部 Skill 的完整动态清单，也不替代运行时文件。

### 用户请求处理主流程

```text
用户请求
  -> 1. 按动作和当前已发现 Skill 判断职责归属
  -> 2. 声明仓库边界，按 Reference 导航选择一个主场景手册
  -> 3. 读取直接源码/测试锚点，完成事实与契约消歧
  -> 4. 必要时检查 LLM 阶段影响
  -> 5. 编码、规划、诊断或 review
  -> 6. 执行或核对验证证据并汇总
```

#### 1. 职责归属与外部 Skill 路由

外部职责只根据当前会话已发现 Skill 的 `name + description` 和目标动作判断，`amct-coder` 不维护完整外部路由表。命中专门 Skill 时实际加载并遵循对应流程，不为外部职责预读 amct-coder 的 repo-map 或主场景手册，也不能只口头回复“应使用某 Skill”后结束；路由之后的加载和交付由目标 Skill 决定，`amct-coder` 不再规定。

修改 eval/deploy 等实现代码仍可由 `amct-coder` 承接；运行命令、判读结果或生成交付产物属于对应操作流程。混合任务按职责拆分，完成后统一核对上下游契约并汇总。

外部 Skill 在主入口只保留四处：§定位一句「实际加载而非口头委派」、§1 判归属、§全局 BLOCKER 中 `model-adapter` 与 `amct-ops-dev` 两道自我围栏、`delegation.md` 的入口。围栏约束的是 `amct-coder` 自己不越界，不是管辖已经路由出去的流程。

BLOCKER 只阻止错误落点或错误方案。正确替代路径明确、仍在用户目标和授权范围内时继续执行；只有缺少会改变实现方案的关键契约或授权时才询问用户。该语义只在 `SKILL.md` §全局 BLOCKER 定义一次。

#### 2. 加载时机就近声明

不设中间层的任务类型矩阵。每份 Reference 的加载时机写在主 `SKILL.md` Reference 导航表对应行的「用途与加载时机」列，其中条件较复杂的两份在自己文件开头也自述一遍：

| Reference | 加载时机 |
| --- | --- |
| `project-rules.md` | 修改代码前，或做规范合规 review 时读取；只做规划、诊断或代码行为 review 时不读 |
| `validation.md` | 实际编码在修改后读取并执行匹配的最小验证；只做规划或 review 时按需读取相关部分 |
| repo-map + 主场景手册 | 由 `SKILL.md` §2 规定：除纯规范合规 review 外都要先声明边界 |
| `contribution-flow.md` | 不在主入口导航表中，由 `project-rules.md` 在实际操作 Issue/PR 前引出 |

“纯规范合规 review”只检查代码风格、版权、贡献门槛和模板等规则，定义写在 `SKILL.md` §2 开头。结论一旦涉及实现行为、架构落点、调用链或上下游契约，就不再是纯合规检查，必须回到 §2 补齐资料。

repo-map 是稳定架构契约，源码和测试是当前行为事实。两者冲突时先判断源码是否越层；源码代表预期的新边界或核心流程时，把 repo-map 修正纳入当前任务。读取 repo-map 不代表默认写回，写回仍受其“更新规则”约束。

#### 3. Reference 导航与多场景裁决

主 `SKILL.md` 列出从本入口直接进入的 Reference（当前 17 份），并标注类型、用途和加载时机；`contribution-flow.md` 由 `project-rules.md` 引出，不占导航位。索引只保证可发现性，不表示所有文档是平级子能力，也不表示一次性全部读取。主场景只保留 dtype/quant module、LLM 算法、模型层、数据/workflow、solver/参数、deploy 实现、classic、ops Python 边界和 public API 等粗粒度用途；具体触发、源码锚点、执行步骤和 BLOCKER 留在各自 Reference。

多个主场景同时可能命中时：

1. 根据用户动作、目标入口、被修改契约和活跃调用方确定真正拥有修改语义的主手册。
2. 仍无法唯一确定时读取 `repo-scenarios.md` 定位代码体系，不预读全部候选主手册。
3. 只有源码已经证明同时改变另一契约时，才把第二个主场景作为辅助加载。
4. 选择仍依赖未冻结的关键契约时，进入契约缺失处理，不按关键词强行归类。

完全未命中现有手册时，先查公共入口、注册点、调用方和直接测试；若该场景形成稳定、重复出现的新代码体系，再更新现有手册或新增 Reference。

#### 4. 事实门禁与契约缺失处理

具体计划形成前按固定顺序消歧：先查源码、测试和调用方；查得到就判定或否决错误方案；查不到时先完成不依赖该答案的部分，再按交付物处理。

| 交付物 | 关键契约缺失时的处理 |
| --- | --- |
| 只读规划 | 输出各条件分支的前提、落点差异和验证差异，不把候选写成既定事实 |
| 实际编码 | 不同答案会产生不兼容实现时询问用户，不写投机版本 |
| 局部等价选择 | 变量名、日志措辞等不影响公共契约的选择由 Agent 决定，并在最终说明中点明；默认值可能改变兼容性，不自动归入此类 |

不得自行发明类名、CLI 参数、schema、文件名或 payload；不得在没有核对调用方、注册和编排入口时把未调用 helper 当成当前运行路径。占位对象不得继承上一题名称和设计决策的规则由 `llm-algorithm.md` 承接，主入口不再复制。

需要询问时收敛成封闭选项并说明影响。只有源码惯例、兼容性或实现成本存在可引用证据时才给推荐项；没有依据时明确无默认推荐。拿不到答复时，只读规划保留条件分支，实际编码停在契约确认前，不替用户冻结未知契约。

#### 5. 阶段影响与渐进加载

阶段只描述 LLM 数据流影响，不决定代码职责归属。classic、ops 和纯 public API/packaging 标记为“不适用”；dtype、quant module 和算法等横向能力可以同时影响多个阶段，不强行选择单阶段。

默认上下文预算为：repo-map 最小切片 + 1 个主 playbook + 首轮 2～4 个源码/测试锚点；实际编码再加项目规范。辅助 Reference 默认最多 1 份，项目规范和验证规则不计入主场景资料预算。纯规范合规 review 和已路由到外部专门 Skill 的任务不套用这份行为分析预算。真实跨契约任务按数据流分批加载，每批重新判断是否已经满足停止条件。预算用于防止预读，不用于阻止完成任务；必须超出时说明触发扩张的冲突或缺失证据。

停止条件是：仓库边界已确定；活跃调用链或未接通状态已有证据；确认事实、待确认契约和条件式方案已区分；修改面和最小验证范围已经明确。

#### 6. 按实际执行动作汇总

最终说明只汇总本轮实际执行过的动作，不为未执行的步骤补写结果：

- 改了代码：主场景、改动层与文件、事实和待确认项、复用抽象、已跑与未跑验证，以及 repo-map/casebook/docs/examples 的更新判断。
- 没改代码：分析边界与主场景、事实、发现或方案，以及已核对和缺失的验证证据；不声称执行过修改后验证。
- 只做了纯规范合规检查：检查过的项目规则、发现和验证证据完整性；没有声明边界时不虚构主场景、改动层或 repo-map 结论。

已路由到外部专门 Skill 的部分由目标 Skill 自己汇总，`amct-coder` 只在混合任务收口时核对共享契约。

### 四个阶段的数据流

当前阶段数据流与 v1 的基本链路一致，但阶段矩阵只在修改 LLM CLI/workflow、阶段产物或跨阶段契约时加载：

| 阶段 | 当前主要职责 | 核心契约 |
| --- | --- | --- |
| eval | `llm-data-workflow`，模型层只参与 adapter forward 契约 | 输入、PPL 口径、bit policy、mask/position 传递 |
| extract_ptq_data | `llm-data-workflow`，模型层保证捕获点与 `PtqUnit.kind` 对齐 | hook、离线输入命名、kwargs、样本与切批 |
| ptq | `llm-data-workflow` + `llm-algorithm` + `ptq-solver-params` | unit 输入/GT、算法状态、solver 调度、参数保存回载 |
| deploy | `deploy-export-code`，dtype/算法/model binding 提供上游契约 | PTQ 参数加载、deploy binding、payload、safetensors/config |

跨阶段改动按上游产物到下游消费者的顺序处理。上游契约未冻结时，不在下游写模型专属或 silent fallback 分支。

### 冲突裁决规则

后续维护和执行遇到冲突时按以下顺序处理：

1. 当前作用域 `AGENTS.md`、用户明确目标与当前已发现专门 Skill 的职责先确定任务归属和授权边界；用户指定的错误实现层不能覆盖仓库边界。
2. `SKILL.md` 与对应 Reference 是 amct-coder 的运行时事实源；本文只记录设计，不反向覆盖已经生效的运行规则。
3. 确认由 amct-coder 承接后，以 repo-map 判断稳定分层，以源码、注册入口、活跃调用链和测试确认当前行为。
4. 主场景按动作和被修改契约选择，多个候选先定位体系和所有权；阶段词和目录相邻不能单独决定主手册。
5. BLOCKER 只停止错误方案；替代路径明确时继续，关键契约确实无法从仓库查出且会改变产物时才询问用户。
6. 设计文档内部冲突时，文末当前设计高于原 v1；发现设计与运行时文件冲突时，以运行时行为为准并同步设计记录。
7. 静态检查、单测、集成和真实硬件证据分级表述，不以低一级证据替代高一级结论。

### 对 `SKILL.md` 的目标形态

当前 `SKILL.md` 保持为短入口，主体结构为：

1. 正向 frontmatter：描述 AMCT 源码编码、实现规划、行为诊断和代码 review 的触发范围。
2. 定位与固定流程：职责归属 -> 声明边界并选主手册 -> 源码证据 -> 交付与验证，共 4 步，与正文章节对应。
3. 职责判断：实现与操作流程分离，命中专门 Skill 时实际切换并继续；路由之后不再规定其加载和交付。
4. 事实门禁与契约缺失处理：规定什么时候自查、什么时候给条件分支、什么时候询问，以及有证据时才推荐。
5. 仓库边界：除纯规范合规 review 外读取 repo-map 最小切片、抽查源码锚点并声明落点。
6. Reference 导航：列出从本入口直接进入的资料并就近标注加载时机，主场景保持粗粒度，提供多场景裁决和未分类探查。
7. 阶段影响与渐进预算：按真实风险扩张上下文，并定义停止条件。
8. 全局 BLOCKER：集中维护跨层禁止事项和继续执行语义。
9. 交付与验证：执行或核对验证，并按本轮实际执行过的动作汇总。

具体算法、数据格式、solver、模型、classic、ops 和 public API 细节继续留在对应 Reference，不上移到主入口。

### 维护检查清单

修改 `amct-coder` 结构时检查：

1. 是否先按动作和当前已发现 Skill 的职责确定任务归属，且路由后继续完成请求。
2. 每份 Reference 的加载时机是否就近声明在导航表对应行或该文件开头，是否没有重新引入中间层的任务类型矩阵。
3. 除纯规范合规 review 外是否读取 repo-map 最小边界并抽查候选落点，纯规范合规 review 是否避免虚构边界结论。
4. Reference 索引是否覆盖所有从本入口直接进入的文件，未列入的是否有其它文件引出且已在表头说明；粗粒度用途是否没有重新枚举动态细节。
5. 多个主场景同时匹配时是否按动作、入口、契约和活跃调用方裁决，而不是预读所有候选。
6. 是否先查源码可确认的事实，再处理真正缺失的契约；实际编码是否避免投机实现，推荐项是否有源码、兼容性或成本证据。
7. 是否区分函数存在、单测覆盖、集成闭环和真实环境验证。
8. 是否只在 LLM workflow、阶段产物和跨阶段契约变化时加载阶段影响规则。
9. 是否应用读取预算和停止条件，避免为了穷举目录而扩张上下文；预算是否仍被表述为软目标而非硬上限。
10. 外部 Skill 职责是否来自当前 metadata，没有在 amct-coder 内复制完整动态清单；主入口提及外部 Skill 的位置是否仍限于判归属、自我围栏和 `delegation.md` 入口，没有管辖已路由出去的流程。
11. `SKILL.md`、Reference、当前设计和迭代记录是否各自承担唯一职责，同一条加载条件是否只有一份正文。
12. forward-test 是否使用物理隔离的新会话，Prompt、答案和评分材料是否对被测 Agent 隔离。
