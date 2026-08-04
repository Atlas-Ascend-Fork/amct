---
name: amct-coder
description: |
  AMCT 仓源代码编码与 review 入口：用于新增、修改、调试或审查 workflow/CLI、solver、dtype、通用 quant module、LLM PTQ 算法、deploy 实现、classic graph、public API、tests、docs 和 examples；也用于判断“代码放哪”“是否越层”“需要补哪些上下游闭环”。
  路由边界：模型 adapter/wrapper/attention/MLP/MoE 结构实现或模型注册转 `model-adapter`；NPU kernel/C++ extension/Meta backend 转 `amct-ops-dev`；实际跑 eval/extract/PTQ 转 `quant-run`；精度判读转 `direct-quant-eval`/`algorithm-validation`；deploy-ready 产物导出转 `deploy-export`。
---

# AMCT Coder

## 定位

把 `amct-coder` 作为 AMCT 编码与 review 的总入口。先判断任务职责归属和仓库边界，再选择一个主子能力 playbook，最后按实际改动做最小验证。

不要把本 skill 当成量化实验或模型适配编排器。命中外部 skill 时切换到对应流程并继续完成请求，不要只回复“应使用某 skill”后结束。BLOCKER 只阻止错误落点或错误方案；正确替代方案明确时继续执行。

默认使用中文。修改代码前，简短说明主落点层和允许改动范围。

## 固定流程

```text
用户请求
  -> 1. 按动作判断职责归属，路由外部 skill
  -> 2. amct-coder 承接后读取 repo-map 边界，声明落点
  -> 3. 判定代码体系，选择一个主子能力
  -> 4. 必要时检查阶段影响并加载辅助资料
  -> 5. 读取源码锚点，编码或 review
  -> 6. 读取 validation.md，执行最小验证
```

## 1. 先判任务职责归属

先根据用户要执行的动作路由，不要只看 `eval`、`deploy`、`attention` 等关键词。

| 任务实质 | 路由 |
| --- | --- |
| 端到端量化生命周期，或需要自动判断适配/方案/实验/deploy 阶段 | `quant-workflow` |
| 新模型族、模型注册、`PtqUnit`/quant block、attention/MLP/MoE wrapper 结构实现 | `model-adapter` |
| `amct_ops/` kernel、Ascend C、C++ extension、schema、Meta backend、算子构建调试 | `amct-ops-dev` |
| 实际运行 bf16/quant eval、extract、PTQ、PTQ 结果评测 | `quant-run` |
| 第一轮方案或下一步 PTQ 算法推荐 | `scheme-recommendation` / `algorithm-recommendation` |
| 直转结果或 PTQ 算法收益判读 | `direct-quant-eval` / `algorithm-validation` |
| 导出 deploy-ready 模型目录和 `deploy_quantization.md` | `deploy-export` |
| GitCode PR/issue 操作 | `gitcode-pr` / `gitcode-issue`；缺失或用户要求更新时用 `default-skills` |

修改 eval/deploy **实现代码**仍属于 amct-coder；运行命令拿结果或导出交付产物属于上表外部 skill。用户点名 skill 时，只有不违反仓库边界和该 skill 的“不适用”条件才遵从。

混合任务按职责归属拆分：本 skill 处理边界、review 和非外部 skill 代码，其余部分进入对应 skill；全部完成后统一检查契约并汇总。

详细边界不清时读 [delegation.md](references/delegation.md)。

## 2. 强制声明仓库边界

只有确认由 amct-coder 承接编码或 review 后，才执行本节；已经路由到外部 skill 的任务不要预读 repo-map。

1. 读取 [repo-map.md](../../docs/repo-map.md) 的「分层边界」和「更新规则」。
2. 抽查候选落点直接相关的源码锚点，确认 map 与当前实现一致。
3. 在内部形成边界声明：主落点层、允许范围、禁止事项、必要辅助层、是否需要写回 repo-map。

repo-map 是稳定架构契约，源码是当前行为事实。两者冲突时，先判断源码是否越层；源码代表预期的新边界或核心流程时，把 repo-map 修正纳入本次任务。

repo-map 前置读取不代表默认写回。仅在框架边界、核心流程、实际注册集合、公共契约变化，或 map 已确认落后于预期源码时更新受影响 section。局部 bug、普通算法文件、未接主链目录和不改变复用边界的模型细节默认不更新。

## 3. 选择主子能力

先判代码体系：LLM blockwise/new-path、classic graph、`amct_ops` Python 调用边界或 public API/packaging。无法定位体系或入口时读 [repo-scenarios.md](references/repo-scenarios.md)。

每次先选择**一个主子能力**；公开面、测试或上下游闭环只作为辅助项。主子能力按动作和被修改契约判断，不按目录相邻或关键词堆叠判断。

| 主子能力 | 使用场景 | Playbook |
| --- | --- | --- |
| `dtype-quant-module` | dtype、bit policy、fake/real quant、quantizer、通用 quant module | [dtype-quant-module.md](references/dtype-quant-module.md) |
| `llm-algorithm` | LWC/LAC/AutoRound/OmniQuant/FlatQuant 类算法、`targets`、训练/统计参数导出 | [llm-algorithm.md](references/llm-algorithm.md) |
| `llm-model-unit` | 模型契约 review、契约对齐、局部非结构 adapter bug；结构实现转 `model-adapter` | [llm-model-unit.md](references/llm-model-unit.md) |
| `llm-data-workflow` | eval/extract/PTQ workflow、dataloader、离线输入、GT、PPL、逐层 forward | [llm-data-workflow.md](references/llm-data-workflow.md) |
| `ptq-solver-params` | solver、optimizer、scheduler、unit 调度、PTQ 参数保存/加载/回载 | [ptq-solver-params.md](references/ptq-solver-params.md) |
| `deploy-export-code` | deploy workflow、safetensors/index、`quantization_config`、导出实现 | [deploy-export-code.md](references/deploy-export-code.md) |
| `classic-graph` | classic quantize/convert/algorithm register、graph_based、QAT、剪枝、蒸馏 | [classic-graph.md](references/classic-graph.md) |
| `ops-boundary` | `amct_pytorch` 调用 `amct_ops` 的 import/schema/wheel/Python 侧边界 | [ops-boundary.md](references/ops-boundary.md) |
| `public-api-tests` | 顶层 API、`__all__`、懒加载、CLI choices、packaging、tests/docs/examples 闭环 | [public-api-tests.md](references/public-api-tests.md) |

读取主 playbook 后再读目标源码和测试。只有实际改变上游契约或下游产物时，才补读相关 playbook 的「上下游依赖」和「BLOCKER」小节。

## 4. 按需检查阶段影响

阶段只描述 LLM 数据流影响，不决定代码职责归属。仅在修改 LLM CLI/workflow、阶段产物或跨阶段契约时检查；classic、ops 和纯 public API/packaging 标记为“不适用”。

| 阶段 | 主要检查 |
| --- | --- |
| eval | 输入、PPL 口径、bit policy、adapter forward |
| extract_ptq_data | hook、离线输入命名、kwargs、`PtqUnit.kind` |
| ptq | unit 输入/GT、算法参数、solver、参数保存/回载 |
| deploy | PTQ 参数加载、deploy binding、dtype/algo payload、safetensors/config |

通用 dtype、quant module 或算法可能同时影响多个阶段，不要强行选择单阶段；列出需要闭环的受影响阶段即可。跨阶段改动按数据流顺序处理，不要预读全部阶段 playbook。

## 5. 渐进加载资料

基础读取固定为：repo-map 最小边界切片 + 主 playbook + 目标源码/测试 + 修改结束前的 [validation.md](references/validation.md)。其余资料按风险追加：

- 落点或越层仍不清楚：读 [layer-rules.md](references/layer-rules.md)。
- attention/wrapper review 或局部模型 bug：读 [attention-rules.md](references/attention-rules.md) 或 [wrapper-rules.md](references/wrapper-rules.md)。
- 模型结构、checkpoint/config 或模型相关精度异常：按 L1 -> L2 -> L3 读取 [casebook](../../docs/casebook/README.md)。
- 主路径或 map/源码差异仍不清楚：只补读 repo-map 对应 section 并扩大源码锚点抽查。

默认一次只加载：repo-map 最小切片 + 1 个主 playbook + `validation.md`，再追加最多 1 份辅助资料。真实跨契约任务按阶段分批加载；读取预算用于防止预读，不用于阻止完成任务。

## 全局 BLOCKER

遇到 BLOCKER 时停止当前错误方案，定位正确层并继续；只有正确方案存在无法从仓库判断的行为、兼容性或范围决策时才询问用户。

- 不在 CLI、workflow、solver 或通用 quant module 中加入模型专属分支。
- 不在本 skill 内新增或结构性修改模型 adapter/wrapper/attention/MLP/MoE；路由 `model-adapter` 后做契约 review。
- 不混用 LLM new-path `ALGO_REGISTRY` 与 classic `AlgorithmRegistry`。
- 新 dtype、算法或 public API 必须完成注册、选择入口、导出/回载和公开面闭环，不能只加局部实现。
- `PtqUnit.kind`、extract 文件命名、PTQ 参数保存/加载、eval/deploy `--algos` 和 deploy binding 必须保持一致。
- 新 NPU 算子实现路由 `amct-ops-dev`；上层调用不得绕过 schema/Meta/import 边界。

## 编码与验证

以源码和测试为事实依据，优先复用现有抽象，限制改动在边界声明范围内。修改前说明主落点；修改后读取 `validation.md`，只运行与实际改动匹配的静态检查、单测或构建。

最终说明：主子能力和改动层、复用的抽象、已跑验证、未跑验证及原因、repo-map/casebook/docs/examples 是否需要更新。缺 NPU、CANN、模型权重或数据集时明确写出，不能把未跑实验表述为已验证。
