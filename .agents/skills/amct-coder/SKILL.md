---
name: amct-coder
description: AMCT 仓专属辅助编码总入口。用于新增或修改 AMCT 代码时的场景识别、分层边界判断、实现路径选择和最小自验：覆盖 amct_pytorch LLM new-path 的量化 dtype/bit policy/quant module、算法实现与 ALGO_REGISTRY 注册、模型 MODEL_REGISTRY/adapter/wrapper/PtqUnit、eval/extract_ptq_data/PTQ/deploy workflow、离线输入/GT/PTQ 参数导入导出、block solver 调度；覆盖 classic PTQ/graph_based 图压缩、QAT/校准/剪枝/蒸馏/auto calibration、public API/packaging/tests；识别 amct_ops NPU 算子构建与 amct_pytorch 调用边界，并在需要写 Ascend C/C++/Meta kernel 时转 amct-ops-dev。也用于 AMCT 代码 review，重点检查越层改动、注册缺失、参数保存加载不闭环、验证缺失。
---

# AMCT Coder

## 定位

`amct-coder` 是 AMCT 仓辅助编码总入口。它先做场景分类，再进入对应子能力 playbook，最后按该场景做边界检查和自验。

本 skill 不把所有知识塞进主文件。具体规则在 `references/` 下按场景拆分；使用时只读取命中的 playbook。

原则性表述和模型个案不在 `references/` 维护：分层职责、attention 适配原则、wrapper 合并原则在 `.agents/docs/repo-map.md`，已踩的坑和模型个案在 `.agents/docs/casebook/`。references 只写编码动作、必查项、BLOCKER 和自验命令。

## 先做三件事

1. 判断任务属于哪个子能力；如果跨多个子能力，按数据流顺序处理。
2. 读取命中的场景 playbook，再读对应代码锚点。
3. 改完读 [references/validation.md](references/validation.md)，选最小验证。

无法判断任务属于哪条代码体系（LLM new-path / classic / amct_ops），或需要入口文件清单时，才读 [references/repo-scenarios.md](references/repo-scenarios.md)。下面的全局边界已覆盖常见判断。

回复默认使用中文。执行代码修改前，先简短说明准备修改的模块边界。

## 读取预算

按需加载只在读得少时才有意义。默认上限是 **3 个 reference**：主 playbook + 最多 1 个辅助规则 + `validation.md`。

- 先只读主触发 playbook 就开工，不要预先读全上下游。
- 实际触到跨层改动时再补读该层 playbook 的「上下游依赖」和「BLOCKER」两节，不必整篇读。
- 已超过 3 个 reference 还没定位到落点，说明任务该拆或该委派，见 [references/delegation.md](references/delegation.md)。
- 跨 3 个以上子能力的大改动，优先委派 subagent 承载源码阅读，主上下文只保留边界结论。
- `.agents/docs/amct-coder-design.md` 是维护者文档，执行编码任务时不要读。

## 依赖关系

LLM new-path 的主依赖链是：

`CLI args` -> `workflow` -> `model adapter / PtqUnit` -> `quant module + dtype + algorithm` -> `data provider` -> `solver / ptq params` -> `eval / deploy` -> `public API / docs / examples`。

其中：

- `dtype-quant-module` 定义数值语义和 deploy payload，被 `llm-algorithm`、`llm-model-unit`、`deploy-export-code` 共同消费。
- `llm-algorithm` 通过 `ALGO_REGISTRY` 和 `targets` 接入 `quantization/modules/quant_base.py`，训练态由 `ptq-solver-params` 消费，导出态由 `deploy-export-code` 消费。
- `llm-model-unit` 提供 block 构建、quant block、`PtqUnit`、离线输入命名和 deploy binding，是 `llm-data-workflow`、`ptq-solver-params`、`deploy-export-code` 的上游。
- `llm-data-workflow` 产出 PTQ 输入和 GT 口径，被 `ptq-solver-params` 消费；它也决定 eval 结果能否和 PTQ 结果比较。
- `ptq-solver-params` 保存和回载算法/quant module 状态，必须与 eval/deploy 使用的 `--algos`、target 和 unit 命名一致。
- `deploy-export-code` 是 dtype payload、adapter binding、PTQ 参数和 safetensors index 的汇合点。
- `classic-graph` 与 LLM new-path 独立，只共享顶层 package/API 边界；不要把两套算法注册体系互相套用。
- `ops-boundary` 是 `amct_pytorch` 调用 `amct_ops` 的边界检查；真正 kernel/C++/Meta 实现交给 `amct-ops-dev`。
- `public-api-tests` 是用户可见 API、CLI、packaging、docs/examples 的下游闭环。

跨子能力时的读取顺序，受上面的读取预算约束：

1. 主触发 playbook，整篇读。
2. 只有改动真的要动上游契约时，才补读上游 playbook 的「上下游依赖」和「BLOCKER」两节。
3. 只有改动会改变导出、回载或测试口径时，才补读下游 playbook 的同两节。
4. 最后读 `validation`，只选择与实际改动匹配的最小验证。

上面的依赖关系已经给出契约方向；能据此判断的，不要再去读对应 playbook。

每个场景 playbook 至少回答：任务输入是什么、依赖哪些上游契约、要改哪些代码锚点、要同步哪些下游、哪些情况必须停下、如何最小自验。

## 子能力路由

| 子能力 | 触发语 / 场景 | 必读 playbook |
| --- | --- | --- |
| `dtype-quant-module` | 新量化数据类型、`quant_dtype`、fake quant、real quant、activation/weight quantizer、`QuantLinear`、`QuantizedMatmul`、bit config | [references/dtype-quant-module.md](references/dtype-quant-module.md) |
| `llm-algorithm` | 新增/修改 LWC/LAC/AutoRound/OmniQuant/FlatQuant 类算法、`trainable_params()`、统计 buffer、`export_ptq_params()`、`targets` | [references/llm-algorithm.md](references/llm-algorithm.md) |
| `llm-model-unit` | 模型注册、adapter、wrapper、attention/MLP/MoE、`PtqUnit`、权重 key、mask/position、tied lm_head | [references/llm-model-unit.md](references/llm-model-unit.md) |
| `llm-data-workflow` | eval、extract_ptq_data、PTQ workflow、离线输入、GT、PPL、dataloader、逐层 forward | [references/llm-data-workflow.md](references/llm-data-workflow.md) |
| `ptq-solver-params` | block solver、optimizer、scheduler、unit 调度、PTQ 参数保存/加载、参数导出回载 | [references/ptq-solver-params.md](references/ptq-solver-params.md) |
| `deploy-export-code` | deploy workflow、safetensors、weight index、`quantization_config`、tensorwise/blockwise 导出 | [references/deploy-export-code.md](references/deploy-export-code.md) |
| `classic-graph` | classic `quantize/convert/algorithm_register`、graph_based、QAT、校准、剪枝、蒸馏、auto calibration、tensor decomposition | [references/classic-graph.md](references/classic-graph.md) |
| `ops-boundary` | `amct_ops` 构建、wheel、`torch.ops.amct.*` 调用边界；真正写 NPU kernel/C++ extension | [references/ops-boundary.md](references/ops-boundary.md) |
| `public-api-tests` | 顶层 API、`__all__`、懒加载、setup/build、测试选择、文档样例同步 | [references/public-api-tests.md](references/public-api-tests.md) |

跨场景示例。「主」整篇读，「补」只在实际触到时读该 playbook 的「上下游依赖」和「BLOCKER」两节：

- “新增 mxfp-like dtype”：主 `dtype-quant-module`；补 `deploy-export-code`（若参与导出）、`public-api-tests`（若改 CLI choices）。
- “新增 LLM PTQ 算法”：主 `llm-algorithm`；补 `ptq-solver-params`（若有可训练参数或统计状态）。
- “适配新模型并支持 PTQ”：主 `llm-model-unit`；补 `llm-data-workflow`（unit 输入命名）。这类任务通常跨 4 个子能力，按读取预算应委派 subagent 承载源码阅读。
- “改 classic 量化算法”：只读 `classic-graph`，不要套用 LLM `ALGO_REGISTRY` 规则。
- “amct_pytorch 调用新 NPU 算子”：只读 `ops-boundary`；若要写 kernel，转 `amct-ops-dev`。

## 全局边界

- `amct_pytorch/quantization/dtypes/`：量化数据类型的 fake/export 语义。
- `amct_pytorch/quantization/modules/`：通用量化模块，不写模型名分支。
- `amct_pytorch/algorithms/quant/`：LLM new-path PTQ 算法，通过 `ALGO_REGISTRY` 和 `targets` 分流。
- `amct_pytorch/common/models/llm/...`：LLM 模型 adapter、wrapper、`PtqUnit`、权重加载、mask/pos 捕获。
- `amct_pytorch/common/datasets/`：离线输入、GT、dataloader。
- `amct_pytorch/common/optimization/`：solver、optimizer、scheduler，不混模型分支。
- `amct_pytorch/workflows/`：CLI 背后的流程编排，不放模型细节和算法细节。
- `amct_pytorch/classic/`：classic eager/module 替换和 graph_based 图压缩体系。
- `amct_ops/`：NPU 自定义算子层；算法逻辑不进入 kernel，kernel 不 import `amct_pytorch`。

## 全局 BLOCKER

遇到以下情况先停下并改方案：

- 把模型专属逻辑写进 CLI、workflow、solver 或通用 quant module。
- 混用 LLM new-path `ALGO_REGISTRY` 和 classic `AlgorithmRegistry`。
- 新 dtype 只加类，不更新 `register_dtype()`、CLI choices、bit policy 或 deploy 载荷。
- 新算法没有 `targets`、没有 `trainable_params()`/统计导出策略，或无法被 `build_algorithms_by_target()` 选中。
- `PtqUnit.kind` 与 extract 产出的 `block_<layer>_<target>_in.pkl` 命名不一致。
- PTQ 参数保存、加载、deploy 三段结构不一致。
- eval/deploy 加载 PTQ 参数时漏掉训练时一致的 `--algos`。
- 新 public API 只改源模块，不更新 `__all__` / re-export / 测试。
- 新 NPU 算子绕过 `TORCH_LIBRARY_FRAGMENT(amct, ...)`、缺 Meta backend 或把算法逻辑塞进 kernel。

## 验证

改完必须读取 [references/validation.md](references/validation.md)，选择与命中子能力匹配的最小验证。最终回复写清：

1. 命中的子能力。
2. 修改了哪些层，是否触及敏感层。
3. 复用了哪些已有抽象。
4. 跑了哪些验证。
5. 哪些验证没跑及原因。
6. 是否需要更新 `.agents/docs/repo-map.md`、casebook、docs/examples。

不要把“未跑实验”包装成“已验证”。缺 NPU、CANN、模型权重或数据集时，明确写阻塞。
