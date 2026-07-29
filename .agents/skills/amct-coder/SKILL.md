---
name: amct-coder
description: |
  AMCT 仓辅助编码入口：新增、修改或 review AMCT 代码时，用于场景识别、分层/越层判断、子能力路由和最小自验。
  使用：改 workflow/CLI/solver/dtype/quant module/LLM PTQ 算法/deploy/classic/public API/tests/docs/examples；问“代码放哪/这样写行不行/是不是越层/怎么映射子能力”；在 workflow/CLI/solver/通用 quant module 加模型专属 attention/MLP/MoE/mask/position/weight-key。
  边界：模型层 adapter/wrapper/attention/MLP/MoE 结构实现或模型注册，先判落点后转 `model-adapter`；NPU kernel/C++/Meta 转 `amct-ops-dev`；跑量化实验转 `quant-run`；判读精度转 `direct-quant-eval`/`algorithm-validation`。
---

# AMCT Coder

## 定位

`amct-coder` 是 AMCT 仓辅助编码总入口。它不直接承接所有实现，而是先判断任务属于哪条代码体系和哪个子能力，再进入对应 playbook，最后按场景做边界检查和自验。

模型层（`amct_pytorch/common/models/llm/...` 下的 adapter、wrapper、attention/MLP/MoE 结构、模型注册）实现交给 `model-adapter`。amct-coder 在模型层只保留三件事：委派前落点判断、契约对齐检查（`PtqUnit.kind`、extract 命名、hook 名、deploy binding 等是否与下游一致）和 review。判断规则见 [references/delegation.md](references/delegation.md)。

本 skill 不把所有知识塞进主文件。具体编码规则在 `references/` 下按子能力拆分；使用时只读取命中的 playbook。原则性表述和模型个案不在 `references/` 维护：分层职责、attention 适配原则、wrapper 合并原则在 `.agents/docs/repo-map.md`，已踩坑和模型个案在 `.agents/docs/casebook/`。

## 执行流程

```
用户请求
  -> 判断是否应委派：delegation.md
  -> 判定代码体系：LLM new-path / classic / amct_ops / public API
  -> 命中一个主子能力 playbook
  -> 必要时补读一个辅助规则：layer-rules / attention-rules / wrapper-rules
  -> 修改或 review
  -> 读取 validation.md，选择最小验证
```

回复默认使用中文。执行代码修改前，先简短说明准备修改的模块边界。

## 先做三件事

1. 判断任务属于哪个子能力；如果跨多个子能力，按数据流顺序处理。
2. 读取命中的场景 playbook，再读对应代码锚点。
3. 改完读 [references/validation.md](references/validation.md)，选最小验证。

无法判断任务属于哪条代码体系（LLM new-path / classic / amct_ops），或需要入口文件清单时，才读 [references/repo-scenarios.md](references/repo-scenarios.md)。下面的全局边界和子能力路由覆盖常见判断。

## 读取预算

按需加载只在读得少时才有意义。这里的预算是**软预算**，限制的是一次性加载 reference 的数量，不限制任务实际命中的子能力数量。默认先控制在 **3 个 reference**：主 playbook + 最多 1 个辅助规则 + `validation.md`。

- 先只读主触发 playbook，不要预先读全上下游。
- 实际触到跨层改动时再补读该层 playbook 的「上下游依赖」和「BLOCKER」两节，不必整篇读。
- 已超过 3 个 reference 还没定位到落点，说明任务可能需要先拆阶段或先做委派判断，见 [references/delegation.md](references/delegation.md)。
- 任务确实命中 3 个以上子能力时，不要硬停；按依赖链分阶段处理，每阶段只读当前需要的 playbook，并在最终回复说明为什么超出默认预算。
- 跨 3 个以上子能力的大改动，可委派 subagent 承载源码阅读，主上下文保留边界结论和当前阶段必须执行的规则。
- `.agents/docs/amct-coder-design.md` 是维护者文档，执行编码任务时不要读。

## 子能力路由

子能力不是独立 skill，而是本 skill 下的 reference playbook。映射入口就在这张表；命中后只读对应文件。

| 子能力 | 触发语 / 场景 | 必读 playbook |
| --- | --- | --- |
| `dtype-quant-module` | 新量化数据类型、`quant_dtype`、fake quant、real quant、activation/weight quantizer、`QuantLinear`、`QuantizedMatmul`、bit config | [references/dtype-quant-module.md](references/dtype-quant-module.md) |
| `llm-algorithm` | 新增/修改 LWC/LAC/AutoRound/OmniQuant/FlatQuant 类算法、`trainable_params()`、统计 buffer、`export_ptq_params()`、`targets`、`ALGO_REGISTRY` | [references/llm-algorithm.md](references/llm-algorithm.md) |
| `llm-model-unit` | **不承接模型层实现，转 `model-adapter`**。仅用于模型层契约对齐、模型层 review、局部 adapter bug 修复 | [references/llm-model-unit.md](references/llm-model-unit.md) |
| `llm-data-workflow` | eval、extract_ptq_data、PTQ workflow、离线输入、GT、PPL、dataloader、逐层 forward | [references/llm-data-workflow.md](references/llm-data-workflow.md) |
| `ptq-solver-params` | block solver、optimizer、scheduler、unit 调度、PTQ 参数保存/加载、参数导出回载 | [references/ptq-solver-params.md](references/ptq-solver-params.md) |
| `deploy-export-code` | deploy workflow、safetensors、weight index、`quantization_config`、tensorwise/blockwise 导出 | [references/deploy-export-code.md](references/deploy-export-code.md) |
| `classic-graph` | classic `quantize/convert/algorithm_register`、graph_based、QAT、校准、剪枝、蒸馏、auto calibration、tensor decomposition | [references/classic-graph.md](references/classic-graph.md) |
| `ops-boundary` | `amct_pytorch` 调用 `amct_ops`、wheel/schema/import/Meta 边界；真正写 NPU kernel/C++ extension 时转 `amct-ops-dev` | [references/ops-boundary.md](references/ops-boundary.md) |
| `public-api-tests` | 顶层 API、`__all__`、懒加载、CLI choices、setup/build、测试选择、文档样例同步 | [references/public-api-tests.md](references/public-api-tests.md) |

辅助规则按需读取：

| 辅助文件 | 何时读取 |
| --- | --- |
| [references/layer-rules.md](references/layer-rules.md) | 涉及跨层落点、越层判断、全局 BLOCKER、repo-map 写回 |
| [references/attention-rules.md](references/attention-rules.md) | 涉及 attention wrapper 的 review、委派说明或局部非结构 bug |
| [references/wrapper-rules.md](references/wrapper-rules.md) | 涉及 MLP/MoE/attention wrapper 复用、合并、命名、PTQ unit 边界 |
| [references/delegation.md](references/delegation.md) | 需要判断是否转 `model-adapter`、`amct-ops-dev`、`quant-run` 等其它 skill |
| [references/validation.md](references/validation.md) | 每次修改或 review 结束前必读 |

## 子能力编排

LLM new-path 的主依赖链：

`CLI args` -> `workflow` -> `model adapter / PtqUnit` -> `quant module + dtype + algorithm` -> `data provider` -> `solver / ptq params` -> `eval / deploy` -> `public API / docs / examples`

依赖方向用于决定补读顺序：

- `dtype-quant-module` 定义数值语义和 deploy payload，被 `llm-algorithm`、模型层 wrapper、`deploy-export-code` 共同消费。
- `llm-algorithm` 通过 `ALGO_REGISTRY` 和 `targets` 接入 `quantization/modules/quant_base.py`，训练态由 `ptq-solver-params` 消费，导出态由 `deploy-export-code` 消费。
- 模型层提供 block 构建、quant block、`PtqUnit`、离线输入命名和 deploy binding，是 `llm-data-workflow`、`ptq-solver-params`、`deploy-export-code` 的上游。模型层实现归 `model-adapter`，本 skill 的 `llm-model-unit` 只做契约检查。
- `llm-data-workflow` 产出 PTQ 输入和 GT 口径，被 `ptq-solver-params` 消费；它也决定 eval 结果能否和 PTQ 结果比较。
- `ptq-solver-params` 保存和回载算法/quant module 状态，必须与 eval/deploy 使用的 `--algos`、target 和 unit 命名一致。
- `deploy-export-code` 是 dtype payload、adapter binding、PTQ 参数和 safetensors index 的汇合点。
- `classic-graph` 与 LLM new-path 独立，只共享顶层 package/API 边界；不要把两套算法注册体系互相套用。
- `ops-boundary` 是 `amct_pytorch` 调用 `amct_ops` 的边界检查；真正 kernel/C++/Meta 实现交给 `amct-ops-dev`。
- `public-api-tests` 是用户可见 API、CLI、packaging、docs/examples 的下游闭环。

跨子能力读取顺序：

1. 主触发 playbook，整篇读。
2. 只有改动真的要动上游契约时，才补读上游 playbook 的「上下游依赖」和「BLOCKER」两节。
3. 只有改动会改变导出、回载或测试口径时，才补读下游 playbook 的同两节。
4. 最后读 `validation.md`，只选择与实际改动匹配的最小验证。

## 常见映射

| 用户请求 | 主子能力 | 补读 / 动作 |
| --- | --- | --- |
| “新增 mxfp-like dtype / bit config / quant_dtype” | `dtype-quant-module` | 若参与导出补 `deploy-export-code`；若改 CLI choices 补 `public-api-tests` |
| “新增一个 LLM PTQ 算法 / 改 LWC 参数导出” | `llm-algorithm` | 若有可训练参数或统计状态补 `ptq-solver-params` |
| “适配新模型并支持 PTQ” | `llm-model-unit` | 模型层实现转 `model-adapter`；非模型层的 dtype / workflow / deploy 改动留在本 skill |
| “改 extract_ptq_data / PPL / 离线输入 / GT” | `llm-data-workflow` | 若影响参数保存回载补 `ptq-solver-params` |
| “改 block solver / optimizer / PTQ 参数加载” | `ptq-solver-params` | 若影响算法状态补 `llm-algorithm` |
| “改 deploy safetensors / index / quantization_config” | `deploy-export-code` | 若新增公开开关补 `public-api-tests` |
| “改 classic 量化算法 / graph_based pass / QAT” | `classic-graph` | 不套用 LLM `ALGO_REGISTRY` |
| “amct_pytorch 要调用新 NPU 算子” | `ops-boundary` | 只处理上层调用边界；写 kernel 转 `amct-ops-dev` |
| “新增公开 API / CLI 参数 / 包内容 / 测试入口” | `public-api-tests` | 若参数会影响 LLM 主链，补对应主子能力 |

## 越层请求对照

同样出现 attention、wrapper、MoE 等词时，判断依据是**落点路径和动作**，不是关键词。

| 用户请求 | 落点判断 | 正确响应 |
| --- | --- | --- |
| 在 `amct_pytorch/workflows/llm_ptq.py` 里加一段 Qwen3 attention 计算 | `workflows/` 出现模型结构计算 | 留在本 skill，报越层 BLOCKER；计算应下沉到模型层 wrapper，workflow 只做编排 |
| 在 solver 里按 Qwen/DeepSeek 模型名分支 | `common/optimization/` 出现模型分支 | 留在本 skill，报越层 BLOCKER；模型差异应由 adapter、`PtqUnit` 或 block 抽象表达 |
| 在 `quantization/modules/` 里判断 MoE 或具体模型名 | 通用 quant module 出现结构家族分支 | 留在本 skill，报越层 BLOCKER；通用模块只能表达通用量化语义 |
| 在 CLI 里 import 某个模型类并按模型名分流 | `cli/llm/` 出现模型逻辑 | 留在本 skill，报越层 BLOCKER；CLI 只解析参数并启动 workflow |
| 给 Qwen3-Next 新增 attention wrapper 类 | `common/models/llm/...` 模型层结构实现 | 先用本 skill 判定落点，再转 `model-adapter`；回来后按 `llm-model-unit` 做契约对齐 |
| 修某个已有 wrapper 的局部非结构 bug | `common/models/llm/...` 局部修复 | 可留在本 skill；按 `llm-model-unit`，必要时补 `attention-rules` 或 `wrapper-rules` |

## 全局边界

- `amct_pytorch/quantization/dtypes/`：量化数据类型的 fake/export 语义。
- `amct_pytorch/quantization/modules/`：通用量化模块，不写模型名分支。
- `amct_pytorch/algorithms/quant/`：LLM new-path PTQ 算法，通过 `ALGO_REGISTRY` 和 `targets` 分流。
- `amct_pytorch/common/models/llm/...`：LLM 模型 adapter、wrapper、`PtqUnit`、权重加载、mask/pos 捕获。新增和结构性修改转 `model-adapter`，本 skill 只做契约对齐、review 和局部 bug 修复。
- `amct_pytorch/common/datasets/`：离线输入、GT、dataloader。
- `amct_pytorch/common/optimization/`：solver、optimizer、scheduler，不混模型分支。
- `amct_pytorch/workflows/`：CLI 背后的流程编排，不放模型细节和算法细节。
- `amct_pytorch/classic/`：classic eager/module 替换和 graph_based 图压缩体系。
- `amct_ops/`：NPU 自定义算子层；算法逻辑不进入 kernel，kernel 不 import `amct_pytorch`。

## 全局 BLOCKER

遇到以下情况先停下并改方案：

- 在本 skill 内新增模型 adapter、新增 wrapper 类、改 attention/MLP/MoE 结构或注册新模型；这些应转 `model-adapter`。
- 把模型专属逻辑写进 CLI、workflow、solver 或通用 quant module。
- 混用 LLM new-path `ALGO_REGISTRY` 和 classic `AlgorithmRegistry`。
- 新 dtype 只加类，不更新 `register_dtype()`、CLI choices、bit policy 或 deploy 载荷。
- 新算法没有 `targets`、没有 `trainable_params()`/统计导出策略，或无法被 `build_algorithms_by_target()` 选中。
- `PtqUnit.kind` 与 extract 产出的 `block_<layer>_<target>_in.pkl` 命名不一致。
- PTQ 参数保存、加载、deploy 三段结构不一致。
- eval/deploy 加载 PTQ 参数时漏掉训练时一致的 `--algos`。
- 新 public API 只改源模块，不更新 `__all__` / re-export / 测试。
- 新 NPU 算子绕过 `TORCH_LIBRARY_FRAGMENT(amct, ...)`、缺 Meta backend 或把算法逻辑塞进 kernel。

## 文档写回

只有框架边界或核心流程语义变化时更新 `.agents/docs/repo-map.md`。新增算法文件、局部 wrapper bug、未接主链模型目录、模型专属细节变更默认不改 repo-map。

改 `amct-coder` 结构时可读 `.agents/docs/amct-coder-design.md`；执行普通编码任务时不要读它。

## 验证

改完必须读取 [references/validation.md](references/validation.md)，选择与命中子能力匹配的最小验证。最终回复写清：

1. 命中的子能力。
2. 修改了哪些层，是否触及敏感层。
3. 复用了哪些已有抽象。
4. 跑了哪些验证。
5. 哪些验证没跑及原因。
6. 是否需要更新 `.agents/docs/repo-map.md`、casebook、docs/examples。

不要把“未跑实验”包装成“已验证”。缺 NPU、CANN、模型权重或数据集时，明确写阻塞。
