# amct-coder Skill 测试报告

**测试范围：** 路由触发、越层拦截、委派边界、多子能力联动


## 一、当前测试通过（4 项）

### 测试 1：新增 fp4 dtype（正常路由）

| 项目 | 内容 |
|---|---|
| **测试输入** | 在 `amct_pytorch/quantization/dtypes` 里新增一个 fp4-like dtype，并接入 CLI choices |
| **期望命中** | `amct-coder` / `dtype-quant-module` |
| **实际命中** | ✅ `amct-coder` / `dtype-quant-module` |
| **期望行为** | 读 dtype playbook；补读 public-api-tests；最后读 validation.md |
| **实际行为** | ✅ 读了 `dtype-quant-module.md`、`public-api-tests.md`、`validation.md`，输出了 6 步可执行操作 |
| **验证结论** | ✅ 通过 |

**验证了：** 非模型层改动正确路由；跨子能力补读生效；读取预算 3 个 reference 未超。


### 测试 2：solver 里按模型名写分支（越层拦截）

| 项目 | 内容 |
|---|---|
| **测试输入** | 在 `common/optimization/blockwise_solver.py` 里按 DeepSeek 模型名写特殊分支 |
| **期望命中** | `amct-coder`，报越层 BLOCKER |
| **实际命中** | ✅ `amct-coder`，识别为越层 BLOCKER |
| **期望行为** | 报 BLOCKER；solver 不应出现模型名分支；模型差异应在 adapter/PtqUnit/block 层 |
| **实际行为** | ✅ 第一次回答凭记忆给出结论（行为正确，依据缺失）；修正后引用了三处 Skill 规则覆盖此场景 |
| **验证结论** | ✅ 通过 |

**验证了：** 越层拦截生效；solver 层职责边界清晰；正确引导到 adapter 层。


### 测试 3：新增 classic graph_based optimizer pass（正常路由）

| 项目 | 内容 |
|---|---|
| **测试输入** | 新增 classic graph_based 的 optimizer pass |
| **期望命中** | `amct-coder` / `classic-graph` |
| **实际命中** | ✅ `amct-coder` / `classic-graph` |
| **期望行为** | 不套用 LLM ALGO_REGISTRY，走 classic/graph_based 体系 |
| **实际行为** | ✅ 读了 `classic-graph.md`、`validation.md`；输出了 7 步路线，包含代码锚点、config 接入、测试、public API 懒加载边界 |
| **验证结论** | ✅ 通过 |

**验证了：** classic 和 LLM 两套体系被区分；正确识别 classic 代码位置和配套更新；public API 懒加载边界识别。


### 测试 4：CLI 里 import 模型名分流（越层拦截）

| 项目 | 内容 |
|---|---|
| **测试输入** | 在 `cli/llm/args.py` 里 import Qwen3Model，根据模型名分流参数 |
| **期望命中** | `amct-coder`，报越层 BLOCKER |
| **实际命中** | ✅ `amct-coder`，报越层 BLOCKER |
| **期望行为** | CLI 层不能出现模型类名 import 和按模型名的 if 分支；模型专属分流应由 adapter 提供统一抽象 |
| **实际行为** | ✅ 引用了 `layer-rules.md` 第 44 行和全局 BLOCKER；没有给实现步骤，直接停下，给出正确落地方向 |
| **验证结论** | ✅ 通过 |

**验证了：** CLI 层越层拦截生效；输出没有继续实现错误方案。



## 三、验收标准达成情况
文档规定测试通过的情况

| 验收项 | 目标 | 当前状态 |
|---|---|---|
| 越层场景拦截率 | ≥5 个越层场景，100% 拦截 | 2/5 已测，100% 通过 |
| 合法改动放行率 | ≥3 个合法改动，≥90% 放行 | 2/3 已测（dtype、classic-graph），100% 通过 |
| description 未被关键词带偏 | dtype 命中 dtype，workflow 加 attention 命中越层 | 已验证 dtype 场景；workflow 场景待测 |
| 读取预算 | 默认上限 3 个 reference | dtype 测试恰好 3 个；classic-graph 测试 2 个 |


## 四、重构后计划模式测试样例（待执行）

以下用例只测试职责归属、边界判断、资料读取和计划质量。所有 prompt 都要求 agent 不修改文件、不运行命令。

### 测试 5：在 workflow 中加入模型专属 attention 计算

**测试 Prompt：**

```text
请使用 amct-coder 分析这个需求：我想直接在
amct_pytorch/workflows/llm_ptq.py 中加入一段 Qwen3-Next 专属 attention
计算，用于处理 linear_attention 和 full_attention 的不同 mask。
先不要修改任何文件，也不要运行命令；请先给出边界判断、正确落点和实施计划供我参考。
```

**期望行为：**

- 识别为代码落点与越层问题，不能因用户指定 workflow 就照做。
- 确认由 amct-coder 做边界判断后，读取 repo-map 的「分层边界」「更新规则」并抽查 workflow/model 锚点。
- 声明 Workflow 层只负责编排，模型专属 attention/mask 计算不能落在该层。
- BLOCKER 只阻止错误落点；给出模型 adapter/wrapper 的正确方案，并路由 `model-adapter` 继续形成计划。
- 不修改文件、不运行测试或模型命令。

### 测试 6：在 solver 中按模型名增加特殊调度

**测试 Prompt：**

```text
请使用 amct-coder 评估一个修改：在
amct_pytorch/common/optimization/blockwise_solver.py 中根据 DeepSeek 模型名增加
特殊 unit 调度分支，绕过某些 MoE unit。
先不要修改任何文件，也不要运行命令；请先给出可执行的修改计划和推荐落点。
```

**期望行为：**

- 读取 repo-map 最小边界切片，声明主检查层为 Optimization。
- 命中越层 BLOCKER：solver 不得出现模型名或模型结构分支。
- 以 `ptq-solver-params` 作为当前错误落点的主检查子能力，同时指出模型差异应由 adapter、`PtqUnit` 或 unit 契约表达。
- 正确方案明确时继续给出迁移计划；涉及模型 unit 结构修改时路由 `model-adapter`，不能只报告 BLOCKER 后结束。
- 计划中说明 solver 保持通用所需的回归验证，但不实际执行。

### 测试 7：在通用 QuantLinear 中判断 Qwen MoE

**测试 Prompt：**

```text
请使用 amct-coder 给出计划：我准备在
amct_pytorch/quantization/modules/quant_linear.py 中判断当前模块是否属于
Qwen MoE expert，并为它增加单独的量化分支。
先不要修改任何文件，也不要运行命令；先分析是否越层，并给出正确实现方案。
```

**期望行为：**

- 读取 repo-map 边界并声明 Quant modules 必须保持模型无关。
- 命中越层 BLOCKER，拒绝在通用 `QuantLinear` 中加入 Qwen/MoE 判断。
- 先检查现有通用抽象能否表达需求；模型结构差异应下沉到模型 wrapper/adapter。
- 需要结构实现时路由 `model-adapter` 并继续给出计划；通用量化语义确需扩展时才把 `dtype-quant-module` 作为辅助部分。
- 不因为同时出现 `quant`、`MoE` 关键词就选择多个并列主子能力。

### 测试 8：新增 Qwen3-Next attention wrapper

**测试 Prompt：**

```text
请使用 amct-coder 帮我规划：为 Qwen3-Next 新增一个 attention wrapper，
注册模型，并调整 PtqUnit 拆分以支持 attn-linear PTQ。
先不要修改任何文件，也不要运行命令；只给出职责归属和后续实施计划。
```

**期望行为：**

- 根据动作直接识别为模型结构实现，优先路由 `model-adapter`。
- amct-coder 不应在外部 skill 职责明确前预读 repo-map 或加载自身子能力 playbook。
- 进入 `model-adapter` 后继续按其流程给出计划，而不是只回复“请使用 model-adapter”。
- 说明完成模型实现后需要由 amct-coder 做 `PtqUnit.kind`、extract 命名和 deploy binding 契约 review。
- 全程保持计划模式，不修改文件。

### 测试 9：修复已有 wrapper 的 kwargs 漏传

**测试 Prompt：**

```text
请使用 amct-coder 分析已有 Qwen3 wrapper 的一个局部 bug：
do_block_forward 调用时漏传 position_embeddings，但不需要新增类、修改 wrapper
结构或调整模型注册。
先不要修改任何文件，也不要运行命令；请给出最小修复计划和验证建议。
```

**期望行为：**

- 判断为 amct-coder 可承接的局部非结构模型 bug，不错误路由整个任务到 `model-adapter`。
- 读取 repo-map 边界和相关源码锚点，声明 Model adapter 层的局部修复范围。
- 选择 `llm-model-unit` 作为唯一主子能力；按需读取 attention/wrapper 规则或模型 casebook。
- 检查 adapter forward 对 eval/extract/PTQ 的影响，但不强行选择单一阶段。
- 给出关闭量化等价性、kwargs 透传和相关 mocked workflow 测试计划，不实际运行。

### 测试 10：新增 NPU 算子并接入 amct_pytorch

**测试 Prompt：**

```text
请使用 amct-coder 规划一个混合改动：新增 torch.ops.amct.fake_quant_v2
NPU 算子，包括 Ascend C kernel、C++/Meta 注册，同时在 amct_pytorch 的
通用 quantizer 中增加 Python 调用。
先不要修改任何文件，也不要运行构建或测试；请先拆分职责并给出实施顺序。
```

**期望行为：**

- 识别为混合任务，不能全部留在 amct-coder，也不能全部交给 `amct-ops-dev`。
- kernel、C++ extension、schema、Meta 和算子构建路由 `amct-ops-dev`。
- `amct_pytorch` Python 调用边界由 amct-coder 的 `ops-boundary` 承接，并读取 repo-map 边界后规划。
- 按“先确定 schema/Meta 契约，再实现 kernel，最后接入 Python 调用和测试”的依赖顺序输出统一计划。
- 不运行构建，不创建算子目录。

### 测试 11：修改 deploy index 写入实现

**测试 Prompt：**

```text
请使用 amct-coder 给出计划：修改 amct_pytorch/workflows/llm_deploy.py，
在生成 model.safetensors.index.json 时补充一个新的量化元数据字段，并同步
config.json 和相关单元测试。
先不要修改任何文件，也不要执行 deploy；只给出代码修改计划。
```

**期望行为：**

- 根据“修改实现代码”保留在 amct-coder，不能因为出现 deploy 就路由 `deploy-export`。
- 读取 repo-map 边界并声明 Workflow/deploy 实现和 public surface 的允许范围。
- 选择 `deploy-export-code` 作为主子能力，`public-api-tests` 只作为辅助闭环。
- 标记 deploy 为主影响阶段，并检查 PTQ 参数、binding、index/config 的契约一致性。
- 给出现有 deploy 单测和静态产物检查计划，不实际导出模型。

### 测试 12：新增 dtype 并接入 CLI 和 deploy

**测试 Prompt：**

```text
请使用 amct-coder 规划一个新 fp4-like quant dtype：需要实现 fake quant 和
export_deploy，注册 dtype，接入 CLI quant_dtype choices，并补充单元测试。
先不要修改任何文件，也不要运行命令；请先给出分层计划和影响范围。
```

**期望行为：**

- 由 amct-coder 承接，读取 repo-map 最小边界并抽查 dtype、CLI、deploy 和测试锚点。
- 选择 `dtype-quant-module` 作为唯一主子能力，`public-api-tests` 和 deploy 闭环作为辅助项。
- 不强行把任务归入单一阶段；列出 eval、ptq、deploy 的受影响契约。
- 计划覆盖 dtype 语义、注册、bit policy、CLI choices、deploy payload 和测试，不能只新增 dtype 类。
- 读取预算按阶段分批，不一次加载所有相关 reference。

### 测试 13：classic pass 误用 LLM ALGO_REGISTRY

**测试 Prompt：**

```text
请使用 amct-coder 评估这个方案：在 amct_pytorch/classic/graph_based 下
新增 optimizer pass，并通过 LLM new-path 的 ALGO_REGISTRY 注册和选择它。
先不要修改任何文件，也不要运行测试；请指出问题并给出正确实施计划。
```

**期望行为：**

- 读取 repo-map 边界，识别为 classic graph 体系。
- 选择 `classic-graph` 作为唯一主子能力，阶段影响标记为“不适用”。
- 命中注册体系混用 BLOCKER，拒绝使用 LLM `ALGO_REGISTRY`。
- 给出 classic `AlgorithmRegistry`、graph pass 接入、public API 和测试的正确计划并继续完成，不只停在报错。
- 不预读 LLM 四阶段 playbook，不修改文件。

### 测试 14：实际运行量化 eval 获取 PPL

**测试 Prompt：**

```text
请使用 amct-coder 帮我运行 Qwen3 的 bf16 和 W4A8 quant eval，拿到两组 PPL
并比较结果。先不要运行任何命令，也不要修改文件；请先给出职责归属和执行计划。
```

**期望行为：**

- 根据“实际运行 eval 拿结果”在第一步路由 `quant-run`，不能误判为修改 `llm-data-workflow`。
- amct-coder 不应为这个纯执行任务预读 repo-map 或加载编码子能力 playbook。
- 继续按 `quant-run` 给出命令准备、环境前置、产物记录计划；由于 prompt 要求计划模式，不实际运行。
- 指出结果达标判读不属于 `quant-run`，结果产出后应分别进入 `direct-quant-eval` 等 reviewer skill。
- 不伪造 PPL 或声称已经验证。

## 七、这组用例的统一验收点

| 验收项 | 通过条件 |
| --- | --- |
| 计划模式遵守 | 10 个用例均不修改文件、不运行命令 |
| 动作优先路由 | 不被 eval/deploy/attention 等单个关键词带偏 |
| repo-map 读取时机 | 外部任务先路由；amct-coder 承接后才读取最小边界切片 |
| 越层处理 | 阻止错误方案，给出正确落点并继续完成计划 |
| 单一主子能力 | 合法编码任务只声明一个主子能力，其他作为辅助闭环 |
| 阶段影响 | 仅按需检查；横向能力允许多阶段，classic/ops 可不适用 |
| 渐进披露 | 基础资料之外默认最多追加一份辅助资料 |
| 验证诚实性 | 只给验证计划，不把未执行内容表述为已验证 |
