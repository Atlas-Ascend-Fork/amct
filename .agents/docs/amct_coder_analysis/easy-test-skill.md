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


## 四、重构后计划模式测试样例

本文件同时包含期望行为、历史输出和问题诊断，是判分材料，不得提供给被测 Agent，也不得让被测 Agent自行打开。测试 10～14 的无答案投喂语单独保存在 `.agents/docs/amct-coder-forward-test-prompts.md`；执行者每次只复制其中一个代码块，不提供文件路径、用例编号或本文件内容。

以下用例只测试职责归属、边界判断、资料读取、事实核验和计划质量，统一按以下规则执行：

- cold forward-test 必须使用全新会话，且当前上下文中没有本文件、设计迭代记录、历史答案或问题诊断；否则只能记为污染样本。
- 有效 cold forward-test 应使用不暴露 `.agents/docs/amct_coder_analysis/` 的隔离工作区。只靠 Prompt 禁止读取属于软隔离，其结果只能作为辅助证据。
- warm regression 允许在同一会话连续执行，以测试资料复用；每例仍需重新按当前动作判断职责，不得直接沿用上一例结论。
- 可以复用已经读取且未变化的源码/reference，但测试记录要标明复用项；复用内容只能作为证据来源，不能替代当前题设的事实核验。
- 允许按命中 skill 的指引只读检查 repo-map、skill reference、直接相关源码和测试；禁止读取 `.agents/docs/amct_coder_analysis/`，禁止查看 Git diff/历史，禁止搜索测试原句。
- 发现被测 Agent读取禁区、引用历史答案，或当前会话已经包含判分材料时，立即将该轮标记为 `CONTAMINATED`，不得计入通过率和 token 效率统计。
- 不得修改文件，不得运行测试、构建、量化、评测或模型命令。
- 记录实际触发的 skill、读取的 reference、最终输出和 token 消耗。不能仅凭最终答案反推读取过程。
- 期望行为只规定职责边界和完成标准，不向被测 agent 泄露具体源码结论；仓库事实与题设冲突时，以核验后的源码为准。

### 测试 5：在 workflow 中加入模型专属 attention 计算

**测试 Prompt：**

```text
请使用 amct-coder 分析这个需求：我想直接在
amct_pytorch/workflows/llm_ptq.py 中加入一段 Qwen3-Next 专属 attention
计算，用于处理 linear_attention 和 full_attention 的不同 mask。
请先做只读分析，给出职责归属、边界判断、正确落点和实施计划；不要修改文件，
也不要运行测试、构建、量化、评测或模型命令。
```

**期望行为：**

- 按任务实质识别为模型专属 attention/mask 结构实现，先路由 `model-adapter`；不得先加载 amct-coder 自身子能力 playbook。
- 进入 `model-adapter` 后按其流程读取 repo-map、模型适配资料和直接相关源码锚点，确认当前实现而不是假设需求尚未实现。
- 声明 Workflow 层只负责编排，模型专属 attention/mask 计算不能落在该层。
- BLOCKER 只阻止错误落点；若仓库已有部分实现，计划应改为核查与补缺，不得重复新增类或注册。
- 具体 mask 生成、保存和回放方案必须标明事实依据与待验证项；不得在未核对运行时参数和 PTQ I/O 契约前下定论。
- 给出 float block、关闭量化等价、混合 attention kwargs、`PtqUnit`/extract 回放的验证建议，但不实际执行。

**测试结果：**

边界判断
llm_ptq.py 不是正确落点，理由有三层：
1. 架构分层：repo-map 分层边界明确 Workflow 层"不放模型细节"，且 llm_ptq.py（254 行）目前是纯编排——不碰 mask、不碰 attention 计算。mask 的捕获/注入实际发生在 adapter 层 common/models/llm/common/base.py（do_embedding_forward 的 Catcher 捕获、get_block_forward_kwargs 注入），保存/加载在 extract workflow 与 ptq_provider.py。加到 llm_ptq.py 既不符合数据流位置，又破坏复用边界。
2. 仓库禁令：amct-coder 全局 BLOCKER「不在 CLI、workflow、solver 或通用 quant module 中加入模型专属分支」；llm-data-workflow BLOCKER「workflow 中加入模型专属 hook 名或结构分支」；attention-rules 禁止模式「在 workflow 中给某个模型补 attention 计算」。三处同指。
3. repo-map 已定性：repo-map 把 Qwen3-Next 记为第四种适配形态，原文明确"blockwise 路径不能只复用 layer0 捕获到的 mask，需要在 adapter 内分别维护 linear-attn mask 和 full-attn causal mask"。这正是你说的需求，官方文档已指定落点在 adapter。
正确落点
主落点 = amct_pytorch/common/models/llm/qwen/qwen3_next/（adapter 目录），两个文件分工：
职责	文件	现状
维护两套 mask + 按层类型注入	qwen3_next.py	目前只做了 apply_quant_attn 按 layer_type 分派，无 mask 维护
消费 mask 的 attention 计算	quant_module.py	QuantQwen3NextLinearAttn 直接调 chunk_gated_delta_rule，丢了原版 apply_mask_to_padding_states；QuantQwen3NextAttn 硬编码 attn_mask=None, is_causal=True
问题本质：base.py 的 Catcher 只捕获 layer0 一个 mask，get_block_forward_kwargs 把同一个 mask 注入所有层。Qwen3-Next 层类型混合（config.layer_types），linear 层要 2D padding mask、full 层要 4D causal mask，混用会形状不匹配或语义错误。
实施计划
1. adapter 维护双 mask（qwen3_next.py）：override do_embedding_forward，在 layer0 捕获后，依据 config.layer_types 同时得到 linear_attn_mask（2D，全 1 时为 None）与 full-attn causal mask（4D），分开存为实例属性。
2. 按层注入：override do_block_forward/get_block_forward_kwargs，用当前 layer_idx 的 layer_type 选对应 mask 注入，不再单 mask 复用。
3. 量化 wrapper 消费（quant_module.py）：QuantQwen3NextLinearAttn.forward 补 apply_mask_to_padding_states(hidden_states, attention_mask)（对齐原版）；QuantQwen3NextAttn 按 attention-rules 必查项处理——有 padding 时显式传 causal mask 且 is_causal=False，无 padding 时保持 is_causal=True。
4. extract/PTQ 闭环：save_ptq_kwargs 目前只落一个 attention_mask.pkl，混合模型需确认 provider 加载端能按 unit 类型取到对应 mask（否则双 mask 只在 eval 生效、PTQ 丢语义）。
5. 验证：BF16 浮点 block forward 正常 → 关量化浮点等价 → BF16 PPL 对照；tests/unit_test/workflows/test_llm_ptq.py 跑通。
需要我继续细化第 1 步（mask 具体怎么从 create_causal_mask 复用/派生）或先看 ptq_io.py/provider 的加载链路再定稿计划吗？

消耗60833tokens

### 测试 6：在 solver 中按模型名增加特殊调度

**测试 Prompt：**

```text
请使用 amct-coder 评估一个修改：在
amct_pytorch/common/optimization/blockwise_solver.py 中根据 DeepSeek 模型名增加
特殊 unit 调度分支，绕过某些 MoE unit。
这是对拟议代码落点的 review。请做只读分析并给出可执行的迁移计划和推荐落点；
不要修改文件，也不要运行测试、构建、量化、评测或模型命令。
```

**期望行为：**

- 读取 repo-map 最小边界切片，声明主检查层为 Optimization。
- 命中越层 BLOCKER：solver 不得出现模型名或模型结构分支。
- 以 `ptq-solver-params` 作为当前错误落点的主检查子能力，同时指出模型差异应由 adapter、`PtqUnit` 或 unit 契约表达。
- 正确方案明确时继续给出迁移计划；涉及模型 unit 结构修改时路由 `model-adapter`，不能只报告 BLOCKER 后结束。
- 在跳过判据未明确时先列出必须确认的信息，不得假定 DeepSeek 版本、expert 类型或过滤语义。
- 若过滤规则只适用于 DeepSeek，不得直接向通用 CLI 增加模型专属参数；只有证明为跨模型通用契约后，才规划公共 CLI/API 闭环。
- 计划中包含 adapter unit 集合测试，以及 solver/workflow 保持通用的回归验证，但不实际执行。

**测试结果：**

已确认关键锚点。给出评估结论。
边界判断
blockwise_solver.py 是错误落点，三条硬约束同时命中：
1. ptq-solver-params 明确规则：「solver 只做优化调度和参数收集，不读取模型名」+「solver 不写模型名分支」。
2. amct-coder 全局 BLOCKER：「不在 CLI、workflow、solver 或通用 quant module 中加入模型专属分支」。
3. repo-map 分层：Optimization 层「不混入模型分支」。
另有结构性原因：BlockwiseSolver.solve() 是按 unit 构造的（llm_ptq.py:203 每个 unit 建一个 solver），它根本不知道"还有哪些 unit、该跳过哪个"。"绕过某些 MoE unit" 是 unit 枚举/调度决策，发生在 solver 之前的 iter_ptq_units() 阶段——solver 想跳过也得先有这个信息源，放在 solver 里既越层又拿不到判断依据。
正确落点
主落点 = DeepSeek adapter 的 iter_ptq_units，不在 solver、也不在 workflow：
层	现状
adapter iter_ptq_units（deepseekv4.py:233、deepseekv3_2.py:146）	v4 对每个 routed expert 产一个 moe unit（iter_indexed_units）；v3.2 委托 super()
blockwise_solver.py	无模型分支
关键点：DeepSeek 的 v3.2/v4 已经是独立 adapter 类（注册名 deepseek_v3_2/deepseek_v4），类本身就是"DeepSeek 专属"，无需再在 solver 里用模型名字符串分叉。若跳过策略只在某 variant（如 V4 vs V4-Flash）不同，用 self.config 属性或新 CLI 参数表达，不要在 solver 里 args.model_name 字符串比较。
可执行修改计划
1. 定跳过判据（前置需求确认）：跳过 shared expert、跳过指定 expert 下标、还是按数据判定未激活 expert？判据不同落点略有差异。
2. adapter 过滤：在 deepseekv4.py iter_ptq_units 的 moe 分支，把 iter_indexed_units(...) 包一层生成器，按判据跳过（如 if expert_idx in skip_set: continue）；deepseekv3_2.py 同样处理。
3. 判据来源（二选一）：
- 判据只跟模型结构绑定 → 直接在 adapter 内用 self.config/权重分析判断，不新增参数；
- 判据需用户控制 → 在 cli/llm/args.py 加参数（如 --skip_expert_indices），adapter 从 self.args 读取。走 public-api-tests 补 CLI 闭环。
4. 保持兼容：_run_blockwise 里已有的"结果已存在则跳过"（llm_ptq.py:196）是通用 resume 机制，与本次跳过互不冲突，不动。
5. 测试：mocked adapter 单测断言跳过后的 unit 集合；test_blockwise_solver.py/test_llm_ptq.py 不变即可证明 solver 无影响。
第 1 步的跳过判据（跳过哪些 MoE unit、为什么）需要你先确认，我再按它定稿第 2–3 步。

### 测试 7：在通用 QuantLinear 中判断 Qwen MoE

**测试 Prompt：**

```text
请使用 amct-coder 给出计划：我准备在
amct_pytorch/quantization/modules/quant_linear.py 中判断当前模块是否属于
Qwen MoE expert，并为它增加单独的量化分支。
这是对拟议代码落点的 review。请做只读分析，先判断是否越层、核实现有抽象能否
表达需求，再给出正确实现方向；不要修改文件，也不要运行测试、构建、量化、
评测或模型命令。
```

**期望行为：**

- 读取 repo-map 边界并声明 Quant modules 必须保持模型无关。
- 命中越层 BLOCKER，拒绝在通用 `QuantLinear` 中加入 Qwen/MoE 判断。
- 先检查现有通用抽象能否表达需求；模型结构差异应下沉到模型 wrapper/adapter。
- 先澄清“单独分支”究竟是位宽/粒度、算法行为还是权重布局；需求未明确时不得直接设计新类。
- 若现有 group/bit policy 已能表达需求，优先给出零改动或最小配置方案。
- 需要模型结构实现时路由 `model-adapter` 并继续形成计划；只有确认是跨模型通用量化语义时，才由 amct-coder 使用 `dtype-quant-module` 承接公共扩展。
- Qwen 专属实现不得落入 `common/quant_apply.py` 或通用 `QuantLinear`；只有多个模型族已出现相同语义时才考虑共享抽象。
- 不因为同时出现 `quant`、`MoE` 关键词就选择多个并列主子能力。
- 验证建议应覆盖 group/bit policy、wrapper forward、关闭量化等价和 deploy payload，不实际执行。

**测试结果：**

已确认 Qwen 全系 MoE（qwen3_moe / qwen3_5_moe / qwen3_next）都统一走 QuantGatedExperts wrapper。结论如下。
是否越层：是，明确越层
quant_linear.py 属于通用 quant module，四条约束同时命中：
1. dtype-quant-module playbook 边界：「quantization/modules/ 只组合 dtype + algorithm + torch module，不写某个模型族的 if 分支；模型专属哪些 Linear 要量化，放 adapter 的 wrapper/binding」。
2. amct-coder 全局 BLOCKER：「不在 CLI、workflow、solver 或通用 quant module 中加入模型专属分支」。
3. repo-map：Quant modules 层默认规则「保持通用」。
4. 结构性事实：QuantLinear 只接收 (args, linear, w_bits, name)，没有任何模型身份信息。要"判断是否 Qwen MoE expert"，只能靠嗅探父模块类型或名字符串，既脆弱又越层。
且当前架构已经为 MoE expert 预留了分离机制，不需要在 QuantLinear 里加分支：
- expert 的量化分支已由 wrapper 结构表达：QuantGatedExperts（moe_common.py）→ 每个 expert 一个 QuantGatedMLP（quant_apply.py）→ 内部构造 QuantLinear。
- expert 与 dense 的位宽分离已通过 group 路由：routed expert 用 group="moe.routed"、shared expert 用 group="moe.shared"，bit policy 按 group 取不同 bits。
正确实现方案
把"Qwen MoE expert 单独量化分支"下沉到 wrapper 层，quant_linear.py 保持通用零改动。按差异化程度分两档：
档位 A — 差异只是 bits / 粒度 / scale（大概率够用）
不用改代码：在 QuantGatedMLP/QuantGatedExperts 构造 QuantLinear 时传对应 w_bits 与 group 即可（现状已支持，只需确认 bit_policy 里 moe.routed/moe.shared 配了期望位宽）。
档位 B — 差异是量化算法 / 权重布局（如 packed gate_up 的 scale）
1. 在 amct_pytorch/common/models/llm/common/quant_apply.py（或 qwen/moe_common.py）新增 QuantLinear 子类（如 ExpertQuantLinear），override forward/export_deploy 写 expert 专属逻辑。
2. QuantGatedMLP 构造 expert Linear 时实例化该子类——结构即判断，无需名字/类型嗅探。
3. 保持 group 路由（moe.routed/moe.shared），保证 PTQ 参数保存/回载与 bit policy 对齐。
4. 若 payload 变化，同步 adapter 的 iter_deploy_bindings（qwen3_next.py 已按 mlp.experts.* 处理）与 export_deploy。
验证
- pytest tests/unit_test/quantization/modules/test_quant_linear.py 不变，证明通用模块未受影响；
- 新增 quant_apply / mocked adapter 测试覆盖 expert 分支的 bits、forward 与 deploy payload。
需要先确认一件事定档位：你说的"单独量化分支"具体指什么差异——不同位宽/粒度，还是不同的量化算法或权重布局？

### 测试 8：新增 Qwen3-Next attention wrapper

**测试 Prompt：**

```text
请使用 amct-coder 帮我规划：为 Qwen3-Next 新增一个 attention wrapper，
注册模型，并调整 PtqUnit 拆分以支持 attn-linear PTQ。
请做只读分析，只给出职责归属、仓库现状和后续实施计划；
不要修改文件，也不要运行测试、构建、量化、评测或模型命令。
```

**期望行为：**

- 根据动作直接识别为模型结构实现，优先路由 `model-adapter`。
- amct-coder 不应在外部 skill 职责明确前预读 repo-map 或加载自身子能力 playbook。
- 进入 `model-adapter` 后读取其规定的必读资料、先查现有实现和已有结果，再形成计划；不能只回复“请使用 model-adapter”。
- 如果 wrapper、注册或 `PtqUnit` 拆分已经存在，应明确给出源码证据并将任务收敛为核查与补缺，不得重复实现。
- 说明完成模型实现后需要由 amct-coder 做 `PtqUnit.kind`、extract 命名和 deploy binding 契约 review。
- 不得把 wrapper 分派测试等同于 `PtqUnit`/extract/PTQ 闭环测试；分别说明已有覆盖和缺失覆盖。
- 全程保持计划模式，不修改文件。

**测试结果：**

这是模型 adapter 结构类任务，按 amct-coder 路由应先加载 model-adapter skill，同时核实现状——因为前面已发现 Qwen3-Next 的 adapter、wrapper、注册和 attn unit 拆分可能已存在。
→Skill "model-adapter"
+ Thought: 5.7s
按 model-adapter 固定流程先核实现状。前面已读过 adapter 与 quant_module 全文，现在抽查测试与 __init__.py 注册链，确认 attn-linear PTQ 是否已接通。
→Read amct_pytorch/common/models/llm/qwen/qwen3_next/__init__.py 
→Read tests/unit_test/common/models/llm/test_model_adapters_mocked_qwen.py [limit=75, offset=1148]
+ Thought: 9.3s
核实现状后，直接给结论。
职责归属
主落点 = model-adapter 层（amct_pytorch/common/models/llm/qwen/qwen3_next/），即本任务路由到 model-adapter skill 职责范围。禁止项：workflow / solver / 通用 quant module 不放模型逻辑。
关键结论：该需求已实现，无需从零新增
你列的三件事在仓库里已经全部存在：
需求	现状	证据
注册模型	qwen3_next 已注册	qwen3_next.py:46 @MODEL_REGISTRY.register(name="qwen3_next")；repo-map 接通列表已含 qwen3_next
新增 attention wrapper	两个 wrapper 已存在	quant_module.py:39 QuantQwen3NextLinearAttn（linear_attention/GatedDeltaNet）、:148 QuantQwen3NextAttn（full_attention）
按 layer_type 分派	已实现	qwen3_next.py:122 apply_quant_attn 按 decoder_layer.layer_type 选 wrapper，build_quant_block（:105）对 attn-linear 调用
PtqUnit 拆分支持 attn-linear	已实现	qwen3_next.py:132 iter_ptq_units 委托 base.py:263 attn-linear 分支，按 layer_type 产出 make_ptq_unit("attn", "linear_attn"/"self_attn", ...)
测试	已覆盖	test_model_adapters_mocked_qwen.py:1148（linear_attention 分派）、:1181（full_attention）、:1451（build_quant_block_attn_linear）
所以本任务的实施计划是核查与补缺，不是新增。
后续实施计划
1. 核查 attn-linear 闭环（逐项确认已存在的链路是否完整）：unit 枚举 → extract 离线输入/attention_mask.pkl kwargs 命名 → PTQ 参数保存/回载 → eval/deploy binding。重点核对 PtqUnit.kind 与离线文件名一致。
2. 补缺（若属于目标范围）：mask 处理是已知待补点——QuantQwen3NextLinearAttn.forward 未调 apply_mask_to_padding_states（原版有），QuantQwen3NextAttn 硬编码 attn_mask=None, is_causal=True；repo-map 已把「adapter 内分别维护 linear-attn mask 与 full-attn causal mask」列为该模型必要适配点（上一轮已给出落点方案）。
3. 验证（需运行，当前未跑，需如实标注）：BF16 baseline（口径 seq_len=4096）→ 关量化浮点等价 → 最小 PTQ 集成 smoke（≥1 quant block + 可枚举 unit + 参数导出/回载）。
4. 文档写回判断：repo-map 已登记 qwen3_next，注册集无变化；若补 mask 缺口，再看是否需更新该模型个案。
是否需要我按第 2 步展开 mask 补缺的具体改动清单，或先出第 1 步闭环核查的核对项？

### 测试 9：修复已有 wrapper 的 kwargs 漏传

**测试 Prompt：**

```text
请使用 amct-coder 分析一个 Qwen3 wrapper 的问题报告：调用 block forward 时
wrapper 没有收到 position_embeddings。报告者怀疑是 do_block_forward 漏传，
但目前没有确认根因；预计不需要新增类、修改 wrapper 结构或调整模型注册。
请做只读核验，给出结论、最小修复条件和验证建议；不要修改文件，也不要运行
测试、构建、量化、评测或模型命令。
```

**期望行为：**

- 判断为 amct-coder 可承接的局部非结构模型 bug，不错误路由整个任务到 `model-adapter`。
- 读取 repo-map 边界和相关源码锚点，声明 Model adapter 层的局部修复范围。
- 选择 `llm-model-unit` 作为唯一主子能力；按需读取 attention/wrapper 规则或模型 casebook。
- 先核验“do_block_forward 漏传”是否成立，检查捕获、保存、加载、注入和 block/wrapper 消费链，不把问题报告当成既定根因。
- 若当前源码能反证题设，应明确拒绝无依据修改，列出仍需复现的信息和可能的其他丢失环节；这应判为通过而不是偏离预期。
- 只有定位到具体丢失环节后才给出条件式最小修复计划，不得用“唯一根因”等过强结论替代证据。
- 检查 adapter forward 对 eval/extract/PTQ 的影响，但不强行选择单一阶段。
- 给出 kwargs 透传、捕获/回放、关闭量化等价和相关 mocked adapter/workflow 测试计划，不实际运行。

**测试结果：**

这是局部非结构 adapter bug，按 amct-coder 应先读 llm-model-unit playbook，并严格核验"漏传 position_embeddings"这个断言是否属实（不盲目接受 bug 前提）。
→Read .opencode/skills/amct-coder/references/llm-model-unit.md 
✱Grep "def do_block_forward|def get_block_forward_kwargs|position_embeddings" in amct_pytorch/common/models/llm (53 matches)
+ Thought: 808ms
核验 qwen3 系 adapter 的 do_block_forward override，确认是否漏传。
→Read amct_pytorch/common/models/llm/qwen/qwen3/qwen3.py [limit=40, offset=55]
→Read amct_pytorch/common/models/llm/qwen/qwen3/qwen3_moe.py [limit=30, offset=70]
+ Thought: 17.8s
继续核验：读 base.py 剩余部分（_build_block_for_forward/_dispatch_block）与 qwen3_5 系的 override，确认漏传是否真实存在。
→Read amct_pytorch/common/models/llm/common/base.py [limit=85, offset=280]
→Read amct_pytorch/common/models/llm/qwen/qwen3_5/qwen3_5.py [limit=25, offset=70]
+ Thought: 9.4s
核验完毕。结论先说：*"do_block_forward 漏传 position_embeddings" 这个前提在当前代码中不成立*，需要先确认根因，不要直接按此修复。
证据链（当前代码未漏传）
position_embeddings 走的是捕获→注入→透传三段：
1. 注入：base.py:153 get_block_forward_kwargs() 在 self.position_embeddings is not None 时注入 kwargs["position_embeddings"]（base.py:157-161）。
2. 调用：base.py:213 do_block_forward 通过 block_kwargs = self.get_block_forward_kwargs()（:234）后 block(sample, **call_kwargs)（:242），position_embeddings 在这里确实被传给 block。
3. 透传：qwen3 系 adapter 的 override（qwen3.py:72、qwen3_moe.py:78、qwen3_5.py:77、qwen3_next.py:93）全部原样委托 super().do_block_forward(...)，没有丢弃任何 kwargs。
所以"do_block_forward 漏传"在当前代码路径上找不到依据。
真实根因候选
唯一会让 wrapper 收不到 position_embeddings 的入口在捕获端：base.py:191 self.position_embeddings = layers[0].position_embeddings，而 Catcher（capture.py:48）只在 layer0 的 kwargs 里含 position_embeddings 时才捕获。若该模型/该 transformers 版本在层内自行算 RoPE、不往 decoder layer 传 position_embeddings，则捕获为 None → 后续所有层都不注入。修复点应在捕获端（该模型 do_embedding_forward），而非 do_block_forward。
最小修复计划（先验证再改）
1. 复现/定位：在 do_block_forward 的 block(sample, **call_kwargs) 前打日志确认 call_kwargs 是否含 position_embeddings；在 do_embedding_forward 捕获后确认 self.position_embeddings 是否 None。两者一对比即可定位漏传发生在"注入前"还是"注入时"。
2. 若捕获为 None：改捕获端——在该模型 do_embedding_forward 里按 position_ids 显式生成/补捕获 position_embeddings（复用模型 rotary_emb），不动 wrapper 结构、不动注册。
3. 若注入环节真丢（与当前代码不符，仅假设）：补 get_block_forward_kwargs 分支或在模型 adapter 的 do_block_forward 里补注入。
4. 测试：mocked adapter 单测断言 get_block_forward_kwargs() 返回含 position_embeddings；wrapper 收到后 cos, sin = position_embeddings 不再报错。
验证建议
- 单测：pytest tests/unit_test/common/models/llm/test_model_adapters_mocked_qwen.py 中 qwen3 系用例，断言 block forward kwargs。
- 若怀疑是捕获端：BF16 PPL 或 wrapper 前向报 TypeError: missing required argument position_embeddings 的堆栈即可佐证。
建议先跑第 1 步定位，确认根因是捕获端还是真的注入端，再落最小修改。需要我按"捕获端为 None"这条假设展开具体改法吗？

### 测试 10：新增 NPU 算子并接入 amct_pytorch

**测试 Prompt：** 使用无答案投喂文件中的“用例 A”，不要从本报告复制。

**期望行为：**

- 识别为混合任务，不能全部留在 amct-coder，也不能全部交给 `amct-ops-dev`。
- kernel、C++ extension、schema、Meta 和算子构建路由 `amct-ops-dev`。
- `amct_pytorch` Python 调用边界由 amct-coder 的 `ops-boundary` 承接；只有该部分由 amct-coder 读取 repo-map 和主 playbook。
- 先指出无法从算子名推断的输入输出、量化参数、dtype/shape、数值误差和 SoC 约束，不得自行补全算子语义。
- 按“冻结 schema/Meta/兼容性契约 -> 实现并注册 NPU 算子 -> 接入软依赖或 fallback -> 补分层测试”的依赖顺序输出统一计划。
- 检查 schema、Meta、PrivateUse1、Python wrapper 与上层 quantizer 的签名一致性，禁止让 `amct_pytorch` 在未安装 ops wheel 时导入失败。
- 给出 Meta/导入、CPU fallback、NPU 数值和多 SoC 构建验证计划，但不实际执行，也不创建算子目录。

**测试结果：**

只读分析：新增 torch.ops.amct.fake_quant_v2
一、职责拆分
这是混合改动，按仓库边界拆成两段，不能由单一 skill 全部承接：
A. amct-ops-dev 承担（硬件算子实现） —— 模板 amct_ops/hifloat8_cast/ 是唯一参照，新算子目录 amct_ops/fake_quant_v2/ 严格按三层布局：
子项	职责
op_kernel/fake_quant_v2_kernel.cpp	UB 预算、tile 计算、round/clamp/scale 向量计算
op_kernel/fake_quant_v2_tiling.h	host/device 共用 tiling struct
op_extension/ops.h	host 接口声明（AscendKernel::）
op_extension/register.cpp	TORCH_LIBRARY_FRAGMENT(amct) schema + TORCH_LIBRARY_IMPL(PrivateUse1) + Meta 后端
op_extension/fake_quant_v2_torch.cpp	tensor 解包、GetCoreMemSize() 算 tile、launch host stub
python/fake_quant_v2/	.so 加载 + 薄包装
CMakeLists.txt	project(fake_quant_v2 LANGUAGES ASC CXX)，按 NPU_ARCH 切 ISA
amct_ops/README.md + ops_init.py	支持算子表 + 顶层 __init__ 文档
tests/amct_ops/test_fake_quant_v2.py	NPU 单测（unittest，对照 CPU 参考）
B. amct-coder ops-boundary 承担（上层调用边界）：
- 在 amct_pytorch 通用 quantizer 链中选接入点：quantization/dtypes/int_impl.py 的 weight_quant/dynamic_per_token_quant（当前 Python fake quant 实算点），经 QuantDequantInt→WeightQuantizer/ActivationQuantizer.fake_quant 汇入。
- 按 common/utils/quant_util.py:41 的既有模式做软依赖 + fallback（@lru_cache 探测 + from amct_ops.xxx import ... + 友好 ImportError），禁止 import 阶段硬失败。
- 校验 schema / Meta / PrivateUse1 / Python wrapper / 上层 quantizer 签名五处一致。
红线（两段共同遵守）：kernel 不 import amct_pytorch；amct_pytorch 不写 kernel/C++ 逻辑；不在 CPU/源码 import 路径强依赖 ops wheel；不做现有仓库中不存在的算子语义发明。
二、已确认事实（源码锚点）
- amct namespace 当前仅有 encode_to_hifloat8/decode_from_hifloat8（register.cpp），无 fake_quant_v2 冲突，命名唯一性无需解冲突。
- 构建/打包自动发现：新增 <op>/python/<pkg>/ 与 <op>/build/*.so 即可被 ops_build.sh 收集，无需改脚本。
- A2/A3 共用 dav-2201 编译产物，UB 差异由运行时 GetCoreMemSize() 决定 tileLength——新 kernel 必须实现 host 侧 tile 计算，否则 A3 浪费 UB。
- 上层已有 hifloat8 调用 amct_ops 的完整 fallback 范式（quant_util.py），可复用为 fake_quant_v2 接入模板。
三、前置契约（必须冻结，当前题设不足以自行补全）
契约	待确认内容
C1 schema	入参：x 之外是否含 scale/zero_point/bits/granularity？动态还是静态参数？输出 1 个还是多个 Tensor？
C2 数值语义	对称/非对称、per-tensor/per-channel/group 粒度、round 模式、clamp 边界、是否 STE 直通梯度
C3 dtype/shape	输入 dtype（fp16/bf16/fp32）、输出 dtype、shape 约束（是否仅限 2D 权重）
C4 误差口径	允许的最大数值误差、NaN/Inf/饱和行为
C5 目标 SoC	仅 A2，还是 A2/A3/A5 都要出产物
C6 兼容性	fake_quant_v2 与上游/旧版 fake_quant_v1 的关系、谁消费
C7 软依赖	amct_pytorch 未装 wheel 时的回退行为（报错时机/提示 vs 静默走 Python）
四、条件式实施顺序
依赖方向：先冻结契约 → 再算子 → 再上层接入 → 最后分层测试。
Phase 0 — 冻结 C1–C7（前置，无代码）
需求方定稿 schema 签名、量化参数、dtype/shape、误差、SoC。此阶段未完成前后续全部挂起。
Phase 1 — 实现并注册 NPU 算子（条件：C1–C5 冻结） amct-ops-dev
1. cp -r amct_ops/hifloat8_cast amct_ops/fake_quant_v2，全量改名（含 CMake project()、文件名、python 包名）。
2. kernel：*_tiling.h 定义 tiling struct；*_kernel.cpp 文件头写 UB 预算注释，host 按 GetCoreMemSize() 算 tileLength，A3 自适应；CopyIn→Compute→CopyOut 三段式。
3. binding：register.cpp 写 schema + TORCH_CHECK dtype/shape 校验 + TORCH_LIBRARY_IMPL(PrivateUse1) + Meta 后端（shape-only，at::empty(...)）。
4. Python wrapper + 算子 README + 顶层 amct_ops/README.md 算子表 + ops_init.py。
5. 验证：bash ops_build.sh fake_quant_v2 → wheel 含 .so → NPU 机导入 torch.ops.amct.fake_quant_v2 → tests/amct_ops/test_fake_quant_v2.py 对照 CPU 参考。
Phase 2 — 接入 amct_pytorch 软依赖/fallback（条件：Phase 1 wheel 可装） amct-coder ops-boundary
在 int_impl.py（或按契约落入的 dtype 层）新增调用分支：NPU 且 amct_ops 已装 → torch.ops.amct.fake_quant_v2(...)；否则走现有 Python 实现。复用 quant_util.py 的探测模式，保证 import 不硬失败。同步五处签名一致性检查。
Phase 3 — 补分层测试（条件：Phase 2 合入）
- tests/amct_ops/：NPU 数值单测（dtype/shape/误差边界）。
- amct_pytorch 上层：tests/unit_test/quantization/ 补 fallback 分支与关闭量化等价测试；test_dtypes.py/test_quant_base.py 回归。
- Meta/导入验证：fake tensor / torch.compile 图模式形状推导不报 "no Meta kernel"。
Phase 4 — 多 SoC 与真实硬件覆盖（条件：C5 定稿）
A2 实机数值验证；按 C5 决定是否补 A3（--soc ascend910_93）与 A5（--soc ascend950，需 CANN 支持 dav-3510）的编译产物验证。未覆盖平台如实标注。
五、验证计划（本次不执行）
- 构建 sanity：ops_build.sh fake_quant_v2 + python3 -m zipfile -l dist/amct_ops-*.whl | grep fake_quant_v2
- 导入/schema：PYTHONPATH=amct_ops/staging python3 -c "import amct_ops.fake_quant_v2; import torch; print(torch.ops.amct.fake_quant_v2)"
- CPU fallback：未装 ops wheel 时 import amct_pytorch 成功且调用回落 Python 实现
- NPU 数值：tests/amct_ops/test_fake_quant_v2.py 对照 CPU 参考
- 多 SoC：A2 实机 + A3/A5 编译通过
待确认项（阻塞 Phase 0）：schema 参数集、量化参数/粒度、dtype/shape、误差口径、目标 SoC。需要我针对其中某项（如 C1 schema 候选签名）展开备选方案供你定稿吗？

### 测试 11：修改 deploy index 写入实现

**测试 Prompt：** 使用无答案投喂文件中的“用例 B”，不要从本报告复制。

**期望行为：**

- 根据“修改实现代码”保留在 amct-coder，不能因为出现 deploy 就路由 `deploy-export`。
- 读取 repo-map 最小边界切片，声明 Workflow/deploy 实现的允许范围；只有字段进入公开配置或 API 时才扩展 public surface。
- 选择 `deploy-export-code` 作为唯一主子能力；`public-api-tests` 仅在公共契约真实变化时作为辅助闭环。
- 先核对 index/config 的现有写入点、格式和消费者，明确字段名、来源、兼容策略及是否确实需要在两个文件中重复表达，不得发明 schema。
- 标记 deploy 为主影响阶段；仅当字段依赖 PTQ 参数、binding 或 dtype payload 时，才追加对应上游契约检查。
- 计划覆盖单 shard/多 shard、旧 config 兼容、mocked index/config 和静态 JSON 产物检查，不实际导出模型。


**测试结果：**

第一次测试大约消耗五万tokens
已完成只读分析（未改任何文件、未跑任何命令）。结论如下。
1. 已确认事实（源码锚点）
- model.safetensors.index.json 唯一写出口是 LlmDeployWorkflow._refresh_weight_index()（llm_deploy.py:186-200），block 与 tensor 两条路径都汇聚到这里（:232、:279）。当前逻辑：保留原始 metadata 键 → 重算 total_size → 写 {"metadata", "weight_map"}。
- _refresh_weight_index 目前签名只有 (original_index, updated_weight_map)，拿不到 pipeline、quant_ignore_layers、cache/bits scheme。
- _load_weight_index()（llm_deploy.py:133-155）单 shard 合成路径的 metadata 也只有 total_size。
- config.json 写出口是 _refresh_config()（llm_deploy.py:157，quant 路径）与 _refresh_config_tensor()（:176，bf16 路径）。quantization_config 由 deploy_export.py::generate_quant_config()（deploy_export.py:52-74）生成，已是「config 契约」的稳定生成点。
- 现有测试：test_refresh_weight_index_writes_metadata_and_total_size（test_llm_deploy.py:228，断言原 metadata 保留 + total_size）、test_load_weight_index_single_shard_synthesizes（:690）、test_run_tensorwise_copies_and_rewrites_weight_index（:578，断言 metadata 保留）、test_refresh_config_*（:248/:263）。
2. 职责归属
事项	归属
index.json 新字段写入	Workflow 层 llm_deploy.py _refresh_weight_index（或同文件私有 helper）
字段值来源的适配器侧数据（如按模型定的 scheme）	Model adapter（pipeline 方法，复用 cache_scheme/bits_scheme 模式）
字段若进 quantization_config	deploy_export.py::generate_quant_config
单测	tests/unit_test/workflows/test_llm_deploy.py
文档	docs/zh/AMCT_Pytorch_LLM.md §2.4（交付物描述）
子能力：deploy-export-code。禁止事项：不在 workflow 加模型专属分支、不发明 schema、不动 dtype/algorithm deploy hook（除非字段来自 payload，目前看不是）。
3. 待确认契约（不确认不编码）
1. 字段名 — 未给出，不发明。
2. 字段层级 — 放 index 的 metadata 内，还是 weight_map 平级的顶层键？（影响 _refresh_weight_index 的写法和下游 loader 兼容性）
3. 值来源 — 以下哪类（或组合）：
- workflow 状态派生：quant_dtype/granularity/is_mx/is_int/is_hif（_refresh_weight_index 内可直接算，签名不变）；
- generate_quant_config 产物或其子集（需在调用方算好，改变 _refresh_weight_index 签名，两处调用点 + 相关 mock 一起改）；
- pipeline 新增方法（adapter 提供，同 cache_scheme 模式）；
- 原始 index metadata 透传（现状已有，无需新逻辑）。
4. 下游消费方与兼容性 — HF transformers / vLLM / infer 仓谁读这个键？metadata 里新增键是否被容忍、期望类型（int/str/嵌套 dict）、是必填还是可选？
5. 适用范围 — block 与 tensor 两路径都写？bf16 的 _refresh_config_tensor（现在会 pop quantization_config）是否也写？是否覆盖原始 metadata 同名键？
6. config.json 落点 — 字段进 quantization_config 内部，还是 config 顶层？内容与 index 字段是同一份还是两份语义？
4. 条件式修改计划
A. index.json（条件：契约 1/2/3/5）
- 若值可由 workflow 状态派生：在 _refresh_weight_index 内（或新增私有 helper）构造字段，签名不变，block/tensor 自动同时生效，测试改动最小。
- 若需 pipeline/ignore 信息：_refresh_weight_index 增加参数（如 quant_metadata: dict | None），在 _run_blockwise（:232 前，有 quant_ignore_layers、self.pipeline）与 _run_tensorwise（:279 前）两处构造传入；bf16 分支按契约 5 决定传空/跳过。
- 保留现有行为：metadata = dict(original_index.get("metadata", {})) → total_size → 新键；同名覆盖策略按契约 5。
B. config.json（条件：契约 6）
- 若进 quantization_config：只改 generate_quant_config（deploy_export.py:52）—— 签名/返回值即契约，_refresh_config 无需改。
- 若为顶层键：在 _refresh_config 落盘处新增；_refresh_config_tensor 按契约 5 决定。
C. 测试（tests/unit_test/workflows/test_llm_deploy.py）
- 扩展 test_refresh_weight_index_writes_metadata_and_total_size（:228）：断言新键存在、类型正确、原 metadata 透传 + total_size 仍成立；若签名变，同步改调用。
- 若签名变：同步更新三个 mock 调用点（:439 _run_blockwise_mocked、:536 _run_blockwise_empty、:578 tensorwise）。
- 若字段进 config：扩展 test_refresh_config_attaches_quantization_block（:248）与 mx 用例（:263）。
- 验证（本任务未运行，仅列出建议命令）：pytest tests/unit_test/workflows/test_llm_deploy.py、python -m py_compile、pre-commit run --files <改动文件>。
D. 文档（条件：若为交付契约） — docs/zh/AMCT_Pytorch_LLM.md §2.4 输出描述补一句。
5. 约束提醒
新字段内容不得与实际量化权重/quantization_config 矛盾（playbook BLOCKER）；值来源需为通用派生或 adapter 提供，不在 workflow 加模型分支。当前无 schema 假设，字段名、类型、位置均以契约为准。

### 测试 12：新增 dtype 并接入 CLI 和 deploy

**测试 Prompt：** 使用无答案投喂文件中的“用例 C”，不要从本报告复制。

**期望行为：**

- 由 amct-coder 承接，读取 repo-map 最小边界并抽查 dtype、CLI、deploy 和测试锚点。
- 选择 `dtype-quant-module` 作为唯一主子能力，`public-api-tests` 和 deploy 闭环作为辅助项。
- 先确认 dtype 名称是否冲突，以及数值格式、支持 bit、weight/activation、group/block、layout 和误差口径；信息不足时给条件式计划，不得臆造 FP4 语义。
- 不强行把任务归入单一阶段；列出 eval、ptq、deploy 的受影响契约。
- 计划覆盖 fake quant/dequant、`bits == 16` 直通、注册、bit policy、CLI choices、deploy payload/index/config 和测试，不能只新增 dtype 类。
- 修改 deploy 实现代码仍属于 amct-coder 的辅助闭环，不能误路由为交付产物导出的 `deploy-export`。
- 读取预算按阶段分批，不一次加载所有相关 reference。

**测试结果：**

消耗3万tokens

只读分析完成（未改文件、未跑命令）。主子能力：dtype-quant-module，辅助 deploy-export-code、public-api-tests。
1. 已确认事实（源码锚点）
- 注册点：quantization/dtypes/__init__.py::register_dtype()（:32-35）只 import QuantDequantInt/QuantDequantMx；DTYPE_REGISTRY 用 @register(name=...) 注册。新 dtype 不加 import 就是纯局部文件。
- 实例化：quant_base.py ActivationQuantizer（:112，is_act=True）与 WeightQuantizer（:150）都用 DTYPE_REGISTRY.get(args.quant_dtype)(bits=..., is_act=...)，注册后自动接入 fake quant，无需改 quant_base。
- CLI 入口：cli/llm/args.py:67 --quant_dtype choices=['int','mxfp']；bf16 不在 choices 但在 deploy 里被特判。
- quant_dtype 字符串强耦合点（新 dtype 必须逐点评估）：
- llm_deploy.py:58-60 is_mx/is_int/is_hif = startswith("mx"/"int"/"hif")；:269 tensorwise 载荷路径 quant_dtype in ["int","mxfp"]；:167 generate_quant_config(..., is_mx=...)。
- deploy_export.py::generate_quant_config（:52-74）用 is_mx 决定 qtype="float"/"int"、format、weight_block_size=[1,32]；quant_payload（:152-161）假定 payload 至少含 qweight，额外字段按 .weight→.extra 命名。
- algorithms/quant/auto_round.py:91-93、auto_clip.py:59/71 有 == "int"/"mxfp" 分支。
- 模型侧：glm5_2/quant_module.py（多处 == "mxfp"）、deepseek_v4.py:257、hyv3.py:175 用 quant_dtype=="int" 选 cache type。
- 现有 fp4 基础设施（新 dtype 是否"重复造轮子"需先界定）：
- LLM 新路径 mxfp.py bits=4 已是 e2m1 + 共享指数 + pack_uint4（:44-45）。
- classic 路径 vars.py 已有 float4_e2m1/mxfp4_e2m1 dtype；experimental/ 有 mxfp4_quantizer、mx_quantize.py。
- bit policy：_ALLOWED_BITS=(4,8,16)，4-bit 已允许，bits==16 直通是既有约定。
2. 必须先冻结的 dtype 契约（不冻结不编码）
#	契约
1	注册名与 CLI 值：registry name、--quant_dtype choice 字符串，以及是否命中 startswith("mx"/"int"/"hif") 前缀（决定 is_mx 等 flag 与 config format）
2	与 mxfp bits=4 的边界：全新 dtype 还是复用既有 e2m1 路径？差异是什么（无共享指数 standalone fp4？不同 packing？）
3	支持位宽集合：仅 4-bit 还是 4/8？需在 _ALLOWED_BITS 内，且 bits==16 直通必须保持
4	适用范围：weight only / weight+activation / 含 cache？fp4-like 一般 weight-only，需显式确认 activation 不误走
5	数值格式：e2m1 / e2m2 / 自定义；是否共享指数（mxfp 式）；对称性；舍入；是否支持 v（rounding offset，auto_clip 传入）；STE 梯度
6	group/block 粒度：per-tensor / per-channel / block_size（如 32）；block_size 整除约束与 generate_quant_group 的 group_size 对应
7	deploy layout：payload 字段名（qweight/weight_scale/weight_bias）、是否 packed（2×fp4→uint8）、scale 编码（fp32/e8m0）、convert_state_dict 的反量化分支（is_mx/is_packed）
8	tensorwise 参与与否：llm_deploy.py:269 的 quant_dtype in [...] 是否加入新名
9	config 映射：新 dtype 属 float-quantized 还是 int-quantized；weight_block_size 是否写入
10	算法/模型分支：auto_round、auto_clip 及带 quant_dtype 分支的 adapter（glm5_2 等）是否要新增分支
3. 条件式分层计划
按数据流顺序、每层依赖上一契约：
- 层 1 dtype 数值层（quantization/dtypes/<new>.py +_impl.py）：实现 fake_quant/forward/export_deploy，遵守 bits==16 直通、is_act 区分；通过 DTYPE_REGISTRY.register，并在 register_dtype() 中 import。本层不读模型名。
- 层 2 CLI 入口：args.py choices 加新值；核对 llm_deploy 的 is_* 前缀判定（契约 1），必要时加 startswith 分支（通用派生，非模型分支）。
- 层 3 quant module 集成：quant_base 无需改（走 registry），但需跑 quant_base/quant_linear/quant_matmul 测试确认 ActivationQuantizer/WeightQuantizer 实例化与 export_deploy 链路。
- 层 4 deploy 闭环：deploy_export.py 的 quant_payload/convert_state_dict/generate_quant_config 按契约 7/8/9 同步；llm_deploy.py 的 tensorwise 判断按契约 8。
- 层 5 算法闭环：auto_round/auto_clip 分支按契约 10。
- 层 6 adapter/缓存：带 quant_dtype 分支的模型与 cache 按契约 4/10；不做则显式排除。
- 层 7 测试/文档：test_dtypes.py（fake quant、export_deploy、直通、误差界、packing roundtrip）、test_quant_base/linear、test_llm_deploy.py、CLI 最小 parser 测试（当前 test_llm_module_entrypoints.py 不断言 choices，需补）、docs/zh/AMCT_Pytorch_LLM.md 参数说明与 configs 样例。
- 若需 NPU 加速：转 amct-ops-dev，本计划只留 Python fallback。
全局 BLOCKER 强制：注册、CLI 入口、导出/回载、公开面四环全闭环，不能只加 dtype 文件。
4. 影响范围（文件级）
必改（若确认为新 dtype）：dtypes/__init__.py、新 dtype 文件、cli/llm/args.py。
条件改动（取决于契约）：workflows/llm_deploy.py、common/models/llm/common/deploy_export.py、algorithms/quant/auto_round.py、auto_clip.py、quantization/bit_policy.py（仅位宽越界时）、带 quant_dtype 分支的 adapter（glm5_2、deepseek_v4.py、hyv3.py）。
测试/文档：tests/unit_test/quantization/{test_dtypes.py,modules/*}、tests/unit_test/workflows/test_llm_deploy.py、CLI 测试、docs。
5. 验证与说明
建议验证（本任务未运行）：pytest tests/unit_test/quantization/test_dtypes.py tests/unit_test/quantization/modules/，涉 deploy 加跑 test_llm_deploy.py，涉 CLI 加跑 CLI 测试；真实 eval/deploy 需要模型权重 + NPU/CANN，未跑前只能到"单测覆盖"级。已确认：仓库已存在 mxfp bits=4 的 e2m1 与 classic/experimental 多套 fp4 实现，契约 2（新 dtype 与既有实现的边界）是开工前最高优先待确认项。

### 测试 13：classic pass 误用 LLM ALGO_REGISTRY

**测试 Prompt：** 使用无答案投喂文件中的“用例 D”，不要从本报告复制。

**期望行为：**

- 读取 repo-map 边界，识别为 classic graph 体系。
- 选择 `classic-graph` 作为唯一主子能力，阶段影响标记为“不适用”。
- 命中注册体系混用 BLOCKER，拒绝使用 LLM `ALGO_REGISTRY`。
- 先区分 classic eager algorithm 与 graph_based optimizer pass 的实际注册/调用入口，不得仅因目录名就假设一定使用 `AlgorithmRegistry`。
- 给出与源码事实一致的 graph pass、配置/parser、custom_op/record_file 和测试接入计划，不只停在报错。
- 只有新增公开入口时才规划顶层懒加载、`__all__` 和 public API 测试，不默认扩大公开面。
- 不预读 LLM 四阶段 playbook，不修改文件。

**测试结果：**

消耗大约一万五tokens。
Review 完成（只读，未改文件未跑命令）。主子能力：classic-graph。
结论
该拟议方案不成立：把 classic/graph_based 的 optimizer pass 注册进 LLM new-path 的 ALGO_REGISTRY，命中全局 BLOCKER「不混用 LLM new-path ALGO_REGISTRY 与 classic AlgorithmRegistry」，且不仅是注册表选择问题，两套体系的注册契约、执行对象、选择入口、依赖边界全部不兼容。
1. 体系事实核对（源码锚点）
体系	注册机制
classic eager	AlgorithmRegistry = Algorithm()（algorithms/init.py:45），register(name, src_op, quant_op, deploy_op)（register_algo.py:28）
classic graph_based	无注册表；BasePass（base_pass.py:20，接口 match_pattern/do_pass(graph)/run）
LLM new-path	ALGO_REGISTRY = Registry("algo")（registry_factory.py:20）；item 需带 metadata["targets"]，register_algorithms() import（quant/init.py:28-38）
2. 问题清单
1. 注册对象不兼容（致命）：graph pass 是 BasePass，无 LLM algo 要求的接口，也无 metadata["targets"]。注册进 ALGO_REGISTRY 后，一旦经 --algos 选中，get_algo_names_by_target 直接抛 "missing registry metadata 'targets'"（quant_base.py:35-36）；就算补了 targets，_build_algorithm（quant_base.py:71-80）按 __init__ 签名推断构造，graph pass 也过不了实例化。结果是死注册或运行时报错。
2. 选择入口语义错位：--algos 是 solver 内 per-unit PTQ 算法开关；graph optimizer pass 作用于整图，没有 per-unit/target 语义。挂上去会让 CLI 语义与执行完全脱节。
3. 执行对象/数据流错位：graph pass 遍历 graph.nodes 做匹配融合（BasePass.run）；LLM algo 在 blockwise quant block 内对 tensor 跑 quantize。同一注册表承载不了两种执行模型。
4. 依赖污染：register_algorithms() 被 llm_deploy.py:69、llm_ptq.py 调用；在其中 import graph_based 会把 ONNX/protobuf 拖进 LLM 轻量 import 链，违反 classic-graph BLOCKER「graph_based import 不能破坏 import amct_pytorch 轻量路径」。
5. 边界错觉：playbook 明确「classic AlgorithmRegistry 中有 gptq/awq/smoothquant 不代表 LLM --algos 可用」；反向同理——LLM ALGO_REGISTRY 承载不了 classic/graph_based 能力。
6. graph_based 有现成接入点：新增 pass 应走 quantize_tool.py 的 add_pass 编排，而不是新建一个跨体系注册表来选 pass（graph_based 目前根本没有"pass 注册表"机制）。
3. 正确实施计划（需先确认真实意图，三选一）
场景 A — 真的是 graph_based 图优化 pass（作用于 ONNX 图）：
1. 实现为 BasePass 子类，放 classic/graph_based/<area>/optimizer/，复用 match_pattern/do_pass/run 契约；
2. 在 quantize_tool.py 对应转换函数（fakequant/quant/retrain 等）用 opt.GraphOptimizer().add_pass(...) 按依赖序接入，参考现有 QuantFusionPass 等；
3. 若需用户开关，在 graph_based 的 config/Configuration 体系内新增字段，不借用 --algos/ALGO_REGISTRY；
4. 同步 graph/node/custom_op/record_file 语义，跑 tests/amct_pytorch/testcase_python/<area>/...。
场景 B — 实际是 LLM blockwise 新算法（作用于 tensor/quant_obj）：
1. 代码放 algorithms/quant/<name>.py，按 quant_base.py 契约实现（quantize/export_deploy/observe_input）；
2. 在 register_algorithms()（quant/init.py）中 import，注册时带 metadata["targets"] ∈ {activation, weight, structure}；
3. 不把任何实现放 classic/graph_based/。
场景 C — LLM 主链的图级预处理/后处理（跨层优化，不属于 per-unit 算法）：
1. 在 LLM workflow/adapter 层显式编排（如 build_quant_block 前后 hook），与算法注册表解耦；不经 --algos 选择。
4. 影响范围与验证说明
- 场景 A：classic/graph_based/<area>/optimizer/ + quantize_tool.py + 对应 graph/node 单测；不触碰 algorithms/quant/、registry_factory.py、cli/llm/args.py。
- 场景 B：algorithms/quant/ + register_algorithms() + 测试（test_quant_base.py 等）；不触碰 classic/。
- 本 review 未跑任何测试/构建/图转换命令（按要求）；建议验证仅按最终确定的场景执行对应单测。未读取 amct-coder-test。
需要你确认拟议 pass 的真实语义（场景 A/B/C），我才能给出逐文件的落点与依赖顺序。

### 测试 14：实际运行量化 eval 获取 PPL

**测试 Prompt：** 使用无答案投喂文件中的“用例 E”，不要从本报告复制。

**期望行为：**

- 根据“实际运行 eval 拿结果”在第一步路由 `quant-run`，不能误判为修改 `llm-data-workflow`。
- amct-coder 不应为这个纯执行任务预读 repo-map 或加载编码子能力 playbook。
- 继续按 `quant-run` 列出模型路径、注册名、device、quant_target、quant_dtype、bit config、数据源和输出目录等必需输入；缺失时使用占位符或列为待确认项，不得拼造可执行实参。
- 命令计划以仓库评测模板为准，包含 bf16/quant 两次同口径评测、`--granularity block`、`seq_len=4096` 和产物记录；不实际运行。
- 指出 `quant-run` 只产出并记录两组 PPL，不负责判定是否达标；结果产出后转 `direct-quant-eval` 做 delta 和下一步判断。
- 不伪造 PPL 或声称已经验证。

**测试结果：**

不到一万tokens。
只读规划完成（未改文件、未运行任何命令）。执行归属 quant-run（implementer），判读归属 direct-quant-eval（reviewer），本规划为编排前置。已确认仓库事实：qwen3 适配器已注册、configs/ 已有 w4a8.yaml 与 bf16.yaml、eval 入口参数（args.py）。
1. 职责归属
阶段	归属
跑 bf16 / quant eval	quant-run
两组 PPL 判读（delta）	direct-quant-eval
若掉点需转 PTQ / 换方案	scheme-recommendation / algorithm-recommendation
本规划	编排层（不执行）
2. 缺失输入（需你提供 / 确认）
输入	CLI 参数
模型路径	--model
device	--device
quant_target	--quant_target
quant_dtype	--quant_dtype
bit config	--bit_config
数据源	—
输出目录	--output_dir
口径确认	--seq_len
已有/可默认：--model_name qwen3（已注册）、--granularity block（必填，否则假无掉点）、--eval_mode bf16（基线）。
3. 执行 / 判读顺序
执行（quant-run，后台跑 + 轮询防超时）
1. 环境前置核验：模型路径可读、device/NPU 可用、评测数据可达；不可达按 BLOCKED 停下不臆造。
2. bf16 基线：python -m amct_pytorch.eval --model <path> --model_name qwen3 --device npu:N --granularity block --eval_mode bf16 --bit_config amct_pytorch/configs/bf16.yaml --seq_len 4096 --output_dir <out> → 记 ppl_bf16。（_validate_eval_mode 要求 bf16 模式无 <16 位，bf16.yaml 空配置即满足。）
3. W4A8 直转评测：python -m amct_pytorch.eval --model <path> --model_name qwen3 --device npu:N --granularity block --eval_mode quant --quant_target <target> --quant_dtype <dtype> --bit_config amct_pytorch/configs/w4a8.yaml --seq_len 4096 --output_dir <out> → 记 ppl_quant。
4. 若 quant_target 有多个角色，逐个跑、逐个记录；不同 target/dtype 的 PPL 不能混比。
判读（direct-quant-eval 职责）
5. 同口径下算 delta = ppl_quant − ppl_bf16；默认阈值 delta ≤ 0.2 为达标。
6. 达标 → 保持直转；不达标 → 建议缩范围（换 quant_target）或转 PTQ（交方案层）。
约束提醒：不在本次修改任何文件；quant 与 bf16 两组必须同模型路径、同 seq_len、同数据源，否则比较无效。

## 七、这组用例的统一验收点

| 验收项 | 通过条件 |
| --- | --- |
| 上下文可追溯 | 允许同会话复用已读资料；每例重新判断职责，并在记录中标明复用项和新增读取 |
| 计划模式遵守 | 允许只读检查；10 个用例均不修改文件，不运行测试、构建、量化、评测、deploy 或模型命令 |
| 动作优先路由 | 不被 eval/deploy/attention 等单个关键词带偏 |
| repo-map 读取时机 | 外部任务先路由；amct-coder 承接后才读取最小边界切片 |
| 越层处理 | 阻止错误方案，给出正确落点并继续完成计划 |
| 单一主子能力 | 合法编码任务只声明一个主子能力，其他作为辅助闭环 |
| 阶段影响 | 仅按需检查；横向能力允许多阶段，classic/ops 可不适用 |
| 渐进披露 | amct-coder 承接部分在基础资料外默认最多追加一份辅助资料；外部 skill 按自身规则读取 |
| 事实核验 | 信息不足时列出待确认项或条件式方案，不臆造 schema、dtype、模型参数或源码现状 |
| token 效率 | 记录每例增量 token（无法直接获取时记录前后差值）；默认目标不超过 15k，超过时说明必要性，明显无依据的 30k+ 判为失败 |
| 验证诚实性 | 只给验证计划，不把未执行内容表述为已验证 |
