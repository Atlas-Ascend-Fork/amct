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

以下用例只测试职责归属、边界判断、资料读取、事实核验和计划质量，统一按以下规则执行：

- 用例允许在同一会话连续执行，以减少重复读取；每例仍需重新按当前动作判断职责，不得直接沿用上一例结论。
- 可以复用已经读取且未变化的源码/reference，但测试记录要标明复用项；复用内容只能作为证据来源，不能替代当前题设的事实核验。
- 允许使用只读工具检查仓库源码、测试和文档；不得修改文件，不得运行测试、构建、量化、评测或模型命令。
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

**测试 Prompt：**

```text
请使用 amct-coder 规划一个混合改动：新增 torch.ops.amct.fake_quant_v2
NPU 算子，包括 Ascend C kernel、C++/Meta 注册，同时在 amct_pytorch 的
通用 quantizer 中增加 Python 调用。
当前只确定了算子名称和集成范围，输入输出、量化参数、支持 dtype/shape 与目标
SoC 尚未定稿。请做只读分析，先拆分职责、列出前置契约，再给出条件式实施顺序；
不要修改文件，也不要运行测试、构建或 NPU 命令。
```

**期望行为：**

- 识别为混合任务，不能全部留在 amct-coder，也不能全部交给 `amct-ops-dev`。
- kernel、C++ extension、schema、Meta 和算子构建路由 `amct-ops-dev`。
- `amct_pytorch` Python 调用边界由 amct-coder 的 `ops-boundary` 承接；只有该部分由 amct-coder 读取 repo-map 和主 playbook。
- 先指出无法从算子名推断的输入输出、量化参数、dtype/shape、数值误差和 SoC 约束，不得自行补全算子语义。
- 按“冻结 schema/Meta/兼容性契约 -> 实现并注册 NPU 算子 -> 接入软依赖或 fallback -> 补分层测试”的依赖顺序输出统一计划。
- 检查 schema、Meta、PrivateUse1、Python wrapper 与上层 quantizer 的签名一致性，禁止让 `amct_pytorch` 在未安装 ops wheel 时导入失败。
- 给出 Meta/导入、CPU fallback、NPU 数值和多 SoC 构建验证计划，但不实际执行，也不创建算子目录。

### 测试 11：修改 deploy index 写入实现

**测试 Prompt：**

```text
请使用 amct-coder 给出计划：修改 amct_pytorch/workflows/llm_deploy.py，
在生成 model.safetensors.index.json 时补充一个新的量化元数据字段，并同步
config.json 和相关单元测试。
字段名、取值来源和下游消费方目前尚未说明。请做只读分析，在不假设 metadata
schema 的前提下给出职责归属、待确认契约和条件式代码修改计划；不要修改文件，
也不要运行测试、构建、deploy 或模型命令。
```

**期望行为：**

- 根据“修改实现代码”保留在 amct-coder，不能因为出现 deploy 就路由 `deploy-export`。
- 读取 repo-map 最小边界切片，声明 Workflow/deploy 实现的允许范围；只有字段进入公开配置或 API 时才扩展 public surface。
- 选择 `deploy-export-code` 作为唯一主子能力；`public-api-tests` 仅在公共契约真实变化时作为辅助闭环。
- 先核对 index/config 的现有写入点、格式和消费者，明确字段名、来源、兼容策略及是否确实需要在两个文件中重复表达，不得发明 schema。
- 标记 deploy 为主影响阶段；仅当字段依赖 PTQ 参数、binding 或 dtype payload 时，才追加对应上游契约检查。
- 计划覆盖单 shard/多 shard、旧 config 兼容、mocked index/config 和静态 JSON 产物检查，不实际导出模型。

### 测试 12：新增 dtype 并接入 CLI 和 deploy

**测试 Prompt：**

```text
请使用 amct-coder 规划一个新 fp4-like quant dtype：需要实现 fake quant 和
export_deploy，注册 dtype，接入 CLI quant_dtype choices，并补充单元测试。
具体编码格式、weight/activation 适用范围、group/block 粒度和部署 layout 尚未确定。
请做只读分析，先列出必须冻结的 dtype 契约，再给出条件式分层计划和影响范围；
不要修改文件，也不要运行测试、构建、量化、deploy 或模型命令。
```

**期望行为：**

- 由 amct-coder 承接，读取 repo-map 最小边界并抽查 dtype、CLI、deploy 和测试锚点。
- 选择 `dtype-quant-module` 作为唯一主子能力，`public-api-tests` 和 deploy 闭环作为辅助项。
- 先确认 dtype 名称是否冲突，以及数值格式、支持 bit、weight/activation、group/block、layout 和误差口径；信息不足时给条件式计划，不得臆造 FP4 语义。
- 不强行把任务归入单一阶段；列出 eval、ptq、deploy 的受影响契约。
- 计划覆盖 fake quant/dequant、`bits == 16` 直通、注册、bit policy、CLI choices、deploy payload/index/config 和测试，不能只新增 dtype 类。
- 修改 deploy 实现代码仍属于 amct-coder 的辅助闭环，不能误路由为交付产物导出的 `deploy-export`。
- 读取预算按阶段分批，不一次加载所有相关 reference。

### 测试 13：classic pass 误用 LLM ALGO_REGISTRY

**测试 Prompt：**

```text
请使用 amct-coder 评估这个方案：在 amct_pytorch/classic/graph_based 下
新增 optimizer pass，并通过 LLM new-path 的 ALGO_REGISTRY 注册和选择它。
这是对拟议注册方案的 review。请做只读分析，核对 classic graph 的现有入口后
指出问题并给出正确实施计划；不要修改文件，也不要运行测试、构建或图转换命令。
```

**期望行为：**

- 读取 repo-map 边界，识别为 classic graph 体系。
- 选择 `classic-graph` 作为唯一主子能力，阶段影响标记为“不适用”。
- 命中注册体系混用 BLOCKER，拒绝使用 LLM `ALGO_REGISTRY`。
- 先区分 classic eager algorithm 与 graph_based optimizer pass 的实际注册/调用入口，不得仅因目录名就假设一定使用 `AlgorithmRegistry`。
- 给出与源码事实一致的 graph pass、配置/parser、custom_op/record_file 和测试接入计划，不只停在报错。
- 只有新增公开入口时才规划顶层懒加载、`__all__` 和 public API 测试，不默认扩大公开面。
- 不预读 LLM 四阶段 playbook，不修改文件。

### 测试 14：实际运行量化 eval 获取 PPL

**测试 Prompt：**

```text
请使用 amct-coder 帮我运行 Qwen3 的 bf16 和 W4A8 quant eval，拿到两组 PPL
并比较结果。当前只给出了模型族和目标精度，尚未提供模型路径、device、
quant_target、quant_dtype、bit config、数据源和输出目录。请先做只读规划，给出
职责归属、缺失输入和执行/判读顺序；不要修改文件，也不要实际运行评测或量化命令。
```

**期望行为：**

- 根据“实际运行 eval 拿结果”在第一步路由 `quant-run`，不能误判为修改 `llm-data-workflow`。
- amct-coder 不应为这个纯执行任务预读 repo-map 或加载编码子能力 playbook。
- 继续按 `quant-run` 列出模型路径、注册名、device、quant_target、quant_dtype、bit config、数据源和输出目录等必需输入；缺失时使用占位符或列为待确认项，不得拼造可执行实参。
- 命令计划以仓库评测模板为准，包含 bf16/quant 两次同口径评测、`--granularity block`、`seq_len=4096` 和产物记录；不实际运行。
- 指出 `quant-run` 只产出并记录两组 PPL，不负责判定是否达标；结果产出后转 `direct-quant-eval` 做 delta 和下一步判断。
- 不伪造 PPL 或声称已经验证。

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
