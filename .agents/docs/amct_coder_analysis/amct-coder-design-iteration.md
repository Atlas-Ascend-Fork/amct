# amct-coder 设计迭代记录

本文记录 `amct-coder` 从初版设计审视到主 `SKILL.md` 重构的讨论过程，供后续维护者理解关键取舍。具体运行规则以 `.agents/skills/amct-coder/SKILL.md` 和命中的 reference 为准，完整设计以 `amct-coder-final-design-v0.md` 为准。

## 1. 初始目标

初版把 `amct-coder` 定位为 AMCT 仓编码总入口，主要负责：

- 识别代码体系和修改落点。
- 拦截 CLI、workflow、solver、通用 quant module 中的模型专属逻辑。
- 路由 dtype、算法、workflow、solver、deploy、classic、ops boundary、public API 等子能力。
- 修改结束后选择最小验证。

初版已经具备 9 个子能力 playbook，但主文件同时包含依赖链、常见映射、越层示例、全局边界和验证说明，运行时入口偏重。

## 2. 第一轮审视：先修职责和继续执行语义

讨论首先发现三个关键问题：

1. 冲突裁决写成“用户要求优先”，可能让用户指定的错误实现路径覆盖仓库边界。
2. BLOCKER 容易被理解为终止整个任务，而不是只阻止错误方案。
3. “委派到其它 skill”容易退化为只回复路由建议，不继续完成用户请求。

形成的结论：

- 当前作用域 `AGENTS.md`、独立 skill 职责、repo-map 分层边界和公共契约优先于用户指定的代码落点。
- 保留用户业务目标，但拒绝越层实现；正确替代方案明确时直接继续。
- 命中外部 skill 后切换其流程并继续执行；混合任务按职责拆分，最后统一做契约对齐和汇总。
- 区分三类 deploy 请求：修改 deploy 实现归 `deploy-export-code`，实际导出交付归 `deploy-export`，eval/extract/PTQ 跑数归 `quant-run`。

## 3. 第二轮审视：阶段不是代码职责

要求 LLM new-path 任务必须先判 eval / extract / ptq / deploy 阶段。讨论后认为，阶段适合描述运行数据流，不适合作为编码任务的首要分类。

原因：

- dtype、quant module、algorithm 是横向能力，可能同时影响 eval、ptq 和 deploy。
- 参数保存、unit 命名、deploy payload 等是跨阶段契约，应表达为“主阶段 + 受影响阶段”。
- classic、ops、public API/packaging 不属于 LLM 四阶段。
- `deploy`、`eval` 等关键词不能区分“修改实现”与“运行命令”。

因此把“必须判阶段”改为“按需检查阶段影响”：

- 先确定任务职责、仓库层和主子能力。
- 修改 LLM CLI/workflow、阶段产物或跨阶段契约时，再检查主阶段和受影响阶段。
- 通用能力列出需要闭环的阶段，不强行选择单阶段。
- `quant-workflow` 仍以生命周期阶段为核心；`amct-coder` 以代码职责为核心。

## 4. 第三轮审视：渐进披露与编码前必读

初版一方面要求渐进披露，另一方面要求所有 LLM/cross-layer 任务固定通读 repo-map、casebook 和 `AGENTS.md`，两者存在冲突。

讨论形成两层读取模型：

- 基础读取：完成当前编码职责所必需的稳定边界、主 playbook、目标源码/测试和 validation。
- 条件读取：repo-map 其它章节、casebook、attention/wrapper 规则、上下游 playbook。

`AGENTS.md` 是当前作用域已经生效的约束，不要求为了摘抄规范重复通读。casebook 只在模型结构、checkpoint/config、模型契约或相关精度异常时加载。

## 5. 第四轮审视：repo-map 边界读取前置

随后进一步明确：既然 `amct-coder` 的核心职责是分层和越层判断，就不应先凭关键词猜落点，判断困难后才补读 repo-map。

最终采用两级入口：

1. 先按用户动作和独立 skill 职责排除跑数、判读、模型结构实现、NPU kernel、deploy 交付、GitCode 操作等外部任务。
2. 确认由 `amct-coder` 承接编码或 review 后，强制读取 repo-map 的「分层边界」「更新规则」，抽查候选源码锚点并形成边界声明。

边界声明包含：主落点层、允许范围、禁止事项、必要辅助层、repo-map 是否需要写回。

repo-map 读取前置不表示默认写回。只有框架边界、核心流程、注册集合、公共契约变化，或 map 已确认落后于预期源码事实时更新；局部 bug、普通算法文件和不改变复用边界的模型细节默认不更新。

## 6. 复杂度控制

讨论确认设计文档可以保留理由和裁决过程，但运行时 `SKILL.md` 不能照搬完整设计。信息按用途分布：

| 内容 | 位置 |
| --- | --- |
| 运行时必须执行的判断和硬规则 | `SKILL.md` |
| 子能力执行步骤、BLOCKER、验证方法 | `references/*.md` |
| 设计理由、方案比较、冲突分析 | 本目录设计文档 |
| 稳定仓库分层和架构事实 | `repo-map.md` |
| 模型经验和已踩坑 | `casebook/` |

复杂度控制规则：

- 每次先确定一个主子能力，其它只作为辅助闭环。
- 固定读取 repo-map 最小边界切片、一个主 playbook 和 validation。
- 默认再追加最多一份辅助资料；真实跨契约任务分阶段加载。
- 不在主文件重复维护完整数据流、模型名单、详细测试命令和大段示例。

## 7. 当前主流程

最终将 `SKILL.md` 收敛为以下流程：

```text
动作与任务职责归属
  -> 外部 skill 路由并继续执行
  -> amct-coder 承接后读取 repo-map，声明边界
  -> 判定代码体系，选择一个主子能力
  -> 按需检查阶段影响并加载辅助资料
  -> 读取源码锚点，编码或 review
  -> 读取 validation.md，执行最小验证
```

## 8. 第五轮审视：测试 10～14 暴露的渐进披露冲突

测试 10～14 的动作路由整体正确，但出现两个相关问题：

1. 主 `SKILL.md` 要求渐进加载和最小源码切片，子能力 reference 却统一使用“必读文件”标题，容易被解释为必须全文读取列表中的所有源码和测试。
2. deploy、dtype 等 playbook 的源码列表较长，只读计划也固定加载 `validation.md`，使简单 review 自然膨胀到 30k～50k token；这与测试报告默认不超过 15k 的目标冲突。

测试同时暴露了渐进读取不足时的事实风险：

- 契约尚未冻结，却提前把具体实现文件当成最终落点。
- 只确认 helper 存在，没有核对调用方、注册或编排入口，就把它描述成活跃运行路径。
- 连续测试虽然允许复用上下文，但没有记录每轮新增读取与复用项，无法审计读取顺序和增量 token。

### 8.1 已完成修改

本轮保留“一个总入口 + 内部子能力 reference + 外部独立 skill”的结构，只调整读取契约：

- 主流程改为只有实际代码改动后才固定读取 `validation.md`；纯只读规划默认不加载，review 仅在判断测试证据等级或验证完整性时加载。
- 明确 reference 是任务规则，reference 中的源码路径是候选锚点，不是全文必读清单。
- 首轮默认用 `rg` 定位并选择 2～4 个最相关的源码/测试锚点，只读直接定义、调用链和断言切片。
- 只有边界、调用链或跨契约影响仍缺证据时才扩大读取；辅助 reference 默认最多一份。
- 增加停止扩张条件：边界已确定、活跃或未接通状态有源码证据、事实/待确认契约/条件式方案已区分、修改面和最小验证范围已明确。
- 契约未冻结时，具体文件和函数只能列为候选锚点；函数存在不能替代活跃调用链核验。
- 连续任务必须记录本轮新增读取和复用项，不能用上一任务结论替代当前事实核验。

9 个主子能力 reference 的“必读文件”已统一改为“源码锚点（按需选择）”，并按各自任务类型补充首轮选择规则：

- `dtype-quant-module`
- `llm-algorithm`
- `llm-model-unit`
- `llm-data-workflow`
- `ptq-solver-params`
- `deploy-export-code`
- `classic-graph`
- `ops-boundary`
- `public-api-tests`

`public-api-tests` 同时删除了重复读取 `AGENTS.md` 的要求；作用域约束已经生效，不应为了摘抄再次加载。

以上决定取代第 4、6、7 节中“固定读取目标源码列表和 validation”的旧口径。没有修改各子能力的职责、执行步骤、BLOCKER、实现规则或验证标准。

### 8.2 本轮验证

- `git diff --check` 通过。
- `quick_validate.py .agents/skills/amct-coder` 输出 `Skill is valid!`。
- 未运行仓库源码单测；本轮只修改 skill Markdown，不涉及运行时代码。

## 9. 子能力定向重构计划

下一阶段不整体重写全部子能力，也不把内部 playbook 升格为独立 skill。优先处理测试 10～13 已经给出失败证据的四个子能力；其余子能力继续保留，等对应测试暴露具体问题后再改。

| 优先级 | 子能力 | 已暴露问题 | 计划修改 | 对应回归目标 |
| --- | --- | --- | --- | --- |
| P0 | `deploy-export-code` | 把未调用的 `_refresh_config_tensor()` 描述为实际 bf16 路径；index/config 测试计划未完整覆盖单/多 shard、旧配置兼容和静态 JSON | 按 index/config、payload/binding、blockwise/tensorwise 分流锚点；强制从写入点反查调用方；为每类产物列最小测试矩阵 | 测试 11 冷启动事实无误，增量 token 目标 ≤12k |
| P0 | `dtype-quant-module` | 注册、CLI、deploy、algorithm、adapter 被一次性展开；未知 dtype 契约下出现 weight-only 倾向 | 按注册入口、数值语义、deploy payload、算法兼容、模型条件影响分批核验；未冻结格式和适用范围前禁止给默认语义 | 测试 12 冷启动保持闭环且增量 token 目标 ≤18k，超 15k 时说明跨契约原因 |
| P1 | `ops-boundary` | schema 未冻结就提前锁定 `int_impl.py`；上层软依赖与算子实现细节混写 | 明确“候选接入点”与“最终落点”门禁；上层只核验 import/schema/wrapper/fallback，kernel/C++/Meta/CMake 交给 `amct-ops-dev`；统一从仓库根目录表达构建命令 | 测试 10 不再臆定实现点，增量 token 目标 ≤20k并记录外部 skill 消耗 |
| P1 | `classic-graph` | `BasePass.do_pass` 签名概括不准；新增 pass 计划遗漏 optimizer `__init__.py`/`__all__`；入口定位过于泛化 | 区分 classic eager 与 graph_based 后，核对准确 pass 接口、导出闭环和实际 `GraphOptimizer.add_pass()` 调用点 | 测试 13 冷启动完整命中注册体系 BLOCKER，增量 token 目标 ≤12k |

`quant-run` 是外部执行 skill，不属于 amct-coder 内部子能力。测试 14 暴露的问题应单独修改其文档：明确 eval 当前固定使用 Wikitext-2，并区分真正必填参数、有默认值但建议显式指定的参数，以及当前不存在的数据源 CLI 参数。目标是保持测试 14 增量 token ≤8k。

### 9.1 四个子能力重构实施结果

以上计划已按 `deploy-export-code`、`dtype-quant-module`、`ops-boundary`、`classic-graph` 的顺序完成。重构前分别核对了子能力映射、repo-map 分层、设计资料、真实源码定义、活跃调用链、注册或编排入口以及现有单测；修改范围只包含四份 playbook，没有改动仓库运行时代码，也没有把内部子能力拆成独立 skill。

| 子能力 | 已完成的重构 | 通过源码核对后固化的关键事实 |
| --- | --- | --- |
| `deploy-export-code` | 以 index/config、blockwise payload、tensorwise 转换和文档产物分流；要求从产物写入点反查调用方；补齐单/多 shard、旧配置兼容、静态 JSON 和 payload/index/config 一致性测试矩阵 | `_refresh_config_tensor()` 和 `_convert_tensor()` 当前只有定义与直接单测，没有活跃运行时调用方；blockwise/tensorwise 当前都调用 `_refresh_config()`；tensorwise 的 `loaded_files` 生命周期需要结合调用链验证，不能仅因逐 shard 循环就声称内存受控 |
| `dtype-quant-module` | 增加 dtype 契约冻结门禁，按 registry、数值语义、quant module、bit policy、deploy、algorithm、adapter 和 ops 条件分批加载；明确未知格式不得默认推断 weight-only 或 weight+activation | 当前 registry 只有 `int`/`mxfp`，bit policy 允许 4/8/16，activation 通过 `is_act=True` 区分，CLI choices 为 `int`/`mxfp`；deploy 同时存在名称前缀判断和 `int`/`mxfp` 精确判断，新增 dtype 不能只改注册或 CLI |
| `ops-boundary` | 将 OWN 范围收敛为 `amct_pytorch` 上层调用、软依赖和 fallback；共享 schema 未冻结时只允许列候选接入点；kernel/C++/schema/Meta/CMake/build/wheel 实现明确转 `amct-ops-dev`；构建命令统一从仓库根目录表达 | 当前 HiFloat8 在原生 `torch_npu` 不可用时于调用点延迟 import `amct_ops`，缺 wheel 时抛 `ImportError`；Meta 只负责 shape/dtype 推导，不是 CPU 数值 fallback；不能在未知语义下预先锁定 `int_impl.py` |
| `classic-graph` | 明确 classic eager 与 graph_based 是两套注册/编排体系；分别列出 pass 基类、导出闭环和真实 `add_pass()` 接入要求；增加 `__init__.py`/`__all__`、编排入口和回归矩阵 | classic eager 使用 `AlgorithmRegistry` 与 classic `ModelOptimizer`；graph_based 没有统一 pass registry，而是由 `GraphOptimizer.add_pass()` / graph_based `ModelOptimizer.add_pass()` 编排；ONNX pass 常用 `BaseFusionPass`，module pass 使用 graph_based `BaseModuleFusionPass`，common `BasePass` 仅局部使用 |

四份 playbook 均保留“源码锚点按需选择”的渐进披露规则，并把当前事实、待确认契约、条件扩展和 BLOCKER 分开表达。这样既避免测试 10～13 中的提前落点和伪活跃调用链，也不要求每次任务全文读取所有候选文件。

本轮没有修改 `quant-run`。测试 14 的外部执行 skill 问题仍按上文作为独立后续项，避免把执行参数规则混入 `amct-coder` 内部子能力。

## 10. 下一轮测试与记录要求

完成上述子能力重构后，重新执行测试 10～14。连续会话用于验证资料复用，冷启动会话用于验证 skill 的独立路由和泛化，两类结果不能互相替代。

每例至少记录：

- `session`：cold 或 warm，以及是否与上一例连续。
- 实际触发的主 skill、外部 skill、主子能力和辅助 reference。
- 本轮新增读取的 reference、源码锚点和测试锚点。
- 从前序任务复用的资料。
- 开始 token、结束 token、增量 token；只能取得近似值时说明统计口径。
- 是否修改文件、运行命令，以及未运行验证的原因。
- 路由、源码事实、计划完整性、过程合规、token 效率五项独立结论。

测试时只从 `.agents/docs/amct-coder-forward-test-prompts.md` 复制单个无答案代码块，不向被测 Agent 提供该文件路径、用例编号、期望答案、既有问题诊断或本迭代计划。测试报告中的原 Prompt 已删除，避免搜索题目原句时直接命中答案。

cold forward-test 应在不暴露 `.agents/docs/amct_coder_analysis/` 的隔离工作区和全新会话中执行；同仓库仅靠 Prompt 禁止读取属于软隔离，只能作为辅助证据。被测 Agent若读取分析目录、Git diff/历史，或者上下文已经含有判分材料，该轮标记为 `CONTAMINATED`，不计入通过率与 token 效率。连续会话只作为 warm regression，不能替代独立 forward-test。

## 11. 重构后回归结果与 token 预算评估

测试 10～14 已完成第二轮回归。当前记录的第二轮增量 token 分别为：测试 10 约 65k、测试 11 约 20k、测试 12 约 25k、测试 13 约 10k、测试 14 约 6k，合计约 126k。第一轮已记录部分合计约 156k～166k，因此第二轮总量下降约 19%～24%，方向正确，但距离合理总预算仍有明显差距。

token 效率必须使用“本轮增量 token”口径，不能把 warm session 的累计上下文直接当成本轮消耗。可取得详细 usage 时，分别记录 uncached input、cached input、工具输出、推理和最终输出；只能取得总量时，记录开始值、结束值及差值。cold forward-test 与 warm regression 分开统计，不互相替代。

### 11.1 分复杂度预算

只读分析不适合统一卡死在 15k，应按职责跨度和待核验契约分档：

| 测试 | 任务复杂度 | 建议区间 | 硬上限 | 第二轮实际 | token 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| 测试 10：NPU 算子并接上层 quantizer | 混合任务，跨 `amct-ops-dev` 与 `ops-boundary` | 18k～25k | 30k | 约 65k | 失败 |
| 测试 11：deploy metadata review | 单主子能力，核对 index/config 活跃链 | 8k～12k | 15k | 约 20k | 超标 |
| 测试 12：新 dtype 跨契约规划 | dtype 主能力，CLI/deploy 为条件闭环 | 12k～18k | 22k | 约 25k | 偏高 |
| 测试 13：classic 注册体系 review | 单体系边界与真实编排入口核验 | 8k～12k | 15k | 约 10k | 通过 |
| 测试 14：eval 执行规划 | 外部 skill 路由与参数门禁 | 4k～8k | 10k | 约 6k | 通过 |

五例合理总预算约 60k～70k。硬上限用于识别无效扩张，不表示接近上限就是优秀；事实和计划已足够时应立即停止读取。

### 11.2 Agent 决策质量

| 测试 | 已符合预期 | 主要问题 | 综合判定 |
| --- | --- | --- | --- |
| 测试 10 | 正确拆分 `amct-ops-dev` 与 `ops-boundary`；识别 schema、数值、dtype/shape、STE、fallback、SoC 均未冻结；保持五接口面一致性要求 | 契约冻结前仍读取 `hifloat8_cast` 全目录，并展开复制目录、双缓冲、UB、stream launch 和多 SoC 命令等具体实现；候选消费点和 kernel 方案展开超过当前决策所需 | 部分通过：路由和门禁正确，过程与 token 失败 |
| 测试 11 | 保留在 `deploy-export-code`；正确核对 `_refresh_weight_index()` 活跃调用链；确认 `_refresh_config_tensor()`、`_convert_tensor()` 当前无运行时调用方；未发明 metadata schema | 测试计划未明确完整覆盖单 shard、多 shard、旧 config 兼容和静态 JSON 产物一致性；读取完整 workflow 测试文件仍偏重 | 条件通过 |
| 测试 12 | 选择 `dtype-quant-module`；识别现有 `mxfp` 4 bit 已覆盖一种 MXFP4 语义；未默认推断 weight-only；覆盖 registry、bit policy、CLI 和 deploy 条件闭环 | 应增加“目标语义等同现有 MXFP4 时不新增 dtype”的零改动分支；eval/PTQ/deploy 阶段影响未显式分开；提前扫描 classic、experimental、algorithm、adapter 和 NPU 条件路径，读取范围偏宽 | 条件通过 |
| 测试 13 | 准确区分 classic eager `AlgorithmRegistry`、graph_based 显式 `add_pass()` 编排与 LLM `ALGO_REGISTRY`；覆盖 pass 基类、`optimizer/__init__.py`、`__all__`、真实调用点和条件式测试 | 未显式标记 LLM 阶段影响“不适用”，属于轻微表达遗漏 | 通过 |
| 测试 14 | 正确路由 `quant-run`，把结果判读交给 `direct-quant-eval`；未伪造 PPL；token 与复杂度匹配 | eval workflow 当前固定使用 Wikitext-2，不应把 Wikitext-2/Pileval描述为可选 eval 数据源；`--data_dir` 是 PTQ 数据目录，不是 eval 数据集参数；两条建议命令遗漏已列为待确认项的 `--output_dir`；纯外部执行任务不需要复用 repo-map | 部分通过：路由和 token 正确，事实与命令计划需修正 |

整体上，五例主职责路由均正确，说明新版 `amct-coder` 的动作优先路由和职责边界已经基本稳定。当前主要问题已从“选错 skill 或落错层”转为两类：一是满足停止条件后仍继续读取和展开，二是外部 skill 自身固化了不准确或混合阶段的事实。

### 11.3 后续收敛项

1. `ops-boundary` 对“共享契约未冻结”的混合任务增加更强的早停规则：只输出职责拆分、契约清单、候选锚点和阶段依赖，不读取 kernel 模板全目录，不展开具体 tiling/UB/stream 实现。
2. `deploy-export-code` 的只读 metadata review 固定最小产物矩阵：单/多 shard、旧字段兼容、原 metadata 保留、`total_size`、静态 JSON 可解析及 index 文件引用有效。
3. `dtype-quant-module` 先做“现有 dtype 是否已覆盖”分流；命中复用时给零改动方案，只有新契约成立后才扩展 deploy、algorithm、adapter 或 ops 链路。
4. 单独修正 `quant-run`：明确 eval 当前固定使用 Wikitext-2，Pileval 只用于 PTQ 数据提取；区分真正必填参数、建议显式指定但有默认值的参数，以及当前不存在的数据集选择参数；评测模板补齐 `--output_dir`。
5. 下一轮验收除路由、事实、计划、过程、token 五项外，增加“达到停止条件后的额外读取”计数。无新证据需求却继续读取完整目录或完整测试文件，直接判过程不通过。

本轮只记录回归分析，没有修改运行时代码、skill 或 repo-map；上述收敛项留待后续独立实施与复测。
