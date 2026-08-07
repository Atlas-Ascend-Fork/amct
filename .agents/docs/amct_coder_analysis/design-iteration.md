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

## 12. 有无 `amct-coder` 的复杂任务对照评估

在第二轮回归之后，进一步对以下两组记录做成对比较：

- 有 `amct-coder`：`complex-test-skill.md`。
- 无 `amct-coder`：`complext-test-noskill.md`。

本实验的唯一目标变量是 `amct-coder` 是否存在。无 skill 组并非完全没有 skill，`amct-ops-dev`、`deploy-export`、`model-adapter`、`quant-workflow`、`quant-run` 等专项 skill 仍然可用，只移除了 `amct-coder`。因此，本节评估的是 `amct-coder` 在现有专项 skill 体系之上的增量价值，而不是“有全部 skill”和“完全无 skill”的差异。

实际执行有 skill 组时，投喂文本没有包含“请使用 amct-coder”这类显式触发语。当前 `complex-test-skill.md` 保存的 Prompt 带有该表述，与真实投喂文本不完全一致；本次归因以实际执行条件为准，不把结果解释为显式名称诱导。这个差异也说明后续测试记录必须保存真实执行 Prompt，不能用事后整理文本替代。

### 12.1 评分口径

对每个用例按相同的 100 分 rubric 人工评分：

| 维度 | 分值 | 判断重点 |
| --- | ---: | --- |
| 职责路由与分层边界 | 25 | 是否选择正确主 skill/子能力，是否拒绝错误落点和越层方案 |
| 源码事实与调用链准确性 | 25 | 是否区分源码存在、活跃调用链、注册入口和实际编排点 |
| 待确认契约与事实门禁 | 20 | 是否避免在 schema、dtype、layout、来源等未冻结时臆造确定方案 |
| 条件式计划与验证闭环 | 20 | 是否覆盖依赖顺序、注册/公开面、测试和证据等级 |
| 上下文与 token 效率 | 10 | 是否达到停止条件后及时停止，避免无关资料和实现细节扩张 |

该评分用于同一组复杂用例的内部横向比较，不是客观精度指标。token 数来自测试文件中的近似记录，按万 token 汇总，适合判断数量级和相对变化，不适合解释小幅差异。

### 12.2 逐例结果

| 用例 | 核心能力 | 有 skill | 无 skill | 质量差值 | Token 对比 |
| --- | --- | ---: | ---: | ---: | ---: |
| A | NPU 算子与上层 quantizer 混合职责拆分 | 94 | 80 | +14 | 65k vs 70k，下降约 7% |
| B | deploy metadata/config 契约与活跃链 | 95 | 84 | +11 | 20k vs 30k，下降约 33% |
| C | 新 dtype 的数值、注册、CLI、deploy 闭环 | 95 | 85 | +10 | 25k vs 25k，基本持平 |
| D | classic graph 与 LLM new-path 注册体系隔离 | 97 | 79 | +18 | 10k vs 30k，下降约 67% |
| E | eval 执行、方案选择与结果判读编排 | 82 | 91 | -9 | 6k vs 20k，下降约 70% |
| **平均/合计** |  | **92.6** | **83.8** | **+8.8** | **126k vs 175k，下降约 28%** |

按这套 rubric，五例中四例质量提升，一例退化。若把 `100 - 得分` 作为问题量，有 skill 组从无 skill 组的 16.2 降至 7.4，问题量约减少 54%。用平均质量除以总 token 构造仅供内部比较的相对效率指数，有 skill 组约提升 54%。后两个比例依赖人工评分上限，只能用于说明“提升质量的同时减少了上下文消耗”，不能作为稳定的统计精度结论。

### 12.3 有效性来源

用例 A～D 是 `amct-coder` 的主要成功区，改进不是简单增加信息量，而是提高判断纪律：

- 用例 A：有 skill 时明确拆成 `amct-ops-dev` 和 `ops-boundary` 两个职责域，并把 `int_impl.py` 等消费点保持为待确认候选；无 skill 组虽然技术细节丰富，但更早锁定 kernel、tiling 和 Python 接入文件。主要收益是防止未冻结契约演变为伪确定计划。
- 用例 B：有 skill 时正确保留在代码侧 `deploy-export-code`，并识别 `_refresh_config_tensor()`、`_convert_tensor()` 没有活跃运行时调用方；无 skill 组使用了偏交付执行的 `deploy-export`，最终计划可用，但职责识别和伪活跃路径门禁较弱。
- 用例 C：有 skill 时先检查现有 new-path `mxfp`、classic 和 experimental FP4 实现，要求新 dtype 说明与既有格式的差异；无 skill 组能覆盖主要接入面，但缺少同等级的防重复和跨代码体系隔离门禁。
- 用例 D：有 skill 时直接命中 BLOCKER，并区分 classic eager `AlgorithmRegistry`、graph_based 显式 pass 编排和 LLM `ALGO_REGISTRY` 三套机制；无 skill 组最终也否决了方案，但加载了不相关的 `model-adapter`，经历额外上下文扩张，并且对三套机制的表述不够精确。

这些结果说明 `amct-coder` 最有效的部分是“编码总入口”的核心职责：按动作选主能力、声明仓库边界、冻结契约、检查活跃调用链，以及拦截不同代码体系之间的错误复用。由于无 skill 组仍然拥有其它专项 skill，这些提升可以视为 `amct-coder` 自身提供的增量效果。

### 12.4 反例与当前缺口

用例 E 是本轮重要反例。无 skill 组形成了较完整的：

```text
quant-workflow
  -> scheme-recommendation
  -> quant-run
  -> direct-quant-eval
```

并结合 casebook 区分 Qwen3 Dense/MoE、已有 W4A8 先验和初始方案确认门禁。有 skill 组虽然正确识别 `quant-run` 与 `direct-quant-eval` 的职责，且 token 明显更低，但存在三点不足：

1. `quant_target`、`quant_dtype` 尚未确认时，没有强制先进入 `scheme-recommendation`，却在示例命令中提前采用 `int` 等候选值。
2. “运行两组 PPL 并比较”是执行与 reviewer 的组合动作，当前规则没有稳定形成多外部 skill 的顺序编排。
3. 纯外部量化流程路由后仍保留“amct-coder 统一编排”的残余职责，没有完全退出不属于编码 skill 的流程。

因此，`amct-coder` 当前不能解释为 AMCT 所有复杂请求的统一生命周期编排器。它对编码和 review 任务有效，但对方案推荐、实际跑数和结果判读等纯外部职责，应更快、更完整地切换到 `quant-workflow` 或对应叶子 skill。

### 12.5 综合结论

本轮将 `amct-coder` 的综合有效程度评为 **8/10：明确有效，但外部流程编排仍需修正**。

可复核的观测结论是：

- 复杂编码规划平均质量得分提高约 8.8 分。
- 五个用例中四个提升，核心编码/review 用例 A～D 均提升。
- 总 token 从约 175k 降到约 126k，下降约 28%。
- 最大收益出现在 classic/new-path 越层识别，最大缺口出现在量化执行与判读的组合编排。
- 在没有显式点名 `amct-coder` 的条件下仍产生上述差异，说明自然语言触发与内部主子能力路由基本有效。

考虑样本只有五例、评分为人工 rubric、token 为近似值，且部分记录可能存在 warm session 资料复用，不能声称稳定提升恰好为 8.8 分。当前更稳妥的效果区间是：复杂编码规划质量提升约 **7～10 分**，token 降低约 **20%～30%**。

### 12.6 下一轮修改与实验要求

1. 为“运行 + 比较”“执行 + 判读”等组合动作增加明确路由模板，强制拆分 implementer 与 reviewer。
2. `quant_target`、`quant_dtype`、算法等方案输入缺失时，先进入 `scheme-recommendation`；不得在命令模板中把候选值写成已确认选择。
3. 命中 `quant-workflow`、`quant-run`、`direct-quant-eval` 等纯外部流程后，删除 amct-coder 的残余编排职责，只保留必要的跨职责契约汇总。
4. 为契约未冻结的 Ops 只读规划增加快速早停路径。只输出职责拆分、契约清单、候选锚点和阶段依赖，不读取 kernel 模板全目录，不展开 tiling、UB、stream 等实现细节。
5. 后续采用相同模型、相同工具环境、全新会话和完全一致的真实 Prompt；每个用例至少重复 3 次，盲评后再汇总均值和方差。
6. 增加 2x2 对照：skill 存在/不存在 × Prompt 显式点名/不点名，以分别测量 skill 内容收益和显式触发收益。
7. 测试记录必须保存真实投喂文本、cold/warm 状态、实际加载的 skill/reference 和增量 token；整理版 Prompt 与真实 Prompt 不一致时单独标注，不得混用。

## 13. 第六轮审视：从“操作说明书”收敛外部路由与知识边界

结合 `quant-workflow` 已上线后的实际分层方式，进一步明确：skill 的职责是提供稳定边界、操作流程、检索入口和停止条件，不是复制整个 AMCT 仓库知识。`amct-coder` 的内部子能力表应当是场景操作手册的目录，具体架构事实仍由当前源码、注册入口、活跃调用链和测试确认。

本轮不否定现有主体结构。事实门禁、repo-map 与源码事实分层、渐进披露、验证证据等级，以及按场景拆分 reference 的方向均应保留。当前主要问题集中在入口层承担了过多外部路由职责，需要做收敛而不是继续增加知识。

### 13.1 主要问题

1. **P0：frontmatter 承担了过多外部路由职责。**
   当前 description 除源码编码与 review 的正向职责外，还枚举 `model-adapter`、`amct-ops-dev`、`quant-run`、精度判读和 deploy 交付等外部能力。客户端主要依据 `name + description` 选择 skill，这会让 `amct-coder` 与更专门的 skill 竞争触发，也让新增外部 skill 反向依赖本 skill 更新。

2. **P0：外部路由存在多份事实源。**
   frontmatter、主 `SKILL.md` 的任务实质路由表和 `references/delegation.md` 的委派表重复维护同一组外部 skill。新增、删除或调整 skill 时容易漏改，且独立 skill 的职责本应由各自 description 作为机读接口表达。`amct-coder` 只应保留与源码编码直接相邻、确实存在歧义的职责交界。

3. **P1：BLOCKER 的继续执行语义不一致。**
   主文件前文规定“正确替代方案明确时继续执行”，全局 BLOCKER 又写成“定位正确层并继续询问用户”。后者可能导致无需新增契约或授权时仍中断任务。应统一为：只停止错误方案；替代路径明确且仍在用户目标内时继续，只有缺少会改变方案的关键输入时才询问用户。

4. **P1：内部子能力容易被理解为封闭架构分类。**
   当前要求每次先选择一个主子能力，但没有明确“现有分类未覆盖”的处理方式。仓库新增代码体系或出现跨域任务时，Agent 可能把任务强行套入最相近 playbook。应把该表定位为“场景操作手册索引”，并允许未命中时先查入口、注册和调用链，再决定是否扩展现有 playbook。

5. **P1：部分全局闭环规则表达过度泛化。**
   “新 dtype、算法或 public API 必须完成注册、选择入口、导出/回载和公开面闭环”容易被解释为三类改动都必须完成同一组动作。实际应分别描述各自的必要闭环，公共检查只保留调用链接通、兼容性和验证证据，避免为无关任务制造额外改动。

6. **P2：委派文档开始吸收易过期仓库知识。**
   `delegation.md` 枚举当前 casebook 已有模型族和个案。此类动态清单应通过目录检索获得，或由 casebook 索引维护，不应复制进委派操作说明书。否则模型新增后会形成又一份需要同步的知识副本。

### 13.2 保留边界

- 保留内部子能力及 reference；它们是不同编码场景下“怎么查、怎么改、怎么验证”的操作说明，不是独立 skill，也不是完整架构百科。
- 保留 repo-map 的稳定分层导航，但继续以源码和测试作为当前行为事实。
- 保留“确认事实 / 待确认契约 / 条件式方案”、活跃调用链核验和最小源码锚点预算。
- 保留修改后的最小验证与证据等级说明，不用单测替代集成、真实模型或硬件结论。

### 13.3 后续更新顺序

1. 收窄 frontmatter，只保留 AMCT 源码编码、调试、review、落点和越层判断的正向触发条件。
2. 删除主文件中的完整外部 skill 清单，改成“优先使用当前已发现且职责更专门的 skill”；`delegation.md` 只保留高歧义的稳定交界。
3. 统一 BLOCKER 的继续执行和询问条件。
4. 将“主子能力”表改为场景操作手册索引，增加未分类任务的源码探查与扩展路径。
5. 按 dtype、算法、public API 分拆闭环规则，删除无关的统一强制项。
6. 删除 casebook 模型枚举，改为按索引或目录动态检索。
7. 用全新会话回归四类场景：amct-coder 直接编码、外部 skill 直接触发、编码与外部流程混合、现有子能力未覆盖的新体系；同时验证 Claude Code 与 OpenCode 的自然语言触发。

本轮结论与 `quant-workflow` 的已上线实践一致：入口 skill 只保存自身编排或编码职责所需的稳定规则，具体操作下沉到叶子 skill/reference，模型经验放 casebook，当前实现事实现场读取。后续不以增加 AMCT 全量知识为目标，而以减少重复、降低触发歧义和提高未知场景下的事实核验能力为目标。
