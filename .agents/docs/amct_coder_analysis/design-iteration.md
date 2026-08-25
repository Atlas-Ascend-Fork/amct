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

7. **P1：跨体系阶段词仍可能触发错误子能力。**
   `classic-graph` 的触发条件单独列出“校准”，但 LLM new-path 同样包含校准样本生成、离线输入保存加载、算法统计和量化器 observe 等动作。若 Agent 把子能力覆盖范围当成关键词分类，就可能把 LLM PTQ 校准误路由到 classic。应明确“校准、评测、部署”等只描述阶段或业务目的，不能单独决定代码职责；先区分 classic 与 LLM new-path，再按实际修改的数据流、算法、通用量化器或模型契约选择操作手册。`classic-graph` 的触发描述也应收紧为 classic 调用链内的校准。

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
7. 收紧 `classic-graph` 的校准触发条件，并在场景操作手册索引中增加跨体系阶段词的判定规则。
8. 用全新会话回归四类场景：amct-coder 直接编码、外部 skill 直接触发、编码与外部流程混合、现有子能力未覆盖的新体系；同时验证 Claude Code 与 OpenCode 的自然语言触发。

本轮结论与 `quant-workflow` 的已上线实践一致：入口 skill 只保存自身编排或编码职责所需的稳定规则，具体操作下沉到叶子 skill/reference，模型经验放 casebook，当前实现事实现场读取。后续不以增加 AMCT 全量知识为目标，而以减少重复、降低触发歧义和提高未知场景下的事实核验能力为目标。

### 13.4 本轮实施结果

第六轮后续更新中的规则修改已经完成：

| 修改项 | 实施结果 |
| --- | --- |
| 收窄 frontmatter | description 只保留源码编码、调试、review、落点、调用链和契约检查等正向触发条件，不再枚举外部 skill 路由 |
| 去重外部路由 | 主文件删除完整外部 skill 表，改为根据当前已发现 skill 的 `name + description` 判断；`delegation.md` 只保留模型层、NPU 算子、eval/extract/PTQ 和 deploy 的高歧义交界 |
| 统一 BLOCKER | 明确只停止错误方案；替代路径明确且无需新增授权时继续，只有关键契约缺失且会改变方案时才询问用户 |
| 开放场景分类 | “主子能力”表改为“场景操作手册索引”，明确它不是封闭架构分类；未命中时先查入口、注册、调用方和测试，不强行套用相近手册 |
| 拆分闭环规则 | dtype、算法和 public API 分别定义必要闭环，导出/回载、packaging/docs 等只在实际受影响时检查 |
| 删除动态知识副本 | `delegation.md` 删除 casebook 模型族和个案枚举，改为读取 casebook 索引或动态检索目录 |
| 修正校准歧义 | `classic-graph` 收紧为 classic 调用链内的校准；主索引和 `llm-data-workflow` 明确先区分 classic/LLM，再按数据流、算法统计或通用量化器契约路由 |

本轮同时把 `validation.md` 的“命中子能力”更新为“命中场景操作手册或说明未分类探查结果”，避免验证规则继续使用封闭分类口径。

验证结果：

- `git diff --check` 通过。
- `quick_validate.py .agents/skills/amct-coder` 输出 `Skill is valid!`。
- 新增 casebook 索引相对链接已确认存在。
- 未运行仓库源码单测；本轮只修改 skill 和设计 Markdown，不涉及运行时代码。
- 未执行第 8 项跨 Claude Code/OpenCode 的全新会话 forward-test。当前会话已经读取问题诊断和预期修改，不能作为无污染回归证据；应在隔离工作区和全新会话中另行执行并记录。

## 14. 第七轮审视：路由歧义与用户澄清门禁

在“场景操作手册不是封闭分类”之后，还需要回答一个实际问题：Agent 已检查当前可用 skill 和仓库入口，但仍无法确定任务落到哪个操作手册时，是否应通过与用户交互完成路由。

结论是：**应允许澄清，但它只能是事实查证后的最后一道保险，不能成为 Agent 缺少把握时的默认动作。** 用户负责确认业务目标和未冻结契约，Agent 仍负责理解仓库、判断边界和选择内部操作手册。不能把内部路由责任转嫁给用户。

### 14.1 路由歧义处理流程

```text
理解用户动作
  -> 检查显式文件、符号、命令和调用入口
  -> 对照当前已发现 skill 的职责
  -> 查 2～4 个直接源码锚点
  -> 能判断：直接路由
  -> 仍有多个候选：
       不影响落点或契约：声明假设并继续
       会改变落点或契约：向用户问一个澄清问题
```

不要仅依赖模型自报的数值置信度决定是否询问。LLM 的置信度未必校准良好，应使用可核验的结构条件判断歧义是否具有实质影响。

### 14.2 必须询问的条件

只有源码和现有上下文无法消除歧义，并且不同答案会导致以下任一变化时，才询问用户：

- 进入不同代码体系或由不同职责主体承接。
- 改变数值语义、输入输出、数据格式或公共 API。
- 决定任务是修改实现、运行实验、结果判读还是生成交付产物。
- 决定是否执行高成本、不可逆或具有外部影响的操作。
- 缺失信息本身属于尚未确定的产品需求，而不是可以从仓库查出的实现事实。

### 14.3 不应询问的条件

以下情况由 Agent 继续查证或作出可逆假设，不询问用户：

- 调用入口、注册关系、活跃调用方或测试能够给出答案。
- 用户已经给出明确文件、函数、命令或报错位置。
- 多个候选最终使用相同的修改层、契约和验证方式。
- 只是 Agent 尚不熟悉代码，需要继续读取最小源码锚点。
- 可以声明假设后进行低风险、可逆的局部工作，并在证据不符时重新路由。

纯只读规划中，若用户也无法确认且仓库证据仍不足，可以给出按条件分支的方案；实际编码中，若不同分支会产生不兼容实现，则必须停在契约确认前。

### 14.4 澄清问题的设计

澄清问题必须面向用户可理解的业务事实或代码入口，不能要求用户选择内部 skill、子能力或 reference。例如，不应询问：

> 请选择 `classic-graph`、`llm-data-workflow` 或 `llm-algorithm`。

应改为：

> 这次要改的是 classic `quantize()/graph_based` 流程，还是 LLM 的 `extract_ptq_data/PTQ` 流程？如果你不确定，请提供目标文件或入口命令，我继续从调用链定位。

若已确认属于 LLM，还可以继续询问真正缺失的契约：

> 你要改变的是校准数据如何生成和保存，还是算法如何根据校准数据计算统计参数？

每轮优先只问一个能够最大幅度缩小方案空间的问题。问题应说明为什么需要该信息，以及不同答案会改变什么；不要一次抛出完整需求问卷。

### 14.5 与现有规则的关系

本轮不引入新的路由体系，而是补全现有两条规则之间的决策过程：

1. 未命中现有场景操作手册时，先查公共入口、注册点、调用方和直接测试。
2. 只有缺少会改变实现方案的关键输入、契约或授权时才询问用户。

交互澄清位于两者之间：先做工具和源码查证，再判断剩余歧义是否具有实质影响。它不替代事实门禁、repo-map、源码锚点或操作手册，也不改变“用户不负责内部路由”的原则。

### 14.6 后续修改与验证建议

1. 在主 `SKILL.md` 的事实门禁后增加“路由歧义处理”，固化“先查证、再判断影响、最后询问”的顺序。
2. 只在主入口维护交互策略；各场景 reference 只描述本领域的关键判别信号，不重复交互流程。
3. 增加三类 forward-test：源码可消除歧义时不询问；低风险歧义时声明假设继续；职责或契约发生变化时只问一个有效问题。
4. 评测同时记录不必要询问率、必要询问召回率、平均澄清轮数，以及用户回答后是否一次路由正确。
5. 加入“用户不知道答案”的测试，验证 Agent 会请求目标文件/入口命令、继续只读定位或输出条件式方案，而不是反复要求用户选择内部能力。

该机制设计得当会提高边界判断的可靠性；若简单实现为“Agent 不确定就问”，则会增加交互成本、降低自主性，并掩盖检索和源码理解能力不足的问题。

## 15. 待重构子能力初次 forward-test 评估

本轮使用用例 F～O 初次覆盖尚未定向重构的五个场景操作手册：`llm-algorithm`、`llm-data-workflow`、`llm-model-unit`、`ptq-solver-params` 和 `public-api-tests`。本节只记录初次测评暴露的问题、需要进入各子能力的修改方向和复测要求，不与第 12 节 A～E 的有无 skill 对照实验混算。

### 15.1 样本有效性

当前记录可以作为发现规则缺口的辅助证据，但不能形成正式 cold forward-test 通过率或 token 效率结论：

1. 用例 F 的 Agent 明确说明全局搜索命中了 `amct-coder-reference-forward-test-prompts.md`。该文件属于执行隔离要求中的禁区，因此 F 应标记为 `CONTAMINATED`，不计入通过率和 token 效率。
2. F、G、H 后附的会话摘要同时把三项任务列为同一 Goal，说明三例至少存在连续会话复用。G、H 只能作为 warm regression，不能替代独立 cold 结果。
3. 用例 I 的 NovaMoE 输出被记录在用例 J 的 Prompt 之后；J 的 kwargs 数据格式扩展没有对应结果。I/J 的结果映射需要修正，J 必须重跑。
4. K～O 的记录没有保存可复核的物理隔离工作区信息，不能仅根据 Prompt 中的禁读声明推断为有效 cold 样本。
5. token 记录混用了 `5k`、`2w`、`5w`、Thought 时间和可能的会话累计值，缺少统一的开始值、结束值和本轮增量，无法评分“上下文与 token 效率”。
6. prompt 文件声明“只保存无答案投喂语”，实际同时保存了工具轨迹、答案和会话摘要。后续必须把无答案 Prompt 与结果/评分材料分开，且让被测 Agent 在物理上无法访问后者。

因此，本轮不输出 100 分制正式均值。以下“强、可用、需修”只评价答案内容，用于确定 reference 的修改优先级，不代表测试通过状态。

### 15.2 逐例内容结论

| 用例 | 覆盖子能力 | 内容初评 | 暴露的问题 |
| --- | --- | --- | --- |
| F | `llm-algorithm` | 可用，但样本污染 | 注册、训练、保存、回载和 deploy 链核对较完整；但在 offset shape、约束、参数格式和 payload 均未冻结时，仍计划先注册带空 `trainable_params()` / 导出 hook 的可选算法，违反事实门禁，也可能让 `--algos` 暴露不可用实现 |
| G | `llm-algorithm` | 强，warm | 正确否决 solver 按算法名实例化、workflow 保存整层 `state_dict`、deploy 按算法名恢复三处越层方案；说明现有 registry、target、参数 hook 和 deploy hook 边界基本清晰 |
| H | `llm-data-workflow` | 强，warm | 正确核对 Qwen3 当前保存/加载前缀静态一致，并要求先比对磁盘产物与两次命令；但“源码静态一致”只能排除当前代码中的直接命名错位，不能写成真实 extract/PTQ 已经正确 |
| I | `llm-model-unit` | 可用，记录错位 | 正确识别新增模型和 wrapper 应转 `model-adapter`，也否决通用层模型分支；但没有可复核证据表明实际加载并遵循了外部 skill，完成标准遗漏 BF16 baseline、最小 PTQ smoke 和文档写回判断 |
| J | `llm-data-workflow` | 缺失 | 尚未验证 kwargs 字段、格式版本、按层/全局存储和 tensor-only 兼容策略，不能据此判断该子能力已覆盖离线数据格式演进 |
| K | `llm-data-workflow` | 强 | 正确区分 LLM 数据流与其它“校准”体系，拒绝 `model_name` 分支和缺输入 silent skip；对 DeepSeek mask/position 的结论仍应限制为已核对的静态捕获链，而不是运行时正确性证明 |
| L | `ptq-solver-params` | 需修 | 正确发现 `global` 未接主链和默认 `model` 死路径；但“逐 unit 独立训练、同层共享 optimizer、最小侵入只动 solver”互相矛盾，尚未冻结 solver 持有的 module、multi-unit batch/GT、finalize 责任和 workflow 调度契约 |
| M | `ptq-solver-params` | 强 | 正确定位训练自动生成 param dir、eval/deploy 未传目录时静默直转的高概率断点，并否决整模型 `state_dict`；但题目未指定具体算法，不应把诊断值固定为 AutoRound 的 `value/min_scale/max_scale` |
| N | `public-api-tests` | 需修 | 正确拒绝顶层 eager import graph_based；但误称现有 `test_non_graph_interface_*` 会检查所有顶层 `__all__` 名，实际只覆盖固定的 `NON_GRAPH_INTERFACES`；建议把懒加载名加入 `__all__` 时，也没有完整处理 `import *` 会触发 graph 依赖的兼容性变化 |
| O | `public-api-tests` | 需修 | entrypoint 和 packaging 锚点核对有效；但目标 helper 未指定时直接锁定 `BitPolicy.summary()`、`--bit_config`、默认 16 和 YAML 输出，违反事实门禁，后续条件式计划建立在未经确认的候选 helper 上 |

### 15.3 子能力修改计划

| 优先级 | 子能力 | 需要固化的修改 | 对应复测目标 |
| --- | --- | --- | --- |
| P0 | `llm-model-unit` | 强化“实际委派”门禁：确认结构实现属于外部职责后，必须加载并遵循 `model-adapter`，不能只写“转 model-adapter”；完成标准从外部 skill 当前规则取得，不在本 reference 复制不完整清单；review 汇总至少核对 BF16 baseline 或阻塞原因、关量化等价、最小 PTQ smoke、参数回载和文档写回判断 | I 能同时完成越层 review、真实 skill 切换和完整完成标准汇总，不替虚构模型写 wrapper |
| P0 | `ptq-solver-params` | 为新 granularity 增加语义冻结表：unit 边界、solver 持有的 module、batch/GT 组织、optimizer/scheduler 生命周期、finalize/文件写入责任、断点续跑和跨模式兼容；这些契约未冻结前，不得先确定 `LayerSolver` 构造签名或宣称“只动 solver”；参数回载诊断先读取实际算法的导出 key，不默认套 AutoRound 字段 | L 不再产生“独立训练却共享 optimizer”的矛盾方案；M 保持目录断点判断，但算法状态检查使用通用步骤和条件分支 |
| P0 | `public-api-tests` | 明确区分普通 `import amct_pytorch`、属性访问触发 `__getattr__`、`from amct_pytorch import *`、源模块 `__all__` 和顶层 `__all__` 五种契约；测试结论必须说明断言遍历的是固定集合还是完整 `__all__`；用户未给 helper 符号时，只能列候选和查证方法，不能据候选发明 CLI、输出和退出码契约 | N 准确说明懒加载名进入 `__all__` 的 wildcard import 代价并设计独立测试；O 在 helper 未确认时停在契约清单和候选锚点，不锁定 `BitPolicy` |
| P1 | `llm-algorithm` | 增加“不可注册空实现”规则：算法核心数值语义、trainable 状态和保存/回载格式未冻结时，只输出契约、候选锚点和阶段依赖；不得先把 no-op 类导入 `register_algorithms()` 并暴露给 `--algos`。只有最小可用语义与失败行为明确后才进入注册闭环 | F 在未冻结 offset 契约时早停，不再提出带空 hook 的阶段 0；G 的零 solver/workflow/deploy 分支结论保持不回退 |
| P2 | `llm-data-workflow` | 增加证据等级措辞：保存端与加载端静态命名一致，只能证明当前源码契约对齐，真实目录、命令参数和运行产物仍需单独核验；缺文件保持显式失败。离线 kwargs 格式扩展是否还需新增版本/兼容规则，等待 J 的有效 cold 结果后再定 | H/K 不把静态链路写成真实运行验证；J 覆盖字段集合、必填性、shape、层级、版本和旧 tensor-only 目录的兼容矩阵 |

`llm-data-workflow` 当前不做大幅重写。H、K 已证明其模型分支和 silent skip 门禁有效，现阶段只补证据等级；J 缺失前，不根据 Prompt 预设答案反向修改 reference。

### 15.4 实施顺序与验收

建议按以下顺序进入下一轮修改：

1. 先修改 `llm-model-unit` 的实际委派与完成标准来源，避免 amct-coder 继续复制外部 skill 的过期规则。
2. 再修改 `ptq-solver-params` 的 granularity 语义冻结表和算法无关的回载诊断步骤。
3. 修改 `public-api-tests` 的五类导入/导出契约与未知 helper 门禁。
4. 为 `llm-algorithm` 增加禁止注册 no-op 算法的早停规则。
5. 为 `llm-data-workflow` 补静态/运行证据分级；完成 J 的独立 cold 测试后，再决定是否增加数据格式版本化规则。

修改后的复测至少满足：

- 每个用例使用全新会话和物理隔离工作区；工作区不包含分析目录、prompt 文件、旧答案和 Git 历史替代入口。
- prompt-only 文件不写答案；结果保存到被测 Agent 不可见的位置。
- 每例记录真实 Prompt 哈希、代码版本、客户端/模型、实际加载的 skill/reference、cold/warm 状态、开始 token、结束 token 和增量 token。
- F～O 逐例重跑；F、G、H 不复用同一会话，I/J 分开记录，J 必须有独立输出。
- 正式评分仍使用第 12.1 节 rubric，但把样本有效性作为评分前置门禁；`CONTAMINATED`、warm 或结果缺失样本不进入均值。

当前 staged 的主 `SKILL.md` / `validation.md` 项目规范门禁修改不属于 F～O 的覆盖目标。本轮结果不能证明 Ruff、版权头、pre-commit 或 Issue/PR 路由规则有效，后续需要单独增加对应 forward-test，不能借用本轮五个 reference 的结果完成验收。

### 15.5 初次测评结论

初次内容结果表明，五个子能力对分层边界、活跃调用链和错误落点的识别整体有效，G、H、K、M 已能给出较强的只读 review。主要缺口已经从“找不到主链”收敛到三类更具体的问题：

1. 外部职责只做口头委派，没有实际加载并继承完整完成标准。
2. 新 granularity 等跨契约设计在语义未冻结前过早确定类、构造签名和共享 optimizer 方案。
3. public API/CLI 规划没有区分多种 import 契约，并会把未确认 helper 候选写成确定实现。

下一轮不整体重写五份 reference，而按 P0～P2 定向修改并独立复测。只有隔离、结果映射和增量 token 记录同时合格后，才能判断这些修改是否提高正式 cold forward-test 质量与效率。

### 15.6 子能力修改实施结果

第 15.3 节列出的五项定向修改已经完成，改动只落在 `amct-coder/references/`，没有修改运行时代码、repo-map、frontmatter 或 UI metadata：

| 子能力 | 已实施修改 | 对应初测问题 |
| --- | --- | --- |
| `llm-model-unit` | 明确“转 `model-adapter`”必须实际加载并遵循外部 skill；本 reference 只保留跨层 review，不再用本地六项摘要替代外部完成标准；汇总时核对 BF16 baseline/阻塞、关量化等价、最小 PTQ smoke、参数回载和文档写回 | 修正 I 的口头委派和完成标准不完整 |
| `ptq-solver-params` | 新增 granularity 七项语义门禁，冻结 unit、module、batch/GT、optimizer、finalize、续跑和兼容关系后才能确定类与接口；新增算法无关的参数回载诊断顺序 | 修正 L 的共享 optimizer 自相矛盾方案和 M 的 AutoRound 字段预设 |
| `public-api-tests` | 新增普通 import、顶层属性、wildcard import、源模块 `__all__`、顶层 `__all__` 五类契约；明确固定测试集合不等于完整 `__all__`；新增未知 helper 身份和稳定 I/O 门禁 | 修正 N 的测试范围误判与 wildcard import 漏项，以及 O 的候选 helper 过早锁定 |
| `llm-algorithm` | 把核心数值/状态契约冻结前置到注册之前；新增“契约未冻结时只做只读规划”和“不得注册可被 `--algos` 选择的 no-op 半成品”规则 | 修正 F 的空 hook 阶段 0，同时保留 G 已验证的 registry/target/hook 边界 |
| `llm-data-workflow` | 新增源码契约、磁盘产物、命令口径、集成闭环四级证据；明确静态前缀一致不能证明真实 extract/PTQ 已正确 | 收敛 H/K 的证据表述；J 的数据格式版本化规则仍等待有效 cold 结果 |

本轮遵循 `skill-creator` 的渐进披露原则：详细场景规则继续保存在按需读取的 reference 中，没有把五类细节上移到主 `SKILL.md`。由于 name、description 和触发边界未变化，未生成或修改 `agents/openai.yaml`。

验证结果：

- `git diff --check` 通过。
- `quick_validate.py .agents/skills/amct-coder` 输出 `Skill is valid!`。
- 未运行 AMCT 源码单测；本轮只修改 skill/reference 与设计 Markdown，不涉及 Python/C++ 运行行为。
- 未执行正式 forward-test；当前工作区已包含初测答案和设计结论，不满足 cold 隔离要求。F～O 应按第 15.4 节在物理隔离的新会话中复测。
- repo-map 不需要更新；本轮没有改变仓库分层、注册集合或核心运行流程。

### 15.7 第二次复测评估与追加收敛

`amct-coder-reference-forward-test-prompts.md` 已追加第二次输出，但当前记录仍不能形成正式 cold 通过率：

1. Prompt、旧答案和第二次答案仍保存在同一文件，且没有物理隔离工作区、Prompt 哈希、代码版本或会话 ID。答案中的“隔离自查”只能作为声明，不能替代可复核的隔离证据。
2. F～K 有第二次输出；L 的第二次输出误贴在 O 之后，并在“L4 回载核验”处截断；M、N、O 没有完整第二次输出。
3. G 明确写入上一题的 `adaptive_round_scale`、`沿用上一轮 C1-C9` 和具体文件名，而 G 的题面只说“一个新算法”。这证明任务局部事实发生串扰，G 只能作为失败的 warm regression，不能计为独立 cold 样本。
4. token 记录仍混用 `92.4k`、`15k`、`20k`、`50k` 等未知口径；F 的数值尤其像累计上下文。没有开始值、结束值和增量值前，不评价 token 改善比例。

第二次输出的内容结论如下：

| 用例 | 内容结论 | 仍存在的问题 |
| --- | --- | --- |
| F | 早停规则生效：先冻结数值/状态契约，明确禁止注册 no-op，再进入注册和消费链 | 调用链展开到 deploy/dtype/store 多层，记录为 `92.4k`，读取与输出明显过宽；效率结论无有效统计口径 |
| G | 三处错误落点仍能准确否决 | 把 F 的算法身份和 C1～C9 带入当前未命名算法，违反事实门禁和连续任务复用规则 |
| H | 四级证据表述生效：结论严格停在“当前源码静态契约一致”，磁盘产物、命令口径和集成闭环继续待证 | 内容可用，但仍缺独立 cold 与增量 token 证据 |
| I | 能识别模型结构实现属于 `model-adapter`，并拒绝通用层模型分支 | 仍只写“切换到 model-adapter”，没有实际加载记录，却自行给出适配完成标准；P0 问题未修复 |
| J | 首次覆盖离线 kwargs 格式演进，能列出字段、shape/切批、必填性、作用域、版本和旧目录行为 | 在这些契约未冻结后，又把“任意 kwargs dict + 版本标记”写成后续确定实现，条件分支仍不彻底 |
| K | 正确区分 LLM 数据流，保持缺主输入 fail-fast，并把静态/产物/命令/集成证据分开 | 条件分支自行命名 `--skip-missing-units`；若要改变缺失策略，必须先冻结用户可见接口和 eval/deploy 结果语义 |
| L | 七项 granularity 门禁和 registry-first 死路径已识别，回载按 per-unit 格式可保持不感知训练粒度 | 仍从 `layer` 名称直接断言“本质是联合求解”并推荐方案 A；结果错位且截断，只能作为问题证据 |
| M/N/O | 无完整第二次输出 | 不能评价定向修改是否生效 |

基于以上结果，本轮只做五项追加收敛：

1. 主 `SKILL.md` 增加任务局部事实隔离：连续任务只复用稳定源码事实，不复用上一题对象的名称、状态、CLI、文件名、payload 或决策。
2. `llm-algorithm` 要求未命名算法保持占位表达，禁止把前序算法写成当前确定落点。
3. `llm-model-unit` 增加完成前硬检查：草稿出现“转 `model-adapter`”但无实际加载记录时，必须先切换 skill，禁止用本地归纳替代外部完成标准。
4. `llm-data-workflow` 新增离线格式演进门禁，覆盖字段所有权、基数、切批、必填性、容器/版本和旧目录行为；契约未冻结前不发明 schema、文件名或 CLI 参数。
5. `ptq-solver-params` 明确 `layer` 名称和“按层保存”不推出联合训练或共享 optimizer；同时把 CLI 闭环表述从不存在的既有 `choices` 修正为接受值、帮助和默认值的兼容性决策。

`public-api-tests` 本轮不追加修改，因为 N/O 没有第二次结果，不能用初次答案重复证明已有修订有效。下一轮应先修正结果文件结构，并独立重跑 I、G、L、M、N、O；J/K 可作为新规则的回归题，但答案必须存放在被测 Agent 物理不可见的位置。

本轮验证结果：

- 对本轮修改的 `SKILL.md`、四份 reference 和本设计记录执行 `git diff HEAD --check -- <files>`，通过。
- `quick_validate.py .agents/skills/amct-coder` 输出 `Skill is valid!`。
- 全工作区 `git diff --cached --check` 仍会报告 `amct-coder-reference-forward-test-prompts.md` 中测试工具轨迹的行尾空格；这些来自本轮输入的已暂存测试记录，未在本轮改写。
- 未运行 AMCT Python/C++ 单测；本轮只修改 skill 与分析 Markdown，不涉及运行时代码。
- 未把现有第二次输出当作修改后正式 forward-test；追加规则仍需在物理隔离的新会话中复测。

### 15.8 需求不明确时的处理形态收敛

本轮针对一个此前未成文的问题：用户需求不明确时，skill 应该用什么方式确定需求，以及是否由 Agent 直接与用户交互。

评审当前 skill 的结论是判据已有、形态缺失：

1. `SKILL.md` 全局 BLOCKER 与 `delegation.md`「继续执行与询问」是同一条规则的两处表述，且相互一致——只有关键契约缺失且不同答案会改变实现方案时才询问用户。判据方向正确。
2. 最终说明已要求列出「已确认事实与待确认项」，待确认项有汇报出口，不会静默丢失。
3. 但「条件式方案」在事实门禁、候选锚点、最终说明三处被提及，全文没有一处定义它的输出形态。Agent 知道要区分三类信息，不知道区分完如何交付。
4. 事实门禁各条只写禁止动作（不发明类名/CLI/schema、只能列候选锚点），没有接续动作，容易停在“待确认”即交付。
5. 缺少“先推进不依赖该答案的部分”，一个待确认契约可能挂住整个任务；原 BLOCKER 一节针对的是错误方案，不覆盖信息缺失。
6. 「不自行发明」若被从宽理解，变量名、日志措辞、默认值等不影响交付物的局部选择也会被推回用户。

#### 三档处理原则

用一句话说明这套规则解决什么：用户没说清算法叫什么、参数存成什么格式、CLI 怎么暴露时，Agent 交付物应该长什么样。此前 skill 只写了「什么时候该问用户」的判据，没写「不问时交付什么」，所以实测里往两头跑偏——一头是自己编类名和 CLI 参数当成既定事实（F、J、K、O），一头是列一串「待确认」就交差。

按顺序分三档，不许跳档：

1. **源码能回答的自己查。** 函数签名、注册集合、活跃调用方、测试断言属于可查事实，不推回用户。越层落点、错误复用和 BLOCKER 查到证据后直接否决，不需要用户确认。
2. **查不到但不影响交付物的自己定。** 变量名、日志措辞、默认值、等价实现取舍，定完在最终说明点明，不为此提问。
3. **查不到且不同答案导出不同的活，按交付物分流。** 只读规划输出条件分支；实际编码直接询问用户。

第 3 档按交付物拆分，与第 14.3 节保持一致：只读规划里条件分支本身就是交付物，写分支成本低；实际编码中不同分支产出不兼容实现时，按错误假设写出完整版本的返工成本高于提问，且假设容易被后续轮次当成既定事实继承。任何一档都先完成不依赖该答案的部分，不让单个待确认项挂住整个任务。

#### 本轮修改

只改 `SKILL.md` 与 `delegation.md`，未新增 reference：

| 位置 | 修改 | 对应问题 |
| --- | --- | --- |
| `SKILL.md` 事实门禁 | 开头新增消歧顺序：先查源码/测试/调用方，查到即判定或否决错误方案，查不到才进入处理形态 | 修正问题 4 的“只有刹车没有下一步” |
| `SKILL.md` 事实门禁 | “不自行发明类名/CLI/schema”一条接续消歧动作；新增局部选择自行决定的划界条目 | 修正问题 4、6 |
| `SKILL.md` 新增「契约缺失时的处理形态」 | 定义查不到时的交付形态：先做完不依赖该答案的部分；只读规划出条件分支，实际编码停下询问，不写投机实现。提问形态四条：封闭选项带推荐项与影响面、一次问完、单项不阻塞整个任务、无答复时只读规划按推荐项续做并标注未确认而编码停在契约确认前 | 修正问题 3、5 |
| `SKILL.md` 全局 BLOCKER | 补充“信息缺失不等于 BLOCKER”，并指向新小节 | 修正问题 5 |
| `delegation.md`「继续执行与询问」 | 同源规则引用新小节，要求询问前先确认该问题不能由源码、注册入口或测试消歧 | 防止两处同源规则各自演化 |

新小节放在主 `SKILL.md` 而非 reference：这些是入口级判断规则，需要每次生效，下沉到按需读取的 reference 会导致不被加载。最终说明未新增字段，条件分支的前提沿用既有「待确认项」出口。

本节初稿曾把“声明假设并按假设实现完整可用版本”作为编码与规划的统一默认。该写法拆掉了第 14.3 节对实际编码的刹车，并会放大实测中已出现的编造契约问题，已按上述交付物分流修正。

本轮验证结果：

- `quick_validate.py .agents/skills/amct-coder` 输出 `Skill is valid!`。
- `git diff HEAD --check -- .agents/skills/amct-coder/SKILL.md .agents/skills/amct-coder/references/delegation.md` 通过。
- 未运行 AMCT Python/C++ 单测；本轮只改 Markdown，不涉及运行时行为。
- 未执行 forward-test。当前工作区包含历次答案与设计结论，不满足 cold 隔离要求。待验证的是两类偏差是否消失：该自查却询问用户、该询问却硬猜契约；应复用 forward-test 中契约不完整的题目（J、K、L、O）在物理隔离新会话中复测。
- repo-map 不需要更新；未改变仓库分层、注册集合或核心运行流程。

### 16. 路由粒度、Reference 可发现性与项目规范前置

本轮根据评审意见重新检查主 `SKILL.md` 的职责。问题不在于现有场景枚举是否足够准确，而在于主入口承担了过多细粒度分类：表格试图用算法名、参数类型、数据对象和文件形态描述每个子能力，却仍不可能覆盖未来全部需求；具体场景同时已在各 reference 中维护，形成重复和双处更新成本。

评审同时指出两个相关问题：

1. 主表只列 9 个场景 playbook，`repo-scenarios.md`、`stage-impact.md`、`layer-rules.md`、`attention-rules.md`、`wrapper-rules.md`、`validation.md` 等辅助 reference 分散在后文，Agent 不能先看到完整资料地图。
2. AMCT 项目规范中的代码风格、pre-commit、版权和贡献要求被混入修改后才读取的 `validation.md` 或按需读取的贡献流程，约束时机晚于编码。

#### 16.1 设计裁决

主 `SKILL.md` 只承担流程编排和文档导航，不在路由层枚举完整触发场景：

- 场景 playbook 的用途压缩为宽泛能力说明，例如 `llm-algorithm.md` 只写“LLM 量化算法相关开发与 review”；算法名、`targets`、训练/统计参数等细节留在该 reference。
- 未命中现有场景时继续按源码入口、注册点、调用方和直接测试探查，不通过补充更多关键词追求穷举覆盖。
- 主入口建立一张完整 Reference 索引，列出 `references/` 下每份文档的用途和加载时机。完整列出只解决可发现性，不表示所有文档是平级子能力，也不表示一次性全部读取。
- Reference 按主场景、定位/边界辅助、模型规则、编码前规范、编码后验证和贡献操作区分；每个任务仍只加载当前需要的资料。

路由、索引和加载的职责分别是：路由提供粗粒度方向，索引保证资料可发现，reference 承载具体规则，源码与测试确认当前事实。

#### 16.2 AMCT 项目规范

新增独立的 `project-rules.md`。`amct-coder` 先判断职责归属；确认由自身承接且需要实际修改 AMCT 文件时，在形成具体修改方案或编辑文件前读取。已经路由到外部专门 skill 的任务遵循目标 skill 自己的前置流程。这里的“项目规范”特指以下 AMCT 仓规则，不以“适用的 `AGENTS.md`”代称：

- Python 遵循 PEP8，使用 Ruff 检查和格式化。
- C/C++ 遵循 Google 代码规范，使用 `.clang-format`，4 空格缩进、120 字符行宽。
- 提交前安装并运行 pre-commit。
- 新增代码保留仓库既有的 Apache 2.0 版权声明。
- 简单 bug 修复可直接提 PR；新功能或接口变更先通过 Issue 讨论。
- PR 关联对应 Issue，并包含代码以及受影响的优化文档和 README 更新。
- 创建 PR 使用 `.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md`；创建 Issue 时从 `.gitcode/ISSUE_TEMPLATE/` 按类型选择模板。
- 涉及 GitCode Issue 或 PR 时使用 `gitcode-issue`、`gitcode-pr` skill；缺失时按 `default-skills` 安装。

`project-rules.md` 负责编码前约束；`validation.md` 只负责修改后的语法、Ruff/pre-commit、测试、构建和证据等级。读取规范不等于任务开始时安装工具或创建 Issue/PR：安装 pre-commit 发生在准备提交且环境未就绪时，GitCode skill 只在实际进行对应操作时加载。

#### 16.3 修改范围

本轮按以下范围实施：

1. 收缩 `SKILL.md` 的场景描述，删除算法名、参数名和数据对象的细粒度枚举。
2. 在 `SKILL.md` 中列出全部 reference，并标注类型、用途和加载时机。
3. 新增 `project-rules.md`，把 AMCT 项目规范放到编码前固定流程。
4. 从 `validation.md` 删除规范定义，只保留对应的修改后检查动作；收缩 `contribution-flow.md` 为实际 Issue/PR 操作补充。
5. 更新相关引用，验证所有 reference 均可从主 `SKILL.md` 直接发现。

repo-map 不需要更新：本轮只调整 Agent 文档的导航和读取时机，没有改变 AMCT 源码分层、注册集合或核心运行流程。frontmatter 的触发范围不变，`agents/openai.yaml` 当前不存在，无需生成 UI metadata。

#### 16.4 实施结果与验证

本轮已按第 16.3 节完成修改：

| 文件 | 实施结果 |
| --- | --- |
| `SKILL.md` | 固定流程先判断职责归属，确认由 amct-coder 承接实际改动后再读 AMCT 项目规范；原细粒度场景表改为粗粒度用途导航；直接列出全部 reference；保留按需读取预算 |
| `references/project-rules.md` | 新增编码前规范，集中维护 Python/C/C++ 风格、pre-commit、版权、Issue/PR 门槛、模板和 GitCode skill 路由 |
| `references/validation.md` | 只保留修改后的语法、Ruff/pre-commit、格式化、测试、构建和证据检查，并反向引用编码前规范 |
| `references/contribution-flow.md` | 删除门槛和模板副本，只保留获得授权后的实际 Issue/PR 操作衔接 |
| `references/repo-scenarios.md` | 将失效的“子能力路由”章节引用更新为 `SKILL.md` 的 Reference 导航 |

实现保持以下边界：完整 Reference 索引只建立可发现性，不触发全量读取；主场景用途不复制算法名、参数、数据对象和源码锚点；具体触发和执行规则继续留在对应 reference。项目规范只在确认由 amct-coder 承接实际改动后前置读取；repo-map、场景手册和辅助 reference 仍按承接职责与风险加载。

验证结果：

- `git diff --check` 通过。
- `quick_validate.py .agents/skills/amct-coder` 输出 `Skill is valid!`。
- `SKILL.md` 直接链接的 reference 集合与 `references/*.md` 实际文件集合一致，共 18 份，无遗漏或多余项。
- `SKILL.md` 正文规模低于 `skill-creator` 建议的 500 行上限；不记录易随正常编辑漂移的精确行数。
- 仓库中 PR 模板和四类 Issue 模板路径均已核对存在。
- 未运行 AMCT Python/C++ 单测；本轮只修改 Agent Markdown，不涉及运行时代码。
- 未执行 forward-test；本轮先完成结构和静态有效性验证，后续需要用无设计答案泄露的新会话检验 Agent 是否做到“粗粒度路由、按需读取、编码前遵守项目规范”。
- repo-map 不需要更新；AMCT 源码分层和核心流程未变化。

## 17. 第八轮审视：多场景裁决、Review 规范分流与设计版本更新

第 16 节完成粗粒度 Reference 导航后，自审发现三个尚未闭环的问题。本轮只修入口级决策和设计文档事实源，不根据未经隔离的旧测试继续扩写领域 reference。

### 17.1 发现的问题

1. **多个主场景同时匹配时缺少裁决。** 主入口要求选择一个主手册，也处理了完全未命中的情况，但没有说明 `llm-algorithm`、`llm-data-workflow`、`public-api-tests` 等粗粒度用途同时可能匹配时如何选择。Agent 可能因题面同时出现算法、CLI、测试或文档而读取多个主手册，削弱渐进披露。
2. **“纯 review 不加载项目规范”范围过宽。** 架构落点 review 不需要代码风格和贡献门槛，但代码或 PR 的规范合规 review 需要检查 Ruff、版权、Issue/PR 门槛和模板。把所有 review 统一跳过会漏掉该类检查。
3. **`design-v1.md` 与运行时规则发生漂移。** v1 仍保留详细外部 Skill 路由表、细粒度子能力表和旧加载顺序，并声明冲突时优先按设计文档修改 `SKILL.md`。这会让后续维护者用旧设计覆盖已经迭代的运行规则。设计记录中维护精确行数也会产生无价值漂移。

### 17.2 设计裁决

#### 多主场景裁决

保持主入口的粗粒度导航，不恢复算法名、参数名或文件枚举。多个主场景同时可能命中时，按以下顺序裁决：

1. 先看用户动作、目标入口、被修改契约和活跃调用方，确定真正拥有修改语义的主手册。
2. 仍无法定位代码体系时读取 `repo-scenarios.md`，不通过预读所有候选手册解决歧义。
3. 只有源码已经证明同时改变另一契约时，才把第二个主场景作为辅助加载。
4. 主手册选择仍依赖未冻结的关键契约时，进入既有「契约缺失时的处理形态」，不增加新的询问体系。

#### Review 规范分流

`project-rules.md` 的加载由 review 目标决定，而不是由 `review` 这个动作词决定：

| 任务形态 | 是否读取项目规范 | 原因 |
| --- | --- | --- |
| 实际修改 AMCT 文件 | 是 | 代码风格、版权和贡献门槛必须在修改前建立约束 |
| 代码或 PR 的规范合规 review | 是 | 需要检查 Ruff、版权、Issue/PR 门槛和模板 |
| 架构落点、分层或越层 review | 否 | 由 repo-map、场景手册和源码事实提供判断依据 |
| 纯只读规划、普通问题分析和故障诊断 | 否 | 不产生需要项目规范约束的修改或合规结论 |

#### 设计文档事实源

`design-v1.md` 保留 v1 原始设计，作为设计演进基线；在文件末尾按 v1 的结构追加当前设计，不重写原始正文。文档优先级调整为：

1. `SKILL.md` 与 `references/*.md` 是 Agent 实际执行的运行时事实源。
2. `design-v1.md` 文末最新的“当前设计”记录结构设计快照；与原始 v1 冲突时，文末更新优先。
3. `design-iteration.md` 记录问题、裁决、修改过程和验证证据，不承担运行时规则。
4. 发现设计记录与运行时文件冲突时，以运行时文件确认当前行为，再同步设计记录；不能用旧设计静默回退已验证的规则。

### 17.3 实施范围

| 文件 | 修改内容 |
| --- | --- |
| `SKILL.md` | 增加多主场景裁决；细分项目规范对实际修改、规范合规 review 和架构 review 的加载条件 |
| `references/project-rules.md` | 同步 Review 加载边界，避免与主入口出现第二种口径 |
| `design-v1.md` | 补充版本与事实源说明，并在原 v1 末尾按相同结构记录当前设计 |
| `design-iteration.md` | 记录本轮问题、设计裁决、实施范围和验证结果；删除易漂移的精确行数 |

本轮不修改九份主场景 reference，不改变 frontmatter、repo-map、casebook 或 AMCT 运行时代码。Reference 的领域规则只有在后续隔离 forward-test 出现可复现失败时再调整。

### 17.4 实施结果与验证

第 17.3 节范围已经完成，验证结果如下：

- `git diff --check` 与 `git diff HEAD --check` 均通过，工作区新增差异和相对 HEAD 的组合差异不存在空白错误。
- `quick_validate.py .agents/skills/amct-coder` 输出 `Skill is valid!`。
- 主入口直接链接的 Reference 集合与 `references/*.md` 实际文件集合一致，共 18 份。
- `SKILL.md` 与 `project-rules.md` 均使用“实际修改或规范合规 review 时读取；架构落点 review、纯规划、普通分析和诊断不读取”的同一口径。
- `design-v1.md` 保留原 v1 正文，并在末尾按“设计目标、文档分工、主流程、阶段数据流、冲突裁决、目标形态、维护清单”记录当前设计；文件开头已明确版本和运行时事实源优先级。
- `SKILL.md` 正文仍低于 `skill-creator` 建议的 500 行上限。
- 不运行 AMCT Python/C++ 单测；本轮只修改 Agent Markdown，不涉及运行时代码。
- 不把本轮静态检查视为 forward-test；多场景裁决和 Review 分流仍需在物理隔离的新会话中验证行为。

## 18. 第九轮审视：按交付物路径统一加载与汇总

第 17 节按 review 类型修正项目规范加载后，再次审查主 `SKILL.md` 正文发现，问题已经从单条规则错误转为多处规则各自维护：固定流程、职责判断、Reference 导航、渐进披露、规范 Reference、验证 Reference 和最终说明都在描述任务类型，但没有共同的执行路径事实源。

### 18.1 发现的问题

1. **加载口径分散。** `project-rules.md`、repo-map、主场景手册和 `validation.md` 的读取条件散落在多个章节；增加实现规划或细分 review 后，需要同时修改多处，容易再次漂移。
2. **实现规划没有成为一等交付物。** 正文实际承接只读实现规划，但 frontmatter、固定流程和最终说明仍主要围绕编码与 review，Agent 可能遗漏规划所需的 repo-map、主手册和验证方案证据。
3. **局部选择边界过宽。** “默认值”与变量名、日志措辞并列为 Agent 可自行决定的局部选择，但默认值经常影响 CLI、兼容性和用户可见行为，不能天然视为等价实现。
4. **提问和最终汇总会制造无依据内容。** 询问规则要求每次都给推荐项，即使源码、兼容性和成本没有提供偏向；统一的最终说明又要求所有任务报告主场景、改动层和修改后验证，纯规范合规 review 或外部 Skill 路径只能虚构这些字段。

### 18.2 设计裁决

主 `SKILL.md` 增加唯一的“按交付物选择执行路径”矩阵：

| 执行路径 | `project-rules.md` | repo-map + 主场景手册 | `validation.md` |
| --- | --- | --- | --- |
| 实际编码 | 修改前读取 | 读取 | 修改后读取并执行匹配的最小验证 |
| 实现规划、行为诊断、架构或代码行为 review | 不读取 | 读取 | 判断既有或建议的验证证据时按需读取相关部分 |
| 纯规范合规 review | 读取 | 默认不读；发现行为、分层或调用链问题后切换路径 | 检查验证完整性时按需读取相关部分 |
| 已路由到外部专门 Skill | 由目标 Skill 决定 | amct-coder 不预读 | 由目标 Skill 决定 |

这张表是加载时机的运行时唯一事实源。固定流程只要求选择路径；Reference 导航只说明资料用途并引用路径；渐进披露只规定预算和扩张顺序；`project-rules.md` 与 `validation.md` 不再维护各自的任务类型清单。

同时作出三项行为收敛：

1. frontmatter 和定位明确包含“实现规划”。
2. 从局部等价选择中删除默认值；需要用户决策时，只有源码惯例、兼容性或实现成本存在证据才给推荐项，没有依据就明确无默认推荐。
3. 最终汇总按四条执行路径区分。纯规范合规 review 不虚构主场景和 repo-map 结论，外部 Skill 路径不声称加载过 amct-coder 手册，无修改的规划、诊断和 review 不声称执行过修改后验证。

### 18.3 实施范围

| 文件 | 修改内容 |
| --- | --- |
| `SKILL.md` | 更新 frontmatter、定位和固定流程；新增执行路径矩阵；收敛边界读取、Reference、渐进披露、提问规则和按路径最终汇总 |
| `references/delegation.md` | 同步入口职责定位，补入实现规划，避免外部职责辅助文档收窄主入口范围 |
| `references/project-rules.md` | 删除独立任务类型枚举，引用主入口执行路径矩阵 |
| `references/validation.md` | 引用主入口执行路径矩阵，并允许规划、诊断和 review 按需核对验证证据 |
| `design-v1.md` | 在文末当前设计中同步设计目标、主流程、路径矩阵、事实门禁、渐进预算、最终汇总、目标形态和维护清单 |
| `design-iteration.md` | 记录本轮问题、裁决、实施范围、验证结果和测试限制 |

本轮不修改主场景 playbook、repo-map、casebook 或 AMCT 运行时代码。加载规则的正文只在主 `SKILL.md` 路径矩阵中维护；设计文档中的矩阵是结构快照，不参与运行时执行。

### 18.4 实施结果与验证

第 18.3 节范围已经完成，验证结果如下：

- `git diff --check` 与 `git diff HEAD --check` 均通过，工作区新增差异和相对 HEAD 的组合差异不存在空白错误。
- `quick_validate.py .agents/skills/amct-coder` 输出 `Skill is valid!`。
- 主入口直接链接的 Reference 去重集合与 `references/*.md` 实际文件集合一致，共 18 份。
- 运行时加载条件只在 `SKILL.md` 的执行路径矩阵中定义；`project-rules.md` 和 `validation.md` 均引用该矩阵，Reference 导航和渐进披露不再维护第二份任务类型清单。
- `design-v1.md` 文末当前设计已同步执行路径、事实门禁、渐进预算和按路径最终汇总；`delegation.md` 的入口定位也已包含实现规划。
- `SKILL.md` 正文规模低于 `skill-creator` 建议的 500 行上限。
- 未运行 AMCT Python/C++ 单测；本轮只修改 Skill 与分析 Markdown，不涉及 AMCT 运行时代码。
- 未执行 forward-test；当前工作区含历次设计和旧测试答案，不满足 cold 隔离条件，静态检查不能证明 Agent 已按新路径稳定行动。
- repo-map、casebook、docs/examples 不需要更新；本轮没有改变 AMCT 源码分层、模型经验或用户功能。

## 19. 第十轮审视：触发面、规则副本与跨文件一致性

第 18 节统一执行路径矩阵后，对主 `SKILL.md` 及本轮改动过的 reference 做了一次静态自查，覆盖：主入口全文、`project-rules.md`、`validation.md`、`delegation.md`、`repo-scenarios.md`、`contribution-flow.md`，并交叉核对 `repo-map.md`、`casebook/README.md`、`layer-rules.md`、`stage-impact.md` 与外部 skill 的 frontmatter。

本节只记录问题与拟定解法，不含实施结果。

### 19.1 静态核对结论

以下项目全部核对通过，不构成问题：

- 主入口 Reference 导航列出的 18 份文档与 `references/` 实际文件集合一致，无失效链接。
- `project-rules.md` 引用的 `.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md` 和四份 Issue 模板均存在，文件名（含 `feature_requeset.yaml`、`documentaion.yaml` 的既有拼写）与仓库一致。
- `validation.md` §场景测试索引的 9 条测试路径在仓库中全部存在。
- 主入口引用的 repo-map §分层边界、§更新规则和 casebook 的 L1/L2/L3 结构均存在。
- 全局 BLOCKER 中的 `ALGO_REGISTRY`、`AlgorithmRegistry`、`MODEL_REGISTRY`、`PtqUnit`、`export_deploy()`、`iter_ptq_units()` 和 `--algos` 都能在 `amct_pytorch` 中定位。
- `repo-scenarios.md` §入口文件列出的非通配路径全部存在。

即当前问题不在引用失效，而在触发面缺口、规则重复和跨文件口径分歧。

### 19.2 发现的问题

1. **description 缺负向边界，且与 `model-adapter` 直接冲突。** 主入口 frontmatter 只有正向枚举，没有其它 7 个仓内 skill 都具备的「不适用：→ `$xxx`」。选 skill 时只有 description 参与判断，正文 §1 的职责归属规则和 §全局 BLOCKER 的模型层拦截条都读不到。更具体的冲突是：`model-adapter` 的 description 把「补模型专属 attention·MLP wrapper」写成肯定项，amct-coder 只在正文把它写成 BLOCKER，排除项不在触发面上。`design_v0.md` §重构记录已诊断过这种不对称，第 18 轮改写 description 时排除项被整体移出，问题回归。

2. **主文件已从可执行手册退化为元规则文档。** 主入口约 8.5K 字符中出现 83 处「不」（含 18 处「不要」），`amct_pytorch/` 具体路径出现 0 次。§1「先判任务职责归属」占正文约三分之一，其中「事实门禁」和「契约缺失时的处理形态」讲的是如何推理、何时提问、如何表述不确定性，与 harness 自身的系统提示大面积重叠；§5「渐进披露与读取预算」进一步规定首轮锚点个数和辅助 reference 份数。导航性内容（Reference 表 + 执行路径表）约 15 行，约束性内容约 70 行，比例与「主入口负责路由、reference 负责领域规则」的既定分工相反。

3. **同一规则存在多份副本。** 至少四组：BLOCKER 只停错误方案 / 替代明确即继续 / 提问按契约缺失处理（正文 §定位、§契约缺失时的处理形态、§全局 BLOCKER 与 `delegation.md` §继续执行与询问，共四处）；「不要预读 repo-map 或场景手册」（§1 开头、§区分实现与错误落点 review、执行路径矩阵、§2 开头，共四处）；「实际加载外部 skill，不能口头声称委派」（§定位与 `delegation.md` 末段）；读取预算（执行路径矩阵、§5 第 1 条、§5 末段）。第 18 轮把加载时机收敛到唯一矩阵，但继续执行语义和预读禁令没有做同样的收敛。

4. **`layer-rules.md` §必读顺序与主入口的条件读取冲突。** 主入口规定 casebook 只在模型结构、checkpoint/config 或模型相关精度异常时按 L1→L2→L3 读取，而 `layer-rules.md` §必读顺序无条件要求第 3 步读 L1 casebook。`layer-rules.md` 的加载条件是「落点或上下游边界仍不明确时」，因此一个纯 dtype 落点问题会被它拉去读 `cross-model-pitfalls.md`，直接抵消主入口的条件读取。

5. **repo-map 写回清单已开始漂移。** 主入口明确声明「写回判据以 repo-map §更新规则为准，本文件不复制该清单」，但 `layer-rules.md` §repo-map 写回规则复制了整份清单，且两份已经不一致：repo-map 写「改变了实际**注册的模型集合**」，`layer-rules.md` 写「改变实际**注册链的含义**」。这是第 18 轮「加载规则只在一处维护」原则在写回规则上的同类缺口。

6. **固定流程 5 步与正文章节编号错位。** step 1 与 step 2 都落在 §1 内，step 3 散在 §2/§3/§5，step 4 与 step 5 没有对应章节；而 §4「按需检查阶段影响」在流程图里没有位置。§4 正文仅数行且全是加载条件，与 Reference 导航表「用途与加载时机」列的职能重复。

7. **次要项。**
   - `validation.md` §最小静态验证只写「对本次改动文件运行仓库 Ruff 配置对应的检查和格式化」，未给命令。`pyproject.toml` 只有 `[tool.ruff.format]`（`quote-style = "preserve"`），没有 lint 段，Agent 需要自行推断；而 `.pre-commit-config.yaml` 已明确 `ruff-check`（带 `--fix`）与 `ruff-format` 两个 hook，可以直接写出等价命令。
   - 主入口 §全局 BLOCKER 与 `delegation.md` §高歧义交界硬编码 `model-adapter`，同时 `delegation.md` 声明「新增或删除外部 skill 不要求同步修改本文件」，两者对漂移风险的处理不一致。
   - `delegation.md` 在主入口有两个入口（§契约缺失时的处理形态末段的正文链接 + Reference 导航表），而导航表已声明覆盖全部 reference。

### 19.3 拟定裁决

1. **恢复 description 的负向边界，并与 `model-adapter` 对齐口径。** 在 frontmatter 正向枚举后追加「不适用」子句，明确三类外部职责：新增或结构性修改模型 adapter/wrapper/attention/MLP/MoE（→ `$model-adapter`）、Ascend C kernel 与 C++ extension（→ `$amct-ops-dev`）、运行量化命令与生成交付产物（→ `$quant-run` / `$deploy-export`）。排除项的措辞必须与 `model-adapter` description 的肯定项可对照，避免同一个词在两侧含义相反。范围待确认，见 19.4。

2. **把推理型元规则压回 reference，主入口只保留路由与硬边界。** 「事实门禁」与「契约缺失时的处理形态」中与 harness 通用行为重叠的部分（如何表述不确定性、不用强结论、不为可自查问题提问）删除；保留 AMCT 特有的三条：区分「函数存在」与「活跃调用链已接通」、上游契约不清回到提供契约的层确认、不发明 CLI 参数/schema/文件名/payload 字段。提问形态整体移入 `delegation.md`，主入口只留一句指针。目标是把约束与导航的比例倒回来，并让主入口能承载具体锚点。

3. **按第 18 轮的同一原则收敛剩余规则副本。** 继续执行语义只在 §全局 BLOCKER 定义一次，§定位与 `delegation.md` 改为引用；预读禁令只在执行路径矩阵的「已路由到外部专门 skill」行定义一次，其余三处删除；读取预算只在 §5 末段保留一处。

4. **改写 `layer-rules.md` §必读顺序为条件读取。** 第 3～5 步加上与主入口一致的触发条件（模型结构、checkpoint/config、模型契约或模型相关精度异常），或直接删除 casebook 步骤、改为指向主入口 §渐进披露的 L1→L2→L3 规则。取后者可少维护一份触发条件。

5. **删除 `layer-rules.md` §repo-map 写回规则的清单副本**，改为指向 repo-map §更新规则。以 repo-map 的「注册的模型集合」为准。

6. **重排固定流程与章节。** 固定流程改为与实际章节一一对应（判职责 → 选路径 → 声明边界 → 选主手册并读锚点 → 实现或分析 → 按路径验证与汇总）；§4「按需检查阶段影响」并入 Reference 导航表中 `stage-impact.md` 行的加载时机，正文不单独设节。

7. **次要项处理。**
   - `validation.md` 补出等价命令：`ruff check --fix <changed files>` 与 `ruff format <changed files>`，并注明配置来源是 `.pre-commit-config.yaml` 与 `pyproject.toml` `[tool.ruff.format]`，仓库当前无独立 lint 段。
   - 硬编码 skill 名保留（触发面必须写出具体名字才有路由效果），但把 `delegation.md` 的「新增或删除外部 skill 不要求同步修改本文件」限定为「不要求同步非高歧义交界的 skill」，与 §高歧义交界表的硬编码自洽。
   - 删除 §契约缺失时的处理形态末段的 `delegation.md` 正文链接，只保留 Reference 导航表入口。

### 19.4 待确认项

- **description 排除项范围。** 三个候选：只写模型层与 NPU 算子两类（最短，覆盖两个最高频误触发）；三类全写（含运行命令与交付产物）；三类全写并额外点出「review 已有落点是否合理仍属 amct-coder」。第三种能压掉「review 模型层文件」被误路由到 `model-adapter` 的情况，但 description 会变长。需要确认后再改，因为改 description 会直接改变触发行为。

### 19.5 实施范围

尚未实施。本次仅完成静态自查并记录问题与拟定解法。预期涉及文件：

| 文件 | 预期修改 |
| --- | --- |
| `SKILL.md` | frontmatter 补排除项；压缩事实门禁与提问规则；收敛继续执行语义、预读禁令和读取预算的副本；重排固定流程；§4 并入 Reference 导航表 |
| `references/delegation.md` | 承接提问形态；限定「不要求同步」的适用范围 |
| `references/layer-rules.md` | §必读顺序改条件读取或指向主入口；删除 repo-map 写回清单副本 |
| `references/validation.md` | 补 Ruff 等价命令与配置来源 |
| `design-v1.md` | 文末当前设计同步 description 边界、主入口职能划分和条件读取口径 |
| `design-iteration.md` | 补记本轮实施结果与验证 |

实施时的验证要求沿用第 18 轮：`git diff --check`、`quick_validate.py`、Reference 集合一致性核对，并在物理隔离的新会话中对「模型层 review 不被误路由」和「纯 dtype 任务不读 casebook」两条做 forward-test；静态检查不作为行为证据。

本节第 2、3 条（压回推理型元规则、收敛规则副本）与第 20 节属同一方向，其中「收敛继续执行语义和预读禁令副本」已在第 20 节实施；第 1 条 description 排除项、第 4、5 条 `layer-rules.md` 修正和第 7 条 Ruff 命令仍未实施。

## 20. 第十一轮审视：撤销执行路径矩阵，加载时机就近声明

第 18 节引入「按交付物选择执行路径」矩阵，目的是把 `project-rules.md`、repo-map、主场景手册和 `validation.md` 的加载时机收敛到唯一事实源。本轮复核确认该目的已经落空，同时按用户判断压缩了外部 skill 的提及面。

### 20.1 发现的问题

1. **矩阵没有真正减少重复，只增加了一层间接引用。** 四格规则在别处都有可独立执行的正文：「不为外部职责预读 repo-map/主手册」在 §1 开头和 §区分实现与错误落点 review；「纯规范合规 review 跳过 repo-map」在 §2 开头，且该处已把四行结论原样复述一遍，读者不需要回查表；`project-rules.md:3` 和 `validation.md:3` 本来就在描述自己的加载条件，只是措辞改成了反向引用主入口那张表。矩阵剩下的作用是一个「唯一事实源」标签，让 4 处引用它，而每处引用又都要重述结论才能读懂。

2. **外部 skill 提及面过大。** 全文 13 处提及，其中「不要预读 repo-map」重复 4 次、「实际加载而非口头声称委派」2 次，§交付与验证 还要求汇总外部 skill 的交付结果——那属于目标 skill 自己的职责。路由之后的加载和交付不该由 amct-coder 规定。

3. **`contribution-flow.md` 曾被从导航表删除，但文件仍在且仍被引用。** `references/` 实际 18 份，表格列 17 份，而表头写「全部 reference」，声明为假。该文件 18 行中门槛、模板、GitCode 路由三项自述「以 `project-rules.md` 为准，不维护第二份副本」，独有内容只有授权边界、按实际模板填写、先查已有 Issue/PR 三条，且 `project-rules.md:38` 已在实际操作前引出它。

4. **上一轮手工编辑引入的缺陷。** `[repo-map.md] (path)` 中间有空格导致链接失效；「实际编码」条目与「询问遵循以下形态：」之间缺空行导致后续四条被吞进列表项；执行路径矩阵引导句句末为半角句号；frontmatter 去掉「调试与 review」但正文 review 路径仍在（矩阵两行、§1 整节、导航表 12 处、§交付与验证两条），且「等」不是有效触发词；「区分实现与错误落点 review」两条判据从互斥（新增或结构性实现 / 评审明确文件或方案）改为可同时命中（属于专门 skill 职责 / 提出编码任务），标题与内容脱节，落点 review 的出口也被删除。

5. **五条规则正文被删但概念仍被引用。** 「区分函数存在与活跃调用链已接通」（frontmatter、§5 停止条件、§契约缺失时的处理形态 三处仍引用「活跃调用链」）；「不发明类名/CLI 参数/schema/文件名/payload」被抽象成「不要可自行编造条件」，丢失可逐项自查的清单；「候选锚点」条目的前置条件「上述契约未冻结时」被删，规则变为无条件，与 §定位「修改代码前说明主落点层」冲突；「不做开放式提问」被删，而同轮又在两处新增了询问用户；读取预算末句「用于防止预读，不用于阻止完成任务」被删，2～4 个锚点从软目标变成硬上限。其中前两条在 §19.3 第 2 条的保留名单内。

### 20.2 设计裁决

1. **删除执行路径矩阵，加载时机就近声明。** 矩阵四行中只有两条是别处没有的信息，回收进 §Reference 导航表对应行的「用途与加载时机」列（`project-rules.md` 写「修改代码前，或做规范合规 review 时读取」；`validation.md` 写「实际编码在修改后…只做规划或 review 时按需读取相关部分」）。「纯规范合规 review」的定义随 repo-map 条件移入 §2 开头。`project-rules.md` 与 `validation.md` 开头改为直接声明自己的加载条件，不再反向引用主入口。固定流程从 5 步收敛为 4 步，与正文章节对应。

2. **外部 skill 提及压到 4 处**：§定位一句（实际加载而非口头委派）、§1 判归属（补「后续加载和交付由它决定」）、§全局 BLOCKER 中 `model-adapter` 与 `amct-ops-dev` 两道自我围栏、`delegation.md` 入口。删除 3 处重复的预读禁令和 §交付与验证 中要求汇总外部 skill 交付结果的条目。围栏保留的理由是它约束 amct-coder 自己不越界，不是管辖已路由出去的流程。

3. **`contribution-flow.md` 不加回导航表**，改表头为「从本入口直接进入的 reference」并注明贡献流程由 `project-rules.md` 引出。这样假声明消除，文件也不失联。

4. **恢复 20.1 第 4、5 项列出的缺陷和规则正文**，包括 frontmatter 补回「行为诊断与代码 review」、「区分实现与落点 review」恢复互斥判据和外部职责出口。§3 末段「未命中现有手册」保留本轮新增的落点确认，但去掉重复的「并询问用户」，只在实际编码前确认落点和改动范围，只读规划直接交付规划；两处 `rg` 改为「搜索工具」（本机未安装 ripgrep，实际走 harness 内置 Grep）。

5. **「占位对象不得继承上一题名称」不在主入口恢复**：`llm-algorithm.md:55` 已有该规则的完整正文，属于算法场景的领域规则。

### 20.3 实施范围

| 文件 | 修改内容 |
| --- | --- |
| `SKILL.md` | 删除执行路径矩阵并回收 2 条加载时机；压缩外部 skill 提及；恢复 5 条规则正文和 4 项编辑缺陷；固定流程改 4 步；导航表表头改「从本入口直接进入」；最终汇总按实际执行动作组织 |
| `references/project-rules.md` | 开头改为直接声明加载条件 |
| `references/validation.md` | 开头改为直接声明加载条件 |
| `design-v1.md` | 文末当前设计同步：设计目标第 2、3 条、核心原则、文档分工两行、主流程 6 步、§1 补路由后不再规定、§2 改「加载时机就近声明」、§3 补 17 份与 `contribution-flow.md` 说明、§4 占位规则改指 playbook、§5 预算补软目标、§6 改按实际动作汇总、目标形态 9 条、维护检查清单第 2、3、4、9、10、11 条 |
| `design-iteration.md` | 本节；并在 §19.5 标注哪些拟定项已由本节实施 |

未修改：9 份主场景 playbook、`repo-map.md`、`casebook/`、AMCT 运行时代码。§19 记录的 description 排除项、`layer-rules.md` 两项修正和 Ruff 命令仍待实施。

### 20.4 实施结果与验证

- `git diff --check` 通过，无空白错误。
- `quick_validate.py .agents/skills/amct-coder` 输出 `Skill is valid!`。
- 主入口 17 个 `references/*.md` 链接全部可解析；`repo-map.md`、`casebook/README.md` 存在。`references/` 实际 18 份，未列入的 `contribution-flow.md` 由 `project-rules.md:38` 引出，表头已说明。
- `.agents/skills/` 下已搜不到「执行路径」；「交付物」仅剩 §契约缺失时的处理形态 两处，指按只读规划与实际编码分流，与已删除的矩阵无关。
- `SKILL.md` 162 行 / 7.7K 字符，低于 `skill-creator` 建议的 500 行上限。
- 未运行 AMCT Python/C++ 单测：本轮只修改 Agent Markdown 与设计文档。
- 未执行 forward-test：当前工作区含历次设计和旧测试答案，不满足 cold 隔离条件。待验证行为包括纯规范合规 review 不读 repo-map、未命中手册的任务先确认落点、review 类请求仍能触发本 skill。
- repo-map 不需要写回：未改变 AMCT 源码分层或核心流程，不满足其§更新规则任一条件。

## 21. 第十二轮审视：有 Skill 测试暴露的子能力闭环缺口

本轮对 `.agents/docs/test-skill.md` 的六题回答做静态审查，并与 `test-noskill.md`、`amct-coder/SKILL.md`、现存 17 份 reference 及当前源码锚点交叉核对。评估参赛回答时只取 `test-skill.md` 第 1～486 行；第 487 行之后是后附审查结论，不作为 Agent 原始回答的一部分。

有 Skill 版本已经改善分层、registry、职责边界和验证矩阵意识，但仍不足以判通过。问题不在所有子能力都缺少内容：`dtype-quant-module`、`llm-algorithm`、`llm-data-workflow`、`ptq-solver-params` 已有较好的局部骨架；主要缺口是这些规则没有形成跨 algorithm、data、solver、save/load、eval/deploy 的强制收口，Agent 因而能引用正确局部概念，同时给出不成立的端到端方案。

本节只记录发现、源码证据和拟定改法，不含实施结果。

### 21.1 P0：算法状态生命周期没有闭环

`llm-algorithm.md` 当前要求训练型算法提供 `trainable_params()`，统计型算法使用 buffer 并实现导出/加载，但没有定义一次性统计优化算法的完整状态机：

1. observe 由谁在何时开始和停止。
2. GT 生成阶段是否允许 observe，如何避免统计或权重变化污染浮点目标。
3. 统计完成后由谁触发一次性权重优化，以及该动作是否幂等。
4. 无 trainable params 时由谁驱动 dataloader；不能假设现有 solver 会自然遍历数据。
5. fake quant、real quant、统计状态和优化后权重之间的先后关系。
6. 优化后的权重、scale/offset 或其它状态如何保存、回载并被 eval/deploy 消费。

当前源码事实直接反证测试 4 的 GPTQ 数据流：

- `QuantLinear.forward()` 在非 eval 分支已经调用 `weight_quantizer.observe_input(hidden_states, weight)`。
- `LlmPtqWorkflow._prepare_unit_batch()` 生成 GT 时关闭 weight/activation quant，同时显式开启 observe。
- `BlockwiseSolver.solve()` 在没有 trainable 参数时于遍历 dataloader 前直接返回。

因此不能只增加 calibration pre-pass，再假设 solver 会触发一次性优化。拟在 `llm-algorithm.md` 增加“算法执行形态与阶段状态机”，要求对 trainable、observe-only、observe-then-finalize 三类形态逐阶段填写状态、触发者、副作用、持久化和失败行为；在 `ptq-solver-params.md` 对应增加无 trainable 参数时的执行归属门禁。

### 21.2 P0：helper 存在与活跃保存链被混淆

`ptq-solver-params.md` 当前描述 `PtqParamHandler.export_unit()` 的递归导出语义，但没有同时指出训练工作流的活跃保存链。当前 `LlmPtqWorkflow._run_blockwise()` 实际执行：

```text
solver.solve(unit_batch.data_loader)
  -> solver.finalize()
  -> _save_unit_result(unit, unit_result)
```

`BaseSolver.finalize()` 只调用当前 unit module 自身的 `export_ptq_params()`，否则导出 `requires_grad` 参数；`_save_unit_result()` 直接 `torch.save(result)`。`PtqParamHandler.export_unit()` 虽然存在并支持递归子模块，但没有被这条训练保存链调用。

这导致测试 1、5 把“存在递归 helper”误写成“训练保存已递归闭环”。拟在 `ptq-solver-params.md` 显式列出训练保存、eval/deploy 回载两条活跃链，要求每次保存方案回答 caller、结果对象、文件命名、回载消费者和严格性；未找到 caller 的 helper 必须标记为未接通，不得作为行为证据。

### 21.3 P0：缺少 composite unit / joint calibration 门禁

现有 `llm-data-workflow.md` 只记录 extract/PTQ 当前为单 `quant_target`，`ptq-solver-params.md` 的 granularity 表也主要面向新 solver/granularity，尚不能约束 Attention+MLP 复合 unit。联合校准计划至少必须逐项闭环：

- CLI `quant_target` choices 与 PTQ 参数目录映射。
- adapter 是否同时注入 attention 和 MLP wrapper。
- `PtqUnit` 的 `kind`、`name`、`layer_idx`、`module`、`save_name` 语义。
- capture 是 decoder block 原始输入还是 norm 输出，避免把 norm 输出再次送入完整 block。
- inputs、per-sample kwargs、GT 与 batch 的对齐方式。
- 是否真的需要新 solver，还是现有 solver 已能消费复合 module。
- finalize 如何拆分或聚合保存，eval/deploy 如何用一致结构回载。
- 公平对照必须比较“分开校准后 attention 与 MLP 均量化”和“联合校准后两者均量化”，不能拿单独只量化一侧的结果代替。

拟在 `llm-data-workflow.md` 增加复合 unit 的 producer/consumer 契约，在 `ptq-solver-params.md` 增加 joint solve/save/reload 门禁，并把以上任一未冻结项列为条件式规划而非确定实现。

### 21.4 P1：已有规则缺少强制消费面证据

`dtype-quant-module.md` 已覆盖 registry、CLI、bit policy、quantizer、deploy 和 algorithm compatibility，但测试 2 仍只规划 dtype 文件和注册导入，遗漏 CLI 与 tensorwise deploy。这说明当前内容是建议性步骤，没有要求最终交付逐项出示结论。

拟增加固定消费面状态表。每个新 dtype 必须对下列项目逐项标记 `supported`、`rejected` 或 `not applicable`，并给出源码或测试证据：

```text
registry / CLI / bit policy / weight quantizer / activation quantizer /
blockwise deploy / tensorwise deploy / quantization_config /
algorithm compatibility / model-specific exact branches
```

同时增加既有语义复用门：若目标格式已被现有 MXFP4 或其它 dtype 覆盖，优先零改动、别名或复用现有分支，不能仅因名称不同重复创建 dtype。

### 21.5 P1：敏感层与自动跳层语义没有归属

测试 3 暴露的未定义项包括：浮点/量化成对输出如何取得；Hessian/Fisher 的梯度或统计来源；多个 transform/linear 的 score 如何聚合到 unit/layer；阈值或排序作用域；跳过 `solver.solve()` 后 quant block 是否仍保持量化；跳层如何写入 PTQ 参数、strict load、eval 和 deploy；跳层是保持浮点还是回退 direct quant。

拟由 `llm-algorithm.md` 定义“统计与决策型能力”的输入、聚合和决策输出，由 `ptq-solver-params.md` 定义 skip/partial result 的保存、回载和续跑生命周期。暂不新增独立 reference，避免把同一状态机再拆散。

### 21.6 P1：skill 路由触发面仍有缺口

`amct-coder` frontmatter 只有辅助编码范围的正向描述；客户端选择 skill 时正文尚未加载，因此正文中的模型 adapter、NPU kernel 和实验流程围栏无法参与首次路由。`model-adapter` 的 description 已明确把新增模型、`PtqUnit` 和模型专属 attention/MLP wrapper 列为肯定触发项，两者在触发面不对称。

拟在 `amct-coder` description 直接补三类负向边界，并保留“评审已有落点是否合理”仍属于 `amct-coder` 的例外：

- 新增或结构性修改模型 adapter/wrapper/attention/MLP/MoE -> `model-adapter`。
- Ascend C kernel、PyTorch C++ extension、schema/Meta/CMake -> `amct-ops-dev`。
- 实际运行量化/评测或生成 deploy 交付产物 -> `quant-workflow`、`quant-run`、`deploy-export`。

### 21.7 P1：自然语言目标到 CLI/config 缺少一致性门

测试 6 把用户要求的 FP16 baseline 实际写成 `eval_mode=bf16`，又把“INT4 权重量化”配置成 W4A4。现有 reference 没有要求把自然语言目标展开成 CLI 可执行语义并逐 role 核验。

拟在 `public-api-tests.md` 增加用户目标到配置的口径门：FP16 与 BF16 不得静默互换；weight-only 必须证明所有 activation role 为 16 bit；W4A4 不得描述为 INT4 weight-only；CLI 不支持用户要求的精度或组合时必须显式报告差异，按只读规划保留条件分支或在实际编码前确认契约。

### 21.8 P1/P2：reference 测试体系与静态一致性不足

六题属于跨契约压力测试，不能证明全部 reference 自身可靠。`classic-graph`、`ops-boundary`、`public-api-tests` 等未被六题充分覆盖，当前只能标记为未验证。后续测试应分层：

1. 每个 reference 至少覆盖正常题、边界路由题、未知契约题、伪活跃调用链题。
2. 跨 reference 覆盖 dtype+deploy、algorithm+solver、data+adapter、model+external skill。
3. cold 与 warm 分开，保存真实投喂 prompt，避免把预期答案泄露给被测 Agent。
4. 记录源码 commit、实际读取资料、token、rubric 和证据等级。
5. 原始测试回答与 reviewer 审查分文件，避免 `test-skill.md` 当前的混写。

静态一致性还存在以下问题：

- `project-rules.md` 引用当前不存在的 `contribution-flow.md`；第 19、20 节关于其存在性和 reference 数量的历史结论已经漂移。
- `layer-rules.md` 无条件要求读取 L1 casebook，与主入口只在模型相关任务读取的规则冲突。
- `layer-rules.md` 复制 repo-map 更新条件，且措辞已经分叉。
- `validation.md` 只写 Ruff 原则，没有给仓库对应的可执行命令。
- `stage-impact.md` 只有阶段摘要，未强制记录每阶段入口、写入产物、消费者、保存/回载和验证证据。

拟修复失效链接、条件读取冲突和规则副本；补 Ruff 等价命令；扩展 `stage-impact.md` 为跨阶段证据表。历史章节保留原文作为迭代记录，不回写伪装成当时已经正确，只在本节和实施结果中说明当前事实。

### 21.9 实施顺序与验证边界

按风险排序实施：

1. `llm-algorithm.md` + `ptq-solver-params.md`：统计生命周期、活跃保存链、skip/partial result。
2. `llm-data-workflow.md`：composite unit / joint calibration 门禁。
3. `dtype-quant-module.md`：强制消费面状态表和既有语义复用门。
4. `SKILL.md`：frontmatter 负向路由边界。
5. `public-api-tests.md`：precision、weight-only 和 bit-policy 口径一致性。
6. `project-rules.md`、`layer-rules.md`、`validation.md`、`stage-impact.md`：静态一致性修复。
7. 完成链接解析、Reference 集合、`git diff --check` 和 `quick_validate.py` 静态验证。

本轮修改 Skill 文档，不改变 AMCT 运行时代码，因此不运行 Python/C++ 单测。当前会话和工作区已包含测试答案及本节诊断，不具备 cold forward-test 的隔离条件；静态验证只能证明文档结构和引用成立，不能证明新 Agent 已稳定遵循这些门禁。

### 21.10 设计记录写入时状态

尚未实施。下一步按 §21.9 修改现有 Skill/reference，并在本节后追加实际变更与验证结果。

### 21.11 实施结果

已按 §21.9 完成现有文件内的修订，没有新增 reference：

| 文件 | 实际修改 |
| --- | --- |
| `SKILL.md` | frontmatter 增加模型结构、NPU 算子、量化/评测与 deploy 交付的负向路由，并保留已有落点 review 例外 |
| `references/llm-algorithm.md` | 增加 trainable、observe-only、observe-then-finalize 三类执行形态；GT/observe/finalize/save/eval-deploy 状态表；统计决策与敏感层门禁 |
| `references/ptq-solver-params.md` | 写明当前训练保存活跃链及 `PtqParamHandler.export_unit()` 未接入事实；增加无 trainable 参数、复合 unit、skip/partial result 生命周期 |
| `references/llm-data-workflow.md` | 增加 composite unit / joint calibration 的 CLI、adapter、输入、kwargs、GT、result producer/consumer 表和公平对照规则 |
| `references/dtype-quant-module.md` | 增加既有数值语义复用门和 registry/CLI/bit policy/quantizer/deploy/algorithm/model 分支强制状态表 |
| `references/public-api-tests.md` | 增加 FP16/BF16、weight-only/W4A4、dtype/bit policy 和 baseline 对比的用户口径门 |
| `references/stage-impact.md` | 从阶段摘要扩展为入口、输入、状态/产物、消费者和证据等级的强制阶段表 |
| `references/project-rules.md` | 删除不存在的 `contribution-flow.md` 链接，把授权、模板和已有 Issue/PR 检查收回本文件 |
| `references/layer-rules.md` | casebook 改为模型相关条件读取；repo-map 写回规则改指唯一事实源，不再复制清单 |
| `references/validation.md` | 补 Ruff 命令；增加 Skill/reference 链接、导航集合、cold/warm forward-test 和测试材料隔离要求 |
| `design-v1.md` | 同步当前 frontmatter、阶段证据、17 份 reference 及不存在额外 contribution reference 的设计事实 |

### 21.12 验证结果与边界

- `git diff --check` 通过，无空白错误。
- 运行当前 `skill-creator` 的 `quick_validate.py .agents/skills/amct-coder`，输出 `Skill is valid!`。
- `SKILL.md` 导航中的 reference 去重集合与 `references/*.md` 实际集合一致，共 17 份，无遗漏或额外项。
- 解析 `.agents/skills/amct-coder/` 下全部 Markdown 相对链接，目标均存在；失效的 `contribution-flow.md` 引用已消除。
- 重新核对 `QuantLinear.observe_input()`、GT 阶段 observe 开关、solver 无 trainable 参数提前返回、`BaseSolver.finalize()`、`PtqParamHandler.export_unit()`、`_save_unit_result()` 及 CLI choices，新增规则与当前源码事实一致。
- `SKILL.md` 仍为 162 行，低于 `skill-creator` 建议的 500 行上限；该目录不存在 `agents/openai.yaml`，无需同步 UI metadata。
- 未运行 AMCT Python/C++ 单测：本轮只修改 Skill/reference 与设计 Markdown，不改变运行时代码。
- 未执行 cold forward-test：当前会话和工作区已经包含原始测试答案、审查结论与本节预期，无法满足隔离要求。同会话静态复核不作为 Agent 行为证据。
- 未更新 repo-map/casebook/docs/examples：本轮没有改变 AMCT 分层、模型经验或用户功能。
- 工作区原有 `test.md`、`test-noskill.md`、`test-skill.md` 及测试 prompt 相关未提交状态均未改动或清理。

### 21.13 有 / 无 Skill 横向对比结论补录

第 21 节最初只记录了有 Skill 版本的残余漏洞和整改，没有把本轮 A/B 横向结论独立写入，这是记录遗漏。补录结论如下：

- **有 Skill 版本约 3.5/10**：该分数来自 `test-skill.md` 第 487 行之后的 reviewer 审查，不属于被测 Agent 原始回答。
- **无 Skill 版本约 2.5/10**：`test-noskill.md` 没有附同格式 reviewer 评分，本分数是按相同关注点做的回顾性启发式估计，不是统一 rubric 自动计算值。
- **相对胜负为有 Skill 4 胜、1 平、1 负**。这里的“胜”只表示两份不合格答案中的相对表现更好，不表示该题已经达到可实施或可合入标准。

| 测试 | 相对结果 | 主要差异 | 两边仍有的问题 |
| --- | --- | --- | --- |
| 1. BATQuant + MXFP4 | 有 Skill 胜 | 有 Skill 版本进入 LLM new-path 的 algorithm、structure transform 和 solver 契约；无 Skill 版本主要套用 classic `BaseQuantizeModule` / GPTQ 式路径 | 有 Skill 仍把可由源码确认的 transform、trainable params 和保存链留成待确认或判断错误 |
| 2. HIF4 dtype | 有 Skill 胜 | 有 Skill 版本定位到 `quantization/dtypes`、registry 和 dtype 契约未知性；无 Skill 版本大量落点来自另一套 classic/experimental 配置与算子体系 | 有 Skill 漏掉 CLI、tensorwise deploy 和完整消费面，且过早固定文件/接口候选 |
| 3. 敏感层识别 | 有 Skill 胜 | 有 Skill 版本至少把能力放回 new-path algorithm/workflow/solver 边界；无 Skill 版本主要围绕 classic quantize/pass 和已有 GPTQ 统计展开 | 有 Skill 没有闭环成对输出、score 聚合以及跳层在保存、eval、deploy 中的语义 |
| 4. GPTQ | 平 | 两版都识别了 new-path `ALGO_REGISTRY`、weight target 和统计型算法与 trainable 算法的差异 | 两版都没有正确闭环 GT observe、无 trainable 参数时的数据遍历、一次性 finalize 以及优化权重的保存回载；无 Skill 的展开更长，但不构成更高等级证据 |
| 5. Attention + MLP 联合校准 | 有 Skill 负 | 无 Skill 版本对联合 target、联合求解和实验矩阵展开得更完整；有 Skill 版本更早把 `attn+mlp` 和完整 block 路线写成确定方案，并错误声称递归保存已闭环 | 两版都没有正确处理 CLI choices、wrapper 双侧注入、block 原始输入与 double norm、完整 `PtqUnit`、保存回载和公平对照；无 Skill 也不合格，只是相对略优 |
| 6. Kimi-K3 端到端流程 | 有 Skill 胜 | 有 Skill 版本更接近 adapter 与端到端 workflow 的职责边界；无 Skill 版本直接假设 Kimi-K3 复用 Qwen3，并转向未经当前主链证明的 experimental FlatQuant 实现 | 有 Skill 仍未真正路由 `quant-workflow`，混淆 FP16/BF16、INT4 weight-only/W4A4，并发明模型落点候选 |

横评说明 Skill 的有效增益主要在三个方面：更容易进入正确代码体系，能识别 registry/adapter/solver 等分层边界，也更愿意保留未知契约。它没有稳定解决三个更关键的问题：确认活跃调用链、描述跨阶段状态机、把局部 helper/规则收口为可执行的端到端契约。因此本轮结论不是“有 Skill 已经可用”，而是“方向有改善，但准确度和闭环性仍不合格”，这也是 §21.1～§21.9 修订的依据。

## 22. 第十三轮审视：隔离 cold forward-test 与规划一致性

第 21 轮静态修订后，在 commit `e181c6e` 的浅克隆副本中移出 `test-skill.md`、`test-noskill.md` 和 `amct_coder_analysis/`，用三个无历史上下文的独立 Agent 分别验证模型落点 review、HIF4 dtype 规划和 GPTQ 统计生命周期。测试只投喂原始任务、`amct-coder` 路径和只读约束，没有提供预期答案或第 21 节诊断。

### 22.1 结果

| 用例 | 结果 | 主要证据 |
| --- | --- | --- |
| Kimi-K3 wrapper 拟放入 workflow | 通过 | 先由 `amct-coder` 否决越层落点，再实际加载 `model-adapter` 形成模型层计划；具体 family、类名和 wrapper 必要性保持为候选 |
| 基于 INT4 扩展 HIF4 | 部分失败 | 正确核对 registry、CLI、bit policy、quantizer 和 deploy，且没有读取模型 casebook；但先声明 role 范围未冻结，后续又自行选择 weight-only 并拒绝 activation/cache |
| GPTQ observe-then-finalize | 语义通过、读取预算失败 | 正确识别 GT observe、副作用边界、solver 无 trainable 参数提前返回、活跃保存链和 tensorwise deploy 缺口；但读取 7 份 Reference 及大量源码/测试，未说明超预算原因，人工催收后才停止 |

这轮说明第 21 节新增的领域事实有效，但两条执行约束仍不够稳定：条件式计划会在后文被悄悄冻结；跨契约题会把“需要闭环”误读成“全文读取所有相关资料”。

### 22.2 拟议修改

本节只记录下一轮候选方案，当前尚未修改 `SKILL.md` 或任何 Reference：

1. 在 `dtype-quant-module.md` 中把规则泛化为：参考实现的 role、bit、粒度和 deploy 范围只构成现状证据，不能自动冻结新 dtype 的目标契约。避免直接写入 HIF4、INT4 或 weight-only 等测试题专属措辞。
2. 允许只读规划把未冻结的消费面标为 `pending`，并列出缺失契约及各条件分支；只有契约冻结后的实现 review 和最终说明才收口为 `supported`、`rejected` 或 `not applicable`。
3. 增加对应 BLOCKER：不得先把 role、bit、粒度或 deploy 范围列为待确认，后续又在没有新证据时选定一个分支并写成目标状态。
4. 暂不修改 `SKILL.md` 或 `llm-algorithm.md` 的读取预算规则。已有两轮隔离复测都没有证明候选停止门有效，继续增加文字只会扩大上下文。

### 22.3 验证结果与边界

- baseline cold 测试中，模型落点 review 通过；GPTQ 语义通过但读取预算失败；HIF4 暴露未知 role 被自行冻结的问题。
- HIF4/GPTQ 各进行两轮隔离复测，均未在预算内完成并被中止；中止后的输出不计作独立 cold 语义证据。因此上述拟议修改尚无修改后行为证据，读取效率问题保持未解决。
- 原始 baseline 输出保留在本轮 Agent trace，未写入被测副本；本轮不满足 `validation.md` 要求的 prompt、输出和评分分文件归档，不能把 trace 当成可复用测试资产。
- 本轮不改变 AMCT 运行时代码、repo-map 或 casebook，不运行 AMCT Python/C++ 单测。

### 22.4 下一步验证

实施前先准备一个不含 HIF4、INT4 和 weight-only 关键词的短 dtype 规划题，验证规则能否迁移到其它格式和 role 组合。实施后用物理隔离的新会话复测，并将原始 prompt、Agent 输出和 reviewer 评分分别保存；只有 Agent 全程保留未知契约为 `pending`，且没有从参考实现偷渡目标范围时，才把拟议修改记为已验证。

### 22.5 后续总体策略

后续采用“小步修改 + 独立 cold test + 达标后再扩面”的迭代方式，不再一次性横向扩写 `SKILL.md` 或全部 Reference。当前证据只稳定支持两个优先问题：

1. 未知目标契约会在规划后半段被 Agent 无证据地冻结。
2. 跨契约任务会把“检查闭环”解释成“全文读取所有相关 Reference”，导致读取预算失控。

第一个问题已有可复现的语义失败，下一轮可以直接修改；第二个问题同时存在规则静态冲突和行为失败，但具体收敛方式需要先记录读取轨迹，再决定调整 Reference 指令还是主入口预算。除这两项外，其它扩写必须由新测试失败驱动。

### 22.6 P0-A：建立 dtype 目标契约状态机

当前 `dtype-quant-module.md` 要求计划、实现 review 和最终说明的消费面状态只能是 `supported`、`rejected` 或 `not applicable`。这会迫使只读规划在信息不足时仍选择一个终态，是 HIF4 cold test 中 role 被自行冻结的直接诱因。

下一轮应把“当前仓库事实”和“目标契约决策”拆成两个维度：

| 维度 | 表达内容 | 允许状态 |
| --- | --- | --- |
| 当前仓库事实 | 现有 registry、CLI、bit policy、quantizer、deploy 和算法分支实际支持什么 | `confirmed` / `not found`，附源码证据 |
| 目标契约 | 新 dtype 计划支持、拒绝或不涉及什么 | `pending` / `supported` / `rejected` / `not applicable` |

状态转换规则：

1. 只读规划允许目标契约保持 `pending`，同时列出缺失输入和各条件分支。
2. `pending` 只能由用户明确输入、已冻结需求、兼容性裁决或可直接确定目标语义的源码契约转成终态；参考实现的当前行为本身不能触发转换。
3. 当前实现是 weight-only、固定 bit 或固定粒度，只能证明现状，不能自动成为新 dtype 的目标范围。
4. 同一回答前文已标记 `pending`，后文没有新增证据时不得改写为 `supported`、`rejected` 或 `not applicable`。
5. 契约仍为 `pending` 时，不把具体 dtype 类名、实现文件、payload 字段、CLI 值或最终改动面写成已经确定的唯一方案。

对应 BLOCKER 应增加：不得先承认 role、bit、粒度或 deploy 范围未知，随后仅根据参考实现选择一个分支；不得把现有实现证据写入目标决策列。

#### 22.6.1 dtype 验证题与通过条件

使用不包含 HIF4、INT4、MXFP4、weight-only 等现有测试关键词的短题，要求为一种新数值格式做只读接入规划，但故意不提供 weight/activation/cache role、bit、粒度和 deploy 范围。

通过条件：

- Agent 正确核对当前 registry、CLI、bit policy、quantizer 和 deploy 消费面。
- 所有未给出的目标 role/bit/粒度保持 `pending`。
- 输出条件分支，不替用户推荐缺少依据的默认范围。
- 不从 INT、MXFP 或 experimental 参考实现偷渡目标契约。
- 未冻结项不产生唯一文件、类名、payload 或命令方案。

至少在三个相互隔离的 cold 会话中全部通过，才能认为该状态机具备稳定行为证据。

### 22.7 P0-B：消除读取预算的规则冲突

主 `SKILL.md` 当前规定默认只加载一个主 playbook、最多一份辅助 Reference 和首轮 2～4 个源码/测试锚点，但三个子能力存在会自动突破该预算的指令：

- `llm-algorithm.md` 要求改变 dtype real quant 语义时同时读取 `dtype-quant-module` 和 `deploy-export-code`。
- `ptq-solver-params.md` 要求新增 solver/granularity 时同时读取 `llm-data-workflow` 和 `public-api-tests`。
- `llm-data-workflow.md` 要求新增 CLI 参数或用户可见数据格式时同时读取 `public-api-tests`。

“辅助 Reference 默认最多一份”和“同时读两份”不能同时作为可执行规则。下一轮不应简单放宽预算，而应把自动扩张改成证据驱动：

```text
完整读取一个主 playbook
  -> 先查主 playbook 已列出的直接源码/测试锚点
  -> 明确仍未解决的跨契约问题及 owner
  -> 只加载该 owner 的一份辅助 Reference
  -> 第二份辅助 Reference 只有在第一份和源码仍无法回答时加载，并记录扩张原因
```

辅助读取前形成最小“扩张凭据”：

| 字段 | 要求 |
| --- | --- |
| 未解决契约 | 精确到 CLI choice、payload、保存 caller、回载结构等具体问题 |
| 已查证据 | 已读取的源码符号、调用方或测试断言 |
| owner Reference | 为什么该 Reference 拥有剩余契约 |
| 超预算理由 | 为什么主手册、源码和第一份辅助资料仍不足 |

不要因为题面同时出现 algorithm、solver、deploy、CLI 等词就读取所有同名 Reference。跨阶段检查可以输出 `affected/not applicable` 状态，但只有实际改变该阶段的输入、状态或产物时才完整加载对应 owner 手册。

#### 22.7.1 GPTQ 读取效率验收

继续使用 observe-then-finalize GPTQ 题做语义与效率双重验证：

- 正确识别 GT 生成阶段 observe、副作用隔离、无 trainable 参数时 solver 提前返回、一次性 finalize caller、优化状态保存/回载和 deploy 消费。
- 默认控制在一个主 Reference、一份辅助 Reference 和首轮 2～4 个源码/测试切片内。
- 必须超出时，在读取第二份辅助 Reference 前留下扩张凭据，不能结束后补写理由。
- 输出达到任务所需证据后停止，不继续枚举无关算法、测试目录或模型实现。
- 语义正确但读取未收敛，仍判失败，不能只按答案内容判通过。

先记录一次不修改规则的完整读取轨迹，确认是上述“同时读”指令、主手册锚点不足还是停止条件失效；再做最小文字修改。不要在原因未定位前继续给主 `SKILL.md` 增加新的预算段落。

### 22.8 建立统一 A/B 评分体系

当前有 Skill 的 3.5/10 来自附在 `test-skill.md` 后的 reviewer，无 Skill 的 2.5/10 是回顾性启发式估计，两者不是严格统一 rubric 的测量结果。后续比较必须使用同一评分表：

| 评分维度 | 权重 | 核心检查 |
| --- | ---: | --- |
| Skill 路由与代码体系 | 15 | 是否进入正确 workflow/new-path/classic/model/ops 体系 |
| 源码事实与活跃调用链 | 25 | 是否找到真实 caller、注册、分派和保存/消费链 |
| 未知契约处理 | 20 | 是否保留 `pending`、给条件分支且不擅自冻结 |
| 跨阶段闭环 | 20 | 输入、状态、保存、回载、eval/deploy 是否一致 |
| CLI/config 准确性 | 10 | 参数、choices、bit policy 和自然语言目标是否一致 |
| 验证与证据等级 | 10 | 是否区分源码、单测、集成和真实环境证据 |

增加硬性封顶规则：使用不存在的 CLI 参数、选择错误代码体系、把没有 caller 的 helper 当成活跃路径、静默混淆 FP16/BF16 或 weight-only/W4A4、以及把未知目标契约写成已冻结方案时，该题不得及格。篇幅、文件数量和细节丰富度不直接加分。

A/B 测试必须保持同一源码 commit、模型版本、reasoning 配置、原始 prompt、权限和超时设置。reviewer 不知道哪份答案来自有 Skill 环境，先独立评分再揭示分组，避免确认偏差。

### 22.9 测试资产与归档方式

每个用例分别保存：

1. 原始用户 prompt。
2. 被测 Agent 原始输出，不追加 reviewer 结论。
3. 实际读取的 Skill、Reference、源码和测试清单。
4. token/时间/中止原因等效率数据。
5. 独立 reviewer rubric、逐项得分和源码证据。
6. 源码 commit、模型版本、cold/warm 模式和运行环境。

prompt、回答和 review 必须分文件；被测副本移除历史测试答案、`amct_coder_analysis/` 和预期结论。中止的输出记录为效率失败或未完成，不能作为完整语义通过样本。相同题至少运行三次，避免把单次偶然行为当成稳定能力。

### 22.10 短回归题矩阵

先用短题验证单一门禁，再重跑六道综合题：

| 短题 | 主要断言 |
| --- | --- |
| 未知 dtype role | 现状与目标分离；未知目标始终为 `pending` |
| helper 存在但无 caller | 明确标为未接通，不声称集成闭环 |
| 无 trainable 参数的统计算法 | 识别 solver 提前返回并要求显式数据驱动/finalize |
| 复合 unit 使用 norm 输出 | 识别完整 block double norm 风险及原始输入/kwargs 契约 |
| FP16/BF16、weight-only/W4A4 | 自然语言目标与 CLI/bit policy 逐 role 对齐 |
| 模型 wrapper 放入 workflow | 首次路由否决越层，并进入 `model-adapter` |
| NPU kernel 放入 Python quant module | 首次路由到 `amct-ops-dev`，只保留上层契约 review |

短题稳定通过后再重跑原六题，随后扩展到尚未被充分覆盖的 `classic-graph`、`ops-boundary`、`deploy-export-code` 和 `public-api-tests`。不能用六道跨契约综合题代替每份 Reference 自身的正常、边界、未知契约和伪活跃链测试。

### 22.11 推荐实施批次

下一批只修改：

1. `dtype-quant-module.md`：加入 `pending`、现状/目标双列和禁止无证据状态转换的 BLOCKER。
2. `llm-algorithm.md`：把一次性“同时读两份”改为证据驱动的 owner 选择。
3. `ptq-solver-params.md`：同样消除同时加载两份辅助 Reference 的指令。
4. `llm-data-workflow.md`：把 public API/CLI 的辅助读取改为真实用户契约变化后按需加载。
5. `validation.md`：补统一 rubric、实际读取资料和效率数据归档要求。
6. `design-iteration.md`：记录实际修改、cold test 原始结果和裁决。

主 `SKILL.md` 在这一批保持不变。只有读取轨迹证明主入口预算或停止条件本身有问题，才单独修改；不要让子能力局部问题再次推动主文件扩张。算法生命周期、保存链、composite unit、路由和精度门禁已经有语义改善证据，本批不做无关重写。

### 22.12 阶段验收门

下一轮完成标准：

- dtype 通用未知契约题三个 cold 会话全部保留 `pending`，无参考实现偷渡。
- GPTQ 三个 cold 会话语义门全部通过，读取遵守默认预算；超预算时有事前扩张凭据。
- 所有答案不出现不存在的 CLI、错误代码体系、未接通 helper 闭环或精度/bit-policy 口径混淆。
- prompt、回答、读取清单和 reviewer 评分全部独立归档，A/B 使用同一 rubric 盲评。
- `quick_validate.py`、Markdown 链接检查、Reference 导航集合和 `git diff --check` 通过。

达到这些门后再评估主 `SKILL.md` 是否需要压缩，并开始覆盖其它子能力。未达到时继续围绕失败用例做最小修订，不新增大范围规则，也不以“平均分提高”掩盖仍存在的 P0 错误。

### 22.13 当前状态

§22.5～§22.12 在本节形成时仅为后续改进方案，尚未修改 `SKILL.md`、Reference、测试资产或评分脚本，也没有产生修改后的 cold forward-test 证据。后续 dtype 状态机的实际修改与 warm 对照结果见第 23 节。

## 23. 第十四轮审视：dtype 状态机 warm 对照与残余决策漂移

第 22 节发现新 dtype 的未知 role 会在规划后半段被 Agent 无证据地冻结，因此在 `dtype-quant-module.md` 中实施了“当前仓库事实 / 目标契约”双维状态模型，并增加 `pending` 状态、状态转换规则、消费面双列表和对应 BLOCKER。本轮使用新的 dtype 行为题检查该修改是否改变 Agent 的规划路线。

精炼结果见 [dtype-test.md](../dtype-test.md)，原始长回答归档于 [dtype-test-raw.md](../dtype-test-raw.md)。

### 23.1 测试范围与证据边界

本轮完成 5 组有 / 无 Skill 的成对只读规划题：

| 用例 | 主要变量 |
| --- | --- |
| D1 `nova-float` | 只有产品名称，其余目标契约基本未知 |
| D2 `orbit4` | 用户给出“复制 4 bit 实现即可自动接通”的错误暗示 |
| D3 `quark4` | weight-only 与 4 bit 已确认，编码、粒度和 deploy 未确认 |
| D4 `luma` | 只有 dtype 类、注册和数值单测，却声称已经完成接入 |
| D5 新格式 + NPU | dtype Python 语义与 NPU 算子实现的职责交界 |

本轮不是严格 cold A/B，证据限制如下：

- 有 Skill 回答能够看到或复用无 Skill 阶段的结论，属于同会话 warm 对比。
- 每题只有一次结果，没有证明行为可稳定复现。
- 没有统一记录模型版本、源码 commit、完整读取轨迹、token、运行时间和中止原因，不能评价读取预算。
- 原计划中用于验证“信息充分时能否正常收敛”的 `q6s` 未执行。
- 两组都没有修改代码或运行单测、集成、真实模型和 NPU 验证。

因此本节只能记录“规划行为方向性改善”，不能把 dtype 子能力标记为 cold 回归通过。

### 23.2 对照结果

| 用例 | 无 Skill 的主要路线 | 有 Skill 的主要路线 | 结果 |
| --- | --- | --- | --- |
| D1 只有名称 | 默认新增 dtype，提前确定命名、骨架和跨层重构 | 先裁决别名、classic 迁移或新语义，未知项保持 `pending` | 明显改善 |
| D2 错误复用 | 能否决复制方案，但仍建议先建骨架并做 CLI/deploy 重构 | 先比较数值语义，再决定复用、别名或新增；搜索命中不等于必改 | 明显改善 |
| D3 部分契约 | 采用已知 role/bit，但仍提前规划新文件、CLI 和 NPU | 已知项收口、未知项保持 `pending`，但默认建议修改注册名 | 有改善，仍有过早决策 |
| D4 注册即完成 | 能追到 import、CLI 和 deploy，但检查面较大 | 用当前事实 / 目标契约表反证“完成”，但 CLI 状态前后不一致 | 有改善，仍有状态漂移 |
| D5 dtype + NPU | 直接按 LLM dtype 与新算子路线展开 | 先裁决 LLM new-path / classic / 两者，再划分 dtype 与 ops owner | 明显改善 |

有 Skill 后没有观察到相对无 Skill 的明显退化。D1、D2、D5 的路线改善最清楚；D3、D4 说明状态模型已经减少无依据方案，但还没有完全约束状态与后续动作的一致性。

### 23.3 已确认的有效改进

1. **规划起点前移。** Agent 不再直接回答“如何新增 dtype”，而是先判断现有语义复用、公开别名、跨体系迁移和真正新增。
2. **现状与目标分离。** INT、MXFP、classic 或 experimental 的当前实现进入事实列，不再当然成为新 dtype 的 role、bit、粒度和 deploy 范围。
3. **消费面证据增强。** Agent 能稳定区分“类存在 / 注册成功”与 CLI、quantizer、blockwise/tensorwise deploy 实际接通。
4. **职责边界改善。** dtype + NPU 场景中，Python 参考语义和上层调用契约留在 `amct-coder`，kernel、C++ schema、Meta、构建和硬件验证进入 `amct-ops-dev`。
5. **证据等级更明确。** 有 Skill 回答普遍说明只完成源码核验，没有把未运行的测试、集成或硬件验证写成已通过。

因此，dtype 状态机不是无效扩写：它已经对 Agent 的规划行为产生了可见改善。

### 23.4 仍存在的问题

#### P0-A：目标状态与后续动作仍会分叉

D4 前文把“是否公开 CLI 值”标为 `pending`，后文却把“加入 CLI choice”列为可立即修改。当前规则限制了 `pending -> supported/rejected` 的文字状态转换，但没有同样限制依赖该状态的动作清单。

同类风险还包括：deploy 参与度为 `pending` 时直接规划 payload；NPU 需求为 `pending` 时直接创建 kernel/schema；新增或别名未裁决时直接创建 dtype 文件。

#### P0-B：仓库惯例仍可能被当成目标决策依据

D3 根据现有格式族命名惯例，建议把用户题面中的 `quark4` 注册名改成 `quark`。命名惯例可以形成候选方案和兼容性说明，但不能单独改变用户给出的产品名、CLI 值或 registry 名。当前状态机对 role/bit/粒度约束较强，对“产品名 / CLI 值 / registry key / Python 标识符”的分离还不够明确。

#### P1：候选骨架仍可能被写成可立即实施

D5 虽然正确停止了数值实现和 schema 决策，但仍把 ops 目录骨架描述成可立即确定。目录结构可引用现有模式作为候选；在代码体系和接口契约未冻结前，不应成为最终落点。

#### P1：效率改善没有证据

两种回答都偏长。本轮没有完整读取轨迹和 token 记录，不能判断是 Reference 读取过多、源码锚点扩张过多，还是最终输出没有收敛。该问题暂不推动主 `SKILL.md` 修改，先在下一轮 cold test 中记录轨迹。

### 23.5 下一步最小修改

下一步只修改 `references/dtype-quant-module.md`，不扩写主 `SKILL.md`：

1. **增加状态与动作一致性门。** 最终计划中的每个立即动作必须引用一个非 `pending` 的目标状态；依赖 `pending` 的内容只能进入条件分支或待确认清单。
2. **拆分命名契约。** 分别记录产品称谓、CLI 值、registry key、Python 文件/类标识符和可选 ops schema 名；仓库惯例只能支持候选映射，不能自动替用户选择或改名。
3. **增加高风险动作表。** 明确以下对应关系：

   | 目标项仍为 `pending` | 禁止写成立即动作 |
   | --- | --- |
   | 新增 / 别名 / 复用 | 创建唯一 dtype 类、文件或注册项 |
   | CLI 是否公开 | 修改 `choices`、help 或配置样例 |
   | deploy 参与范围 | 修改 payload、packing、index 或 `quantization_config` |
   | NPU 加速与 schema | 创建确定的 kernel、schema、Meta 或算子目录 |

4. **增加交付前一致性复核。** 输出前逐项比较“目标状态”和“动作清单”；没有新增证据时，前文 `pending` 的项目在落点、测试和实施顺序中也必须保持条件式表达。
5. **限制只读规划输出。** 默认只保留边界、关键当前事实、目标状态、条件动作和验证建议，不重复铺开全部搜索命中和完整实现清单。

这批修改只针对本轮可复现的 D3/D4/D5 问题，不改算法、solver、data workflow 或主入口读取预算。

### 23.6 修改后验收

1. 先补跑信息充分的 `q6s`，确认 Agent 能采用已冻结契约并收敛到最小第一阶段，而不是机械保留 `pending`。
2. 在物理隔离的新会话中重跑 D1、D3、D4，每题至少三次：
   - D1：未知 role、bit、粒度、CLI 和 deploy 全程保持 `pending`，不产生唯一文件或命名方案。
   - D3：采用已给出的 weight-only / 4 bit，不依据仓库惯例改写 `quark4`；未知编码、粒度和 deploy 保持条件式。
   - D4：能反证“注册即完成”，但 CLI/deploy 是否修改严格依赖目标参与范围，不从当前白名单事实自动推出必改动作。
3. 记录同一模型配置、源码 commit、原始 prompt、实际读取资料、token、时间和 reviewer 评分。
4. 语义正确且读取满足预算，或超预算前提供扩张凭据，才判该轮通过。
5. `git diff --check`、Markdown 链接检查和 `quick_validate.py` 继续作为结构验证，不替代 cold 行为证据。

### 23.7 当前状态

- dtype 目标契约状态机已经实施。
- 5 组 warm 对照观察到明确方向性改善。
- D3 命名决策和 D4 CLI 动作仍暴露状态一致性缺口。
- 本节形成时只记录结果与下一步方案，尚未实施 §23.5 的第二轮修改；后续实际修改见第 24 节，修改后的 cold 验收仍未完成。

## 24. 第十五轮实施：dtype 动作一致性与算法读取收敛

本轮先按第 23 节的 dtype warm 对照修复残余决策漂移，再按第 21、22 节的 BATQuant、敏感层、GPTQ 和 cold 读取轨迹修正算法子能力。修改限定在两份现有 Reference，不改主 `SKILL.md`，也不新增脚本或 Reference。

### 24.1 dtype 第二轮修改

`references/dtype-quant-module.md` 增加以下约束：

1. 产品称谓、CLI 值、registry key、Python 标识符和 ops schema 分别作为目标契约记录，仓库命名惯例只能形成候选映射，不能自动改写用户名称。
2. 每个改变实现、公开契约或交付产物的立即动作必须回指一个非 `pending` 的目标状态；新增/别名/复用、CLI、deploy 和 NPU 任一目标仍为 `pending` 时，对应文件、公开值、payload 或 schema 只能写成条件分支。源码核验和现状测试不受此限制。
3. 分层实施顺序改为仅对 `supported` 项成立，避免步骤清单把前文未知契约暗中收口。
4. 交付前比较目标状态与动作、落点、实施顺序和测试清单，并压缩只读规划输出。

该修改直接对应 D3 根据命名惯例改写 `quark4`、D4 CLI 状态与立即动作冲突，以及 D5 接口未冻结却提前确定 ops 骨架的问题。

### 24.2 llm-algorithm 修改

`references/llm-algorithm.md` 保留第 21 节已经补齐的 trainable、observe-only、observe-then-finalize、敏感度决策和保存回载语义，本轮不重复扩写算法生命周期，只修正历史测试仍暴露的执行问题：

1. 增加可查事实门：registry import、`targets`、构造签名、quantizer hook、trainable/observe/finalize caller、保存 caller 和 CLI 参数必须先从源码与直接测试核对，不能继续留作开放问题。
2. helper 或 hook 找不到 caller 时标记为“源码存在但未接通”，不得推导 solver 驱动、递归保存或 eval/deploy 闭环。
3. 删除“改变 real quant 就同时读取 dtype 和 deploy”的无条件扩张，按 dtype、solver、workflow、deploy、public API 的未解决契约选择一份 owner Reference。
4. 第二份辅助 Reference 只有在第一份和新增源码证据仍不足时加载，并在加载前记录未解决契约、已查证据、owner 和超预算原因。
5. 增加停止条件与精简输出要求，避免 GPTQ 语义已经闭环后继续枚举无关 Reference、算法和测试目录。

该修改对应 BATQuant 中可查事实仍被留待确认、敏感层中 skip 保存/消费边界未按 owner 收口、GPTQ 中 helper/caller 容易混淆，以及 cold 测试一次读取 7 份 Reference 的问题。

### 24.3 验证边界与下一步

本轮只修改 Skill Reference 和设计记录，不改变 AMCT 运行时代码，因此不运行 Python/C++ 单测。已完成以下静态验证：

- `quick_validate.py .agents/skills/amct-coder` 输出 `Skill is valid!`。
- 三个修改文件通过 `git diff --check`。
- 主 `SKILL.md` 的 Reference 导航去重后仍为 17 份，与 `references/*.md` 实际集合一致；本轮没有新增导航项。
- 两份修改后的 Reference 及第 23 节涉及的相对链接目标均存在。

静态验证只能证明文档结构成立，不能证明 Agent 已稳定遵循新规则。下一步行为回归分两组：

- dtype：补跑信息充分的 `q6s`，再隔离重跑 D1、D3、D4，检查未知状态是否贯穿动作清单和命名映射。
- algorithm：隔离重跑 GPTQ observe-then-finalize、dtype real quant + deploy、敏感层自动跳层三类题，记录实际读取资料；语义正确但无扩张凭据地超过默认预算仍判失败。

在 cold 回归完成前，dtype 只能记为“warm 对照后完成第二轮规则修订”，算法只能记为“根据既有失败完成静态修订”，都不能标记为行为验收通过。
