# 验证规则：validation

实际编码在修改后读取本文件并执行匹配的最小验证；实现规划、行为诊断和各类 review 只在判断验证方案或既有验证证据时读取相关部分。

## 通用

交付说明该包含哪些内容，以 [SKILL.md「交付格式」](../SKILL.md#交付格式) 为唯一出处，本节不复制清单。本文件只约束其中「已跑/未跑验证」两项的判定标准：

- 证据写到哪一级，按下方「证据等级」表述；不得用低级证据替代高级结论。
- 缺 NPU、CANN、模型权重、外网数据集、protobuf 生成物时，不要声称已完成对应验证；按「验证失败的处理」的「环境缺失」行写成「X 未执行：缺 Y」，不得写成已验证或跳过。

## 证据等级

结论按下面等级分别表述，不能互相替代：

1. **源码存在**：实现或注册能在源码中定位。
2. **单测覆盖**：测试直接断言了目标行为；只测 wrapper 分派不等于覆盖 `PtqUnit` 或 PTQ I/O。
3. **集成闭环**：输入、GT、保存、回载等跨模块路径实际跑通。
4. **真实环境验证**：真实模型、数据集、NPU/CANN 或目标硬件结果已取得。

计划中的测试写为“建议运行”或“未运行”，不能表述为已经覆盖。「唯一根因」的判定条件是完整相关链路已被证据排除（禁令本身在 [SKILL.md「全程标记纪律」](../SKILL.md#全程标记纪律)，本节只给判定条件）。

## 最小静态验证

- Python 语法：`python -m py_compile <files>`。
- Python 风格检查：对本次改动文件运行 `ruff check <changed files>` 和 `ruff format --check <changed files>`，只验证、不改文件。
- Python 风格修复：确认需要自动修复时运行 `ruff check --fix <changed files>` 和 `ruff format <changed files>`；对应 `.pre-commit-config.yaml` 的 `ruff-check --fix` / `ruff-format` hook，格式选项来自 `pyproject.toml` 的 `[tool.ruff.format]`。
- pre-commit：环境已就绪时运行 `pre-commit run --files <changed files>`（**不要用 `--all-files`**，避免全仓存量问题掩盖本次改动信号）。未安装 pre-commit 时**不主动安装**——安装有网络、环境和权限副作用；在交付说明中报告「pre-commit 未执行：环境未就绪」即可。仅当用户明确要求完整提交前检查时，经确认后再安装（`python -m pip install pre-commit` + `pre-commit install`）。
- 版权：检查新增代码包含仓库既有的 Apache 2.0 版权声明，现有版权头未被删除或破坏。
- public API：`pytest tests/unit_test/test_public_api.py`（其中 ast 解析的静态检查无需构建环境，import 行为用例在未构建源码树时自动跳过）。

本节只定义修改后的检查动作。代码风格和版权要求必须已在修改前按 [coding-rules.md](coding-rules.md) 建立约束，不能等运行本节时才首次得知。

## 验证失败的处理

验证失败按类型分派，处理顺序固定：先判断失败属于哪类，再按对应动作执行，不跳类。

| 失败类型 | 判定 | 处理 | 交付措辞 |
| --- | --- | --- | --- |
| 静态检查失败（py_compile/ruff） | 失败点在本次改动文件内 | 修复后重跑直至通过；ruff 的存量告警（stash 对照确认非本次引入）不修，如实引用 | 「静态检查通过；X 为存量问题（对照 HEAD 确认）」 |
| 本次改动引入的单测失败 | 失败用例断言的是本次改动目标或其直接下游 | 修复实现或测试（实现错误改实现，断言过时改断言并说明），重跑至绿 | 「单测通过；修正了 X（原因）」 |
| 与本次改动无关的单测失败 | 失败用例与改动面无调用关系 | 不顺手修（范围控制，另开 issue）；验证证据标注该失败与本改动的关系判断依据 | 「本次改动相关用例全过；Y 失败与本改动无关（依据：调用链不相交），已单独记录」 |
| 环境缺失（NPU/CANN/权重/数据集/pre-commit） | 验证目标需要外部资源 | 不假装已验证，不阻塞可执行部分；能 mock 的走单测级 | 「X 未执行：缺 Y」——不得写成已验证或跳过 |
| 间歇性失败（flake） | 同一用例重跑结果不一致 | 重跑 2~3 次确认；仍复现按无关失败或本次引入分类 | 标注重跑次数与结果分布 |

任何失败都不改变证据等级表述：修复后通过写「单测覆盖」，不写「曾失败现已修复」作为降级理由。

## 按变更类型选验证

先按本次实际改动选择最小验证，再叠加场景测试索引。不要为了看起来完整而运行无关大集合，也不要用低一级证据替代高一级结论。

| 变更类型 | 最小验证 | 说明 |
| --- | --- | --- |
| 只改 `SKILL.md` 或 reference | 自检脚本 `python .agents/skills/amct-coder/scripts/check_skill.py`（链接+锚点+节名引用+快照符号四项，替代手工检查）；`git diff --check`；修改「当前接通事实」「主链状态」等快照时，确认脚本中该条核对命令通过；skill 结构校验（frontmatter、命名）用 skill-creator 的 `quick_validate.py`，脚本随 skill-creator skill 分发，不在本仓 | 路由语义或场景规则明显变化时跑 `evals/` 题集：`routing.json`（路径选择与委派）+ `behavior.json`（证据措辞、升级、询问形态、范围控制）。每题带 rubric，按下方「forward-test 执行方式」判定 |
| docs/examples 纯措辞 | `git diff --check`，必要时检查链接和示例命令是否仍存在 | 不声称运行代码或覆盖实现行为 |
| docs/examples 新增 API/CLI 承诺 | 对应实现 owner 的单测 + 公共接口与打包检查 | 文档承诺必须能被源码和测试支撑 |
| Python 局部实现 | `python -m py_compile <changed .py>` + 命中场景单测 | 若改 public API、workflow 或 solver，再叠加对应集成测试 |
| CLI/parser/default 行为 | parser 测试、命令 help 或最小 workflow dry-run | 默认值变化按兼容性风险说明 |
| 保存/加载/文件格式 | producer 和 consumer 双侧单测或 smoke | 只测保存不等于回载闭环 |
| 构建、packaging、ops 边界 | 对应 build/import 测试；ops 内部实现按 `amct-ops-dev` | 缺 CANN/NPU/protobuf 时明确阻塞 |

### 自造词准入（只在改 skill 文案时适用）

本 skill 不设术语表：自造词一律在使用点就地解释，定义只有一份，改一处即生效，不存在两处定义漂移。要引入一个新自造词，三条全中才立，否则直接写白话：

1. **跨文件术语**：≥3 个文件引用同一概念；单文件词汇就地展开。
2. **白话不更含糊**：白话版只是更长、并不更含糊时，直接写白话，不立条。
3. **不撞既有词**：与本仓领域词、代码符号，以及读者从行业带进来的既有词**不同形也不近形**（含跨语言标准译名、含同形异义）。`rg` 只查得出前两类；第三类在仓内零命中却照样撞车，只能自己想一遍。撞了必须在**使用点**就地覆盖语义，不能只写在别处。

领域词与代码符号（`PtqUnit`、granularity、`export_deploy`、bit policy 等）一律不另立名，以源码为准——抄一份只会造出第二份会腐化的真相。

**已留用的跨文件术语**（定义在使用点，此处只登记去向、不复述内容）：主手册 → [process-rules「Reference 读取」](process-rules.md#reference-读取)；门禁对象 → [process-rules「契约核对的触发条件」](process-rules.md#契约核对的触发条件)①；契约状态表 → [quant-dtype-module「契约状态表」](quant-dtype-module.md#契约状态表)；快照／核对命令 → [process-rules「快照与核对命令纪律」](process-rules.md#快照与核对命令纪律)；事实纪律 → [SKILL.md「事实纪律」](../SKILL.md#事实纪律q1q2q3)；禁止项的语义覆盖 → [SKILL.md「禁止项」](../SKILL.md#禁止项)；禁止项完整清单 → [process-rules「全局禁止项」](process-rules.md#全局禁止项)。

**已试过并废弃的自造词**（不要再当术语用，也不要重新造回来；括号内为废弃原因）：敏感层（撞本仓产品术语量化敏感层：`README.md`「回退敏感层」、`ROADMAP_2026Q4.md`「LLM 量化敏感度分析」在建能力；`llm-algorithm.md` 同词用领域义，一词两义已实际发生）→ 改写「共用架构层」，五层枚举内联在 SKILL.md「交付格式」（定义在首次使用点，别处只回指不重述）；契约状态模型（「模型」在本仓指神经网络模型：`模型` 1448 处 / `model` 12614 处）与门禁表（与前者二名一物）→ 均收敛为「契约状态表」；收束（与领域侧「收敛」近形且同仓并存）→「停止扩张、开始交付」；落点层 →「（改动落在哪一）架构层」；契约缺失形态 →「契约缺失的处理形态」；双端验证 →「producer/consumer 两端核对」；消费面 →「下游接入点」。后四条按第 2 条不立：白话只是更短，并不更含糊。第二轮又废弃四条（都是入口级高频词，看不懂就直接卡住流程）：BLOCKER（撞读者行业先验且**一名反义**：issue tracker 里 blocker＝阻塞发布/任务挂住，本 skill 的定义恰是「只停错误方案不停任务」；`amct_pytorch/`·`docs/`·`tests/`·`examples/` rg 零命中，第 3 条前两类查得出、这一类查不出）→「禁止项」，命中即该方案不可用、换方案继续；直达卡（「卡」是装饰性容器隐喻，正文本就写「四步直达」，读者要先解码隐喻才拿到信息，按第 2 条不立）→「四步直达」（该词第三轮又废，见下）；执行形态走查（**false friend，比生词更误导**：「执行形态」是 `llm-algorithm.md`「算法执行形态与阶段状态机」已定义的领域词，指算法属训练型还是统计型，而该文件示范的是全流程如何停止扩张与交付，与执行形态无关；且标题单文件出现，不过第 1 条）→「走查示范」；句柄（与本 skill `coding-rules.md`「临时文件/目录、句柄和 device 内存」的 CS 本义同形异义，一个 skill 内一词两义；且 2 文件不过第 1 条）→「跨文件术语」。第三轮废弃一条：四步直达（标题级裸词——「直达」没说跳过的是什么，读者要读完整节才能解码）→ 四步本身折进 SKILL.md「执行顺序」的第 3~4 步，信息不丢。第四轮废弃三条（都是「两条流程」框架的产物，这一轮把框架本身拆了）：快速路径／完整流程（入口为说清「什么时候能少读」而造出一对流程名，代价是 SKILL.md 必须抄一份「禁止项」摘要副本、evals 要同时考两条路、读者动手前先得分辨自己在哪条路上；换成「读取范围」——判据①~④ 全中就按表读，不全中或有疑就按 process-rules 节索引读——省读文件的收益全留，副本与双路分辨全消）；单向升级（同一条补读规则在 SKILL.md 与 process-rules「仓库边界」各写一遍，两处措辞已漂移）→「判错了就补读」（该条第五轮又废，见下）。第五轮把这道门整个拆掉，废弃四条：读取范围／判据①~④／判据全中／判错了就补读（第四轮只把「快速路径」改名成「读取范围」，支路还在——入口仍要先判一道门，门本身要 20 行判据加一张表来描述，读者动手前先得判门，还多出一条「改判据必须与契约核对触发条件同轴修改」的长期约束；删掉支路后所有任务走同一条流程，读多少资料由 process-rules「读取预算与停止条件」的停止扩张条件收口，防预读的效果一样，同轴约束消失）→ 共用架构层五层枚举移入 SKILL.md「交付格式」，门禁对象定义移入 process-rules「契约核对的触发条件」①，跨契约的两端解释并入同节②；「判错了就补读」不再需要（本来就该按流程读），其内容由 process-rules「仓库边界」的「结论涉及实现行为/落点/调用链/契约就必须回本节补齐」覆盖。第六轮把路由阶梯整个移出 SKILL.md，废弃一节两名：职责归属（SKILL.md 节名）／职责归属细则（process-rules 节名）。那四级阶梯（用户点名 > 拿已发现 skill 的 `name + description` 对一遍 > 去 `.agents/skills/` 补查 > 兜底自己做）写的是**选哪个 skill**，而选 skill 发生在 agent 读到 SKILL.md 之前、依据 frontmatter description（`.agents/docs/architecture.md`「能力 + 触发：见 frontmatter `description`（机读接口）」），body 里再排一遍等于让 skill 重做已经做完的路由。其中第一级「用户明确指定了 skill，就按用户的目标执行」在本 skill 已被加载时恒为空转：点名别的 skill 时本文件根本不会被读到，点名本 skill 时它就是默认行为；唯一可能有作用的场景——点名本 skill 但活的实质归别处——恰好被它写成歧义（读成「留下」与 frontmatter「不用于量化实验执行」矛盾，读成「该转就转」又与第二级重复）。留下的只有进来之后才可能发生的两件事 → SKILL.md「不归本 skill 的部分」（发现实质归别处就实际加载对应 skill、不只口头委派；混合任务按动作拆开，源码级部分与共享契约由本 skill 收口；该节第七轮又废，见下）；实现代码与运行命令的分界、落点 review 区分 → process-rules「动作分界与混合任务」。同轮消掉三处副本与一处无守护引用：SKILL.md／process-rules／delegation 各写一遍的「按 name+description 判职责」收成 delegation 一句「本文件不参与路由」；「用户指定 > 专门 skill description > 本 skill 流程」这条优先级删除；delegation 里按序数指的「职责归属」②改成按节名指（序数引用 `check_skill.py` 查不出，删一条就会静默错位）。第七轮把第六轮留下的那一节也删掉，废弃一节：不归本 skill 的部分。它是**负空间描述**——通篇在说「本 skill 不做什么、什么不归我」，而 skill body 的读者是已经被 frontmatter description 选中、正在动手的 agent：路由决策早已完成，此处再申辩一遍边界只增加读文件的成本，不改变行为；真要声明适用范围，唯一有效位置是 frontmatter description（机读接口，`.agents/docs/architecture.md`「能力 + 触发：见 frontmatter `description`」），而该处已写了「不用于量化实验执行、精度/算法收益判读、模型结构适配、classic/graph_based 或 NPU 算子实现」。逐条核对后确认整节零信息丢失：「实际加载、不只口头委派」与「混合任务只对共享签名/数据格式/保存加载/验证证据收口」→ `delegation.md`「继续执行与询问」末段本就有原文；「按动作拆开、不看 eval/deploy/attention/校准 关键词」→ `process-rules`「动作分界与混合任务」本就有原文；「会话里找不到对应 skill 就去 `.agents/skills/` 补查」与「名字撞车/查不确认不了就按本 skill 流程做并在交付说明点明重叠」→ 移入 `delegation.md`「使用原则」（这两条本来就是「已经进来之后」的处置，属 delegation 的职责，而非入口级无条件约束）；仓库级 skill 清单另由 `.agents/README.md` 目录树覆盖。同轮清掉两处指向该节的引用：`process-rules`「动作分界与混合任务」与 `delegation`「使用原则」，两处均只改指 delegation 本身，不再回指 SKILL.md 节名。

## forward-test 执行方式

`evals/` 下的题集验证的是「agent 读了本 skill 会不会做出正确判断」，**需要 agent 实际作答，不能由脚本自动判定**——`check_skill.py` 只查链接、锚点、节名引用和快照事实，不跑题。

执行步骤：

1. 开一个不加载本 skill 上下文的新会话，把 `evals/*.json` 中某题的 `prompt` 原样作为用户请求发进去。
2. 让 agent 按 skill 正常流程作答（路由题只需判走哪条流程并说明理由；行为题需要给出它会交付的说明）。
3. 对照该题 `rubric` 判定，三个键都按**语义**判、不做字面子串匹配：
   - `must_mention`：每个子数组至少命中一项，同义表述算命中。
   - `must_not_claim`：任一项的语义出现即失败；答案**否定**该说法时不算命中（「这不用叠加契约核对」不触「契约核对」）。
   - `allow`：看似违反 `must_not_claim`、实则被本 skill 规则要求的对冲表述，命中不扣分。
4. 失败时先分辨三种原因，处置完全不同：skill 文本没说清 → 改 skill；agent 没读到 → 查加载路径与 description；**rubric 与 skill 规则冲突 → 改 rubric，不是改答案**。第三种最容易漏，见下方纪律。

### 出题纪律（违反会把正确答案判成失败）

1. **rubric 判行为，不判复述。** 不得要求答案出现规则名或工具名（「一次问完」「用 rg 搜」）——沉默地做对也是做对。`must_mention` 只放「正确答案不可能不说到」的事实性内容。
2. **`must_not_claim` 不放裸状态词。** 「契约核对」「门禁对象」会字面命中 skill 自己要求的对冲（口径/实现分辨、按触发条件判要不要核对）。要写就写成完整断言（「为改一行文案整读 repo-map 与主手册全文」）。
3. **skill 要求的对冲必须进 `allow`。** 执行中发现改动面变大或触到契约就回 process-rules「仓库边界」「契约核对的触发条件」补齐资料、`delegation.md` 的先分辨再分流、禁止项「只停错误方案不停任务」的继续推进——这些都是合规答案的组成部分，不豁免就必然误判。
4. **环境前提写进 `prompt`，不写进 `rubric`。** 判分器不知道被测 agent 跑在什么机器上；「缺 NPU 时不得声称已验证」这类题必须把「本机没有 NPU」钉在题面里，否则在有 NPU 的机器上真跑通并如实上报的答案会被判失败。
5. **`expected` 里的仓库事实必须核实，且不写会漂移的形态。** 题目嵌入的源码断言（某 helper 无 caller、某默认值指向会 raise 的分支）一旦过期，就会拿错误事实当判分依据，比没有 rubric 更糟。三条约束：① **不写行号**——按 `process-rules.md`「只锚定稳定符号」，改用符号描述（写「granularity 分支分派到 `_run_modelwise()`」，不写 `llm_ptq.py:86`）；② **不复制可枚举计数**（「三处调用」），改为指向手册快照的运行时查询；③ 核对命令以场景手册「主链状态」为唯一出处，题目引用而不复制。第 ③ 条是硬约束而非风格偏好：`check_skill.py` 只扫 `*.md`，**题集不在其快照判定范围内**，写进 `expected` 的事实没有任何机器守护，只能靠人工——所以题目一律不承载事实副本，改指向手册里那条被机器守住的核对命令。事实变了要连 `expected` 与 `rubric` 一起改。
6. **题目不是第二份规则来源。** `expected` 写判断理由（人读），`rubric` 写可判定的语义点（评审读）；两者与 skill 正文不一致时，以正文为准并同步修正题目。若一道题暴露出正文自相矛盾（例如某变更同时命中门禁对象与单点契约豁免），先修正文，再让题目回归为对该修正的回归测试。

## 越界自查

改完按实际调用关系逐个确认改动文件所在层、是否越层以及上下游契约是否闭环。四个高频问题：

1. 改动是否停在声明的架构层，没有顺手改上下游？
2. 是否先复用了现有抽象（`QuantLinear` / `QuantGatedMLP` / family helper）再新增？
3. 关闭量化时原浮点行为是否保住？
4. 是否有没必要的框架级重构或 solver 改动？

## 公共接口与打包检查

用户可见改动需要同时核对源码实现、入口转发、安装包和文档示例：

- 公共 API：分别检查普通 import、顶层属性访问、通配符 import、源模块 `__all__` 和顶层 `__all__`；懒加载 API 不能只证明属性可达。
- CLI：核对 parser 参数、默认值、choices、帮助文本和 `python -m` 入口是否与实际 registry/workflow 一致。
- 打包：按改动面检查——运行依赖变更查 `requirements.txt`，打包变更查 `setup.py`、`pyproject.toml`、`build.sh`、`package_data` 和 `AMCT_EXPERIMENTAL` 开关；不提交构建产物或 generated staging。
- 配置口径：文档中的 BF16、weight-only、W4A4、dtype 和 bit policy 必须与 parser 和实际每个 role 的配置一致。
- 文档示例：新增 API、CLI 参数、dtype 或 workflow 行为时同步更新，并确认示例命令在当前 parser 中存在。

## 场景测试索引

这里集中维护测试定位提示与命令；场景手册只描述需要验证的行为。先按实现符号、导入关系或目标行为在 `tests/` 中搜索直接断言，确认文件仍存在并覆盖本次改动，再运行对应测试。下表的目录是搜索范围，不要求整目录执行；文件改名后以搜索结果为准。

通用执行方式：`pytest <已确认的测试文件>`；需要缩小范围时使用 `pytest <已确认的测试文件> -k <相关测试名>`。新增行为没有直接断言时，在所属测试模块补覆盖，不把邻近测试通过当成该行为已验证。

| 场景操作手册 | 当前测试定位提示 |
| --- | --- |
| `quant-dtype-module` | `tests/unit_test/quantization/test_dtypes.py`、`tests/unit_test/quantization/modules/`；按 dtype/quantizer/BitPolicy 符号定位 |
| `llm-algorithm` | `tests/unit_test/algorithms/`；注册参考 `test_algorithms_registry_factory.py`，分流参考 `tests/unit_test/quantization/modules/test_quant_base.py` |
| `model-adapter-review` | `tests/unit_test/common/models/llm/`；按目标 adapter、wrapper 或 `PtqUnit` 符号定位，跨层行为再查 mocked workflow 测试 |
| `llm-cli-dataflow` | `tests/unit_test/workflows/`、`tests/unit_test/common/datasets/`、`tests/unit_test/common/evaluate/`；按 workflow、provider/I/O、PPL 行为定位 |
| `llm-solving-lifecycle` | `tests/unit_test/common/optimization/`、`tests/unit_test/common/models/llm/common/test_ptq_params.py`、`tests/unit_test/workflows/test_llm_ptq.py` |
| `llm-deploy` | workflow：`tests/unit_test/workflows/test_llm_deploy.py`；helper：`tests/unit_test/common/models/llm/common/test_deploy_export.py` |
| `amct-ops-dev` 上层接入 | 上层 fallback/import 参考 `tests/amct_pytorch/test_hifloat8_fallback.py`；算子硬件测试在 `tests/amct_ops/`，按 `amct-ops-dev` 执行 |
| 公共接口与打包 | `tests/unit_test/test_public_api.py`、`tests/unit_test/test_packaging.py`、`tests/unit_test/cli/test_llm_module_entrypoints.py` |

表中目录是搜索范围，不要求整目录执行；同一文件被多个场景引用时（如 `test_llm_ptq.py` 同时覆盖 `llm-cli-dataflow` 的 ptq 命令和 `llm-solving-lifecycle` 的求解链路）按本次改动选择相关用例，命令详见下节。

常用定向命令，按实际改动选取；与上表重复的文件不再列出第二份清单，以实际改动所属场景和搜索结果为准：

- dtype 与 quantizer 分流：`pytest tests/unit_test/quantization/test_dtypes.py tests/unit_test/quantization/modules/test_quant_base.py`。
- workflow：`pytest tests/unit_test/workflows/test_llm_eval.py tests/unit_test/workflows/test_llm_extract_ptq_data.py tests/unit_test/workflows/test_llm_ptq.py`，只涉及其中一个命令时只选该文件。
- 数据保存与加载：`pytest tests/unit_test/common/datasets/test_ptq_io.py tests/unit_test/common/datasets/test_ptq_provider.py`。
- PPL：`pytest tests/unit_test/common/evaluate/test_eval_ppl.py`。
- solver 与 factory：`pytest tests/unit_test/common/optimization/test_base_solver.py tests/unit_test/common/optimization/test_blockwise_solver.py tests/unit_test/common/optimization/test_factory.py`。
- 导出：`pytest tests/unit_test/workflows/test_llm_deploy.py tests/unit_test/common/models/llm/common/test_deploy_export.py`。
- public API / packaging：`pytest tests/unit_test/test_public_api.py tests/unit_test/test_packaging.py`。
- CLI 入口：`pytest tests/unit_test/cli/test_llm_module_entrypoints.py`；parser/default 行为还需确认有相应断言。

算法自身、classic custom algorithm/config、graph pass、quantize/save、prune/distill/retrain/cali 等按符号与领域定位；不维护逐算法或逐 pass 的文件清单。跨契约任务按生产者和消费者补充上述验证，测试直接 import 未接通模块不证明正常注册入口已接通。

## 链接检查

skill 文件内的仓库相对链接（reference、repo-map、casebook、README 等相对路径）必须实际存在，带锚点的链接（`file.md#section`）还要确认目标文件含对应标题（GitHub 风格归一化）。散文里写成「节名」的引用没有锚点可查，同样必须与实际标题精确一致——近似而不一致即判漂移（豁免项在脚本 `QUOTED_WHITELIST` 里逐条写理由）。

用自检脚本执行，**不在此维护脚本副本**——脚本逻辑只有一份，改一处即可：

```bash
python .agents/skills/amct-coder/scripts/check_skill.py
```

它一次性覆盖四项：路径存在性、锚点可达性、散文节名引用与标题的一致性、快照核对命令的断言判定。无输出即通过（exit 0）；失败时逐项打印。

外链（`https://`）不在本仓维护范围内；纯锚点链接（`#section`，同文件内跳转）由脚本按同文件标题查。脚本自身的判定逻辑用 `--self-test` 验证（含负向用例）。

## 构建

- 查看选项：`bash build.sh --help`。
- 构建 amct_pytorch 包：`bash build.sh --torch`。
- 完整包：`bash build.sh --pkg`。
- 单元测试构建：`bash build.sh -u`。
- `bash build.sh` 不带构建选项不会产分发包。

`amct_ops` 构建走 `cd amct_ops && bash ops_build.sh [--soc <soc>] [<op>]`。
