# dtype 子能力有无 Skill 规划行为对比

## 1. 测试目的与范围

本轮关注 Agent 面对 dtype 任务时的规划行为，而不是最终数据流写得是否完整。主要观察职责判断、读取顺序、仓库事实核验、未知契约处理、消费面检查、上下文收敛和证据等级。

本文件整理了 5 组有 / 无 `amct-coder` 的成对回答：

| 用例 | 主要变量 |
| --- | --- |
| D1 `nova-float` | 只有产品名称，目标契约基本未知 |
| D2 `orbit4` | 用户给出“复制 4 bit 实现即可自动接通”的错误暗示 |
| D3 `quark4` | role 与 bit 已知，其余契约未知 |
| D4 `luma` | 已有 dtype 类、注册和数值单测，声称接入完成 |
| D5 新格式 + NPU | dtype 与 NPU 算子的职责交界 |

原计划中的“信息充分时正常收敛”用例 `q6s` 未出现在本次结果中，因此尚不能证明 Skill 在契约充分时不会机械地继续保留 `pending` 或扩大检查面。

## 2. 证据边界

本轮可以用于观察方向，但不是严格的 cold A/B：

- 有 Skill 回答多次直接比较“无 Skill 版本”，并注明复用本会话已确认事实，说明两种情况没有物理隔离。
- 每题目前各只有一组回答，不能证明行为稳定复现。
- 文档没有统一记录模型版本、源码 commit、完整工具读取轨迹、token 和运行时间，不能评价上下文效率。
- 两组都只做只读规划，没有运行单测、集成流程、真实模型或 NPU 验证。

因此本轮结论表述为“有 Skill 的规划行为方向性改善”，不能表述为“dtype 子能力已经通过 cold 回归”。

## 3. 两种情况精炼总结

### 3.1 无 Skill

典型处理路线：

```text
题面关键词
  -> 全仓搜索相似 dtype 和硬编码分支
  -> 发现 CLI/deploy/algorithm/model 侧缺口
  -> 默认任务需要新增 dtype
  -> 给出骨架、全局重构和端到端完成清单
```

有效行为：

- 能查到 `register_dtype()`、CLI 白名单、bit policy、quantizer 和 deploy 静默错分等真实问题。
- D2、D4 能明确否决“注册成功等于框架接入完成”。
- 能区分响亮失败和静默错误，并说明未运行测试。

主要偏差：

- 容易把“新增 dtype”当作既定前提，未先判断零改动、别名、跨体系迁移或真正新增四种可能。
- 在数值语义、role 和 deploy 范围未冻结时，就确定文件名、注册键、算子名、骨架和全局去硬编码重构。
- 把搜索命中的 algorithm、model、deploy 分支倾向性地列为必改面，修改范围容易越过 dtype 主职责。
- D5 未先判断 LLM new-path 与 classic helper 两种接入形态，就直接按 `DTYPE_REGISTRY + CLI + deploy + amct_ops` 展开。

### 3.2 有 Skill

典型处理路线：

```text
判断职责与代码体系
  -> 读取 repo-map 最小边界和 dtype 主手册
  -> 先过既有语义复用门
  -> 分离当前仓库事实与目标契约
  -> 对未知项保持 pending 并给条件分支
  -> 只对真实成立的消费面继续取证或分流
```

有效行为：

- D1 先判断 `nova-float` 是否真是新 dtype，再分成别名、classic 迁移和新语义三条路线。
- D2、D4 使用 registry、CLI、bit policy、quantizer、blockwise/tensorwise deploy 等消费面逐项反证错误方案。
- D3 能采用已确认的 weight-only 与 4 bit，同时让数值编码、粒度和 deploy 范围保持未决。
- D5 识别出 LLM `DTYPE_REGISTRY` 与 classic HiFloat helper 两种接入形态，并把 kernel/schema/Meta/build 路由给 `amct-ops-dev`。
- 回答普遍区分源码存在、未运行测试、集成未闭环和真实环境未验证。

仍有问题：

- 输出仍然偏长，读取和事实枚举较多；缺少完整工具轨迹，无法确认是否满足一个主 Reference、一个辅助 Reference 和首轮 2～4 个源码锚点的预算。
- D3 根据格式族命名惯例建议把注册名从 `quark4` 改成 `quark`，证据不足以把该建议升级为默认选择。
- D4 一边把 CLI 是否公开标为 `pending`，一边又把“CLI choices 加入新值”列为可立即修改，目标状态前后不完全一致。
- D5 仍提前认为 ops 目录骨架可以确定；在集成形态、schema 和数值契约未冻结时，这最多只能作为候选布局。

## 4. 逐题对比

| 用例 | 无 Skill 的规划行为 | 有 Skill 的规划行为 | 对比判断 |
| --- | --- | --- | --- |
| D1 只有名称 | 承认数值未知，但仍确定命名映射、dtype 骨架、NPU 名称和多处全局重构 | 先判断是否需要新 dtype，未知 role/bit/粒度/deploy 保持未决，按三种契约分支规划 | 明显改善；状态机方向有效 |
| D2 错误复用暗示 | 正确否决复制方案并找到 CLI/deploy 缺口，但仍建议先建骨架和做跨层去硬编码重构 | 先比较数值语义，再区分别名、复用和新增；搜索命中只作为兼容性信号 | 明显改善；复用门和消费面门有效 |
| D3 部分契约 | 能识别 weight-only、4 bit 和若干未知项，但仍预设新增文件、CLI 和可选 NPU 路径 | 已知项正常采用，未知项保持 `pending`，补出 `is_act=True` 构造和 16 bit 直通约束 | 有改善；存在注册名默认选择过早的问题 |
| D4 注册即完成 | 正确追到 import、CLI 和 deploy 活跃链，但混入未由题面触发的 NPU 和较大范围清单 | 用当前事实 / 目标契约表反证“完成”，证据等级更清楚 | 有改善；CLI 是否立即公开仍有状态不一致 |
| D5 dtype + NPU | 直接按 LLM dtype 和新算子展开，并提前锁定命名、schema/Meta 骨架与通用重构 | 先裁决 LLM/classic/两者，再划分 Python 契约、ops 实现和 deploy owner | 明显改善；跨 Skill 职责边界有效 |

综合看，有 Skill 的 5 题都没有出现相对无 Skill 的明显退化，其中 D1、D2、D5 的路线改善最清楚，D3、D4 仍有局部契约被提前收口的问题。

## 5. 当前结论

本轮支持以下结论：

1. dtype 状态机已经改变 Agent 的规划起点：从“如何新增一个 dtype”前移到“是否需要新增，以及哪些目标契约已经冻结”。
2. “当前仓库事实 / 目标契约”双列能够减少从 INT、MXFP 或题面名称偷渡 role、bit、粒度和 deploy 范围。
3. 消费面检查能稳定暴露“类存在、注册成功，但 CLI、quantizer 或 deploy 未接通”的问题。
4. `ops-boundary` 能帮助 Agent 把 Python 参考语义和 NPU kernel 实现分开，不在 dtype 规划中直接虚构 kernel 细节。

本轮不能支持以下结论：

- 不能证明修改后行为在独立 cold 会话中稳定。
- 不能证明读取预算已经收敛。
- 不能证明信息充分时 Agent 会及时结束提问和检查。
- 不能把源码规划结果表述为代码可用、集成完成或真实硬件验证通过。

## 6. 下一轮验证

1. 补跑信息充分的 `q6s` 用例，检查 Agent 是否能直接采用已冻结契约并限制第一阶段改动面。
2. 用物理隔离的新会话重跑 D1、D3、D4，每题至少三次；有 Skill 环境不得看到无 Skill 输出、本文件或 reviewer 结论。
3. D1 验证未知项全程保持 `pending`；D3 验证只采用题面明确项；D4 验证 CLI/deploy 是否严格按目标参与范围决定，而不是默认必改。
4. 记录实际读取的 Skill、Reference、源码和测试切片，以及模型版本、commit、token、时间和中止原因。
5. 使用文末统一 rubric 独立评分；只有语义和读取效率同时达标，才把 dtype 状态机记为回归通过。

## 附录 A：原始测试记录

以下保留原始测试语和两种情况下的回答，作为上述结论的复核证据。Reviewer 内容不得投喂给后续被测 Agent。

### D1：只有产品名称的新格式

原始测试语：

> AMCT 准备支持一种暂定名为 `nova-float` 的新量化格式，在进行编码。目前只有这个产品名称，产品侧还没有给出更详细的格式说明。请基于当前仓库给出接入前的任务规划。本次只做只读分析，不修改代码，不执行测试或量化。


#### 无 Skill 原始回答

核对完成。只有产品名这一个输入的情况下，能立即定下来的事比预期多 —— 因为名字本身就触发了三个具体决策，而且 nova-float
  这个名字在本仓有两处会静默出错。

  先说一句范围判断：数值编码未定时，nova_float_impl.py 的实际量化逻辑是无法开工的。现在可编码的是格式无关的骨架与三处既有
  欠债。如果「正在进行编码」指的是前者，建议先停下来对齐规格；如果指后者，方向是对的。

  名字带来的三个可立即决策项

  nova-float 含连字符，这在本仓有硬约束：

  - 模块名：nova-float.py 不可导入 —— register_dtype() 里的 from .nova-float import ... 是语法错误。必须是 nova_float.py。
  - 算子标识：若要做 NPU 算子，torch.ops.amct.encode_to_nova-float 不是合法 C++ 标识符，TORCH_LIBRARY_FRAGMENT 的 schema
    会编译失败。
  - 注册键：本仓 ALGO_REGISTRY 的既有命名一个分隔符都没有 —— autoround、flatquant、omniquant、lwc、lac。注意是 autoround
    而非 auto-round，说明本仓惯例是去掉分隔符而不是换成下划线。

  argparse 的取值里连字符是合法的（不同于选项名），所以 --quant_dtype nova-float 能跑通。这恰恰是风险所在：CLI
  字符串可以保留连字符，而模块名和注册键不能，于是同一个格式会有多个写法并存。

  需要显式定下并记录映射的是这四个：产品名 nova-float / CLI 取值 / 注册键 / 模块与算子标识符。建议注册键取 novafloat 或
  nova_float（前者贴合既有惯例），CLI 取值与注册键保持一致以免再引入一层转换 —— quant_base.py:112,150 是直接把
  args.quant_dtype 传给 DTYPE_REGISTRY.get() 的，中间没有归一化层，两者不一致就必须新增映射。

  「暂定名」的直接推论

  既然名字是暂定的，那么格式名字面量的集中化优先级应当提前。目前格式名以裸字符串散落在至少这些位置：args.py:67、llm_deploy
  .py:58、llm_deploy.py:269、deploy_export.py:52/58/65/72、glm5_2.py:152、glm5_2/quant_module.py:192/225/300、deepseekv4.p
  y:257、hyv3.py:175、auto_clip.py:59/71、auto_round.py:91-93。若 nova-float
  按现有方式铺进去，之后产品改名就是一次十几处的横扫，且漏一处即静默错误。

  先集中再接入，成本低得多。这项工作完全不依赖格式规格。

  名字里的 float 造成的两处静默错误

  这是本次核对最值得注意的发现 —— 两处判定对同一个格式给出相反结论：

  llm_deploy.py:58：
  self.is_mx = self.quant_dtype.startswith("mx")
  字符串前缀判断，不查注册表。nova-float 不以 mx 开头 → is_mx=False → deploy_export.py:58,65 产出
  qtype="int"、format="int-quantized"，且 :72 不写 weight_block_size。一个名字里带 float 的格式被导出成 int-quantized。

  deepseekv4.py:257 / hyv3.py:175：
  cache_type = "int" if self.quant_dtype == "int" else "float"
  else 兜底到 float → nova-float 在这里偶然落到 "float"。

  于是 deploy 的 quant_config 说 int，cache scheme 说 float。两处都不报错，单测也测不到 —— 只有跑完整 deploy
  并检查产物元数据才会暴露。

  更深一层：is_mx 同时驱动 qtype 和 weight_block_size，把「是否浮点类型」与「是否 32 block
  缩放」两件事绑成了一个布尔量。一个浮点但非 MX 编码的格式正是拆开这个耦合的触发条件。这项重构同样不依赖规格。

  现在可以开工的

  1. 敲定四处名字与映射关系（上文），并同步给产品侧确认 —— 若产品名后续变更，需知道哪些是对外可见的。
  2. 格式名集中化：把散落字面量收敛到单一定义处，为改名和新增格式各留一个入口。
  3. is_mx 解耦：改为按格式属性分派，拆开「浮点/整型」与「block 粒度」两个维度。这一步会把上面那处矛盾一并消除。
  4. 两处白名单改为注册表派生：args.py:67 的 choices=['int', 'mxfp']、llm_deploy.py:269 的 if self.quant_dtype in ["int", 
     "mxfp"]。Registry.list_all()（registry_factory.py:88-89）已存在但生产代码无一处调用。后一处未命中是静默跳过量化、导出
     未量化模型，比 CLI 拒收危险。
  5. 建 dtype 骨架：nova_float.py + nova_float_impl.py，数值部分留 NotImplementedError，并在 register_dtype() 里加
     import（漏这行会导致格式从未注册，而直接 import 类的单测照样全绿）。
  6. 写格式无关的契约测试：注册闭环（只调 register_dtype() 后查 list_all()，不直接 import 模块）、bits=16 直通、bits=16, 
     is_act=True 可构造（ActivationQuantizer 在 quant_apply.py:167-168 等六处是无条件构造的，不看 a_bits）、export_deploy
     返回含 qweight 的 dict。
  7. 拟规格问询清单并发给产品侧（下节即是）。

  阻塞项与对应的待确认问题

  ┌────────────────────────────────────────────────────────────────────┬──────────────────────────────────────────────┐
  │                             阻塞的工作                             │                需要产品侧给出                │
  ├────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ nova_float_impl.py 全部数值实现                                    │ 编码（指数/尾数位宽、特殊值、舍入规则）      │
  ├────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ 位宽是否需放开 bit_policy.py:23 的 _ALLOWED_BITS = (4, 8, 16)      │ 支持位宽                                     │
  ├────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ is_act 分支、ActivationQuantizer 接入、cache 相关                  │ 支持对象（权重 / 激活 / KV-cache）           │
  ├────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ scale 张量形状、auto_clip.py:59,71、auto_round.py:91-97            │ 量化粒度（per-channel / block 及 block       │
  │                                                                    │ 大小）                                       │
  ├────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ export_deploy payload 键、quant_payload 落盘名、quant_config       │ 部署格式与下游消费方                         │
  │ 元数据                                                             │                                              │
  ├────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ amct_ops/nova_float_cast/ 三层                                     │ 是否需要 NPU 加速、目标 SoC（A2/A3/A5）      │
  └────────────────────────────────────────────────────────────────────┴──────────────────────────────────────────────┘

  其中两条会硬崩而非静默：auto_round.py:91-97 对未知 dtype 落 else 并 raise ValueError("Not supported hifx for 
  now")；quant_util.py:198-207 的 convert_dtype 对未知 dtype raise。是否阻塞取决于要搭配哪些 algos、以及是否走 NPU 落盘。

  排序与判据

  第 1 项先做，它决定后面所有文件名和符号名。第 2–4 项是格式无关欠债，建议在规格到位前完成 —— 它们不仅服务
  nova-float，也是前面几种格式反复撞到的同一批问题，做完之后新格式接入面会明显收窄。第 5–6 项建骨架但不填数值。第 7
  项越早发出越好，它是唯一能解除阻塞的动作。

  完成判据建议定为：CLI 能跑通一次 nova-float 的 PTQ → deploy 全流程，且产出的 quant_config 与 cache scheme
  中的格式标识一致且正确。数值单测全绿不构成完成 —— 上面那处 int/float 自相矛盾正是单测覆盖不到的地方。

  本次仅只读分析，未修改代码，未执行测试或量化。

#### 有 Skill 原始回答
amct-coder（主场景手册 references/dtype-quant-module.md）。补充一点准确性：它不是我自己调 Skill
  工具拉进来的，而是在我提出要不要重跑之后，由系统以规则注入的方式加载的,所以第一轮"没触发"的原因仍在会话的 skill
  清单装配，未变。

  任务路线

  按 skill 的固定流程走了 4 步：

  1. 职责归属 — 判定为「实现规划」，属 amct-coder 自身；不涉及 adapter/kernel/跑实验，无需路由外部 skill
  2. 仓库边界 — 读 repo-map 分层边界与更新规则，声明主落点层为 amct_pytorch/quantization/，并判定本次不需要写回
  repo-map（新增 dtype 属"不要重写"情形）
  3. 主场景 + 源码锚点 — 选 dtype-quant-module，先过「既有语义复用门」，再按消费面清单逐项取证
  4. 交付与验证 — 只读规划，按证据级别声明"仅源码存在，未跑任何验证"

  与无 skill 的区别

  关键差异不在格式，在于多做了一步复用门核验，而这一步改变了结论：

  ┌──────────┬────────────────────────────────────────────────────────────────┬──────────────────────────────────────┐
  │          │                            无 skill                            │               有 skill               │
  ├──────────┼────────────────────────────────────────────────────────────────┼──────────────────────────────────────┤
  │ 起点假设 │ 直接假定「新增 dtype 类」                                      │ 先问「这是不是新 dtype」             │
  ├──────────┼────────────────────────────────────────────────────────────────┼──────────────────────────────────────┤
  │ 交付形态 │ P0–P3 单一线性计划                                             │ A/B/C 三条件分支 + 8 项封闭阻塞输入  │
  ├──────────┼────────────────────────────────────────────────────────────────┼──────────────────────────────────────┤
  │ 契约处理 │ 默认了 weight+activation+KV cache 适用范围（从 MXFP 现状复制） │ 九个消费面全部标 pending，无推荐默认 │
  ├──────────┼────────────────────────────────────────────────────────────────┼──────────────────────────────────────┤
  │ 落点     │ 给出唯一文件名与 registry 名                                   │ 契约未冻结，只列候选锚点             │
  └──────────┴────────────────────────────────────────────────────────────────┴──────────────────────────────────────┘

  复用门带出两个会改变方案的事实，无 skill 那轮完全没查：仓库已有五种浮点格式分布在 LLM new-path 与 classic
  两套互不相通的体系；llm_deploy.py:60 的 is_hif 分支在 registry
  里取不到对应条目（未接通），不是可参照的先例。有这两条，"新增 dtype"就从既定前提降级为三分之一的可能。

  还有一处是无 skill 那轮判断错了方向：我把 rg quant_dtype
  命中的十余处硬编码称为"真实工作量"，并建议先做一轮去硬编码重构当 P0。playbook
  明确这些命中只是兼容性审计信号，不是必改列表；那轮重构还跨出了 dtype 层 OWN 边界（触及 deploy_export.py
  和多个模型适配器），在 role 与 deploy 参与度都没冻结时，根本无法判定它有没有必要做。

  一句话概括：无 skill 版本更像一份看着可执行的计划，但它的可执行性建立在几个未经核验的假设上；有 skill
  版本交付的是"哪些是事实、哪些待确认、各分支分别怎么做"。规格未定的接入前规划，后者才是对的形态。

有 skill 版本的计划

  核心判断：在拿到数值语义前，nova-float 到底是不是一个新 dtype 无法确定。所以交付的不是一条路线，而是「一个前置裁决 +
  三条分支」。

  边界
  
  主落点 amct_pytorch/quantization/（dtypes / modules / bit_policy）。CLI、deploy payload、algorithm 兼容、模型侧 cache
  分支都是下游闭环，不是主修改面。repo-map 不写回。

  前置裁决（决定走哪条分支）

  仓库已有五种浮点格式，分布在两套互不相通的体系：
  - LLM new-path DTYPE_REGISTRY — 只有 int、mxfp
  - classic 白名单 — hifloat8、float8_e4m3fn、float4_e2m1、mxfp8_e4m3fn、mxfp4_e2m1

  先比对 nova-float 的数值语义、粒度、packing、scale 语义、适用 role 是否已被上述某一个完整覆盖。

  三条分支

  A｜语义已被现有实现覆盖 → 零改动或公开别名。落点只在 registry 别名 + CLI choices + 别名可解析的最小 parser 测试。不新增
  dtype 类，不新增 deploy format。

  B｜classic 已有等价格式，需迁入 new-path → 在 new-path 按已冻结语义实现 dtype 类并接入 register_dtype()。不可复用
  classic 注册表或白名单常量（跨代码体系）。额外需裁决两侧命名是否统一。

  C｜确为新数值语义 → 按 playbook 分层顺序 1→9：dtype/impl 数值语义（含 bits==16 直通）→ 接入 register_dtype() →
  仅在通用接口不兼容时才动 quant module → bit policy/CLI 只放开真实支持的组合 → 参与 deploy 时先完整读 deploy-export-code
  再动 payload 与 quantization_config → 逐个确认算法支持/拒绝/fallback → 模型侧先做兼容 review，需结构实现转 model-adapter
  → 需 NPU 加速时保留 Python 语义基线，kernel 转 amct-ops-dev。

  三分支共用测试矩阵

  dtype 数值（shape/dtype、零与边界、误差界、round/clamp、STE、非法 bit/shape/block）、16-bit 直通、real quant
  payload（必含 qweight、scale shape/dtype、pack/unpack roundtrip、v 生效）、registry（可取／不冲突／重复调用幂等）。入口
  tests/unit_test/quantization/test_dtypes.py。

  8 项阻塞输入（封闭选项，一次问完，无推荐默认）

  定位（A/B/C）、编码格式、缩放粒度、适用 role、bit 集合、是否格式族、deploy 参与度（blockwise 与 tensorwise
  独立不互推）、NPU 依赖。

Reviewer 主要观察：

- 是否先定位 dtype 主职责和当前 registry/消费入口；
- 是否把名称解释成某种既有 FP4/FP8 编码；
- 未知的 bit、role、粒度、数值语义和 deploy 范围是否在回答前后保持一致；
- 是否在证据不足时仍给出唯一类名、文件、CLI 值或 payload。

典型失败信号：从名称推导编码或位宽；默认选择 weight-only；直接给出确定改动清单。

### D2：带有错误复用暗示的方案评审

原始测试语：

> 有人建议新增 `orbit4` dtype 时直接复制仓库现有的一套 4 bit 实现，只替换注册名，这样 CLI 和部署应该就能自动支持。请评审这个方案并给出下一步规划。本次只读，不修改任何文件。

#### 无 Skill 原始回答
成立，且「自动支持」这一条是反的

  「CLI 和部署应该就能自动支持」在本仓恰好是两处硬编码白名单，都不读注册表：

  - cli/llm/args.py:67 — choices=['int', 'mxfp']。--quant_dtype orbit4 会被 argparse 直接拒收，代码一行都进不去。
  - workflows/llm_deploy.py:269 — if self.quant_dtype in ["int", 
    "mxfp"]:。未命中就静默跳过量化，导出一个未量化的模型，不报错、不告警。

  Registry 确实提供了 list_all() 和 has()（registry_factory.py:85-92），但生产代码里无一处调用 ——
  全仓只有定义处的命中。注册表只管一件事：DTYPE_REGISTRY.get(args.quant_dtype) 的实例化（quant_base.py:112,150）。注册成功
  ≠ 可达。

  所以复制加改名的实际结果是：CLI 拒收；即使绕过 CLI 直接构造，deploy 也会安静地产出未量化模型。后者比前者危险 ——
  报错能被发现，静默走错会一路落盘。

  「现有一套 4 bit 实现」是个歧义项

  本仓有两套互不兼容的 4 bit 实现，各自绑定不同下游：

  ┌──────────┬─────────────────────────────────┬────────────────────────────────────┐
  │          │          int 的 4 bit           │           mxfp 的 4 bit            │
  ├──────────┼─────────────────────────────────┼────────────────────────────────────┤
  │ 编码     │ 对称 per-channel，qmax = 2^3-1  │ fp4_e2m1，block 32 共享 e8m0 指数  │
  ├──────────┼─────────────────────────────────┼────────────────────────────────────┤
  │ 打包     │ pack_4bit，沿输出通道转置后打包 │ pack_uint4，沿最后一维相邻两元素   │
  ├──────────┼─────────────────────────────────┼────────────────────────────────────┤
  │ scale    │ scale_fp32_to_u64 → uint64      │ e8m0 偏移 127 存 uint8             │
  ├──────────┼─────────────────────────────────┼────────────────────────────────────┤
  │ 附加张量 │ int4_assistance_bias            │ 无                                 │
  ├──────────┼─────────────────────────────────┼────────────────────────────────────┤
  │ 下游     │ 注释明确写 W4A8 MoEGMM 专用     │ compressed-tensors float-quantized │
  └──────────┴─────────────────────────────────┴────────────────────────────────────┘

  复制哪一套，就完整继承它的数值编码和部署布局。int_impl.py 里三个 4 bit 函数的 docstring 都写着 "for W4A8
  MoEGMM"，这不是通用实现。方案没说复制哪套，而这个选择等于替 orbit4 定了编码 ——
  通常这该是格式定义者的决定，不是实现者的。

  更根本的问题：改名不产生新格式

  如果数值实现照抄、只换注册名，那么 orbit4 与被抄的那个格式逐位等价。它不是新格式，是别名。当初要新增 orbit4
  的理由（无论是什么）一条也没落地。

  反过来，如果 orbit4 的数值编码与两者都不同，那么复制来的 *_impl.py 整个都要重写 ——
  复制省下的只有文件骨架，而骨架本来就是最便宜的部分。

  这里存在一个隐含的自相矛盾：方案要么产出一个别名（做了但没意义），要么复制来的核心逻辑全部要换（省不下工作量）。两种情况
  下「复制现有实现」都不是有效策略。

  复制还会带进几处隐藏耦合

  - int_impl.py:72 的 assert tensor.dim() == 2 —— 2D 假设。
  - mxfp.py:35 的 self.block_size = 32 与 _get_format_params()（:75-79）里 bits == 8 / else 的二分 —— 抄过来后 orbit4
    只要不是 8 就走 fp4 参数。
  - pack_4bit / pack_uint4 都有形状可除性约束（前者转置后按 2 打包，后者 assert shape[-1] % 2 == 0）。
  - int.py 的 is_act 分支会走 dynamic_per_token_quant —— orbit4 若是 weight-only，这条路径是死代码，但会被
    ActivationQuantizer 无条件构造时以 is_act=True 触及。
  - 复制文件不会自动把 import 加进 dtypes/__init__.py 的 register_dtype()。漏这一行，三个 workflow 里 orbit4
    从未注册，而直接 import 类的单测照样全绿。
  - 数值单测一并复制的话，测的是被抄格式的性质，对 orbit4 无验证力，但会呈现为「测试已覆盖」。
  - 大段重复实现大概率触发 ruff 检查（仓库用 ruff，且近期有 F841/E741/F811 清理提交）。

  方案里对的部分

  结构上照现有格式的样子做是对的，这点值得保留：<name>.py + <name>_impl.py 的拆分、__init__(bits, is_act) 签名、forward 里
  bits==16 直通、export_deploy 返回含 qweight 的 dict、注册装饰器写法 —— 这些正是本仓 dtype 的既有契约，int 与 mxfp
  真就是这样并存的。

  问题只在两点：把结构对齐误当成自动支持，以及复制了实现而非骨架。正确说法是「照 int/mxfp
  的骨架新建，数值部分自行实现，并显式改动 CLI 与 deploy 两处白名单」。

  下一步规划

  第一步 —— 先定性质，这是分岔口

  1. orbit4 与现有两套 4 bit 是什么关系？三选一：(a) 真别名 —— 那么该做的是注册别名或加 CLI 取值映射，不是复制文件；(b)
     编码不同 —— 那么 orbit4_impl.py 必须自研，复制无意义；(c) 编码待定 —— 那么现在只能做骨架，数值留空。
  2. 若为 (b)/(c)，明确编码、粒度（per-channel 还是 block、block 多大）、是否 weight-only。
  3. 明确是否进入部署导出。这一项对工作量近乎二分：不进则不需要 packing、落盘和 quant_config 元数据。

  第二步 —— 与编码无关、可立即做

  4. 建 dtypes/orbit4.py / orbit4_impl.py 骨架，并在 register_dtype() 里加 import。
  5. args.py:67 放行 —— 建议改为从 DTYPE_REGISTRY.list_all() 派生（list_all() 已存在但无人用），一次性消除这类白名单遗漏。
  6. llm_deploy.py:269 的白名单与 deploy_export.py 的 is_mx 布尔量（:52,58,65,72）改为按格式名分派 ——
     第三种格式在布尔量里无法表达，不改就会被误标成 int-quantized。
  7. 写自研的契约测试：注册闭环（只调 register_dtype() 后查 list_all()，不直接 import 模块）、bits=16, is_act=True
     构造与直通、export_deploy 键完整性。

  第三步 —— 编码确定后

  8. 实现数值部分并与参考值比对。
  9. 处理 auto_clip.py:59,71（按 == "mxfp" 决定 block 32 还是 per-channel，orbit4 会静默落到 per-channel）与
     auto_round.py:91-97（未知 dtype 落 else 并 raise ValueError("Not supported hifx for now")，硬崩）——
     是否阻塞取决于要搭配哪些 algos。
  10. 文档与样例：docs/zh/AMCT_Pytorch_LLM.md 第 280 行参数表、297 行语义说明、examples/algorithms/。

  完成判据建议定为：CLI 能跑通一次 orbit4 的 PTQ → deploy 全流程，且导出的 quant_config
  格式标识正确。数值单测全绿不构成完成 —— 它恰好是这个方案最容易误判的地方。

  本次只读，未修改任何文件。


#### 有 Skill 原始回答
方案不成立，三个前提都与源码冲突，建议不要按“复制 + 改注册名”执行。

  前提 1「有一套现成 4bit 实现可整体复制」——不成立。
  amct_pytorch/quantization/dtypes/ 下只有 int.py / int_impl.py / mxfp.py / mxfp_impl.py。4bit 不是独立 dtype，而是
  int/mxfp 内部的 bit 分支：int_impl.py:83 在 real_quant 且 bits == 4 时走 pack_4bit()（int_impl.py:53，沿输出通道两个
  int4 压一字节，为 W4A8 MoEGMM 服务），mxfp.py:41/76 以 bits == 8 二分 float8 与 packed E2M1。所以没有“一个 4bit
  文件”可复制，只能复制某个 dtype 类的整体（含它的 8/16bit 语义），复制出来的东西的适用范围、粒度、packing
  全部继承自被复制方，而不是来自 orbit4 的需求。

  前提 2「CLI 自动支持」——不成立。 amct_pytorch/cli/llm/args.py:67 是硬编码白名单 choices=['int', 'mxfp']，不从
  DTYPE_REGISTRY 派生。不改这一行，--quant_dtype orbit4 直接被 argparse 拒绝。

  前提 3「部署自动支持」——不成立，且失败方式是静默的。 amct_pytorch/workflows/llm_deploy.py:58-60 用前缀分类
  startswith("mx"/"int"/"hif")；orbit4 三个都不匹配，于是 is_mx/is_int/is_hif 全
  False，不会报错，而是走到一个没人设计过的组合。下游 deploy_export.py:52 的 generate_quant_config(is_mx=...) 只做
  float/int 二分（qtype、format），新类别会被默认写成 int-quantized。再叠加 llm_deploy.py:269 的精确白名单 quant_dtype in 
  ["int", "mxfp"]，orbit4 会被静默排除在该分支之外。

  旁证：is_hif 分支已经存在，但 dtypes/ 下没有 hif 注册、CLI choices 也没有该值——这正是“加了 dtype
  却没接通入口”的既有残留形态。

  另外还有两处会静默错分的下游：deepseek_v4/deepseekv4.py:257 与 hyv3/hyv3.py:175 的 "int" if quant_dtype == "int" else 
  "float"，orbit4 落到 float cache；algorithms/quant/auto_clip.py:59/71 的 == "mxfp" 判断，orbit4 落到非 mxfp
  分支。这些都是命中信号，不是必改清单。

  更前置的问题：可能根本不需要新 dtype。 按复用门，若 orbit4 已冻结的数值语义（编码、block/group 粒度、packing
  layout、scale 语义、weight/activation 适用范围）与现有 int4 或 mxfp4
  逐项相同，那正确做法是零改动或做公开别名，而不是造一个等价 dtype 类。“名字不同”“产品称谓不同”“基于 INT4
  扩展”都不足以证明需要新 dtype。这一点在拿到 orbit4 的数值定义前无法判定。

  真实落点与消费面

  registry 构造侧是通用的，不需要改：quantization/modules/quant_base.py:112/150 统一用
  DTYPE_REGISTRY.get(args.quant_dtype)(bits=..., is_act=True/False) 实例化。bit policy 侧 bit_policy.py:23 的
  _ALLOWED_BITS = (4, 8, 16) 已含 4，是否需要动取决于 role 与粒度决策。

  ┌──────────────────┬────────────────────────────────────────────────────────────────┬───────────────────────────────┐
  │      消费面      │                       当前事实（已核验）                       │           目标契约            │
  ├──────────────────┼────────────────────────────────────────────────────────────────┼───────────────────────────────┤
  │ registry/import  │ dtypes/__init__.py register_dtype() 仅 import                  │ pending：新增 / 别名 /        │
  │                  │ .int、.mxfp；只加文件不进 import 链则 registry 不可达          │ 复用未定                      │
  ├──────────────────┼────────────────────────────────────────────────────────────────┼───────────────────────────────┤
  │ CLI              │ args.py:67 硬编码 choices=['int','mxfp']，与 registry 无联动   │ pending：是否公开该值        │
  ├──────────────────┼────────────────────────────────────────────────────────────────┼──────────────────────────────┤
  │ bit              │ _ALLOWED_BITS = (4,8,16)；w_bits/a_bits/cache_bits 默认 16     │ pending：role、bit、粒度未定 │
  │ policy/config    │                                                                │                              │
  ├──────────────────┼────────────────────────────────────────────────────────────────┼──────────────────────────────┤
  │                  │ int.py:47 export_deploy() →                                    │                              │
  │                  │ int.py:47 export_deploy() →                              │                                     │
  │ weight quantizer │ weight_quant(real_quant=True)，4bit 时 packing + 辅助    │ pending                             │
  │                  │ bias；bits==16 直通（int.py:43）                         │                                     │
  ├─────────────────┼───────────────────────────────────────────────────────────┼─────────────────────────────────────┤
  │ activation      │ int.py:37 dynamic per-token；is_act=True 实例化           │ pending：是否支持 activation        │
  │ quantizer       │                                                           │                                     │
  ├─────────────────┼───────────────────────────────────────────────────────────┼─────────────────────────────────────┤
  │ blockwise       │ llm_deploy.py:167 经 is_mx 传入                           │                                     │
  │ deploy          │ generate_quant_config；回载 deploy_export.py:143          │ pending                             │
  │                 │ weight_dequant(is_mx=True, is_packed=True)                │                                     │
  ├─────────────────┼───────────────────────────────────────────────────────────┼─────────────────────────────────────┤
  │ tensorwise      │ llm_deploy.py:269 精确白名单 ["int","mxfp"]               │ pending：不能从 blockwise 推导      │
  │ deploy          │                                                           │                                     │
  ├─────────────────┼───────────────────────────────────────────────────────────┼─────────────────────────────────────┤
  │ algorithm 兼容  │ auto_clip.py:59/71 显式 == "mxfp"                         │ pending：逐个决定支持/拒绝/fallback │
  ├─────────────────┼───────────────────────────────────────────────────────────┼─────────────────────────────────────┤
  │ 模型专属分支    │ glm5_2（glm5_2.py:152、quant_module.py:192/225/300）精确  │ pending：需 orbit4                  │
  │                 │ == "mxfp"；deepseekv4:257、hyv3:175 else-float            │ 语义才能判是否真受影响              │
  └─────────────────┴───────────────────────────────────────────────────────────┴─────────────────────────────────────┘

  下一步规划

  第 0 步先回答下面四项，它们决定后面是否有代码要写。这些是封闭选项，任选其一即可推进；我不替你选，因为仓库里没有可引用的
  orbit4 语义证据。

  1. 数值语义：orbit4 的编码格式（符号/指数/尾数、是否共享指数）、对称性、round/clamp、零与边界处理，与现有 int4 或 mxfp4
     是否逐项相同？相同 → 走别名/复用，不新增 dtype；不同 → 列出差异项。
  2. 适用范围：weight-only / weight+activation / 含 cache，三者选一。
  3. 粒度：per-channel(row) / per-group / per-block（含 block size 与非整除行为）。
  4. 是否参与 deploy：不参与 / 只 blockwise / blockwise+tensorwise。参与则需给出 payload 字段、packing 顺序、scale 的
     dtype 与 shape。

  拿到答复后按分支展开：

  - 若语义与现有实现等价 → 落点收缩到别名层与 CLI choices，改动面最小，验证只需 registry 与 CLI parser 测试。
  - 若语义确有差异 → 落点顺序为：dtype/impl 实现数值与 packing（保持 bits==16 直通）→ 接入 register_dtype() → 同步 CLI
    choices 与 bit policy 允许组合 → 若参与 deploy，把 startswith 前缀分类与 in ["int","mxfp"]
    精确白名单两处逐一判断（此时需完整读 deploy-export-code），并对齐 generate_quant_config 的 float/int 二分与回载
    weight_dequant 的 layout → 最后逐个裁定 auto_clip 与模型侧分支是支持、拒绝还是 fallback。不支持的组合应在 CLI/bit
    policy 这类最上游入口显式拒绝，而不是等下游偶然报错。
  - 若需要 NPU kernel 加速 → Python 侧保留确定语义作基线，kernel 实现转 amct-ops-dev。



Reviewer 主要观察：

- 是否把用户建议当作待核验方案，而不是已确认事实；
- 是否先比较数值语义和实际消费入口，再判断复用、别名或新增实现；
- 是否核对注册可达、CLI、bit policy、weight/activation quantizer 和 deploy caller；
- 是否能明确指出“注册成功”与“下游已支持”的证据等级不同。

典型失败信号：接受“复制并改名”；只检查 `DTYPE_REGISTRY`；声称 CLI/deploy 会自动支持。

### D3：部分契约已经冻结

原始测试语：

> 计划在 AMCT 中增加 `quark4`。目前已确认它只用于权重、固定 4 bit，并且不量化 activation 和 cache；数值编码、量化粒度以及是否进入部署导出仍未确定。请给出实现前规划。本次只读，不修改代码或运行测试。


#### 无 Skill 原始回答
核对完成。三项已确认的约束确实消掉了一些障碍，但其中「固定 4 bit」与本仓的一处无条件构造直接冲突 ——
  这是实现前必须先解决的。

  已确认项各自换来什么

  ┌─────────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │   已确认    │                                          在本仓的实际效果                                           │
  ├─────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 固定 4 bit  │ bit_policy.py:23 的 _ALLOWED_BITS = (4, 8, 16) 已含 4，无需改动                                     │
  ├─────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 不量化      │ has_quant_cache()（bit_policy.py:77-79）在 cache 全 16 时返回 False，cache_scheme 走 None           │
  │ cache       │ 分支，天然适配                                                                                      │
  ├─────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 只用于权重  │ has_quant_linear()（:82）在 w_bits < 16 时返回 True，w4a16 能被正确识别为量化配置                   │
  └─────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────┘

  配置形态因此可以定下来：顶层 w_bits: 4, a_bits: 16。注意 bit_policy 要求嵌套条目里 w_bits/a_bits
  必须成对出现（_check_complete，:150-158），只写 w_bits 会 raise。

  一处必须先解决的冲突

  「不量化 activation」不等于激活侧代码不会被执行。ActivationQuantizer 在 quant_apply.py:167-168 是无条件构造的，完全不看
  a_bits：

  self.input_quant = ActivationQuantizer(quant_args, gate.a)   # gate.a == 16
  self.hidden_quant = ActivationQuantizer(quant_args, down.a)

  而 ActivationQuantizer.__init__（quant_base.py:112）会执行：

  self.quant_obj = DTYPE_REGISTRY.get(args.quant_dtype)(bits=self.bits, is_act=True)

  也就是说 QuantDequantQuark4(bits=16, is_act=True) 一定会被构造。同样的无条件构造还出现在 glm5_2/quant_module.py:191-195
  、qwen3_next/quant_module.py:76-77、deepseek_v3_2/quant_module.py:144-145、longcat_lite/quant_module.py:194-195。

  所以「固定 4 bit」若实现成 assert bits == 4 或在 __init__ 里硬写 4 并拒绝其他值，模型构造阶段就会崩，且崩在 MLP
  包装而非量化逻辑里，排查方向容易跑偏。正确形态是沿用 int.py:42-45 / mxfp.py:61-64 的既有约定：

  def __init__(self, bits=4, is_act=False): ...
  def forward(self, x, v=0.0):
      if self.bits == 16:
          return x       # 激活侧在此直通
      return self.fake_quant(x, v=v)

  另外 ActivationQuantizer.fake_quant 调用时不传 v（quant_base.py:118-119），签名要能单参调用。

  「固定 4 bit」的准确表述应是：权重侧只支持 4，激活侧以 16 
  直通。建议在方案里就改成这个措辞，避免实现者理解成构造期强校验。

  现在可以开工的部分

  1. dtype 骨架与注册闭环。dtypes/quark4.py + quark4_impl.py，注册名 quark4，并且务必同时在 dtypes/__init__.py 的 
     register_dtype() 里加 import —— 装饰器只在模块被导入时执行，漏这一行会导致三个
     workflow（llm_ptq.py:59、llm_eval.py:59、llm_deploy.py:69）里 quark4 从未注册，而直接 import 类的单测照样全绿。
  2. CLI 放行。cli/llm/args.py:67 的 choices=['int', 'mxfp'] 需要加 quark4，或改为从 DTYPE_REGISTRY
     派生（后者更符合注册表设计，且一次性解决后续格式）。
  3. weight-only 契约测试。不依赖编码的不变量现在就能写：bits=16, is_act=True 构造不抛异常且 forward 直通、bits=4 时
     shape/dtype 保持、单参调用可用、export_deploy 返回含 qweight 的 dict。这几条正好覆盖上面那个冲突。
  4. 算法兼容性排查。确认 quark4 要与哪些 algos 组合使用 —— 这决定下面两条是不是阻塞项。

  被阻塞的部分

  数值编码未定 → quark4_impl.py 全部数值实现。注意 weight_quant 有 assert tensor.dim() == 2（int_impl.py:72），2D
  假设是否沿用需明确。

  量化粒度未定 → 影响面比编码更广，且多数是静默错误：

  - auto_clip.py:59,71 按 == "mxfp" 决定 view(-1, 32) 还是 per-channel，clip_dim 同理。quark4 会默认落到 per-channel 
    分支，若实际是 block-wise 则算错且不报错。
  - auto_round.py:91-97 的 _get_group_size 对未知 dtype 落到 else 并 raise ValueError("Not supported hifx for now") ——
    硬崩。用不用 auto_round 决定这条是否阻塞。
  - deploy_export.py:72 的 weight_block_size = [1, 32] 仅在 is_mx 下写入。
  - pipeline 的 block_size hook 各适配器不一致：base.py:59-60 返回 32、deepseekv4.py:74-76 按 dtype 返回
    32/128、hyv3.py:78-79 返回 1。
  - 粒度还直接决定 export_deploy 里 scale 的张量形状。

  是否进入部署导出未定 → 这一项对工作量的影响最大，几乎是二分：

  - 若不进：quark4 只服务 PTQ/eval 的 fake-quant，不需要 packing、不需要 safetensors 落盘、不需要 quant_config
    元数据。上面第 2 组阻塞项大半消失。
  - 若进：要处理 llm_deploy.py:269 的第二道白名单 if self.quant_dtype in ["int", 
    "mxfp"]（未命中是静默跳过量化、导出未量化模型且不报错）、deploy_export.py:52/58/65/72 的 is_mx
    布尔量（第三种格式无处表达，会被误标成 int-quantized）、以及 quant_payload（:152-161）的键名到 safetensors
    张量名映射约定。

  关于 4 bit 部署布局有一点要特别提醒：本仓现有的 4 bit 落盘路径（int_impl.py 的 pack_4bit + scale_fp32_to_u64 +
  int4_assistance_bias）注释明确写着 W4A8 MoEGMM 专用，是与特定消费方强绑定的。quark4 是 w4a16、且可能不限于
  MoE，不能默认复用这套布局，需要独立确认下游消费方期望的排布与 scale 类型。

  排序建议

  先解决那个构造冲突（改方案措辞，不需要代码），再做第 1–3 项骨架工作。同时把「是否进入部署导出」提到最前面问 ——
  它决定后续一半工作是否存在，比编码和粒度更该先定。粒度定下来之后，auto_clip/auto_round 两处按实际组合需求处理。

  另外需要确认一项范围：quark4 是否也要做 NPU 算子。若要，amct_ops/quark4_cast/ 三层加 ops_init.py
  挂载是独立工作量，且应以 Python
  参考实现为数值基准，一致性判据（逐位相同还是给定容差）建议在算子开工前就定，否则返工成本会集中爆发。

  本次只读，未修改代码、未运行测试。
  
#### 有 Skill 原始回答
 给的三项（weight-only、固定 4 bit、不量化 activation/cache）是用户输入级确认事实，我据此核验了它们在当前框架里能否直接
  表达。结论是其中两项无法按字面实现，需要在规划阶段就改变落点形态。

  三个必须先纠正的实现前提

  1「不量化 activation」不能由 dtype 类拒绝 is_act 来实现。
  quant_base.py:112 的 ActivationQuantizer.__init__ 无条件执行 DTYPE_REGISTRY.get(args.quant_dtype)(bits=self.bits, 
  is_act=True)，没有任何 dtype 白名单或 role 开关。而 ActivationQuantizer 的构造点遍布通用层与模型层（通用侧
  common/quant_apply.py:167-168，模型侧 glm5_2、deepseek_v3_2、qwen3_next、longcat_lite
  等十余处）。也就是说：只要模型被构建，quark4 类一定会被以 is_act=True 实例化，与你是否想量化 activation 无关。

  若在 __init__ 里对 is_act=True 抛异常，PTQ 会在建图阶段直接崩溃。所以 weight-only 的正确表达方式是：构造必须成功，由
  a_bits=16 的直通路径承载“不量化”，拒绝逻辑放到 bit policy / CLI 这类上游入口（对应手册“在最上游稳定入口拒绝”）。

  2「固定 4 bit」不代表类内可以假定 bits == 4。
  BitPolicy.__init__（bit_policy.py:57-58）中 w_bits/a_bits 默认均为 16，linear_bits()
  对未命中的层回落到这对全局默认值。WeightQuantizer.__init__（quant_base.py:147）就用这个值构造 dtype。因此非目标层的
  quark4 实例会拿到 bits=16，必须保持直通——这也是手册的 BLOCKER 项（bits==16 不再直通即为违规，参照
  int.py:43、mxfp.py:62）。

  同时，_ALLOWED_BITS = (4, 8, 16) 是全局校验，不区分 dtype；仓库当前没有 per-dtype 的 bit 合法性校验点。所以“固定 4
  bit”需要新增一个约束位置（拒绝 bits == 8），这是现有代码里不存在的能力，不是复制能得到的。

  3 名字里带 bit 数与现有架构不一致。
  现有两个注册名 int、mxfp 都是格式族，bit 数是正交维度：由构造参数 bits= 传入、由 bit_policy 按层解析。quark4 把 bit
  数烧进注册名，会产生 DTYPE_REGISTRY.get("quark4")(bits=16) 这类自相矛盾的实例（而如上所述这个实例必然会被创建）。

  建议注册名用格式族 quark，4 bit 约束通过 bit policy 校验表达；对外 CLI 若需要 quark4 这个产品称谓，可在 CLI
  层做值映射。这是有源码惯例支撑的推荐项，但改变用户可见的 CLI 取值属于公开契约，最终由你定。

  消费面状态（weight-only 前提代入后）

  ┌────────────────┬──────────────────────────┬──────────────────────────────────────────────────────────────────────┐
  │     消费面     │         当前事实         │                               目标契约                               │
  ├────────────────┼──────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │                │ dtypes/__init__.py       │                                                                      │
  │ registry/impor │ register_dtype() 仅      │ 待定名（quark vs quark4）；新增，非别名（前提已排除等价复用可能性之  │
  │ t              │ import .int/.mxfp；不进  │ 外的判断仍需编码语义）                                               │
  消费面: registry/import
  当前事实: dtypes/__init__.py register_dtype() 仅 import .int/.mxfp；不进 import 链则不可达
  目标契约: 待定名（quark vs quark4）；新增，非别名（前提已排除等价复用可能性之外的判断仍需编码语义）
  ────────────────────────────────────────
  消费面: CLI
  当前事实: args.py:67 硬编码 choices=['int','mxfp']，与 registry 无联动
  目标契约: 必改：需公开新值
  ────────────────────────────────────────
  消费面: bit policy
  当前事实: _ALLOWED_BITS=(4,8,16) 全局；无 per-dtype 校验；w_bits/a_bits 默认 16
  目标契约: 必改：新增“weight 仅允许 4/16、activation 必须 16”的拒绝点
  ────────────────────────────────────────
  消费面: weight quantizer
  当前事实: quant_base.py:165-177 优先用 algorithm 的 export_deploy()，否则回退 quant_obj.export_deploy()；缺失则
  NotImplementedError
  目标契约: supported；具体数值/粒度 pending
  ────────────────────────────────────────
  消费面: activation quantizer
  当前事实: :112 无条件以 is_act=True 构造；enable 默认 False；16bit 直通
  目标契约: 构造必须成功 + 16bit 直通；bits<16 时的 activation 行为按产品契约拒绝
  ────────────────────────────────────────
  消费面: cache
  当前事实: cache_bits() 默认 16；has_quant_cache() 判 <16
  目标契约: not applicable（你已确认不量化 cache），但需确保配置层无法把 cache 设成 quark
  ────────────────────────────────────────
  消费面: blockwise deploy
  当前事实: llm_deploy.py:167 经 is_mx 传 generate_quant_config；回载 deploy_export.py:143
  目标契约: pending
  ────────────────────────────────────────
  │ blockwise │ llm_deploy.py:167 经 is_mx 传                       │ pending                                         │
  │  deploy   │ generate_quant_config；回载 deploy_export.py:143    │                                                 │
  ├───────────┼─────────────────────────────────────────────────────┼─────────────────────────────────────────────────┤
  │ tensorwis │ llm_deploy.py:269 精确白名单 ["int","mxfp"]         │ pending                                         │
  │ e deploy  │                                                     │                                                 │
  ├───────────┼─────────────────────────────────────────────────────┼─────────────────────────────────────────────────┤
  │ deploy    │ llm_deploy.py:58-60 startswith("mx"/"int"/"hif")    │ 若参与 deploy 则必改；quark*                    │
  │ 前缀分类  │                                                     │ 三者全不匹配且不报错，静默走未设计组合          │
  ├───────────┼─────────────────────────────────────────────────────┼─────────────────────────────────────────────────┤
  │ algorithm │ auto_clip.py:59/71 显式 == "mxfp"                   │ pending：逐个裁定支持/拒绝/fallback             │
  │  兼容     │                                                     │                                                 │
  ├───────────┼─────────────────────────────────────────────────────┼─────────────────────────────────────────────────┤
  │ 模型专属  │ glm5_2（glm5_2.py:152、quant_module.py:192/225/300  │ pending：weight-only 下多为 cache/layout        │
  │ 分支      │ ）、deepseekv4:257、hyv3:175 均为精确判断 +         │ 判断，需按最终 deploy 决策复核                  │
  │           │ else-float                                          │                                                 │
  └───────────┴─────────────────────────────────────────────────────┴─────────────────────────────────────────────────┘

  下一步规划

  可以立即开工、不依赖未定项的部分：

  - dtype 类骨架与 registry 接入：类定义 + register_dtype() 内 import，构造签名对齐 (bits=..., is_act=False)，bits==16
    直通，is_act=True 构造不抛异常。
  - bit 约束点：新增 per-dtype 的 weight bit 校验（拒绝 8）与 activation 必须 16 的校验，落在 bit policy 或 args
    校验层（候选锚点 bit_policy.py 的 _validate / args.py:104-106 的 ensure_bit_policy 附近，最终位置待编码时确认）。
  - CLI choices 扩展 + 对应 parser 测试。
  - registry / 直通 / 构造不崩三类测试，可照 test_dtypes.py:104 test_int_weight_bits16_is_passthrough 与 :110 的 is_act
    参数化模式写。


Reviewer 主要观察：

- 是否只固定题面已经明确的 role 和 bit；
- 是否继续核验仓库当前 bit policy、quantizer 和入口事实；
- 未给出的数值编码、粒度和 deploy 范围是否保持未决；
- 条件分支是否只影响相应落点，而不是阻塞全部事实核验。

典型失败信号：把所有项目都标成未知；反过来根据“4 bit 权重”补出编码、per-channel 粒度或 packing 格式。

### D4：只有 dtype 类和注册的“已完成”方案

原始测试语：

> 某份 `luma` dtype 接入方案只包含一个新的 dtype 类、一次注册和对应数值单测，方案作者认为这样已经完成框架接入。请结合当前仓库判断这个完成结论是否成立，并给出后续检查计划。本次只读，不修改代码。

Reviewer 主要观察：

- 是否从定义和注册继续反查真实消费者，而不是直接接受“已完成”；
- 是否按当前调用关系选择必要源码锚点，而非机械全文读取所有 Reference；
- 是否区分源码存在、单测覆盖、集成闭环和真实环境验证；
- 是否在目标契约未知时避免替方案作者补齐具体语义。

典型失败信号：只补注册测试；把类存在视为 quantizer、CLI 或 deploy 已接通；为了完整性无条件加载全部子手册。
#### 无 Skill 原始回答
不成立
     
  luma 目前在仓库里零引用（全仓 grep 无命中），所以这是纯方案评审。方案覆盖的是最内层 —— dtype
  类本身的数值行为。但本仓的格式名会被注册表之外的至少 6 处硬编码分支读到，这些地方不会因为注册成功而自动适配。

  按这个方案交付出来的 luma，实际状态会是：CLI 拒收、deploy 静默跳过、导出元数据标错。数值单测全绿也照样如此，因为单测直接
  import 类、绕过了全部这些路径。

  硬阻塞（新格式无论作何取舍都要处理）

  ┌─────┬──────────────────────────────┬────────────────────────────────┬─────────────────────────────────────────────┐
  │  #  │             位置             │              问题              │                  失败形态                   │
  ├─────┼──────────────────────────────┼────────────────────────────────┼─────────────────────────────────────────────┤
  │     │                              │ register_dtype() 里硬编码 from │                                             │
  │ 1   │ dtypes/__init__.py:31-33     │  .int import ... / from .mxfp  │ 类完全不生效（见下）                        │
  │     │                              │ import ...。装饰器只在模块被   │                                             │
  │     │                              │ import 时才执行                │                                             │
  ├─────┼──────────────────────────────┼────────────────────────────────┼─────────────────────────────────────────────┤
  │ 2   │ cli/llm/args.py:67           │ choices=['int', 'mxfp']        │ argparse 直接拒绝 --quant_dtype             │
  │     │                              │                                │ luma，进不到任何代码                        │
  ├─────┼──────────────────────────────┼────────────────────────────────┼─────────────────────────────────────────────┤
  │ 3   │ workflows/llm_deploy.py:269  │ if self.quant_dtype in ["int", │ 静默跳过量化，导出一个未量化的模型，不报错  │
  │     │                              │  "mxfp"]: 第二道白名单         │                                             │
  ├─────┼──────────────────────────────┼────────────────────────────────┼─────────────────────────────────────────────┤
  │     │                              │ is_mx 布尔量，qtype = "float"  │ 第三种格式无处表达，会被误标成              │
  │ 4   │ deploy_export.py:52,58,65,72 │ if is_mx else "int"、format    │ int-quantized                               │
  │     │                              │ 二选一                         │                                             │
  └─────┴──────────────────────────────┴────────────────────────────────┴─────────────────────────────────────────────┘

  第 1 条要单独说：方案所称的「一次注册」有歧义。如果指的只是类上的 @DTYPE_REGISTRY.register(name="luma", ...)
  装饰器，那么 luma 在运行时从未注册 —— 三个 workflow（llm_ptq.py:59、llm_eval.py:59、llm_deploy.py:69）都调
  register_dtype()，而它只 import int 和 mxfp。数值单测因为直接 from ...dtypes.luma import ...
  触发了模块导入，所以照样通过，掩盖了这个问题。这是本次评审里最需要先问清的一点。

  第 3 条比第 2 条更危险：CLI 拒收是响亮的失败，deploy 静默跳过是安静的错误，会一路走到落盘。

  条件阻塞（取决于上一轮那三个待定项）

  - 位宽 —— bit_policy.py:23 的 _ALLOWED_BITS = (4, 8, 16)。luma 位宽若不在其中，yaml 配置校验直接 raise。
  - 算法组合 —— auto_round.py:91-97 的 _get_group_size 对未知 dtype 落到 else 分支并 raise ValueError("Not supported hifx 
    for now")，硬崩；auto_clip.py:59,71 按 == "mxfp" 决定 block 32 还是 per-channel，luma 会静默走成 per-channel，clip_dim
    算错但不报错。
  - NPU 落地 —— quant_util.py:198-207 的 convert_dtype 对未知 dtype raise。另外方案完全没提
    amct_ops/luma_cast/，而「同时提供 Python 参考实现和 NPU 加速」是这个特性的既定目标，算子三层加 ops_init.py
    挂载一个都没有。
  - 模型适配层 —— glm5_2/quant_module.py:192,225,300、glm5_2.py:152、deepseekv4.py:257、hyv3.py:175 都在按 == "mxfp" / == 
    "int" 决定 cache 类型和某些模块是否量化。luma 会因为「不是 mxfp」而落到 float 侧，是巧合而非设计。

  契约完整性

  数值单测覆盖不到的接口约束：

  - WeightQuantizer.export_deploy（quant_base.py:172-177）缺 export_deploy 就 NotImplementedError。
  - quant_payload（deploy_export.py:152-161）要求 payload 是 dict 且含 "qweight" 键，其余键按 .weight → .{key} 映射成
    safetensors 张量名。键名拼错不会报错，只会产出下游读不到的张量名。
  - 若 luma 要支持激活，ActivationQuantizer（quant_base.py:112）以 is_act=True 构造、且 fake_quant 调用时不传 
    v，签名必须兼容。
  - Registry._register（registry_factory.py:96）对重名 raise KeyError，名字冲突会在 import 期暴露 ——
    这条是好事，属于已有保护。

  后续检查计划

  按「最便宜且最能定性」排序，全部只读：

  第一步 —— 先问，不用看代码
  1. 「一次注册」具体指什么？是否包含 register_dtype() 里的 import 一行。这一条决定后面所有检查是否还有意义。
  2. luma 的支持对象范围（仅权重 / 含激活 / 含 KV-cache）与位宽。这决定条件阻塞里哪几条生效。

  第二步 —— 可达性验证（判断「是否真的接进来了」）
  3. 在干净进程里只调 register_dtype()，再查 DTYPE_REGISTRY.list_all() 是否含 luma —— 不要在测试里直接 import luma
  模块，那会掩盖第 1 条。
  4. 沿 --quant_dtype luma 走一遍参数解析，确认能否通过 args.py:67。
  5. 走 WeightQuantizer → export_deploy → quant_payload 的完整链路，而不是直接调 dtype 类的方法。

  第三步 —— 分支穷举（判断「是否有静默错误」）
  6. 以 quant_dtype ==、is_mx、"mxfp"、"int" 四个模式全仓 grep，逐个判定 luma 落到哪个分支、是有意还是默认。重点看 else
  分支 —— 崩溃能被发现，静默走错不能。
  7. 核对 _ALLOWED_BITS 与 luma 位宽。

  第四步 —— 交付面缺口
  8. 对照上一轮的接入点清单，确认 amct_ops/luma_cast/（kernel / extension / python 三层 + ops_init.py）、deploy_export.py
  的格式分派、docs/zh/AMCT_Pytorch_LLM.md（280 行参数表、297 行语义说明）、examples/ 的缺失项。
  9. 确认参考实现与算子的数值一致性判据是否已定。

  建议的完成定义：能在 CLI 里跑通一次 luma 的 PTQ 到 deploy 全流程，且导出的 quant_config 里格式标识正确 ——
  而不是数值单测通过。前者会自然逼出上面 1–4 号全部问题。

  本次只读，未修改任何代码。

#### 有 Skill 原始回答
  “已完成框架接入”不成立。可达性只满足了一半：注册链确实通了，但用户入口和下游消费面都没接。
     
  先确认方案里成立的部分——我本轮特意核实了 register_dtype() 是否真被生产代码调用（只有定义不等于活跃调用链）：它在
  workflows/llm_ptq.py:59、llm_eval.py:59、llm_deploy.py:69 的 _register_components() 中被调用。所以只要新类进入
  dtypes/__init__.py 的 import 链，PTQ / eval / deploy 三个 workflow 都能从 DTYPE_REGISTRY 取到它。这一项是 confirmed。

  （另外 luma 这个名字不带 bit 数，与现有 int/mxfp 的格式族命名惯例一致，bit 数留给 bits= 参数与 bit policy
  表达——这点是对的。）

  但下面这些缺口使“接入完成”这个结论无法成立。

  阻断性缺口

  1 用户根本选不到它。 cli/llm/args.py:67 的 choices=['int','mxfp'] 是硬编码白名单，不从 DTYPE_REGISTRY
  派生。--quant_dtype luma 被 argparse 直接拒绝。注册成功但入口不开，等于整条路径不可达——这一项单独就足以否决“完成”。

  2 类必然会被以 is_act=True 构造，而数值单测覆盖不到这条路径。 quantization/modules/quant_base.py:112 的
  ActivationQuantizer.__init__ 无条件执行 DTYPE_REGISTRY.get(args.quant_dtype)(bits=self.bits, is_act=True)，没有任何
  dtype 白名单或 role 开关；其构造点遍布通用层（common/quant_apply.py:167-168）与各模型 quant_module.py。若 luma 的
  __init__ 不接受 is_act 或在该分支抛异常，模型建图阶段就崩，而纯数值单测不会暴露这一点。

  3 bits=16 直通路径未覆盖。 BitPolicy.__init__（bit_policy.py:57-58）中 w_bits/a_bits 默认均为 16，linear_bits()
  对未命中层回落到该默认值，WeightQuantizer.__init__（quant_base.py:147）就用它构造 dtype。所以非目标层的 luma 实例会拿到
  bits=16，必须原样返回输入（参照 int.py:43、mxfp.py:62）。这是手册明列的 BLOCKER 项，只测 4bit 数值测不到。

  4 bit 合法性无 per-dtype 校验。 _ALLOWED_BITS = (4, 8, 16) 是全局常量，不区分 dtype；仓库当前没有“某 dtype 只支持某些
  bit / 某些 role”的校验点。若 luma 只支持部分 bit 或只用于
  weight，现在没有任何地方会拒绝非法组合，会一路走到下游才出问题或静默出错。

  5 deploy 静默错分（若参与 deploy）。 workflows/llm_deploy.py:58-60 是前缀分类 startswith("mx"/"int"/"hif")；luma
  三者全不匹配，is_mx/is_int/is_hif 全为 False，不报错。下游 common/models/llm/common/deploy_export.py:52 的
  generate_quant_config(is_mx=...) 只做 float/int 二分，luma 会被默认写成 int-quantized；再叠加 llm_deploy.py:269
  的精确白名单 quant_dtype in ["int","mxfp"]，luma 被静默排除。产物看似生成成功但 config 是错的。

  6 export_deploy() 缺失会在运行期才炸。 quant_base.py:172-177：若 dtype 没有 export_deploy()，抛 NotImplementedError: 
  Quant dtype 'xxx' does not implement export_deploy()。方案只提数值单测，未说明是否实现该方法。

  7 算法与模型侧静默落入错误分支。 algorithms/quant/auto_clip.py:59/71 显式判 == "mxfp"；模型侧
  deepseek/deepseek_v4/deepseekv4.py:257 与 hyv3/hyv3.py:175 是 "int" if quant_dtype == "int" else "float"，luma 落到
  float cache；glm5_2（glm5_2.py:152、quant_module.py:192/225/300）同为精确 == "mxfp"
  判断。这些是命中信号，不是必改清单，需按 luma 的实际语义逐项裁定支持 / 拒绝 / fallback。

  8 一个待核实信号： workflows/llm_extract_ptq_data.py:45-46 的 _register_components() 只调 register_llm_models()，没有
  register_dtype()。该阶段是否会构造 quant module 我本轮未核验；若会，则 extract 阶段取 dtype 会失败。这对 int/mxfp
  同样成立，可能是既有设计（extract 不建量化图），需确认后再判定是否为缺口。

  旁证：is_hif 分支已存在于 deploy，但 dtypes/ 无 hif 注册、CLI
  也无该值——仓库里已经有一例“分支存在但入口未接通”的残留，正是这类不完整接入的典型形态。

  消费面状态

  luma 的 role、bit 集合、粒度、是否参与 deploy 本轮都未给出，因此目标列多为 pending。我不从格式名或前一轮别的 dtype
  的前提推断这些。

  ┌────────────────────┬─────────────────────────────────────────────────────────────────┬────────────────────────────┐
  │      消费面       │                        当前事实                         │              目标契约               │
  ├───────────────────┼─────────────────────────────────────────────────────────┼─────────────────────────────────────┤
  │ registry/import   │ register_dtype() 被 ptq/eval/deploy 三处调用；进 import │ supported（方案已覆盖）             │
  │                   │  链即可达                                               │                                     │
  ├───────────────────┼─────────────────────────────────────────────────────────┼─────────────────────────────────────┤
  │ CLI               │ args.py:67 硬编码白名单，与 registry 无联动             │ 必改，方案未覆盖                    │
  ├───────────────────┼─────────────────────────────────────────────────────────┼─────────────────────────────────────┤
  │ bit policy        │ _ALLOWED_BITS=(4,8,16) 全局；无 per-dtype 校验；默认 16 │ pending：role/bit/粒度未定          │
  ├───────────────────┼─────────────────────────────────────────────────────────┼─────────────────────────────────────┤
  │ weight quantizer  │ quant_base.py:165-177 优先算法 hook，回退               │ pending                             │
  │                   │ quant_obj.export_deploy()，缺失即抛错                   │                                     │
  ├───────────────────┼─────────────────────────────────────────────────────────┼─────────────────────────────────────┤
  ├──────────────────┼──────────────────────────────────────────────────────┼────────────────────────────────────────┤
  │ activation       │ :112 无条件 is_act=True 构造；enable 默认 False      │ 构造必须成功 + 16bit 直通；bits<16     │
  │ quantizer        │                                                      │ 行为 pending                           │
  ├──────────────────┼──────────────────────────────────────────────────────┼────────────────────────────────────────┤
  │ blockwise deploy │ llm_deploy.py:167 经 is_mx 传参；回载                │ pending                                │
  │                  │ deploy_export.py:143                                 │                                        │
  ├──────────────────┼──────────────────────────────────────────────────────┼────────────────────────────────────────┤
  │ tensorwise       │ llm_deploy.py:269 精确白名单                         │ pending，不能由 blockwise 推导         │
  │ deploy           │                                                      │                                        │
  ├──────────────────┼──────────────────────────────────────────────────────┼────────────────────────────────────────┤
  │ deploy 前缀分类  │ :58-60 startswith 三分类，luma 全不匹配且不报错      │ 参与 deploy 则必改                     │
  ├──────────────────┼──────────────────────────────────────────────────────┼────────────────────────────────────────┤
  │ algorithm 兼容   │ auto_clip.py:59/71 精确判 mxfp                       │ pending，逐项裁定                      │
  ├──────────────────┼──────────────────────────────────────────────────────┼────────────────────────────────────────┤
  │ 模型专属分支     │ glm5_2 / deepseekv4:257 / hyv3:175 精确判断 +        │ pending，需语义才能判是否真受影响      │
  │                  │ else-float                                           │                                        │
  └──────────────────┴──────────────────────────────────────────────────────┴────────────────────────────────────────┘

  后续检查计划

  不依赖未定项、可立即补的：

  1. CLI choices 加入新值，并补 parser 测试（tests/unit_test/cli/test_llm_module_entrypoints.py 或新增）。
  2. 补三类测试：is_act=True 构造不抛异常、weight 与 activation 的 bits=16 直通、registry 取值与重复注册幂等。可照
     test_dtypes.py:104 的 test_int_weight_bits16_is_passthrough 与 :110 的 is_act 参数化写法。
  3. 确认类是否实现 export_deploy()；不实现则需在上游明确拒绝 deploy 路径，而非等 NotImplementedError。
  4. 核实 llm_extract_ptq_data 是否需要 register_dtype()。

  需先确认语义才能定的（互相独立，答一项推一项）：

  5. role：weight-only / weight+activation / 含 cache。决定 activation 侧是实现还是在上游拒绝。
  6. bit 集合与粒度：支持哪些 bit、per-channel / per-group / per-block（含 block size 与非整除行为）。决定是否需要新增
     per-dtype bit 校验点，以及 scale 的 shape。
  7. 是否参与 deploy：不参与 / 只 blockwise / blockwise+tensorwise。选“不参与”则须在 deploy 入口显式拒绝，避免第 5
     条的静默错分；选参与则需 payload 字段名、packing 顺序、scale 的 dtype/shape，届时完整读 deploy-export-code 并对齐回载
     weight_dequant 的 layout。
  8. 语义确定后，逐项裁定 auto_clip 与模型侧分支，不按 grep 命中盲改。

  另需回答一个更前置的问题：luma 的数值语义是否与现有 int4 或 mxfp4 逐项等价？若等价，正确做法是别名或复用，而非新增 dtype
  类——这一点在拿到编码定义前无法判定。

  证据等级

  只读，未修改任何文件，未运行任何测试。以上均为当前 commit 的源码事实（“源码存在”级）。test_dtypes.py
  只读了断言清单未执行，现有测试通过状况未核验；llm_extract_ptq_data 是否构造 quant module 未核验。未做真实模型 eval 或
  deploy 导出（缺模型权重与 NPU/CANN 环境）。本轮复用已确认事实，新增读取为 register_dtype 调用点、QuantConfig、workflow
  注册段。repo-map 与源码在 dtype 分层上一致，无写回需要。

### D5：dtype 与 NPU 加速的职责交界

原始测试语：

> 团队准备为一种尚未进入 AMCT 的新量化格式同时提供 Python 参考实现和昇腾 NPU 加速。格式名称已经确定，但数值编码、支持对象和部署格式还在讨论中。请给出当前可以推进的工作及后续职责划分。本次只做规划，不修改代码、不编译算子。
#### 无 Skill 原始回答
 已经摸清了接入点。结论先说：在编码方案未定的前提下，仍有相当一部分工作可以立即开工，因为本仓的量化格式是注册表驱动的，格
  式无关的骨架、契约测试和两处「加第三种格式必然要改」的重构都不依赖那三个待定项。

  仓库现状（本次核对结果）

  新格式要落地，需要碰的点是这些：

  - dtype 扩展点：amct_pytorch/quantization/dtypes/。用 @DTYPE_REGISTRY.register(name=..., description=...) 注册，再在
    dtypes/__init__.py 的 register_dtype() 里 import 才生效。现有 int 与 mxfp 都遵循 <name>.py（注册 + nn.Module 封装）与
    <name>_impl.py（纯数值函数）的拆分。
  - dtype 契约（鸭子类型，从 int.py/mxfp.py 归纳）：__init__(bits, is_act)、forward(x, v)（bits==16
    直通）、fake_quant、export_deploy(x, v) -> {"qweight", "weight_scale", ...}，mxfp 另有 quant/deploy。
  - 消费方：quantization/modules/quant_base.py 里 WeightQuantizer/ActivationQuantizer 通过
    DTYPE_REGISTRY.get(args.quant_dtype)(bits=..., is_act=...) 实例化。
  - NPU 算子模板：amct_ops/hifloat8_cast/ 的三层结构 —— op_kernel/（Ascend
    C）、op_extension/（TORCH_LIBRARY_FRAGMENT(amct, ...) + PrivateUse1 实现 + Meta 实现）、python/<pkg>/。构建入口
    amct_ops/ops_build.sh [--soc ascend910b|ascend910_93|ascend950] [<op>]，并在 ops_init.py 的 __all__ 里挂出。
  - 部署导出：common/models/llm/common/deploy_export.py，走 compressed-tensors，qtype = "float" if is_mx else 
    "int"、format 取 float-quantized/int-quantized。
  - 测试与文档：tests/unit_test/quantization/test_dtypes.py（CPU，按 bits
    参数化）、tests/amct_ops/test_hifloat8_cast.py（NPU）、docs/zh/AMCT_Pytorch_LLM.md（第 280 行参数表、297
    行语义说明）。

  立即可以推进的工作

  1. 先开 Issue 立项。AGENTS.md 明确要求新特性先经 Issue 达成共识再提 PR，模板在
     .gitcode/ISSUE_TEMPLATE/feature_requeset.yaml。把「数值编码 / 支持对象 /
     部署格式」三项作为显式待决条目写进去，后续讨论有锚点。
  2. 锁定命名。格式名已定，那么派生的名字全都可以定：dtypes/<name>.py 与 <name>_impl.py、注册表里的 name
     字符串、amct_ops/<name>_cast/ 目录、torch.ops.amct.encode_to_<name> /
     decode_from_<name>。这些一旦定下，两条工作流可以并行而不撞车。
  3. 两处「必然要改」的重构（推荐现在做，收益与编码无关）：
     - cli/llm/args.py:67 的 --quant_dtype 是硬编码 choices=['int', 'mxfp']，与注册表驱动的设计不一致。改为从
       DTYPE_REGISTRY 派生，新格式接入时就不必再动 CLI。
     - deploy_export.py 用布尔 is_mx
       区分格式，第三种格式进来必然容纳不下。趁现在改成按格式名分派，这个重构不需要知道新格式的编码细节。
  4. 写格式无关的契约测试。与编码无关的不变量现在就能落：forward 保持 shape/dtype、bits==16 直通、export_deploy 返回 dict
     且含必需键、CPU 可跑。这套测试同时会把 dtype 契约固化成可执行文档 —— 目前它只是隐式约定。
  5. 确认 SoC 支持范围。ops_build.sh 支持 A2/A3/A5
     三档，新格式要覆盖哪几档是产品决策，但不依赖编码，可以先问清楚。它会直接影响算子开发的工作量。
  6. 搭 Python 与 NPU 的一致性比对脚手架，参照 tests/amct_ops/test_hifloat8_cast.py。

  被阻塞的工作及其对应决策

  ┌──────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │  待定项  │                                               解锁的工作                                                │
  ├──────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 数值编码 │ <name>_impl.py 全部数值实现、Ascend C kernel 数学部分；若位宽超出 bit_policy.py 的 _ALLOWED_BITS = (4,  │
  │          │ 8, 16)，该处需同步放开                                                                                  │
  ├──────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 支持对象 │ is_act 分支、ActivationQuantizer 接入、KV-cache（bit_policy 的                                          │
  │          │ attn-cache）、quant_linear/quant_matmul、MoE 路径                                                       │
  ├──────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ 部署格式 │ export_deploy 的 payload 键、deploy_export.py 的 quant_config 生成、safetensors 落盘布局                │
  └──────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  职责划分建议

  分三个角色，关键是排序：

  - Python 参考实现：负责 dtypes/<name>*.py 与 CPU 单测。这条线必须先落，因为它是 NPU 算子的数值基准（oracle）。
  - NPU 算子：负责 amct_ops/<name>_cast/ 三层与 tests/amct_ops/。开发时可参照
    .agents/skills/amct-ops-dev/SKILL.md（内含新增算子流程与实现注意事项）。在参考实现就绪前，可以先做 schema、Meta
    实现和构建接入 —— 这些不需要最终数学。
  - 接口与集成：负责 dtype 契约、CLI、deploy_export.py
    的分派重构，以及文档（docs/zh/AMCT_Pytorch_LLM.md、examples/algorithms/<name>/）。第 3、4 项工作归这条线。

  有一处建议提前明确：参考实现与算子的数值一致性判据（逐位相同，还是给定容差）。这个判据不取决于具体编码，但如果等到算子写
  完才定，返工成本会集中爆发。

  本次只做规划，未修改任何代码，也未编译算子。

#### 有 Skill 原始回答
新增：ops-boundary.md（NPU 加速跨契约影响成立，作为辅助手册）、quant_util.py:60-100 的 HiFloat8
  fallback、amct_ops/hifloat8_cast/ 目录形态。复用本会话已确认的 dtype 侧事实，源码未变。

  顺带确证了之前的旁证：quantization/dtypes/ 下没有任何 hif 注册（grep 命中全是 shift/shif 子串），CLI choices
  也无该值，但 llm_deploy.py:60 的 is_hif 分支存在。

  一个必须先定的前置选择：走哪种集成形态

  这个发现会改变后面全部落点，比数值编码更前置。仓库里“带 NPU 算子的数值格式”已有两种并存形态，不是一种：

  形态 A — LLM new path（int、mxfp）：@DTYPE_REGISTRY.register(name=...) 注册，register_dtype() 在 llm_ptq.py:59 /
  llm_eval.py:59 / llm_deploy.py:69 被调用，用户经 --quant_dtype 选择，由 quant_base.py:112/150 构造，deploy 经 startswith
  三分类消费。

  形态 B — HiFloat8 现状：不是注册 dtype、没有 CLI 值。以 quant_util.py 的 hifloat8_fake_quant() helper 形式存在，被
  classic/quantize_op/cast_module.py 消费，NPU 侧先探测原生 torch_npu、不支持时延迟 import amct_ops。

  所以“新增一种带 NPU 加速的格式”并不自动等于“新增一个 DTYPE_REGISTRY 条目”。选 A 还是 B 取决于该格式服务 LLM new path
  还是 classic 体系（或两者），这决定是否要动 CLI、bit policy、deploy
  三分类——改动面差一个量级。这是有源码证据的封闭选择，需要你定。

  现在就能推进的工作

  数值编码未定，所以 fake quant 与 kernel 数值都写不了。但下面这些不依赖编码语义，可以立即定稿：

  1 冻结 Python 参考实现为唯一数值 oracle。 这是 ops-boundary 混合任务的第 1 步，也是硬约束：NPU
  算子的正确性判定必须对照一个可在无 NPU 环境运行的参考实现。团队既然同时要 Python 参考 + NPU
  加速，参考实现必须先存在且可独立验证，不能等 kernel 写完再反推语义。

  2 定可选依赖与 fallback 策略。 可直接照 HiFloat8 既有形态（quant_util.py:83-95）：调用点延迟 import，wheel
  缺失抛带指引的 ImportError，模块 import 阶段不硬依赖 ops wheel（否则 CPU/源码环境 import amct_pytorch 直接失败，这是
  BLOCKER）。优先级顺序需明确：原生 torch_npu → amct_ops → Python 参考 → 显式报错。HiFloat8 当前是“原生 → amct_ops →
  报错”，没有 Python fallback；新格式既然有参考实现，可插入该层，但这是行为决策，需你确认。

  3 定 ops 侧目录骨架。 ops_build.sh 自动发现 <op>/CMakeLists.txt 与
  <op>/python/<pkg>/，新算子通常不需改通用脚本；布局可照
  hifloat8_cast/（op_kernel/、op_extension/、python/<pkg>/）。这部分与数值无关。

  4 若走形态 A，补结构性防护。 构造签名对齐 (bits=..., is_act=False)，is_act=True 必须构造成功不抛异常（quant_base.py:112
  无条件以该参数实例化），bits=16 必须直通（bit_policy 默认 16，非目标层必然拿到
  16）。这三项与编码无关，且纯数值单测覆盖不到。

  5 记录 deploy 静默错分风险。 若最终参与 deploy，llm_deploy.py:58-60
  的前缀匹配对新名字全不命中且不报错，generate_quant_config 的 float/int 二分会默认写成 int-quantized，:269
  的精确白名单会静默排除。无论最终是否参与，都需在入口显式处理，不能依赖下游偶然报错。

  被阻塞的部分与需确认的封闭问题

  以下互相独立，答一项推一项，不需要一次答完：

  1. 集成形态：A（DTYPE_REGISTRY + CLI）/ B（helper + classic）/ 两者。
  2. 数值编码：符号/指数/尾数布局、是否共享指数、round/clamp、特殊值（NaN/Inf/零）、STE
     形式、确定性与误差口径。同时需回答复用门：是否与现有 int4 / mxfp4 / HiFloat8
     逐项等价？等价则应做别名或复用，而非新增。
  3. 适用对象：weight-only / weight+activation / 含 cache；以及支持的 bit 集合与粒度（per-channel / per-group / per-block
     含 block size 与非整除行为）。当前 _ALLOWED_BITS=(4,8,16) 是全局常量，无 per-dtype 校验点，若需限制则要新增。
  4. 部署格式：不参与 / 只 blockwise / blockwise+tensorwise；参与则需 payload 字段名、packing 顺序、scale 的 dtype 与
     shape。
  5. 目标 SoC 与 CANN 范围：A2 ascend910b / A3 ascend910_93（dav-2201）、A5
     ascend950（dav-3510）。本轮未提及，它决定编译与硬件验证范围。

  第 2、3 项冻结后才能定 schema（参数名/顺序/类型/输出数量、是否 mutation/alias、是否需 autograd）。schema
  未冻结前不锁定具体上层实现文件或 kernel 方案——fake_quant 这类名字不能替代 weight/activation 与 shape 语义。

  职责划分

  ┌────────────────────────────────────────────────────────────────────────────────┬──────────────────────────────────┐
  │                                      范围                                      │               归属               │
  ├────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────┤
  │ Python 参考实现、dtype 类或 helper、registry/CLI/bit policy、quant module      │ 本 skill（dtype-quant-module +   │
  │ 兼容、上层调用点、延迟 import 与 fallback、deploy 的 Python 侧消费             │ ops-boundary）                   │
  ├────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────┤
  │ Ascend C kernel、C++ extension、TORCH_LIBRARY_FRAGMENT schema、PrivateUse1     │                                  │
  │ 实算与 Meta 形状推导、算子 CMakeLists、wheel/ops_build.sh、NPU 数值测试与多    │ amct-ops-dev                     │
  │ SoC 编译                                                                       │                                  │
  ├────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────┤
  │ deploy payload 与 config schema 细节                                           │ 部署参与范围冻结后，完整读       │
  │                                                                                │ deploy-export-code               │
  └────────────────────────────────────────────────────────────────────────────────┴──────────────────────────────────┘

  边界两条不能越：amct_pytorch 内不写 kernel/C++ 逻辑，amct_ops 不 import amct_pytorch 的算法或 workflow；反过来，上层
  fallback 与报错时机不交给 amct-ops-dev。

  推进顺序：冻结共享契约 → amct-ops-dev 实现算子 → 核对 wheel 可导入且 schema 已注册 → 本 skill 接上层调用点 →
  统一核对五个接口面（C++ schema、PrivateUse1/Meta 签名、Python wrapper、上层调用、测试 oracle）。schema
  若有破坏性变更，优先新函数名并保留旧接口，同时给出新旧 wheel 与调用方的兼容矩阵。

  证据等级

  只做规划，未修改任何代码、未编译算子、未运行测试。以上均为当前 commit
  的源码事实（“源码存在”级）。tests/amct_pytorch/test_hifloat8_fallback.py 与 test_dtypes.py
  均未执行，现有测试通过状况未核验。无 NPU/CANN 环境，未做算子编译、.so 导入、Meta 推导或 NPU
  数值验证；这几项后续须分层记录，不能用编译成功替代数值正确性，也不能用 Meta 成功替代实算。repo-map 与源码在 dtype/ops
  分层上一致；若最终不改变主流程编排语义，按其「更新规则」无需写回。


Reviewer 主要观察：

- 是否先处理 dtype 数值和上层调用契约，而不是直接规划 kernel；
- 是否正确区分 `amct-coder` 的 Python 侧契约 review 与 `amct-ops-dev` 的 kernel/schema/Meta/build 职责；
- 是否只停止依赖未知格式的实现部分，同时继续完成仓库现状核验；
- 是否避免虚构算子名、schema、tiling 方式和支持硬件范围。

典型失败信号：直接给出 Ascend C 文件和接口；把 NPU 实现当作定义数值语义的地方；因契约未知而停止全部规划。

## 附录 B：统一行为评分

每项按 `0/1/2` 评分：`0` 表示缺失或明显错误，`1` 表示提及但执行不稳定，`2` 表示有可核验证据且前后一致。总分用于比较同一题的多次运行，不跨不同难度题目直接比较。

| 维度 | 2 分判据 |
| --- | --- |
| 职责与主路径 | 选择 dtype 主场景；涉及外部职责时只在真实交界成立后分流 |
| 读取顺序 | 先读主规则和少量直接锚点，再由未解决契约驱动扩张 |
| 仓库事实 | 注册、CLI、bit policy、quantizer 或 deploy 结论有当前源码/caller/测试证据 |
| 契约控制 | 已知项正常采用；未知项不被名称、参考实现或后续叙述静默冻结 |
| 条件式规划 | 分支与缺失契约一一对应，不把候选落点写成唯一实现 |
| 消费面核验 | 不用“新增类/已注册”代替实际 quantizer、CLI 或 deploy 消费证据 |
| 上下文收敛 | 达到任务所需证据后停止；超出默认预算前能说明具体未解决契约 |
| 交付可信度 | 区分源码、单测、集成和真实环境证据，不声称执行未做的动作 |

以下任一情况出现时，该题最高记 8/16：

- 使用当前仓库不存在的 CLI 参数或调用入口；
- 把没有 caller 的 helper 当成活跃路径；
- 在没有新证据时把未知目标契约改成确定方案；
- 选择错误代码体系，或把模型/NPU 专属实现写入通用 dtype 层；
- 声称已经运行实际未执行的测试、量化、部署或硬件验证。

## 附录 C：建议后续执行顺序

1. 先用 D1 连续运行三次，验证 dtype 目标契约状态是否稳定。
2. 运行 D3 和尚未测试的 `q6s`，区分 Agent 是机械保留未知，还是能够正确处理部分和完整契约。
3. 使用 D2 和 D4 检查反证候选方案及活跃消费链核验。
4. 最后运行 D5，验证 dtype 与 NPU 子能力的职责交界。

旧 HIF4 题只作为综合回归题保留，不再作为 dtype 状态机的首个验证题。短题通过后再重跑旧题，可以判断改进是否迁移到更复杂的上下文。
