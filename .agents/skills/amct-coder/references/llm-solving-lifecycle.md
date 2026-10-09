# 子能力：llm-solving-lifecycle

## 职责边界

- 上游：solver 消费 adapter 的 quant block / `PtqUnit`、data provider 的输入/GT、algorithm 的 `trainable_params()` 和 PTQ 参数 hook。
- 平级：solver 只做优化调度和参数收集，不读取模型名、不解析离线输入文件、不决定 dtype payload。
- 下游：eval/deploy 通过 `PtqParamHandler` / `PtqParamStore` 回载状态，必须用训练时一致的 target、unit 名和 `--algos`。
- 若新增 solver 或 granularity，要同时读 `llm-cli-dataflow` 和 `validation.md` 的公共接口检查，同步 CLI 接受值/帮助/默认值、workflow 分支和测试。

## 调用链

```text
workflow -> adapter PtqUnit + provider batch/GT
  -> solver registry -> solve -> finalize/export_ptq_params
  -> workflow 保存 -> eval/deploy 回载
```

## 实现定位

### 主要入口

- `amct_pytorch/common/optimization/`：solver 注册、优化循环及 optimizer/scheduler 构建。
- `amct_pytorch/common/models/llm/common/ptq_params.py`：参数导出与回载。

### 关键符号

搜索 `SOLVER_REGISTRY`、`register_solvers`、`BlockwiseSolver`、`trainable_params`、`finalize`、`PtqParamHandler` 和 `PtqParamStore`；沿 `LlmPtqWorkflow` 的调用核对 solver 选择、unit 执行和保存顺序。

### 验证入口

核对参数收集与更新、registry 分派、finalize、保存回载和部分结果处理。测试定位与命令见 [validation.md](validation.md#场景测试索引)，具体覆盖要求见本文件“自验”。

## 当前事实

事实附「核对」命令；作为判断依据前先执行，结果与断言冲突时以命令结果为准。命令失效（符号搜不到）本身即过期信号。发现不符时不改本节，把差异写进交付说明。

- `global_solver.py` 文件存在且有注册装饰器，但当前 `register_solvers()` 只导入 `BlockwiseSolver`，主流程接通的是 `block`。（核对：`rg -n -A4 "def register_solvers" amct_pytorch/common/optimization/__init__.py`）
- `llm_ptq.run()` 当前在 granularity 分支前先查 registry。（核对：`rg -n "SOLVER_REGISTRY.get" amct_pytorch/workflows/llm_ptq.py`）
- `model` 分支仍在 `run()` 的分派里，但 `LlmPtqWorkflow.__init__` 先拒绝非 `block` 取值，`_run_modelwise()` 是占位、无条件抛 `ValueError`，在 ptq 路径上不可达——ptq 只接受 `block`。（核对：`grep -n -A6 "def __init__" amct_pytorch/workflows/llm_ptq.py`，应看到 `!= "block"` 的 `raise ValueError`）
- `--granularity` 默认值已是 `block`，与 `register_solvers()` 实际接通的 solver 一致；改默认值属对外接口变更，不是仓内可自决的实现细节。（核对：`rg -n -A3 "granularity" amct_pytorch/cli/llm/args.py`，应看到 `default='block'`）

### 当前训练保存活跃链

当前 blockwise PTQ 训练侧必须按以下 caller 链判断，不要用回载侧 helper 替代：

```text
LlmPtqWorkflow._run_blockwise()
  -> solver.solve(unit_batch.data_loader)
  -> solver.finalize()
  -> LlmPtqWorkflow._save_unit_result()
  -> torch.save(result, 按现有 unit 参数文件名)
```

`BaseSolver.finalize()` 只在 unit module 自身提供 `export_ptq_params()` 时调用该方法，否则 fallback 到 unit module 的 `requires_grad` 参数。（核对：`rg -n -A8 "def finalize" amct_pytorch/common/optimization/base_solver.py`，应看到 `hasattr` 分派与 requires_grad fallback 两支）`PtqParamHandler.export_unit()` 提供 module 导出、递归子模块导出和 trainable fallback，但当前训练保存链没有调用它；它的存在不能证明任意子 wrapper 会被递归保存。（核对：`grep -rn "\.export_unit(" amct_pytorch/ tests/ | grep -v "def export_unit"` 应只命中 `tests/` 下文件；`amct_pytorch/` 内非定义命中即说明该论断过期——只搜 `amct_pytorch/` 时无输出不能区分「无调用」与「调用都在测试里」，必须含测试目录才有区分力）

任何保存/回载方案必须逐项给出：活跃 caller、传入的 module/unit、result 的真实 key/shape、文件名与 target 目录、加载 caller、strict 行为以及 eval/deploy 的最终消费者。

### 无 trainable 参数时

当前 `BlockwiseSolver.solve()` 收集不到 trainable 参数时，在遍历 dataloader 前直接返回。（核对：`rg -n -B3 -A2 "param_groups" amct_pytorch/common/optimization/blockwise_solver.py`）统计型或一次性优化算法不能依赖该方法自然完成 observe。新增路径前必须在 `llm-algorithm` 的运行方式基础上确定：

- 样本遍历由既有 GT forward、独立 calibration、solver 的统计分支还是其它已定义 caller 驱动。
- GT forward 的 observe 副作用是否可复用；若不可复用，如何隔离或清空统计。
- finalize 是 algorithm、module、solver 还是 workflow 的稳定约定，以及重复调用行为。
- 统计完成但没有 trainable 参数时，solver 返回值和保存结果如何区分“成功统计”与“什么都没做”。

这些语义未确定时，不新增空 optimizer、伪 trainable 参数或无 caller 的 post-calibration hook 来绕过提前返回。

## 执行步骤

1. 先确认改动是优化循环、solver factory、参数 store，还是算法状态导出回载；新增 solver/granularity 时先确定下一节的语义约定。
2. 检查 unit 输入和 GT 来自 provider，不在 solver 内直接读文件或重放模型。
3. 收集参数时只依赖 module/algorithm 的 `trainable_params()` 或导出 hook。
4. 训练型算法要验证参数真实进入 optimizer；统计型算法先证明谁驱动样本遍历和 finalize，再验证 buffer 进入保存和回载。
5. 保存文件名保持 `layer_<idx>_<unit.save_name>.pt`，并确认 target 目录不会混用。
6. 补 solver factory、blockwise solver、参数 store、workflow 或新增 granularity 的最小测试。

## 前置检查

### 新 granularity 语义

`block`、`layer`、`model` 等名字不能替代执行语义；语义未确定时只输出条件分支。`--granularity=layer`、“按层保存”或“介于 block 与 model 之间”都不能单独证明要做整层联合 forward、跨 unit 共享 optimizer、使用 layer 输入/输出或建立 layer 完成标记。用户没有给出优化目标、资源约束或精度语义时，只能并列这些候选及其代价，不推荐其中一种。形成具体类名、构造签名或源码位置前，逐项确定：

| 跨端约定 | 必须回答的问题 |
| --- | --- |
| unit 边界 | 一次 solve 面向单个 `PtqUnit`、一层全部 units，还是完整模型；unit 仍由哪个 adapter 接口枚举 |
| module 所有权 | solver 持有 unit module、quant block、多个 module 的集合，还是完整模型；forward 实际消费哪一个对象 |
| batch / GT | provider 返回单 unit batch 还是联合 batch；不同 unit 的 kwargs、shape 和 GT 如何组织与对应 |
| optimizer 生命周期 | 参数集合在何时一次性收集；optimizer/scheduler 是 per-unit、per-layer 还是全局；共享是否会引入孤立参数或错过计算图 |
| finalize / 保存 | 谁把 solver 结果拆回各 unit；是否继续沿用现有 unit 参数文件名；eval/deploy 现有 store 能否直接回载 |
| 断点续跑 | 完成标记属于 unit、layer 还是 model；部分产物存在时恢复、重训或报错的确定语义 |
| 兼容关系 | 与现有 granularity 是否共享参数目录和产物格式；CLI、registry、workflow、eval/deploy 分别接受、映射还是拒绝新值 |

领域补充：共享 optimizer 只有在 solver 同时持有相关 module、batch/GT 和活跃计算图时才成立。

### 复合 unit 与联合求解

一个 `PtqUnit` 同时承载 attention、MLP 或其它多个子模块时，不等同于新增 granularity，也不自动需要新 solver。形成类名或 workflow 分支前，除「新 granularity 语义」外还要回答：

| 跨端约定 | 必须回答的问题 |
| --- | --- |
| 构造 | `PtqUnit.kind/name/layer_idx/module/save_name` 如何生成；adapter 是否同时注入全部目标 wrapper |
| forward | module 是完整 block 还是复合容器；输入是在 norm 前还是 norm 后；kwargs 能否完整重放 |
| 参数集合 | 现有递归 `trainable_params()` 是否已覆盖全部目标；为何需要或不需要新 solver |
| GT/loss | 联合输出的比较边界和 loss；分开校准的公平对照是否最终量化同一组模块 |
| finalize | 单一结果还是按子 unit 拆分；当前 `BaseSolver.finalize()` 能否真实导出全部状态 |
| 回载 | 文件名、target 目录、unit 枚举、strict load 和 deploy binding 如何重建相同结构 |

任一项未确定时，只能把完整 block 或新 solver 列为候选，不能声称通过复用递归 helper 已完成保存闭环。

## 实现规则

### solver

- solver 不写模型名分支。
- unit 是否枚举或跳过由 adapter 的 `iter_ptq_units()` 或已定义的通用 unit 约定表达，solver 不查看其它 unit 决定过滤策略。
- 仅服务单个模型族的 unit 过滤参数不能直接加入通用 CLI；只有证明为跨模型通用能力并定义稳定语义后，才同步 CLI/API/workflow 闭环。
- trainable 参数只从 module 的 `trainable_params()` 收集。
- 当前训练型 solver 无 trainable 参数时应直接返回，不假装训练；统计型路径的样本驱动与 finalize 按上面的前置检查单独核对。
- reconstruction loss 默认比较 unit output 与 GT。
- epoch loss 恒为 0 或 NaN 是强失败信号，必须检查输入、GT、梯度和 unit 策略。
- 新 solver 若要进入主流程，必须更新 `register_solvers()`、CLI 接受值/帮助/默认值、workflow 分支和测试；当前 CLI 没有 `choices` 时，不要把“新增 choices”冒充既有约定，应明确这是兼容性决策。

### 参数

- unit 参数文件名默认 `layer_<idx>_<unit.save_name>.pt`；unit 不属于任何层时省略 `layer_<idx>_` 前缀，保存与回载同规则。
- `PtqParamHandler.export_unit()` 优先 module 自己的 `export_ptq_params()`，再递归子模块，最后 fallback 到 requires_grad 参数；先确认当前 caller 是否实际使用该 helper。
- `load_unit()` 优先 module 自己的 `load_ptq_params()`，再递归子模块，最后按 trainable 参数加载。
- eval/deploy 加载 PTQ 参数时，必须用训练时一致的 `--algos` 构建出相同子模块结构。

## skip / partial result 生命周期

跳过一个 unit、layer 或部分算法前，必须确定并测试：

1. 跳过的是优化、统计、fake quant、real quant 还是整个量化；“不调用 solve”不等于保持浮点。
2. 结果文件是缺失、显式 skip marker、direct-quant 状态还是浮点状态；参数 store 如何区分“尚未运行”和“有意跳过”。
3. 断点续跑看到部分产物时继续、重跑或报错的确定语义。
4. eval/deploy 用什么结构和 bit policy 重现该决定，`strict=True/False` 各自允许哪些缺失。
5. 决策改变模型数值语义时对应的用户可见配置、日志和比较基线。

在统一约定建立前，solver 不根据 loss、层名或其它 unit 自行静默跳过；算法侧的 score 或 decision 也不能仅靠不保存参数来表达跳层。

## 参数回载诊断

训练日志显示“参数更新”只能证明日志所指的训练侧状态发生变化，不能直接证明保存文件包含该状态，也不能证明 eval/deploy 已加载。按以下顺序取证：

1. 从当前实际启用的算法/module 查 `trainable_params()`、`export_ptq_params()`、`load_ptq_params()` 及真实 key；算法未指定时不套用 AutoRound 的 `value`、`min_scale`、`max_scale` 等字段。
2. 核对 `unit.save_name`、target 参数目录和实际文件名是否一致，并检查文件内容是否相对该算法初始状态发生变化。
3. 核对 eval/deploy 是否用相同 target、`--algos` 和参数目录重建模块；区分目录为空、文件缺失、unit 名错位与 payload key 不匹配。
4. `strict=False` 仅能表达已定义的可选缺失；本应存在的参数未加载必须产生可见日志或错误。

只有保存内容和加载后同一模块状态都已对比，才能把问题收敛到具体导出或回载实现。

## PTQ smoke 健康信号

按算法的实际运行方式选择最小 PTQ 集成 smoke 标准；组合算法分别核对每种方式。适用项不满足时停下排查，不把训练型条件套到纯统计路径。

共同要求：unit 输入加载成功；流程需要 GT 时 materialize 成功且参考输出未被统计或权重更新污染；需要持久化的参数、统计状态或优化结果保存后可回载，并被后续量化或导出消费。

| 运行方式 | 必须取得的健康信号 |
| --- | --- |
| trainable | solver 至少跑一个 batch，loss 不是恒 0 或 NaN，目标参数真实更新 |
| observe-only | 已确认的 GT forward、独立 calibration 或其它 caller 实际遍历样本，统计 buffer 按预期写入，后续量化或导出消费该统计 |
| observe-then-finalize | 样本遍历与统计写入成功；已接通的 caller 在统计完成后触发 finalize，生成预期的优化权重或状态，后续量化或导出消费该结果 |

纯统计路径不要求 optimizer、训练 loss 或 solver 遍历 batch。统计由 GT forward 驱动时，solver 因无 trainable 参数提前返回可以是预期行为；仍需验证统计次数、GT 不受污染，以及 finalize（若需要）和保存回载链已实际执行。

## 禁止项

- 新 granularity 的 module、batch/GT、optimizer、保存和续跑语义未确定，却已确定类名、构造签名或共享 optimizer 方案。
- 仅凭 `layer` 名称就断言或推荐整层联合训练，并发明 block 级数据文件或完成标记。
- 复合 unit 没有闭环 adapter 构造、原始输入/kwargs、finalize 拆分和回载结构，却声称现有 solver/helper 可直接复用。
- 无 trainable 参数时未定义样本驱动与 finalize，却声称统计算法已由 solver 执行。
- 用结果文件缺失表达主动跳过，导致续跑、strict load、eval 或 deploy 无法区分未执行与已决策。
- 算法有状态但没有导出/加载闭环。
- 参数目录按 target 混用，导致 attn/MLP/MoE 互相加载。
- strict=False 下吞掉本应存在的参数而无日志说明。
- 新 granularity 只加 solver 文件，没接 CLI/workflow/register。
- solver 自己读离线输入文件，绕过 data provider。

## 自验

- 测试定位与命令见 [validation.md](validation.md#场景测试索引)；按改动验证参数收集、优化循环及 optimizer/scheduler 构建。
- 涉及注册：核对正常注册入口导入后的 registry 与 workflow 分派，不能由测试直接导入 solver 后的 registry 推断 CLI 可用。
- 涉及 PTQ workflow：验证 unit 调度、solver 调用与结果保存顺序。
- 涉及参数 store：补针对 `PtqParamHandler/Store` 的 unit 测试或模型 adapter mocked 测试。
- 涉及新 granularity：覆盖 registry/CLI/workflow 分派、module 与 batch/GT 所有权、optimizer 参数集合、逐 unit 保存回载和部分产物续跑语义。
- 涉及复合 unit：覆盖 block 原始输入与 kwargs、全部目标参数收集、finalize 内容、公平分开基线和 eval/deploy 回载。
- 涉及 skip/partial result：覆盖主动跳过与未执行的区分、断点续跑、strict load 和最终量化开关。
