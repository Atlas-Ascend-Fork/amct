# 子能力：ptq-solver-params

## 触发

用户提到 solver、blockwise/global、optimizer、scheduler、epoch loss、`trainable_params()`、PTQ 参数保存/加载、`PtqParamHandler`、`PtqParamStore`、导出回载、unit 调度时使用。

## 源码锚点（按需选择）

以下路径不是全文必读清单。先按优化循环、factory、参数 store 或回载动作选择直接定义、调用方和测试锚点。

- `amct_pytorch/common/optimization/__init__.py`
- `amct_pytorch/common/optimization/base_solver.py`
- `amct_pytorch/common/optimization/blockwise_solver.py`
- `amct_pytorch/common/optimization/factory.py`
- `amct_pytorch/common/optimization/global_solver.py`
- `amct_pytorch/common/models/llm/common/ptq_params.py`
- `amct_pytorch/workflows/llm_ptq.py`
- `tests/unit_test/common/optimization/`
- `tests/unit_test/workflows/test_llm_ptq.py`

## 上下游依赖

- 上游：solver 消费 adapter 的 quant block / `PtqUnit`、data provider 的输入/GT、algorithm 的 `trainable_params()` 和 PTQ 参数 hook。
- 平级：solver 只做优化调度和参数收集，不读取模型名、不解析离线输入文件、不决定 dtype payload。
- 下游：eval/deploy 通过 `PtqParamHandler` / `PtqParamStore` 回载状态，必须用训练时一致的 target、unit 名和 `--algos`。
- 若新增 solver 或 granularity，要同时读 `llm-data-workflow` 和 `public-api-tests`，同步 CLI 接受值/帮助/默认值、workflow 分支和测试。

## 执行步骤

1. 先确认改动是优化循环、solver factory、参数 store，还是算法状态导出回载；新增 solver/granularity 时先冻结下节语义契约。
2. 检查 unit 输入和 GT 来自 provider，不在 solver 内直接读文件或重放模型。
3. 收集参数时只依赖 module/algorithm 的 `trainable_params()` 或导出 hook。
4. 训练型算法要验证参数真实进入 optimizer；统计型算法要验证 buffer 进入保存和回载。
5. 保存文件名保持 `layer_<idx>_<unit.save_name>.pt`，并确认 target 目录不会混用。
6. eval/deploy 回载失败时区分“确实可选”和“本应存在却缺失”，避免 `strict=False` 吞错。
7. 补 solver factory、blockwise solver、参数 store、workflow 或新增 granularity 的最小测试。

## 新 granularity 语义门禁

`block`、`layer`、`model` 等名字不能替代执行语义。形成具体类名、构造签名或源码落点前，逐项冻结：

`--granularity=layer`、“按层保存”或“介于 block 与 model 之间”都不能单独证明要做整层联合 forward、跨 unit 共享 optimizer、使用 layer 输入/输出或建立 layer 完成标记。用户没有给出优化目标、资源约束或精度语义时，只能并列这些候选及其代价，不推荐其中一种。

| 契约 | 必须回答的问题 |
| --- | --- |
| unit 边界 | 一次 solve 面向单个 `PtqUnit`、一层全部 units，还是完整模型；unit 仍由哪个 adapter 接口枚举 |
| module 所有权 | solver 持有 unit module、quant block、多个 module 的集合，还是完整模型；forward 实际消费哪一个对象 |
| batch / GT | provider 返回单 unit batch 还是联合 batch；不同 unit 的 kwargs、shape 和 GT 如何组织与对应 |
| optimizer 生命周期 | 参数集合在何时一次性收集；optimizer/scheduler 是 per-unit、per-layer 还是全局；共享是否会引入孤立参数或错过计算图 |
| finalize / 保存 | 谁把 solver 结果拆回各 unit；是否继续写 `layer_<idx>_<unit.save_name>.pt`；eval/deploy 现有 store 能否直接回载 |
| 断点续跑 | 完成标记属于 unit、layer 还是 model；部分产物存在时恢复、重训或报错的确定语义 |
| 兼容关系 | 与现有 granularity 是否共享参数目录和产物格式；CLI、registry、workflow、eval/deploy 分别接受、映射还是拒绝新值 |

以上任一项会改变接口而尚未冻结时，只输出条件分支和候选锚点。不得先发明 `LayerSolver` 等类名、固定构造参数、block 级文件名或把“跨 unit 共享 optimizer”描述为最小改动；共享 optimizer 只有在 solver 同时持有相关 module、batch/GT 和活跃计算图时才成立。具体类、文件和 runner 名只能出现在对应契约已冻结的条件分支内，不能写成当前确定落点。

## 主链状态

`global_solver.py` 文件存在且有注册装饰器，但当前 `register_solvers()` 只导入 `BlockwiseSolver`，主流程接通的是 `block`。`llm_ptq.run()` 当前在 granularity 分支前先查 registry，因此未注册的默认 `model` 会先失败，`_run_modelwise()` 不可达。文件、分支或装饰器存在都不等于 workflow 可选；分析时仍以本轮源码复核为准。

## solver 规则

- solver 不写模型名分支。
- unit 是否枚举或跳过由 adapter 的 `iter_ptq_units()` 或已定义的通用 unit 契约表达，solver 不查看其它 unit 决定过滤策略。
- 仅服务单个模型族的 unit 过滤参数不能直接加入通用 CLI；只有证明为跨模型通用能力并定义稳定语义后，才同步 CLI/API/workflow 闭环。
- trainable 参数只从 module 的 `trainable_params()` 收集。
- 无 trainable 参数时 solver 应直接返回，不假装训练。
- reconstruction loss 默认比较 unit output 与 GT。
- epoch loss 恒为 0 或 NaN 是强失败信号，必须检查输入、GT、梯度和 unit 策略。
- 新 solver 若要进入主流程，必须更新 `register_solvers()`、CLI 接受值/帮助/默认值、workflow 分支和测试；当前 CLI 没有 `choices` 时，不要把“新增 choices”冒充既有契约，应明确这是兼容性决策。

## 参数规则

- unit 参数文件名：`layer_<idx>_<unit.save_name>.pt`。
- `PtqParamHandler.export_unit()` 优先 module 自己的 `export_ptq_params()`，再递归子模块，最后 fallback 到 requires_grad 参数。
- `load_unit()` 优先 module 自己的 `load_ptq_params()`，再递归子模块，最后按 trainable 参数加载。
- eval/deploy 加载 PTQ 参数时，必须用训练时一致的 `--algos` 构建出相同子模块结构。

## 参数回载诊断

训练日志显示“参数更新”只能证明日志所指的训练侧状态发生变化，不能直接证明保存文件包含该状态，也不能证明 eval/deploy 已加载。按以下顺序取证：

1. 从当前实际启用的算法/module 查 `trainable_params()`、`export_ptq_params()`、`load_ptq_params()` 及真实 key；算法未指定时不套用 AutoRound 的 `value`、`min_scale`、`max_scale` 等字段。
2. 核对 `unit.save_name`、target 参数目录和 `layer_<idx>_<unit.save_name>.pt` 是否一致，并检查文件内容是否相对该算法初始状态发生变化。
3. 核对 eval/deploy 是否用相同 target、`--algos` 和参数目录重建模块；区分目录为空、文件缺失、unit 名错位与 payload key 不匹配。
4. `strict=False` 仅能表达已定义的可选缺失；本应存在的参数未加载必须产生可见日志或错误。

只有保存内容和加载后同一模块状态都已对比，才能把问题收敛到具体导出或回载实现。

## BLOCKER

- 算法有状态但没有导出/加载闭环。
- 参数目录按 target 混用，导致 attn/MLP/MoE 互相加载。
- strict=False 下吞掉本应存在的参数而无日志说明。
- 新 granularity 只加 solver 文件，没接 CLI/workflow/register。
- 新 granularity 的 module、batch/GT、optimizer、保存和续跑语义未冻结，却已确定类名、构造签名或共享 optimizer 方案。
- 仅凭 `layer` 名称就断言或推荐整层联合训练，并发明 block 级数据文件或完成标记。
- solver 自己读离线输入文件，绕过 data provider。

## 自验

- `pytest tests/unit_test/common/optimization/test_base_solver.py tests/unit_test/common/optimization/test_blockwise_solver.py tests/unit_test/common/optimization/test_factory.py`
- 涉及注册：加跑 `tests/unit_test/common/optimization/test_global_solver.py` 或新增对应测试。
- 涉及 PTQ workflow：`pytest tests/unit_test/workflows/test_llm_ptq.py`。
- 涉及参数 store：补针对 `PtqParamHandler/Store` 的 unit 测试或模型 adapter mocked 测试。
- 涉及新 granularity：覆盖 registry/CLI/workflow 分派、module 与 batch/GT 所有权、optimizer 参数集合、逐 unit 保存回载和部分产物续跑语义。
