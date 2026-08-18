# 子能力：llm-algorithm

## 触发

用户提到 LLM new-path 算法、LWC、LAC、AutoRound、OmniQuant、FlatQuant、`ALGO_REGISTRY`、`targets`、可训练参数、统计参数、clip、round、structure transform 时使用。

## 源码锚点（按需选择）

以下路径不是全文必读清单。按主 skill 的读取预算，先根据当前改动选择直接定义、调用方和测试锚点，再按已确认的上下游影响扩展。

- `amct_pytorch/algorithms/registry_factory.py`
- `amct_pytorch/algorithms/quant/__init__.py`
- `amct_pytorch/algorithms/quant/auto_clip.py`
- `amct_pytorch/algorithms/quant/auto_round.py`
- `amct_pytorch/algorithms/quant/omniquant.py`
- `amct_pytorch/algorithms/quant/flatquant.py`
- `amct_pytorch/quantization/modules/quant_base.py`
- `amct_pytorch/common/optimization/blockwise_solver.py`
- `amct_pytorch/common/models/llm/common/ptq_params.py`
- `tests/unit_test/algorithms/`
- `tests/unit_test/quantization/modules/test_quant_base.py`

## 上下游依赖

- 上游：算法由 CLI `--algos` 传入，经 `ALGO_REGISTRY`、`register_algorithms()` 和 `build_algorithms_by_target()` 构建。
- 平级：算法只通过 quantizer hook、module buffer/parameter 或 structure transform 影响量化，不直接读取 dataset 或 workflow。
- 下游：`ptq-solver-params` 消费 `trainable_params()`、buffer 和 PTQ 参数导出；`deploy-export-code` 消费算法的 `export_deploy()` 或已回载状态。
- 若算法改变 dtype real quant 语义，要同时读 `dtype-quant-module` 和 `deploy-export-code`。

## 执行步骤

1. 明确算法作用域：`weight`、`activation`、`structure` 中的一个或多个；不要用宽泛 target 掩盖实现不匹配。
2. 冻结核心契约：数值变换、状态 shape/初始化/约束、与其它算法的组合关系、保存回载格式，以及 fake/real quant 或 deploy 是否需要自定义 hook。
3. 查现有同类算法的构造签名、状态保存和测试口径，优先沿用本地基类/辅助函数。
4. 核心契约冻结后实现最小可用算法，再用 `@ALGO_REGISTRY.register(...)` 注册，并确保 `amct_pytorch/algorithms/quant/__init__.py` 导入。
5. 接入 quantizer 或 structure transform，检查同一位置是否已有互斥 hook。
6. 对训练型算法实现 `trainable_params()`；对统计型算法实现 buffer、`export_ptq_params()`、`load_ptq_params()`。
7. 若 eval/deploy 需要复现训练态，检查 `--algos`、参数文件和回载路径是否能构建出相同模块结构。
8. 补 registry、算法本体、quantizer 分流、solver 或 deploy 的最小测试。

## 算法执行形态与阶段状态机

新算法在确定 workflow 或 solver 改动前，先按真实执行语义归入一种形态；算法名称不能替代该判断：

| 执行形态 | 数据驱动 | 状态变化 | 完成动作 |
| --- | --- | --- | --- |
| trainable | solver 遍历 batch 并执行 forward/backward | optimizer 更新 `trainable_params()` | 训练结束后导出参数 |
| observe-only | 已明确的 workflow/module forward 遍历样本 | 只更新统计 buffer，不修改待量化权重 | 统计状态直接供后续量化或导出 |
| observe-then-finalize | 已明确的观察阶段遍历样本 | 先更新统计状态，再一次性修改权重或生成优化状态 | 显式 finalize/post-calibration hook |

对 observe-only 或 observe-then-finalize，规划和实现必须给出下表，不能只写“增加 calibration pass”或“solver 自动触发”：

| 阶段 | 必须核对的事实 |
| --- | --- |
| GT 生成 | quant/observe 开关的真实状态；是否允许更新统计；是否会修改权重或污染浮点 GT |
| 样本遍历 | 由 workflow、solver 还是 module forward 驱动；遍历次数、开始/停止条件和重复进入语义 |
| 一次性优化 | 精确 caller、触发时机、输入状态、幂等性和失败行为 |
| fake/real quant | 统计前后调用哪个权重；优化权重是否再次被普通 fake quant 覆盖 |
| 保存 | 活跃 caller 最终取得哪些 buffer/权重/scale/offset；不能只引用未接通 helper |
| eval/deploy | 用什么 `--algos` 和参数重建结构；回载后哪个 hook 消费状态 |

当前源码需要特别复核三点：`QuantLinear.forward()` 的非 eval 分支会调用 `WeightQuantizer.observe_input()`；`LlmPtqWorkflow._prepare_unit_batch()` 生成 GT 时关闭量化但开启 observe；`BlockwiseSolver.solve()` 在没有 trainable 参数时于遍历 dataloader 前返回。因此，一次性统计算法若没有额外已接通的驱动和 finalize 调用，不能声称现有 PTQ 流程能够完成统计优化。

## 统计与决策型能力

敏感度评分、自动跳层或其它“统计后决策”能力不能只定义一个 score。形成 workflow 接口前必须冻结：

1. score 的原始输入；activation error 必须说明浮点/量化成对输出如何取得，Hessian/Fisher 必须说明统计或梯度来源。
2. 一个 unit 内多个 algorithm、transform 或 linear 的聚合规则，以及 layer/model 范围内阈值、排序或预算的作用域。
3. 决策结果的数据结构、确定性、分布式一致性和保存/回载格式。
4. 决策的执行者；algorithm 产生统计或决策状态，solver 不按模型结构自行猜测 unit，workflow 只有在通用 skip 契约冻结后才执行。
5. “跳过”的数值语义：保持浮点、回退 direct quant、跳过 PTQ 优化但仍量化，三者不能混写。

skip/partial result 的文件、续跑、strict load、eval 和 deploy 行为按 `ptq-solver-params` 检查。在这些契约冻结前，只给条件式计划，不发明通用阈值参数或把“跳过 `solver.solve()`”表述为跳过量化。

## 契约未冻结时的早停

新算法的名称和业务目标不等于实现契约。下列任一项尚未确定且会改变实现时，只做只读规划：

- 参数或统计状态的 shape、初始化、约束和 dtype/device 语义。
- 算法对输入的具体数值变换，以及关闭算法时的等价基线。
- 与其它同 target 算法的顺序、共存或互斥关系。
- PTQ 参数 key、导出/加载行为和旧参数兼容性。
- fake quant、real quant 与 deploy payload 是否复用同一路径。

此时输出必须冻结的契约、候选源码锚点、条件分支、阶段依赖和测试范围，不创建或建议先接入一个可由 `--algos` 选中的占位实现。空 `trainable_params()`、空导出/加载 hook、直通 `quantize()` 或伪造 payload 会让注册成功掩盖算法不可用，不是安全的“阶段 0”。

只有最小数值语义、状态生命周期和失败行为已经明确，才能把算法导入 `register_algorithms()`；注册、用户选择入口和最小行为测试应在同一可用阶段闭环。

题目只说“一个新算法”或未给算法身份时，保持 `<new_algo>` 等占位表达。现有算法和前序任务只能作为注册、hook 或测试模式的证据，不能把其名称、状态 key、shape、payload 或已冻结决策带入当前算法；条件式计划也不得把前序算法文件写成当前最终落点。

## 两套算法系统

LLM new-path 只认 `ALGO_REGISTRY`：

- 注册在 `amct_pytorch/algorithms/quant/*.py`。
- `register_algorithms()` 必须导入算法类。
- 通过 `targets=("weight",)`、`("activation",)`、`("structure",)` 分流。
- `WeightQuantizer` / `ActivationQuantizer` / structure transform 按 target 构建。

classic path 只认 `AlgorithmRegistry`，规则见 [classic-graph.md](classic-graph.md)。不要混用。

## 实现规则

- weight/activation 算法是 `nn.Module`，在 quantizer forward 前后参与变换。
- 需要训练的算法实现 `trainable_params()`，solver 只优化这些参数。
- 统计型算法用 buffer 保存观测结果，并实现 `export_ptq_params()` / `load_ptq_params()`。
- 自定义 weight `quantize()` hook 同一 `WeightQuantizer` 中只能有一个。
- 若算法要改 deploy real quant，提供 `export_deploy(weight, quant_obj)`，不要在 deploy workflow 写算法分支。
- structure 算法接收 `AlgoBuildContext`，同一位置当前只允许一个 structure 算法。
- 算法构造函数签名要匹配 `_build_algorithm()`：只需 `args` 或 `args, *ctor_args`。
- 已进入 registry 的算法必须具备与声明 target 一致的最小可用行为；不要注册只包含空 hook 或直通占位逻辑的半成品。

## BLOCKER

- 新算法没有 `@ALGO_REGISTRY.register(...)`。
- 核心数值/状态契约未冻结，却先注册 no-op 算法并暴露给 `--algos`。
- registry metadata 没有 `targets`。
- `targets` 与实际作用域不一致。
- 未命名的新算法继承了前序任务的具体名称、状态或 payload 契约。
- `register_algorithms()` 没导入，导致运行时不可选。
- 有 trainable 参数但 `trainable_params()` 返回空。
- observe-only / observe-then-finalize 算法没有明确的数据驱动者、observe 开关或完成 hook，却假设现有 solver 会遍历数据并触发优化。
- GT 生成阶段会更新统计或权重，但没有证明该副作用不会污染目标或重复计数。
- 保存了 PTQ 参数但没有 `load_ptq_params()`。
- 统计 buffer 没进入 `export_ptq_params()`，eval/deploy 无法复现训练态。
- 敏感度 score 没有输入与聚合契约，或把跳过 solver 等同于保持浮点。

## 自验

- 算法注册：`pytest tests/unit_test/algorithms/test_algorithms_registry_factory.py`。
- 算法自身：新增/更新 `tests/unit_test/algorithms/test_<algo>.py`。
- 分流与 quantizer：`pytest tests/unit_test/quantization/modules/test_quant_base.py`。
- 涉及 solver 训练：`pytest tests/unit_test/common/optimization/test_blockwise_solver.py`。
- 涉及统计/一次性优化：覆盖 GT 阶段不污染、observe 次数、无 trainable 参数的数据遍历、finalize 幂等和保存回载。
- 涉及敏感层/跳层：覆盖 score 聚合、决策持久化、续跑、strict load、eval/deploy 的一致语义。
- 涉及 deploy hook：`pytest tests/unit_test/workflows/test_llm_deploy.py`。
