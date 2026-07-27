# 子能力：ptq-solver-params

## 触发

用户提到 solver、blockwise/global、optimizer、scheduler、epoch loss、`trainable_params()`、PTQ 参数保存/加载、`PtqParamHandler`、`PtqParamStore`、导出回载、unit 调度时使用。

## 必读文件

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
- 若新增 solver 或 granularity，要同时读 `llm-data-workflow` 和 `public-api-tests`，同步 CLI choices、workflow 分支和测试。

## 执行步骤

1. 先确认改动是优化循环、solver factory、参数 store，还是算法状态导出回载。
2. 检查 unit 输入和 GT 来自 provider，不在 solver 内直接读文件或重放模型。
3. 收集参数时只依赖 module/algorithm 的 `trainable_params()` 或导出 hook。
4. 训练型算法要验证参数真实进入 optimizer；统计型算法要验证 buffer 进入保存和回载。
5. 保存文件名保持 `layer_<idx>_<unit.save_name>.pt`，并确认 target 目录不会混用。
6. eval/deploy 回载失败时区分“确实可选”和“本应存在却缺失”，避免 `strict=False` 吞错。
7. 补 solver factory、blockwise solver、参数 store、workflow 或新增 granularity 的最小测试。

## 主链状态

`global_solver.py` 文件存在且有注册装饰器，但当前 `register_solvers()` 只导入 `BlockwiseSolver`，主流程接通的是 `block`。文件存在不等于 workflow 可选。

## solver 规则

- solver 不写模型名分支。
- trainable 参数只从 module 的 `trainable_params()` 收集。
- 无 trainable 参数时 solver 应直接返回，不假装训练。
- reconstruction loss 默认比较 unit output 与 GT。
- epoch loss 恒为 0 或 NaN 是强失败信号，必须检查输入、GT、梯度和 unit 策略。
- 新 solver 若要进入主流程，必须更新 `register_solvers()`、CLI granularity choices、workflow 分支和测试。

## 参数规则

- unit 参数文件名：`layer_<idx>_<unit.save_name>.pt`。
- `PtqParamHandler.export_unit()` 优先 module 自己的 `export_ptq_params()`，再递归子模块，最后 fallback 到 requires_grad 参数。
- `load_unit()` 优先 module 自己的 `load_ptq_params()`，再递归子模块，最后按 trainable 参数加载。
- eval/deploy 加载 PTQ 参数时，必须用训练时一致的 `--algos` 构建出相同子模块结构。

## BLOCKER

- 算法有状态但没有导出/加载闭环。
- 参数目录按 target 混用，导致 attn/MLP/MoE 互相加载。
- strict=False 下吞掉本应存在的参数而无日志说明。
- 新 granularity 只加 solver 文件，没接 CLI/workflow/register。
- solver 自己读离线输入文件，绕过 data provider。

## 自验

- `pytest tests/unit_test/common/optimization/test_base_solver.py tests/unit_test/common/optimization/test_blockwise_solver.py tests/unit_test/common/optimization/test_factory.py`
- 涉及注册：加跑 `tests/unit_test/common/optimization/test_global_solver.py` 或新增对应测试。
- 涉及 PTQ workflow：`pytest tests/unit_test/workflows/test_llm_ptq.py`。
- 涉及参数 store：补针对 `PtqParamHandler/Store` 的 unit 测试或模型 adapter mocked 测试。
