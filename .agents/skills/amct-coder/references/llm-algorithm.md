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
2. 查现有同类算法的构造签名、状态保存和测试口径，优先沿用本地基类/辅助函数。
3. 用 `@ALGO_REGISTRY.register(...)` 注册，并确保 `amct_pytorch/algorithms/quant/__init__.py` 导入。
4. 接入 quantizer 或 structure transform，检查同一位置是否已有互斥 hook。
5. 对训练型算法实现 `trainable_params()`；对统计型算法实现 buffer、`export_ptq_params()`、`load_ptq_params()`。
6. 若 eval/deploy 需要复现训练态，检查 `--algos`、参数文件和回载路径是否能构建出相同模块结构。
7. 补 registry、算法本体、quantizer 分流、solver 或 deploy 的最小测试。

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

## BLOCKER

- 新算法没有 `@ALGO_REGISTRY.register(...)`。
- registry metadata 没有 `targets`。
- `targets` 与实际作用域不一致。
- `register_algorithms()` 没导入，导致运行时不可选。
- 有 trainable 参数但 `trainable_params()` 返回空。
- 保存了 PTQ 参数但没有 `load_ptq_params()`。
- 统计 buffer 没进入 `export_ptq_params()`，eval/deploy 无法复现训练态。

## 自验

- 算法注册：`pytest tests/unit_test/algorithms/test_algorithms_registry_factory.py`。
- 算法自身：新增/更新 `tests/unit_test/algorithms/test_<algo>.py`。
- 分流与 quantizer：`pytest tests/unit_test/quantization/modules/test_quant_base.py`。
- 涉及 solver 训练：`pytest tests/unit_test/common/optimization/test_blockwise_solver.py`。
- 涉及 deploy hook：`pytest tests/unit_test/workflows/test_llm_deploy.py`。
