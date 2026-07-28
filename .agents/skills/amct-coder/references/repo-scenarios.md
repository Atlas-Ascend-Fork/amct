# AMCT 编码场景总览

## 先验结论

AMCT 当前至少有三条代码体系：

1. **LLM new-path**：`cli/llm` -> `workflows` -> `common/models/llm` -> `quantization` / `algorithms/quant` -> `common/optimization` -> `common/datasets` -> deploy safetensors。
2. **classic PTQ / graph_based**：`amct_pytorch.classic` 与 `classic/graph_based/amct_pytorch`，围绕 config、ONNX graph、optimizer pass、record_file、custom_op、QAT/剪枝/蒸馏。
3. **amct_ops**：独立 NPU 自定义算子包，通过 `torch.ops.amct.*` 暴露给上层。

编码前先确认任务属于哪条体系。不要把其中一条体系的注册、配置或测试规则套到另一条。

## 体系依赖拓扑

### LLM new-path

主路径不是单个线性文件，而是多个契约的组合：

1. `cli/llm/args.py` 定义用户入口、choices 和默认值。
2. `workflows/llm_*.py` 只负责编排，不承载模型、算法或 dtype 特例。
3. `common/models/llm/...` 提供 adapter、float block、quant block、`PtqUnit`、输入捕获、权重绑定和 deploy binding。
4. `quantization/dtypes/` 定义数值量化和 deploy payload；`quantization/modules/` 组合 dtype、算法和 torch module。
5. `algorithms/quant/` 通过 `ALGO_REGISTRY` 和 `targets` 注入 weight、activation 或 structure 路径。
6. `common/datasets/` 保存和加载 `PtqUnit` 所需的输入、GT 和 kwargs。
7. `common/optimization/` 消费 quant block、unit、data provider 和 algorithm 参数，输出 PTQ 参数。
8. `llm_eval.py`、`llm_deploy.py` 重建同一 adapter / dtype / algorithm 结构，加载 PTQ 参数后评测或导出。
9. `amct_pytorch/__init__.py`、`setup.py`、docs/examples 只在用户可见接口变化时同步。

上游契约不清时，先回到提供契约的层确认，不要在下游写兜底分支。例如 deploy 载荷不清先查 dtype/algorithm 的 `export_deploy()`，unit 输入不清先查 adapter 的 `iter_ptq_units()` 和 extract 命名。

### classic / graph_based

classic eager 依赖 `common/config`、`classic/optimizer`、`classic/quantize_op`、`classic/deploy_op` 和 `AlgorithmRegistry`。graph_based 依赖 ONNX graph、配置解析、record file、custom op 和 optimizer pass。两者可以通过顶层 public API 暴露，但不依赖 LLM `ALGO_REGISTRY`、`PtqUnit` 或 safetensors deploy。

### amct_ops

`amct_ops` 独立构建 wheel，通过 `torch.ops.amct.*` 或 Python wrapper 被上层调用。`amct_pytorch` 可以声明软依赖和 fallback，但不能在 import 阶段强制依赖未安装的 ops wheel，也不能把 kernel/C++ 实现写进 Python 量化层。

## 入口文件

LLM new-path：

- `amct_pytorch/cli/llm/args.py`
- `amct_pytorch/workflows/llm_eval.py`
- `amct_pytorch/workflows/llm_extract_ptq_data.py`
- `amct_pytorch/workflows/llm_ptq.py`
- `amct_pytorch/workflows/llm_deploy.py`
- `amct_pytorch/common/models/llm/__init__.py`
- `amct_pytorch/common/models/llm/common/base.py`
- `amct_pytorch/common/models/llm/common/quant_apply.py`
- `amct_pytorch/common/models/llm/common/ptq_units.py`
- `amct_pytorch/common/models/llm/common/ptq_params.py`
- `amct_pytorch/common/datasets/ptq_io.py`
- `amct_pytorch/common/datasets/ptq_provider.py`
- `amct_pytorch/common/optimization/blockwise_solver.py`
- `amct_pytorch/quantization/modules/quant_base.py`
- `amct_pytorch/quantization/modules/quant_linear.py`
- `amct_pytorch/quantization/modules/quant_matmul.py`
- `amct_pytorch/quantization/dtypes/__init__.py`
- `amct_pytorch/algorithms/quant/__init__.py`

classic / graph_based：

- `amct_pytorch/classic/quantize.py`
- `amct_pytorch/algorithms/__init__.py`
- `amct_pytorch/algorithms/register_algo.py`
- `amct_pytorch/classic/optimizer/*.py`
- `amct_pytorch/classic/quantize_op/*.py`
- `amct_pytorch/classic/deploy_op/*.py`
- `amct_pytorch/classic/graph_based/amct_pytorch/quantize_tool.py`
- `amct_pytorch/classic/graph_based/amct_pytorch/quant_calibration_interface.py`
- `amct_pytorch/classic/graph_based/amct_pytorch/prune_interface.py`
- `amct_pytorch/classic/graph_based/amct_pytorch/distillation_interface.py`
- `amct_pytorch/classic/graph_based/amct_pytorch/accuracy_based_auto_calibration.py`
- `amct_pytorch/classic/graph_based/amct_pytorch/tensor_decompose/tensor_decompose.py`

ops / packaging / API：

- `amct_ops/README.md`
- `amct_ops/ops_build.sh`
- `amct_ops/setup.py`
- `amct_ops/*/CMakeLists.txt`
- `amct_ops/*/op_extension/register.cpp`
- `amct_ops/*/python/*/ops.py`
- `amct_pytorch/__init__.py`
- `setup.py`
- `build.sh`
- `pyproject.toml`

## 场景落点

各场景的默认落点见 `SKILL.md` §全局边界与 §子能力路由，本文件不复制。不该改的位置见 [layer-rules.md](layer-rules.md) §BLOCKER 检查。

## repo-map 与 casebook

`.agents/docs/repo-map.md` 是 LLM new-path 稳定架构说明，不是全仓实时索引。新任务至少抽查命中入口文件；只有核心边界变化才更新 repo-map。

模型适配、attention、MoE、MLA、混合 attention、自定义 modeling 任务还要读：

- `.agents/docs/casebook/cross-model-pitfalls.md`
- `.agents/docs/casebook/structure-family-pitfalls.md`
- 命中模型族的 `.agents/docs/casebook/<family>/...`
