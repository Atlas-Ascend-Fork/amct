# 子能力：classic-graph

## 触发

用户提到 classic `quantize()`、`convert()`、`algorithm_register()`、AWQ/GPTQ/SmoothQuant/MinMax/OFMR/Cast/Quantile 的旧路径、graph_based、ONNX graph、record_file、QAT、校准、剪枝、蒸馏、auto calibration、tensor decomposition 时使用。

## 必读文件

- `amct_pytorch/classic/quantize.py`
- `amct_pytorch/algorithms/__init__.py`
- `amct_pytorch/algorithms/register_algo.py`
- `amct_pytorch/common/config/config.py`
- `amct_pytorch/common/config/parser.py`
- `amct_pytorch/classic/optimizer/*.py`
- `amct_pytorch/classic/quantize_op/*.py`
- `amct_pytorch/classic/deploy_op/*.py`
- `amct_pytorch/classic/graph_based/amct_pytorch/quantize_tool.py`
- `amct_pytorch/classic/graph_based/amct_pytorch/quant_calibration_interface.py`
- `amct_pytorch/classic/graph_based/amct_pytorch/prune_interface.py`
- `amct_pytorch/classic/graph_based/amct_pytorch/distillation_interface.py`
- `amct_pytorch/classic/graph_based/amct_pytorch/accuracy_based_auto_calibration.py`
- `amct_pytorch/classic/graph_based/amct_pytorch/tensor_decompose/tensor_decompose.py`

## 上下游依赖

- classic eager 上游是 config/parser 和 `AlgorithmRegistry`，下游是 `quantize()`、`convert()`、deploy op 和 public API。
- graph_based 上游是 ONNX graph、Configuration、record_file 和 custom_op，下游是 save_model、QAT/retrain/prune/distill/auto calibration 输出。
- 两者只在 packaging/public API 层和 LLM new-path 相邻；算法注册、PTQ unit、safetensors deploy 均不共享。
- 涉及顶层导出、懒加载、build 或 docs/examples 时，同时读 `public-api-tests`。

## 执行步骤

1. 先区分 classic eager、graph_based、QAT/retrain、prune、distill、auto calibration 或 tensor decomposition。
2. 修改 classic 算法时，同步 config、quant op、deploy op、`AlgorithmRegistry` 和对应测试。
3. 修改 graph optimizer pass 时，同时检查 graph node、custom_op、record_file 和 parser 的语义。
4. 新增 public API 时，用懒加载或显式 re-export 保持 `import amct_pytorch` 轻量。
5. 涉及 protobuf/onnx 依赖时，确认 LLM-only import 不被拖入强依赖。
6. 补 classic、graph_based、config、public API 或 testcase_python 的最小测试。

## classic PTQ 规则

- `amct_pytorch.classic.quantize.quantize()` 走 `parse_config()` -> `InsertQuantizeModulePass`。
- `convert()` 走 `ReplaceNpuQuantModulePass`。
- classic 算法注册使用 `AlgorithmRegistry.register(name, src_op, quant_op, deploy_op)`。
- quant op 必须继承或兼容 `BaseQuantizeModule`。
- deploy op 是 classic 部署替换模块，不是 LLM safetensors deploy payload。
- 预定义 config 常量在 `common/config/config.py`，顶层 API 需要 re-export。

## graph_based 规则

- 入口包括 `create_quant_config()`、`quantize_model()`、`save_model()`、`create_quant_cali_config()`、`create_quant_retrain_config()`、`create_prune_retrain_model()`、`accuracy_based_auto_calibration()` 等。
- graph_based 依赖 ONNX parse、Configuration、RecordFileParser、SingletonScaleOffsetRecord、optimizer pass。
- 修改 optimizer pass 时要检查 graph/node/custom_op/record_file 的配套语义。
- QAT / retrain / prune / distill 分别有独立 config 与 custom_op，不要互相偷用字段。
- tensor decomposition 只支持特定层类型时要保留显式校验。

## 与 LLM new-path 的边界

- classic `AlgorithmRegistry` 中有 gptq/awq/smoothquant，不代表 LLM PTQ `--algos` 可用。
- classic deploy op 不等于 LLM `export_deploy()` payload。
- graph_based public API 懒加载到 `amct_pytorch.__getattr__`，不要让 LLM import 强依赖 protobuf/onnx。

## BLOCKER

- 修改 classic 算法却只更新 LLM `ALGO_REGISTRY`。
- 新 classic public API 没进 `__all__` 或顶层懒加载测试。
- record_file 为空仍继续 save_model。
- optimizer pass 改图但不更新对应 parser/custom_op 测试。
- graph_based import 破坏 `import amct_pytorch` 的轻量路径。

## 自验

- classic PTQ：`pytest tests/unit_test/classic tests/amct_pytorch/test_quantize.py tests/amct_pytorch/test_customized_algo.py`。
- graph_based：按改动目录跑 `tests/amct_pytorch/testcase_python/<area>/...`。
- public API：`pytest tests/unit_test/test_public_api.py tests/amct_pytorch/testcase_python/test_public_forwarding.py`。
- 配置：`pytest tests/unit_test/common/config tests/amct_pytorch/test_config.py`。
