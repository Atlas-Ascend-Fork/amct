# 子能力：classic-graph

## 简介

解决 AMCT 旧版量化流程和计算图优化功能应该怎样新增、修改及接入的问题，帮助区分即时执行的 classic 算法与基于计算图的优化步骤，并找到各自正确的注册、编排和验证入口。本子能力不处理大模型新量化流程中的算法和部署实现。

## 触发与体系边界

修改 classic `quantize()`/`convert()`/`algorithm_register()`、旧路径 AWQ/GPTQ/SmoothQuant/MinMax/OFMR/Cast/Quantile、graph_based ONNX pass，以及 classic/graph_based 调用链内的 QAT/retrain、quant calibration、剪枝、蒸馏、auto calibration 或 tensor decomposition 时使用。

本子能力覆盖两个彼此独立、但共享 classic/public 包空间的体系：

| 体系 | 核心入口 | 选择/执行机制 |
| --- | --- | --- |
| classic eager | `amct_pytorch/classic/quantize.py` | `AlgorithmRegistry` + classic `ModelOptimizer`，对 `nn.Module` 插入/替换 quant module |
| classic graph_based | `classic/graph_based/amct_pytorch/` 各 interface | 无统一 pass 注册表；在具体流程中显式 `GraphOptimizer.add_pass()` 或 `ModelOptimizer.add_pass()` |

两者都不使用 LLM new-path `ALGO_REGISTRY`、`--algos`、`PtqUnit` 或 safetensors deploy payload。LLM new-path 任务转相应 LLM 子能力。

“校准”不是本手册的独占关键词。只出现“校准”时先确认活跃入口和被修改契约：LLM 校准样本、离线输入、kwargs 和 GT 属于 `llm-data-workflow`；LLM 算法消费的统计属于 `llm-algorithm`；通用 observe/min-max/scale 语义属于 `dtype-quant-module`。只有确认进入上述 classic eager 或 graph_based 调用链后，才使用本手册。

## 按任务选择源码锚点

| 当前动作 | 首轮锚点 | 确认影响后再读 |
| --- | --- | --- |
| eager algorithm/register | `classic/quantize.py`、`algorithms/__init__.py`/`register_algo.py`、目标 quant/deploy op | `common/config/parser.py`、目标 optimizer pass、对应 classic 测试 |
| eager quantize/convert | `quantize()`/`convert()` 与 `InsertQuantizeModulePass`/`ReplaceNpuQuantModulePass` | config、deploy op 和目标 module 测试 |
| graph_based ONNX pass | 目标 pass、`BaseFusionPass`、`GraphOptimizer`、实际 `add_pass()` 调用点 | graph/node/record/custom_op/config 中被该 pass 读写的部分 |
| graph_based module pass | 目标 pass、graph_based `BaseModuleFusionPass`/`ModelOptimizer`、实际调用点 | PyTorch module、graph 辅助信息和对应测试 |
| prune/distill/retrain/cali/tensor decomposition | 对应 interface 和 config 类 | 只扩展该领域的 custom_op、record、parser、proto 和 tests |
| 新 public API | graph_based 子包 `__init__.py`、顶层 `amct_pytorch.__getattr__`、public forwarding 测试 | `public-api-tests`、docs/examples |

不要通读 `quantize_tool.py` 或全部 optimizer。先用类名/调用动作搜索 `add_pass()`，确定真实编排函数后读取其相邻 pass 顺序。

## 当前接通事实

- eager `quantize()` 走 `set_default_config()`/`parse_config()`，再由 classic `ModelOptimizer` 执行 `InsertQuantizeModulePass`。
- eager `convert()` 由同一 optimizer 执行 `ReplaceNpuQuantModulePass`；quant-op 到 deploy-op 的映射来自 `AlgorithmRegistry.quant_to_deploy`。
- `AlgorithmRegistry.register(name, src_op, quant_op, deploy_op)` 是 eager 算法契约，不是 graph_based pass 注册入口。
- graph_based ONNX pass 通常继承 `BaseFusionPass`，对 graph node 执行 `match_pattern(node)` 与 `do_pass(graph, node, model=None)`，由 `GraphOptimizer` 调度。
- graph_based module pass 通常继承其 `BaseModuleFusionPass`，对 `nn.Module` 执行 `match_pattern(module, name, graph)` 与 `do_pass(model, module, name, graph)`，由 graph_based `ModelOptimizer` 调度。
- `classic/graph_based/common/optimizer/BasePass` 是局部公共实现，当前不能代表全部 graph_based pass；选基类必须以目标调用链为准。
- graph_based optimizer 通过 `optimizer/__init__.py` import/re-export 后以 `opt.<Pass>` 使用；新增 pass 若走该命名空间，要同步 import 和 `__all__`。
- graph_based public API 定义在子包 `__all__`；顶层 `amct_pytorch.__getattr__` 延迟导入该子包，避免 LLM-only import 强依赖 ONNX/protobuf。

## 实施流程

### A. classic eager algorithm

1. 冻结算法名、源 module 类型、quant module、deploy module 和 config 组合。
2. 实现兼容 `BaseQuantizeModule` 的 quant op 与对应 deploy op。
3. 更新 `AlgorithmRegistry`；确认 `parse_config()` 能选择该算法和 dtype/粒度组合。
4. 核对 insert/convert 两段映射，不能只完成 fake quant 或只完成 deploy。
5. 公开配置常量或 API 变化时再补 `public-api-tests`。

### B. graph_based optimizer pass

1. 先确认执行对象是 ONNX graph node 还是 PyTorch module，选择对应 pass 基类和 optimizer。
2. 核对目标 pass 在相邻 pass 前后的数据不变量，确定唯一实际 `add_pass()` 编排点；不要把目录存在当成已接通。
3. 实现 `match_pattern`、`do_pass` 和必要的 setup/teardown，保持 graph 拓扑、anchor、initializer 与 model/module 状态一致。
4. 若调用方通过 `opt.<Pass>` 使用，同步 `optimizer/__init__.py` 的 module import、class import 和 `__all__`。
5. 只检查该 pass 实际读写的 Configuration、RecordFileParser、SingletonScaleOffsetRecord、custom_op/proto；没有依赖就不扩展。
6. 为 match 成功/失败、图修改、pass 顺序和异常输入增加目标测试。

### C. 独立 graph_based 领域

QAT/retrain、prune、distill、quant calibration、auto calibration 和 tensor decomposition 各有独立 interface/config/custom_op。改动前先确定领域，不跨领域复用字段或 record 语义；保存产物前保持已有空 record/文件检查。

## 测试选择

| 改动面 | 最小验证 |
| --- | --- |
| eager quantize/convert | `tests/unit_test/classic/` + `tests/amct_pytorch/test_quantize.py` |
| eager custom algorithm/config | `test_customized_algo.py`、`test_config.py` 和命中的 quant/deploy op 测试 |
| 单个 graph pass | `tests/amct_pytorch/testcase_python/optimizer/test_<pass>.py`，必要时加 graph/node 测试 |
| quantize/save 主流程 | `testcase_python/single_file/test_quantize_tool.py` 或 `test_quantize_model.py` |
| prune/distill/retrain/cali | 只跑对应领域目录和 configuration/custom_op 测试 |
| public forwarding | `tests/amct_pytorch/testcase_python/test_public_forwarding.py` 和 public API 测试 |

测试计划必须写出具体领域和文件，不能用 `tests/amct_pytorch/testcase_python/<area>/...` 代替已经能确定的目标。

## BLOCKER

- classic eager 或 graph_based 能力注册到 LLM `ALGO_REGISTRY`，或借用 `--algos` 选择图 pass。
- graph_based pass 无实际 `add_pass()` 调用点，或新增 `opt.<Pass>` 却漏改 optimizer import/`__all__`。
- ONNX graph pass、module pass 和 common `BasePass` 契约混用，导致 `run`/`do_pass` 签名或执行对象不匹配。
- pass 改图却未维护 node anchor、initializer、拓扑排序或被消费的 record/config/custom_op 语义。
- record_file 为空仍继续生成依赖量化参数的模型。
- 新 graph_based public API 破坏顶层懒加载，使 `import amct_pytorch` 强依赖 ONNX/protobuf。
- classic deploy op 被当成 LLM `export_deploy()` payload，或反向混用。

## 自验说明

classic/graph_based 测试可能依赖 ONNX、protobuf、NPU 或生成文件。只运行与改动领域相符的最小集合；缺依赖时说明未运行原因，不用 eager 单测替代 graph_based 集成，也不用 import 成功替代图转换结果。
