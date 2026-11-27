# Python API 参考

> 本文说明可以被用户依赖的 AMCT Python API，并区分 LLM 工作流接口和内部实现。

## 1. API 分类

AMCT 的 Python 接口分两层：

| 层级 | 内容 | 依赖建议 |
|---|---|---|
| **LLM 工作流接口**（推荐） | 4 个 workflow 类（`amct_pytorch.workflows.*`）+ `python -m amct_pytorch.{eval,ptq,deploy,extract_ptq_data}` 模块入口 | 面向 LLM 量化工作流的稳定入口；本文第 2 节逐个说明 |
| 内部实现 | 模型适配器（`amct_pytorch/common/models/llm/` 各 adapter）、`BaseModel`、`PtqUnit`、`PtqParamStore`、quantization 模块内部类 | **不是公共 API**——无兼容性承诺，升级版本可能随任变更，不要在脚本中依赖 |

判断标准：import `amct_pytorch.workflows` 下的 4 个 workflow 类、`python -m amct_pytorch.*` 模块入口的用法是受支持的；import 路径进入 `common/models/`、`quantization/modules/`、`algorithms/` 内部的用法属于内部实现。

## 2. LLM 工作流接口

LLM 量化主流程以 CLI 为推荐入口；需要在脚本中编排时，用 4 个 workflow 类。它们的构造签名一致（接收一个 argparse 风格的 `args` namespace），入口统一为 `run()`：

| 接口 | 模块路径 | 职责 |
|---|---|---|
| `LlmEvalWorkflow` | `amct_pytorch.workflows.llm_eval` | BF16 / 量化模型 PPL 评估（block 或 model 粒度） |
| `LlmExtractPtqDataWorkflow` | `amct_pytorch.workflows.llm_extract_ptq_data` | 提取 PTQ 校准数据（仅 block 粒度） |
| `LlmPtqWorkflow` | `amct_pytorch.workflows.llm_ptq` | 训练并保存 PTQ 量化参数（仅 block 粒度） |
| `LlmDeployWorkflow` | `amct_pytorch.workflows.llm_deploy` | 部署导出（CLI 仅 tensor 粒度） |

### LlmEvalWorkflow

- 模块路径：`amct_pytorch.workflows.llm_eval.LlmEvalWorkflow`
- 签名：`__init__(self, args)`，`run(self)`
- 参数：`args` 为 `argparse.Namespace`，字段与 CLI 同名（`model`、`model_name`、`eval_mode`、`granularity`、`quant_dtype`、`algos`、`dump` 等）。注意：workflow 不解析 `--bit_config`，也不注入 CLI 默认值——量化位宽必须用 `bit_policy=BitPolicy.from_yaml(path)` 或显式 `w_bits`/`a_bits` 提供，其余字段也要显式设置（完整清单见[CLI 参考](02_cli.md)第 2、3 节）
- 返回：`run()` 无返回值；PPL 写入 `{output_dir}/logs/eval_{model_name}_{eval_mode}.log` 的汇总行（包含 `PPL: <值>`）
- 最小调用示例：

```python
import argparse
from amct_pytorch.workflows.llm_eval import LlmEvalWorkflow

args = argparse.Namespace(
    model="/path/to/Qwen3-4B",
    model_name="qwen3",
    eval_mode="bf16",
    granularity="block",
    device="npu:0",
    seq_len=2048,
    nsamples=128,
    seed=42,
    trust_remote_code=False,
    dump=False,
    output_dir="./outputs/eval-bf16",
    quant_target=[],
    quant_dtype="",
    algos=[],
    data_dir="",
    bit_config=None,
)
LlmEvalWorkflow(args).run()
# PPL 从 ./outputs/eval-bf16/logs/eval_qwen3_bf16.log 的汇总行读取
```

- 量化评估：把示例中的 `bit_config=None` 换成 `bit_policy=BitPolicy.from_yaml("amct_pytorch/configs/w4a8.yaml")`，并从 `amct_pytorch.quantization.bit_policy` 导入 `BitPolicy`；只传 `bit_config` 路径不会生效
- 版本和设备限制：Ascend NPU；`granularity` 仅 `block` / `model`；`eval_mode=bf16` 要求 bit 配置无低于 16bit 的条目
- 常见异常：`eval_mode=bf16` 且 bit 配置含低于 16bit 的条目 → `ValueError`；加载 PTQ 参数时 `--algos` 与训练不一致 → `KeyError: Submodule ... is not found`

### LlmExtractPtqDataWorkflow

- 模块路径：`amct_pytorch.workflows.llm_extract_ptq_data.LlmExtractPtqDataWorkflow`
- 签名：`__init__(self, args)`，`run(self)`
- 参数：`quant_target` 必须恰好一个；`data_dir` 必填；`granularity` 仅 `block`
- 返回：`run()` 写出校准数据目录（见[输出产物参考](05_artifacts.md)第 2 节）
- 最小调用示例：构造 Namespace 后同样 `LlmExtractPtqDataWorkflow(args).run()`（字段同上，`quant_target=["mlp"]`、`data_dir="./ptq-data"`）
- 常见异常：`quant_target` 为空或多个 → 校验报错；`data_dir` 缺失 → 校验报错

### LlmPtqWorkflow

- 模块路径：`amct_pytorch.workflows.llm_ptq.LlmPtqWorkflow`
- 签名：`__init__(self, args)`，`run(self)`
- 参数：`quant_target` 单目标；`data_dir` 必填；`start_block_idx` / `end_block_idx`（排他上界）；PTQ 参数目录三选一（`attn_linear_param_dir` / `attn_cache_param_dir` / `moe_mlp_param_dir`），未指定时自动创建默认目录
- 返回：`run()` 默认写出 `ptq_params/{model_name}/{quant_target}/layer_*.pt`，显式传参数目录时写到指定目录；断点续跑语义与 CLI 相同
- 最小调用示例：同上模式，并补充以下字段：
  - `quant_target=["mlp"]`
  - `bit_policy=BitPolicy.from_yaml("amct_pytorch/configs/w4a8.yaml")`（`BitPolicy` 通过 `from amct_pytorch.quantization.bit_policy import BitPolicy` 导入）
  - `quant_dtype="int"`、`algos=["lwc", "lac"]`
  - `epochs=15`、`base_lr=1e-5`、`start_block_idx=0`、`end_block_idx=<模型层数>`
  - 未设置 `bit_policy` 时按 16bit 处理（等于不量化）；缺少 `epochs` / `base_lr` / `start_block_idx` / `end_block_idx` 会直接报 `AttributeError`
- 常见异常：找不到校准数据 → `Run extract_ptq_data first.`；`end_block_idx` 超出层数自动钳制并告警

### LlmDeployWorkflow

- 模块路径：`amct_pytorch.workflows.llm_deploy.LlmDeployWorkflow`
- 签名：`__init__(self, args)`，`run(self)`
- 参数：`granularity="tensor"` + `quant_layers_config`（JSON 路径）；`deploy_platform` 默认解析为 `A3`，但 DeepSeek V4.1 的 tensor 部署必须显式指定
- 返回：`run()` 写出部署产物五件套并执行内置校验（见[输出产物参考](05_artifacts.md)第 3、4 节）
- 最小调用示例：同上模式，`granularity="tensor"`、`quant_layers_config="./quant_layers.json"`
- 版本和设备限制：源模型必须是未量化、未旋转的 BF16 checkpoint（DeepSeek V4/V4.1 的 FP8 官方源例外）；输出目录必须为空或不存在
- 常见异常：CLI 路径下 `granularity != "tensor"` 或缺 `quant_layers_config` → `ValueError`；JSON 规则不合法 → 加载阶段报错；校验失败 → `Deployment artifact validation failed; see {validation_path}`

### 模块入口

`python -m amct_pytorch.<command>` 等价于 CLI 调用（`main()` 内部转发到对应 CLI 模块）：

```text
amct_pytorch.eval / amct_pytorch.extract_ptq_data / amct_pytorch.ptq / amct_pytorch.deploy
```

它们没有可复用的函数级 API——`main()` 读取 `sys.argv`。脚本编排请直接用上面的 workflow 类。

## 3. 兼容性

**公共 API 与内部模块的边界**：第 2 节的 4 个 workflow 类、模块入口和 CLI 属于公共 API，升级时保持「构造 args → run()」的调用形态稳定。`args` 的字段集合随功能演进可能增加，建议用 CLI 参数名构造 namespace 并显式设置全部用到的字段（如示例所示），不要依赖默认注入；`bit_config` 只在 CLI 和 deploy 的参数准备阶段被解析，直接构造 namespace 时要自己转成 `bit_policy`。

**内部实现无兼容承诺**：模型适配器（`MODEL_REGISTRY` 注册的各 adapter 类）、`BaseModel`、`PtqUnit`、`PtqParamStore`、量化模块（`QuantLinear` 等）和算法实现的内部结构会随版本重构。一个现成例子：`LlmDeployWorkflow` 曾经历 tensorwise 重构——CLI 的 blockwise 部署入口被移除，仅 workflow 类的程序化调用可达；直接依赖旧 CLI 行为的脚本随即失效。凡 import 进 `common/`、`quantization/modules/`、`algorithms/` 内部路径的用法都可能遇到同类变化。

**懒加载机制**：`amct_pytorch` 顶层部分导出名使用 PEP 562 懒加载，访问时才触发加载。环境探测代码不要遍历顶层 `__all__`，那会意外拉起重依赖（如 onnx/protobuf）。

**程序化 deploy 的现状**：`LlmDeployWorkflow` 支持通过预先设置私有属性 `args._deploy_args_prepared` 跳过 CLI 参数校验以到达 blockwise 路径；这是内部机制，随重构演进，不构成稳定接口。blockwise 路径不生成 `deployment_validation.json` 自检报告，产物需自行校验（说明见[输出产物参考](05_artifacts.md)第 3 节）。

**版本升级建议**：升级 AMCT 后，PTQ 参数文件与部署产物建议重新生成（产物结构随版本演进，跨版本不承诺兼容）；工作流脚本先跑 `--help` 与最小评估确认参数面无变化。
