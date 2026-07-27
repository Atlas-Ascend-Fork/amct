# 子能力：dtype-quant-module

## 触发

用户提到新量化数据类型、`quant_dtype`、INT/MXFP/HiFloat、fake quant、real quant、activation/weight quant、`QuantLinear`、`QuantizedMatmul`、bit policy、bit yaml、部署量化载荷时使用。

## 必读文件

- `amct_pytorch/quantization/dtypes/__init__.py`
- `amct_pytorch/quantization/dtypes/int.py`
- `amct_pytorch/quantization/dtypes/mxfp.py`
- `amct_pytorch/quantization/modules/quant_base.py`
- `amct_pytorch/quantization/modules/quant_linear.py`
- `amct_pytorch/quantization/modules/quant_matmul.py`
- `amct_pytorch/quantization/bit_policy.py`
- `amct_pytorch/cli/llm/args.py`
- `amct_pytorch/common/models/llm/common/deploy_export.py`
- `tests/unit_test/quantization/test_dtypes.py`
- `tests/unit_test/quantization/modules/test_quant_base.py`
- `tests/unit_test/quantization/modules/test_quant_linear.py`

## 上下游依赖

- 上游：用户入口来自 CLI `--quant_dtype`、bit policy、模型 adapter 传入的 quant config。
- 平级：`llm-algorithm` 可能包裹或改写 dtype 的 weight/activation 行为，二者必须通过 quantizer hook 协作。
- 下游：`deploy-export-code` 消费 dtype 的 `export_deploy()` 返回字段；`public-api-tests` 消费 CLI choices、docs/examples 和 config 示例。
- 若 dtype 需要 NPU 加速，还要读 `ops-boundary`，确认 Python fallback 与 `amct_ops` 软依赖策略。

## 执行步骤

1. 先确认 dtype 是 weight、activation、cache 还是多场景共用，并列出支持 bit、group、block size 和 layout。
2. 查 `DTYPE_REGISTRY` 与 `register_dtype()`，确认命名不会和现有 dtype 冲突。
3. 实现 fake quant / dequant 语义，保留 `bits == 16` 直通路径。
4. 接入 `ActivationQuantizer`、`WeightQuantizer` 或相关 quant module，避免在模型 adapter 中硬编码 dtype 分支。
5. 同步 CLI choices、bit policy、配置样例和必要的 docs/examples。
6. 若参与 deploy，实现或更新 `export_deploy()`，并检查 safetensors 字段、index 和 `quantization_config`。
7. 补 dtype、quant module、CLI 或 deploy 的最小测试。

## 实现规则

- 新 dtype 类必须通过 `DTYPE_REGISTRY.register(name=...)` 注册，并在 `register_dtype()` 中导入。
- `ActivationQuantizer` 实例化 dtype 时传 `is_act=True`，weight dtype 默认 `is_act=False`。
- dtype `forward()` 中 `bits == 16` 应保持直通，确保 bf16/float 等价路径可用。
- dtype 若参与 deploy，必须实现 `export_deploy(weight, v=...)` 并返回至少 `qweight`；额外 tensor 用稳定字段名，如 `weight_scale`、`weight_bias`。
- 若 dtype 需要 block size、packing、共享指数或特殊 layout，部署侧 `quant_payload()` / `convert_state_dict()` / `generate_quant_config()` 要同步。
- CLI `--quant_dtype choices` 要包含新 dtype，否则 workflow 无法进入。
- bit policy `_ALLOWED_BITS`、group 规则、config yaml 示例要与 dtype 支持位宽一致。

## 边界

- `quantization/dtypes/` 只描述数值量化/反量化与 deploy payload，不读取模型名。
- `quantization/modules/` 只组合 dtype + algorithm + torch module，不写某个模型族的 if 分支。
- 模型专属哪些 Linear 要量化，放 adapter 的 wrapper/binding。
- 硬件高性能实现若需要 NPU kernel，转 `amct-ops-dev`；本层只调用 `torch.ops.amct.*` 或保持 Python fallback。

## BLOCKER

- 新 dtype 只加文件但没有更新 `register_dtype()`。
- CLI 不允许该 `quant_dtype`。
- `export_deploy()` 返回字段与 deploy 写 safetensors 逻辑不匹配。
- activation 误走 weight quant 逻辑。
- bit policy 允许某个位宽，但 dtype 实际不支持。
- dtype 修改破坏 `bits == 16` 直通。

## 自验

- 运行 dtype 单测：`pytest tests/unit_test/quantization/test_dtypes.py`。
- 涉及 quant module：加跑 `tests/unit_test/quantization/modules/test_quant_base.py` 和对应 `test_quant_linear.py` / `test_quant_matmul.py`。
- 涉及 CLI choices：加跑 `pytest tests/unit_test/cli/test_llm_module_entrypoints.py` 或针对 args 的最小解析测试。
- 涉及 deploy payload：加跑 `pytest tests/unit_test/workflows/test_llm_deploy.py`。
