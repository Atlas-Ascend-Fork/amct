# 子能力：deploy-export-code

## 触发

用户提到 deploy workflow、deploy-ready、safetensors、`model.safetensors.index.json`、`quantization_config`、blockwise/tensorwise 导出、`export_deploy()`、weight scale/bias、FP8 转 BF16/MXFP/INT 时使用。

## 必读文件

- `amct_pytorch/workflows/llm_deploy.py`
- `amct_pytorch/common/models/llm/common/deploy_export.py`
- `amct_pytorch/common/models/llm/common/base.py`
- `amct_pytorch/quantization/dtypes/int.py`
- `amct_pytorch/quantization/dtypes/mxfp.py`
- `amct_pytorch/quantization/modules/quant_linear.py`
- `tests/unit_test/workflows/test_llm_deploy.py`
- 相关 adapter 的 `iter_deploy_bindings()`、`generate_tensorwise_quant_layers()`、`bits_scheme()`、`cache_scheme()`

## 上下游依赖

- 上游：deploy 消费 adapter 的 deploy binding、dtype 的 `export_deploy()`、algorithm 的导出 hook、PTQ 参数 store 和原始 safetensors index。
- 平级：deploy workflow 只编排导出，不在 workflow 中重新定义 real quant 数值细节或模型结构特例。
- 下游：导出目录、`config.json`、`quantization_config`、weight index、docs/examples 是交付契约。
- 若新增 dtype、algorithm deploy hook 或 public config 字段，要同时读 `dtype-quant-module`、`llm-algorithm`、`public-api-tests`。

## 执行步骤

1. 明确导出模式是 blockwise 还是 tensorwise，以及是否需要加载 PTQ 参数。
2. 检查 adapter binding 是否能定位原始 weight、量化模块和 ignore layers。
3. 检查 dtype/algorithm `export_deploy()` 返回字段，确定 safetensors 命名和 `tensor_routes`。
4. 写出 quantized tensor 后刷新 index，并确保被替换的原始 weight 不重复保留。
5. 处理单 shard 和多 shard 两种原始权重布局。
6. 更新 `config.json` 的 `quantization_config`、bits scheme、cache scheme 和 ignore 列表。
7. 补 mocked safetensors/index、PTQ 参数回载、新 dtype payload 或 adapter binding 的最小测试。

## 导出路径

blockwise:

1. 复制非权重支持文件。
2. 加载原始 weight index。
3. 每层 `build_quant_block()`。
4. `load_selected_layer_ptq_params()`。
5. `export_block_deploy()` 遍历 `QuantLinear.export_deploy()`。
6. 每层写 `layer_xxx.safetensors`。
7. 剩余原始权重写 `rest_xxxxx.safetensors`。
8. 刷新 index 与 `config.json`。

tensorwise:

1. adapter 给出 quant/ignore layer。
2. 按原 safetensors 文件流式转换。
3. dtype `export_deploy()` 产出 qweight/scale。
4. 刷新 index 与 config。

## 规则

- dtype 的 real quant 细节应在 dtype 类或 algorithm `export_deploy()` 中，不写在 workflow 分支里。
- `tensor_routes` 用于标记新 tensor 替换了哪个原始 weight，避免重复保留旧权重。
- `quant_ignore_layers` 应收集未量化 Linear，写入 config ignore。
- 单 shard 模型要合成 index，不能假设一定有 `model.safetensors.index.json`。
- `config.json` 中 `quantization_config` 要与 dtype、cache scheme、bits scheme 一致。

## BLOCKER

- 新 dtype deploy 载荷字段 deploy workflow 不认识。
- PTQ 参数未加载却导出成“PTQ 后模型”。
- quantized tensor 写出后 index 未刷新或遗漏剩余原始权重。
- `config.json` 保留了与实际权重不一致的 quantization_config。
- tensorwise 直接加载所有 shard 导致内存不可控。

## 自验

- `pytest tests/unit_test/workflows/test_llm_deploy.py`。
- 新 dtype：加跑 dtype/quant module 单测。
- 新 adapter deploy：用 mocked safetensors/index 测试 `iter_deploy_bindings()` 与 ignore 列表。
- 真实导出若未跑，明确缺模型权重/NPU/CANN。
