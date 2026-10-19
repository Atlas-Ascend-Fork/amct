# 子能力：llm-deploy

解决量化模型导出代码应该怎样修改和验证的问题，确保模型权重、分片索引、量化配置及导出文档彼此一致，并同时覆盖分块导出和逐张量导出路径。

## 职责边界

本文件覆盖导出实现代码的修改与 review。

## 调用链

```text
deploy CLI -> LlmDeployWorkflow -> adapter/dtype/algorithm 状态
  -> blockwise 或 tensorwise 导出
  -> 权重分片 + index.json + config.json
```

## 实现定位

### 主要入口

- `amct_pytorch/workflows/llm_deploy.py`：导出编排。
- `amct_pytorch/common/models/llm/common/deploy_export.py`：通用导出辅助。

文件移动或拆分后，沿下面的关键符号、调用方和实际产物重新定位；dtype、algorithm 和 adapter 仍负责各自的上游语义。

### 关键符号

以下分支互斥起步：

| 当前动作 | 搜索定义及调用方 | 确认影响后再读 |
| --- | --- | --- |
| index metadata、单/多 shard、剩余权重 | `_load_weight_index()`、`_refresh_weight_index()` | `_write_remaining_original_weights()`、下游 loader 约定 |
| `config.json` / `quantization_config` | `_refresh_config()`、`generate_quant_config()` | adapter `cache_scheme()` / `bits_scheme()`；公开字段成立时按 `validation.md` 的公共接口检查 |
| blockwise payload/binding | `_run_blockwise()`、`export_block_deploy()`、`QuantLinear.export_deploy()` | 命中 adapter 的 `iter_deploy_bindings()`、PTQ 参数回载；payload 变化时读 `quant-dtype-module` 或 `llm-algorithm` |
| tensorwise 转换 | `_run_tensorwise()`、`convert_state_dict()`、`quant_payload()` | 命中 adapter 的 `generate_tensorwise_*()`、`get_scale_name()`、`block_size()` |
| 用户文档或 deploy 产物格式 | 实际产物写入点、`docs/zh/AMCT_Pytorch_LLM.md` 的 deploy 小节 | 命中的 examples 和 `validation.md` 的文档检查 |

先用 `rg` 确认定义和调用方。当前源码中 `_refresh_config_tensor()`、`_convert_tensor()` 只有定义和直接单测，没有运行时调用方；不得把它们描述成当前 bf16/tensorwise 主路径。（核对：`grep -n "self\._convert_tensor\b\|self\._refresh_config_tensor" amct_pytorch/workflows/llm_deploy.py` 应无命中，出现命中即说明已接通、本论断过期）blockwise 与 tensorwise 当前都在结束时调用 `_refresh_config()`。（核对：`rg -n "self\._refresh_config\(" amct_pytorch/workflows/llm_deploy.py`）

### 验证入口

分别核对 workflow 产物组织与 helper 的 payload/binding 语义。测试定位与命令见 [validation.md](validation.md#场景测试索引)，覆盖目标见本文件「测试矩阵」。

## 当前事实

blockwise：

1. 复制非权重支持文件，读取或为单 shard 合成原始 index。
2. 每层构建 quant block，回载所选 PTQ 参数，通过 adapter binding 调 `module.export_deploy()`。
3. `tensor_routes` 把 qweight/scale/bias 等新 tensor 映射回被替换的原始 weight。
4. 写 `layer_xxx.safetensors`，把未替换权重重分片为 `rest_xxxxx.safetensors`。
5. 重算 index `total_size`、保留原 metadata、刷新 weight map，再写 config。

tensorwise：

1. 按原始 weight map 分 shard 读取，必要时用 scale inverse 把 FP8/MXFP 权重转换为浮点。
2. adapter 给出 quant/ignore layer；当前只对 `quant_dtype in ["int", "mxfp"]` 调 `quant_payload()`。（核对：`rg -n "quant_dtype in" amct_pytorch/workflows/llm_deploy.py`）
3. 复用原 shard 文件名写新 tensor，刷新 index 和 config。

## 前置检查

1. 先确认 blockwise/tensorwise、单/多 shard、量化或 bf16 转换范围。
2. 新 metadata/config 字段先确定字段名、层级、类型、值来源、适用路径、覆盖策略、下游消费者和旧产物兼容性；还未确认时只列候选 helper，不改签名。
3. 字段要求同时写 index 和 config 时，先确认两处是否表达同一语义以及谁是权威来源，避免复制后漂移。
4. 修改已有 helper 前必须反查所有调用方；无调用 helper 不能作为当前行为依据。
5. 新 dtype payload 先确认 `export_deploy()` 字段和 `quant_payload()`/`export_block_deploy()` 命名规则，再修改 workflow；数值与 packing 留在 dtype。
6. algorithm 有自定义 `quantize()` 或 `export_deploy()` 时，确认 PTQ 参数已回载且 hook 约定可被 `WeightQuantizer.export_deploy()` 消费。
7. adapter 只提供 binding、ignore、bits/cache scheme 等模型事实；workflow 不加入模型名分支。
8. tensorwise 逐 shard 行为要核对 `loaded_files` 生命周期和峰值内存，不把“按 shard 循环”等同于内存已受控。

## 测试矩阵

| 改动面 | 最小断言 |
| --- | --- |
| index metadata | 新旧字段、类型、同名覆盖策略、原 metadata 保留、`total_size` 重算、weight map 不丢项 |
| shard 兼容 | 有 index 的多 shard路径、无 index 的单 shard合成、源权重文件缺失报错 |
| blockwise | replaced weight 不重复写入、extra tensor 路由、空层、rest 分片、PTQ 参数回载 |
| tensorwise | FP8/非 FP8 转换、缺 scale 告警、int/mxfp payload、ignore layers、每个输出 tensor 都进 index |
| config | 原 config 字段保留、quantization_config 格式/bits/cache/ignore 一致、旧 config 缺字段可兼容 |
| 静态产物 | JSON 可解析、safetensors 文件真实存在、index 中每个文件名可解析到输出目录 |

分别用 workflow 测试覆盖产物组织、helper 测试覆盖纯数据转换；不能由 helper 单测推断 workflow 已接通。真实模型导出未运行时只能声称 mocked/单测覆盖。

## 禁止项

- 新字段的 schema、来源或消费者未确认，却直接写入 index/config。
- 把未调用 helper 当成当前主路径，或只改 helper 不接调用方。
- dtype/algorithm payload 字段变化后，safetensors 命名、`tensor_routes`、index 或 config 未同步。
- PTQ 参数未回载却导出成“PTQ 后模型”。
- quantized tensor 写出后仍重复保留原 weight，或 index 指向不存在的文件。
- `config.json` 与真实权重 dtype、bits、cache scheme 或 ignore 列表不一致。
- workflow 出现模型专属分支，或 tensorwise 改动无峰值内存分析。

## 自验

- 测试定位与命令见 [validation.md](validation.md#场景测试索引)；按上面的测试矩阵选择 workflow 或 helper 验证。
- 新 dtype/algorithm/adapter：再加跑对应子能力测试。
- 真实导出需要模型权重和目标运行环境；未执行时明确说明。
