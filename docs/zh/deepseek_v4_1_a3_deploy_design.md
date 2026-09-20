# DeepSeek V4.1 A3 Tensorwise Deploy 适配设计

## 背景

DeepSeek V4.1 官方 checkpoint 中包含 MXFP8 和 MXFP4 格式的源权重。A3/vLLM-Ascend
目标格式为 W8A8_DYNAMIC：选中的二维权重需要导出为 INT8 `weight`、BF16
`weight_scale` 和 BF16 `weight_offset`，未选中的权重需要以 BF16 形式保留。

当前 `master` 已经提供 DeepSeek V4.1 的 deploy-only adapter 和 tensorwise workflow
扩展接口，但 A3 路径的三个 hook 尚未实现：

- `_convert_tensorwise_shard_a3`
- `_refresh_config_a3`
- `_make_ascend_description_a3`

本分支只实现 A3 + `int` + `ascend`，950/MXFP legacy 路径暂不调整。

## 目标与非目标

目标：

1. 在逐 shard 导出时，将 MXFP8/MXFP4 源权重实时解码为 BF16。
2. 复用已有 tensorwise INT8 导出逻辑，将 JSON 选中的 BF16 权重量化为 W8A8_DYNAMIC。
3. 生成与实际输出 tensor 集合一致的 `quant_model_description.json`。
4. 删除输出 `config.json` 中官方源权重携带的顶层 `quantization_config`。
5. 保持 950 legacy 路径和通用模型 Ascend 路径的现有行为不变。

非目标：

- 不修改 DeepSeek V4.1 的 PTQ、模型构建或 tokenizer 逻辑。
- 不在本分支实现 950 Ascend 或 950 legacy 的新行为。
- 不改变通用 MXFP/NVFP4 解码算法；直接复用 `convert_state_dict()`。

## 现有调用链

```text
deploy.py
  -> LlmDeployWorkflow.run()
    -> _run_tensorwise()
      -> DeepseekV41.prepare_tensor_deploy_plan()
      -> DeepseekV41.convert_tensorwise_shard()
        -> _convert_tensorwise_shard_a3()
          -> convert_state_dict()       # MXFP8/MXFP4 -> BF16
          -> quant_payload()            # BF16 -> INT8 payload
          -> adapt_ascend_payload()    # BF16 scale + zero offset
      -> make_ascend_description()
        -> _make_ascend_description_a3()
      -> refresh_config()               # A3 专属 config 刷新
      -> validate_ascend_artifacts()
```

当前 workflow 的通用 Ascend 分支默认不刷新 `config.json`。因此本适配会增加一个
模型专属 `refresh_config()` 调用点；只有 adapter 提供该 hook 时才调用，不影响没有
专属 hook 的模型。

## 权重转换设计

### 源权重解码

对每个 `.weight`：

1. 根据同名 `.scale` 构造 scale tensor 名称。
2. 使用 `convert_state_dict()`：
   - MXFP8：FP8 weight + UE8M0 block scale -> BF16；
   - MXFP4：INT8 packed weight + UE8M0 scale -> BF16。
3. 若源 weight 没有配对 scale，保持原 tensor；对选中模块最终由 Ascend payload 校验
     保证必须是二维 BF16。

对应的源 `.scale` 不写入输出，避免源格式量化辅助 tensor 残留。

### W8A8 导出

对 JSON 选中的模块调用现有 `quant_payload()`，再调用
`adapt_ascend_payload()`。这复用了 workflow 的同一套 INT8 导出实现，负责：

- 按 tensorwise bit 配置生成 INT8 weight 和 scale；
- 在 Ascend 模式下转为 BF16 scale；
- 生成全零 BF16 offset；
- 校验输入是非空二维、有限的 BF16 tensor。

未选中的权重写出解码后的 BF16 tensor，其他非权重 tensor 原样保留。

## Config 与 description

`_refresh_config_a3()` 深拷贝源 `config.json`，删除顶层 `quantization_config`，其余
字段保持不变，并以临时文件原子替换输出配置。

`_make_ascend_description_a3()` 复用通用
`amct_pytorch.common.models.llm.common.ascend_deploy_export.make_ascend_description`：

- 选中模块的 `.weight`、`.weight_scale`、`.weight_offset` 标记为 `W8A8_DYNAMIC`；
- 其他实际写出的 tensor 标记为 `FLOAT`；
- 缺少任一量化输出 tensor 时提前报错。

## 错误处理

- JSON 未覆盖实际需要处理的源量化模块：沿用现有 plan 校验并在写出前报错。
- 选中权重解码后不是二维 BF16、包含 NaN/Inf：由 `adapt_ascend_payload()` 报错。
- 源 scale 缺失：`convert_state_dict()` 保持原权重；若该模块被选中，则由 Ascend
  payload 校验阻止错误产物生成。
- 输出目录使用现有 workflow 的新目录/空目录约束，不覆盖已有产物。

## 测试计划

补充 DSV4.1 adapter 单元测试，使用小型 safetensors checkpoint 覆盖：

1. MXFP8 解码后导出 INT8 W8A8_DYNAMIC；
2. MXFP4 解码后导出 INT8 W8A8_DYNAMIC；
3. 未选中源量化权重输出为 BF16，源 `.scale` 不残留；
4. description 的三件套标记与实际输出一致；
5. `config.json` 删除顶层 `quantization_config` 并保持其他字段；
6. 950 legacy 既有测试继续通过。

## 当前限制

本设计只覆盖 A3 Ascend 导出。实际 vLLM-Ascend 服务启动和端到端精度需要在目标
Ascend 环境中进一步验证，本地测试只验证静态产物、dtype、shape、索引和描述契约。
