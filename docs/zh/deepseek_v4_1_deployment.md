# DeepSeek V4.1 tensor deployment

本适配器只支持官方 DeepSeek-V4.1-Flash 权重的 `tensor + legacy + ascend950 + mxfp`
部署转换，不构建模型或 tokenizer，不实现 PTQ。

## 用法

```bash
python -m amct_pytorch.cli.llm.deploy \
  --model /path/to/DeepSeek-V4.1-Flash \
  --model_name deepseek_v4_1 \
  --granularity tensor \
  --deploy_format legacy \
  --deploy_platform ascend950 \
  --quant_dtype mxfp \
  --quant_layers_config amct_pytorch/configs/deploy/deepseek_v4_1_tensor_mxfp.json \
  --output_dir /path/to/DeepSeek-V4.1-Flash-950
```

输出目录必须不存在或为空，且不能与输入目录重叠。需要支持
`torch.float8_e8m0fnu` 和对应 safetensors 序列化的运行环境。
V4.1 必须显式指定 `deploy_platform`；旧模型不指定该参数时维持原行为。
本路径保留源位宽，不接收 `bit_config` 或 PTQ 参数。

## 选层与转换规则

JSON 延用 `quant_layers` / `ignore_layers` 模块名 glob，名称不带 `.weight` 后缀，
ignore 优先。示例 `*` 覆盖整个 checkpoint；实际只转换满足以下条件的层：

- 同一分片内存在 `.weight` 和 `.scale`。
- weight 是二维 `float8_e4m3fn`，形状 `[N,K]`，N、K 为 32 的正整数倍。
- scale 是二维 `float8_e8m0fnu`，形状 `[N/32,K/32]`。
- scale 按 uint8 字节视图沿 dim 0 重复每行 32 次，再恢复原 dtype，得到 `[N,K/32]`。

weight 不反量化、不重新量化，字节保持不变。FP4 专家、已是 1×32 的 scale、BF16/FP32
以及其他不满足条件的张量原样保留。不跨分片查找配对，与参考脚本一致。
因此本路径面向官方同分片配对 checkpoint；不保证重新分片后的 checkpoint 满足部署格式。

在输出之前只读取分片头检查处理目标。JSON 必须覆盖所有符合条件的 32×32 FP8 层，
遗漏或 ignore 掉其中任意一层都会报错，避免部分转换后写出全局 MXFP 配置。
JSON 匹配到的 FP4 专家不计入实际处理层数，也不会因此加入 config 的 ignore。
转换接口保持 `{tensor_name: shard_name}` 返回值，workflow 汇总并刷新权重索引。

## config 刷新

保持 `text_config`、`vision_config` 和其他字段不变，替换顶层 `quantization_config`：

- `compressed-tensors`、`float-quantized`、`quantization_status=compressed`。
- Linear 使用 MXFP W8A8，MoEGMM 使用 MXFP W4A8，group_size=32。
- `weight_block_size=[1,32]`。
- cache scheme 按参考脚本设置：KV 8 bit，compressed cache 和 indexer cache 4 bit。
- ignore 按 KV/index source layers 生成，并包含 `head`。

兼容 `kv_source_layers` / `kv_source_layer_ids` 以及
`index_source_layers` / `index_source_layer_ids`。同一组两者同时存在时必须一致；
缺失或冲突在输出前报错。嵌套 `text_config` 优先，也支持平铺配置。

## 扩展接口与当前边界

模型提供 `prepare_tensor_deploy_plan`、`convert_tensorwise_shard`、`refresh_config`，
workflow 优先调用这些接口，缺失时保留通用路径；自定义接口报错不会回退。
分片接口的参数、返回值与通用 `_convert_tensorwise_shard` 一致，自定义实现负责落盘。
`validate_tensorwise_result(plan)` 返回实际处理模块集合，由 workflow 核对计划。

`make_ascend_description` 已接入通用 workflow 的自定义优先分支，但 V4.1 的 ascend
描述接口仅占位。平台决定实现，dtype 校验组合：`A3/int` 与 `ascend950/mxfp`。
A3 转换/config 刷新及两个平台的 ascend description 均明确抛出 `NotImplementedError`；
CLI 在开始转换前拒绝 A3 与 ascend 模式。

参考：cann-recipes-infer 的 `models/deepseek_v4_1/utils/convert_model.py` 和
`convert_config.py`（本次按用户提供的本地副本核实）。
