# DeepSeek V4.1 tensor deployment

本适配器支持官方 DeepSeek-V4.1-Flash 权重的 tensorwise 部署转换：
`legacy + ascend950 + mxfp` 和 `ascend + A3 + int`，不构建模型或 tokenizer，
不实现 PTQ。

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

A3 平台导出 W8A8 动态量化权重：

```bash
python3 amct_pytorch/cli/llm/deploy.py \
  --trust_remote_code \
  --model <MODEL_DIR> \
  --model_name deepseek_v4_1 \
  --device cpu \
  --granularity tensor \
  --quant_dtype int \
  --bit_config amct_pytorch/configs/deploy/tensor_w8a8.yaml \
  --deploy_format ascend \
  --quant_layers_config amct_pytorch/configs/deploy/dsv4_1_ascend_w8a8.json \
  --output_dir <OUTPUT_DIR> \
  --deploy_platform A3
```

其中 `<MODEL_DIR>` 是官方 DeepSeek-V4.1-Flash 源权重目录，`<OUTPUT_DIR>` 必须是
不存在或为空的目录，且不能与输入目录重叠。A3 路径会将源权重中的 MXFP/FP8/FP4
张量先转换为 BF16，再按 JSON 选择的层导出 INT8 W8A8 动态量化权重，同时生成
`quant_model_description.json`。

输出目录必须不存在或为空，且不能与输入目录重叠。需要支持
`torch.float8_e8m0fnu` 和对应 safetensors 序列化的运行环境。
V4.1 必须显式指定 `deploy_platform`；旧模型不指定该参数时维持原行为。
ascend950 路径保留源位宽，不接收 `bit_config`；A3 路径通过 `bit_config`
指定目标位宽。两条路径均不接收 PTQ 参数。

## 选层与转换规则

JSON 延用 `quant_layers` / `ignore_layers` 模块名 glob，名称不带 `.weight` 后缀，
ignore 优先。示例 `*` 覆盖整个 checkpoint；实际只转换满足以下条件的层：

- 同一分片内存在 `.weight` 和 `.scale`。
- weight 是二维 `float8_e4m3fn`，形状 `[N,K]`，N、K 为 32 的正整数倍。
- scale 是二维 `float8_e8m0fnu`，形状 `[N/32,K/32]`。
- `ascend950` 路径会将 scale 按 uint8 字节视图沿 dim 0 重复每行 32 次，
  再恢复原 dtype，得到 `[N,K/32]`；A3 路径由 DSV4.1 适配器先将官方
  `float8_e8m0fnu` scale 规范化为 MXFP 形式，再调用通用 `convert_state_dict`
  完成反量化。

ascend950 路径不反量化、不重新量化 weight，源字节保持不变；FP4 专家、已按行展开的
Engram scale、BF16/FP32 以及其他不满足条件的张量原样保留。A3 路径则在逐 shard
处理中按 index 查找配对 scale，将源 MXFP8/MXFP4 权重反量化后再执行 INT8 导出。
其中 Engram embed 的 scale 形状为 `[M, N/32]`，适配器会保留其按行展开语义；
普通 MXFP8 权重和 Engram wkv 使用 `[M/32, N/32]` 的 block scale。

在输出之前只读取分片头检查处理目标。对于 `ascend950`，JSON 必须覆盖所有符合条件的
32×32 FP8 层，遗漏或 ignore 掉其中任意一层都会报错，避免部分转换后写出全局 MXFP
配置；对于 A3，JSON 选择需要导出 INT8 的层，未选中的源量化权重会解码为 BF16。
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

`make_ascend_description` 已接入通用 workflow 的自定义优先分支。平台与 dtype
组合固定为 `A3/int` 和 `ascend950/mxfp`；A3 输出使用 vLLM-Ascend 的
`W8A8_DYNAMIC` description，950 输出保持 legacy 配置格式。

参考：cann-recipes-infer 的 `models/deepseek_v4_1/utils/convert_model.py` 和
`convert_config.py`（本次按用户提供的本地副本核实）。
