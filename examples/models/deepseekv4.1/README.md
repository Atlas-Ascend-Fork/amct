# DeepSeek V4.1 部署权重导出

本样例介绍如何使用 AMCT 将官方 DeepSeek-V4.1-Flash 权重转换为可部署权重，支持以下两种目标环境：

| 目标平台 | 推理框架 | 导出格式 | 权重格式 |
| --- | --- | --- | --- |
| Atlas A3 | vLLM-Ascend | `ascend` | W8A8 动态量化 |
| Atlas A5（Ascend 950PR/950DT） | cann-recipes-infer | `legacy` | 保留源 MXFP/FP8/FP4 位宽 |
| Atlas A5（Ascend 950PR/950DT） | vLLM-Ascend | `legacy` | 注意力 Linear 转 HiFloat8 |

该适配只执行 tensorwise 权重转换，不加载模型或 tokenizer，也不执行 PTQ。

## 前置条件

- 从官方渠道获取 DeepSeek-V4.1-Flash 权重，并保存为本地 Hugging Face safetensors 模型目录。
- 安装 AMCT 及其依赖，具体步骤请参见[环境安装与验证](../../../README.md#安装验证)。
- Python、PyTorch 和 safetensors 环境需要支持 `torch.float8_e8m0fnu` 及其序列化。
- 导出目录必须不存在或为空，且不能与源模型目录相同、互为父目录或子目录。
- 以下命令均在 AMCT 仓库根目录执行。

## Atlas A3 部署

A3 路径面向 vLLM-Ascend。它会将官方权重中的 MXFP/FP8/FP4 张量还原为 BF16，再将配置选中的模块导出为 INT8 W8A8 动态量化权重；未选中的源量化权重会转换为 BF16。

```shell
python3 amct_pytorch/cli/llm/deploy.py \
  --trust_remote_code \
  --model <MODEL_DIR> \
  --model_name deepseek_v4_1 \
  --device cpu \
  --granularity tensor \
  --quant_dtype int \
  --bit_config amct_pytorch/configs/deploy/tensor_w8a8.yaml \
  --deploy_format ascend \
  --quant_layers_config amct_pytorch/configs/deploy/deepseek_v4_1_ascend_w8a8.json \
  --output_dir <A3_OUTPUT_DIR> \
  --deploy_platform A3
```

参数说明：

- `<MODEL_DIR>`：官方 DeepSeek-V4.1-Flash 权重目录。
- `<A3_OUTPUT_DIR>`：A3 部署权重的输出目录。
- `--granularity tensor`：逐张量处理 checkpoint，不构建完整模型。
- `--deploy_format ascend`：生成 vLLM-Ascend ModelSlim 兼容的部署产物。
- `--deploy_platform A3`：启用 DeepSeek V4.1 的 A3 转换路径。

主要输出包括：

- 转换后的 safetensors 权重分片。
- `model.safetensors.index.json`：更新后的权重索引。
- `config.json`：包含 `quant_method=ascend` 和 `model_quant_type=W8A8_DYNAMIC`。
- `quant_model_description.json`：逐张量记录 `W8A8_DYNAMIC` 或 `FLOAT` 类型。
- `deployment_validation.json`：记录权重、索引、dtype、shape 和量化描述的完整性校验结果。

导出成功后，确认 `deployment_validation.json` 中的 `status` 为 `passed`，再使用输出目录在 vLLM-Ascend A3 环境中拉起模型。

## Atlas A5（Ascend 950）部署

A5 路径面向 cann-recipes-infer 的 Ascend 950PR/950DT 部署格式。该路径不会反量化或重新量化权重，而是保留源权重的位宽和字节内容，并将符合条件的 32x32 FP8 scale 转换为 infer 仓需要的布局。

```shell
python -m amct_pytorch.cli.llm.deploy \
  --model <MODEL_DIR> \
  --model_name deepseek_v4_1 \
  --granularity tensor \
  --deploy_format legacy \
  --deploy_platform ascend950 \
  --quant_dtype mxfp \
  --quant_layers_config amct_pytorch/configs/deploy/deepseek_v4_1_tensor_mxfp.json \
  --output_dir <A5_OUTPUT_DIR>
```

参数说明：

- `<MODEL_DIR>`：官方 DeepSeek-V4.1-Flash 权重目录。
- `<A5_OUTPUT_DIR>`：供 cann-recipes-infer 使用的 A5 部署权重目录。
- `--deploy_format legacy`：生成 infer 仓兼容的 legacy 配置。
- `--deploy_platform ascend950`：启用 A5/Ascend 950 转换路径。
- `--quant_dtype mxfp`：保留源 MXFP 位宽语义。

转换过程中，满足条件的 scale 会从 `[N/32, K/32]` 扩展为 `[N, K/32]`。FP4 专家、已经按行展开的 Engram scale、BF16/FP32 张量及其他不满足转换条件的张量保持不变。导出流程同时刷新 `model.safetensors.index.json` 和 `config.json`，输出目录可交付 cann-recipes-infer 的 Ascend 950PR/950DT 环境使用。

## Atlas A5（Ascend 950）HiFloat8 部署

HiFloat8 路径面向 Atlas A5 上的 vLLM-Ascend。它把配置选中的注意力 Linear 从 32x32 块 FP8 反量化后重新编码为 HiFloat8，其余张量保持不变。

这条链路的完整实践与精度结果见 [DeepSeekV4.1-Flash-HiFloat8 量化Agent实践](./DeepSeekV4.1-Flash-Quantization-Agent.md)。

```shell
python -m amct_pytorch.cli.llm.deploy \
  --model <MODEL_DIR> \
  --model_name deepseek_v4_1 \
  --granularity tensor \
  --deploy_format legacy \
  --deploy_platform ascend950 \
  --quant_dtype hifp \
  --quant_layers_config amct_pytorch/configs/deploy/deepseek_v4_1_hif8.json \
  --output_dir <HIF8_OUTPUT_DIR>
```

参数说明：

- `<HIF8_OUTPUT_DIR>`：供 vLLM-Ascend 使用的 HiFloat8 权重目录。
- `--quant_dtype hifp`：启用 HiFloat8 重编码，与 `mxfp` 的保位宽路径并列。
- `--quant_layers_config`：选中 attn 的 `wq_a`/`wkv`/`wq_b`/`wo_a`/`wo_b`、indexer 的 `wq_b`、engram 的 `wkv`，以及 MTP 块的同名 Linear。

转换后选中层的 weight 为 `U8 [out, in]`，scale 为 `F32 [out, 1]`；`config.json` 保持源文件不变。

## 使用限制

- DeepSeek V4.1 当前只支持 `granularity=tensor` 的部署转换，不支持通过该适配器执行 PTQ。
- 平台与格式组合固定为：
  - A3：`--deploy_platform A3 --deploy_format ascend --quant_dtype int`。
  - A5：`--deploy_platform ascend950 --deploy_format legacy --quant_dtype mxfp`。
  - A5 HiFloat8：`--deploy_platform ascend950 --deploy_format legacy --quant_dtype hifp`。
- A5 的 `mxfp` 和 `hifp` 路径均不接受 `--bit_config`。
- HiFloat8 路径选中的层必须是 32x32 块 FP8，其余层不做重编码。
- 以上路径均不接受 `--algos`、`--attn_linear_param_dir`、`--attn_cache_param_dir` 或 `--moe_mlp_param_dir`。
- A5 mxfp 路径的配置必须覆盖 checkpoint 中所有符合条件的 32x32 FP8 层，避免只转换部分 scale 却写入全局 MXFP 配置。
