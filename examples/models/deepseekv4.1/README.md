# DeepSeek V4.1 部署权重导出

本样例介绍如何使用 AMCT 将官方 DeepSeek-V4.1-Flash 权重转换为可部署权重，支持以下两种目标环境：

| 目标平台 | 推理框架 | 导出格式 | 权重格式 |
| --- | --- | --- | --- |
| Atlas A3 | vLLM-Ascend | `ascend` | `int` / W8A8 动态量化 |
| Ascend 950（950PR/950DT） | vLLM-Ascend | `ascend` | `hifp` / HiFloat8 W8A8 |

该适配只执行 tensorwise 权重转换，不加载模型或 tokenizer，也不执行 PTQ。

## 前置条件

- 从官方渠道获取 DeepSeek-V4.1-Flash 权重，并保存为本地 Hugging Face safetensors 模型目录。
- 安装 AMCT 及其依赖，具体步骤请参见[环境安装与验证](../../../README.md#安装验证)。
- Python、PyTorch 和 safetensors 环境需要支持 `torch.float8_e8m0fnu` 及其序列化。
- 导出目录必须不存在或为空，且不能与源模型目录相同、互为父目录或子目录。
- 以下命令均在 AMCT 仓库根目录执行。

## Atlas A3（int）部署

A3 `int` 路径面向 vLLM-Ascend。它会将官方权重中的 MXFP/FP8/FP4 张量还原为 BF16，再将配置选中的模块导出为 INT8 W8A8 动态量化权重；未选中的源量化权重会转换为 BF16。

```shell
python3 amct_pytorch/cli/llm/deploy.py \
  --trust_remote_code \
  --model <MODEL_DIR> \
  --model_name deepseek_v4_1 \
  --device cpu \
  --granularity tensor \
  --quant_layers_config amct_pytorch/configs/deploy/deepseek/deepseek4_1/deepseek4_1_w8a8_int.json \
  --output_dir <A3_OUTPUT_DIR> \
  --deploy_platform A3
```

参数说明：

- `<MODEL_DIR>`：官方 DeepSeek-V4.1-Flash 权重目录。
- `<A3_OUTPUT_DIR>`：A3 部署权重的输出目录。
- `--trust_remote_code`：允许加载模型仓库中的自定义代码。
- `--model_name deepseek_v4_1`：选择 DeepSeek V4.1 模型适配器。
- `--device cpu`：在 CPU 上执行 tensor deploy 的权重处理。
- `--granularity tensor`：逐张量处理 checkpoint，不构建完整模型。
- `--quant_layers_config`：结构化 JSON 配置，通过 `pattern`、`quant_dtype=int`、`w_bits=8`、`a_bits=8` 指定量化层与方案。
- `--deploy_platform A3`：启用 DeepSeek V4.1 的 A3 转换路径。

主要输出包括：

- 转换后的 safetensors 权重分片。
- `model.safetensors.index.json`：更新后的权重索引。
- `config.json`：包含 `quant_method=ascend` 和 `model_quant_type=W8A8_DYNAMIC`。
- `quant_model_description.json`：逐张量记录 `W8A8_DYNAMIC` 或 `FLOAT` 类型。
- `deployment_validation.json`：记录权重、索引、dtype、shape 和量化描述的完整性校验结果。

导出成功后，确认 `deployment_validation.json` 中的 `status` 为 `passed`，再使用输出目录在 vLLM-Ascend A3 环境中拉起模型。

## Ascend 950 HiFloat8 部署

HiFloat8 路径面向 Ascend 950 上的 vLLM-Ascend。它将配置选中的 Linear/MoE 权重编码为 HiFloat8 W8A8，支持浮点源权重及带 scale 的 FP8 权重；未选中的权重按 FLOAT 路径处理。

```shell
python -m amct_pytorch.cli.llm.deploy \
  --model <MODEL_DIR> \
  --model_name deepseek_v4_1 \
  --granularity tensor \
  --deploy_platform ascend950 \
  --quant_layers_config amct_pytorch/configs/deploy/deepseek/deepseek4_1/deepseek4_1_w8a8_hifp.json \
  --output_dir <HIF8_OUTPUT_DIR>
```

参数说明：

- `<MODEL_DIR>`：官方 DeepSeek-V4.1-Flash 权重目录。
- `<HIF8_OUTPUT_DIR>`：供 vLLM-Ascend 使用的 HiFloat8 权重目录。
- `--model_name deepseek_v4_1`：选择 DeepSeek V4.1 模型适配器。
- `--granularity tensor`：逐张量处理 checkpoint，不构建完整模型。
- `--deploy_platform ascend950`：启用 Ascend 950 HiFloat8 转换路径。
- `--quant_layers_config`：结构化 JSON 配置，使用 `quant_dtype=hifp`、`w_bits=8`、`a_bits=8`。

转换后选中层的 weight 为 `U8 [out, in]`，scale 为 `F32 [out, 1]`；`config.json` 保持源文件不变。

## 使用限制

- DeepSeek V4.1 当前只支持 `granularity=tensor` 的部署转换，不支持通过该适配器执行 PTQ。
- A3 支持 `int` W8A8；Ascend 950 支持 `int` W8A8 和 `hifp`（HiFloat8）W8A8。
- 量化类型与位宽由 `--quant_layers_config` 中的规则指定。
- HiFloat8 路径选中的层必须是非空二维张量。支持 BF16/F16/F32/F64 及带 scale 的 FP8 输入；编码需要 NPU 以及具备原生 HiFloat8 转换支持的 `torch_npu`。
- 以上路径均不接受 `--algos`、`--attn_linear_param_dir`、`--attn_cache_param_dir` 或 `--moe_mlp_param_dir`。
