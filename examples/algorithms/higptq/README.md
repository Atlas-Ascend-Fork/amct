# AMCT大模型 HiGPTQ 量化

## 1 算法简介

HiGPTQ 是面向华为自研 **HiFloat4**（`hifx4`）格式的 GPTQ 变体算法：沿用标准 GPTQ 基于 Hessian 逆矩阵的 OBS（Optimal Brain Surgeon）误差补偿框架，但将逐列量化改造为逐 **64 元素分组**量化，以匹配 HiFloat4 的三级层次化 block scale（L1 E6M2 覆盖 64 个元素，L2/L3 micro-exponent 分别覆盖 8/4 个元素）。分组内的列必须联合量化，无法像 int/mxfp 那样按单列独立求解。HiGPTQ 当前只做**权重量化**（weight-only），激活始终保持 BF16，`bit_config` 中的激活位宽配置会被忽略。

与本目录下 [GPTQ](../gptq/README.md) 等其余样例不同，HiGPTQ 基于 AMCT 新一代 **CLI 工作流框架**（`amct_pytorch/cli/llm`）实现，通过 `python -m amct_pytorch.{extract_ptq_data,ptq,eval}` 等命令行入口驱动，配置以 `--bit_config` 指定的 YAML 文件传入；不再通过 Python 接口直接构造 `quant_cfg` dict 调用。完整的 CLI 参数说明参见 [AMCT PyTorch LLM 量化工具使用说明](../../../docs/zh/AMCT_Pytorch_LLM.md)。

## 2 量化前提

### 2.1 安装依赖

参考仓库根目录 [环境安装&验证](../../../README.md#安装验证) 完成 `amct_pytorch` 的源码构建与安装。

### 2.2 模型和数据集准备

本样例以 **Qwen3.5-9B**（dense 模型，`--model_name qwen3_5`）为例，请自行下载模型权重并保存到本地路径（如 `/path/to/Qwen3.5-9B`）。

- 校准数据：`extract_ptq_data` 从 **Pileval** 在线加载，也可通过环境变量 `PILEVAL_LOCAL_PATH` 指定本地缓存路径。
- 评估数据集：`eval` 使用 **WikiText2** 在线加载，也可通过环境变量 `WIKITEXT2_LOCAL_PATH` 指定本地缓存路径。

### 2.3 量化配置

HiGPTQ 使用内置的 [`hifloat4.yaml`](../../../amct_pytorch/configs/hifloat4.yaml) 作为 `bit_config`：权重 4bit（HiFloat4），激活保持 16bit，MoE 路由/共享专家权重同样为 4bit。对应的 CLI 参数为：

```bash
--quant_dtype hifp --algos higptq --bit_config amct_pytorch/configs/hifloat4.yaml
```

`--algos higptq` 会被 `LlmPtqWorkflow` 特判为 `calibrate` 类求解器（单次前向的闭式 Hessian 求解，无梯度/无 epoch 概念），与 `lwc`/`lac`/`autoround` 等基于梯度训练的算法互斥，不能混用。

## 3 量化示例

Qwen3.5-9B 是 dense 模型，本样例对 **attn-linear**（Attention 中的线性层）和 **mlp**（FFN）两个 `quant_target` 分别执行完整流程。`ptq`/`extract_ptq_data` 当前每次只能处理一个 `quant_target`，因此下述涉及这两个入口的步骤都需要对两个目标各跑一次。

### 3.1 提取 PTQ 校准数据

```bash
python -m amct_pytorch.extract_ptq_data \
    --model /path/to/Qwen3.5-9B \
    --model_name qwen3_5 \
    --seq_len 4096 \
    --granularity block \
    --device npu:0 \
    --data_dir ptq_data/qwen3_5/attn-linear \
    --quant_target attn-linear
```

`mlp` 重复上述命令即可：`--quant_target` 换成 `mlp`，`--data_dir` 换成 `ptq_data/qwen3_5/mlp`。

### 3.2 执行 HiGPTQ PTQ 训练（Hessian 校准）

```bash
python -m amct_pytorch.ptq \
    --model /path/to/Qwen3.5-9B \
    --model_name qwen3_5 \
    --seq_len 4096 \
    --granularity block \
    --device npu:0 \
    --data_dir ptq_data/qwen3_5/attn-linear \
    --quant_dtype hifp \
    --algos higptq \
    --bit_config amct_pytorch/configs/hifloat4.yaml \
    --quant_target attn-linear \
    --attn_linear_param_dir ptq_result/ptq_params/qwen3_5/attn-linear \
    --output_dir ptq_result/
```

`mlp` 重复上述命令即可：`--quant_target`、`--data_dir` 同 3.1 换成 `mlp` 对应的值，参数保存目录改用 `--moe_mlp_param_dir ptq_result/ptq_params/qwen3_5/mlp`（替换掉 `--attn_linear_param_dir`）。

### 3.3 量化精度评估

与 `extract_ptq_data`/`ptq` 不同，`eval` 支持一次传入多个 `--quant_target`，会加载两个目标各自的 PTQ 参数并联合应用：

```bash
python -m amct_pytorch.eval \
    --model /path/to/Qwen3.5-9B \
    --model_name qwen3_5 \
    --seq_len 4096 \
    --granularity block \
    --device npu:0 \
    --eval_mode quant \
    --quant_target mlp attn-linear \
    --quant_dtype hifp \
    --bit_config amct_pytorch/configs/hifloat4.yaml \
    --attn_linear_param_dir ptq_result/ptq_params/qwen3_5/attn-linear \
    --moe_mlp_param_dir ptq_result/ptq_params/qwen3_5/mlp
```

评估完成后，在 `output_dir/logs/`（默认 `./outputs/logs/`）下会生成包含 WikiText2 PPL 的评估日志。

### 3.4 导出 BF16 部署权重（可选）

若需要在没有 HiFloat4 内核的推理引擎（如 vLLM）上部署，可先用 `amct_pytorch.deploy` 把 HiGPTQ 校准后的权重打包成真实的 4bit HiFloat4 checkpoint，再反量化回 BF16。反量化用的 `hif4_unpack` 是打包 `hif4_pack` 的精确逆运算，数值上和直接 fake-quant 完全等价。

```bash
python -m amct_pytorch.deploy \
    --model /path/to/Qwen3.5-9B \
    --model_name qwen3_5 \
    --granularity block \
    --device npu:0 \
    --quant_target attn-linear mlp \
    --quant_dtype hifp \
    --bit_config amct_pytorch/configs/hifloat4.yaml \
    --attn_linear_param_dir ptq_result/ptq_params/qwen3_5/attn-linear \
    --moe_mlp_param_dir ptq_result/ptq_params/qwen3_5/mlp \
    --output_dir ptq_result/deploy/qwen3_5

python examples/algorithms/higptq/src/dequantize_higptq_deploy.py \
    --input_dir ptq_result/deploy/qwen3_5 \
    --output_dir ptq_result/bf16_export/qwen3_5
```

以上 3.1-3.4 的完整命令也整理在 [`src/run_higptq_example.sh`](src/run_higptq_example.sh) 中（运行前请修改脚本中的 `MODEL_PATH`）。

## 4 常见注意事项

- `ptq`、`extract_ptq_data` 每次只处理一个 `quant_target`，`eval`、`deploy` 支持多个 `quant_target` 一起加载。
- HiGPTQ 仅支持权重量化，`bit_config` 中任何 `<16bit` 的激活配置都会被忽略（运行时会打印 warning 提示）。
- `--granularity` 请使用 `block`；`--algos higptq` 不能与 `lwc`/`lac`/`autoround` 等基于梯度训练的算法混用。
- 更多通用参数（`--seq_len`、`--data_dir`、`--attn_linear_param_dir` 等）说明参见 [AMCT PyTorch LLM 量化工具使用说明 · 参数说明](../../../docs/zh/AMCT_Pytorch_LLM.md#3-参数说明)。
