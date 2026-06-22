# Qwen3.6-MoE Quantization on NPU
## Overview
The Tongyi team has released the Qwen3.6 series models. This practice uses the quantization tool in amct_pytorch to perform quantization, data extraction, and PTQ training on the Qwen3.6-MoE model, achieving model PPL drop within 0.1 under BF16 and A8W4 quantization, supporting deployment on the Ascend `Atlas A3 Pod` platform and `950PR/DT` platform.

---

## Hardware Requirements
Product Model: Atlas A3 Pod Series

Operating System: Linux ARM

Image Version: amct_llm_images:v1

Driver Version: Ascend HDK 25.5.1
> Use npu-smi info to check whether Ascend NPU firmware and driver are correctly installed. If installed, use the command `npu-smi info` to confirm whether the version is `25.5.1`. If not installed or the version is not `25.5.1`, please first download the [firmware and driver package](https://www.hiascend.com/hardware/firmware-drivers/community?product=7&model=33&cann=9.0.0-beta.2&driver=Ascend+HDK+25.5.1), and install it yourself according to the [guide](https://hiascend.com/document/redirect/CannCommunityInstSoftware).

---

## One-Stop Platform Guide

The one-stop platform has pre-configured deployment runtime environment. When using the one-stop platform, please follow this chapter; no need to execute docker-related steps in the standard process.

- **Model Support**: The one-stop platform environment is an Atlas A3 single-card environment
- **Environment Deployment**: The platform has set up the runtime environment; no need to obtain docker image or launch docker container.
- **CANN Path**: The CANN installation path is `/home/developer/Ascend/cann`. For scripts involving `cann_path` (such as the `source` command before weight conversion), please use this path.


> The standard operations for each step in the following quick start chapter are applicable to non-one-stop platform environments. One-stop platform users please adjust corresponding steps according to the above differences.

---

## Quick Start

### Download Source Code

  Execute the following command on each node to download amct-pytorch source code.
  ```shell
  mkdir -p /home/code; cd /home/code/
  git clone https://gitcode.com/cann/amct.git
  cd amct
  ```
### Download Dataset
  When executing eval in amct_pytorch, the required dataset will be automatically downloaded

### Download Weights

  Download [Qwen/Qwen3.6-35B-A3B original weights](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) and upload to a fixed path on each node, such as `/data/models/Qwen3.6-35B-A3B`.

### Local Package Build
  For local package build process, please check [Environment Installation & Verification](../../../README.md#installation-verification)

### Baseline Test

  After completing local package build, you can test the environment path through baseline test, providing baseline data for subsequent direct conversion quantization test and direct conversion quantization test with PTQ

  ```shell
python -m amct_pytorch.eval \
     --model /data/models/Qwen3.6-35B-A3B \
     --model_name qwen3_6_moe \
     --seq_len 4096 \
     --granularity block \
     --device npu:0 \
     --eval_mode bf16 \
     --bit_config amct_pytorch/configs/bf16.yaml
  ```
Required parameter explanation:
- seq_len: Input sequence length used for calibration and evaluation, can be adjusted according to memory
- granularity: Supports blockwise and modelwise inference, currently supports block
- eval_mode: In `quant` mode, need to synchronously configure `bit_config`; under `bf16`, can not configure or refer to sample
- bit_config: Quantization configuration file

Baseline test accuracy result:
`Wikitext2-ppl=6.2825`

For more detailed parameter explanation, please refer to [Parameter Description](../../../docs/zh/AMCT_Pytorch_LLM.md#31-general-parameters)

### Direct Conversion Quantization Accuracy Evaluation
According to YAML bit configuration, perform direct conversion quantization accuracy test, evaluate the gap with baseline accuracy. Current solution defaults to full A8W4 int quantization on `quant-target`:
  ```shell
python -m amct_pytorch.eval \
    --model /data/models/Qwen3.6-35B-A3B \
    --model_name qwen3_6_moe \
    --seq_len 4096 \
    --granularity block \
    --device npu:0 \
    --eval_mode quant \
    --quant_target attn-linear \
    --quant_dtype int \
    --bit_config amct_pytorch/configs/w4a8.yaml
  ```
Required parameter explanation:
- quant_target: Quantization object, currently quantize linear layers in Attention
- quant_dtype: Quantization data format, currently supports int, mxfp

Direct conversion quantization accuracy result:
`Wikitext2-ppl=7.0407`

For more detailed parameter explanation, please refer to [Parameter Description](../../../docs/zh/AMCT_Pytorch_LLM.md#32-ppl-evaluation-parameters)

### PTQ Data Extraction
Extract corresponding PTQ calibration dataset according to different quantization objects `quant_target`:
  ```shell
python -m amct_pytorch.extract_ptq_data \
    --model /data/models/Qwen3.6-35B-A3B \
    --model_name qwen3_6_moe \
    --seq_len 4096 \
    --granularity block \
    --device npu:0 \
    --data_dir ptq_data/qwen3_6_moe/attn-linear \
    --quant_target attn-linear
  ```
Required parameter explanation:
- data_dir: Extracted data directory

For more detailed parameter explanation, please refer to [Parameter Description](../../../docs/zh/AMCT_Pytorch_LLM.md#33-data-extraction-parameters)

### Post-Training Quantization
Introduce quantization algorithm to optimize the quantization process to reduce quantization loss, using autoround as an example:
#### Single-Card Environment
  ```shell
python -m amct_pytorch.ptq \
    --model /data/models/Qwen3.6-35B-A3B \
    --model_name qwen3_6_moe \
    --seq_len 4096 \
    --granularity block \
    --device npu:0 \
    --data_dir ptq_data/qwen3_6_moe/attn-linear \
    --quant_dtype int \
    --algos autoround \
    --bit_config amct_pytorch/configs/w4a8.yaml \
    --base_lr 1e-3 \
    --quant_target attn-linear \
    --epochs 10 \
    --output_dir ptq_result/
  ```
Required parameter explanation:
- base_lr: Learning rate, can be adjusted according to model/algorithm, etc.
- algos: Quantization algorithm used, currently supports lwc/lac/omniquant/autoround
- output_dir: PTQ training result save path
- epoches: Iteration rounds, adjust according to algorithm and optimization effect

#### Multi-Card Environment
To improve training efficiency, we provide training scripts under multi-card
For multi-card environment, please refer to script [ptq_multi_npu](https://gitcode.com/fujun19/amct_llm/blob/master/examples/ptq_multi_npu.sh)

For more detailed parameter explanation, please refer to [Parameter Description](../../../docs/zh/AMCT_Pytorch_LLM.md#35-ptq-parameters)

### Direct Conversion Quantization Accuracy Evaluation Based on Post-Training Quantization
After completing PTQ, add quantization algorithm in direct conversion quantization accuracy evaluation, compare with baseline test and direct conversion quantization accuracy without quantization algorithm, verify quantization algorithm effectiveness:
  ```shell
python -m amct_pytorch.eval \
  --model /data/models/Qwen3.6-35B-A3B \
  --model_name qwen3_6_moe \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --eval_mode quant \
  --quant_target attn-linear \
  --bit_config amct_pytorch/configs/w4a8.yaml \
  --quant_dtype int \
  --algos autoround \
  --attn_linear_param_dir ptq_result/ptq_params/qwen3_6_moe/attn-linear
  ```
Required parameter explanation:
- attn_linear_param_dir: When `quant_target` is `attn-linear`, quantization algorithm parameter save path

### Quantized Weight Exporting
After all the preceding steps are complete, export the quantized weigths for loading. The weight_map corresponding to the weigths is consistent with taht on the huggingface official website.
  ```shell
python -m amct_pytorch.deploy \
  --model /data/models/Qwen3.6-35B-A3B \
  --model_name qwen3_6_moe \
  --granularity block \
  --quant_target attn-linear \
  --bit_config amct_pytorch/configs/w4a8.yaml \
  --quant_dtype int \
  --algos autoround \
  --attn_linear_param_dir ptq_result/ptq_params/qwen3_6_moe/attn-linear \
  --output_dir ./output/Qwen3.6-35B-A8W4-INT
  ```
Required parameter explanation:
- output_dir：Path for saving the exported weights