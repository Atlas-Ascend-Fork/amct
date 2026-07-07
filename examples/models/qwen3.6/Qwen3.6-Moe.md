# Qwen3.6-MoE Quantization on NPU
## 概述
通义团队发布了Qwen3.6系列模型，本实践基于amct_pytorch中的量化工具，对Qwen3.6-MoE模型做了量化、数据提取与PTQ训练，使得模型在BF16和A8W4量化下ppl掉点在0.1以内，支持在昇腾`Atlas A3 Pod`平台和`950PR/DT`平台部署。

---

## 硬件要求
产品型号：Atlas A3 Pod 系列

操作系统：Linux ARM

镜像版本：amct_llm_images:v1

驱动版本：Ascend HDK 25.5.1
> npu-smi info 检查Ascend NPU固件和驱动是否正确安装。如果已安装，通过命令`npu-smi info`确认版本是否为`25.5.1`。如果未安装或者版本不是`25.5.1`，请先下载[固件和驱动包](https://www.hiascend.com/hardware/firmware-drivers/community?product=7&model=33&cann=9.0.0-beta.2&driver=Ascend+HDK+25.5.1)，并根据[指导](https://hiascend.com/document/redirect/CannCommunityInstSoftware)自行安装。

---

## 一站式平台指南

一站式平台已预置部署运行环境，使用一站式平台时请以本章节为准，无需执行标准流程中的 docker 相关步骤。

- **模型支持**：一站式平台环境为 Atlas A3 单卡环境
- **环境部署**：平台已搭建好运行环境，无需获取 docker 镜像，也无需拉起 docker 容器。
- **CANN 路径**：CANN 安装路径为 `/home/developer/Ascend/cann`，涉及 `cann_path` 的脚本（如权重转换前的 `source` 命令）均需使用此路径。


> 以下快速启动章节中各步骤的标准操作适用于非一站式平台环境，一站式平台用户请根据上述差异调整对应步骤。

---

## 快速启动

### 下载源码

  在各个节点上执行如下命令下载 amct-pytorch 源码。
  ```shell
  mkdir -p /home/code; cd /home/code/
  git clone https://gitcode.com/cann/amct.git
  cd amct
  ```
### 下载数据集
  在amct_pytorch中执行eval时，会自动下载所需要的数据集

### 下载权重

  下载[Qwen/Qwen3.6-35B-A3B原始权重](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)并上传到各节点的某个固定的路径下，比如`/data/models/Qwen3.6-35B-A3B`。

### 本地编包
  本地编包的流程请查看[环境安装&验证](../../../README.md#安装验证)

### 基准测试

完成本地编包后，可以通过基准测试来测试环境通路，为后续的直转量化测试以及带ptq的直转量化测试提供基准数据

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
必要传参解释：
- seq_len：校准和评估时使用的输入序列长度，可根据显存调整
- granularity：支持blockwise和modelwise推理，目前支持block
- eval_mode：`quant`模式下，需同步配置`bit_config`，`bf16`下可不配置或参考样例
- bit_config：量化配置文件

基准测试精度结果：
`Wikitext2-ppl=6.2825`

更多参数详细解释请参见[参数说明](../../../docs/zh/AMCT_Pytorch_LLM.md#31-通用参数)

### 直转量化精度评估
根据YAML的bit配置，进行直转量化精度测试，评估与基准精度的差距，当前方案默认为对`quant-target`做全A8W4的int量化：
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
必要传参解释：
- quant_target：量化对象，当前对Attention中的线性层进行量化
- quant_dtype：量化数据格式，当前支持int、mxfp

直转量化精度结果：
`Wikitext2-ppl=7.0407`

更多参数详细解释请参见[参数说明](../../../docs/zh/AMCT_Pytorch_LLM.md#32-ppl-评估参数)

### PTQ数据提取
根据不同的量化对象`quant_target`提取相应的 PTQ 校准数据集：
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
必要传参解释：
- data_dir：提取数据目录

更多参数详细解释请参见[参数说明](../../../docs/zh/AMCT_Pytorch_LLM.md#33-数据提取参数)

### Post-Training Quantization
引入量化算法，对量化过程做优化，以减少量化带来的损失，以autoround为例：
#### 单卡环境
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
必要传参解释：
- base_lr：学习率，可根据模型/算法等自行调节
- algos：所用量化算法，目前支持lwc/lac/omniquant/autoround
- output_dir：ptq训练结果保存路径
- epoches：迭代轮数，根据算法和优化效果进行调整

#### 多卡环境
为提升训练效率，我们提供多卡下的训练脚本
多卡环境请参照脚本[ptq_multi_npu](https://gitcode.com/fujun19/amct_llm/blob/master/examples/ptq_multi_npu.sh)

更多参数详细解释请参见[参数说明](../../../docs/zh/AMCT_Pytorch_LLM.md#35-ptq-参数)

### 基于Post-Training Quantization的直转量化精度评估
完成PTQ后，在直转量化精度评估中加入量化算法，与基准测试、无量化算法的直转量化精度比对，验证量化算法的效果：
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
必要传参解释：
- attn_linear_param_dir：`quant_target`为`attn-linear`时，量化算法系数保存路径