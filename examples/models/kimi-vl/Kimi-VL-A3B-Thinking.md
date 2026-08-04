# Kimi-VL-A3B-Thinking Quantization on NPU
## 概述
Moonshot AI 发布的 Kimi-VL-A3B-Thinking 是一个多模态（视觉-语言）MoE 模型，总参数 16B、每 token 激活 2.8B，BF16 权重约 32GB。本实践基于 amct_pytorch 中的量化工具，对其**语言侧 MoE 主干**做量化、数据提取与 PTQ。

模型由两部分组成：视觉塔 MoonViT 与 DeepSeek-V3 结构的语言主干（MLA 注意力 + 64 路由专家 + 2 共享专家，第 0 层为稠密 MLP）。本样例只量化语言主干的前馈部分，视觉塔与多模态投影层保持 BF16 不变。

与仓内其他一站式样例相比，本样例的特点是**单卡即可跑完全流程**：32GB 的 BF16 权重可以整模型放到一张 Atlas A3 卡上，基准评测无需多卡切分。

---

## 硬件要求
产品型号：Atlas A3 Pod 系列

操作系统：Linux ARM

镜像版本：amct_llm_images:v1

驱动版本：Ascend HDK 25.5.1
> npu-smi info 检查 Ascend NPU 固件和驱动是否正确安装。如果已安装，通过命令`npu-smi info`确认版本是否为`25.5.1`。如果未安装或者版本不是`25.5.1`，请先下载[固件和驱动包](https://www.hiascend.com/hardware/firmware-drivers/community?product=7&model=33&cann=9.0.0-beta.2&driver=Ascend+HDK+25.5.1)，并根据[指导](https://hiascend.com/document/redirect/CannCommunityInstSoftware)自行安装。

---

## 一站式平台指南

一站式平台已预置部署运行环境，使用一站式平台时请以本章节为准，无需执行标准流程中的 docker 相关步骤。

- **模型支持**：一站式平台环境为 Atlas A3 单卡环境，本模型 BF16 权重约 32GB，单卡容量充足。
- **环境部署**：平台已搭建好运行环境，无需获取 docker 镜像，也无需拉起 docker 容器。
- **CANN 路径**：CANN 安装路径为 `/home/developer/Ascend/cann`。

> 以下快速启动章节中各步骤的标准操作适用于非一站式平台环境，一站式平台用户请根据上述差异调整对应步骤。

---

## 支持范围

| quant_target | 是否支持 | 说明 |
|---|---|---|
| `moe` | 是 | 第 1~26 层的路由专家（`moe.routed`）与共享专家（`moe.shared`） |
| `mlp` | 是 | 第 0 层的稠密 MLP（`first_k_dense_replace=1`） |
| `attn-linear` | 否 | 语言主干为 `q_lora_rank=None` 的 MLA，需要单独的注意力量化模块，暂未适配 |
| `attn-cache` | 否 | 同上 |

传入 `attn-linear` / `attn-cache` 时适配层会直接报错并给出上述提示，不会静默跳过。

---

## 快速启动

### 下载源码

  ```shell
  mkdir -p /home/code; cd /home/code/
  git clone https://gitcode.com/cann/amct.git
  cd amct
  ```

### 下载数据集
  在 amct_pytorch 中执行 eval 时，会自动下载所需要的数据集

### 下载权重

  下载 [moonshotai/Kimi-VL-A3B-Thinking 原始权重](https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking) 并上传到某个固定路径下，比如 `/data/models/Kimi-VL-A3B-Thinking`。

  该权重为 `bfloat16` 格式，无需像 FP8/FP4 权重那样先做格式转换，可直接进入下面的流程。模型使用 `trust_remote_code` 加载，权重目录需要包含官方的 `modeling_kimi_vl.py` 与 `configuration_kimi_vl.py`。

### 本地编包
  本地编包的流程请查看[环境安装&验证](../../../README.md#安装验证)

### 基准测试

完成本地编包后，可以通过基准测试来测试环境通路，为后续的直转量化测试以及带 ptq 的直转量化测试提供基准数据

  ```shell
python -m amct_pytorch.eval \
     --model /data/models/Kimi-VL-A3B-Thinking \
     --model_name kimi_vl \
     --seq_len 4096 \
     --granularity block \
     --device npu:0 \
     --eval_mode bf16 \
     --bit_config amct_pytorch/configs/bf16.yaml
  ```
必要传参解释：
- seq_len：校准和评估时使用的输入序列长度，可根据显存调整
- granularity：支持 blockwise 和 modelwise 推理，目前支持 block
- eval_mode：`quant` 模式下，需同步配置 `bit_config`，`bf16` 下可不配置或参考样例
- bit_config：量化配置文件

基准测试精度结果：
`Wikitext2-ppl=30.0658`

> 说明：评测数据集为纯文本的 Wikitext2，衡量的是语言主干的能力；视觉侧不参与该指标。
> 该值明显高于同量级的纯文本模型（如 Qwen3.6-35B-A3B 为 6.2825），这是模型本身的特性而非流程异常：
> Kimi-VL-A3B-Thinking 面向多模态长思维链训练，词表 163840，在原始英文语料上的语言建模困惑度本就偏高。
> 我们用一条完全独立的路径复核过——整模型加载到单卡后按标准滑动窗口计算，得到 `30.2493`，
> 与本流程的 blockwise 结果相差 0.6% 以内（差异来自 blockwise 的 bf16 累积顺序与末位 token 处理）。
> 因此请以本基准值作为量化掉点的比较基线，不要与纯文本模型的 ppl 横向比较。

更多参数详细解释请参见[参数说明](../../../docs/zh/AMCT_Pytorch_LLM.md#31-通用参数)

### 直转量化精度评估
根据 YAML 的 bit 配置，进行直转量化精度测试，评估与基准精度的差距，当前方案默认为对 `quant-target` 做全 A8W4 的 int 量化：
  ```shell
python -m amct_pytorch.eval \
    --model /data/models/Kimi-VL-A3B-Thinking \
    --model_name kimi_vl \
    --seq_len 4096 \
    --granularity block \
    --device npu:0 \
    --eval_mode quant \
    --quant_target moe \
    --quant_dtype int \
    --bit_config amct_pytorch/configs/w4a8.yaml
  ```
必要传参解释：
- quant_target：量化对象，当前对 MoE 专家中的线性层进行量化
- quant_dtype：量化数据格式，当前支持 int、mxfp

直转量化精度结果：
`Wikitext2-ppl=29.7433`

> 该值略低于 BF16 基准（30.0658），即 A8W4 直转在本模型上基本无掉点。
> 量化后 ppl 反而下降约 1%，属于该模型损失曲面上的正常波动，并非量化未生效——
> 用同一路径将 `bit_config` 换成 `w4a4.yaml`（激活也压到 4 bit）后 ppl 升至 `78.8130`，
> 说明量化确实作用在专家线性层上。若你复现时得到与基准几乎完全相同的数值，
> 才需要怀疑 `quant_target` 未匹配到专家。

更多参数详细解释请参见[参数说明](../../../docs/zh/AMCT_Pytorch_LLM.md#32-ppl-评估参数)

### PTQ数据提取
根据不同的量化对象 `quant_target` 提取相应的 PTQ 校准数据集：
  ```shell
python -m amct_pytorch.extract_ptq_data \
    --model /data/models/Kimi-VL-A3B-Thinking \
    --model_name kimi_vl \
    --seq_len 4096 \
    --granularity block \
    --device npu:0 \
    --data_dir ptq_data/kimi_vl/moe \
    --quant_target moe
  ```
必要传参解释：
- data_dir：提取数据目录

更多参数详细解释请参见[参数说明](../../../docs/zh/AMCT_Pytorch_LLM.md#33-数据提取参数)

### Post-Training Quantization
引入量化算法，对量化过程做优化，以减少量化带来的损失，以 autoround 为例：
#### 单卡环境
  ```shell
python -m amct_pytorch.ptq \
    --model /data/models/Kimi-VL-A3B-Thinking \
    --model_name kimi_vl \
    --seq_len 4096 \
    --granularity block \
    --device npu:0 \
    --data_dir ptq_data/kimi_vl/moe \
    --quant_dtype int \
    --algos autoround \
    --bit_config amct_pytorch/configs/w4a8.yaml \
    --base_lr 1e-3 \
    --quant_target moe \
    --epochs 10 \
    --output_dir ptq_result/
  ```
必要传参解释：
- base_lr：学习率，可根据模型/算法等自行调节
- algos：所用量化算法，目前支持 lwc/lac/omniquant/autoround
- output_dir：ptq 训练结果保存路径
- epochs：迭代轮数，根据算法和优化效果进行调整

> 本模型单卡即可完成 PTQ；如需提升训练效率也可参照多卡脚本 [ptq_multi_npu](../../ptq_multi_npu.sh)。

更多参数详细解释请参见[参数说明](../../../docs/zh/AMCT_Pytorch_LLM.md#35-ptq-参数)

### 基于Post-Training Quantization的直转量化精度评估
完成 PTQ 后，在直转量化精度评估中加入量化算法，与基准测试、无量化算法的直转量化精度比对，验证量化算法的效果：
  ```shell
python -m amct_pytorch.eval \
  --model /data/models/Kimi-VL-A3B-Thinking \
  --model_name kimi_vl \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --eval_mode quant \
  --quant_target moe \
  --bit_config amct_pytorch/configs/w4a8.yaml \
  --quant_dtype int \
  --algos autoround \
  --moe_mlp_param_dir ptq_result/ptq_params/kimi_vl/moe
  ```
必要传参解释：
- moe_mlp_param_dir：`quant_target` 为 `moe` 或 `mlp` 时，量化算法系数保存路径

### 量化权重导出
完成上述所有步骤后，导出量化权重用于加载，权重对应的 weight_map 和 huggingface 官方对齐
  ```shell
python -m amct_pytorch.deploy \
  --model /data/models/Kimi-VL-A3B-Thinking \
  --model_name kimi_vl \
  --granularity block \
  --quant_target moe \
  --bit_config amct_pytorch/configs/w4a8.yaml \
  --quant_dtype int \
  --algos autoround \
  --moe_mlp_param_dir ptq_result/ptq_params/kimi_vl/moe \
  --output_dir ./output/Kimi-VL-A3B-Thinking-A8W4-INT
  ```
必要传参解释：
- output_dir：导出权重保存路径

> 导出的权重保留视觉塔与多模态投影层的原始 BF16 张量，键名与官方 `language_model.` / `vision_tower.` 前缀保持一致。
