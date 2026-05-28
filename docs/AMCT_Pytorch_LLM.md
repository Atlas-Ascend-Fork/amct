# AMCT PyTorch LLM 量化工具使用说明

本文档介绍了 `amct_pytorch` 中面向大语言模型（LLM）的量化工具。该工具以**命令行工作流**为核心，串联起 PPL（Perplexity，困惑度） 测量、PTQ（Post-Training Quantization，训练后量化） 数据提取、PTQ 参数训练及量化权重部署导出等关键能力。同时，借助**模型适配器**、**量化算法注册表**和**量化数据类型注册表**，实现了不同模型架构与量化策略的灵活组合。

## 1. 概述

### 1.1 能力概览

量化工具主要由以下模块组成：

| 模块 | 路径 | 作用 |
|------|------|------|
| CLI 入口 | `amct_pytorch/cli/llm/` | 提供 `eval`、`extract_ptq_data`、`ptq`、`deploy` 四类命令行入口。 |
| 工作流 | `amct_pytorch/workflows/` | 支持编排 PPL 评估、校准数据提取、PTQ 优化训练和部署导出等功能。 |
| 模型适配器 | `amct_pytorch/common/models/llm/` | 适配 DeepSeek、Qwen、LongCat、Pangu、GLM 等模型结构。 |
| 量化算法 | `amct_pytorch/algorithms/quant/` | 提供 LWC、LAC、OmniQuant、Learnable Hadamard、AutoRound 等算法组件。 |
| 量化数据类型 | `amct_pytorch/quantization/dtypes/` | 提供 `mxfp`、`int` 等量化/反量化实现。 |
| 量化配置 | `amct_pytorch/configs/` | 提供 BF16、W8A8、W4A4、W4A8 示例配置。 |

支持的典型能力：

- **基准评估**：支持在 WikiText 数据集上计算 BF16 或量化模型的困惑度（PPL）。
- **数据提取**：利用 Pileval 数据高效提取 PTQ 校准所需的输入激活及中间层结果。
- **块级优化**：基于 Block 粒度，对指定量化目标执行 PTQ 参数训练与优化。
- **模型导出**：输出适配部署环境的 `safetensors` 权重文件及对应的量化配置信息。
- **灵活配置**：通过 `bit_config` 灵活设定全局或分组的权重/激活（W/A）量化比特数。
- **算法选择**：通过 `algos` 参数自由组合并选择可训练的量化算法策略。

### 1.2 环境准备

确保模型目录为 HuggingFace/safetensors 格式，并包含 `config.json`、tokenizer 相关文件以及 `model.safetensors.index.json`。支持的模型适配器名称以及对应的大模型为：

| 模型名称（`--model_name`） | 说明 |
|------|------|
|[`deepseek_v3_2`](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/tree/main) | DeepSeek V3.2 |
|[`deepseek_v4`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main) | DeepSeek V4 |
|[`qwen3`](https://huggingface.co/Qwen/Qwen3-4B/tree/main) | Qwen3 Dense |
|[`qwen3_moe`](https://huggingface.co/Qwen/Qwen3-235B-A22B/tree/main) | Qwen3 MoE |
|[`qwen3_5`](https://huggingface.co/Qwen/Qwen3.5-4B/tree/main) | Qwen3.5 Dense |
|[`qwen3_5_moe`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/tree/main) | Qwen3.5 MoE |
|[`qwen3_6_moe`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/tree/main) | Qwen3.6 MoE |
|[`qwen3_next`](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/tree/main) | Qwen3 Next |
|[`longcat_lite`](https://huggingface.co/meituan-longcat/LongCat-Flash-Lite/tree/main) | LongCat Flash Lite |
|[`longcat_next`](https://huggingface.co/meituan-longcat/LongCat-Next/tree/main) | LongCat Next |
|[`glm5`](https://huggingface.co/zai-org/GLM-5.1/tree/main) | GLM-5.1 |

## 2. CLI入口及对应实例化操作

### 2.1 `eval`

**入口**：`amct_pytorch/cli/llm/eval.py`

**功能特性**：该入口用来进行困惑度PPL 评估，支持两种评估模式：

- BF16基准性能评估：加载原始模型 block，以 BF16 精度计算困惑度（PPL）（`eval_mode=bf16`）。
- 量化性能评估：构建量化 block，根据 `bit_config` 和 `quant_target` 启用量化器，计算量化后的困惑度 PPL（`eval_mode=quant`）。

**输出**：在`output_dir/logs/` 目录下会生成评估日志，日志中包含 PPL。

**约束**：
- 考虑到用户环境的限制，`granularity`请尽量选择block。
- `bit_config`、`model`请配置本地路径，`bit_config`对应的yaml文件可以参考仓上amct_pytorch/configs内的文件

#### 2.1.1 BF16 基线 PPL 测量

首先计算 BF16 全精度模型的困惑度（PPL），将其作为量化后精度的评估基线：

```bash
python -m amct_pytorch.eval \
  --model /path/to/model \
  --model_name qwen3 \
  --device npu:0 \
  --granularity block \
  --eval_mode bf16 \
  --bit_config amct_pytorch/configs/bf16.yaml
```

参数详细解释请参见[参数说明](#3-参数说明)。

#### 2.1.2 量化模型 PPL 测量

量化评估会重建量化 block，并根据 `bit_config` 判断是否启用量化器。

```bash
python -m amct_pytorch.eval \
  --model /path/to/model \
  --model_name qwen3 \
  --device npu:0 \
  --granularity block \
  --eval_mode quant \
  --quant_target mlp \
  --quant_dtype int \
  --bit_config amct_pytorch/configs/example_w4a8.yaml
```

参数详细解释请参见[参数说明](#3-参数说明)。

#### 2.1.3 带ptq训练结果的量化模型 PPL 测试

量化评估会重建量化 block，并根据 `bit_config` 判断是否启用量化器。

```bash
python -m amct_pytorch.eval \
  --model /path/to/model \
  --model_name qwen3 \
  --device npu:0 \
  --granularity block \
  --eval_mode quant \
  --quant_target mlp \
  --algos lwc lac \
  --quant_dtype int \
  --bit_config amct_pytorch/configs/example_w4a8.yaml \
  --moe_mlp_param_dir ./outputs/qwen3_ptq_params/mlp
```

参数详细解释请参见[参数说明](#3-参数说明)。

`eval`相关启动指令也可参考[eval.sh](../examples/eval.sh)

### 2.2 `extract_ptq_data`

**入口**：`amct_pytorch/cli/llm/extract_ptq_data.py`

**功能特性**：该入口用来提取PTQ校准数据，流程如下：

1. 加载校准数据：加载 Pileval 校准样本。
2. 执行推理：执行 Embedding 编码及逐 Block 的前向传播。
3. 采集中间数据：在指定的 Hook 位置捕获 PTQ 所需的输入（Input）、关键字参数（Kwargs）等中间数据，供后续 PTQ 流程使用。

**输出**：`--data_dir` 下的 block/unit 输入、kwargs 等 PTQ 数据。

**约束**：

- `quant_target` 参数必须且仅能指定一个目标。
- `granularity` 必须和ptq中保持一致。

#### 2.2.1 提取 PTQ 数据

在基于 Block 的 PTQ 数据提取流程中，系统根据 `quant_target` 配置自动定位目标层并注册 Hook，以捕获激活值或权重数据。不同量化目标对应的数据采集位置如下

- Attention 模块（`attn-linear`, `attn-cache`）：在 Attention Norm 之后采集数据。
- FFN/MoE 模块（`mlp`, `moe`）：在 FFN Norm 之后采集数据。

```bash
python -m amct_pytorch.extract_ptq_data \
  --model /path/to/model \
  --model_name qwen3 \
  --device npu:0 \
  --quant_target mlp \
  --seq_len 4096 \
  --data_dir ./outputs/qwen3_ptq_data
```

参数详细解释请参见[参数说明](#3-参数说明)。

`extract_ptq_data`相关启动指令也可参考[extract_ptq_data.sh](../examples/extract_ptq_data.sh)

### 2.3 `ptq`

**入口**：`amct_pytorch/cli/llm/ptq.py`

**功能特性**：该入口用来进行PTQ优化训练，流程如下：

1. 数据加载：加载由 `extract_ptq_data` 生成的校准数据集。
2. 逐层优化：对指定 `quant_target` 单元执行逐层量化优化。
3. 损失最小化：以原始未量化模型的输出作为真值GT（GroundTruth），通过最小化量化模块输出与真值之间的均方误差（MSE）来优化量化参数。
4. 参数导出：优化完成后，自动导出各单元的 PTQ 参数。

**输出**：PTQ 参数目录下的 `layer_*.pt` 文件。

**约束**：

- `quant_target` 参数必须且仅能指定一个量化目标单元。
- 当前仅支持 `granularity=block`， `model` 粒度的 PTQ 逻辑目前为预留接口。
- `granularity` 必须和 extract_ptq_data 中保持一致。

#### 2.3.1 训练 PTQ 参数

`PTQ` 当前要求一次只处理一个 `quant_target`量化目标，并支持 block 粒度的量化训练。训练完成后，PTQ 量化参数会保存到对应的参数目录。

```bash
python -m amct_pytorch.ptq \
  --model /path/to/model \
  --model_name qwen3 \
  --device npu:0 \
  --granularity block \
  --quant_target mlp \
  --quant_dtype int \
  --bit_config amct_pytorch/configs/example_w4a8.yaml \
  --algos lwc lac \
  --data_dir ./outputs/qwen3_ptq_data \
  --start_block_idx 0 \
  --end_block_idx 32 \
  --epochs 15 \
  --base_lr 1e-5 \
  --optimizer adamw \
  --output_dir ./outputs/qwen3_ptq
```

参数详细解释请参见[参数说明](#3-参数说明)。

### 2.4 `deploy`

**入口**：`amct_pytorch/cli/llm/deploy.py`

**功能特性**：该入口用来导出部署的权重，流程如下：

- 加载基础safetensors文件：系统将从源文件中加载原始的 safetensors 权重索引，以作为后续操作的基础。
- 构建量化模块：基于指定的 PTQ参数，系统会构建用于量化处理的 block 模块。
- 导出部署张量：系统将生成并导出量化的权重张量，以及相关的 scale 参数等用于部署的张量数据。
- 管理未替换权重：在量化过程中未被处理的原始权重将被重新分片，并存储在 `rest_*.safetensors` 文件中以供恢复或进一步使用。
- 更新配置文件：完成操作后，系统将更新核心配置文件 `model.safetensors.index.json` 和 `config.json`，以确保文件的完整性以及一致性。

**输出**：部署模型目录，包括 `layer_*.safetensors`、`rest_*.safetensors`、更新后的 `model.safetensors.index.json` 和 `config.json`。

**约束**：

- 模块粒度：当前仅支持以block 为粒度的处理（`granularity=block`） 。
- 自定义 hook 限制：若使用了自定义的 `quantize()` hook（用于权重量化算法），则当前版本不支持通过 `export_deploy()` 功能进行部署。

`ptq`单卡相关启动指令也可参考[ptq_single_npu.sh](../examples/ptq_single_npu.sh)
`ptq`多卡相关启动指令也可参考[ptq_multi_npu.sh](../examples/ptq_multi_npu.sh)

#### 2.4.1 导出部署权重

`deploy` 当前支持 block 粒度导出，该过程会复制模型的非权重辅助文件，逐层导出量化权重，重写 `model.safetensors.index.json`，并在 `config.json` 中写入 `quantization_config`。

```bash
python -m amct_pytorch.deploy \
  --model /path/to/model \
  --model_name qwen3 \
  --granularity block \
  --quant_target mlp \
  --quant_dtype mxfp \
  --bit_config amct_pytorch/configs/example_w4a8.yaml \
  --moe_mlp_param_dir ./outputs/qwen3_ptq_params/mlp \
  --output_dir ./outputs/qwen3_deploy
```

参数详细解释请参见[参数说明](#3-参数说明)。

## 3. 参数说明

### 3.1 通用参数

| 参数 | 默认值 | 含义 |
|------|------|------|
| `--model` | `deepseek-ai/DeepSeek-V4-Pro` | 模型权重路径或模型标识，部署导出时应为本地模型目录。 |
| `--model_name` | `deepseek-ai/DeepSeek-V4-Pro` | AMCT 内部模型适配器名称，需匹配已注册模型，如 `qwen3`、`deepseek_v4`。 |
| `--device` | `npu:0` | 指定运行设备。建议选择NPU/GPU，以加速计算。 |
| `--granularity` | `model` | 工作粒度，即量化或处理的单位，常用选项：<br> `block`：按模块块处理（更细粒度，可能效果更好但耗时）。<br> `model`：整个模型统一处理（粗粒度，速度快），部分流程不支持。 |
| `--seed` | `0` | 随机种子，用于控制实验中随机行为的一致性（如数据采样、初始化等），设置为固定值可保证结果可复现。 |
| `--quant_target` | `[]` | 量化目标，支持以下组合：<br>mlp：Multi-Layer Perceptron，多层感知机。<br>moe：Mixture of Experts，混合专家模型。<br>attn-linear：Attention Linear Layer，注意力机制中的线性层。<br>attn-cache：Attention Cache，注意力缓存。 |
| `--seq_len` | `4096` | 校准和评估时使用的输入序列长度。用于确定模型在处理多长文本时的表现（尤其在量化校准阶段很重要）。 可选值：`1024`、`2048`、`4096`|
| `--data_dir` | 空字符串 | PTQ 中间数据保存/读取目录。 |
| `--output_dir` | `./outputs` | 输出目录，存放日志文件、PTQ 参数配置、最终部署模型等所有生成内容。 |

### 3.2 PPL 评估参数

| 参数 | 默认值 | 含义 |
|------|------|------|
| `--eval_mode` | `bf16` | 评估模式，指定模型在计算困惑度（PPL）时使用的精度格式，支持如下两种格式：<br> `bf16`：半精度浮点数（BF16），常用于加速推理并保持较高精度。<br>`quant`：量化模式，可能指 INT8/INT4 等量化格式，用于降低内存和计算开销，但可能牺牲少量精度。 |
| `--wikitext_final_out` | 空字符串 | WikiText 末端输出路径参数，用于指定WikiText 数据集处理后最终输出的文件路径，目前该参数预留。 |

说明：

- `eval_mode=bf16` 时，`bit_config` 中不应存在低于 16 bit 的 linear/cache 配置。
- `eval_mode=quant` 时，如果 `bit_config` 没有任何低于 16 bit 的配置，会构建量化模块但关闭量化器。

### 3.3 数据提取参数

| 参数 | 默认值 | 含义 |
|------|------|------|
| `--nsamples` | `128` | 校准样本数量，`extract_ptq_data` 从 Pileval 中加载该数量样本。 |

### 3.4 量化配置参数

| 参数 | 默认值 | 含义 |
|------|------|------|
| `--quant_dtype` | 空  |量化后数据的类型，当前适配包括mxfp、int。`eval_mode=quant`时，为必要参数|
| `--bit_config` | `None` | YAML 格式的配置文件路径，该文件定义了具体的位宽策略。如果不提供此参数，则使用默认的 `W16A16` 配置。 |
| `--algos` | `[]` | 启用的量化算法列表，算法会根据其注册的 `target` 自动应用到模型的权重（weight）、激活值（activation）或结构（structure）上。 |
| `--is_per_tensor` | `False` | 对于LAC（Learned Activation Clipping，学习式激活裁剪）算法，是否使用 per-tensor统计信息来确定裁剪范围。 |
| `--k_size` | `128` | 可学习的 Hadamard 类结构变换矩阵的尺寸。 |

常用 `bit_config`：

| 文件 | 含义 |
|------|------|
| [bf16.yaml](../amct_pytorch/configs/bf16.yaml) | 全部保持 16 bit。 |
| [w8a8.yaml](../amct_pytorch/configs/w8a8.yaml) | 全局 W8A8（权重激活都为8bit）。 |
| [w4a8.yaml](../amct_pytorch/configs/w4a8.yaml) | 全局 W4A8（权重8bit，激活4bit）。 |
| [w4a4.yaml](../amct_pytorch/configs/w4a4.yaml) | 全局 W4A4（权重激活都为4bit）。 |
| [example_w4a8.yaml](../amct_pytorch/configs/example_w4a8.yaml) | 自定义量化方案示例。 |

- `bit_config` 支持顶层 `w_bits`、`a_bits`，也支持按 `attn-linear`、`mlp`、`moe`、`attn-cache` 分组覆盖。linear 组内如果写 `w_bits` 或 `a_bits`，必须两者同时出现。
- `quant_dtype`传入`int`时，权重采用`per-channel`量化，激活采用`dynamic-per-token`量化，均采用**对称量化**。
- `bit_config`需要传入yaml，推荐用户在本地自行配置yaml，可参考[configs路径下的yaml样例](../amct_pytorch/configs/)或直接点进上方软连接跳转

### 3.5 PTQ 参数

| 参数 | 默认值 | 含义 |
|------|------|------|
| `--cali_bsz` | `4` | PTQ 校准训练batch size。<br>PTQ过程中，需要使用一小部分数据（称为校准数据集）来调整量化参数（如缩放因子和零点），该参数定义了每次处理多少个样本进行校准。 |
| `--base_lr` | `1e-5` | 基础学习率，该参数决定了模型参数在每次迭代中更新的步长。 |
| `--optimizer` | `adamw` | 优化器，指定用于更新模型参数的算法；支持如下几种优化器。<br> `adamw`：Adam的改进版，通常能更好地处理权重衰减。<br>`adam`：经典的Adam优化器。<br/>`sgd`：随机梯度下降。<br/>`cayley：`一种较少见的优化器。 |
| `--weight_decay` | `0.0` | 权重衰减，适用于部分优化器；一种正则化技术，通过在损失函数中加入一个与权重大小成正比的惩罚项，来防止模型过拟合。值为`0.0`表示在此优化过程中不使用权重衰减。 |
| `--momentum` | `0.9` | SGD 动量。<br>当使用SGD优化器时，动量可以帮助加速收敛并帮助模型跳出局部最优解,它通过累积之前梯度的指数加权移动平均来更新参数。 |
| `--lr_scheduler` | `cosine` | 学习率调度器，定义了在训练过程中如何动态地调整学习率；支持如下几种：<br> `none`：学习率保持不变。<br>`cosine`：余弦退火，学习率会按照余弦函数的曲线平滑地降低。<br/>`step`：阶梯式衰减，每隔固定的步数将学习率乘以一个系数。<br/>默认使用`cosine`，因为它能提供更平滑的学习率下降，有助于模型收敛到更好的状态。 |
| `--min_lr` | `0.0` | 最小学习率参数，当前通用 scheduler 中未直接使用。<br>在使用某些调度器（如`cosine`或`step`）时，可以设置学习率的下限。即使调度器要求继续减小，学习率也不会低于这个值。 |
| `--lr_step_size` | `1` | StepLR 的 step size<br>该参数仅在`--lr_scheduler`设置为`step`时生效，它定义了每隔多少个epoch（轮次）对学习率进行一次衰减。 |
| `--lr_gamma` | `0.1` | StepLR 的衰减系数。<br>该参数仅在`--lr_scheduler`为`step`时有效，它定义了每次衰减时学习率要乘以的系数。 |
| `--epochs` | `15` | PTQ 优化 epoch 数，指定了校准和优化过程的总轮数，一个epoch意味着整个校准数据集被遍历一次。 |
| `--start_block_idx` | `0` | 起始 block 下标，包含该层。<br>该参数用于指定从模型的哪个部分（block）开始进行优化。 |
| `--end_block_idx` | `61` | 结束 block 下标（指定优化的最后一个block），索引不包含该层。 |
| `--attn_linear_param_dir` | 空字符串 | 指定注意力机制（attention）中线性层（linear）的PTQ参数目录。 |
| `--attn_cache_param_dir` | 空字符串 | 指定注意力机制（attention）中缓存（cache）部分的PTQ参数目录。 |
| `--moe_mlp_param_dir` | 空字符串 | 指定混合专家模型（MoE）或多层感知机（MLP）部分的PTQ参数目录。 |

如果 `ptq` 时未显式传入对应参数目录，系统会按以下形式自动创建：

```text
{output_dir}/ptq_params/{model_name}/{quant_target}/
```

每个 PTQ unit 的参数文件命名通常为：

```text
layer_{layer_idx}_{unit_name}.pt
```

## 4. 常见注意事项

- `extract_ptq_data` 和 `ptq` 的 `quant_target` 必须保持一致。
- `ptq`、`extract_ptq_data` 当前一次只处理一个 `quant_target`，需要多目标量化时建议分目标依次执行。
- `deploy` 读取 PTQ 参数目录时，应确保对应 `quant_target` 的参数已训练完成。
- `--granularity block` 是当前 LLM Agent 的主要可用路径；`ptq` 的 model 粒度仍为预留实现。
- `--model_name` 是内部适配器名称，不一定等同于 HuggingFace 模型路径。
- `bit_config`、`model`请配置本地路径，`bit_config`对应的yaml文件可以参考仓上amct_pytorch/configs内的文件

## 5. 一站式样例

本章节提供一站式样例，帮助开发者更好熟悉本特性的流程，以`qwen3.6moe`模型为例，可参考[Qwen-3.6-MoE一站式样例](../examples/models/qwen3.6/Qwen3.6-Moe.md)