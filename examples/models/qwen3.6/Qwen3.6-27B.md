# Qwen3.6-27B Dense A4W4 MXFP4 Attn-Linear 量化实践 CaseBook

## 1. 实践概述

本实践基于 AMCT PyTorch，对 Qwen3.6-27B Dense 模型执行
A4W4、MXFP4、Attn-Linear 训练后量化，并比较不同 PTQ 算法的
量化精度、运行耗时、Deploy 结果和 clean reload 完整性。

| 配置项 | 本次配置 |
|---|---|
| 模型 | Qwen3.6-27B Dense |
| 模型层数 | 64 |
| 位宽 | A4W4 |
| 数据格式 | MXFP4 |
| 量化目标 | Attn-Linear |
| 量化粒度 | Block |
| Block 范围 | `[0, 64)` |
| 校准数据 | PileVal |
| 精度评测数据 | WikiText2 test |
| 核心精度指标 | Perplexity（PPL） |
| AMCT 内部适配名称 | `qwen3_5` |

本次实际加载的模型、Tokenizer 和权重均为 Qwen3.6-27B。
模型配置中的关键字段为：

```text
model_type = qwen3_5
architectures = Qwen3_5ForConditionalGeneration
```

本次正式执行的量化方案包括：

- Direct
- LWC
- LAC
- LWC+LAC
- OmniQuant
- AutoRound
- FlatQuant

## 2. 校准与评测配置

所有正式 PTQ 算法共用同一份 PileVal 校准数据，避免因校准样本变化
影响不同算法之间的可比性。

| 配置项 | 取值 |
|---|---:|
| 校准样本数 | 16 |
| 校准序列长度 | 1024 |
| 粒度 | block |
| `start_block_idx` | 0 |
| `end_block_idx` | 64 |
| Block 范围 | `[0, 64)` |
| 校准数据提取 Block 数 | 64 |
| 校准数据提取耗时 | 4003 秒 |
| 校准数据目录大小 | 约 11G |
| 校准数据提取退出码 | 0 |

正式校准数据相对目录为：

```text
outputs/formal/extract_attn_linear_n16_s1024_v2
```
七种正式量化方案均生成了完整的 64 个 Block 参数：

```text
PARAM_COUNT=64
```

## 3. 流程与指标说明

本次实践的主要流程为：

```text
原始 BF16 模型
    ↓
提取 PileVal 校准数据
    ↓
PTQ 生成量化参数
    ↓
WikiText2 Eval
    ↓
计算 PPL
    ↓
Deploy 导出部署模型
    ↓
Clean reload 独立重载验证
```

各阶段含义如下：

- **PTQ**：Post-Training Quantization，使用校准数据计算量化参数，
  不重新进行完整模型训练。
- **Eval**：加载模型及量化参数，在 WikiText2 上执行前向评测。
- **PPL**：Perplexity，语言模型预测下一个 token 的困惑度，
  在相同评测配置下通常越低越好。
- **Deploy**：将模型结构、权重、量化参数和配置导出为部署目录。
- **Clean reload**：在新的干净进程中，仅依靠 Deploy 目录重新加载，
  验证导出模型是否完整。

一次 blockwise PPL Eval 主要包括：

```text
Chunking
→ Embedding Processing
→ Block Processing
→ Head Processing
→ Evaluating
→ 输出 PPL
```

其中：

- Chunking：将 WikiText2 token 序列切分为评测样本；
- Embedding Processing：将 token ID 转换为初始隐藏向量；
- Block Processing：依次执行 64 个 Transformer Block；
- Head Processing：执行末端归一化和语言模型输出头；
- Evaluating：计算交叉熵损失并换算为 PPL。
## 4. 正式实验结果

### 4.1 WikiText2 PPL

BF16 基线 WikiText2 PPL 为 `7.782411575317383`。

“相对 BF16”定义为当前方法 PPL 减去 BF16 PPL，数值越小越好。

“相对 Direct 恢复量”定义为 Direct PPL 减去当前方法 PPL，
正数表示相对于直接量化恢复了精度。

| 方法 | WikiText2 PPL | 相对 BF16 | 相对 Direct 恢复量 |
|---|---:|---:|---:|
| BF16 | 7.782411575317383 | — | — |
| Direct | 8.050390243530273 | +0.267979 | 0.000000 |
| LWC | 7.966102123260498 | +0.183691 | +0.084288 |
| LAC | 7.583184719085693 | -0.199227 | +0.467206 |
| LWC+LAC | 7.579893112182617 | -0.202518 | +0.470497 |
| OmniQuant | 7.799318790435791 | +0.016907 | +0.251071 |
| AutoRound | 8.158552169799805 | +0.376141 | -0.108162 |
| FlatQuant | 7.207714557647705 | -0.574697 | +0.842676 |


说明：
- LAC、LWC+LAC 和 FlatQuant 的 PPL 低于 BF16，仅代表在当前
  WikiText2 数据集和当前评测配置下的实测结果。 不能据此推导量化方法在一般情况下必然提高模型精度。
- LAC 和 FlatQuant 均进行过独立复测，PPL 与正式结果逐位一致。

### 4.2 标准化单卡串行 Eval


| 方法 | 单卡串行 Eval/s | 历史重叠或并发 Eval/s | 标准化退出码 |
|---|---:|---:|---:|
| BF16 | 1807 | — | 0 |
| Direct | 2164 | 2129 | 0 |
| LWC | 2021 | 1990 | 0 |
| LAC | 1975 | 2071 | 0 |
| LWC+LAC | 1948 | 68008 | 0 |
| OmniQuant | 2008 | 68346 | 0 |
| AutoRound | 1961 | 66749 | 0 |
| FlatQuant | 2013 | 68144 | 0 |

标准化复测采用以下统一条件：

- 同一 Qwen3.6-27B 模型；
- 同一 WikiText2 数据；
- 同一正式 PTQ 参数；
- 同一物理 NPU 4；
- 进程内统一使用逻辑 `npu:0`；
- 同一 Block 范围 `[0, 64)`；
- 同一 Eval CLI；
- 每个方法严格串行运行；
- 测试期间不并发执行其他 PTQ、PPL、Deploy 或大模型任务。

单卡串行 Eval 耗时集中在 1948～2164 秒。

本表中的 Eval 时间是完整 WikiText2 PPL 评测的端到端耗时，
包含数据切分、Embedding Processing、Block Processing、
Head Processing、Evaluating 以及相关框架开销。

该指标不等同于在线推理 tokens/s、首 token 延迟或单 token 延迟。
### 4.3 PTQ、Deploy 与 clean reload

PTQ 耗时来自历史正式运行，尚未复测，只作为工程记录。

| 方法 | 历史 PTQ/s | Deploy/s | 文件数 | 导出大小 | Clean reload |
|---|---:|---:|---:|---:|---|
| Direct | 786 | 1418 | 83 | 约 42G | PASS |
| LWC | 969 | 1399 | 83 | 约 42G | PASS |
| LAC | 918 | 1379 | 83 | 约 42G | PASS |
| LWC+LAC | 1196 | 1383 | 83 | 约 42G | PASS |
| OmniQuant | 991 | 1474 | 83 | 约 42G | PASS |
| AutoRound | 2583 | 1308 | 83 | 约 42G | PASS |
| FlatQuant | 1158 | 1372 | 83 | 约 42G | PASS |
| Learnable Hadamard | SKIPPED | SKIPPED | SKIPPED | SKIPPED | SKIPPED |

约 42G 的 Deploy 目录不仅包含打包后的量化权重，还包含配置文件、
未量化组件以及保持浮点格式的模型部分。

## 5. 多任务并发的工程教训

历史实验中存在两类任务重叠：

1. BF16、Direct、LWC 和 LAC 的 recheck 任务分别绑定不同物理
   NPU，但仅间隔约 60 秒启动，执行时间存在重叠。
2. LWC+LAC、OmniQuant、AutoRound 和 FlatQuant 使用四个独立
   PTQ+PPL pipeline，在物理 NPU 4～7 上并发执行。

虽然不同任务使用不同 NPU，但仍会竞争以下共享资源：

- CPU 计算资源；
- 系统内存及内存带宽；
- 磁盘读取带宽；
- Python 和 PyTorch 调度资源；
- CPU fallback 算子；
- 模型加载及中间数据处理资源。

历史结果中，后四种算法的 PPL Eval 耗时约为 6.7～6.8 万秒；
严格单卡串行复测后，耗时下降至 1948～2013 秒。

两轮测试的 PPL 逐位一致，说明正式 PTQ 参数和精度结果可复现，
但历史耗时受到并发资源竞争严重污染。

因此，后续实验需要注意：

1. 精度批量验证尽量避免并发，否则可能耗时很长长长长。
2. 性能基准必须统一模型、数据、设备、命令和并发条件。
3. PPL、PTQ、Deploy 和在线推理性能应分别计时。

## 6. 运行环境

| 项目 | 配置 |
|---|---|
| 操作系统 | EulerOS 2.0 SP10 |
| NPU | Ascend 910B3 |
| Python | 3.12.13 |
| PyTorch | 2.10.0+cpu |
| NPU 接口 | torch_npu |
| Transformers | 5.5.4 |
| 模型权重 | 15 个 safetensors，约 51.75 GiB |


本实践通过 `torch_npu` 接入 NPU。

为避免在公开文档中写入个人目录，复现时建议使用以下通用变量：

```bash
ROOT="<PROJECT_ROOT>"
AMCT_DIR="$ROOT/code/amct"
MODEL_DIR="$ROOT/models/Qwen3.6-27B"
DATA_DIR="$ROOT/data/formal"
FORMAL_DIR="$ROOT/outputs/formal"
DEPLOY_DIR="$ROOT/outputs/deploy"
LOG_DIR="$ROOT/logs"
```

## 7. 算法与输出目录

七种正式方案的 PTQ 参数相对目录如下：

| 方法 | PTQ 参数目录 |
|---|---|
| Direct | `outputs/formal/direct_a4w4_mxfp4_full64/ptq_params/qwen3_5/attn-linear` |
| LWC | `outputs/formal/lwc_a4w4_mxfp4_full64/ptq_params/qwen3_5/attn-linear` |
| LAC | `outputs/formal/lac_a4w4_mxfp4_full64/ptq_params/qwen3_5/attn-linear` |
| LWC+LAC | `outputs/formal/lwc_lac_a4w4_mxfp4_full64/ptq_params/qwen3_5/attn-linear` |
| OmniQuant | `outputs/formal/omniquant_a4w4_mxfp4_full64/ptq_params/qwen3_5/attn-linear` |
| AutoRound | `outputs/formal/autoround_a4w4_mxfp4_full64/ptq_params/qwen3_5/attn-linear` |
| FlatQuant | `outputs/formal/flatquant_a4w4_mxfp4_full64/ptq_params/qwen3_5/attn-linear` |

正式 Deploy 输出相对目录如下：

| 方法 | Deploy 输出目录 |
|---|---|
| Direct | `outputs/deploy/direct_a4w4_mxfp4_full64` |
| LWC | `outputs/deploy/lwc_a4w4_mxfp4_full64` |
| LAC | `outputs/deploy/lac_a4w4_mxfp4_full64` |
| LWC+LAC | `outputs/deploy/lwc_lac_a4w4_mxfp4_full64` |
| OmniQuant | `outputs/deploy/omniquant_a4w4_mxfp4_full64` |
| AutoRound | `outputs/deploy/autoround_a4w4_mxfp4_full64` |
| FlatQuant | `outputs/deploy/flatquant_a4w4_mxfp4_full64` |
## 8. 标准化 Eval 复现命令

### 8.1 BF16 Eval

```bash
env ASCEND_RT_VISIBLE_DEVICES=4 \
python -m amct_pytorch.eval \
  --model "$MODEL_DIR" \
  --model_name qwen3_5 \
  --device npu:0 \
  --granularity block \
  --eval_mode bf16 \
  --quant_target attn-linear \
  --bit_config amct_pytorch/configs/bf16.yaml \
  --data_dir "$DATA_DIR" \
  --start_block_idx 0 \
  --end_block_idx 64 \
  --output_dir "$OUTPUT_DIR"
```

### 8.2 量化 Eval

```bash
env ASCEND_RT_VISIBLE_DEVICES=4 \
python -m amct_pytorch.eval \
  --model "$MODEL_DIR" \
  --model_name qwen3_5 \
  --device npu:0 \
  --granularity block \
  --eval_mode quant \
  --quant_target attn-linear \
  --quant_dtype mxfp \
  --algos "$ALGO" \
  --bit_config amct_pytorch/configs/w4a4.yaml \
  --attn_linear_param_dir "$PARAM_DIR" \
  --data_dir "$DATA_DIR" \
  --start_block_idx 0 \
  --end_block_idx 64 \
  --output_dir "$OUTPUT_DIR"
```

算法参数如下：

| 方法 | `--algos` 参数 |
|---|---|
| Direct | 不传 `--algos` |
| LWC | `lwc` |
| LAC | `lac` |
| LWC+LAC | `lwc lac` |
| OmniQuant | `omniquant` |
| AutoRound | `autoround` |
| FlatQuant | `flatquant` |## 10. Deploy 与 clean reload

七种量化方案均完成 Deploy，并通过 clean reload。

统一检查结果为：

```text
MISSING_KEYS=[]
UNEXPECTED_KEYS_COUNT=0
MISMATCHED_KEYS_COUNT=0
ERROR_MSGS_COUNT=0
TOTAL_QUANTIZED_COUNT=304
LAYER0_QUANTIZED_COUNT=5
VISION_QUANTIZED_COUNT=0
LM_HEAD_HAS_WEIGHT_SCALE=False
CLEAN_RELOAD=PASS
EXIT_CODE=0
```

Clean reload 中采用了以下兼容处理：

1. 仅在内存配置中移除两个 `activation_use_clip` 字段。
2. 使用 `torch.bfloat16` 初始化模型骨架和未量化浮点组件。
3. 忽略 Vision、Visual 和 Image 相关分支。
4. 忽略 `lm_head`。
5. 忽略输入维度不能被 MXFP block size 32 整除的 Linear。
6. 不修改正式 Deploy 目录中的原始 `config.json`。

## 9. 已知兼容性问题

### 9.1 `activation_use_clip`

Deploy 生成的量化配置组中包含 `activation_use_clip` 字段。

当前环境中的 compressed-tensors 配置模型不接受该字段，因此
clean reload 只在内存配置副本中移除该字段：

```text
REMOVED_ACTIVATION_USE_CLIP=2
```

正式 Deploy 目录中的 `config.json` 不作修改。

### 9.2 uint8 模型骨架初始化

Deploy 模型中的量化权重使用 uint8 打包存储。

如果使用 `dtype=auto`，Transformers 可能将模型骨架初始化类型错误
推断为 uint8，并出现：

```text
dtype=torch.uint8 is not a floating-point dtype
```

因此，模型骨架和未量化浮点组件在 clean reload 时显式使用
`torch.bfloat16`。

部分视觉 Linear 的输入维度不能被 MXFP block size 32 整除。
Clean reload 时需要精确排除这些模块以及 `lm_head`。

### 9.3 Eval 警告

正式评测过程中观察到：

- FLA fast path 相关库未安装，部分计算回退至 Torch 实现；
- Tokenizer 先编码完整 WikiText2，再切分为 72 个评测样本，
  因而出现序列长度警告；
- FlatQuant 的部分矩阵求逆算子不受当前 NPU 后端原生支持，
  会发生 CPU fallback；
- 部分张量因 NPU internal format 条件不满足而使用 base format。

这些警告未导致正式评测失败，但可能影响端到端 Eval 性能。

## 10. Deploy 与 clean reload

七种量化方案均完成 Deploy，并通过 clean reload。

统一检查结果为：

```text
MISSING_KEYS=[]
UNEXPECTED_KEYS_COUNT=0
MISMATCHED_KEYS_COUNT=0
ERROR_MSGS_COUNT=0
TOTAL_QUANTIZED_COUNT=304
LAYER0_QUANTIZED_COUNT=5
VISION_QUANTIZED_COUNT=0
LM_HEAD_HAS_WEIGHT_SCALE=False
CLEAN_RELOAD=PASS
EXIT_CODE=0
```

Clean reload 中采用了以下兼容处理：

1. 仅在内存配置中移除两个 `activation_use_clip` 字段。
2. 使用 `torch.bfloat16` 初始化模型骨架和未量化浮点组件。
3. 忽略 Vision、Visual 和 Image 相关分支。
4. 忽略 `lm_head`。
5. 忽略输入维度不能被 MXFP block size 32 整除的 Linear。
6. 不修改正式 Deploy 目录中的原始 `config.json`。

这些处理不会将已经打包的 MXFP4 权重恢复为 BF16。


## 11. 阶段结论

在当前 Qwen3.6-27B Dense、A4W4、MXFP4、Attn-Linear 配置下：

1. Direct、LWC、LAC、LWC+LAC、OmniQuant、AutoRound 和
   FlatQuant 均完成正式 64 层 PTQ 参数生成。
2. 七种量化方案的正式参数目录均包含 64 个 Block 参数。
3. BF16 和七种量化方案均完成 WikiText2 PPL 评测。
4. 七种量化方案均完成 Deploy，退出码均为 0。
5. 七种量化方案均通过 clean reload 完整性检查。
6. FlatQuant 在当前 WikiText2 数据集和当前评测配置下获得最低
   实测 PPL。
7. 严格单卡串行 Eval 耗时集中在 1948～2164 秒。
8. 历史多任务并发耗时不能用于算法速度排名。
9. AutoRound 的历史 PTQ 耗时最高，但 PTQ 耗时尚未完成统一条件下
   的严格串行复测。
10. Learnable Hadamard 本阶段暂不执行。

七种方案最终使用相同的 A4W4、MXFP4、Attn-Linear 部署配置。

其在线推理吞吐、延迟和峰值显存仍需通过独立部署推理基准验证。
## 12. 维护建议

| 字段 | 建议或结果 |
|---|---|
| 最低 NPU 数量 | 1 |
| 推荐 NPU 数量 | 1 张卡用于标准化性能基准 |
| 是否支持单卡 | 是 |
| 单卡启动方式 | 只暴露一张物理卡，AMCT 内使用逻辑 `npu:0` |
| 多任务多卡方式 | 每个进程只暴露一张物理卡，并在进程内使用 `npu:0` |
| 单任务多卡并行 | 本 CaseBook 未验证 |
| 已验证芯片 | Ascend 910B3 |
| 峰值 NPU HBM | 本轮未独立采集 |
| tokens/s | 本轮未独立采集 |


## 13. 参考链接

- Qwen3.6-MoE AMCT 示例：
  <https://gitcode.com/cann/amct/blob/master/examples/models/qwen3.6/Qwen3.6-Moe.md>
- AMCT 已支持量化算法介绍：
  <https://gitcode.com/cann/amct/blob/master/docs/zh/algorithm_brief.md>
- CaseBook 参考模板：
  <https://gitcode.com/cann/amct/blob/master/.agents/docs/casebook/qwen/qwen3.5-3.6.md>
- Qwen3.6-27B 模型：
  <https://modelscope.cn/models/Qwen/Qwen3.6-27B>