# DeepSeek-V4-Flash W4A4 量化样例（lwc + lac）

本样例使用 AMCT 对 DeepSeek-V4-Flash 完成 `w4a4`（权重 4bit + 激活 4bit）的 `lwc` + `lac` 后训练量化，覆盖 Attention 与 MoE 两类量化目标，交付 `int4/int4` 与 `mxfp4/mxfp4` 两种量化格式的可复现流程。

## 运行说明

量化工作流共四步：BF16 精度基线评测 → PTQ 离线数据提取 → PTQ 训练 → 带 PTQ 参数的量化精度评估。所有脚本位于 `scripts/`，位宽配置使用仓库自带的 `amct_pytorch/configs/w4a4.yaml`。

### 前置条件

- 模型权重：从 [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) 下载（FP8+FP4 混合格式 checkpoint，约 149 GB）。`deepseek_v4` 适配器会在加载时自动将 1 字节权重反量化为高精度，无需预先转换 bf16 副本，具体包括：
  - Attention 权重（FP8 e4m3）：按 128x128 块 scale 反量化；
  - MoE expert 权重（打包 int8 的 MXFP4）：按每行 32 元素组的 E8M0 scale 解包反量化；
  - checkpoint 中的 `.scale` 键在加载时消费并丢弃，`load_state_dict(strict=True)` 不会被多余键阻塞。
  本样例同步修复了该适配器的两处加载缺陷：`_block_sharded` 之前对 FP8 权重做裸 dtype cast（数值错误）并以 fp32 默认 dtype 构建 meta 模块导致权重被静默升为 fp32（单层显存 24.1G -> 12.1G）；`_block_staged` 之前全量 `.bfloat16()` 会破坏 fp32 声明的 HC/RMSNorm 参数。
- 校准集：Pileval（`mit-han-lab/pile-val-backup`），评估集：WikiText2（`wikitext-2-raw-v1` test split），`seq_len=4096`。首次运行时自动下载。
- 硬件：单卡即可运行；双卡可通过 `--start_block_idx/--end_block_idx` 按 block 分片并行 PTQ。注意单卡 64GB HBM 下 PTQ 需 `--cali_bsz 1`。

### 第一步：BF16 精度基线

```shell
bash scripts/eval_bf16.sh
```

默认参数：`MODEL_PATH`（模型目录）、`BIT_CONFIG=amct_pytorch/configs/bf16.yaml`、`SEQ_LEN=4096`、`DEVICE=npu:0`。基线 PPL 用于计算量化精度损失。

### 第二步：PTQ 离线数据提取

`extract_ptq_data` 每次只支持一个 `quant_target`，需分别对 Attention 与 MoE 各执行一次（默认 `NSAMPLES=16`，每次提取约 22 GiB 数据，两 target 合计约 43 GiB，请预留磁盘空间）：

```shell
QUANT_TARGET=attn-linear bash scripts/extract_ptq_data.sh
QUANT_TARGET=moe bash scripts/extract_ptq_data.sh
```

`--nsamples` 受磁盘预算约束：本模型 hidden 状态为 4D Hyper-Connection 形状（`hc_mult=4`），每样本每层约 134 MB，`nsamples=128` 时单 target 需约 170 GiB，本样例取 16。

### 第三步：PTQ 训练

Attention 与 MoE 分别训练（DeepSeek-V4 为纯 MoE 结构，无独立 MLP 分支，`quant_target=mlp` 会被适配器拒绝）：

```shell
# int4/int4 格式
QUANT_DTYPE=int bash scripts/ptq_attn.sh
QUANT_DTYPE=int bash scripts/ptq_moe.sh

# mxfp4/mxfp4 格式
QUANT_DTYPE=mxfp bash scripts/ptq_attn.sh
QUANT_DTYPE=mxfp bash scripts/ptq_moe.sh
```

关键参数说明：

- `QUANT_DTYPE`：`int` 或 `mxfp`。位宽配置使用仓库自带的 `amct_pytorch/configs/w4a4.yaml`（`w_bits: 4 / a_bits: 4`），量化格式由 `--quant_dtype` 决定。
- `BASE_LR=1e-3`：学习率。**不要使用框架默认值 1e-5**——实测默认学习率下 `lwc/lac` 的裁剪系数几乎不更新（clip factor 停留在初始值 4.0 附近），量化精度显著劣化。
- `EPOCHS`：Attention 侧 15、MoE 侧 3。MoE 每层有 256 个 routed expert，逐 expert 训练开销大，3 epoch 已可收敛。
- `CALI_BSZ=1`：64GB 单卡 HBM 所需的最小 batch size。

MoE 训练支持按 block 分片多卡并行（例如双卡各跑一半层）：

```shell
ASCEND_RT_VISIBLE_DEVICES=0 START_BLOCK_IDX=0 END_BLOCK_IDX=22 QUANT_DTYPE=int bash scripts/ptq_moe.sh &
ASCEND_RT_VISIBLE_DEVICES=1 START_BLOCK_IDX=22 END_BLOCK_IDX=43 QUANT_DTYPE=int bash scripts/ptq_moe.sh &
wait
```

### 第四步：量化精度评估

同时加载 Attention 与 MoE 两组 PTQ 参数评估量化模型：

```shell
QUANT_DTYPE=int bash scripts/eval_quant.sh
QUANT_DTYPE=mxfp bash scripts/eval_quant.sh
```

## 产物说明

| 产物 | 路径 | 大小 |
|---|---|---|
| PTQ 离线数据（Attention） | `${OUTPUT_DIR}/ptq_data/attn-linear/`（43 个 `block_{i}_attn_in.pkl`） | 21.5 GiB |
| PTQ 离线数据（MoE） | `${OUTPUT_DIR}/ptq_data/moe/`（43 个 `block_{i}_moe_in.pkl`） | 21.5 GiB |
| PTQ 参数（Attention，int/mxfp） | `${OUTPUT_DIR}/ptq_params/{int,mxfp}/attn-linear/`（43 个 `layer_{i}_attn.pt`） | 18 MB / 1.2 GB |
| PTQ 参数（MoE，int/mxfp） | `${OUTPUT_DIR}/ptq_params/{int,mxfp}/moe/`（43×256 个 `layer_{i}_expert_{j}.pt`） | 789 MB / 69 GB |

说明：mxfp 格式的 `lwc` 裁剪系数按 32 元素一组存储（int 为 per-channel），故 mxfp 的 MoE 参数体积远大于 int。两种 target 的离线数据与 PTQ 参数在流程中可先后复用：先提取一种 target 的数据并完成两种格式的 PTQ，再处理另一种 target。

## 结果表

评测环境：2×Atlas A3（64GB HBM）、`seq_len=4096`、WikiText2 test split、Pileval 校准（`nsamples=16`）、`base_lr=1e-3`。

| 模型 | 数据类型 | 算法 | 量化格式 | BF16 PPL | 量化 PPL | PPL 差值 | 量化时长（min） | 离线数据大小（GiB） |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| DeepSeek-V4-Flash | `w4a4` | `lwc` + `lac` | `int4/int4` | 4.146 | 12.700 | 8.554 | 273 | 43.0 |
| DeepSeek-V4-Flash | `w4a4` | `lwc` + `lac` | `mxfp4/mxfp4` | 4.146 | 4.538 | 0.393 | 380 | 43.0 |

指标口径：

- **BF16 PPL**：`eval_mode=bf16` 下的 WikiText2 PPL。
- **量化 PPL**：加载 Attention 与 MoE 的 PTQ 参数后，量化模型的 WikiText2 PPL。
- **PPL 差值**：`量化 PPL - BF16 PPL`。
- **量化时长**：从开始执行 PTQ 到全部量化参数生成完成的时间（Attention 与 MoE 双卡并行执行；不含模型下载和环境构建）。
- **离线数据大小**：`extract_ptq_data` 输出目录的总大小，Attention 与 MoE 数据目录合计。

结果说明：`mxfp4/mxfp4` 在 4bit 激活量化下仍保持极小的精度损失（PPL 差值 0.393）；`int4/int4` 的激活对称量化（16 电平）对该模型 outlier 较重的激活分布较为敏感，PPL 差值 8.554，如对精度有要求建议优先选择 mxfp 格式。
