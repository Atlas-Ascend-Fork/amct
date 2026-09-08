# Qwen3.6-35B-A3B MoE 结构化剪枝

Qwen3.6-35B-A3B 包含 40 个 MoE 层，每层 256 个路由专家，每个 token 激活 8 个专家。原始 BF16 权重约占 64.56 GiB，完整模型可能超过单张 Ascend 910 卡的可用显存，因此剪枝阶段在 CPU 上加载完整模型；剪枝后模型可以继续进行 WikiText2 评测、量化或部署。

本次剪枝固定使用：

```text
method=activation_count
target=moe
mode=size_budget
size_budget=0.9, 0.8, 0.5
calibration=Pileval (mit-han-lab/pile-val-backup, validation)
evaluation=WikiText2 (wikitext-2-raw-v1, test)
seq_len=4096
```

接口说明请参阅：

- [AMCT 结构化剪枝接口](../../../amct_pytorch/pruning/README.md)
- [中文 API 文档](../../../docs/zh/api/README.md)

## 1. 环境要求

### 1.1 硬件和软件

- Ascend 910/A3 NPU ；
- Python 3.11；
- CPU 版 PyTorch；
- 与 PyTorch 匹配的 `torch_npu`；
- CANN Toolkit 和 Ops；
- 至少约 70 GiB 磁盘空间保存模型权重；
- 建议主机内存不低于 150 GiB。

依赖版本见 [requirements.txt](requirements.txt)。

安装 Python 依赖：

```bash
python3 -m pip install -r requirements.txt
```

NPU 环境中的 `torch`、`torch_npu` 和 CANN 需要按照对应硬件环境单独安装。

### 1.2 当前验证环境

本样例已在以下环境完成验证：

```text
Python:       3.11.4
torch:        2.7.1+cpu
torch_npu:    2.7.1.post4
transformers: 5.12.1
datasets:     4.8.4
```

### 1.3 torchaudio 兼容说明

如果系统预装的 `torchaudio` 与 `torch` 版本不匹配，Transformers 导入模型时可能出现动态库错误。正式 Qwen 脚本会在进程内屏蔽该无关依赖，不需要安装或使用 `torchaudio`。

## 2. 目录结构

```text
examples/algorithms/pruning/
├── README.md
├── README_en.md
├── requirements.txt
└── src/
    ├── run_qwen3_6_35b_a3b_pruning_activation_count_budget.py
    └── utils.py
```

## 3. Qwen3.6-35B-A3B 任务样例

### 3.1 任务目标

本样例对应 Qwen3.6-35B-A3B 的 MoE 专家结构化剪枝：

```text
模型：Qwen3.6-35B-A3B
方法：activation_count
目标：MoE experts
模式：size_budget
校准集：Pileval（官方数据集为 `mit-han-lab/pile-val-backup`）
评估集：WikiText2（`wikitext-2-raw-v1` 的 test split）
序列长度：4096
校准样本数：4
后处理：`size_budget=0.5` 档执行 300 步恢复训练
```

`activation_count` 根据校准数据统计专家激活情况，优先删除激活次数较少的专家，并同步收缩路由器和专家权重。

### 3.2 模型准备

下载 [Qwen/Qwen3.6-35B-A3B 原始权重](https://link.gitcode.com/?target=https%3A%2F%2Fhuggingface.co%2FQwen%2FQwen3.6-35B-A3B&from=https%3A%2F%2Fgitcode.com%2Fcann%2Famct%2Fblob%2Fmaster%2Fexamples%2Fmodels%2Fqwen3.6%2FQwen3.6-Moe-Pruning.md&lang=zh&theme=white) 到固定路径， 例如 `/path/to/Qwen3.6-35B-A3B`。该权重为 `bfloat16`，无需格式转换。

目录中应包含：

```text
config.json
model.safetensors.index.json
model-00001-of-00026.safetensors
...
model-00026-of-00026.safetensors
tokenizer.json
tokenizer_config.json
```

首次运行脚本时会自动下载并缓存校准与评估数据集。

## 运行说明

正式脚本路径：

```text
src/run_qwen3_6_35b_a3b_pruning_activation_count_budget.py
```

运行实际命令前，请先根据实际环境设置路径变量：

```bash
export AMCT_ROOT=/path/to/amct
export MODEL_DIR=/path/to/Qwen3.6-35B-A3B
export RESULT_DIR=/path/to/qwen3_6_pruning_results
export BUDGET09_DIR=/path/to/Qwen3.6-35B-A3B-pruned-budget09
export BUDGET08_DIR=/path/to/Qwen3.6-35B-A3B-pruned-budget08
export BUDGET05_DIR=/path/to/Qwen3.6-35B-A3B-pruned-budget05
export RECOVERY_OUTPUT_DIR=/path/to/Qwen3.6-35B-A3B-pruned-budget05-finetuned
export RECOVERY_REPORT=/path/to/recovery_train_budget05_lr1e-2.json
```

运行 `size_budget=0.9`：

```bash
cd $AMCT_ROOT

PYTHONPATH=$AMCT_ROOT \
python3 examples/algorithms/pruning/src/run_qwen3_6_35b_a3b_pruning_activation_count_budget.py \
  --model_path "$MODEL_DIR" \
  --size_budget 0.9 \
  --seq_len 4096 \
  --calibration_samples 4 \
  --pileval_source modelscope \
  --ratio_grid 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 \
  --report_dir "$RESULT_DIR" \
  --save_model_dir $BUDGET09_DIR
```

运行 `size_budget=0.8`：

```bash
PYTHONPATH=$AMCT_ROOT \
python3 examples/algorithms/pruning/src/run_qwen3_6_35b_a3b_pruning_activation_count_budget.py \
  --model_path "$MODEL_DIR" \
  --size_budget 0.8 \
  --seq_len 4096 \
  --calibration_samples 4 \
  --pileval_source modelscope \
  --ratio_grid 0.2,0.3,0.4 \
  --report_dir "$RESULT_DIR" \
  --save_model_dir $BUDGET08_DIR
```

运行 `size_budget=0.5`：

```bash
PYTHONPATH=$AMCT_ROOT \
python3 examples/algorithms/pruning/src/run_qwen3_6_35b_a3b_pruning_activation_count_budget.py \
  --model_path "$MODEL_DIR" \
  --size_budget 0.5 \
  --seq_len 4096 \
  --calibration_samples 4 \
  --pileval_source modelscope \
  --ratio_grid 0.5,0.6,0.7 \
  --report_dir "$RESULT_DIR" \
  --save_model_dir $BUDGET05_DIR
```

参数说明：

| 参数                    | 含义                                       |
| ----------------------- | ------------------------------------------ |
| `--model_path`          | 本地模型目录                               |
| `--size_budget`         | 目标参数预算                               |
| `--seq_len`             | 校准序列长度                               |
| `--calibration_samples` | 校准样本数量                               |
| `--pileval_source`      | 官方 Hugging Face 数据集或 ModelScope 镜像 |
| `--ratio_grid`          | AMCT 尝试的候选专家剪枝比例                |
| `--report_dir`          | JSON 报告输出目录                          |
| `--save_model_dir`      | 可选的剪枝模型保存目录；不传则不保存权重   |

脚本在 CPU 上执行模型加载和结构化剪枝，完成后对剪枝模型执行一次前向检查。剪枝模型先存在于当前 Python 进程的内存中；只有传入 `--save_model_dir` 时，脚本才会将剪枝后的 config、tokenizer 和 safetensors 权重写入该目录。未传该参数时，进程退出后只保留 JSON 报告，剪枝模型不会持久化。

### 3.3 输出报告

默认报告目录：

```text
$RESULT_DIR
```

输出文件：

```text
qwen3_6_35b_a3b_activation_count_budget_0.9.json
qwen3_6_35b_a3b_activation_count_budget_0.8.json
qwen3_6_35b_a3b_activation_count_budget_0.5.json
```

报告包含：

- 剪枝前后参数量；
- 实际参数削减比例；
- 每个 MoE 层的专家稀疏率；
- 预算是否不可达；
- 可剪枝参数占比；
- 剪枝过程事件和警告。

## 诊断与报告

正式脚本会调用：

```python
from amct_pytorch.pruning import prune_diagnose
```

诊断结果确认：

```text
cnn targets:   0
dense targets: 40
moe targets:   40
fixed-ratio pruning: available
accuracy search:     available
```

本样例的配置只启用了 `moe`，因此实际只修改 MoE 专家结构。

剪枝完成后，脚本检查：

```text
logits shape: (1, 4096, 248320)
logits finite: True
```

这证明剪枝后的模型可以完成一次正常前向计算。

正式剪枝后的 `PruneReport` 核心内容：

```text
size_budget=0.9
params_before=34660610688
params_after=28239147648
experts=256 -> 205
budget_unreachable=False

size_budget=0.8
params_before=34660610688
params_after=24965460608
experts=256 -> 179
budget_unreachable=False

size_budget=0.5
params_before=34660610688
params_after=15270310528
experts=256 -> 102
budget_unreachable=False
```

完整 `events`、`per_layer_sparsity` 和 `warnings` 保存在脚本输出的 JSON 报告中。上述诊断使用 `prune_ratio=0.1` 的诊断试剪，正式剪枝使用各档 `size_budget` 和 `ratio_grid`。

## 结果表

| 模型            | 方法               | 实验设置          |          基线 PPL |         剪枝后 PPL |        后处理 PPL |               参数量（前 → 后） | 参数削减率 | 剪枝时长（min） |
| --------------- | ------------------ | ----------------- | ----------------: | -----------------: | ----------------: | ------------------------------: | ---------: | --------------: |
| Qwen3.6-35B-A3B | `activation_count` | `size_budget=0.9` | 6.308547019958496 |  6.664599895477295 |               N/A | 34,660,610,688 → 28,239,147,648 |   18.5267% |           28.89 |
| Qwen3.6-35B-A3B | `activation_count` | `size_budget=0.8` | 6.308547019958496 | 7.1991167068481445 |               N/A | 34,660,610,688 → 24,965,460,608 |   27.9717% |           16.84 |
| Qwen3.6-35B-A3B | `activation_count` | `size_budget=0.5` | 6.308547019958496 |  9.780414581298828 | 8.294227600097656 | 34,660,610,688 → 15,270,310,528 |   55.9433% |           20.46 |

所有正式实验均使用：

```text
seq_len=4096
calibration_samples=4
budget_unreachable=False
logits finite=True
```

## WikiText2 精度评测

样例使用 `wikitext-2-raw-v1` 的 test split，并固定 `seq_len=4096`。Pileval 只用于校准，不能用作 PPL 评估。

原始模型无法完整放入单卡时，使用 AMCT 的 blockwise 评测入口。这里解决的是 NPU 显存问题：评测程序按 block 读取模型并逐块放到 NPU 上计算；模型目录仍然位于本地磁盘。这里的 `--model` 不是要求把完整原始模型再次保存到显存，而是指定权重文件所在的磁盘目录。

基线模型已经存在于 `$MODEL_DIR`，不需要通过剪枝脚本另存一份。基线模型和每个剪枝后模型必须使用同一条命令：

```bash
python3 -m amct_pytorch.eval \
  --trust_remote_code \
  --model $MODEL_DIR \
  --model_name qwen3_6_moe \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --eval_mode bf16 \
  --bit_config amct_pytorch/configs/bf16.yaml
```

要评测剪枝模型，必须在剪枝时传入 `--save_model_dir`，因为剪枝是原地修改当前进程中的模型；进程结束后，如果没有保存目录，剪枝结果会丢失。评测时只替换 `--model` 路径，例如：

```bash
python3 -m amct_pytorch.eval \
  --trust_remote_code \
  --model $BUDGET09_DIR \
  --model_name qwen3_6_moe \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --eval_mode bf16 \
  --bit_config amct_pytorch/configs/bf16.yaml
```

`budget08` 和 `budget05` 只需替换模型目录。这里同样需要注意torchaudio 兼容问题。

## 恢复训练

本次仅对参数量最小、可在单张 64 GiB Ascend 910 上完成全参数训练的 `size_budget=0.5` 模型执行恢复训练。恢复训练从剪枝模型开始，未重新执行剪枝，也未覆盖剪枝模型：

```text
输入模型：$BUDGET05_DIR
输出模型：$RECOVERY_OUTPUT_DIR
训练集：WikiText2 wikitext-2-raw-v1 train split
训练批次：300 个互不重复的连续 token 块
batch size：1
seq_len：512
训练 token 数：153,600
训练步数：300
优化器：SGD
学习率：1e-2
momentum：0.0
warmup：20 步
梯度裁剪：1.0
gradient checkpointing：启用，use_reentrant=False
训练设备：npu:0
训练耗时：27.371 min
峰值已分配显存：58.219 GiB
峰值保留显存：59.793 GiB
```

选择无动量 SGD 是为了避免 AdamW 为参数额外维护一阶、二阶动量状态。SGD 不会像 AdamW 一样按梯度统计量自适应缩放，因此本次恢复训练使用 `lr=1e-2`，而不是直接沿用 AdamW 常见的 `1e-5` 或 `2e-5` 量级。

### 运行恢复训练

`prune_finetune()` 的默认因果语言模型损失接受包含 `input_ids` 的字典，并将 `input_ids` 同时作为 labels。本次显式传入无动量 SGD，避免该接口默认创建 AdamW。

```python
import os
import torch
from amct_pytorch.pruning import prune_finetune

model.config.use_cache = False
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
model.to("npu:0")

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=1e-2, momentum=0.0)

result = prune_finetune(
    model,
    batches,
    steps=300,
    lr=1e-2,
    warmup=20,
    optimizer=optimizer,
    device="npu:0",
    grad_clip=1.0,
    log_every=25,
)

model.to("cpu")
model.save_pretrained(
    os.environ["RECOVERY_OUTPUT_DIR"],
    safe_serialization=True,
)
```

实际运行时还需像剪枝脚本一样在导入 Transformers 前屏蔽不兼容的可选 `torchaudio`，保存 tokenizer，并保留剪枝模型外层 `config.json` 中的 Qwen3.6 wrapper 配置。训练参数和结果另存于 `$RECOVERY_REPORT`。恢复训练完成后，使用上一节相同的 WikiText2 test、`seq_len=4096`、blockwise BF16 命令评测输出目录。

### 当前恢复程度

本次恢复训练的 WikiText2 test 结果为：

```text
原始模型 PPL：      6.308547019958496
剪枝后 PPL：        9.780414581298828
恢复训练后 PPL：    8.294227600097656
```

PPL 越低越好。相对未经恢复的剪枝模型，PPL 绝对下降 `1.486186981201172`，相对下降约 `15.20%`。若以“剪枝造成的 PPL 增量”为损失，则本次恢复了约 `42.81%`：

```text
(9.780414581298828 - 8.294227600097656)
------------------------------------------------ = 42.81%
(9.780414581298828 - 6.308547019958496)
```

恢复后 PPL 仍比原始模型高 `1.985680580139160`，因此当前结果是明显恢复但尚未恢复到基线。训练日志中的首步 loss 和末步 loss 来自不同 batch，不能据此单独判断泛化效果；最终结论以 WikiText2 test PPL 为准。

### 继续恢复时可调整的参数

若需要继续提高恢复程度，可每次只改变少量参数，并用相同的 WikiText2 test 命令比较 PPL：

1. `steps` 和训练数据量：可增加独立训练 batch 和训练步数。若 `steps` 超过 batch 数，`prune_finetune()` 会循环复用数据，重复过多可能过拟合，因此应优先增加不同的训练 token。
2. `lr`：本次 `1e-2` 是无动量 SGD 的起点。可围绕该值进行小范围对照，例如 `3e-3`、`1e-2`；学习率过高可能使 loss 发散，过低则几乎不能恢复权重。
3. `warmup`：训练步数增加时可同步增加 warmup，并保持其占总步数的比例合理。本次为 `20/300`。
4. `seq_len`：更长上下文可能更贴近 `seq_len=4096` 的评测，但会显著增加激活显存。本次峰值保留显存已达到 `59.793 GiB`，在单张 64 GiB 卡上提高前必须重新做 1 步和 10 步显存测试。
5. 训练数据：可增加数量和文本多样性，但不得使用 WikiText2 test split 训练。测试集只用于最终评估。
6. `grad_clip`：本次为 `1.0`。出现梯度或 loss 不稳定时可以调小；不应仅凭训练 loss 选择配置，应同时观察固定验证集或最终 test PPL。

若要比较多组恢复参数，应为每组保留独立的训练报告，并避免覆盖剪枝模型。BF16 的 `budget05` 权重约占 28.4 GiB；磁盘不足时可在记录 PPL 和报告后清理不再需要的恢复模型，再运行下一组实验。