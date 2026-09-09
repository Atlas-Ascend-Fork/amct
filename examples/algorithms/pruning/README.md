# AMCT 结构化剪枝样例

演示 `amct_pytorch.pruning` 接口在三种域上的调用：dense FFN（剪中间维）/ CNN（剪通道）/
MoE（剪专家）。目录里既有随机初始化的微型模型演示，也有面向社区任务的 Qwen3.6-35B-A3B
MoE 剪枝样例。

> 接口详见 [`amct_pytorch/pruning/README.md`](../../../amct_pytorch/pruning/README.md)。

## 1 剪枝前提

### 1.1 安装依赖

依赖见 [requirements.txt](requirements.txt)：`torch` 与 `transformers`（随 amct_pytorch 导入链需要）。
在 NPU 上运行时另需与 Python/torch 版本匹配的 `torch_npu` 及已安装的 CANN 包。

### 1.2 模型与数据准备

样例模型与数据由 [src/utils.py](src/utils.py) 用固定随机种子构造（`MiniMLP`/`MiniCNN`/`MiniMoE`），
无需下载、无需联网。替换成真实模型与校准数据即可用于实际场景。

### 1.3 剪枝配置

以 dict 配置直接传入 `prune()`（与 `amct.quantize` 同风格），按域选择方法：

| 域 | 方法 | 说明 |
|:--|:--|:--|
| dense | `low_variance` | 按激活方差剪 FFN 中间维（自动避开注意力投影） |
| dense | `reconstruct` | 剪后用最小二乘重构补偿，recovery ∈ {none, bias, ls} |
| cnn | `variance_channel` | 按激活方差朴素切片通道 |
| cnn | `reconstruct` | 输出重构补偿的通道剪枝 |
| moe | `activation_count` | 按专家激活频次剪专家，同步收缩 gate |
| moe | `mass_variance` | 按专家质量方差剪专家 |

只给 `tolerance` 时走自动剪枝：在 `ratio_grid` 上二分查找满足容差的最大剪枝率。传入 menu 配置
（`MOE_VARIANCE_MENU_CFG` / `DENSE_RECOVERY_MENU_CFG` / `CNN_RECOVERY_MENU_CFG`）时，`prune` 改走 MENU 择优：
在 `eval_data` 指定的独立小验证集上实测每个候选，择优应用。

## 2 剪枝示例

### 2.1 使用接口方式调用

在当前目录执行以下命令运行样例（纯 CPU 可跑）：

```bash
python3 src/run_dense_samples.py   # dense：固定率 / 容差自动 / recovery-menu / 剪后量化 / evaluator
python3 src/run_cnn_samples.py     # cnn：variance vs reconstruct 通道剪枝 / recovery-menu
python3 src/run_moe_samples.py     # moe：activation_count vs mass_variance 专家剪枝 / variance-menu
```

每个样例打印剪枝前后参数量、削减比例并做一次前向校验。

> 若当前环境未安装 amct_pytorch wheel，可加 `PYTHONPATH=<amct 仓根目录>` 直接从源码运行，
> 例如 `PYTHONPATH=../../.. python3 src/run_dense_samples.py`。

## 3 社区任务样例：Qwen3.6-35B-A3B MoE 专家剪枝

本节对应 issue 183 的任务 1：`activation_count` + `tolerance`。

### 3.1 运行说明

- 校准集：Pileval，默认 `1` 个样本，`seq_len=512`
- 评估集：WikiText2 test split，固定 `seq_len=4096`，默认使用全量测试集
- 搜索模式：`tolerance`
- 输出目录：相对路径占位，脚本会保存剪枝后的模型、`PruneReport` 和运行参数
- 剪枝默认通过 `device_map=auto` 分布到两张可见 NPU 卡上
- `--device-map single` 可退回单卡；单卡默认输入设备是 `npu:0`
- Qwen3.6-35B-A3B 需要两张卡的总显存空间
- 数据集通过 `datasets` 在线拉取：Pileval 校准集与 WikiText2 test split

运行命令：

```bash
python3 src/run_qwen3_6_35b_a3b_pruning_activation_count_tolerance.py \
  --model-path ./path/to/Qwen3.6-35B-A3B \
  --save-dir ./outputs/qwen3.6_activation_count_tolerance/t0_1 \
  --tolerance 0.1 \
  --calib-samples 1 \
  --calib-seq-len 512

python3 src/run_qwen3_6_35b_a3b_pruning_activation_count_tolerance.py \
  --model-path ./path/to/Qwen3.6-35B-A3B \
  --save-dir ./outputs/qwen3.6_activation_count_tolerance/t0_2 \
  --tolerance 0.2 \
  --calib-samples 1 \
  --calib-seq-len 512

python3 src/run_qwen3_6_35b_a3b_pruning_activation_count_tolerance.py \
  --model-path ./path/to/Qwen3.6-35B-A3B \
  --save-dir ./outputs/qwen3.6_activation_count_tolerance/t0_5 \
  --tolerance 0.5 \
  --calib-samples 1 \
  --calib-seq-len 512
```

默认会自动把模型分布到可见的两张 NPU 上；如需单卡回退，可加 `--device-map single --device npu:0`。
`--preview-prune-ratio` 只用于 `prune_diagnose()` 的 dry-run，默认 `0.1`。`--eval-batches` 仅用于快速
截断评测批次，默认不传即使用全量 WikiText2 test split。搜索网格使用库内默认值
`0.1/0.2/0.3/0.4/0.5/0.6/0.7/0.8`；若需要自定义，可额外传 `--ratio-grid`。

如果要把剪枝结果接到 amct 的 eval / quantization 流程，`config.json` 的 VL 结构处理见
[Qwen3.6-Moe-Pruning.md](../../models/qwen3.6/Qwen3.6-Moe-Pruning.md)。

### 3.2 诊断与报告

脚本启动后会依次打印：

1. 完整运行参数
2. `prune_diagnose()` 的诊断摘要
3. 正式剪枝后的 `PruneReport`
4. baseline / pruned PPL 与参数量变化

`tolerance=0.1` 在 2 卡 NPU + 全量 WikiText2 test split 上的实测输出。剪枝率搜索按二分在
`ratio_grid` 上进行，依次评估 0.4 / 0.2 / 0.1 三档候选，均不满足容差，模型保持原样：

```text
[prune-diagnose]
[prune-diagnose] prunable targets: cnn=0, dense=40, moe=40
  fixed-ratio prune: ineffective (0 cut) (cut 0.0%)
  acc binary search: available (chosen prune_ratio=0.7)
  - fixed-ratio prune dry-run error: OutOfMemoryError: NPU out of memory. Tried to allocate
    1.00 GiB (NPU 0; 61.27 GiB total capacity; ...)
    # 说明：诊断干跑需复制模型，2 卡显存不足导致 OOM，仅影响该预览项，不影响正式剪枝搜索

PPL evaluation completed: 6.309916   # 搜索内部基线
PPL evaluation completed: 9.857123   # 候选 prune_ratio=0.4
PPL evaluation completed: 7.833244   # 候选 prune_ratio=0.2
PPL evaluation completed: 6.890347   # 候选 prune_ratio=0.1
PPL evaluation completed: 6.309916   # 模型未改动，复测

[PruneReport]
{
  "backend": "ModelBackend(name='huggingface')",
  "params_before": 34660610688,
  "params_after": 34660610688,
  "events": [],
  "budget_unreachable": false,
  "prunable_fraction": null,
  "warnings": [
    "no prune ratio met tolerance 0.100 across 3 candidates; model left unchanged. Relax the tolerance or add a smaller prune ratio to ratio_grid."
  ],
  "per_layer_sparsity": {},
  "allocation_choice": null
}
```

### 3.3 结果表

以下结果在 2 × Ascend 910 上实测：评估集为全量 WikiText2 test split，`seq_len=4096`（共 72 个
batch，逐条前向）；校准集 Pileval 1 个样本、`seq_len=512`。`tolerance`
语义为允许的 PPL 绝对增量，即 `剪枝后 PPL − 基线 PPL ≤ tolerance`。

| 模型 | 方法 | 实验设置 | 基线 PPL | 剪枝后 PPL | 后处理 PPL | 参数量（前 → 后） | 参数削减率 | 剪枝时长（min） |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| Qwen3.6-35B-A3B | `activation_count` | `tolerance=0.1` | 6.309916 | 6.309916 | N/A | 34660610688 -> 34660610688 | 0.00% | 18.63 |
| Qwen3.6-35B-A3B | `activation_count` | `tolerance=0.2` | 6.309916 | 6.309916 | N/A | 34660610688 -> 34660610688 | 0.00% | 17.44 |
| Qwen3.6-35B-A3B | `activation_count` | `tolerance=0.5` | 6.309916 | 6.309916 | N/A | 34660610688 -> 34660610688 | 0.00% | 18.28 |

三档搜索中评估的候选剪枝率与对应 PPL（二分顺序，跨档可复现）：

| 候选 prune_ratio | 剪枝后 PPL | ΔPPL | 满足 tolerance=0.1 / 0.2 / 0.5 |
| ---: | ---: | ---: | --- |
| 0.1 | 6.890347 | 0.580431 | 否 / 否 / 否 |
| 0.2 | 7.833244 | 1.523328 | 否 / 否 / 否 |
| 0.4 | 9.857123 | 3.547207 | 否 / 否 / 否 |

默认 `ratio_grid` 中最小候选 0.1 的 ΔPPL 已达 0.58，超出全部三档容差，因此三档实验下模型均保持
原样；如需剪枝生效，可通过 `--ratio-grid` 引入更小的候选（如 `0.02,0.05,0.1`）。


## 4 社区任务样例：Qwen3.6-35B-A3B MoE 专家剪枝

本节对应 issue 183 的任务 3：基于 `activation_count` 的 MoE 专家结构化剪枝，使用
`size_budget=0.9`、`0.8` 和 `0.5` 三档目标参数预算。

### 4.1 任务目标与实验设置

Qwen3.6-35B-A3B 包含 40 个 MoE 层，每层 256 个路由专家，每个 token 激活 8 个专家。原始 BF16 权重约占 64.56 GiB，完整模型可能超过单张 Ascend 910 卡的可用显存，因此剪枝阶段在 CPU 上加载完整模型；剪枝后模型可以继续进行 WikiText2 评测、量化或部署。

```text
模型：Qwen3.6-35B-A3B
方法：activation_count
目标：MoE experts
模式：size_budget
size_budget：0.9、0.8、0.5
校准集：Pileval（mit-han-lab/pile-val-backup，validation）
评估集：WikiText2（wikitext-2-raw-v1，test split）
序列长度：4096
校准样本数：4
后处理：size_budget=0.5 档执行 300 步恢复训练
```

`activation_count` 根据校准数据统计专家激活情况，优先删除激活次数较少的专家，并同步收缩路由器和专家权重。

### 4.2 模型准备

将 [Qwen/Qwen3.6-35B-A3B 原始权重](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) 下载到本地模型目录，例如：

```text
./path/to/Qwen3.6-35B-A3B
```

该权重为 `bfloat16`，无需格式转换。目录中应包含：

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

### 4.3 运行剪枝

正式脚本路径：

```text
src/run_qwen3_6_35b_a3b_pruning_activation_count_budget.py
```

在 `examples/algorithms/pruning` 目录执行以下命令。每个预算都应从同一个原始模型目录独立运行：

```bash
python3 src/run_qwen3_6_35b_a3b_pruning_activation_count_budget.py \
  --model_path ./path/to/Qwen3.6-35B-A3B \
  --size_budget 0.9 \
  --seq_len 4096 \
  --calibration_samples 4 \
  --pileval_source modelscope \
  --ratio_grid 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 \
  --report_dir ./outputs/qwen3_6_pruning_results \
  --save_model_dir ./outputs/Qwen3.6-35B-A3B-pruned-budget09

python3 src/run_qwen3_6_35b_a3b_pruning_activation_count_budget.py \
  --model_path ./path/to/Qwen3.6-35B-A3B \
  --size_budget 0.8 \
  --seq_len 4096 \
  --calibration_samples 4 \
  --pileval_source modelscope \
  --ratio_grid 0.2,0.3,0.4 \
  --report_dir ./outputs/qwen3_6_pruning_results \
  --save_model_dir ./outputs/Qwen3.6-35B-A3B-pruned-budget08

python3 src/run_qwen3_6_35b_a3b_pruning_activation_count_budget.py \
  --model_path ./path/to/Qwen3.6-35B-A3B \
  --size_budget 0.5 \
  --seq_len 4096 \
  --calibration_samples 4 \
  --pileval_source modelscope \
  --ratio_grid 0.5,0.6,0.7 \
  --report_dir ./outputs/qwen3_6_pruning_results \
  --save_model_dir ./outputs/Qwen3.6-35B-A3B-pruned-budget05
```

脚本在 CPU 上加载完整模型并执行结构化剪枝，完成后对剪枝模型执行一次前向检查。只有传入
`--save_model_dir` 时，剪枝后的 config、tokenizer 和 safetensors 权重才会写入输出目录；未传该参数时，进程退出后只保留 JSON 报告。

### 4.4 诊断与报告

正式脚本调用 `amct_pytorch.pruning.prune_diagnose`，诊断结果确认：

```text
cnn targets:   0
dense targets: 40
moe targets:   40
fixed-ratio pruning: available
accuracy search:     available
```

本样例只启用 `moe` 目标，因此实际只修改 MoE 专家结构。剪枝完成后，脚本检查：

```text
logits shape: (1, 4096, 248320)
logits finite: True
```

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

完整 `events`、`per_layer_sparsity` 和 `warnings` 保存在脚本输出的 JSON 报告中，默认目录为：

```text
./outputs/qwen3_6_pruning_results/
```

### 4.5 WikiText2 评测与结果

PPL 评测使用 `wikitext-2-raw-v1` 的 test split，固定 `seq_len=4096`。Pileval 只用于校准，不能用于 PPL 评估。原始模型或剪枝模型无法完整放入单卡时，使用 AMCT 的 blockwise 评测入口：

```bash
python3 -m amct_pytorch.eval \
  --trust_remote_code \
  --model ./path/to/Qwen3.6-35B-A3B \
  --model_name qwen3_6_moe \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --eval_mode bf16 \
  --bit_config amct_pytorch/configs/bf16.yaml
```

评测剪枝模型时，只替换 `--model` 路径，例如：

```bash
python3 -m amct_pytorch.eval \
  --trust_remote_code \
  --model ./outputs/Qwen3.6-35B-A3B-pruned-budget09 \
  --model_name qwen3_6_moe \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --eval_mode bf16 \
  --bit_config amct_pytorch/configs/bf16.yaml
```

| 模型            | 方法               | 实验设置          |          基线 PPL |         剪枝后 PPL |        后处理 PPL | 参数量（前 → 后）               | 参数削减率 | 剪枝时长（min） |
| --------------- | ------------------ | ----------------- | ----------------: | -----------------: | ----------------: | ------------------------------- | ---------: | --------------: |
| Qwen3.6-35B-A3B | `activation_count` | `size_budget=0.9` | 6.308547019958496 |  6.664599895477295 |               N/A | 34,660,610,688 → 28,239,147,648 |   18.5267% |           28.89 |
| Qwen3.6-35B-A3B | `activation_count` | `size_budget=0.8` | 6.308547019958496 | 7.1991167068481445 |               N/A | 34,660,610,688 → 24,965,460,608 |   27.9717% |           16.84 |
| Qwen3.6-35B-A3B | `activation_count` | `size_budget=0.5` | 6.308547019958496 |  9.780414581298828 | 8.294227600097656 | 34,660,610,688 → 15,270,310,528 |   55.9433% |           20.46 |

### 4.6 `size_budget=0.5` 恢复训练

本次仅对参数量最小、可在单张 64 GiB Ascend 910 上完成全参数训练的 `size_budget=0.5` 模型执行恢复训练。恢复训练从剪枝模型开始，未重新执行剪枝，也未覆盖剪枝模型。

```text
输入模型：./outputs/Qwen3.6-35B-A3B-pruned-budget05
输出模型：./outputs/Qwen3.6-35B-A3B-pruned-budget05-finetuned
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

选择无动量 SGD 是为了避免 AdamW 为参数额外维护一阶、二阶动量状态。由于 SGD 不会像 AdamW 一样按梯度统计量自适应缩放，本次使用 `lr=1e-2`，而不是直接沿用 AdamW 常见的 `1e-5` 或 `2e-5` 量级。

`prune_finetune()` 的默认因果语言模型损失接受包含 `input_ids` 的字典，并将 `input_ids` 同时作为 labels。本次显式传入无动量 SGD：

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

实际运行时还需在导入 Transformers 前屏蔽不兼容的可选 `torchaudio`，保存 tokenizer，并保留剪枝模型外层 `config.json` 中的 Qwen3.6 wrapper 配置。恢复训练完成后，使用本节相同的 WikiText2 test、`seq_len=4096`、blockwise BF16 命令评测输出目录。

本次恢复训练结果：

```text
原始模型 PPL：      6.308547019958496
剪枝后 PPL：        9.780414581298828
恢复训练后 PPL：    8.294227600097656
```

相对未经恢复的剪枝模型，PPL 绝对下降 `1.486186981201172`，相对下降约 `15.20%`；若以剪枝造成的 PPL 增量为损失，本次恢复约 `42.81%`。恢复后 PPL 仍比原始模型高 `1.985680580139160`，因此当前结果是明显恢复但尚未恢复到基线。

### 4.7 继续恢复时可调整的参数

若需要继续提高恢复程度，可每次只改变少量参数，并用相同的 WikiText2 test 命令比较 PPL：

1. `steps` 和训练数据量：增加独立训练 batch 和训练步数。若 `steps` 超过 batch 数，`prune_finetune()` 会循环复用数据，应优先增加不同的训练 token。
2. `lr`：`1e-2` 是无动量 SGD 的起点，可对照 `3e-3` 和 `1e-2` 等值；学习率过高可能使 loss 发散。
3. `warmup`：训练步数增加时同步增加 warmup，并保持合理比例。本次为 `20/300`。
4. `seq_len`：更长上下文更接近 `seq_len=4096` 评测，但会增加激活显存；本次峰值保留显存已达 `59.793 GiB`，提高前必须重新做 1 步和 10 步显存测试。
5. 训练数据：可增加数量和多样性，但不得使用 WikiText2 test split 训练。
6. `grad_clip`：本次为 `1.0`；若梯度或 loss 不稳定可以调小，最终仍以固定评测集 PPL 为准。

多组恢复参数应使用独立报告和输出目录，避免覆盖剪枝模型。BF16 的 `size_budget=0.5` 权重约占 28.4 GiB；磁盘不足时，记录 PPL 和报告后再清理不再需要的恢复模型。
