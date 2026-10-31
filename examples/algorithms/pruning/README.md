# AMCT 结构化剪枝样例

演示 `amct_pytorch.pruning` 接口在三种域上的调用：dense FFN（剪中间维）/ CNN（剪通道）/
MoE（剪专家）。目录里既有随机初始化的微型模型演示，也有面向社区任务的 Qwen3.6-35B-A3B
MoE 剪枝样例。

> 接口详见 [`amct_pytorch/pruning/README.md`](../../../amct_pytorch/pruning/README.md)。

## 1 剪枝前提

### 1.1 安装依赖

依赖见 [requirements.txt](requirements.txt)：`torch` 与 `transformers`（随 amct_pytorch 导入链需要）。
在 NPU 上运行时另需与 Python/torch 版本匹配的 `TorchNPU` 及已安装的 CANN 包。

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
INFO [prune-diagnose]
INFO [prune-diagnose] prunable targets: cnn=0, dense=40, moe=40
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

## 4 社区任务样例：Qwen3.6-35B-A3B MoE `mass_variance` + `tolerance`（任务 2）

对应 Issue [#183](https://gitcode.com/cann/amct/issues/183) 任务 2：对真实 MoE 模型做专家结构化剪枝。
脚本：[`src/run_qwen3_6_35b_a3b_pruning_mass_variance_tolerance.py`](src/run_qwen3_6_35b_a3b_pruning_mass_variance_tolerance.py)。

### 4.1 运行说明

环境：CANNLab / 云开发 **NPU A3**（建议可见 **2 张 NPU**，主机内存 ≥150GB）。
默认 `--device-map auto`：模型分片到 NPU，**tolerance 搜索与验收均在 NPU 上算 WikiText2 PPL**
（与任务 1 一致）。仅在 NPU 显存不足时改用 `--device-map cpu`。

`tolerance` 语义为允许的 **绝对 PPL 增量**：`剪枝后 PPL − 基线 PPL ≤ tolerance`。

```bash
# 仓库根目录
source /home/developer/Ascend/cann/set_env.sh
export MODEL_DIR=/data/models/Qwen3.6-35B-A3B   # 占位路径，改为本地权重目录
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

# 先跑基线（只需一次）
python3 -m amct_pytorch.eval \
  --trust_remote_code \
  --model "$MODEL_DIR" \
  --model_name qwen3_6_moe \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --eval_mode bf16 \
  --bit_config amct_pytorch/configs/bf16.yaml

# 三档 tolerance 各自从原始模型独立跑（默认 WikiText2 PPL 搜索 + 默认 ratio_grid）
# Qwen3.6 本地权重需显式打开 --trust-remote-code（仅用于你信任的本地目录）
# 将 BASELINE_PPL 换成你测到的基线
BASELINE_PPL=6.308547
for T in 0.1 0.2 0.5; do
  python3 examples/algorithms/pruning/src/run_qwen3_6_35b_a3b_pruning_mass_variance_tolerance.py \
    --model "$MODEL_DIR" \
    --trust-remote-code \
    --tolerance "$T" \
    --baseline-ppl "$BASELINE_PPL" \
    --skip-baseline-eval \
    --search-evaluator wikitext2_ppl \
    --device-map auto \
    --output-dir ./output/qwen3_6_35b_a3b_mass_variance_tolerance
done
```

说明：默认 `ratio_grid` 最小候选为 0.1；若该档 ΔPPL 已超过 0.1/0.2/0.5，则三档均可
能保持原模型不变（与任务 1 同类现象，属有效结果）。若需剪枝生效，可传更细网格，例如
`--ratio-grid 0.02,0.05,0.1`。

主要参数：

| 参数 | 含义 |
|:--|:--|
| `--tolerance` | 允许的 WikiText2 绝对 PPL 增量；任务要求分别跑 `0.1` / `0.2` / `0.5` |
| `--trust-remote-code` | 允许执行模型目录内自定义代码；默认关闭。`--model` 必须是本地目录；仅对信任的本地权重开启 |
| `--device-map auto` | 默认；模型分片到可见 NPU（同任务 1）。`single`/`cpu` 为回退 |
| `--search-evaluator wikitext2_ppl` | 默认；搜索与验收口径一致（WikiText2 PPL）。`fidelity` / `proxy_ppl` 仅供对比，不可与结果表混用 |
| `--eval-batches` | 可选，截断搜索用 WikiText2 batch 数；默认全量 test split |
| `--ratio-grid` | 可选，逗号分隔剪枝率；默认库内 `DEFAULT_RATIO_GRID` |
| `--calib-nsamples` / `--calib-seq-len` | Pileval 校准条数与长度（默认 8 / 512） |
| `--eval-seq-len` | 评估序列长度，固定 4096 |
| `--output-dir` | 产物目录（相对/占位路径） |

校准集：`mit-han-lab/pile-val-backup`；评估集：WikiText2 `wikitext-2-raw-v1` test。
保存剪后权重后会把 `config.json` 恢复为 VL 嵌套结构，并更新 `text_config.num_experts`，以便
`qwen3_6_moe` 评估通路加载。

### 4.2 诊断与报告

实测配置：`--device-map auto`、`--search-evaluator wikitext2_ppl`、`search_device=npu:0`。

`tolerance=0.1` / `0.2`：默认 `ratio_grid` 上候选均不满足容差，模型保持原样，`PruneReport`
含类似 warning：

```text
no prune ratio met tolerance 0.100 across 3 candidates; model left unchanged.
```

`tolerance=0.5`：搜索选中可剪比例并真正剪枝，参数量下降；`PruneReport` 中 `params_after`
与各层 `mass_variance` 事件见对应 `result.json`。

### 4.3 结果表

实测环境：云开发 A3，`--device-map auto`，WikiText2 PPL，`seq_len=4096`，
`--search-evaluator wikitext2_ppl`。

| 模型 | 方法 | 实验设置 | 基线 PPL | 剪枝后 PPL | 后处理 PPL | 参数量（前 → 后） | 参数削减率 | 剪枝时长（min） |
|:--|:--|:--|--:|--:|:--|:--|--:|--:|
| Qwen3.6-35B-A3B | `mass_variance` | `tolerance=0.1` | 6.308547 | 6.308547 | N/A | 34660610688 → 34660610688 | 0.0000 | 17.95 |
| Qwen3.6-35B-A3B | `mass_variance` | `tolerance=0.2` | 6.308547 | 6.308547 | N/A | 34660610688 → 34660610688 | 0.0000 | 17.12 |
| Qwen3.6-35B-A3B | `mass_variance` | `tolerance=0.5` | 6.308547 | 6.586955 | N/A | 34660610688 → 31386923648 | 0.0945 | 17.99 |

说明：`0.1` / `0.2` 在默认网格下无候选满足绝对 PPL 容差（与任务 1 同类，属有效结果）；
`0.5` 档 ΔPPL≈0.278 ≤ 0.5，剪枝生效，削减率约 9.45%。字段来自各档 `result.json`。

## 5 社区任务样例：`mass_variance` + 量化（任务 18）

任务 18 在 `tolerance=0.1` 的 MoE 专家剪枝后，继续使用 Qwen3.6 的公开
`amct_pytorch.eval` blockwise 流程做 W8A8 `attn-linear + moe` 量化评估。脚本位于
[`src/run_qwen3_6_35b_a3b_pruning_mass_variance_tolerance_quant.py`](src/run_qwen3_6_35b_a3b_pruning_mass_variance_tolerance_quant.py)。
模型权重和数据集均不提交，路径为占位的相对路径。

### 5.1 运行说明

```bash
python3 examples/algorithms/pruning/src/run_qwen3_6_35b_a3b_pruning_mass_variance_tolerance_quant.py \
  --model ./path/to/Qwen3.6-35B-A3B \
  --trust-remote-code \
  --device-map auto \
  --tolerance 0.1 \
  --ratio-grid 0.05 \
  --bit-config amct_pytorch/configs/w8a8.yaml \
  --quant-target attn-linear moe --quant-dtype int \
  --output-dir ./output/qwen3_6_35b_a3b_mass_variance_quant
```

脚本会先从原始模型收集 Pileval 校准数据，并执行 `mass_variance` 容差搜索；随后对剪枝模型执行
WikiText2 `test` split、`seq_len=4096` 的量化 PPL 评估。若默认网格下没有候选满足容差，剪枝模型
保持原样，量化仍会对原始模型执行。完整命令、剪枝日志尾部、量化 PPL 和返回码写入 `result.json`。
`--skip-quant-eval` 只用于检查剪枝流程，不属于正式结果。

### 5.2 诊断与报告

启动时打印完整参数、剪枝命令和量化命令。剪枝阶段的 `prune_diagnose()` 与 `PruneReport` 保存在
子任务输出目录的 `result.json`；量化阶段输出 `PPL evaluation completed: ...`，并记录为 `quant_ppl`。
量化使用 `amct_pytorch.eval` 而不是经典 `amct.quantize`，因为 Qwen3.6 的融合 MoE 需要专用模型工作流。

### 5.3 结果表

| 模型 | 方法 | 实验设置 | 基线 PPL | 剪枝后 PPL | 后处理 PPL | 参数量（前 → 后） | 参数削减率 | 剪枝时长（min） |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| Qwen3.6-35B-A3B | `mass_variance` + `quant` | `tolerance=0.1`, `ratio_grid=0.05`, W8A8 `attn-linear + moe` | 6.308547 | 6.354577 | 6.449746 | 34660610688 → 33023767168 | 0.0472 | 10.39 |

结果来自 A3 双 NPU 实测；剪枝报告和量化日志保存在输出目录的 `result.json`。

## 6 社区任务样例：Qwen3.6-35B-A3B MoE `activation_count` + `budget`（任务 3）

本节对应 issue 183 的任务 3：基于 `activation_count` 的 MoE 专家结构化剪枝，使用
`size_budget=0.9`、`0.8` 和 `0.5` 三档目标参数预算。

### 6.1 任务目标与实验设置

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

### 6.2 模型准备

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

### 6.3 运行剪枝

在 `amct/examples/algorithms/pruning` 目录执行以下命令。每个预算都应从同一个原始模型目录独立运行：

```bash
python3 src/run_qwen3_6_35b_a3b_pruning_activation_count_budget.py \
  --model_path ./path/to/Qwen3.6-35B-A3B \
  --size_budget 0.9 \
  --seq_len 4096 \
  --calibration_samples 4 \
  --pileval_source modelscope \
  --ratio_grid 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8 \
  --report_dir ./outputs/qwen3_6_pruning_results \
  --save_model_dir ./outputs/Qwen3.6-35B-A3B-pruned-budget09 \
  --trust_remote_code

python3 src/run_qwen3_6_35b_a3b_pruning_activation_count_budget.py \
  --model_path ./path/to/Qwen3.6-35B-A3B \
  --size_budget 0.8 \
  --seq_len 4096 \
  --calibration_samples 4 \
  --pileval_source modelscope \
  --ratio_grid 0.2,0.3,0.4 \
  --report_dir ./outputs/qwen3_6_pruning_results \
  --save_model_dir ./outputs/Qwen3.6-35B-A3B-pruned-budget08 \
  --trust_remote_code

python3 src/run_qwen3_6_35b_a3b_pruning_activation_count_budget.py \
  --model_path ./path/to/Qwen3.6-35B-A3B \
  --size_budget 0.5 \
  --seq_len 4096 \
  --calibration_samples 4 \
  --pileval_source modelscope \
  --ratio_grid 0.5,0.6,0.7 \
  --report_dir ./outputs/qwen3_6_pruning_results \
  --save_model_dir ./outputs/Qwen3.6-35B-A3B-pruned-budget05 \
  --trust_remote_code
```

脚本在 CPU 上加载完整模型并执行结构化剪枝，完成后对剪枝模型执行一次前向检查。只有传入
`--save_model_dir` 时，剪枝后的 config、tokenizer 和 safetensors 权重才会写入输出目录；未传该参数时，进程退出后只保留 JSON 报告。
参数说明：
| 参数 | 默认值 | 必填 | 含义 |
| --- | --- | --- | --- |
| `--model_path` | 无 | 是 | 本地模型目录 |
| `--size_budget` | 无 | 是 | 目标参数预算，可选 0.9 / 0.8 / 0.5 |
| `--seq_len` | 4096 | 否 | 校准序列长度 |
| `--calibration_samples` | 4 | 否 | 校准样本数量 |
| `--pileval_source` | `huggingface` | 否 | Pileval 数据来源：`huggingface` 或 `modelscope` |
| `--ratio_grid` | `0.1,0.2,0.3` | 否 | AMCT 尝试的候选专家剪枝比例，逗号分隔且须在 0~1 之间 |
| `--report_dir` | 无 | 是 | JSON 剪枝报告输出目录 |
| `--save_model_dir` | `""` | 否 | 剪枝模型保存目录，为空则不保存权重 |
| `--trust_remote_code` | `False` | 否 | 允许执行模型目录内自定义代码；默认关闭。--model_path 必须是本地目录；仅对信任的本地权重开启 |

### 6.4 诊断与报告

正式脚本调用 `amct_pytorch.pruning.prune_diagnose`，例如 size_budget=0.9 时诊断结果如下：

```text
{
  "targets": {
    "cnn": 0,
    "dense": 40,
    "moe": 40
  },
  "prune_works": true,
  "prune_reduction": 0.09444977959183498,
  "prune_forward_ok": true,
  "search_works": true,
  "search_chosen_ratio": 0.5,
  "notes": []
}
```

正式剪枝后的 `PruneReport` 核心内容：

```text
params_before: 34660610688
params_after: 28239147648
budget_unreachable: False
prunable_fraction: 0.9335997444269842
```

完整内容保存在脚本输出的 JSON 报告中。

### 6.5 WikiText2 评测与结果

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
| Qwen3.6-35B-A3B | `activation_count` | `size_budget=0.9` | 6.308547019958496 |  6.664599895477295 |               N/A | 34,660,610,688 → 28,239,147,648 |   18.5267% |           20.11 |
| Qwen3.6-35B-A3B | `activation_count` | `size_budget=0.8` | 6.308547019958496 | 7.1991167068481445 |               N/A | 34,660,610,688 → 24,965,460,608 |   27.9717% |           15.44 |
| Qwen3.6-35B-A3B | `activation_count` | `size_budget=0.5` | 6.308547019958496 |  9.780414581298828 | 8.294227600097656 | 34,660,610,688 → 15,270,310,528 |   55.9433% |           15.07 |

### 6.6 `size_budget=0.5` 恢复训练

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

相对未经恢复的剪枝模型，PPL 绝对下降 `1.486186981201172`，相对下降约 `15.20%`；若以剪枝造成的 PPL 增量为损失，本次恢复约 `42.81%`。恢复后 PPL 仍比原始模型高 `1.985680580139160`，因此当前结果是有所恢复但尚未恢复到基线。

### 6.7 继续恢复时可调整的参数

若需要继续提高恢复程度，可每次只改变少量参数，并用相同的 WikiText2 test 命令比较 PPL：

1. `steps` 和训练数据量：增加独立训练 batch 和训练步数。若 `steps` 超过 batch 数，`prune_finetune()` 会循环复用数据，应优先增加不同的训练 token。
2. `lr`：`1e-2` 是无动量 SGD 的起点，可对照 `3e-3` 和 `1e-2` 等值；学习率过高可能使 loss 发散。
3. `warmup`：训练步数增加时同步增加 warmup，并保持合理比例。本次为 `20/300`。
4. `seq_len`：更长上下文更接近 `seq_len=4096` 评测，但会增加激活显存；本次峰值保留显存已达 `59.793 GiB`，提高前必须重新做 1 步和 10 步显存测试。
5. 训练数据：可增加数量和多样性，但不得使用 WikiText2 test split 训练。
6. `grad_clip`：本次为 `1.0`；若梯度或 loss 不稳定可以调小，最终仍以固定评测集 PPL 为准。

多组恢复参数应使用独立报告和输出目录，避免覆盖剪枝模型。BF16 的 `size_budget=0.5` 权重约占 28.4 GiB；磁盘不足时，记录 PPL 和报告后再清理不再需要的恢复模型。

> 剪枝完成后，如需将模型导出为 torchair 离线模型（.air）用于 ATC 转 om 板端推理，请参考 [AMCT大模型torchair离线模型导出](../export/README.md)。

## 7 社区任务样例：Qwen3-0.6B Dense FFN `low_variance` + `ratio`（任务 5）

本节对应 issue [#183](https://gitcode.com/cann/amct/issues/183) 任务 5：对 Qwen3-0.6B 的
dense FFN 中间维做 `low_variance` 结构化剪枝，`ratio` 模式固定剪枝率，`prune_ratio=0.1/0.2/0.5`
三档。脚本：[`src/run_qwen3_0_6b_pruning_low_variance_ratio.py`](src/run_qwen3_0_6b_pruning_low_variance_ratio.py)。

### 7.1 运行说明

```text
模型：Qwen3-0.6B（ModelScope 下载到本地，约 1.5 GiB bf16）
方法：low_variance（dense FFN 中间维，自动避开注意力投影）
模式：ratio（config 内直接指定 prune_ratio，无搜索）
prune_ratio：0.1、0.2、0.5（三档均从未剪枝原始模型独立开始）
校准集：Pileval（mit-han-lab/pile-val-backup，validation），8 个样本、seq_len=512
评估集：WikiText2（wikitext-2-raw-v1，test split），全量、seq_len=4096
设备：单卡 npu:0
```

模型从 [ModelScope](https://modelscope.cn) 或 HuggingFace 下载到本地目录（占位路径如下），
权重与数据集均不提交仓库：

```bash
# 可选：从 ModelScope 下载（或任选 hf-mirror /官方 HF）
pip3 install modelscope
modelscope download --model Qwen/Qwen3-0.6B --local_dir ./path/to/Qwen3-0.6B
```

在 `examples/algorithms/pruning` 目录逐档运行（每档都从原始模型独立开始）：

```bash
export PYTHONPATH=<amct 仓根目录>   # 源码运行时需要
# 若 huggingface.co 不可达，数据集走镜像：export HF_ENDPOINT=https://hf-mirror.com

python3 src/run_qwen3_0_6b_pruning_low_variance_ratio.py \
  --model-path ./path/to/Qwen3-0.6B \
  --save-dir ./outputs/qwen3_0_6b_low_variance_ratio/r0_1 \
  --prune-ratio 0.1 --calib-samples 8 --calib-seq-len 512

python3 src/run_qwen3_0_6b_pruning_low_variance_ratio.py \
  --model-path ./path/to/Qwen3-0.6B \
  --save-dir ./outputs/qwen3_0_6b_low_variance_ratio/r0_2 \
  --prune-ratio 0.2 --calib-samples 8 --calib-seq-len 512

python3 src/run_qwen3_0_6b_pruning_low_variance_ratio.py \
  --model-path ./path/to/Qwen3-0.6B \
  --save-dir ./outputs/qwen3_0_6b_low_variance_ratio/r0_5 \
  --prune-ratio 0.5 --calib-samples 8 --calib-seq-len 512
```

主要参数：

| 参数 | 默认值 | 含义 |
|:--|:--|:--|
| `--model-path` | 无（必填） | 本地 Qwen3-0.6B 权重目录，相对/占位路径 |
| `--prune-ratio` | 无（必填） | FFN 中间维固定剪枝率，任务要求 0.1 / 0.2 / 0.5 |
| `--save-dir` | `./outputs/qwen3_0_6b_low_variance_ratio` | 报告与（可选）剪枝模型输出目录 |
| `--device` | `npu:0` | 模型放置与评测设备；单卡即可 |
| `--calib-samples` / `--calib-seq-len` | 8 / 512 | Pileval 校准条数与长度 |
| `--seq-len` | 4096 | WikiText2 评测序列长度 |
| `--eval-batches` | 全量 | 可选，截断评测 batch 数（正整数），仅用于快速验证 |
| `--preview-prune-ratio` | 0.1 | `prune_diagnose()` dry-run 用的预览剪枝率，不影响正式剪枝 |
| `--skip-diagnose` | 关闭 | 追加后跳过 `prune_diagnose()` 预览，节省其耗时（三档实测约 0.1 min） |
| `--save-model` | 关闭 | 追加后保存剪枝模型与 tokenizer 到 `--save-dir` |

脚本启动时打印完整运行参数，并保存 `run_command.txt` / `run_environment.txt` /
`run_config.json` 以便复现；`run_environment.txt` 同时记录 python / amct_pytorch / torch /
torch_npu / transformers / CANN toolkit / NPU 驱动版本号（不可得时记为 null）。输出目录被
`.gitignore` 排除，不会提交权重或数据。

### 7.2 诊断与报告

脚本依次打印：完整运行参数 → baseline PPL → `prune_diagnose()` 诊断摘要 → 正式剪枝
`PruneReport` → 剪枝后 PPL 与参数量变化。`prune_ratio=0.1` 档实测输出（单卡 npu:0）：

```text
INFO [qwen3_0_6b.low_variance.ratio] baseline ppl: 19.153452
INFO [prune-diagnose] prunable targets: cnn=0, dense=28, moe=0
  fixed-ratio prune: available (cut 4.4%, forward OK)
  acc binary search: unavailable
  - acc search found no acceptable prune ratio at this tolerance (relax the tolerance).
INFO [qwen3_0_6b.low_variance.ratio] diagnose time: 0.11 min
```

说明：诊断的 fixed-ratio 前向校验要求校准数据与模型同设备，脚本已把校准 batch 对齐到
`--device`。acc binary search 预览未找到可接受剪枝率属正常——`low_variance` 切片在默认
容差内损失过大；`ratio` 模式的正式剪枝不依赖该搜索结果。诊断耗时单列打印并写入
`run_metrics.json` 的 `diagnose_time_minutes`（三档实测约 0.1 min），不需要预览时可加
`--skip-diagnose` 跳过。

`prune_ratio=0.1` 档正式剪枝后的 `PruneReport` 核心内容（完整 JSON 保存在
`outputs/.../prune_report.json`）：

```json
{
  "backend": "huggingface",
  "params_before": 596049920,
  "params_after": 569643008,
  "budget_unreachable": false,
  "prunable_fraction": null,
  "warnings": [],
  "per_layer_sparsity": {"model.layers.0.mlp": 0.099935, "...": "共 28 层，数值相同"},
  "events": [
    {"stage": "dense", "method": "low_variance", "module": "model.layers.0.mlp",
     "detail": "Activation variance prune 3072 -> 2765"}
  ]
}
```

三档的每层中间维变化（28 层一致）：`0.1 → 3072→2765`（削减 9.99%）、
`0.2 → 3072→2458`（19.99%）、`0.5 → 3072→1536`（50.00%）。

### 7.3 结果表

实测环境：单卡 Ascend 910（npu:0），bf16，全量 WikiText2 test split（73 个 4096-token batch，
逐条前向）；校准集 Pileval 8 样本、`seq_len=512`。基线 PPL 三次独立运行均为 `19.153452`。

| 模型 | 方法 | 实验设置 | 基线 PPL | 剪枝后 PPL | 后处理 PPL | 参数量（前 → 后） | 参数削减率 | 剪枝时长（min） | 诊断耗时（min） |
|:--|:--|:--|--:|--:|:--|:--|--:|--:|--:|
| Qwen3-0.6B | `low_variance` | `prune_ratio=0.1` | 19.153452 | 35.085606 | N/A | 596,049,920 → 569,643,008 | 4.43% | 0.02 | 0.11 |
| Qwen3-0.6B | `low_variance` | `prune_ratio=0.2` | 19.153452 | 75.769569 | N/A | 596,049,920 → 543,236,096 | 8.86% | 0.02 | 0.11 |
| Qwen3-0.6B | `low_variance` | `prune_ratio=0.5` | 19.153452 | 2809.957520 | N/A | 596,049,920 → 463,929,344 | 22.17% | 0.02 | 0.10 |

说明：`low_variance` 无后处理步骤，后处理 PPL 记为 N/A。参数削减率低于剪枝率是因为
dense FFN 中间维只占 Qwen3-0.6B 总参数的一部分（embedding、attention 与 lm_head 未剪）。
剪枝本身只做方差统计与权重切片，三档耗时均约 0.02 min；表中的 `剪枝时长` 仅指
`amct.prune()` 正式剪枝一步，诊断 dry-run 耗时单列（可通过 `--skip-diagnose` 跳过）。
剪枝后 PPL 随剪枝率快速上升，
0.5 档已明显劣化，说明该规模下纯 `low_variance` 切片无补偿时高剪枝率不可用，可结合
`reconstruct`（recovery 补偿，接口调用见[第 2 节](#2-剪枝示例)）或恢复训练
（做法参考[第 6.6 节](#66-size_budget05-恢复训练)）缓解。

> 剪枝完成后，如需将模型导出为 torchair 离线模型（.air）用于 ATC 转 om 板端推理，请参考 [AMCT大模型torchair离线模型导出](../export/README.md)。

## 8 社区任务样例：Qwen3-0.6B dense FFN 剪枝（`reconstruct` + `finetune`，任务 17）

本节对应 issue 183 的任务 17：dense FFN 中间维 `reconstruct`（最小二乘恢复）固定 `prune_ratio=0.5`
剪枝，剪枝后进行恢复微调（finetune）。

### 8.1 运行说明

- 校准集：Pileval，`8` 个样本、`seq_len=2048`（共 16384 行校准 token；`reconstruct` 的最小二乘
  恢复要求校准行数不低于保留神经元数，否则自动回退朴素切片并在日志中给出告警）
- 评估集：WikiText2 test split，固定 `seq_len=4096`，使用全量测试集（73 个 batch）
- 微调集：WikiText2 train split，300 个互不重复的连续 512-token 块（不使用 test split 训练）
- 微调设置：无动量 SGD（lr=1e-2）、warmup=20、300 步、grad_clip=1.0、gradient checkpointing
  （优化器选型经同协议对照实验验证：同设置下 AdamW 发散，见 8.3 对照表）
- 输出目录为相对路径占位，脚本保存 `PruneReport`、诊断摘要与运行参数（启动时打印完整参数）；如需保存微调后的模型权重，可在运行命令中追加可选参数 `--save-model`
- 模型加载：默认 `trust_remote_code=False` 并强制 `local_files_only=True`（仅从本地目录离线加载；
  Qwen3 为 transformers 原生支持的架构，无需执行模型目录中的自定义代码）。确有需要时可显式传入
  `--trust-remote-code`，脚本会打印安全告警

运行命令：

```bash
python3 src/run_qwen3_0_6b_pruning_reconstruct_ratio_finetune.py \
  --model-path ./path/to/Qwen3-0.6B \
  --save-dir ./outputs/qwen3_0_6b_reconstruct_ratio_finetune/r0_5 \
  --prune-ratio 0.5
```

不使用微调（仅剪枝）时可加 `--no-finetune`；`--preview-prune-ratio` 仅用于 `prune_diagnose()` 的
dry-run，默认 `0.1`；`--eval-batches` 仅用于快速截断评测批次，默认不传即全量 WikiText2 test split。

### 8.2 诊断与报告

`prune_diagnose()` 在单卡 910B3 上识别出 dense 域 28 个可剪目标（28 层 FFN）；剪枝后模型
`config` 的 `intermediate_size` 已同步收窄，可直接 `save_pretrained/from_pretrained` 往返。
`prune_ratio=0.5` 主运行（2026-10-21 复测）的诊断实际输出与耗时：

```text
INFO [qwen3.0.6b.reconstruct.ratio.finetune] diagnose time: 30.31 min
INFO [qwen3.0.6b.reconstruct.ratio.finetune] prune_diagnose():
INFO [prune-diagnose] prunable targets: cnn=0, dense=28, moe=0
  fixed-ratio prune: available (cut 4.4%, forward OK)
  acc binary search: available (chosen prune_ratio=0.1)
```

acc 搜索的每轮候选都要跑真实 reconstruct 剪枝试验（28 层稠密层各解一次岭回归），故诊断耗时
显著长于 low_variance 类样例；`run_metrics.json` 以 `diagnose_time_minutes` 单列记录，本次
实测 `30.31 min`，不需要预览时可加 `--skip-diagnose` 跳过。

正式剪枝后的 `PruneReport` 核心内容（完整 JSON 见 `outputs/.../prune_report.json`，30 条事件、
无 warnings）：

```json
{
  "backend": "huggingface",
  "params_before": 596049920,
  "params_after": 463929344,
  "budget_unreachable": false,
  "prunable_fraction": null,
  "warnings": [],
  "per_layer_sparsity": {"model.layers.0.mlp.down_proj": 0.5, "...": "共 28 层，数值相同"},
  "events": [
    {"stage": "dense", "method": "reconstruct", "module": "model.layers.0.mlp.down_proj",
     "detail": "output-preserving (ls) prune 3072 -> 1536"}
  ]
}
```

### 8.3 结果表

以下结果在单张 Ascend 910B3（64 GiB）上实测；评估集为全量 WikiText2 test split、`seq_len=4096`
（73 个 batch，逐条前向）；校准集 Pileval 8 个样本、`seq_len=2048`。

| 模型 | 方法 | 实验设置 | 基线 PPL | 剪枝后 PPL | 后处理 PPL | 参数量（前 → 后） | 参数削减率 | 剪枝墙钟（min） |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| Qwen3-0.6B | `reconstruct` + `finetune` | `prune_ratio=0.5` | 19.157549 | 74.244453 | 31.369858 | 596049920 -> 463929344 | 22.17% | 36.5 |

> 后处理 PPL 为恢复微调（finetune）后的结果；微调从剪枝模型开始，未重新执行剪枝。相对剪枝模型
> PPL 绝对下降 `42.874595`，按"剪枝造成的 PPL 增量"计恢复约 `77.8%`；恢复后 PPL 仍高于原始模型
> `12.212309`。
>
> 计时口径（2026-10-21 复测，全程记录于 `run_metrics.json`）：issue 要求的剪枝墙钟（自校准数据
> 收集起、至剪枝完成止）为 `36.5 min`，其中校准数据收集 `0.1 min`、`prune_diagnose()` 诊断
> `30.3 min`（单列，见 8.2）、`amct.prune` 调用 `5.8 min`；300 步恢复微调 `1.6 min`，单列不计入
> 剪枝墙钟。复测的基线/剪枝后 PPL 与主表逐位一致，微调后 PPL `31.302790` 与下表 SGD 对照行逐位
> 一致（管线确定性验证）。

**微调优化器对照（2026-10-18，910B3 复测）**：两组其余设置完全相同（`lr=1e-2`、300 步、
warmup=20、grad_clip=1.0、同一批 300 个训练块），仅替换优化器；两组剪枝起点逐位一致
（剪枝后 PPL 均为 74.244453），确认唯一变量为优化器本身。

关于两组 SGD 数值：主表的 `31.369858` 与表下注脚的 `77.8%` 取自 issue 183 引用的主运行；下表 SGD 行取自 2026-10-18 的独立对照运行，`31.302790` / `77.9%`。两次运行从同一剪枝检查点出发、协议相同，但微调结果并非逐位一致，PPL 相差 `0.067068`（约 0.2%）。两张表各自保留当次运行的数值，issue 183 与主表继续为准。

| 微调优化器 | 基线 PPL | 剪枝后 PPL | 微调后 PPL | 微调 loss（初 → 末） | 恢复率 |
| --- | ---: | ---: | ---: | --- | ---: |
| 无动量 SGD（本样例默认） | 19.157549 | 74.244453 | 31.302790 | 8.0483 → 3.7260 | ≈77.9% |
| AdamW | 19.157549 | 74.244453 | 163033.640625 | 8.0483 → 11.3496 | 发散 |

AdamW 在本样例的 `lr=1e-2` 协议下直接发散：loss 不降反升，微调后 PPL 较剪枝模型恶化约三个
数量级。该对照验证了 8.1 节的优化器选型——无动量 SGD 搭配 `lr=1e-2` 是本样例的合理默认；
AdamW 需要的学习率量级（`1e-5` 级）与本样例的微调协议不兼容。


> 剪枝完成后，如需将模型导出为 torchair 离线模型（.air）用于 ATC 转 om 板端推理，请参考 [AMCT大模型torchair离线模型导出](../export/README.md)。

## 9 社区任务样例：Qwen3-0.6B Dense FFN `reconstruct` + `ratio`（任务 6）

对应 issue [#183](https://gitcode.com/cann/amct/issues/183) 的任务 6：在原始 Qwen3-0.6B 上，分别以 `prune_ratio=0.1/0.2/0.5` 对 Dense FFN 中间维度进行结构化剪枝，使用 `reconstruct` 和默认的最小二乘恢复 `recovery=ls`。脚本：[src/run_qwen3_0_6b_pruning_reconstruct_ratio.py](src/run_qwen3_0_6b_pruning_reconstruct_ratio.py)。

### 9.1 准备与运行

本次实测：Linux aarch64，Python 3.12.13，CANN 9.1.0，torch 2.7.1+cpu，torch_npu 2.7.1.post4，transformers 5.12.1，datasets 4.8.4，huggingface-hub 1.33.0，AMCT 1.1.0。运行设备为单个可见 NPU，`torch.npu.get_device_name(0)` 返回 `Ascend910_9382`，显存约 61.27 GiB。这是已验证的组合，并非所有版本的兼容性声明。

将 [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) 下载到本地目录，确保包含权重、config 和 tokenizer 文件。安装与 Python/torch 匹配的 torch_npu 和 AMCT，加载实际 CANN 安装路径中的 `set_env.sh`。`torch` 版本带 `+cpu` 不表示不能通过 torch_npu 使用 NPU。首次运行需联网下载并缓存数据集。

从 `examples/algorithms/pruning` 目录运行，路径是占位符：

```bash
# 先在实际环境加载 CANN，例如本次使用：
source /usr/local/Ascend/cann-9.1.0/set_env.sh
# 仅在选择从仓库源码运行时设置；使用已安装 AMCT 时跳过此行：
# export PYTHONPATH=../../..

python3 src/run_qwen3_0_6b_pruning_reconstruct_ratio.py \
  --model-path ./path/to/Qwen3-0.6B \
  --prune-ratio 0.1 --calib-samples 32 --calib-seq-len 512 \
  --seq-len 4096 --device npu:0 --save-model \
  --save-dir ./outputs/qwen3_0_6b_reconstruct_ratio/r0_1

python3 src/run_qwen3_0_6b_pruning_reconstruct_ratio.py \
  --model-path ./path/to/Qwen3-0.6B \
  --prune-ratio 0.2 --calib-samples 32 --calib-seq-len 512 \
  --seq-len 4096 --device npu:0 --save-model \
  --save-dir ./outputs/qwen3_0_6b_reconstruct_ratio/r0_2

python3 src/run_qwen3_0_6b_pruning_reconstruct_ratio.py \
  --model-path ./path/to/Qwen3-0.6B \
  --prune-ratio 0.5 --calib-samples 32 --calib-seq-len 512 \
  --seq-len 4096 --device npu:0 --save-model \
  --save-dir ./outputs/qwen3_0_6b_reconstruct_ratio/r0_5
```

每条命令启动新进程并重新加载同一原始模型；不要将上一档剪枝模型作为下一档输入。输出目录必须为空。单卡已足够完成本例；本次只验证了上述设备，未测试 A2/A5。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--model-path` | 必填 | 原始 Qwen3-0.6B 本地模型目录 |
| `--prune-ratio` | 必填 | 仅允许 0.1、0.2、0.5 |
| `--calib-samples` / `--calib-seq-len` | 32 / 512 | Pileval 校准块数量及长度 |
| `--seq-len` | 4096 | 固定 WikiText2 评估长度，其他值会报错 |
| `--eval-batches` | 全量 | 正整数上限，仅用于快速检查；正式实验不传 |
| `--device` | npu:0 | 模型和校准数据所在设备 |
| `--save-dir` | ./outputs/qwen3_0_6b_reconstruct_ratio | 本次结果目录；不允许覆盖已有内容 |
| `--save-model` | 关闭 | 保存到 model/ 并自动重新加载验证 |
| `--skip-diagnose` | 关闭 | 跳过诊断预览，正式报告仍会产生 |

### 9.2 校准、评估与计时口径

复用现有 `src/utils.py` 的模型加载、校准、评估和参数统计接口，不修改共用 helper。Pileval 使用 `mit-han-lab/pile-val-backup` 的 validation，经共享 `get_pileval/pileval_awq` 以 seed=42 打乱、去首尾空白、编码、跳过超过 512 token 的文本行，再拼接取前 32 个完整块，共 16,384 token。校准块与模型均放置到 npu:0。

WikiText2 使用 `Salesforce/wikitext` 的 `wikitext-2-raw-v1` test；共享 `get_wiki_inputs` 用两个换行连接所有行，一次编码后切为不重叠的 4096-token 块，丢弃尾部不足一块的内容。本次得到 73 块，共 299,008 输入 token；共享 PPL helper 使用各块内部的 next-token loss，块间不预测跨边界 token。三档模型输入文件、校准 token 和评估 token 的 SHA256 均一致。

表中剪枝时间为校准数据加载/预处理/搬运时间与实际 `amct.prune()` 时间之和，包含实际剪枝时的激活采集和 LS 求解；NPU 同步后计时。不含模型加载、文件哈希、评估数据准备、基线/剪枝后评估、诊断预览、保存和重载验证。诊断时间单独列出。缓存与设备状态会影响时间；本表不是推理性能测试。没有额外微调或量化，因此后处理 PPL 为 N/A；LS 是 reconstruct 剪枝内部步骤。

早期独立 starter 的 72 块评估口径已废弃，其结果不提交到仓库，不能与本表的 73 块口径混用。

### 9.3 诊断与报告要点

三档诊断均识别 `cnn=0, dense=28, moe=0`，fixed-ratio dry run 的前向检查通过。诊断中的 `acc binary search: unavailable` 表示默认 tolerance 下未找到候选，仅为诊断结果；本任务使用固定 ratio，正式剪枝不依赖该搜索。诊断预览比例与本档实际比例一致。

真实 PruneReport 各含 28 条不同的 Dense reconstruct 事件，全部为 `output-preserving (ls)`；warnings 为空。另有 CNN/MoE 的 `<none>` 事件，因为 Qwen3-0.6B 不含对应目标。三档 28 层 FFN 中间维度分别从 3072 变为 2765、2458、1536；gate/up 输出维度、down 输入维度和 config.intermediate_size 一致。

完整原始记录（诊断输出、`prune_report.json`、`run_metrics.json`、数据/模型/源码哈希、环境与实跑命令）随实验产物保存在各档 `--save-dir` 输出目录中。示例使用的 `outputs/` 目录已被 `.gitignore` 排除，不提交到仓库；可复核的关键字段见本节摘录与 9.4 结果表。

`prune_ratio=0.1` 实测诊断及正式报告摘录（省略其余层和字段，完整记录保存在输出目录）：

```text
[prune-diagnose] prunable targets: cnn=0, dense=28, moe=0
  fixed-ratio prune: available (cut 4.4%, forward OK)
  acc binary search: unavailable
  - acc search found no acceptable prune ratio at this tolerance (relax the tolerance).
```

```json
{
  "backend": "huggingface",
  "params_before": 596049920,
  "params_after": 569643008,
  "warnings": [],
  "events": [
    {
      "stage": "dense",
      "method": "reconstruct",
      "module": "model.layers.0.mlp.down_proj",
      "detail": "output-preserving (ls) prune 3072 -> 2765"
    }
  ]
}
```

### 9.4 全量实测结果

| 模型 | 方法 | 实验设置 | 基线 PPL | 剪枝后 PPL | 后处理 PPL | 参数量（前 → 后） | 参数削减率 | 剪枝时长（min） | 诊断耗时（min） |
| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| Qwen3-0.6B | reconstruct (ls) | prune_ratio=0.1 | 19.157549 | 24.783754 | N/A | 596049920 → 569643008 | 4.4303% | 0.4071 | 1.0841 |
| Qwen3-0.6B | reconstruct (ls) | prune_ratio=0.2 | 19.157549 | 28.937954 | N/A | 596049920 → 543236096 | 8.8606% | 0.4632 | 1.4742 |
| Qwen3-0.6B | reconstruct (ls) | prune_ratio=0.5 | 19.157549 | 59.402245 | N/A | 596049920 → 463929344 | 22.1660% | 0.2663 | 0.9786 |

本表基于已安装的 AMCT 1.1.0。源码运行时，实际导入的 AMCT 及共用 helper 可能随仓库版本变化，应以本次输出 `run_environment.json` 中的 `amct_import_path`、`source_hashes` 和 `helper_sha256` 为准；表中数值不代表任意 master 版本的复测结果。环境记录还会尽可能读取 CANN Toolkit 与 NPU 驱动版本，无法读取时记为 `null`。

三档均已保存并重新加载：逐一检查全部 311 个 state_dict tensor，权重完全相等；参数数、28 层 FFN 形状、config 和 tokenizer 抽样编码一致；首个 4096-token 评估块的重载前后 PPL 在本次实测中完全一致（分别为 21.380943、24.609314、87.247597）；脚本的通过判据为 `rel_tol=1e-4`、`abs_tol=1e-4`。这一步是保存一致性验证，完整评估值以表中 73 块结果为准。

剪枝比例针对 FFN 中间维度，而参数下降针对整个模型，因此 0.5 不等于总参数减少 50%。PPL 越低越好，三档 PPL 均高于原始模型；0.5 档有明显精度损失。本例验证任务要求的流程和实测结果，不据此宣称可直接用于部署，也不与不同校准量/环境的其他任务结果直接比较。

输出目录保存 `run_config.json`、`run_command.txt`、`run_environment.json`、`model_input_sha256.json`、`data_protocol.json`、`baseline_metrics.json`、`prune_config.json`、`diagnose_summary.txt`（未跳过时）、`prune_report.json`、`run_metrics.json`、`run_status.json` 和可选的 `model/`、`reload_validation.json`。权重和数据集不提交到仓库。
