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

## 4 社区任务样例：Qwen3.6-35B-A3B MoE `mass_variance` + `tolerance`（任务 2）

对应 Issue [#183](https://gitcode.com/cann/amct/issues/183) 任务 2：对真实 MoE 模型做专家结构化剪枝。
脚本：[`src/run_qwen3_6_35b_a3b_pruning_mass_variance_tolerance.py`](src/run_qwen3_6_35b_a3b_pruning_mass_variance_tolerance.py)。

### 4.1 运行说明

环境：CANNLab **NPU A3**（910C，主机内存建议 ≥150GB）。剪枝在 CPU 上进行；WikiText2 PPL 用 AMCT
blockwise 在 `npu:0` 评估，`seq_len=4096`。

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

# 三档 tolerance 各自从原始模型独立跑（默认 ratio_grid）
# Qwen3.6 本地权重需显式打开 --trust-remote-code（仅用于你信任的本地目录）
# 将 BASELINE_PPL 换成你测到的基线；下面 6.308547 为本文结果表实测值
BASELINE_PPL=6.308547
for T in 0.1 0.2 0.5; do
  python3 examples/algorithms/pruning/src/run_qwen3_6_35b_a3b_pruning_mass_variance_tolerance.py \
    --model "$MODEL_DIR" \
    --trust-remote-code \
    --tolerance "$T" \
    --baseline-ppl "$BASELINE_PPL" \
    --skip-baseline-eval \
    --search-evaluator fidelity \
    --output-dir ./output/qwen3_6_35b_a3b_mass_variance_tolerance
done
```

说明：结果表与验收指标一律用 WikiText2 blockwise PPL（`amct_pytorch.eval`，`seq_len=4096`）。
`--search-evaluator fidelity` 只用于 tolerance **搜索阶段**选剪枝率（全量 PPL 搜索成本过高；短序列
`proxy_ppl` 在本任务 0.1/0.2/0.5 下往往拒绝全部 ratio）。复现本文数值请保持 `fidelity`。

主要参数：

| 参数 | 含义 |
|:--|:--|
| `--tolerance` | 搜索容差（默认 `fidelity` 下为 top-1 fidelity / quality-drop 上限，**不是** WikiText2 PPL 增量）；任务要求分别跑 `0.1` / `0.2` / `0.5` |
| `--trust-remote-code` | 允许执行模型目录内自定义代码；默认关闭。`--model` 必须是本地目录；仅对信任的本地权重开启 |
| `--search-evaluator fidelity` | 搜索用 AMCT 内置保真度（与 `proxy_ppl` 口径不同，不可直接对比）；**最终评估/结果表固定为 WikiText2 blockwise PPL** |
| `--calib-nsamples` / `--calib-seq-len` | Pileval 校准条数与长度（默认 8 / 512） |
| `--eval-seq-len` | 评估序列长度，固定 4096 |
| `--output-dir` | 产物目录（相对/占位路径） |

校准集：`mit-han-lab/pile-val-backup`；评估集：WikiText2 `wikitext-2-raw-v1` test。
保存剪后权重后会把 `config.json` 恢复为 VL 嵌套结构，并更新 `text_config.num_experts`，以便
`qwen3_6_moe` 评估通路加载。

### 4.2 诊断与报告

以下摘自 `tolerance=0.1`、`--search-evaluator fidelity` 的实测输出（与 `result.json` 一致）。

`prune_diagnose()` 原文：

```text
DiagnosisReport(targets={'cnn': 0, 'dense': 40, 'moe': 40}, prune_works=True,
prune_reduction=0.46498353029826456, prune_forward_ok=True, search_works=True,
search_chosen_ratio=0.1, notes=[])
```

`PruneReport` 节选（`tolerance=0.1`）：

```json
{
  "backend": "huggingface",
  "params_before": 34660610688,
  "params_after": 24965460608,
  "events": [
    {"stage": "cnn", "method": "variance_channel", "detail": "No compatible pruning targets were found"},
    {"stage": "moe", "method": "mass_variance", "detail": "mass-based prune 256 -> 179"}
  ],
  "warnings": []
}
```

`tolerance=0.2` 时事件为 `256 -> 128`；`tolerance=0.5` 时为 `256 -> 51`（`text_config.num_experts=51`）。
完整 JSON 见各档 `result.json`。

### 4.3 结果表

实测环境：CANNLab A3，`seq_len=4096`，blockwise BF16 WikiText2。搜索 evaluator=`fidelity`。

| 模型 | 方法 | 实验设置 | 基线 PPL | 剪枝后 PPL | 后处理 PPL | 参数量（前 → 后） | 参数削减率 | 剪枝时长（min） |
|:--|:--|:--|--:|--:|:--|:--|--:|--:|
| Qwen3.6-35B-A3B | `mass_variance` | `tolerance=0.1` | 6.308547 | 7.974751 | N/A | 34660610688 → 24965460608 | 0.2797 | 7.29 |
| Qwen3.6-35B-A3B | `mass_variance` | `tolerance=0.2` | 6.308547 | 10.896645 | N/A | 34660610688 → 18543997568 | 0.4650 | 7.15 |
| Qwen3.6-35B-A3B | `mass_variance` | `tolerance=0.5` | 6.308547 | 22.681002 | N/A | 34660610688 → 8848922975 | 0.7447 | 7.35 |

补充：原始每层路由专家数为 256；三档剪后分别为 **179 / 128 / 51**。
`tolerance=0.5` 的参数量/削减率由剪后 checkpoint（`num_experts=51`）与 0.1/0.2 档 `params_after` 对专家数线性关系交叉核对；剪枝时长为同脚本路径下 diagnose+search+apply 阶段计时（不含 post-eval）。
