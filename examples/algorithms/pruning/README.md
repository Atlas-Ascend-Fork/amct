# AMCT 结构化剪枝样例

演示 `amct_pytorch.pruning` 接口在三种域上的调用：dense FFN（剪中间维）/ CNN（剪通道）/
MoE（剪专家）。样例使用随机初始化的微型模型，纯 CPU 运行，不下载任何权重。

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

## 3 社区任务样例：Qwen3.6-35B-A3B MoE `mass_variance` + `tolerance`

对应 Issue [#183](https://gitcode.com/cann/amct/issues/183) 任务 2：对真实 MoE 模型做专家结构化剪枝。
脚本：[`src/run_qwen3_6_35b_a3b_pruning_mass_variance_tolerance.py`](src/run_qwen3_6_35b_a3b_pruning_mass_variance_tolerance.py)。

### 3.1 运行说明

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
for T in 0.1 0.2 0.5; do
  python3 examples/algorithms/pruning/src/run_qwen3_6_35b_a3b_pruning_mass_variance_tolerance.py \
    --model "$MODEL_DIR" \
    --tolerance "$T" \
    --baseline-ppl <ppl_bf16> \
    --skip-baseline-eval \
    --search-evaluator fidelity \
    --output-dir ./output/qwen3_6_35b_a3b_mass_variance_tolerance
done
```

主要参数：

| 参数 | 含义 |
|:--|:--|
| `--tolerance` | 搜索容差，任务要求分别跑 `0.1` / `0.2` / `0.5` |
| `--search-evaluator fidelity` | 搜索阶段用 AMCT 内置保真度；最终表内 PPL 仍以 WikiText2 blockwise 为准 |
| `--calib-nsamples` / `--calib-seq-len` | Pileval 校准条数与长度（默认 8 / 512） |
| `--eval-seq-len` | 评估序列长度，固定 4096 |
| `--output-dir` | 产物目录（相对/占位路径） |

校准集：`mit-han-lab/pile-val-backup`；评估集：WikiText2 `wikitext-2-raw-v1` test。
保存剪后权重后会把 `config.json` 恢复为 VL 嵌套结构，并更新 `text_config.num_experts`，以便
`qwen3_6_moe` 评估通路加载。

### 3.2 诊断与报告

`prune_diagnose()` 在本样例中识别到 `moe` 目标可用，`prune_works=True`、`search_works=True`。
正式剪枝后 `PruneReport` 记录各层 `mass_variance` 事件（例如 `256 -> 179` / `256 -> 128`），以及
`params_before` / `params_after`。完整 JSON 见各档 `result.json`。

### 3.3 结果表

实测环境：CANNLab A3，`seq_len=4096`，blockwise BF16 WikiText2。搜索 evaluator=`fidelity`。

| 模型 | 方法 | 实验设置 | 基线 PPL | 剪枝后 PPL | 后处理 PPL | 参数量（前 → 后） | 参数削减率 | 剪枝时长（min） |
|:--|:--|:--|--:|--:|:--|:--|--:|--:|
| Qwen3.6-35B-A3B | `mass_variance` | `tolerance=0.1` | 6.308547 | 7.974751 | N/A | 34660610688 → 24965460608 | 0.2797 | 7.29 |
| Qwen3.6-35B-A3B | `mass_variance` | `tolerance=0.2` | 6.308547 | 10.896645 | N/A | 34660610688 → 18543997568 | 0.4650 | 7.15 |
| Qwen3.6-35B-A3B | `mass_variance` | `tolerance=0.5` | 6.308547 | 22.681002 | N/A | 见对应 `result.json` / `num_experts=51` | — | — |

补充：原始每层路由专家数为 256；三档剪后分别为 **179 / 128 / 51**。
