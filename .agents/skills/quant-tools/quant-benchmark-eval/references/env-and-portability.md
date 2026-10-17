# 环境与可移植性

目标：换一台 Ascend + amct 机器时，只改 env 变量与装依赖，不改代码、不写死路径。

## 依赖边界

- **硬依赖（无法去除）**：amct（装配依赖 `MODEL_REGISTRY`、`build_quant_block`、`load_selected_layer_ptq_params` 与 ptq 参数格式）、`torch_npu` 与 Ascend NPU（代码使用 `npu:*` 设备）、lm-eval 0.4.12。因此无法原样跑在 GPU / CPU 上，也无法脱离 amct。
- **LiveCodeBench 官方分数额外依赖** `lcb_runner`：本 skill 只在 Ascend 环境里生成官方 `custom_evaluator` 所需 JSON（`question_id` + `code_list`），hidden-tests pass@1 由 `python -m lcb_runner.runner.custom_evaluator` 计算（详见 [livecodebench.md](livecodebench.md)）。
- **版本基线（原验证机器）**：lm-eval 0.4.12、torch 2.7.1+cpu、torch_npu、transformers、Python 3.12；2× Ascend 910（每卡 64GB HBM）。8B 单卡可放下；35B 需 `device_map=auto`（双卡，不允许 offload）。

## 可配置项（`scripts/config.py` 读取）

| env 变量 | 含义 | 默认值 |
|---|---|---|
| `AMCT_HOME` | amct 源码根目录（用于默认 `--bit_config`） | 从脚本位置向上查找同时含 `amct_pytorch/` 与 `setup.py` 的目录；找不到即 fail-fast 报错，不静默用错路径 |
| `BENCH_ASSETS` | 本地数据集缓存根目录 | `$XDG_CACHE_HOME/amct-bench-assets`，无则 `~/.cache/amct-bench-assets` |
| `HF_ENDPOINT` | prefetch 用的 HF 镜像 | `https://hf-mirror.com` |

模型与量化参数始终经 CLI 传入（`--model`、`--model_name`、`--param_dir`），不进配置文件。

数据集注册表数据驱动，集中在 `scripts/datasets.json`（repo_id、subdir、split → parquet/jsonl/glob），prefetch 与离线 shim 共用同一份。低资源任务与 MMLU-Pro / MATH500 走 lm-eval；LiveCodeBench 走本 skill 的 `scripts/livecodebench_eval.py` + 官方 `lcb_runner` 评分。

## 环境变量模板

```bash
source <ascend-toolkit>/set_env.sh                       # CANN 环境
export ASCEND_RT_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1     # 35B 多卡用 0,1
export AMCT_HOME=<amct_repo>                              # 省略则自动从脚本位置推导
export PYTHONPATH=$AMCT_HOME:$PYTHONPATH                  # 必须前置（见 constraints.md #4）
export BENCH_ASSETS=<dataset_cache_dir>                   # 省略则落 ~/.cache/amct-bench-assets
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_ENDPOINT=https://hf-mirror.com
```

安装：`pip install --user lm-eval==0.4.12 fastparquet`；要 LiveCodeBench 官方评分再 `pip install --user livecodebench`（或按官方仓 README 安装）。

## 部署到新机器的步骤

1. 设好上表 env 变量（多数情况下 `AMCT_HOME` / `BENCH_ASSETS` 可不设，用默认值）。
2. `pip install lm-eval==0.4.12`（+ LiveCodeBench 评分所需的 `lcb_runner`）。
3. 依赖检查：`python scripts/check_env.py`（PYTHONPATH 需含仓库根）；要官方评分再加 `--livecodebench_score`。
4. 轻量自检（不加载模型、不占 NPU）：`python scripts/self_check.py`。
5. 预下载数据：`python scripts/prefetch_data.py --task <task>`，缓存落 `$BENCH_ASSETS/<subdir>/`。
