# 环境与可移植性

目标：在一个新环境上跑起来，只改 env 变量与装依赖，不改代码、不写死路径。「新环境」指任何没跑过这套脚本的地方——另一台机器、另一个容器、另一套 CANN / transformers 版本、另一份数据集缓存路径。

## 1. 依赖

- **装什么包、版本对不对由脚本判**：`python <skill>/scripts/check_env.py` 逐个查依赖能否找到、对 lm-eval 校钉住的版本，缺包或版本不对直接 fail-fast 报出来；它不查任务名、不查数据是否预取、不查 NPU 可见性。安装命令在 §6「装依赖」那步。
- **条件依赖**：只有要 LiveCodeBench 官方分数才需要 `lcb_runner`，装法与评分流程见 [livecodebench.md](livecodebench.md)。

## 2. 运行环境限制

- **量化 MoE 跑长 prompt 在 npu:0 上 OOM**：量化专家 forward 有额外中间张量，而 npu:0 上又集中了前半层和 embedding，最先被填满——短输入冒烟能过，长 prompt 就 OOM。这是真实容量不足（非显存碎片），`max_split_size_mb` 无效；改用 `device_map` 的 `max_memory` 把部分层挪离 npu:0，或缩短 `--seq_len`。
- **page cache 与 cgroup OOM**：数据集反复读会把 page cache 涨满，它算进容器内存配额，配额吃紧时进程被 kill（exit 137 / 143，dmesg 里有 cgroup OOM 记录）。不需要特权的处置是缩 --limit 重跑（lm-eval 入口还可缩 --batch_size）；echo 3 > /proc/sys/vm/drop_caches 能立刻释放，但要 root，且影响整机所有租户，动手前先问用户。

## 3. 依赖版本兼容：权重 dtype 参数名随 transformers 版本变化

`from_pretrained` 的权重 dtype 参数在 **transformers 4.56 才从 `torch_dtype` 改名为 `dtype`**。写死新名字在旧版环境上不在调用点报错：该值落进 `**kwargs` → 被写进 config → 延后炸成一个不提参数名的 `TypeError`，很难归因。

本 skill 的脚本按已安装版本选参数名（≥4.56 用 `dtype`，更旧用 `torch_dtype`；4.57+ 只是弃用告警、未删除），不要为消错升降级用户的 transformers。`check_env.py` 会打印当前 transformers 版本与解析出的参数名，环境不符时先看这一行。

## 4. 可配置项（`scripts/config.py` 读取）

| env 变量 | 含义 | 默认值 |
|---|---|---|
| `AMCT_HOME` | amct 源码根目录；默认 `--bit_config` 用它拼出 amct 的默认位宽 yaml，加载权重前先验文件在盘上。推到 pip 安装位置时这份 yaml 不在包里，此时显式传 `--bit_config`，或把 `AMCT_HOME` 指到一份完整 checkout | 三级推导：① 从脚本位置向上查找同时含 `amct_pytorch/` 与 `setup.py` 的目录；② 找不到则取可导入的 `amct_pytorch` 包所在目录（`PYTHONPATH` 含仓库根、或 pip 装过，都能命中）；③ 两级都空才 fail-fast 报错，不静默用错路径 |
| `BENCH_ASSETS` | 本地数据集缓存根目录 | `$XDG_CACHE_HOME/amct-bench-assets`，无则 `~/.cache/amct-bench-assets` |

模型与量化参数始终经 CLI 传入（`--model`、`--model_name`、`--param_dir`），不进配置文件。

## 5. 环境变量模板

```bash
source <ascend-toolkit>/set_env.sh                       # CANN 环境
export ASCEND_RT_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1     # 多卡填 §6 第 1 步打出的卡号列表
export AMCT_HOME=<amct_repo>                              # 省略则按 §4 的三级推导
export PYTHONPATH=$AMCT_HOME:$PYTHONPATH                  # 必须前置；整个覆盖会丢 CANN 塞进去的 python/site-packages 与 op_impl/ai_core/tbe，大模型在 aclop 编译阶段崩溃（症状见 SKILL.md §5）
export BENCH_ASSETS=<dataset_cache_dir>                   # 省略则落 ~/.cache/amct-bench-assets
export HF_ENDPOINT=https://hf-mirror.com                  # 官方端点不通时自己指到镜像
```

`HF_DATASETS_OFFLINE` / `HF_HUB_OFFLINE` 不用设，设不设都不改变评测结果：注册任务的读取走本地文件、到不了网络，未注册任务在加载模型之前就被拦住（constraints.md §2）。

`HF_ENDPOINT` 不是本 skill 的配置项：`huggingface_hub` 自己读它，`prefetch_data.py` 不转手、不设默认、不诊断连通性。

## 6. 在新环境上的部署步骤

1. 定卡数：`python <skill>/scripts/plan_device.py --model <model_path>`（只读 `config.json`，不加载权重、不占卡）。它打出本次要传的 `--device_map` 与 `ASCEND_RT_VISIBLE_DEVICES` 卡号列表，卡数 = 权重估算 ÷ `utils.py` 的 `SINGLE_CARD_WEIGHT_BUDGET`（每卡承担的权重预算，已给激活、无 KV cache 的全上下文重算、amct 量化中间张量留了余量，不是每卡显存本身）。本机每卡可用显存低于这个预算时，用 `npu-smi info` 查出每卡可用显存后按比例多开卡，以冒烟实测为准。按打出的卡号填 `ASCEND_RT_VISIBLE_DEVICES`，评测时同样传 `--device_map`。
2. 设好上表 env 变量（多数情况下 `AMCT_HOME` / `BENCH_ASSETS` 可不设，用默认值）。
3. 装依赖：`pip install --user fastparquet lm-eval`（不带版本号装到的是最新版）。lm-eval 有钉住版本，以第 4 步 `check_env.py` 的判定为准：缺包或版本不符时，它把期望版本和可直接复制的 `pip install` 命令一起打出来，照那行重装。要 LiveCodeBench 官方评分，从官方仓装（`git clone https://github.com/LiveCodeBench/LiveCodeBench.git` 后在仓内 `pip install -e .`），PyPI 上没有这个包。
4. 依赖检查：`python <skill>/scripts/check_env.py`（PYTHONPATH 需含仓库根）；要官方评分再加 `--livecodebench_score`，只跑 LiveCodeBench 生成、环境里不装 lm-eval 时加 `--skip_lm_eval`。
5. 预下载数据：`python <skill>/scripts/prefetch_data.py --task <task>`，缓存落 `$BENCH_ASSETS/<subdir>/`。
