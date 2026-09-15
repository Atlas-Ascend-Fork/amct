# Qwen3-0.6B w4a4 autoround 量化样例

对应 Issue [#182](https://gitcode.com/cann/amct/issues/182) 任务 8：对 Qwen3-0.6B 做 `w4a4` + `autoround` 量化，**int4/int4 与 mxfp4/mxfp4 各跑一遍**。
本目录与 `qwen3.6`、`deepseekv4` 并列，不修改 Qwen3.6 文档。


## 运行说明

环境：单卡 `npu:0`。先 `source /home/developer/Ascend/cann/set_env.sh`，并安装 `amct_pytorch`（或设置 `PYTHONPATH` 指向仓库根目录）。

- 模型：本地 Qwen3-0.6B 权重，`--model_name qwen3`，`--trust_remote_code`
- 校准集：Pileval（`mit-han-lab/pile-val-backup`，CLI 默认 `nsamples=128`）
- 评估集：WikiText2 `wikitext-2-raw-v1` test，`--seq_len 4096`，`--granularity block`
- BF16 与量化评估使用同一权重目录、同一评估配置
- 量化覆盖 Attention（`attn-linear`）与 MLP；每种格式都要跑完 extract → PTQ(attn) → PTQ(mlp) → 带 PTQ 参数的量化评估
- 脚本启动时打印完整运行参数；产物目录为相对占位路径，不提交数据/权重

```bash
# 仓库根目录
source /home/developer/Ascend/cann/set_env.sh
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
# 下载Qwen3-0.6B模型
hf download Qwen/Qwen3-0.6B --local-dir /workspace/models/Qwen3-0.6B
export MODEL=/path/to/Qwen3-0.6B   # 占位，改为本地权重

cd examples/models/qwen3-0.6B

# 1) BF16 基线（两种格式共用）
bash scripts/eval_bf16.sh

# 2) 提取 PTQ 离线数据（attn + mlp）
bash scripts/extract_ptq_data.sh

# 3) int4/int4
QUANT_DTYPE=int bash scripts/ptq_attn.sh
QUANT_DTYPE=int bash scripts/ptq_mlp.sh
QUANT_DTYPE=int bash scripts/eval_quant.sh

# 4) mxfp4/mxfp4
QUANT_DTYPE=mxfp bash scripts/ptq_attn.sh
QUANT_DTYPE=mxfp bash scripts/ptq_mlp.sh
QUANT_DTYPE=mxfp bash scripts/eval_quant.sh
```

主要参数：

| 参数 / 环境变量 | 含义 |
|:--|:--|
| `MODEL` | 本地权重目录（占位路径） |
| `--model_name qwen3` | Dense Qwen3 通路 |
| `--seq_len 4096` | 校准与评估序列长度 |
| `--granularity block` | Block 粒度 |
| `--quant_dtype int \| mxfp` | int4/int4 或 mxfp4/mxfp4 |
| `--algos autoround` | 量化算法 |
| `--bit_config` | `configs/w4a4_int_autoround.yaml` 或 `configs/w4a4_mxfp_autoround.yaml`（由 `w4a4.yaml` 派生） |
| `--quant_target` | PTQ 每次一个目标：`attn-linear` 或 `mlp`；量化评估同时加载两者 |
| `NSAMPLES` | Pileval 校准条数，默认 128 |

## 产物说明

相对占位目录（默认在本样例目录下）：

| 产物 | 路径 | 大小 |
|:--|:--|:--|
| PTQ 离线数据 Attention | `./outputs/ptq_data/attn-linear` | 28.675 GiB |
| PTQ 离线数据 MLP | `./outputs/ptq_data/mlp` | 28.675 GiB |
| 离线数据合计（attn+mlp） | 上述两目录之和 | 57.349 GiB |
| int PTQ 参数 Attention | `./outputs/int/ptq_params/qwen3/attn-linear` | 674 MiB |
| int PTQ 参数 MLP | `./outputs/int/ptq_params/qwen3/mlp` | 1010 MiB |
| mxfp PTQ 参数 Attention | `./outputs/mxfp/ptq_params/qwen3/attn-linear` | 715 MiB |
| mxfp PTQ 参数 MLP | `./outputs/mxfp/ptq_params/qwen3/mlp` | 1010 MiB |

量化时长口径：从该格式第一次 `ptq` 开始到 attn+mlp 参数全部写完（不含模型下载和环境构建）。离线数据大小为 ptq_data 输出目录合计，57.349 GiB。


## 结果表

BF16 与量化评估使用同一 `MODEL` 与同一 `seq_len=4096` / `granularity=block` 配置。


| 模型         | 数据类型 | 算法        | 量化格式        | BF16 PPL    | 量化 PPL | PPL差值 | 量化时长(min) | 离线数据大小(GiB) |
| ---------- | ---- | --------- | ----------- | ----------- | ------ | ----- | --------- | ----------- |
| Qwen3-0.6B | `w4a4` | `autoround` | `int4/int4`   | **19.157549** | **934986.312500**     | **934,967.154951**     | **87.68**         | **57.349**           |
| Qwen3-0.6B | `w4a4` | `autoround` | `mxfp4/mxfp4` | **19.157549** | **31.795736**      | **12.638187**     | **88.51**         | **57.349**           |
