# Qwen3-0.6B w4a8 `lwc` + `lac` 量化样例

对应 Issue [#182](https://gitcode.com/cann/amct/issues/182) 任务 1：对 Qwen3-0.6B 做 `w4a8` + `lwc`/`lac` 量化，**int4/int8 与 mxfp4/mxfp8 各跑一遍**。
本目录与 `qwen3.6`、`deepseekv4` 并列，不修改 Qwen3.6 文档。

CLI 入口与 [Qwen3.6-Moe.md](../qwen3.6/Qwen3.6-Moe.md) 一致：`python3 -m amct_pytorch.eval` / `extract_ptq_data` / `ptq`。
本机部分镜像的 `torchaudio` 在导入 `transformers` 时会崩溃；样例脚本通过 `scripts/amct_cli.py` 屏蔽该依赖后再调用同一 CLI。若官方 `python3 -m` 可直接运行，也可不用该包装。

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
export MODEL=/path/to/Qwen3-0.6B   # 占位，改为本地权重

cd examples/models/qwen3

# 1) BF16 基线（两种格式共用）
bash scripts/eval_bf16.sh

# 2) 提取 PTQ 离线数据（attn + mlp）
bash scripts/extract_ptq_data.sh

# 3) int4/int8
QUANT_DTYPE=int bash scripts/ptq_attn.sh
QUANT_DTYPE=int bash scripts/ptq_mlp.sh
QUANT_DTYPE=int bash scripts/eval_quant.sh

# 4) mxfp4/mxfp8
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
| `--quant_dtype int\|mxfp` | int4/int8 或 mxfp4/mxfp8 |
| `--algos lwc lac` | 量化算法 |
| `--bit_config` | `configs/w4a8_int_lwc_lac.yaml` 或 `configs/w4a8_mxfp_lwc_lac.yaml`（由 `w4a8.yaml` 派生） |
| `--quant_target` | extract/PTQ 每次一个目标：`attn-linear` 或 `mlp`；量化评估同时加载两者 |
| `NSAMPLES` | Pileval 校准条数，默认 128 |
| `END_BLOCK_IDX` | Qwen3-0.6B 默认 28 层 |

## 产物说明

相对占位目录（默认在本样例目录下）：

| 产物 | 路径 | 大小 |
|:--|:--|:--|
| PTQ 离线数据 Attention | `./outputs/qwen3_0_6b/ptq_data/attn-linear` | 28.002 GiB |
| PTQ 离线数据 MLP | `./outputs/qwen3_0_6b/ptq_data/mlp` | 28.002 GiB |
| 离线数据合计（attn+mlp） | 上述两目录之和 | 56.004 GiB |
| int PTQ 参数 Attention | `./outputs/qwen3_0_6b_w4a8_int/ptq_params/qwen3/attn-linear` | 1.30 MiB |
| int PTQ 参数 MLP | `./outputs/qwen3_0_6b_w4a8_int/ptq_params/qwen3/mlp` | 1.67 MiB |
| mxfp PTQ 参数 Attention | `./outputs/qwen3_0_6b_w4a8_mxfp/ptq_params/qwen3/attn-linear` | 42.20 MiB |
| mxfp PTQ 参数 MLP | `./outputs/qwen3_0_6b_w4a8_mxfp/ptq_params/qwen3/mlp` | 63.14 MiB |

量化时长口径：从该格式第一次 `ptq` 开始到 attn+mlp 参数全部写完（不含模型下载和环境构建）。离线数据大小为 extract 输出目录合计，GiB。

## 结果表

BF16 与量化评估使用同一 `MODEL` 与同一 `seq_len=4096` / `granularity=block` 配置。

| 模型 | 数据类型 | 算法 | 量化格式 | BF16 PPL | 量化 PPL | PPL 差值 | 量化时长（min） | 离线数据大小（GiB） |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | `w4a8` | `lwc` + `lac` | `int4/int8` | 19.157549 | 47.486488 | 28.328939 | 37.35 | 56.004 |
| Qwen3-0.6B | `w4a8` | `lwc` + `lac` | `mxfp4/mxfp8` | 19.157549 | 23.898159 | 4.740610 | 48.88 | 56.004 |
