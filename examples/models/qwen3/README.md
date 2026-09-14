# Qwen3-0.6B w4a8 `lwc` + `lac` 量化样例

对应 Issue [#182](https://gitcode.com/cann/amct/issues/182) 任务 1：对 Qwen3-0.6B 做 `w4a8` + `lwc`/`lac` 量化，**int4/int8 与 mxfp4/mxfp8 各跑一遍**。
本目录与 `qwen3.6`、`deepseekv4` 并列，不修改 Qwen3.6 文档。

CLI 入口与 [Qwen3.6-Moe.md](../qwen3.6/Qwen3.6-Moe.md) 一致：`python3 -m amct_pytorch.eval` / `extract_ptq_data` / `ptq`。
量化配置直接使用仓内 [`amct_pytorch/configs/w4a8.yaml`](../../../amct_pytorch/configs/w4a8.yaml)，int / mxfp 由 `--quant_dtype` 区分，不再另拷一份。

## 运行说明

环境：单卡 `npu:0`。先按本机 CANN 安装路径配置环境，并安装 `amct_pytorch`（或设置 `PYTHONPATH` 指向仓库根目录）。
一站式平台示例：`source /home/developer/Ascend/cann/set_env.sh`。若需镜像下载校准/评估数据，可自行设置 `HF_ENDPOINT`。
部分镜像的 `torchaudio` 会在导入 `transformers` 时崩溃，属本机依赖问题，修复后再使用官方 `python3 -m` 入口即可，样例不提供环境包装脚本。

- 模型：本地 Qwen3-0.6B 权重，`--model_name qwen3`，`--trust_remote_code`
- 校准集：Pileval（`mit-han-lab/pile-val-backup`，CLI 默认 `nsamples=128`）
- 评估集：WikiText2 `wikitext-2-raw-v1` test，`--seq_len 4096`，`--granularity block`
- BF16 与量化评估使用同一权重目录、同一评估配置
- 量化覆盖 Attention（`attn-linear`）与 MLP；每种格式都要跑完 extract → PTQ(attn) → PTQ(mlp) → 带 PTQ 参数的量化评估
- 产物目录为相对占位路径，不提交数据/权重

在仓库根目录执行：

```bash
source /path/to/cann/set_env.sh
export MODEL=/path/to/Qwen3-0.6B   # 占位，改为本地权重

# 1) BF16 基线（两种格式共用）
python3 -m amct_pytorch.eval \
  --trust_remote_code \
  --model "${MODEL}" \
  --model_name qwen3 \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --eval_mode bf16 \
  --bit_config amct_pytorch/configs/bf16.yaml

# 2) 提取 PTQ 离线数据（attn + mlp）
python3 -m amct_pytorch.extract_ptq_data \
  --trust_remote_code \
  --model "${MODEL}" \
  --model_name qwen3 \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --quant_target attn-linear \
  --nsamples 128 \
  --data_dir examples/models/qwen3/outputs/qwen3_0_6b/ptq_data/attn-linear

python3 -m amct_pytorch.extract_ptq_data \
  --trust_remote_code \
  --model "${MODEL}" \
  --model_name qwen3 \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --quant_target mlp \
  --nsamples 128 \
  --data_dir examples/models/qwen3/outputs/qwen3_0_6b/ptq_data/mlp

# 3) int4/int8
python3 -m amct_pytorch.ptq \
  --trust_remote_code \
  --model "${MODEL}" \
  --model_name qwen3 \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --data_dir examples/models/qwen3/outputs/qwen3_0_6b/ptq_data/attn-linear \
  --quant_dtype int \
  --algos lwc lac \
  --bit_config amct_pytorch/configs/w4a8.yaml \
  --quant_target attn-linear \
  --start_block_idx 0 \
  --end_block_idx 28 \
  --epochs 15 \
  --base_lr 1e-5 \
  --output_dir examples/models/qwen3/outputs/qwen3_0_6b_w4a8_int

python3 -m amct_pytorch.ptq \
  --trust_remote_code \
  --model "${MODEL}" \
  --model_name qwen3 \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --data_dir examples/models/qwen3/outputs/qwen3_0_6b/ptq_data/mlp \
  --quant_dtype int \
  --algos lwc lac \
  --bit_config amct_pytorch/configs/w4a8.yaml \
  --quant_target mlp \
  --start_block_idx 0 \
  --end_block_idx 28 \
  --epochs 15 \
  --base_lr 1e-5 \
  --output_dir examples/models/qwen3/outputs/qwen3_0_6b_w4a8_int

python3 -m amct_pytorch.eval \
  --trust_remote_code \
  --model "${MODEL}" \
  --model_name qwen3 \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --eval_mode quant \
  --quant_target attn-linear mlp \
  --quant_dtype int \
  --algos lwc lac \
  --bit_config amct_pytorch/configs/w4a8.yaml \
  --attn_linear_param_dir examples/models/qwen3/outputs/qwen3_0_6b_w4a8_int/ptq_params/qwen3/attn-linear \
  --moe_mlp_param_dir examples/models/qwen3/outputs/qwen3_0_6b_w4a8_int/ptq_params/qwen3/mlp

# 4) mxfp4/mxfp8：将上一步 --quant_dtype int 改为 mxfp，
#    --output_dir / 参数目录改为 outputs/qwen3_0_6b_w4a8_mxfp 后各跑一遍。
```

也可在设置 `MODEL` 后调用 `scripts/*.sh`；脚本启动时打印完整运行参数，`--bit_config` 同样指向仓内 `w4a8.yaml`。

主要参数：

| 参数 / 环境变量 | 含义 |
|:--|:--|
| `MODEL` | 本地权重目录（占位路径） |
| `--model_name qwen3` | Dense Qwen3 通路 |
| `--seq_len 4096` | 校准与评估序列长度 |
| `--granularity block` | Block 粒度 |
| `--quant_dtype int\|mxfp` | int4/int8 或 mxfp4/mxfp8 |
| `--algos lwc lac` | 量化算法 |
| `--bit_config` | 仓内 `amct_pytorch/configs/w4a8.yaml` |
| `--quant_target` | extract/PTQ 每次一个目标：`attn-linear` 或 `mlp`；量化评估同时加载两者 |
| `--nsamples` | Pileval 校准条数，默认 128 |
| `--end_block_idx` | Qwen3-0.6B 默认 28 层 |

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
