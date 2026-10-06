# Qwen3.5-9B w4a4 autoround 量化样例

对应 Issue [#182](https://gitcode.com/cann/amct/issues/182) 任务 16：对 Qwen3.5-9B 做 `w4a4` + `autoround` 量化，**int4/int4 与 mxfp4/mxfp4 各跑一遍**。
本目录与 `qwen3.6`、`deepseekv4` 并列，不修改其他文档内容。

## 运行说明

环境：单卡 `npu:0`。先按本机 CANN 安装路径配置环境，并安装 `amct_pytorch`（或设置 `PYTHONPATH` 指向仓库根目录）。
一站式平台示例：`source /home/developer/Ascend/cann/set_env.sh`。若需镜像下载校准/评估数据，可自行设置 `HF_ENDPOINT`。

- 模型：本地 Qwen3.5-9B 权重，`--model_name qwen3_5`，`--trust_remote_code`
- 校准集：Pileval（`mit-han-lab/pile-val-backup`，CLI 默认 `nsamples=128`）
- 评估集：WikiText2 `wikitext-2-raw-v1` test，`--seq_len 4096`，`--granularity block`
- BF16 与量化评估使用同一权重目录、同一评估配置
- 量化覆盖 Attention（`attn-linear`）与 MLP（`mlp`）；每种格式都要跑完 extract → PTQ(attn) → PTQ(mlp) → 带 PTQ 参数的量化评估
- 脚本启动时打印完整运行参数；产物目录为相对占位路径，不提交数据/权重

```bash
export HF_ENDPOINT=https://hf-mirror.com
# 下载Qwen3.5-9B模型
hf download Qwen/Qwen3.5-9B --local-dir /path/to/Qwen3.5-9B # 占位，改为本地权重
export MODEL=/path/to/Qwen3.5-9B  # 占位，改为本地权重

cd examples/models/qwen3.5-9B

# 1) BF16 基线
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
| `MODEL` | 本地权重目录 |
| `--model_name qwen3_5` | Qwen3_5 模型类型 |
| `--seq_len 4096` | 校准与评估序列长度 |
| `--granularity block` | `Block` 粒度 |
| `--quant_dtype int \| mxfp` | `int4/int4 或 mxfp4/mxfp4` |
| `--algos autoround` | 量化算法 |
| `--bit_config` | `amct_pytorch/configs/w4a4.yaml` |
| `--quant_target` | PTQ 每次一个目标：`attn-linear` 或 `mlp`，量化评估同时加载两者 |
| `NSAMPLES` | Pileval 校准条数，默认 128 |
| `EPOCHS` | AutoRound 优化迭代轮数，本样例设置为`15` |
| `BASE_LR` | AutoRound 优化的基础学习率，默认 `1e-5` |
| `--start_block_idx` | PTQ 起始 Transformer Block 索引，本样例设置为 `0` |
| `--end_block_idx` | PTQ 结束 Transformer Block 索引，本样例设置为 `32` |

## 产物说明

相对占位目录（默认在本样例目录下）：

| 产物 | 路径 | 大小 |
|:--|:--|:--|
| PTQ 离线数据 Attention | `./outputs/ptq_data/attn-linear` |  129 GiB |
| PTQ 离线数据 MLP | `./outputs/ptq_data/mlp` |  129 GiB |
| 离线数据合计（attn+mlp） | 上述两目录之和 |  258 GiB |
| int PTQ 参数 Attention | `./outputs/int/ptq_params/qwen3_5/attn-linear` |  7965 MiB |
| int PTQ 参数 MLP | `./outputs/int/ptq_params/qwen3_5/mlp` |  18440 MiB |
| mxfp PTQ 参数 Attention | `./outputs/mxfp/ptq_params/qwen3_5/attn-linear` |  8458 MiB |
| mxfp PTQ 参数 MLP | `./outputs/mxfp/ptq_params/qwen3_5/mlp` |  19585 MiB |

量化时长口径：从该格式第一次 `ptq` 开始到 attn+mlp 参数全部写完（不含模型下载和环境构建）。离线数据大小为 ptq_data 输出目录合计， 258 GiB。


## 结果表

BF16 与量化评估使用同一 `MODEL` 与同一 `seq_len=4096` / `granularity=block` 配置。


| 模型         | 数据类型 | 算法        | 量化格式        | BF16 PPL    | 量化 PPL | PPL差值 | 量化时长(min) | 离线数据大小(GiB) |
| ---------- | ---- | --------- | ----------- | ----------- | ------ | ----- | --------- | ----------- |
| Qwen3.5-9B | `w4a4` | `autoround` | `int4/int4`   | **8.209169** | **15138.553711**     | **15130.344542**     | **247.68**         | **258**           |
| Qwen3.5-9B | `w4a4` | `autoround` | `mxfp4/mxfp4` | **8.209169** | **9.210809**      | **1.001640**     | **242.53**         | **258**           |

**注：** 当前`int4` `AutoRound` 算法量化的PPL值特别高，主要由于 `4-bit` 激活量化带来的较大误差，`INT4` 对激活中的离群值较为敏感。当少量较大的激活值决定量化 `scale` 后，大量正常范围内的激活值可能会被压缩到少数几个整数级别，甚至量化为 0，从而造成明显的信息损失。
