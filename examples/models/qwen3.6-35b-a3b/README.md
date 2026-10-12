# Qwen3.6-35B-A3B w4a8 `autoround` 量化样例

对应 Issue [#182](https://gitcode.com/cann/amct/issues/182) **任务 20**：对 Qwen3.6-35B-A3B 做 `w4a8` + `autoround`，**int4/int8 与 mxfp4/mxfp8 各跑一遍**。

CLI 入口与 [Qwen3.6-Moe.md](../qwen3.6/Qwen3.6-Moe.md) 一致：`python3 -m amct_pytorch.eval` / `extract_ptq_data` / `ptq`。
位宽配置复用仓内 [`amct_pytorch/configs/w4a8.yaml`](../../../amct_pytorch/configs/w4a8.yaml)；`int` / `mxfp` 由 CLI `--quant_dtype` 选择。

| 格式 | 配置文件 | CLI `--quant_dtype` |
| --- | --- | --- |
| int4/int8 | `amct_pytorch/configs/w4a8.yaml` | `int` |
| mxfp4/mxfp8 | `amct_pytorch/configs/w4a8.yaml` | `mxfp` |

`w4a8.yaml` 的 MoE 策略：routed 专家 w4/a8，shared expert w8/a8。

**量化范围覆盖 Attention 与 MoE**：`attn-linear` 与 `moe` 两个目标的 PTQ 默认都实际执行（`scripts/ptq_mlp.sh` 默认 `RUN_MOE_PTQ=1`），量化评估同时加载两者的 PTQ 参数。

## 运行说明

环境：单卡 `npu:0` 起步。默认协议（`MOE_NSAMPLES=128` / `MOE_EPOCHS=10`）下 MoE PTQ 每格式约需数天；本样例结果表改用 `MOE_NSAMPLES=32` / `MOE_EPOCHS=1`，单卡实测 int 全流程 243.5 min、mxfp 全流程 303.2 min（见结果表）。需按默认协议跑或进一步提速时，可用 [`examples/ptq_multi_npu.sh`](../../../examples/ptq_multi_npu.sh) 的按层切分方式并行。先按本机 CANN 安装路径配置环境，并安装 `amct_pytorch`（或设置 `PYTHONPATH` 指向仓库根目录）。
一站式平台示例：`source /home/developer/Ascend/cann/set_env.sh`。若需镜像下载校准/评估数据，可自行设置 `HF_ENDPOINT`。

- 模型：本地 Qwen3.6-35B-A3B 权重；`--model_name qwen3_6_moe`；`--trust_remote_code`
- 校准集：Pileval（`mit-han-lab/pile-val-backup`），CLI 默认 `nsamples=128`
- 评估集：WikiText2 `wikitext-2-raw-v1` test；`--seq_len 4096`；`--granularity block`
- BF16 与量化评估使用同一权重目录、同一评估配置
- 每种格式：extract → PTQ(attn) → PTQ(moe) → 带 PTQ 参数的量化评估
- 产物目录为相对占位路径，不提交数据/权重

在仓库根目录执行：

```bash
source /path/to/cann/set_env.sh
export MODEL=/path/to/Qwen3.6-35B-A3B   # 占位，改为本地权重

# 1) BF16 基线（两种格式共用）
bash examples/models/qwen3.6-35b-a3b/scripts/eval_bf16.sh

# 2) 提取 PTQ 离线数据（attn + moe，两种格式共用，跑一次即可）
# 结果表口径：MoE 侧覆盖为 32/1/8（Attention 保持默认 128/10/4）。
# 不设下面三项时 MoE 会按默认 128 条提取，数据量约为表中口径的 4 倍，也复现不出表中 PPL。
export MOE_NSAMPLES=32 MOE_EPOCHS=1 MOE_CALI_BSZ=8
bash examples/models/qwen3.6-35b-a3b/scripts/extract_ptq_data.sh

# 3) int4/int8：PTQ(attn) → PTQ(moe) → 量化评估
export QUANT_DTYPE=int
bash examples/models/qwen3.6-35b-a3b/scripts/ptq_attn.sh
bash examples/models/qwen3.6-35b-a3b/scripts/ptq_mlp.sh
bash examples/models/qwen3.6-35b-a3b/scripts/eval_quant.sh

# 4) mxfp4/mxfp8：换 QUANT_DTYPE 重跑 PTQ(attn) → PTQ(moe) → 量化评估
export QUANT_DTYPE=mxfp
bash examples/models/qwen3.6-35b-a3b/scripts/ptq_attn.sh
bash examples/models/qwen3.6-35b-a3b/scripts/ptq_mlp.sh
bash examples/models/qwen3.6-35b-a3b/scripts/eval_quant.sh
```

也可直接使用 `python3 -m ...` 等价命令（以 int 为例）：

```bash
python3 -m amct_pytorch.ptq \
  --trust_remote_code \
  --model "${MODEL}" \
  --model_name qwen3_6_moe \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --data_dir examples/models/qwen3.6-35b-a3b/outputs/qwen3_6_35b_a3b/ptq_data/attn-linear \
  --quant_dtype int \
  --algos autoround \
  --bit_config amct_pytorch/configs/w4a8.yaml \
  --quant_target attn-linear \
  --start_block_idx 0 \
  --end_block_idx 40 \
  --epochs 10 \
  --base_lr 1e-3 \
  --output_dir examples/models/qwen3.6-35b-a3b/outputs/qwen3_6_35b_a3b_w4a8_int
```

MoE 目标在 Attention 命令基础上改 `--quant_target moe`、`--data_dir .../ptq_data/moe`，并把 `--nsamples` / `--cali_bsz` / `--epochs` 换成 MoE 取值（本样例 `32` / `8` / `1`，`--base_lr` 不变仍为 `1e-3`，即脚本里的 `MOE_NSAMPLES` / `MOE_CALI_BSZ` / `MOE_EPOCHS`）；量化评估同时传 `--attn_linear_param_dir` 与 `--moe_mlp_param_dir`（`--quant_target attn-linear moe`）。直接照 Attention 命令只换 `--quant_target` / `--data_dir` 的话，`--nsamples` / `--cali_bsz` 会落到 CLI 默认 `128 / 4`，`--epochs` 仍为命令中的 `10`，与结果表口径不一致。

主要参数：

| 参数 / 环境变量 | 含义 |
|:--|:--|
| `MODEL` | 本地权重目录（占位路径） |
| `--model_name qwen3_6_moe` | Qwen3.6 MoE 通路 |
| `--seq_len 4096` | 校准与评估序列长度 |
| `--granularity block` | Block 粒度 |
| `--quant_dtype int\|mxfp` | int4/int8 或 mxfp4/mxfp8 |
| `--algos autoround` | 量化算法 |
| `--bit_config` | `amct_pytorch/configs/w4a8.yaml` |
| `--quant_target` | extract/PTQ 每次一个目标：`attn-linear` 或 `moe`；量化评估同时加载两者 |
| `--nsamples` / `--cali_bsz` | Pileval 校准条数（CLI 默认 128）/ 校准 batch size（CLI 默认 4） |
| `--epochs` / `--base_lr` | 本样例取值 `10` / `1e-3`（CLI 默认 `15` / `1e-5`）；可调 |
| `MOE_NSAMPLES` / `MOE_EPOCHS` / `MOE_BASE_LR` / `MOE_CALI_BSZ` | MoE 侧单独覆盖校准条数 / 迭代轮数 / 学习率 / 校准 batch size（默认与 Attention 相同）；专家级 autoround 单元数约为 `40×256`，预算紧张时可下调以控制卡时，结果表按实际取值填写。注意 `MOE_NSAMPLES`、`MOE_EPOCHS`、`MOE_CALI_BSZ` 均需不小于 1（当前 factory 的余弦调度 `T_max = epochs × ⌈nsamples / cali_bsz⌉`，三者非零即合法，脚本会校验并报错） |
| `--end_block_idx` | 本样例取值 40（exclusive；Qwen3.6-35B-A3B 共 40 层，覆盖全部层）；CLI 默认 61，大于层数时会被 PTQ 截断 |
| `RUN_MOE_PTQ` | 默认 `1`（执行 MoE PTQ）；置 `0` 仅用于冒烟测试，结果表不适用 |

## 产物说明

相对占位目录（默认在本样例目录下）：

| 产物 | 路径 | 大小（实测，GiB） |
|:--|:--|--:|
| PTQ 离线数据 Attention | `./outputs/qwen3_6_35b_a3b/ptq_data/attn-linear` | ~81 |
| PTQ 离线数据 MoE | `./outputs/qwen3_6_35b_a3b/ptq_data/moe` | ~21 |
| int PTQ 参数 Attention | `./outputs/qwen3_6_35b_a3b_w4a8_int/ptq_params/qwen3_6_moe/attn-linear` | ~5 |
| int PTQ 参数 MoE | `./outputs/qwen3_6_35b_a3b_w4a8_int/ptq_params/qwen3_6_moe/moe` | ~120 |
| mxfp PTQ 参数 Attention | `./outputs/qwen3_6_35b_a3b_w4a8_mxfp/ptq_params/qwen3_6_moe/attn-linear` | ~5 |
| mxfp PTQ 参数 MoE | `./outputs/qwen3_6_35b_a3b_w4a8_mxfp/ptq_params/qwen3_6_moe/moe` | ~128 |
| BF16 基线日志（两种格式共用） | `./outputs/qwen3_6_35b_a3b_bf16/logs/eval_qwen3_6_moe_bf16.log` | — |

量化时长口径：从该格式第一次 `ptq`（`ptq_attn.sh`）开始到 attn+moe 参数全部写完（不含模型下载和环境构建），`ptq_attn.sh` 与 `ptq_mlp.sh` 均打印 `start:` / `end:` 时间戳。int 行为全流程实测；mxfp 行时长取其首次全流程实测——之后仅按同脚本重跑过 moe 段（86.7 min）并回填量化 PPL，attn 段脚本与调度未变故未重跑，需要 mxfp 全流程口径时补跑 attn 段即可。离线数据大小为 extract 输出目录合计，GiB。

评估日志：`eval_quant.sh` 写 `<OUTPUT_DIR>/logs/eval_*.log`，int 与 mxfp 各自在自己的 `OUTPUT_DIR` 下，不会混进同一个文件；`eval_bf16.sh` 写与 `QUANT_DTYPE` 无关的 `<BF16_OUTPUT_DIR>/logs/`（默认 `./outputs/qwen3_6_35b_a3b_bf16/logs/`），两种格式共用同一份 BF16 基线日志。loguru 为追加写。

## 结果表

BF16 与量化评估使用同一 `MODEL` 与同一 `seq_len=4096` / `granularity=block` 配置。
评估覆盖 `attn-linear` + `moe`，两者均加载 autoround PTQ 参数。

结果表实测取值：Attention 用 `nsamples=128` / `epochs=10` / `cali_bsz=4`（nsamples、cali_bsz 与 CLI 默认一致，epochs 为本样例取值）；
MoE 覆盖为 `MOE_NSAMPLES=32` / `MOE_EPOCHS=1` / `MOE_CALI_BSZ=8`（Atlas A3 单卡，10240 个专家单元）。
复现结果表请使用同一组取值；量化时长为墙钟时间，含单元 I/O。

| 模型 | 数据类型 | 算法 | 量化格式 | BF16 PPL | 量化 PPL | PPL 差值 | 量化时长（min） | 离线数据大小（GiB） |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | `w4a8` | `autoround` | `int4/int8` | 6.3085 | 6.8263 | +0.5178 | 243.5 | 102 |
| Qwen3.6-35B-A3B | `w4a8` | `autoround` | `mxfp4/mxfp8` | 6.3085 | 6.4729 | +0.1644 | 303.2 | 102 |

> 差值列由**全精度**（未四舍五入）的量化 PPL 与 BF16 PPL 相减得出，与表中 4 位小数直接相减的结果在末位可能略有差异。
