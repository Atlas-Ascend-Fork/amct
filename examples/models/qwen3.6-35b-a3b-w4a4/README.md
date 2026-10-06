# Qwen3.6-35B-A3B w4a4 `autoround` 量化样例

对应 Issue [#182](https://gitcode.com/cann/amct/issues/182) **任务 24**（目录 `qwen3.6-35b-a3b-w4a4/`，与同模型其它量化配置分目录，避免路径冲突）：对 Qwen3.6-35B-A3B 做 `w4a4` + `autoround`，**int4/int4 与 mxfp4/mxfp4 各跑一遍**。

CLI 入口与 [Qwen3.6-Moe.md](../qwen3.6/Qwen3.6-Moe.md) 一致：`python3 -m amct_pytorch.eval` / `extract_ptq_data` / `ptq`。  
位宽配置复用仓内 [`amct_pytorch/configs/w4a4.yaml`](../../../amct_pytorch/configs/w4a4.yaml)；`int` / `mxfp` 由 CLI `--quant_dtype` 选择。

| 格式 | 配置文件 | CLI `--quant_dtype` |
| --- | --- | --- |
| int4/int4 | `amct_pytorch/configs/w4a4.yaml` | `int` |
| mxfp4/mxfp4 | `amct_pytorch/configs/w4a4.yaml` | `mxfp` |

MoE 模型量化目标为 `attn-linear` + `moe`（适配器不支持 `mlp`）。

## 运行说明

环境：单卡 `npu:0`。先按本机 CANN 安装路径配置环境，并安装 `amct_pytorch`（或设置 `PYTHONPATH` 指向仓库根目录）。  
一站式平台示例：`source /home/developer/Ascend/cann/set_env.sh`。若需镜像下载校准/评估数据，可自行设置 `HF_ENDPOINT`。

- 模型：本地 Qwen3.6-35B-A3B 权重；`--model_name qwen3_6_moe`；`--trust_remote_code`
- 校准集：Pileval（`mit-han-lab/pile-val-backup`），CLI 默认 `nsamples=128`
- 评估集：WikiText2 `wikitext-2-raw-v1` test；`--seq_len 4096`；`--granularity block`
- BF16 与量化评估使用同一权重目录、同一评估配置
- 量化覆盖 Attention（`attn-linear`）与 MoE（评估时 `--quant_target attn-linear moe`）
- **autoround PTQ 只跑 Attention**；**MoE 采用直转假量化**（评估时 `--quant_target attn-linear moe`，不加载 per-expert autoround 参数）。与官方 [Qwen3.6-Moe.md](../qwen3.6/Qwen3.6-Moe.md) 一致，也符合「耗时过长则对 MoE 做直转」的交付口径
- `ptq_mlp.sh` 默认 SKIP；仅当显式 `RUN_MOE_PTQ=1` 时才跑全量专家 autoround（不推荐作为社区样例默认路径）
- 每种格式：extract → PTQ(attn) → 量化评估（加载 attn PTQ 参数；MoE 按 bit_config 直转）
- 产物目录为相对占位路径，不提交数据/权重

### 时间消耗评估

口径：单卡 `npu:0`（非多卡）；`autoround`、`epochs=10`、`seq_len=4096`、`end_block_idx=40`。

| 步骤 | 墙钟时间 | 依据 |
| --- | --- | --- |
| extract（attn + moe） | 约 1.5–2.5 h | 实测墙钟（attn + moe 两次 extract 合计；离线数据约 162 GiB） |
| Attention autoround PTQ（每种 dtype） | 约 2 h（~120 min） | 实测 int / mxfp 各约 120 min；结果表「量化时长」按此口径 |
| 全量 MoE autoround PTQ | **约 40 天量级（单卡）** | 见下方外推；故默认不做，改为 MoE 直转 |
| 量化评估 | 约 0.5–1 h | 实测墙钟（含 block 假量化前向 + WikiText2 PPL） |

**全量 MoE 外推（单卡 `npu:0`，mxfp 试跑）：**

- 专家 unit 数：`40 层 × 256 专家 ≈ 10240`
- 实测进度：写出 `layer_0_expert_205.pt`、进行到 `expert_206`（第 0 层约 **206/256**）时，该次 MoE PTQ 已连续跑约 **20 h** 墙钟
- 单专家均耗：`20 h / 206 ≈ 5.8 min/expert`
- 第 0 层估满：`20 × 256 / 206 ≈ 24.9 h/层`
- 全层估满：`24.9 h/层 × 40 ≈ 996 h ≈ **41.5 天**`（或 `10240 × 5.8 min ≈ 990 h`）

结果表中的「量化时长」= 该格式 **Attention autoround PTQ** 墙钟时间（不含 extract / eval / 模型下载 / 环境构建）。

在仓库根目录执行：

```bash
source /path/to/cann/set_env.sh
export MODEL=/path/to/Qwen3.6-35B-A3B   # 占位，改为本地权重

# 1) BF16 基线（两种格式共用）
bash examples/models/qwen3.6-35b-a3b-w4a4/scripts/eval_bf16.sh

# 2) 提取 PTQ 离线数据（attn + moe，两种格式共用；moe 数据可留作可选全量 PTQ）
bash examples/models/qwen3.6-35b-a3b-w4a4/scripts/extract_ptq_data.sh

# 3) int4/int4：PTQ(attn) → 量化评估
export QUANT_DTYPE=int
bash examples/models/qwen3.6-35b-a3b-w4a4/scripts/ptq_attn.sh
bash examples/models/qwen3.6-35b-a3b-w4a4/scripts/ptq_mlp.sh   # 默认 SKIP
bash examples/models/qwen3.6-35b-a3b-w4a4/scripts/eval_quant.sh

# 4) mxfp4/mxfp4
export QUANT_DTYPE=mxfp
bash examples/models/qwen3.6-35b-a3b-w4a4/scripts/ptq_attn.sh
bash examples/models/qwen3.6-35b-a3b-w4a4/scripts/eval_quant.sh
```

也可直接使用 README 中的 `python3 -m ...` 命令；脚本启动时会打印完整运行参数。

主要参数：

| 参数 / 环境变量 | 含义 |
|:--|:--|
| `MODEL` | 本地权重目录（占位路径） |
| `--model_name qwen3_6_moe` | Qwen3.6 MoE 通路 |
| `--seq_len 4096` | 校准与评估序列长度 |
| `--granularity block` | Block 粒度 |
| `--quant_dtype int\|mxfp` | int4/int4 或 mxfp4/mxfp4 |
| `--algos autoround` | 量化算法 |
| `--bit_config` | `amct_pytorch/configs/w4a4.yaml` |
| `--quant_target` | extract/PTQ 每次一个目标：`attn-linear` 或 `moe`；量化评估同时加载两者 |
| `--nsamples` | Pileval 校准条数，默认 128 |
| `--epochs` / `--base_lr` | 本样例取值 `10` / `1e-3`（CLI 默认多为 `15` / `1e-5`，由 `scripts/common.sh` 覆盖） |
| `--end_block_idx` | 本样例取值 `40`（exclusive）。该模型共 40 层，已覆盖全部层；CLI 默认 `61` 会在 PTQ 内截断到层数 |

等价 CLI 示例（int）：

```bash
python3 -m amct_pytorch.ptq \
  --trust_remote_code \
  --model "${MODEL}" \
  --model_name qwen3_6_moe \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --data_dir examples/models/qwen3.6-35b-a3b-w4a4/outputs/qwen3_6_35b_a3b/ptq_data/attn-linear \
  --quant_dtype int \
  --algos autoround \
  --bit_config amct_pytorch/configs/w4a4.yaml \
  --quant_target attn-linear \
  --start_block_idx 0 \
  --end_block_idx 40 \
  --epochs 10 \
  --base_lr 1e-3 \
  --output_dir examples/models/qwen3.6-35b-a3b-w4a4/outputs/qwen3_6_35b_a3b_w4a4_int
```

## 产物说明

相对占位目录（默认在本样例目录下）：

| 产物 | 路径 | 大小 |
|:--|:--|:--|
| PTQ 离线数据 Attention | `./outputs/qwen3_6_35b_a3b/ptq_data/attn-linear` | 81 GiB |
| PTQ 离线数据 MoE | `./outputs/qwen3_6_35b_a3b/ptq_data/moe` | 81 GiB |
| 离线数据合计（attn+moe） | 上述两目录之和 | 162 GiB |
| int PTQ 参数 Attention | `./outputs/qwen3_6_35b_a3b_w4a4_int/ptq_params/qwen3_6_moe/attn-linear` | 4.8 GiB |
| int PTQ 参数 MoE | （默认不生成；`RUN_MOE_PTQ=1` 时才有） | — |
| mxfp PTQ 参数 Attention | `./outputs/qwen3_6_35b_a3b_w4a4_mxfp/ptq_params/qwen3_6_moe/attn-linear` | 5.1 GiB |
| mxfp PTQ 参数 MoE | （默认不生成） | — |
| BF16 / int 评估日志 | `./outputs/qwen3_6_35b_a3b_w4a4_int/logs/`（如 `eval_qwen3_6_moe_bf16.log` / `eval_qwen3_6_moe_quant.log`） | — |
| mxfp 评估日志 | `./outputs/qwen3_6_35b_a3b_w4a4_mxfp/logs/`（如 `eval_qwen3_6_moe_quant.log`） | — |
| extract 日志 | `./outputs/qwen3_6_35b_a3b_w4a4_int/logs/`（默认 `QUANT_DTYPE=int`；如 `extract_ptq_data.log`；attn/moe 两次追加同一文件） | — |

离线数据大小为 extract 输出目录合计（GiB）。评估 / extract 日志与对应 `--output_dir`（`OUTPUT_DIR`）同目录，避免落到仓库根 `./outputs/logs/`。

## 结果表

BF16 与量化评估使用同一 `MODEL` 与同一 `seq_len=4096` / `granularity=block` 配置。  
评估覆盖 `attn-linear` + `moe`：Attention 加载 autoround PTQ 参数，**MoE 为 w4a4 直转假量化**（因全量专家 autoround 耗时过长）。

| 模型 | 数据类型 | 算法 | 量化格式 | BF16 PPL | 量化 PPL | PPL 差值 | 量化时长（min） | 离线数据大小（GiB） |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | `w4a4` | `autoround` | `int4/int4` | 6.308547 | 264.116119 | 257.807572 | 120 | 162 |
| Qwen3.6-35B-A3B | `w4a4` | `autoround` | `mxfp4/mxfp4` | 6.308547 | 6.876903 | 0.568356 | 120 | 162 |

**精度口径**：`int4/int4` 的 264.116119 是默认路径「仅 Attention autoround + MoE int4 直转」的**预期结果**（与同次 `eval_quant` 日志一致），不是配错；同配置下 `mxfp4/mxfp4` 因数值格式对直转更友好，PPL 可到 6.88。若要压低 int 掉点，需 `RUN_MOE_PTQ=1` 跑全量专家 autoround（约 41.5 天墙钟，默认关闭）。
