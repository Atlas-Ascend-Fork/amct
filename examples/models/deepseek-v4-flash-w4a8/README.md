# DeepSeek-V4-Flash w4a8 `lwc` + `lac` 量化样例

对应 Issue [#182](https://gitcode.com/cann/amct/issues/182) **任务 25**（目录 `deepseek-v4-flash-w4a8/`，与同模型 W4A4 样例 `deepseekv4/` 分目录）：对 DeepSeek-V4-Flash 做 `w4a8` + `lwc` + `lac`，**int4/int8 与 mxfp4/mxfp8 各跑一遍**。

CLI 入口：`python3 -m amct_pytorch.eval` / `extract_ptq_data` / `ptq`。  
位宽配置复用仓内 [`amct_pytorch/configs/w4a8.yaml`](../../../amct_pytorch/configs/w4a8.yaml)（routed expert `w4a8`，shared expert `w8a8`）；`int` / `mxfp` 由 CLI `--quant_dtype` 选择。

| 格式 | 配置文件 | CLI `--quant_dtype` |
| --- | --- | --- |
| int4/int8 | `amct_pytorch/configs/w4a8.yaml` | `int` |
| mxfp4/mxfp8 | `amct_pytorch/configs/w4a8.yaml` | `mxfp` |

模型量化目标为 `attn-linear` + `moe`（DeepSeek-V4-Flash 无独立 `mlp`）。

## 运行说明

环境：**单卡** `npu:0`（64GB HBM）。机器上若能看到两张卡，评测前设置 `ASCEND_RT_VISIBLE_DEVICES=0`，避免激活与专家权重落到不同卡。先按本机 CANN 安装路径配置环境，并安装 `amct_pytorch`（或设置 `PYTHONPATH` 指向仓库根目录）。  
一站式平台示例：`source /home/developer/Ascend/cann/set_env.sh`。若需镜像下载校准/评估数据，可自行设置 `HF_ENDPOINT`。

- 模型：本地 DeepSeek-V4-Flash 权重；`--model_name deepseek_v4`；`--trust_remote_code`
- 官方 checkpoint 是混合格式：Attention / shared expert 为 FP8 e4m3，配套 `float8_e8m0fnu` scale（128×128 分块）；routed expert 为打包 int8 的 MXFP4，配套 e8m0 scale（最后一维每 32 个元素一组）。本次结果在**加载期反量化**后评测，没有另存整份 bf16 副本。复现时可将 `MODEL` 指向按 [DeepSeekV4-Flash-Walkthrough.md](../deepseekv4/DeepSeekV4-Flash-Walkthrough.md) 导出的 bf16 目录，或在加载期用 `weight_dequant` 做同样的反量化
- 校准集：Pileval（`mit-han-lab/pile-val-backup`），`nsamples=128`
- 评估集：WikiText2 `wikitext-2-raw-v1` test；`--seq_len 4096`；`--granularity block`
- BF16 与量化评估使用同一权重、同一评估配置
- **`lwc` + `lac` 只跑 Attention**；**MoE 采用 w4a8 直转假量化**（评估时 `--quant_target attn-linear moe`，不加载 per-expert PTQ 参数）
- `ptq_mlp.sh` 默认 SKIP；仅当显式 `RUN_MOE_PTQ=1` 时才跑全量专家 PTQ（不作为社区样例默认路径）
- 每种格式：extract → PTQ(attn) → 量化评估（加载 attn PTQ 参数；MoE 按 bit_config 直转）
- 产物目录为相对占位路径，不提交数据/权重
- **量化评估（单卡 OOM 与样例补丁）**：结果表在**单卡 + 样例本地 monkeypatch**下测得，不是多卡分片。裸跑 `python3 -m amct_pytorch.eval` 时，`llm_eval` 开 `sharded_block=True`，单卡下 `_build_block_device_map` 返回 `None`，整层（含 256 个 routed expert）一次上 `npu:0` 会 OOM；逐 expert 上下卡只在 PTQ 路径（`llm_ptq.py`）存在。本样例 `eval_quant.sh` 调用 [`scripts/eval_quant_expert_offload.py`](scripts/eval_quant_expert_offload.py)：在 `_dispatch_block` 后把 routed expert park 到 CPU，MoE `forward` 里用到哪个再 `.to(npu:0)`。`amct_pytorch/` **零改动**。上游跟踪：[Issue #248](https://gitcode.com/cann/amct/issues/248)。同模型 [W4A4 样例](../deepseekv4/README.md) 评测环境是 2×Atlas A3（多卡 device map），本样例用单卡 + 上述补丁复现结果表。

### 时间消耗评估

口径：单卡 `npu:0`；`lwc` + `lac`、`epochs=15`、`nsamples=128`、`cali_bsz=1`、`seq_len=4096`、`end_block_idx=43`。

| 步骤 | 墙钟时间 | 依据 |
| --- | --- | --- |
| extract（attn + moe） | 约 5.3 h | 实测：每个 target 的 block 循环约 2 h 39 min（43 层，约 222 s/层），两个 target 合计 |
| Attention `lwc`+`lac` PTQ（每种 dtype） | 约 30 h（1806 min） | 实测稳定段约 **42 min/层** × 43 层。结果表「量化时长」按此口径 |
| 全量 MoE `lwc`+`lac` | **数周至数百天（单卡）** | 见下方外推；故默认不做，改为 MoE 直转 |
| 量化评估 | 约 6.5–7 h / 格式 | 实测 mxfp 的 block 循环 6 h 39 min（单卡 + `eval_quant_expert_offload.py` 专家逐个上下卡）；int 同量级 |
| BF16 评估 | 约 1.5–2 h | 实测 block 循环约 1 h 30 min |

**全量 MoE 外推（单卡 `npu:0`）：**

- 层数：43。routed expert：与同模型 [W4A4 样例](../deepseekv4/README.md) 一致，每层 **256** 个
- Attention 只有 1 个 PTQ unit/层，实测约 42 min/层（15 epoch × 128 step，`cali_bsz=1`）
- MoE 是每个 expert 一个 unit：`43 × 256 = 11008` 个 unit
- 若单专家耗时与 Attention unit 同量级：`42 min × 11008 ≈ 462336 min` ≈ **321 天**/格式
- 即便单专家只有 Attention 的 1/10，单格式仍约 **32 天**。因此默认直转

结果表中的「量化时长」= 该格式 **Attention PTQ** 墙钟时间（不含 extract / eval / 模型下载 / 环境构建）。

在仓库根目录执行：

```bash
source /path/to/cann/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export MODEL=/path/to/DeepSeek-V4-Flash   # 占位，改为本地权重

# 1) BF16 基线（两种格式共用）
bash examples/models/deepseek-v4-flash-w4a8/scripts/eval_bf16.sh

# 2) 提取 PTQ 离线数据（attn + moe，两种格式共用）
bash examples/models/deepseek-v4-flash-w4a8/scripts/extract_ptq_data.sh

# 3) int4/int8：PTQ(attn) → 量化评估
export QUANT_DTYPE=int
bash examples/models/deepseek-v4-flash-w4a8/scripts/ptq_attn.sh
bash examples/models/deepseek-v4-flash-w4a8/scripts/ptq_mlp.sh   # 默认 SKIP
bash examples/models/deepseek-v4-flash-w4a8/scripts/eval_quant.sh

# 4) mxfp4/mxfp8
export QUANT_DTYPE=mxfp
bash examples/models/deepseek-v4-flash-w4a8/scripts/ptq_attn.sh
bash examples/models/deepseek-v4-flash-w4a8/scripts/eval_quant.sh
```

也可直接使用下方 `python3 -m ...` 命令；脚本启动时会打印完整运行参数。

主要参数：

| 参数 / 环境变量 | 含义 |
|:--|:--|
| `MODEL` | 本地权重目录（占位路径） |
| `--model_name deepseek_v4` | DeepSeek-V4 通路 |
| `--seq_len 4096` | 校准与评估序列长度 |
| `--granularity block` | Block 粒度 |
| `--quant_dtype int\|mxfp` | int4/int8 或 mxfp4/mxfp8 |
| `--algos lwc lac` | 量化算法 |
| `--bit_config` | `amct_pytorch/configs/w4a8.yaml` |
| `--quant_target` | extract/PTQ 每次一个目标：`attn-linear` 或 `moe`；量化评估同时覆盖两者 |
| `--nsamples` | Pileval 校准条数。本样例取值 `128`，extract 与 PTQ 必须一致 |
| `--cali_bsz` | 本样例取值 `1`（CLI 默认 4，64GB 单卡会 OOM） |
| `--epochs` / `--base_lr` | 本样例取值 `15` / `1e-3`（`base_lr` 的 CLI 默认是 `1e-5`，裁剪系数几乎不更新） |
| `--end_block_idx` | 本样例取值 `43`（exclusive）。该模型共 43 层，已覆盖全部层；CLI 默认 `61` 会在 PTQ 内截断到层数 |

等价 CLI 示例（int，Attention PTQ）：

```bash
python3 -m amct_pytorch.ptq \
  --trust_remote_code \
  --model "${MODEL}" \
  --model_name deepseek_v4 \
  --seq_len 4096 \
  --granularity block \
  --device npu:0 \
  --data_dir examples/models/deepseek-v4-flash-w4a8/outputs/deepseek_v4_flash/ptq_data/attn-linear \
  --quant_dtype int \
  --algos lwc lac \
  --bit_config amct_pytorch/configs/w4a8.yaml \
  --quant_target attn-linear \
  --nsamples 128 \
  --cali_bsz 1 \
  --start_block_idx 0 \
  --end_block_idx 43 \
  --epochs 15 \
  --base_lr 1e-3 \
  --output_dir examples/models/deepseek-v4-flash-w4a8/outputs/deepseek_v4_flash_w4a8_int
```

## 产物说明

相对占位目录（默认在本样例目录下）：

| 产物 | 路径 | 大小 |
|:--|:--|:--|
| PTQ 离线数据 Attention | `./outputs/deepseek_v4_flash/ptq_data/attn-linear` | 173 GiB |
| PTQ 离线数据 MoE | `./outputs/deepseek_v4_flash/ptq_data/moe` | 173 GiB |
| 离线数据合计（attn+moe） | 上述两目录之和 | 346 GiB |
| int PTQ 参数 Attention | `./outputs/deepseek_v4_flash_w4a8_int/ptq_params/deepseek_v4/attn-linear` | 18 MiB |
| int PTQ 参数 MoE | （默认不生成；`RUN_MOE_PTQ=1` 时才有） | — |
| mxfp PTQ 参数 Attention | `./outputs/deepseek_v4_flash_w4a8_mxfp/ptq_params/deepseek_v4/attn-linear` | 1.2 GiB |
| mxfp PTQ 参数 MoE | （默认不生成） | — |
| BF16 / int 评估日志 | `./outputs/deepseek_v4_flash_w4a8_int/logs/` | — |
| mxfp 评估日志 | `./outputs/deepseek_v4_flash_w4a8_mxfp/logs/` | — |
| extract 日志 | `./outputs/deepseek_v4_flash_w4a8_int/logs/`（默认 `QUANT_DTYPE=int`；attn/moe 追加同一文件） | — |

离线数据大小为 extract 输出目录合计（GiB）。评估 / extract / PTQ 日志与对应 `--output_dir`（`OUTPUT_DIR`）同目录，避免落到仓库根 `./outputs/logs/`。

## 结果表

BF16 与量化评估使用同一 `MODEL` 与同一 `seq_len=4096` / `granularity=block` 配置。  
评估覆盖 `attn-linear` + `moe`：Attention 加载 `lwc`+`lac` PTQ 参数，**MoE 为 w4a8 直转假量化**。

| 模型 | 数据类型 | 算法 | 量化格式 | BF16 PPL | 量化 PPL | PPL 差值 | 量化时长（min） | 离线数据大小（GiB） |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| DeepSeek-V4-Flash | `w4a8` | `lwc` + `lac` | `int4/int8` | 4.145633 | 4.779413 | 0.633780 | 1806 | 346 |
| DeepSeek-V4-Flash | `w4a8` | `lwc` + `lac` | `mxfp4/mxfp8` | 4.145633 | 4.270695 | 0.125062 | 1806 | 346 |

**精度口径**：两行都是默认路径「仅 Attention `lwc`+`lac` + MoE w4a8 直转」的结果，与同次 `eval_quant` 日志一致。int 差值 0.633780、mxfp 差值 0.125062 都是该路径的预期结果，不是漏加载 Attention 参数。若要再压 MoE 直转掉点，需 `RUN_MOE_PTQ=1` 跑全量专家 PTQ（单格式约数十天到数百天，默认关闭）。
