# DeepSeek-V4-Flash w4a4 `flatquant` 量化样例

对应 Issue [#182](https://gitcode.com/cann/amct/issues/182) **任务 30**（L3，`w4a4` + `flatquant`，华为手环 11）：对 DeepSeek-V4-Flash 做 `w4a4` + `flatquant`，**int4/int4 与 mxfp4/mxfp4 各跑一遍**。

CLI 入口：`amct_pytorch.eval` / `extract_ptq_data` / `ptq`（经 `scripts/run_amct.py` 包装）。  
位宽配置复用仓内 [`amct_pytorch/configs/w4a4.yaml`](../../../amct_pytorch/configs/w4a4.yaml)（`w_bits: 4, a_bits: 4`）；`int` / `mxfp` 由 CLI `--quant_dtype` 选择。对应本任务书的配置映射：`w4a4_int_flatquant.yaml` ≡ `w4a4.yaml` + `--quant_dtype int`，`w4a4_mxfp_flatquant.yaml` ≡ `w4a4.yaml` + `--quant_dtype mxfp`。

## 官方权重加载（加载期反量化）

DeepSeek-V4-Flash 官方 checkpoint 为混合格式：Attention / shared expert 为 FP8 e4m3（配 128×128 块 e8m0 scale），routed expert 为打包 int8 的 MXFP4（配每 32 元素一组 e8m0 scale）。全量 bf16 反导出约 600 GB，无法与 149 GB 源权重共存于 300 GB 盘，因此本样例采用**加载期反量化**：[`scripts/dequant_load_patch.py`](scripts/dequant_load_patch.py) 以 `safe_open` 代理在逐块加载时于 CPU 上按官方格式定义反量化为 bf16（数学实现与仓内 `amct_pytorch/experimental/deepseek-v4/convert_model.py` 一致），单层瞬时内存约 13 GB，`amct_pytorch/` 零改动。

## 运行说明

环境：2×Atlas A3（64GB HBM），extract/PTQ 以单进程串行方式占用双卡分片（同机并发两个分片加载进程会争抢 npu:0 stem 导致 OOM）。先按本机 CANN 安装路径配置环境，安装 `amct_pytorch`（或设置 `PYTHONPATH` 指向仓库根目录）。校准集 Pileval（`mit-han-lab/pile-val-backup`）与评估集 WikiText2（`wikitext-2-raw-v1` test）首次运行自动下载（可设 `HF_ENDPOINT`）。

- 模型：本地 DeepSeek-V4-Flash 官方权重目录；`--model_name deepseek_v4`；`--trust_remote_code`
- `flatquant` 覆盖 `attn-linear` 与 `moe` 两个量化目标，均加载真实 PTQ 参数
- 每种格式：extract（attn + moe）→ PTQ(attn) → PTQ(moe) → 量化评估（同时加载两组参数）
- 环境变量与默认值见各脚本头部；所有路径为相对/占位路径，脚本启动时打印完整运行参数

### 第一步：BF16 精度基线（多卡 device map）

```shell
bash scripts/eval_bf16.sh
```

### 第二步：PTQ 离线数据提取

`extract_ptq_data` 单次只支持一个 `--quant_target`，按 `QUANT_TARGET=attn-linear`、`QUANT_TARGET=moe` 各跑一次：

```shell
QUANT_TARGET=attn-linear bash scripts/extract_ptq_data.sh
QUANT_TARGET=moe bash scripts/extract_ptq_data.sh
```

### 第三步：flatquant PTQ

```shell
QUANT_DTYPE=int bash scripts/ptq_attn.sh
QUANT_DTYPE=int bash scripts/ptq_moe.sh
QUANT_DTYPE=mxfp bash scripts/ptq_attn.sh
QUANT_DTYPE=mxfp bash scripts/ptq_moe.sh
```

### 第四步：量化精度评估（单卡专家下卡路径）

```shell
QUANT_DTYPE=int bash scripts/eval_quant.sh
QUANT_DTYPE=mxfp bash scripts/eval_quant.sh
```

## 产物说明

产物根目录 `outputs/dsv4_flash_w4a4_flatquant/`：

| 产物 | 目录 | 大小 |
| --- | --- | --- |
| BF16 基线评估 | `01_baseline_eval/` | 日志（<0.1 GiB） |
| attn-linear 校准数据 | `ptq_data/attn-linear/`（`02_extract_attnlinear/`） | ~22 GiB |
| moe 校准数据 | `ptq_data/moe/`（`02_extract_moe/`） | ~22 GiB |
| int 格式 PTQ 参数（attn-linear + moe） | `ptq_params/int/{attn-linear,moe}/` | MB 级 |
| int 格式 PTQ 参数（moe） | `04_ptq_moe_int/`（运行日志） | 日志 |
| int 格式 PTQ 运行日志（attn） | `03_ptq_attn_int/`（运行日志） | 日志 |
| mxfp 格式 PTQ 参数（attn-linear + moe） | `ptq_params/mxfp/{attn-linear,moe}/` | MB 级 |
| mxfp 格式 PTQ 参数（moe） | `04_ptq_moe_mxfp/`（运行日志） | 日志 |
| mxfp 格式 PTQ 运行日志（attn） | `03_ptq_attn_mxfp/`（运行日志） | 日志 |
| int 量化评估 | `05_eval_quant_int/` | 日志（<0.1 GiB） |
| mxfp 量化评估 | `05_eval_quant_mxfp/` | 日志（<0.1 GiB） |

## 结果表

评测环境：2×Atlas A3（64GB HBM）、`seq_len=4096`、WikiText2 test split、Pileval 校准（`nsamples=16`）。

| 模型 | 数据类型 | 算法 | 量化格式 | BF16 PPL | 量化 PPL | PPL 差值 | 量化时长（min） | 离线数据大小（GiB） |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| DeepSeek-V4-Flash | `w4a4` | `flatquant` | `int4/int4` | 4.1456 | 26.0635 | 21.9179 | 572 | 44 |
| DeepSeek-V4-Flash | `w4a4` | `flatquant` | `mxfp4/mxfp4` | 4.1456 | 4.6865 | 0.5409 | 797 | 44 |

指标口径：

- **BF16 PPL**：`eval_mode=bf16` 下的 WikiText2 PPL。
- **量化 PPL**：加载 Attention 与 MoE 的 flatquant PTQ 参数后，量化模型的 WikiText2 PPL。
- **PPL 差值**：`量化 PPL - BF16 PPL`。
- **量化时长**：从开始执行 PTQ 到全部量化参数生成完成的时间（不含模型下载和环境构建）。
- **离线数据大小**：`extract_ptq_data` 输出目录总大小，Attention 与 MoE 数据目录合计。

## 备注

- 运行环境为 2×Atlas A3（64GB HBM）。各步实际路径：`extract_ptq_data.sh` 与 `ptq_*.sh` 以单进程分片加载占双卡（attn-linear 与 moe 两目标串行执行）；`eval_bf16.sh` 走多卡 device map；`eval_quant.sh` 走 `scripts/eval_quant_expert_offload.py` 单卡专家下卡路径（`DEVICE=npu:0`，同模型 w4a8 样例方案），单卡 64GB 环境可直接复用该路径。
- 结果表中 int 与 mxfp 两行均在上述环境实测：BF16 基线走多卡 device map，量化评估走专家下卡路径——两条路径显存行为不同，复现时请按本备注逐步对齐卡配置。
- flatquant 为本仓 DeepSeek-V4-Flash 组合首次运行。实际超参：`ptq_attn.sh` 默认 `EPOCHS=15`、`ptq_moe.sh` 默认 `EPOCHS=3`（两者脚本默认不同，结果表中两行分别按各自实际值执行），`BASE_LR=1e-3`、`CALI_BSZ=1` 为共用默认。
