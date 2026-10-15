# Qwen3-0.6B W4A4 量化样例（FlatQuant，社区任务 6）

本样例演示在昇腾 NPU 上使用 AMCT 对 **Qwen3-0.6B** 进行 **W4A4** 量化，PTQ 算法为 **flatquant**，分别验证 **int4/int4** 与 **mxfp4/mxfp4** 两种量化格式的 fake quant 精度（WikiText2 PPL）。

社区任务：任务 6（量化格式 w4a4 + PTQ 算法 flatquant）。

| 项 | 值 |
| --- | --- |
| 模型 | Qwen/Qwen3-0.6B（dense，28 层） |
| AMCT 模型名 | qwen3 |
| 量化格式 / 数据类型 | w4a4：int4/int4 与 mxfp4/mxfp4 |
| 量化目标 | attn-linear、mlp |
| PTQ 算法 | flatquant |
| 校准集 / 评估集 | Pileval（mit-han-lab/pile-val-backup）/ WikiText2（wikitext-2-raw-v1 test），seq_len=4096 |

## 运行说明

所有命令在 **AMCT 仓库根目录**执行。`<MODEL_DIR>` 为本地模型权重目录（占位路径），`NPU_ID` 按 `npu-smi info` 选择空闲卡（默认 0），`SEQ_LEN` 默认 4096 且 **BF16 评估、PTQ 数据提取、量化评估三阶段必须一致**。每个脚本启动时会打印完整运行参数。

**Step 1. BF16 基线评估**

```bash
bash examples/models/qwen3-0.6b_w4a4_flatquant/scripts/eval_bf16.sh <MODEL_DIR> <NPU_ID>
```

**Step 2. 提取 PTQ 离线数据**（Pileval；attn-linear 与 mlp 依次提取，Qwen3-0.6B 期望各 28 个 pkl）

```bash
bash examples/models/qwen3-0.6b_w4a4_flatquant/scripts/extract_ptq_data.sh <MODEL_DIR> <NPU_ID>
```

**Step 3. PTQ 训练**（int 与 mxfp 分别训练；`<dtype>` 取 `int` 或 `mxfp`，对应配置 `configs/w4a4_<dtype>_flatquant.yaml`）

```bash
bash examples/models/qwen3-0.6b_w4a4_flatquant/scripts/ptq_attn.sh <MODEL_DIR> int  <NPU_ID>
bash examples/models/qwen3-0.6b_w4a4_flatquant/scripts/ptq_mlp.sh  <MODEL_DIR> int  <NPU_ID>
bash examples/models/qwen3-0.6b_w4a4_flatquant/scripts/ptq_attn.sh <MODEL_DIR> mxfp <NPU_ID>
bash examples/models/qwen3-0.6b_w4a4_flatquant/scripts/ptq_mlp.sh  <MODEL_DIR> mxfp <NPU_ID>
```

**Step 4. 量化精度评估**（同时加载 attn + mlp 两组 PTQ 参数）

```bash
bash examples/models/qwen3-0.6b_w4a4_flatquant/scripts/eval_quant.sh <MODEL_DIR> int  <NPU_ID>
bash examples/models/qwen3-0.6b_w4a4_flatquant/scripts/eval_quant.sh <MODEL_DIR> mxfp <NPU_ID>
```

### 量化配置

| 文件 | 内容 | 用途 |
| --- | --- | --- |
| `configs/w4a4_int_flatquant.yaml` | `w_bits: 4` + `a_bits: 4` | 与 `--quant_dtype int --algos flatquant` 搭配 |
| `configs/w4a4_mxfp_flatquant.yaml` | `w_bits: 4` + `a_bits: 4` | 与 `--quant_dtype mxfp --algos flatquant` 搭配 |

量化数据类型由 CLI 参数 `--quant_dtype` 选择，配置文件固定权重/激活位宽（W4/A4）。

### 参数含义

| 参数 | 含义 | 说明 |
| --- | --- | --- |
| `<MODEL_DIR>` | 本地模型权重目录 | 脚本第 1 个参数（占位路径） |
| `<int\|mxfp>` | 量化数据类型（int4/int4、mxfp4/mxfp4） | 脚本第 2 个参数，PTQ 与 eval 必须一致 |
| `NPU_ID` | 运行设备卡号 | 默认 0 |
| `SEQ_LEN` | 序列长度 | 默认 4096，三个阶段必须一致 |
| `DATA_DIR` | PTQ 离线数据目录 | 默认 `ptq_data/qwen3/{attn-linear,mlp}`，提取与训练一致 |
| `OUTPUT_DIR` | PTQ 参数输出目录 | 默认 `ptq_result/<dtype>`，int 与 mxfp 分开保存 |

## 产物说明

以下目录由脚本生成，**均不提交到仓库**；模型权重与数据集由使用者自行准备：

| 产物 | 目录 | 内容 | 实测大小（Qwen3-0.6B，seq_len=4096） |
| --- | --- | --- | --- |
| PTQ 离线数据（Attention） | `ptq_data/qwen3/attn-linear/` | 28 个 `block_*_attn_in.pkl` | 29 GiB |
| PTQ 离线数据（MLP） | `ptq_data/qwen3/mlp/` | 28 个 `block_*_mlp_in.pkl` | 29 GiB |
| PTQ 离线数据合计 | — | — | **58 GiB** |
| 量化参数（int4/int4） | `ptq_result/int/ptq_params/qwen3/{attn-linear,mlp}/` | 每层一个 `.pt`，共 56 个 | 约 1.7 GB |
| 量化参数（mxfp4/mxfp4） | `ptq_result/mxfp/ptq_params/qwen3/{attn-linear,mlp}/` | 每层一个 `.pt`，共 56 个 | 约 1.8 GB |

## 结果表

环境：Atlas 910B（16 卡，独占单卡），CANN 8.5.0，torch_npu 2.7.1；flatquant 默认训练超参；量化时长为 PTQ 开始至该格式全部量化参数生成完成的耗时（不含模型下载与环境构建，按参数文件生成时间统计）。

| 模型 | 数据类型 | 算法 | 量化格式 | BF16 PPL | 量化 PPL | PPL 差值 | 量化时长（min） | 离线数据大小（GiB） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | w4a4 | flatquant | int4/int4 | 19.166 | 48.659 | +29.493 | 34.1 | 58 |
| Qwen3-0.6B | w4a4 | flatquant | mxfp4/mxfp4 | 19.166 | 28.150 | +8.984 | 93.2 | 58 |

**结论**：W4A4 全量化下 mxfp4/mxfp4 显著优于 int4/int4（PPL 差值 +8.98 vs +29.49）；mxfp 训练耗时更长（93.2 vs 34.1 min）但精度优势明显。

## 注意事项

1. **数据集准备**：Pileval 与 WikiText2 默认在线加载（HuggingFace）。离线环境可设 `HF_ENDPOINT=https://hf-mirror.com` 与 `HF_HUB_DISABLE_XET=1` 预下载，或按 `load_dataset(data_files=...)` 本地加载。
2. **口径一致性**：BF16 与量化评估必须使用同一模型版本、同一数据集、同一 `seq_len`，否则 PPL 差值无意义。
3. **共享环境选卡**：先 `npu-smi info` 确认空闲卡并通过 `NPU_ID` 参数指定，避免与常驻服务冲突。
