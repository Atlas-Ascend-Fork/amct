# Qwen3-0.6B W4A4 量化样例（FlatQuant，社区任务 6）

本样例演示在昇腾 NPU 上使用 AMCT 对 **Qwen3-0.6B** 进行 **W4A4** 量化，PTQ 算法为 **flatquant**，分别验证 **int** 与 **mxfp** 两种量化数据格式的 fake quant 精度（WikiText2 PPL）。

社区任务：任务 6（量化格式 w4a4 + PTQ 算法 flatquant）。

## 量化配置

| 项 | 值 |
| --- | --- |
| 模型 | Qwen/Qwen3-0.6B（dense，28 层） |
| AMCT 模型名 | qwen3 |
| 量化格式 | w4a4（4bit 权重 + 4bit 激活） |
| 量化数据类型 | int、mxfp |
| 量化目标 | attn-linear、mlp |
| PTQ 算法 | flatquant（`--algos flatquant`） |
| 量化配置文件 | `amct_pytorch/configs/w4a4.yaml` |
| 评估口径 | WikiText2 PPL，seq_len=4096 |

## 运行说明

以下命令均在 **AMCT 仓库根目录** 执行。先设置模型路径，NPU 卡号可按 `npu-smi info` 空闲情况传入（默认 0）：

```bash
export MODEL_DIR=/path/to/Qwen3-0.6B
```

**Step 1. BF16 基线**

```bash
bash examples/models/qwen3-0.6b_w4a4_flatquant/eval_bf16.sh "$MODEL_DIR"
```

**Step 2. 提取 PTQ 离线数据**（attn-linear 与 mlp 各一遍，期望各 28 个 pkl；校准数据与算法/量化格式无关，可跨样例复用）

```bash
bash examples/models/qwen3-0.6b_w4a4_flatquant/extract_ptq_data.sh "$MODEL_DIR"
```

**Step 3. PTQ 训练**（int 与 mxfp 分别训练，输出目录分开）

```bash
bash examples/models/qwen3-0.6b_w4a4_flatquant/ptq_attn.sh "$MODEL_DIR" int
bash examples/models/qwen3-0.6b_w4a4_flatquant/ptq_mlp.sh  "$MODEL_DIR" int
bash examples/models/qwen3-0.6b_w4a4_flatquant/ptq_attn.sh "$MODEL_DIR" mxfp
bash examples/models/qwen3-0.6b_w4a4_flatquant/ptq_mlp.sh  "$MODEL_DIR" mxfp
```

**Step 4. 量化精度评估**（同时加载 attn + mlp 两组 PTQ 参数）

```bash
bash examples/models/qwen3-0.6b_w4a4_flatquant/eval_quant.sh "$MODEL_DIR" int
bash examples/models/qwen3-0.6b_w4a4_flatquant/eval_quant.sh "$MODEL_DIR" mxfp
```

## 参数说明

| 参数 | 含义 | 说明 |
| --- | --- | --- |
| `MODEL_DIR` | 本地模型权重目录 | 脚本第 1 个参数 |
| `int`/`mxfp` | 量化数据类型 | 脚本第 2 个参数，PTQ 与 eval 必须一致 |
| NPU 卡号 | 运行设备 | 脚本第 3 个参数，默认 0 |
| `--seq_len` | 序列长度 | 脚本内固定 4096，BF16 / 提取 / 评估必须一致 |
| `--data_dir` | PTQ 离线数据目录 | 提取与 PTQ 训练阶段一致 |
| `--output_dir` | PTQ 参数输出目录 | int 与 mxfp 分开保存于 `ptq_result/{int,mxfp}` |

## 产物说明

以下目录由脚本生成，**均不入库**（体积较大，README 仅说明结构与大小）：

| 目录 | 内容 | 实测大小（Qwen3-0.6B, seq_len=4096） |
| --- | --- | --- |
| `ptq_data/qwen3/attn-linear/` | Attention 逐层输入 pkl（28 个 `block_*_attn_in.pkl`） | 约 29 GB |
| `ptq_data/qwen3/mlp/` | MLP 逐层输入 pkl（28 个 `block_*_mlp_in.pkl`） | 约 29 GB |
| `ptq_result/{int,mxfp}/ptq_params/qwen3/{attn-linear,mlp}/` | PTQ 参数（每层一个 `.pt`，每种格式 56 个） | 每种格式约 1.7–1.8 GB |

## 结果

Atlas 910B（16 卡，独占单卡），CANN 8.5.0，torch_npu 2.7.1，WikiText2 PPL，seq_len=4096，flatquant 默认训练超参：

| 模型 | 量化格式 | 量化类型 | BF16 PPL | 量化 PPL | PPL 差值 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | w4a4 | 4/int8 | 19.166 | 48.659 | +29.493 | 激活 4bit 下 int 格式劣化显著 |
| Qwen3-0.6B | w4a4 | 4/mxfp8 | 19.166 | 28.150 | +8.984 | mxfp 显著优于 int |

**结论**：W4A4 全量化场景下，mxfp（块浮点）格式相比 int 在 0.6B 小模型上优势明显（PPL 差值 +8.98 vs +29.49），与 W4A8 场景下的对比趋势一致。

## 注意事项

1. **数据集离线准备**：WikiText2（评测）与 pile-val-backup（校准）默认从 HuggingFace 在线加载。无法直连时可设置 `HF_ENDPOINT=https://hf-mirror.com` 与 `HF_HUB_DISABLE_XET=1`，或预先下载后按 `load_dataset(data_files=...)` 方式改造数据加载。
2. **校准数据复用**：`extract_ptq_data` 产物与 PTQ 算法/量化格式无关，同模型同 seq_len 的其他 W4A4/W4A8 样例可直接复用 `--data_dir`。
3. **共享环境选卡**：多卡共享机器上建议先 `npu-smi info` 确认空闲卡，通过 `ASCEND_RT_VISIBLE_DEVICES` 指定，避免与常驻服务冲突。
