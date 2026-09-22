# DeepSeek-V4-Flash W4A4 量化样例（lwc + lac）

本样例使用 AMCT 对 DeepSeek-V4-Flash 完成 `w4a4`（权重 4bit + 激活 4bit）的 `lwc` + `lac` 后训练量化，覆盖 Attention 与 MoE 两类量化目标，交付 `int4/int4` 与 `mxfp4/mxfp4` 两种量化格式的可复现流程。

## 运行说明

量化工作流共四步：BF16 精度基线评测 → PTQ 离线数据提取 → PTQ 训练 → 带 PTQ 参数的量化精度评估。所有脚本位于 `scripts/`，位宽配置使用仓库自带的 `amct_pytorch/configs/w4a4.yaml`。

以下命令均在样例目录 `examples/models/deepseekv4/` 下执行（脚本已把位宽配置与产物目录锚定为绝对路径，从仓库根等其他目录调用同样可用）。四步必须复用同一个 `OUTPUT_DIR`（默认 `<样例目录>/outputs/dsv4_flash_w4a4`）：第二步的校准数据与第三步的量化参数都由它派生，中途换目录会导致第三步找不到第二步的数据。

### 前置条件

- 模型权重：从 [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) 下载（约 149 GB）。官方 checkpoint 是 1 字节混合格式：Attention 权重为 FP8 e4m3（配 128x128 块 scale），MoE expert 权重为打包 int8 的 MXFP4（配每行 32 元素组的 E8M0 scale），两者都要乘回配对的 `.scale` 键才能还原数值。
- **主干 `deepseek_v4` 适配器在加载期不做反量化**：`_block_sharded` 只对浮点张量做裸 dtype cast（不乘 block scale，FP8 数值错），打包 int8 的 MXFP4 连 cast 都不做；PTQ 路径按 `layers.{i}.` 前缀收全部键后走 `load_state_dict(strict=True)`，多出的 `.scale` 键会直接报错。因此直接拿官方 checkpoint 跑本样例的四步流程，会得到错误结果或加载失败。
- 本样例结果表是在额外一层**加载期反量化胶水**上跑出来的：在适配器中新增 `_dequant_tensor` 并 override `load_layer_weight`，把 `*.weight` 与其配对 `.scale` 一并取出后，调用 AMCT 已有的 `from amct_pytorch.quantization.dtypes.fp_impl import weight_dequant`（源文件 `amct_pytorch/quantization/dtypes/fp_impl.py`；MX 系列的解包/反量化同样在该文件中，仓库内没有 `mxfp_impl.py`）还原为 bf16（FP8 走 `block_size=128`；MXFP4 走 `block_size=32, is_mx=True, is_packed=True`，与适配器的 `block_size` 约定一致：int8 权重取 32、其余取 128），并丢弃已消费的 `.scale` 键。该胶水属样例配套改动、**未随本 PR 提交，`amct_pytorch/` 零改动**；同类的 FP8 / MXFP4 / MXFP8 / NVFP4 / HiF4 分派逻辑在导出路径 `amct_pytorch/common/models/llm/common/deploy_export.py` 中已有现成实现（其 `convert_state_dict` 就是这套调用的参考写法），可据此在加载期补齐。
- 复现结果前二选一：
  - **(a) 先把权重转成 bf16（推荐，无需改代码）**：按 [DeepSeekV4-Flash-Walkthrough.md](DeepSeekV4-Flash-Walkthrough.md) 第 2 节，用 `deploy` 接口配 `--granularity tensor` 导出 bf16 副本，再把 `MODEL_PATH` 指向该副本；
  - **(b) 自行在适配器加载期补上上述反量化胶水**，之后 `MODEL_PATH` 可直接指向官方 checkpoint。
- 校准集：Pileval（`mit-han-lab/pile-val-backup`），评估集：WikiText2（`wikitext-2-raw-v1` test split），`seq_len=4096`。首次运行时自动下载。
- 硬件：单卡即可运行；双卡可通过 `--start_block_idx/--end_block_idx` 按 block 分片并行 PTQ。注意单卡 64GB HBM 下 PTQ 需 `--cali_bsz 1`。

### 第一步：BF16 精度基线

```shell
bash scripts/eval_bf16.sh
```

默认参数：`MODEL_PATH`（模型目录）、`BIT_CONFIG=<仓库根>/amct_pytorch/configs/bf16.yaml`、`SEQ_LEN=4096`、`DEVICE=npu:0`。基线 PPL 用于计算量化精度损失。

### 第二步：PTQ 离线数据提取

`extract_ptq_data` 每次只支持一个 `quant_target`，需分别对 Attention 与 MoE 各执行一次（默认 `NSAMPLES=16`，每次提取约 22 GiB 数据，两 target 合计约 43 GiB，请预留磁盘空间）：

```shell
QUANT_TARGET=attn-linear bash scripts/extract_ptq_data.sh
QUANT_TARGET=moe bash scripts/extract_ptq_data.sh
```

`--nsamples` 受磁盘预算约束：提取钩子落盘的是 `attn_norm` / `ffn_norm` 的输出，而 `Block.forward` 会先经 `hc_pre` 把 Hyper-Connection 的 hc 维（`hc_mult=4`）加权求和收敛成 `[b, s, d]`，RMSNorm 再 `.to(dtype)` 转回 bf16，因此落盘张量是 3D bf16，每样本每层约 32 MiB（`seq_len 4096 × hidden 4096 × 2 B`）；`nsamples=128` 时单 target 需约 172 GiB，本样例取 16（16 样本 × 43 层 ≈ 21.5 GiB）。

> **第三步必须复用同一个 `NSAMPLES` 取值**（脚本默认已对齐为 16）：PTQ 的 cosine 学习率调度按 `T_max = epochs × (nsamples // cali_bsz)` 计算周期，而实际优化步数由本步提取出的样本数决定。若第三步沿用 CLI 默认的 `nsamples=128`，周期会是真实步数的 8 倍，训练结束时学习率仍停在 `base_lr` 的约 96%，`lwc/lac` 的裁剪系数几乎没有退火。

### 第三步：PTQ 训练

Attention 与 MoE 分别训练（DeepSeek-V4 为纯 MoE 结构，无独立 MLP 分支，`quant_target=mlp` 会被适配器拒绝）：

```shell
# int4/int4 格式
QUANT_DTYPE=int bash scripts/ptq_attn.sh
QUANT_DTYPE=int bash scripts/ptq_moe.sh

# mxfp4/mxfp4 格式
QUANT_DTYPE=mxfp bash scripts/ptq_attn.sh
QUANT_DTYPE=mxfp bash scripts/ptq_moe.sh
```

关键参数说明：

- `QUANT_DTYPE`：`int` 或 `mxfp`。位宽配置使用仓库自带的 `amct_pytorch/configs/w4a4.yaml`（`w_bits: 4 / a_bits: 4`），量化格式由 `--quant_dtype` 决定。
- `BASE_LR=1e-3`：学习率。**不要使用框架默认值 1e-5**——实测默认学习率下 `lwc/lac` 的裁剪系数几乎不更新（clip factor 停留在初始值 4.0 附近），量化精度显著劣化。
- `EPOCHS`：Attention 侧 15、MoE 侧 3。MoE 每层有 256 个 routed expert，逐 expert 训练开销大，3 epoch 已可收敛。
- `CALI_BSZ=1`：64GB 单卡 HBM 所需的最小 batch size。
- `NSAMPLES=16`：须与第二步 `extract_ptq_data.sh` 的取值一致（见上文提示），否则 cosine 调度周期与实际步数失配。

Attention 与 MoE 训练都支持按 block 分片多卡并行（例如双卡各跑一半层）：

```shell
ASCEND_RT_VISIBLE_DEVICES=0 START_BLOCK_IDX=0 END_BLOCK_IDX=22 QUANT_DTYPE=int bash scripts/ptq_attn.sh &
ASCEND_RT_VISIBLE_DEVICES=1 START_BLOCK_IDX=22 END_BLOCK_IDX=43 QUANT_DTYPE=int bash scripts/ptq_attn.sh &
wait

ASCEND_RT_VISIBLE_DEVICES=0 START_BLOCK_IDX=0 END_BLOCK_IDX=22 QUANT_DTYPE=int bash scripts/ptq_moe.sh &
ASCEND_RT_VISIBLE_DEVICES=1 START_BLOCK_IDX=22 END_BLOCK_IDX=43 QUANT_DTYPE=int bash scripts/ptq_moe.sh &
wait
```

分片并行时有两条硬约束：

- 各卡必须复用同一个 `PARAM_DIR` / `OUTPUT_DIR`（脚本默认值已一致），这样各分片产出的 `layer_{i}_*.pt` 才能汇到一处供第四步加载；
- 各卡的 `[START_BLOCK_IDX, END_BLOCK_IDX)` 必须**互补且不重叠**地覆盖 `[0, 43)`（如上例的 `[0, 22)` 与 `[22, 43)`）。区间不重叠，各分片才只写各自层号的文件。

反例：只给其中一张卡设置这两个变量。此时设了的卡按子区间跑，没设的卡按脚本兜底值 `0..43` 跑完整 43 层，两卡层区间重叠——重叠层会被两个进程并发写同一个 `layer_{i}_*.pt`（`amct_pytorch/workflows/llm_ptq.py` 仅按「文件是否已存在」跳过单元，`torch.save` 又不是原子写），可能留下截断的参数文件，且要到第四步 eval 加载时才暴露。因此某个 target 不做分片时，两张卡都不要设置这两个变量。

### 第四步：量化精度评估

同时加载 Attention 与 MoE 两组 PTQ 参数评估量化模型：

```shell
QUANT_DTYPE=int bash scripts/eval_quant.sh
QUANT_DTYPE=mxfp bash scripts/eval_quant.sh
```

## 产物说明

| 产物 | 路径 | 大小 |
|---|---|---|
| PTQ 离线数据（Attention） | `${OUTPUT_DIR}/ptq_data/attn-linear/`（43 个 `block_{i}_attn_in.pkl`） | 21.5 GiB |
| PTQ 离线数据（MoE） | `${OUTPUT_DIR}/ptq_data/moe/`（43 个 `block_{i}_moe_in.pkl`） | 21.5 GiB |
| PTQ 参数（Attention，int/mxfp） | `${OUTPUT_DIR}/ptq_params/{int,mxfp}/attn-linear/`（43 个 `layer_{i}_attn.pt`） | 18 MB / 1.2 GB |
| PTQ 参数（MoE，int/mxfp） | `${OUTPUT_DIR}/ptq_params/{int,mxfp}/moe/`（43×256 个 `layer_{i}_expert_{j}.pt`） | 789 MB / 69 GB |

说明：mxfp 格式的 `lwc` 裁剪系数按 32 元素一组存储（int 为 per-channel），故 mxfp 的 MoE 参数体积远大于 int。两种 target 的离线数据与 PTQ 参数在流程中可先后复用：先提取一种 target 的数据并完成两种格式的 PTQ，再处理另一种 target。

## 结果表

评测环境：2×Atlas A3（64GB HBM）、`seq_len=4096`、WikiText2 test split、Pileval 校准（`nsamples=16`）、`base_lr=1e-3`。权重按前置条件 (b) 在加载期反量化为 bf16 后参与评测（等价于 (a) 预先转换出的 bf16 副本）。

| 模型 | 数据类型 | 算法 | 量化格式 | BF16 PPL | 量化 PPL | PPL 差值 | 量化时长（min） | 离线数据大小（GiB） |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| DeepSeek-V4-Flash | `w4a4` | `lwc` + `lac` | `int4/int4` | 4.146 | 12.700 | 8.554 | 273 | 43.0 |
| DeepSeek-V4-Flash | `w4a4` | `lwc` + `lac` | `mxfp4/mxfp4` | 4.146 | 4.538 | 0.393 | 380 | 43.0 |

指标口径：

- **BF16 PPL**：`eval_mode=bf16` 下的 WikiText2 PPL。
- **量化 PPL**：加载 Attention 与 MoE 的 PTQ 参数后，量化模型的 WikiText2 PPL。
- **PPL 差值**：`量化 PPL - BF16 PPL`。
- **量化时长**：从开始执行 PTQ 到全部量化参数生成完成的时间（Attention 与 MoE 双卡并行执行；不含模型下载和环境构建）。
- **离线数据大小**：`extract_ptq_data` 输出目录的总大小，Attention 与 MoE 数据目录合计。

结果说明：`mxfp4/mxfp4` 在 4bit 激活量化下仍保持极小的精度损失（PPL 差值 0.393）；`int4/int4` 的激活对称量化（16 电平）对该模型 outlier 较重的激活分布较为敏感，PPL 差值 8.554，如对精度有要求建议优先选择 mxfp 格式。
