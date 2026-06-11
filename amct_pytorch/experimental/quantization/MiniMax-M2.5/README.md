# MiniMax-M2.5 SmoothQuant MXFP4 Sample

本示例提供 MiniMax-M2.5 的两阶段 SmoothQuant + MXFP4 量化流程：

1. `scripts/run_vllm_stage1.sh`：在 vLLM Ascend 路径上拉起 MiniMax-M2.5，记录 SmoothQuant 所需激活统计。
2. `scripts/run_stage2.sh`：读取 stage1 激活，执行 SmoothQuant 融合，并导出 MXFP4 HuggingFace safetensors。

## 目录说明

- `scripts/`：运行入口脚本
- `src/`：stage1/stage2 核心实现
- `mxfp4_quantizer/`：MXFP4 量化与打包逻辑
- `patches/`：MiniMax 在 Ascend vLLM 上运行所需 patch

## 环境

推荐容器：`quay.io/ascend/vllm-ascend:v0.14.0rc1-a3`

容器中已包含以下 vLLM 相关组件（以 editable 模式安装在 `/vllm-workspace/` 下）：

| 组件 | 仓库 | Commit |
|------|------|--------|
| vLLM | https://github.com/vllm-project/vllm.git | `d7de043d55d1dd629554467e23874097e1c48993` |
| vLLM-Ascend | https://github.com/vllm-project/vllm-ascend | `52d4acfa51fb868823d1070b81cbd2d97e9e4696` |

建议在容器内运行，并提前把以下目录挂载进去：

- 代码目录：例如挂到 `/workspace/amct`
- BF16 或 FP8 MiniMax-M2.5 模型目录：例如挂到 `/model/MiniMax/MiniMax-M2.5-bf16`
- 校验数据目录：例如挂到 `/data/minimax/data.jsonl`

下面的示例默认你已经进入仓库目录：

```bash
cd /workspace/amct/amct_pytorch/experimental/quantization/MiniMax-M2.5
```

## 外部输入如何提供

运行本样例时，需要你自行提供以下外部输入。

### `MODEL_DIR`

`MODEL_DIR` 是 MiniMax-M2.5 的bf16的模型目录，供 stage1 和 stage2 共同使用。

- 目录中至少应包含可被 `transformers` / `vllm` 正常加载的模型权重与配置
- 推荐直接挂载 BF16 模型目录
- 如果使用 FP8 模型，需确认当前 vLLM patch 和运行环境支持对应加载路径

示例：

```bash
export MODEL_DIR=/model/MiniMax/MiniMax-M2.5-bf16
```

### `CALIB_DATA`

`CALIB_DATA` 是 stage1 使用的校验集文件，格式为 `jsonl`。脚本会逐行读取，并从每一行中提取 `messages` 字段，然后调用 tokenizer 的 chat template 构造输入。

单行示例：

```json
{"messages":[{"role":"user","content":"介绍一下量化的基本原理。"}]}
```

示例：

```bash
export CALIB_DATA=/data/minimax/data.jsonl
```

### `VLLM_REPO_DIR`

`VLLM_REPO_DIR` 是 Ascend vLLM 源码目录，仅 stage1 需要。`scripts/run_vllm_stage1.sh` 会检查并尝试应用本目录下 `patches/` 中的 MiniMax patch。

示例：

```bash
export VLLM_REPO_DIR=/vllm-workspace/vllm
```

### `VLLM_PATCH_PATH`

`VLLM_PATCH_PATH` 默认指向当前目录下的 patch 文件，一般不需要手动修改；只有当你使用了其他 patch 文件时才需要覆盖。

示例：

```bash
export VLLM_PATCH_PATH=$(pwd)/patches/0001-MiniMax-M2-adapt-Ascend-fp8-loading-and-qk-norm-path.patch
```

## Stage1：记录激活

stage1 会在真实的 vLLM Ascend 执行路径上启动 MiniMax-M2.5，给注意力相关模块挂 hook，统计 SmoothQuant 所需激活最大值，并把结果落盘到 `RECORD_DIR`。

### 启动示例

```bash
MODEL_DIR=/model/MiniMax/MiniMax-M2.5-bf16 \
CALIB_DATA=/data/minimax/data.jsonl \
RECORD_DIR=$(pwd)/record_data_vllm \
NUM_CALIB_DATA=2048 \
SEQ_LEN=32768 \
TP_SIZE=16 \
GPU_MEMORY_UTILIZATION=0.9 \
VLLM_REPO_DIR=/vllm-workspace/vllm \
VLLM_PATCH_PATH=$(pwd)/patches/0001-MiniMax-M2-adapt-Ascend-fp8-loading-and-qk-norm-path.patch \
ENABLE_EXPERT_PARALLEL=1 \
VLLM_MAX_NUM_SEQS=32 \
VLLM_MAX_NUM_BATCHED_TOKENS=32768 \
VLLM_ASCEND_ENABLE_FLASHCOMM1=1 \
VLLM_ENFORCE_EAGER=1 \
VLLM_COMPILATION_CONFIG='{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
bash scripts/run_vllm_stage1.sh
```

### 产物

stage1 完成后，`RECORD_DIR` 下会生成每层的激活统计文件，例如：

- `layers_0_self_attn_q_proj.pt`
- `layers_0_self_attn_k_proj.pt`
- `layers_0_self_attn_v_proj.pt`
- `layers_0_self_attn_o_proj.pt`
- `metadata.json`

### 参数说明

- `MODEL_DIR`：MiniMax-M2.5 原始模型目录。
- `CALIB_DATA`：校验集 `jsonl` 文件路径，每行需要有 `messages` 字段。
- `RECORD_DIR`：stage1 激活统计输出目录，stage2 会直接读取这里的 `.pt` 文件。
- `NUM_CALIB_DATA`：使用多少条校验样本进行统计。
- `SEQ_LEN`：校验样本截断长度，同时决定 vLLM 的最大上下文长度设置。
- `TP_SIZE`：`torchrun` 拉起的并行卡数，对应张量并行大小。
- `GPU_MEMORY_UTILIZATION`：vLLM 可使用的设备内存比例。
- `VLLM_REPO_DIR`：vLLM 源码目录。
- `VLLM_PATCH_PATH`：MiniMax Ascend 适配 patch 路径。
- `ENABLE_EXPERT_PARALLEL`：是否启用 expert parallel，`1` 表示开启，`0` 表示关闭。
- `VLLM_MAX_NUM_SEQS`：vLLM 单批次允许的最大请求数。
- `VLLM_MAX_NUM_BATCHED_TOKENS`：vLLM 单批次允许的最大 token 数。
- `VLLM_ASCEND_ENABLE_FLASHCOMM1`：是否开启 Ascend FlashComm1 相关能力。
- `VLLM_ENFORCE_EAGER`：是否强制 eager 执行，`1` 为开启，`0` 为关闭。
- `VLLM_COMPILATION_CONFIG`：传给 vLLM 的编译配置，使用 JSON 字符串表示。

## Stage2：SmoothQuant + MXFP4 导出

stage2 不再进行前向校验，而是直接读取 stage1 的激活统计，计算 SmoothQuant scale，将 scale 融合进相关参数，然后把可量化权重导出为 MXFP4 格式。

### 启动示例

```bash
MODEL_DIR=/model/MiniMax/MiniMax-M2.5-bf16 \
RECORD_DIR=$(pwd)/record_data_vllm \
OUTPUT_DIR=$(pwd)/exported_model_vllm \
LOAD_DEVICE_MAP=auto \
ALPHA=0.8 \
bash scripts/run_stage2.sh
```

如果你希望显式指定 device map 文件，可以这样启动：

```bash
MODEL_DIR=/model/MiniMax/MiniMax-M2.5-bf16 \
RECORD_DIR=$(pwd)/record_data_vllm \
OUTPUT_DIR=$(pwd)/exported_model_vllm \
LOAD_DEVICE_MAP=auto \
DEVICE_MAP_FILE=/path/to/device_map.json \
ALPHA=0.8 \
bash scripts/run_stage2.sh
```

### 产物

stage2 完成后，`OUTPUT_DIR` 下会生成 HuggingFace safetensors 导出结果，包括：

- `model-00001-of-xxxxx.safetensors`
- `model.safetensors.index.json`
- `config.json`
- `generation_config.json`
- tokenizer 相关文件

### 参数说明

- `MODEL_DIR`：MiniMax-M2.5 bf16模型目录。
- `RECORD_DIR`：stage1 输出的激活统计目录。
- `OUTPUT_DIR`：量化后 MXFP4 模型导出目录。要求该目录为空，或事先不存在。
- `LOAD_DEVICE_MAP`：模型加载策略，可选 `auto` 或 `cpu`。`auto` 会按设备自动分配，`cpu` 则先加载到 CPU。
- `DEVICE_MAP_FILE`：可选的 device map JSON 文件，用于显式指定模型子模块到设备的映射关系。
- `ALPHA`：SmoothQuant 的平衡系数，取值范围为 `[0, 1]`。

## 运行建议

- 先完成 stage1，再运行 stage2；stage2 依赖 stage1 输出的激活统计文件。
- `SEQ_LEN`、`NUM_CALIB_DATA`、`TP_SIZE` 会显著影响 stage1 的显存占用和运行时长，应根据实际卡数和显存情况调整。
- 若 stage1 启动时报 patch 不匹配，需要先确认 `VLLM_REPO_DIR` 指向的 vLLM 版本与 `patches/` 中 patch 兼容。

## SmoothQuant 原理

SmoothQuant 的核心思想是：将激活中难以量化的异常值（outlier）通过数学等价变换"平滑"到权重中，使激活和权重都更易于低比特量化。

### 基本公式

对于线性层 $Y = XW$，引入逐通道缩放因子 $s$，等价变换为：

$$Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W)$$

即：激活除以 $s$，权重乘以 $s$。由于线性运算的结合律，最终输出不变。

缩放因子 $s$ 的计算方式为：

$$s_j = \frac{\max(|X_j|)^{\alpha}}{\max(|W_j|)^{1-\alpha}}$$

其中 $\alpha \in [0, 1]$ 控制"难度"在激活与权重之间的分配：
- $\alpha$ 越大，越多的量化难度从激活转移到权重
- $\alpha$ 越小，权重保持原样，激活自行承担量化误差

### Stage2 中 scale 的融合方式

在推理部署场景下，我们不在运行时对激活做 $\text{diag}(s)^{-1}$ 缩放（否则引入额外算子开销）。因此 stage2 将 $s$ **融合到前一个 module 的权重中**，利用网络结构的天然连接关系消除运行时开销：

#### Group 1：LayerNorm → QKV Projection

```
input_layernorm.weight  ÷=  s
q_proj.weight           *=  s     (沿 input_channel 维度)
k_proj.weight           *=  s
v_proj.weight           *=  s
```

由于 LayerNorm 的输出直接作为 QKV 的输入，`layernorm.weight /= s` 等效于对激活执行了 `/= s`，而 QKV 权重 `*= s` 则补偿了这个缩放。两者合在一起数学等价于原始计算，但激活的数值范围被平滑了。

$s$ 的计算中，`act_scale` 取自 stage1 记录的 `q_proj` 输入激活的逐通道最大绝对值，`weight_scale` 取 q/k/v 三个权重矩阵逐 input channel 最大绝对值的联合最大值。

#### Group 2：V Projection → O Projection

```
v_proj.weight  ÷=  s     (沿 output_channel 维度)
v_proj.bias    ÷=  s     (如有)
o_proj.weight  *=  s     (沿 input_channel 维度)
```

`v_proj` 的输出是 `o_proj` 的输入（经 attention 计算后），因此对 `v_proj` 输出除以 $s$ 等效于对 `o_proj` 输入除以 $s$。具体实现上，将 $s$ 融合进 `v_proj.weight` 的输出维度和 `o_proj.weight` 的输入维度。

对于 GQA（Grouped Query Attention），由于 KV head 数量少于 Q head 数量，需要将 $s$ 按 head group 取 max 后再分别应用到 v_proj 和 o_proj，确保同一 KV head 对应的多个 Q head 共享相同的缩放因子。

### 总结

通过上述两组融合，stage2 实现了"零运行时开销"的 SmoothQuant：所有缩放操作都在离线阶段融入了现有权重，推理时无需额外算子。融合后的权重再经过 MXFP4 量化打包，最终导出为可直接部署的 safetensors 格式。

## 量化损失测试结果

为评估 SmoothQuant + MXFP4 量化方案对模型精度的影响，我们在多个主流评测集上对比了 BF16 基线（Baseline）与 PTQ 量化模型（PTQ）的表现，并给出 PTQ 相对 Baseline 的精度保持比例（比例 = PTQ / Baseline）。

| Task     | drop   | gpqa_diamond | gsm8k  | humaneval+ | math500 | lcb    | longbenchV2 |
|----------|--------|--------------|--------|------------|---------|--------|-------------|
| Baseline | 90.478 | 85.15        | 96.01  | 91.33      | 93.06   | 62.8   | 56.76       |
| PTQ      | 90.04  | 84.64        | 95.89  | 91.21      | 92.92   | 63.83  | 55.09       |
| 比例     | 0.99   | 0.99         | 0.99   | 0.99       | 0.99    | 1.0    | 0.97        |

### 说明

- 上述精度评测使用的推理框架为 [cann-recipes-infer / minimax_m2.5_mxfp4](https://gitcode.com/cann/cann-recipes-infer/tree/master/contrib/minimax_m2.5_mxfp4)，该框架在昇腾 NPU 上对 MiniMax-M2.5 的 MXFP4 量化推理进行**模拟实现**（即按 MXFP4 的量化/反量化规则在高精度算子上等价计算，并非真正调用 MXFP4 低比特硬件算子），用于在端到端推理流程中评估量化方案的精度表现。
- 从结果可以看出，PTQ 量化模型在 drop、gpqa_diamond、gsm8k、humaneval+、math500 等任务上的精度保持比例均达到 **0.99** 及以上，lcb 任务上 PTQ 略优于 Baseline（比例 1.0），longbenchV2 任务上比例为 **0.97**，整体精度损失可控。
- 评测过程中受采样温度、随机性、推理后端实现差异等因素影响，**部分结果可能存在一定波动**，复现时同一配置下数值可能与上表存在小幅差异，建议以多次取均值或在相同框架版本下复现为准。
