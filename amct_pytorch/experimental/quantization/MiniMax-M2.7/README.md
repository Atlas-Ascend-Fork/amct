# MiniMax-M2.7 OSPlus SmoothQuant Sample

本示例提供 MiniMax-M2.7 的 **OSPlus SmoothQuant** 两阶段量化流程。与经典 SmoothQuant
仅按单一超参数 `alpha` 平衡激活/权重动态范围不同，OSPlus 将逐通道缩放**重新参数化为
单参数阈值搜索**，并**直接以 W4A4 MXFP4 伪量化后的逐层输出重建误差**作为搜索目标，使搜出
的等价缩放 scale 直接面向部署侧的实际数值格式。

三个阶段分别是：

1. `scripts/run_stage1.sh`（校准搜索阶段）：以 BF16 模型为后端做单批 eager 前向，通过
   forward hook 在两类位置收集校准激活——`input_layernorm` 输出侧（Group 1，供 q/k/v
   投影缩放）与 `self_attn.o_proj` 输入侧（Group 2，供 v→o 路径缩放）——再逐层运行
   OSPlus 阈值搜索，为每层产出两组 SmoothQuant 等价缩放 scale。
2. `scripts/run_stage2_bf16.sh`（融合导出阶段）：读取原始 BF16 检查点与 stage1 产出的
   全部 scale，按等价变换公式逐层在 fp32 下完成融合后再 cast 回 BF16，导出一份与原模型
   同 shape、同 dtype 的 **融合后 BF16 HuggingFace 检查点**（不携带任何 `quantization_config`）。
3. `scripts/run_stage3_mxfp4.sh`（MXFP4 转换阶段）：读取 stage2 的融合后 BF16 检查点，
   对可量化的 Linear 权重做 round-to-nearest（RTN）MXFP4 量化并打包，导出可直接部署的
   **MXFP4 HuggingFace 检查点**（`config.json` 携带 quark 风格的 `quantization_config`；
   Quark 风格说明见文末附录）。

> stage2 的融合后 BF16 检查点在数学上与原模型仅相差 BF16 舍入误差，本身可作为后续量化
> （MXFP4 / W4A8 / W8A8 / INT4 …）的标准 BF16 起点；stage3 即在此基础上完成到 MXFP4 的
> RTN 权重转换与打包。若只想得到融合后的 BF16 模型，运行到 stage2 即可，无需 stage3。

## 目录说明

- `scripts/`：运行入口脚本（stage1 校准搜索、stage2 融合导出、stage3 MXFP4 转换）
- `src/`：stage1/stage2/stage3 核心实现
  - `stage1_calibrate.py`：加载模型 → 一次前向收集激活 → 逐层 OSPlus 搜索 → 落盘 scale
    （仅 channel-wise scale，不含经典 OS+ 的 channel-wise shift）
  - `run_search_from_cache.py`：stage1 「仅记录激活」模式下，基于缓存激活 + safetensors
    权重离线、可并行地补跑 scale 搜索
  - `stage2_export_bf16.py`：读取 scale，在 fp32 下融合并导出融合后的 BF16 检查点
  - `export_rtn_mxfp4.py`：把融合后的 BF16 检查点用 RTN 转换为 Quark 风格 MXFP4 并导出
    （Quark 风格具体信息见文末附录）
  - `mxfp4_fake_quant.py`：OSPlus 搜索目标所用的 MXFP4 伪量化算子
  - `common.py`：MXFP4 导出所需的架构常量与工具函数
- `mxfp4_quantizer/`：与昇腾 MXFP4 反量化路径对齐的纯 torch MXFP4 量化/反量化实现
- `requirements.txt`：Python 依赖

## 环境

推荐容器：`quay.io/ascend/vllm-ascend:v0.18.0rc1-a3`

`torch` / `torch_npu` 建议直接使用容器内自带版本；其余 Python 依赖见 `requirements.txt`：

```bash
cd /workspace/amct/amct_pytorch/experimental/quantization/MiniMax-M2.7
pip install -r requirements.txt
```

建议在容器内运行，并提前把以下目录挂载进去：

- 代码目录：例如挂到 `/workspace/amct`
- BF16 MiniMax-M2.7 模型目录：例如挂到 `/model/MiniMax-M2.7-bf16`
- 校准数据文件：例如挂到 `/data/minimax/calib.json`

下面的示例默认你已经进入本样例目录：

```bash
cd /workspace/amct/amct_pytorch/experimental/quantization/MiniMax-M2.7
```

## 外部输入如何提供

运行本样例时，需要你自行提供以下外部输入。

### `MODEL_DIR`

`MODEL_DIR` 是 MiniMax-M2.7 的 **BF16** 模型目录，供 stage1 和 stage2 共同使用。

- 目录中至少应包含可被 `transformers` 正常加载的模型权重与配置（含 `model.safetensors.index.json`、
  tokenizer 与 `configuration_minimax_m2.py` / `modeling_minimax_m2.py` 等）
- 推荐直接挂载 BF16 模型目录

示例：

```bash
export MODEL_DIR=/model/MiniMax-M2.7-bf16
```

### `CALIB_DATA`

`CALIB_DATA` 是 stage1 使用的校准语料文件，格式为 `jsonl`：脚本逐行读取，优先从每行的
`messages` 字段按对话模板（chat template）渲染为输入；若无 `messages` 而有 `text` 字段，
则直接使用 `text`。单行示例：

```json
{"messages":[{"role":"user","content":"介绍一下量化的基本原理。"}]}
```

## Stage 1：校准与 OSPlus 搜索

stage1 以 BF16 模型为后端做单批 eager 前向，在两类位置挂 hook 收集激活——`input_layernorm`
输出侧（Group 1）与 `self_attn.o_proj` 输入侧（Group 2）——逐层运行 OSPlus 阈值搜索，并
把每层的两组 scale（以及激活缓存）落盘到 `RECORD_DIR`。

该阶段支持两种工作模式（由 `RECORD_ONLY` 控制），以及断点续算（由 `RESUME` 控制）：

- `RECORD_ONLY=1`（脚本默认）：**只记录并缓存激活，跳过搜索**。适合把「一次前向收集激活」
  与「离线并行搜索」解耦；随后由脚本内置的多 worker（`run_search_from_cache.py` + `numactl`
  绑核）在 CPU 上并行补跑搜索。
- `RECORD_ONLY=0`：一次前向收集激活后，**直接在同一进程内**用线程池完成 OSPlus 搜索。

### 启动示例

```bash
MODEL_DIR=/model/MiniMax-M2.7-bf16 \
CALIB_DATA=/data/minimax/calib.json \
RECORD_DIR=$(pwd)/data/record_data \
NUM_CALIB_DATA=512 \
SEQ_LEN=32768 \
BATCH_SIZE=1 \
MAX_TOKENS_PER_LAYER=4096 \
LOAD_DEVICE_MAP=auto \
RECORD_ONLY=0 \
RESUME=1 \
bash scripts/run_stage1.sh
```

若使用「先记录、后并行搜索」的方式，可设置 `RECORD_ONLY=1` 先跑记录，再复用脚本内置的并行
搜索段（该段依据 `NUM_WORKERS` / `THREADS_PER_WORKER` / `NUMA_NODES` / `CORES_PER_NUMA`
进行绑核并行）。

### 产物

stage1 完成后，`RECORD_DIR` 下会生成：

- `layer_{i}_attn_scale.pt`：Group 1 scale，形状 `[hidden_size]`（共 62 层）
- `layer_{i}_oproj_scale.pt`：Group 2 scale，形状 `[num_attn_heads * head_dim]`（已 GQA 折叠，共 62 层）
- `activations/layer_{i}_attn_act.pt` / `activations/layer_{i}_oproj_act.pt`：bf16 激活缓存
- `metadata.json`：本次校准与搜索的配置元信息

### 参数说明

- `MODEL_DIR`：MiniMax-M2.7 BF16 模型目录（**必填**）。
- `CALIB_DATA`：校准语料 `jsonl` 文件路径（**必填**）。
- `RECORD_DIR`：stage1 输出目录，stage2 会直接读取这里的 scale。
- `NUM_CALIB_DATA`：使用多少条校准样本。
- `SEQ_LEN`：校准样本截断长度。
- `BATCH_SIZE`：前向批大小。
- `MAX_TOKENS_PER_LAYER`：单层激活按 token 维保留的上限（尾部截断），用于固定单层激活缓存上限。
- `LOAD_DEVICE_MAP`：模型加载策略，可选 `auto`（按设备自动分配）或 `cpu`。
- `RECORD_ONLY`：`1` 只记录激活跳过搜索；`0` 记录后立即在同进程内搜索。
- `RESUME`：`1` 跳过已存在 scale 文件的层，用于断点续算。
- `NUM_WORKERS` / `THREADS_PER_WORKER` / `NUMA_NODES` / `CORES_PER_NUMA` / `NUM_LAYERS`：
  仅当 `RECORD_ONLY=1` 走并行搜索段时用于控制 CPU 并行度与 NUMA 绑核。

## Stage 2：融合导出 BF16

stage2 不再前向校验，而是读取 stage1 的全部 scale，逐层在 fp32 下完成 SmoothQuant 等价缩放
融合后 cast 回 BF16，导出一份纯 BF16 的 HuggingFace safetensors 检查点。

### 启动示例

```bash
MODEL_DIR=/model/MiniMax-M2.7-bf16 \
RECORD_DIR=$(pwd)/data/record_data \
OUTPUT_DIR=$(pwd)/data/exported/MiniMax-M2.7-osplus_sq_fused_bf16 \
bash scripts/run_stage2_bf16.sh
```

### 产物

stage2 完成后，`OUTPUT_DIR` 下会生成标准 HuggingFace 导出结果：

- `model-00001-of-xxxxx.safetensors` 等分片
- `model.safetensors.index.json`
- `config.json`（**不含** `quantization_config`）
- `generation_config.json`
- tokenizer 与 modeling 相关文件

### 参数说明

- `MODEL_DIR`：源 BF16 模型目录（**必填**）。
- `RECORD_DIR`：stage1 输出的 scale 目录（需含 62 个 `*_attn_scale.pt` 与 62 个 `*_oproj_scale.pt`）。
- `OUTPUT_DIR`：融合后 BF16 模型导出目录，要求该目录为空或事先不存在。
- `MODEL_FILES_DIR`：tokenizer / config / modeling 等随权重导出的辅助文件来源目录，
  默认回退到 `MODEL_DIR`（BF16 模型目录本身即含这些文件）；仅当这些文件不在 `MODEL_DIR`
  时才需显式指定。

### 资源需求

- `MODEL_DIR` 下的 BF16 参考模型（MiniMax-M2.7-bf16 约 427 GB）；
- 约 430 GB 空闲内存（融合期间整份 BF16 模型驻留 CPU 内存）；
- `OUTPUT_DIR` 下约 430 GB 空闲磁盘。

## Stage 3：MXFP4 转换（RTN）

stage3 读取 stage2 导出的融合后 BF16 检查点，对可量化的 Linear 权重做 round-to-nearest
（RTN）MXFP4 量化并打包，导出可直接部署的 MXFP4 HuggingFace 检查点。本步骤**不使用任何
校准数据**；MoE 门控（`*block_sparse_moe.gate*`）与 `lm_head` 等模块保持原 dtype 不量化。

### 启动示例

默认输入为 stage2 的默认输出目录，通常无需显式指定：

```bash
BF16_MODEL_DIR=$(pwd)/data/exported/MiniMax-M2.7-osplus_sq_fused_bf16_hf \
OUTPUT_DIR=$(pwd)/data/exported/MiniMax-M2.7-osplus_sq_mxfp4_hf \
QUANT_DEVICE=auto \
bash scripts/run_stage3_mxfp4.sh
```

也可直接对任意可加载的 BF16 MiniMax-M2.7 检查点使用（例如原始 BF16 模型），只需把
`BF16_MODEL_DIR` 指向对应目录。

### 产物

stage3 完成后，`OUTPUT_DIR` 下会生成 Quark 风格的 MXFP4 导出结果（详见文末附录）：

- `model-00001-of-xxxxx.safetensors` 等分片（可量化 Linear 权重以 `*.weight` 打包 uint8
  FP4 载荷 + `*.weight_scale` uint8 e8m0 scale 存储，其余张量保持原 dtype）
- `model.safetensors.index.json`
- `config.json`（**携带** quark 风格 `quantization_config`）
- tokenizer 与 modeling 相关文件

### 参数说明

- `BF16_MODEL_DIR`：stage3 的输入 BF16 检查点目录，默认取 stage2 的默认输出目录。
- `OUTPUT_DIR`：MXFP4 模型导出目录，要求该目录为空或事先不存在。
- `LOAD_DEVICE_MAP`：模型加载策略，可选 `cpu`（默认）或 `auto`。
- `DEVICE_MAP_FILE`：可选的 device map JSON 文件，用于显式指定模型子模块到设备的映射。
- `QUANT_DEVICE`：执行 MXFP4 量化的设备，`auto` 会自动使用可用的 NPU / CUDA（否则回退 CPU），
  也可写成如 `npu:0,npu:1` 的逗号分隔列表做多卡并行量化。
- `MAX_INFLIGHT_JOBS`：多设备并行量化时的最大在途任务数。
- `ROW_CHUNK_SIZE`：单个权重按行分块量化的块大小，用于控制量化时的峰值内存。

## 运行建议

- 阶段顺序：先 stage1 → stage2 → （可选）stage3；stage2 依赖 stage1 的 scale，stage3 依赖
  stage2 的融合后 BF16 检查点。若只需融合后的 BF16 模型，运行到 stage2 即可。
- `SEQ_LEN`、`NUM_CALIB_DATA`、`BATCH_SIZE` 会显著影响 stage1 的设备内存占用与运行时长，
  应根据实际卡数与内存情况调整。
- 若走 `RECORD_ONLY=1` 的并行搜索段，`NUM_WORKERS` / `THREADS_PER_WORKER` 等应结合宿主机
  NUMA 拓扑设置，避免跨 NUMA 抢占。
- stage3 可用 NPU / CUDA 加速量化（`QUANT_DEVICE`），无加速器时会回退 CPU；若显存/内存吃紧，
  可调小 `ROW_CHUNK_SIZE`。

## OSPlus SmoothQuant 原理

### 1. SmoothQuant 等价变换的基本形式

对任意 Linear 算子 $Y = X W^\top$（$X \in \mathbb{R}^{N \times C_{\text{in}}}$），沿输入通道方向
引入正向量 $s \in \mathbb{R}_{>0}^{C_{\text{in}}}$，可得严格等价的分解：

$$Y = X W^\top = \big(X / s\big)\big(W \odot s\big)^\top,$$

其中 $X / s$ 与 $W \odot s$ 分别表示按通道除以与按通道乘以 $s$。

在 MXFP4 这一 block 量化格式下，由于 block 内元素共享单一 E8M0 指数 scale，单通道的极端幅值
会显著抬高该 block 的 scale 并压低同 block 内其他元素的有效精度。SmoothQuant 的核心观察在于：
激活的逐通道幅度分布远比权重更不均匀，因此通过选取适当大于 1 的 $s$，在不改变模型函数的前提下
**缩小激活、放大权重**，可显著降低激活与权重在 MXFP4 下的合计量化误差。

本项目将上述变换应用于两个位置，二者的插入点、缩放维度与吸收方向如下表所示。

**表：SmoothQuant 等价缩放的两组插入位置**

| 编组 | 插入位置 | 缩放维度 | 吸收方向 | 备注 |
|------|----------|----------|----------|------|
| Group 1 | 介于 `input_layernorm` 与 q/k/v 投影之间 | 隐藏维（`hidden_size = 3072`） | 归一层权重按通道除以 $s$；q、k、v 投影权重按输入通道乘以 $s$ | 对应 Attention QKV 输入侧的离群压制 |
| Group 2 | 介于 v 投影与 o 投影之间 | 注意力查询维（`num_attn_heads × head_dim = 48 × 128`） | v 投影权重在输出通道维除以 KV 头粒度的 $s$；o 投影权重在输入通道维乘以 Q 头粒度的 $s$ | GQA 约束下，同一 KV 头内 6 份重复查询头必须共享同一 scale，否则 v 侧无法以单一向量吸收 |

由于上述全部融合在 fp32 下完成后再回到 BF16，且 Group 2 的 GQA 折叠在搜索阶段即被显式建模
（见下文），整个等价变换在数学上严格成立，仅引入 BF16 表示本身的舍入误差，融合后模型的 logits
应与原模型在 BF16 舍入精度范围内一致。

### 2. OSPlus 在 MXFP4 上的搜索目标

直接套用原始 SmoothQuant 中以 $s_i = \max|X_{:,i}|^{\alpha} / \max|W_{i,:}|^{1-\alpha}$ 形式
参数化的逐通道缩放，仅能针对单一超参数 $\alpha$ 平衡激活与权重的整体动态范围，无法直接最小化
W4A4 MXFP4 配置下的实际重建误差。OSPlus 将该缩放重新参数化为**单参数阈值搜索**：给定阈值
$s_t > 0$，对每个通道 $i$ 按下式构造 scale，

$$s_i(s_t) = \max\!\Big( \max\big(\tfrac{c_{\max,i}}{s_t},\, 1\big),\; \max\big(\tfrac{-c_{\min,i}}{s_t},\, 1\big) \Big),$$

其中 $c_{\max,i}$ 与 $c_{\min,i}$ 为通道 $i$ 在校准激活上的逐通道最大、最小值。该参数化仅对幅值
显著超过 $s_t$ 的通道施加缩放，其余通道保持 $s_i = 1$，整条候选 scale 由单一标量 $s_t$ 决定，
搜索空间因此从 $C_{\text{in}}$ 维退化为**一维**。

在阈值 $s_t$ 上，OSPlus 直接以 MXFP4 量化后的输出重建误差作为目标函数。对当前层的校准激活 $X$
与权重 $W$，定义

$$\mathcal{L}(s_t) = \frac{1}{N}\sum_{n=1}^{N}\Big\|\,\widetilde{Q}_{a}\!\big(X_n / s(s_t)\big)\;\widetilde{Q}_{w}\!\big(W \odot s(s_t)\big)^{\!\top} - X_n W^{\!\top}\Big\|_2^2,$$

其中 $\widetilde{Q}_a$ 与 $\widetilde{Q}_w$ 为沿 reduction 轴执行的 MXFP4 伪量化算子，采用 E2M1
元素编码与 E8M0 共享指数 scale，并以与昇腾 MXFP4 GEMM 反量化路径严格一致的舍入规则实现
（见 `src/mxfp4_fake_quant.py` 与 `mxfp4_quantizer/`）。该损失等价于把 SmoothQuant 等价缩放
与 W4A4 MXFP4 伪量化串联后在层级层面的输出均方误差，因而其最优解 $s^\star$ 直接面向部署侧的
实际数值格式，而非仅平衡激活与权重的统计幅度。

搜索时将 $s_t$ 在区间 $[0.1,\; \max(|c_{\max}|, |c_{\min}|)]$ 上等距采样 200 个候选值并按降序
遍历，记录使 $\mathcal{L}(s_t)$ 取最小值的 $s^\star$。两组 scale 分别由通用搜索器
（`OSPlusMigrator`）与 GQA 约束搜索器（`OSPlusMigratorGQA`）实现：后者在每一步候选 scale 上
额外执行 GQA 折叠 $\mathrm{fold}(s)_{h,r,d} = \max_{r'} s_{h,r',d}$，使得搜索过程在评估
$\mathcal{L}(s_t)$ 时已经满足「同一 KV 头内 6 份重复共享 scale」的约束，从而避免出现搜索得到、
但融合阶段无法吸收的非法 scale。

### 3. 校准搜索阶段的实现（Stage 1）

校准搜索阶段以 BF16 模型为后端，按对话模板渲染校准语料为输入序列后做单批 eager 前向，并在前向
过程中通过钩子在两类位置上同时收集激活样本：在 `input_layernorm` 的输出侧收集 Group 1 激活，
在 o 投影的输入侧收集 Group 2 激活。由于 OSPlus 搜索只依赖激活的通道级幅值分布，每层按 token
维度对收集结果做尾部截断（`MAX_TOKENS_PER_LAYER`），既保证逐通道极值估计的稳定性，也把单层
激活缓存上限固定。

全部层的激活在同一次前向中并行收集，避免重复加载模型；其后逐层调用 OSPlus 搜索器，在工作设备上
以信号量约束并发度，最终为每一层产出两份 scale 张量并保存于 `RECORD_DIR`。该阶段支持断点续算
（`RESUME`）与「仅记录激活、跳过搜索」（`RECORD_ONLY`）两种工作模式，分别用于失败重试与离线
复用激活缓存并行搜索。

### 4. 融合导出阶段的实现（Stage 2）

融合导出阶段读取原始 BF16 检查点与上一子阶段产出的全部 scale 张量（共 62 层 × 2 组），按第 1 节
的等价变换公式逐层在 fp32 中完成融合后再 cast 回 BF16，最终输出一份与原模型同 shape、同 dtype 的
BF16 HuggingFace 检查点。

其中 v 投影同时承担 Group 1 与 Group 2 两组缩放的吸收：先沿输入通道维乘以 Group 1 的 scale，
再沿输出通道维除以由 Group 2 scale 在 GQA 维度上每组首份元素构成的 KV 头粒度向量；o 投影则沿
输入通道维乘以 Q 头粒度的完整 Group 2 scale。GQA 折叠保证了上述 KV 头粒度向量与 Q 头粒度完整
向量在每组 6 份重复上严格相等，从而上述两步吸收互不冲突。

融合阶段的全部算术在 fp32 下完成后再 cast 回 BF16，以避免 BF16 累计舍入误差污染等价变换的精度。
最终导出的检查点不携带任何 `quantization_config`，对外行为与原 BF16 模型在 BF16 舍入精度范围内
一致，可被后续量化感知训练与 MSE 引导 MXFP4 权重转换两阶段直接当作新的 BF16 起点接续使用，
无需任何特殊加载逻辑。这一在权重检查点层面完成的解耦使本阶段可作为后续阶段的标准 BF16 输入，
同时保留独立运行与回滚能力。

## 阶段性结果

经过约 4906 个 token 的激活记录与 scale 搜索之后，OSPlus SmoothQuant 模型在 W4A4 的阶段性结果
如下表所示：

| 方法 | HumanEval+ | GSM8K | MATH500 | LongBench v2 | GPQA Diamond |
|------|-----------|-------|---------|--------------|--------------|
| OSPlus SmoothQuant | 89.02 | 95.20 | 91.00 | 51.29 | 82.58 |
| M2.7 baseline（BF16） | 91.60 | 95.55 | 91.26 | 54.87 | 88.88 |

上述精度评测使用的推理框架为 [cann-recipes-infer / minimax_m2.5_mxfp4](https://gitcode.com/cann/cann-recipes-infer/tree/master/integration/vllm/minimax_m2.5_mxfp4)，并额外开启了
KV cache 的 MXFP4 量化：对 KV 做 token-wise、group size 32 的 MXFP4 量化；前 32 个 token 的 KV
保持 BF16 精度，之后 token 的 KV 再量化为 MXFP4；同时对 Q 与 K cache 做 block=32 的 Hadamard
旋转，以降低量化带来的误差。

> 说明：上述为 OSPlus SmoothQuant 等价缩放 + W4A4 MXFP4 的阶段性精度，评测受采样温度、随机性
> 与推理后端实现差异等因素影响，复现时同一配置下数值可能存在小幅波动，建议以多次取均值或在相同
> 框架版本下复现为准。

## 附录：Quark 风格 MXFP4 与常见 HF MXFP4 存储差异

「Quark 风格」指的是检查点落盘约定（张量命名 + `quantization_config`），不是另一套数值编码。
底层仍是 OCP MXFP4：E2M1 元素、每 32 个共享一个 E8M0 scale、两枚 FP4 打成一个 `uint8`。

与常见 HF / GPT-OSS MXFP4（`quant_method: mxfp4`）的差异主要如下：

| | 本样例（Quark 风格） | 常见 HF / GPT-OSS MXFP4 |
|---|---|---|
| 权重张量名 | 仍叫 `*.weight`（packed uint8） | `*_blocks`（packed uint8） |
| scale 张量名 | 并列 `*.weight_scale`（uint8 e8m0） | 并列 `*_scales` |
| `quant_method` | `"quark"` | `"mxfp4"` |
| config 结构 | Quark 嵌套 schema（`global_quant_config`、`export.pack_method=reorder`、`weight_format=real_quantized`、`scale_format=e8m0` 等） | 较简，多为 `modules_to_not_convert` + `quant_method` |
| 消费方 | Quark / 对齐 Quark 的推理路径（如 cann-recipes-infer MiniMax MXFP4） | Transformers MXFP4 / GPT-OSS 加载路径 |

一句话：数值格式一样，差在「键名叫什么、config 长什么样、给谁加载」；本 stage3 刻意对齐 AMD Quark 的 HF 导出，所以称为 Quark 风格。
