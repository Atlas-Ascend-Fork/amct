# QMM 理论模型

这份文档定义 Linear `dynamic quant + quant matmul` 的粗粒度理论估算模型。目标不是复现真实 runtime，而是用主导时间项估算收益方向、speedup、瓶颈和置信度。

## 1. 基本原则

- 搬运理论耗时：`bytes / theoretical_path_rate`
- 计算理论耗时：`work / throughput`
- 只保留主导时间项，不模拟完整 kernel timeline
- `dynamic quant` 和 qmm 分开建模
- baseline 使用默认 tile 参数，不让用户输入 `baseM/baseN`
- qmm 默认使用 ops-style 默认候选集，不做开放式 tile search
- 理论估算不能替代真实 runtime 验证

默认数据流：

```text
Baseline:
X(fp/bf16) -> float/bf16 matmul with W(fp/bf16) -> Y

INT Target:
X(fp/bf16)
-> dynamic quant writes X_q + x_scale/meta to GM
-> qmm reads X_q + x_scale/meta + W_q + weight scale/meta
-> qmm cube
-> output path: FixPipe result handling + INT antiquant/rescale + final MTE3 write -> Y

MX Target:
X(fp/bf16)
-> dynamic quant writes X_mx + block scale/meta to GM
-> qmm reads X_mx + block scale/meta + W_mx + block scale/meta
-> qmm cube
-> output path: FixPipe result handling + final MTE3 write -> Y
```

`dynamic quant` 负责 activation 量化。量化产物和 scale/meta 的写出计入 `T_dynamic_quant`，qmm 读入计入 `T_qmm_gm_to_l1` 或 `T_qmm_l1_to_l0`，不建立独立交接时间项。

## 2. 估算流程

### 第一步：确认估算口径

必选/可选输入和来源以 `SKILL.md` 的 `输入与来源` 为准。本文件只定义这些输入进入公式后的计算方式。

`m/k/n` 对应 Linear `X[M,K] @ W[K,N] -> Y[M,N]`。

常用默认值：

```text
baseline_dtype = bf16
output_dtype = bf16 or fp16, depending on caller contract
weight_preparation = default_api_prepared or offline_prepared
T_weight_transform = 0 when weight is prepared offline
a5_bandwidth_variant = high_bandwidth unless user specifies low_bandwidth
baseline_default_baseM = 128
baseline_default_baseN = 256
mx_default_block_size = 32 elements/block
mx_default_scale_dtype_size = 1 byte/block scale
default_activation_granularity = PER_TOKEN or PER_ROW
default_weight_granularity = PER_CHANNEL
```

dtype byte size 由量化格式决定：

```text
bytes(X_fp) = M * K * size(baseline_dtype)
bytes(W_fp) = N * K * size(baseline_dtype)
bytes(Y) = M * N * size(output_dtype)
bytes(X_q) = M * K * size(activation_quant_dtype)
bytes(W_q) = N * K * size(weight_quant_dtype)
bytes(scale/meta) = count(scale/meta elements) * size(scale/meta dtype)
```

量化粒度直接决定 scale/meta 数量：

```text
activation PER_TENSOR: activation_scale_count = 1
activation PER_TOKEN or PER_ROW: activation_scale_count = M
weight PER_TENSOR: weight_scale_count = 1
weight PER_CHANNEL: weight_scale_count = N
PER_BLOCK: scale_count = ceil(elements / block_size)
MX: scale_count = ceil(elements / 32)
```

### 第二步：准备平台计算和搬运能力

从 `references/platform-assumptions.md` 选择平台 bucket，并换算理论路径速率和理论吞吐。

| 参数 | 来源 | 策略 |
| --- | --- | --- |
| `platform_cube_core_count` | `platform-assumptions.md` | 按 `A2/A3` 或 `A5` 固定取值 |
| `cube_frequency` | `platform-assumptions.md` | 固定取值 |
| `gm_rate_per_core` | `platform-assumptions.md` | 只作为 theoretical path rate |
| `l2_rate_per_core` | `platform-assumptions.md` | 只作为 theoretical path rate |
| `mac_per_cycle_per_core(dtype)` | `platform-assumptions.md` | 按 dtype 取 16-bit、8-bit、4-bit 或 mx/fp8 显式假设 |
| `vector_ops_per_cycle_per_core(dtype)` | `platform-assumptions.md` | 按 vector 数据位宽选择 |
| `used_core_num` | qmm base candidate 估算 | 大 shape 可先按满核乐观估计，最终需由 base candidate 修正 |
| `active_core_count` | 公式计算 | `min(used_core_num, platform_cube_core_count)` |
| fixed overhead | 校准值或显式假设 | 无校准时用 `0` optimistic lower-bound，并降低 confidence |

```text
active_core_count = min(used_core_num, platform_cube_core_count)

theoretical_gm_path_rate
= gm_rate_per_core * active_core_count * cube_frequency

theoretical_l2_path_rate
= l2_rate_per_core * active_core_count * cube_frequency

cube_throughput(dtype)
= mac_per_cycle_per_core(dtype) * active_core_count * cube_frequency

vector_throughput(dtype)
= vector_ops_per_cycle_per_core(dtype) * active_core_count * cube_frequency
```

使用规则：

- baseline `bf16` 使用 16-bit cube throughput
- `A2/A3` 量化格式能力：`INT8`、`INT4`
- `A5` 量化格式能力：`INT8`、`FP8`、`MXFP8`、`MXFP6`、`MXFP4`
- `A5` 默认使用 high-bandwidth `gm_rate_per_core = 80 B/cycle`；用户指定 low-bandwidth A5 时改用 `31 B/cycle`
- `theoretical_path_rate` 是理论路径速率，不是真实有效带宽

### 第三步：估算 Baseline 时间

baseline 只估算 float/bf16 matmul 路径。

```text
MACs = M * K * N
T_base_cube = MACs / cube_throughput(baseline_dtype)
```

baseline MTE2 输入搬运采用默认 tile 冷/热数据模型。默认值固定在模型内部，不作为用户输入：

```text
baseM = 128
baseN = 256
dataType = size(baseline_dtype)

cold_hbm_bytes
= 1 * M * K * dataType
+ 4 * K * N * dataType

hot_l2_bytes
= max((N / baseN) - 1, 0) * M * K * dataType
+ max((M / baseM) - 4, 0) * K * N * dataType

T_base_mte2
= hot_l2_bytes / theoretical_l2_path_rate
+ cold_hbm_bytes / theoretical_gm_path_rate
```

MTE3 结果写回：

```text
T_base_mte3 = bytes(Y) / theoretical_gm_path_rate
```

组合方式：

```text
T_baseline
= max(T_base_cube, T_base_mte2, T_base_mte3)
+ T_base_fixed
```

这个组合表达的是：baseline matmul 的 cube compute、MTE2 input movement 和 MTE3 result write 可流水重叠，主导项决定理论耗时。

### 第四步：估算 Dynamic Quant 时间

```text
T_dq_move_in = bytes(X_fp read) / theoretical_gm_path_rate

T_dq_move_out
= bytes(X_quant + activation scale/meta written to GM)
/ theoretical_gm_path_rate

bytes(activation scale/meta written)
= activation_scale_count * scale_dtype_size

dq_vector_ops
= elements * ops_per_element
+ reduction_ops_for_scale
+ format_convert_ops

T_dq_vector = dq_vector_ops / vector_throughput(quant_vector_dtype)

T_dynamic_quant
= max(T_dq_move_in, T_dq_move_out, T_dq_vector)
+ T_dq_fixed
```

如果已有稳定经验口径，也可以写成：

```text
T_dynamic_quant ~= T_dq_vector / 0.7 + T_dq_fixed
```

`T_dq_fixed` 不是平台配置字段。无 calibration 时使用 `0` 做 optimistic lower-bound，并在报告中降低 confidence。

### 第五步：估算 QMM 默认候选集

qmm 默认使用 ops-style 默认候选集，并选择最小 target candidate。默认候选集不是开放式 tile search。

默认候选集生成规则：

- 遍历 4 种 transpose interpretation：`NN`、`NT`、`TN`、`TT`
- 每种 transpose interpretation 最多生成 2 个 base candidate：`basic_base` 和 `small_shape_base`
- `basic_base` 按 shape、dtype bytes、对齐和量化粒度生成
- `small_shape_base` 只在 small shape 或用核不足时尝试，用于改善 used core 和 tail
- 使用 ASW-style sliding-window 估算 used core、tail split 和 L2 hit approximation
- 对合法 candidate 计算 `T_qmm_candidate`，最后取最小值

每个候选只用于估算：

- transpose interpretation
- base tile
- used core
- tail split
- 冷/热数据比例
- `T_qmm_fixed`
- `T_qmm_gm_to_l1`
- `T_qmm_l1_to_l0`
- `T_qmm_cube`
- `T_qmm_fixpipe`
- `T_qmm_antiquant_rescale`
- `T_qmm_mte3`

#### 5.1 base candidate 和冷热数据

base candidate 选择需要兼顾对齐、量化粒度和本地存储容量，但默认只使用有限候选集：

- 普通 qmm 路径的 `basic_base` 以 `min(M, 256)`、`min(N, 256)` 为起点，再按 transpose interpretation 和 dtype bytes 对齐
- `baseK` 以 `min(256 / dtype_bytes, K)` 为起点，再按 reduce block 对齐
- `PER_BLOCK` 类路径使用 block-size 约束生成 base candidate
- small shape 可额外生成 `small_shape_base`，用于提高有效用核数并降低尾块浪费

L2 reuse 只是一种理论复用提示，不应解释为真实 cache counter。报告应记录被选中的 `transpose_interpretation`、`base_candidate`、`used_core_num` 和 tail summary。

#### 5.2 qmm 输入搬运

```text
T_qmm_gm_to_l1
= qmm_cold_gm_bytes / theoretical_gm_path_rate
+ qmm_hot_l2_bytes / theoretical_l2_path_rate

qmm_scale_meta_bytes
= activation_scale_bytes + weight_scale_bytes

T_qmm_l1_to_l0
= l1_to_l0_bytes / local_path_rate_or_documented_assumption

T_qmm_input_movement
= T_qmm_gm_to_l1
+ T_qmm_l1_to_l0
+ T_weight_transform
```

`T_weight_transform` 只在 runtime 做 weight transform 或 pack 时计入。权重离线准备时默认取 `0`，但仍要说明权重和 metadata 的搬运是否计入。

#### 5.3 qmm cube 计算

```text
T_qmm_cube
= repeated_groups
 * align(M, 16) * align(N, 16) * align(K, C0)
 / effective_cube_parallelism
```

快速粗估时可写为：

```text
T_qmm_cube ~= MACs / cube_throughput(qmm_dtype)
```

#### 5.4 FixPipe、antiquant 和 MTE3

```text
T_qmm_fixpipe = fixpipe_bytes_or_ops / fixpipe_rate_or_documented_assumption

T_qmm_antiquant_rescale
= antiquant_or_rescale_ops / vector_throughput(antiquant_rescale_vector_dtype)

T_qmm_mte3 = bytes(Y) / theoretical_gm_path_rate
```

流程口径：

- INT qmm：`FixPipe result handling + INT antiquant/rescale + final MTE3 write`
- MX qmm：`FixPipe result handling + final MTE3 write`

MX 路径如果明确存在额外 convert 或 antiquant/rescale vector work，应显式加入 `T_qmm_antiquant_rescale`，否则默认取 `0`。

#### 5.5 qmm candidate 时间

```text
T_qmm_candidate
= T_qmm_fixed
+ max(T_qmm_cube, T_qmm_input_movement)
+ T_qmm_fixpipe
+ T_qmm_antiquant_rescale
+ T_qmm_mte3
```

`T_qmm_fixed` 不是平台配置字段。它只能来自校准值、显式输入或文档化默认区间；缺失时应降低 confidence。

### 第六步：汇总 Target 时间

```text
T_target
= T_dynamic_quant
+ min(T_qmm_candidate candidates)
```

### 第七步：输出收益结论

```text
estimated_speedup = T_baseline / T_target
```

主瓶颈优先从下面几类中选：

- `dynamic quant`
- `qmm input movement`
- `qmm cube`
- `qmm FixPipe`
- `qmm antiquant/rescale`
- `qmm MTE3`
- `fixed overhead`
- `tiling inefficiency`

confidence 只由理论输入完整度和关键假设决定，不由 runtime validation 状态决定。runtime validation 应作为单独的 `validation_status` 或验证覆盖度记录。

- `High`: `m/k/n`、量化方案、平台、dtype、baseline/target 语义都明确
- `Medium`: 部分 scale/meta、weight preparation、candidate policy 或 A5 bandwidth variant 依赖默认假设
- `Low`: 缺少平台参数、bit-width throughput mapping、scale/meta 口径或量化粒度，只能做方向性估计

## 3. 权重准备

默认理论估算不显式建模具体权重存储 layout。使用 `weight_preparation = default_api_prepared` 或 `offline_prepared` 表示权重已按调用方和 qmm API 所需方式准备好。

理论候选搜索使用 ops-style 默认候选集估算 transpose interpretation、base candidate、对齐、L2 reuse 和有效用核数，不搜索具体权重 layout。如果部署路径 runtime 做 weight transform 或 pack，则把 `T_weight_transform` 显式计入 target；否则默认取 `0`。

## 4. 量化格式规则

| 格式 | Value bytes | Scale/meta 默认 | 主要影响 |
| --- | ---: | --- | --- |
| `INT8` | `1` | 按量化粒度显式计入 | `T_dynamic_quant`、`T_qmm_gm_to_l1`、`T_qmm_cube`、`T_qmm_antiquant_rescale` |
| `INT4` | `0.5` | 按量化粒度显式计入 | `T_dynamic_quant`、`T_qmm_gm_to_l1`、`T_qmm_cube`、packing/alignment |
| `FP8` | `1` | 按平台格式假设 | A5 8-bit cube-like throughput assumption |
| `MXFP8` | `1` | per block, default `32` elements/block, `1` byte scale/block | block scale movement, tile compatibility |
| `MXFP6` | `0.75` | per block, default `32` elements/block, `1` byte scale/block | block scale movement, packed bytes, tile compatibility |
| `MXFP4` | `0.5` | per block, default `32` elements/block, `1` byte scale/block | block scale movement, packed bytes, tile compatibility |
MX 是 per-block/block-scaled 格式。默认按 `32 elements/block` 估算 scale/meta：

```text
mx_block_count = ceil(elements / 32)
mx_scale_bytes = mx_block_count * 1 byte
```

MX activation scale/meta 写出计入 `T_dynamic_quant`；qmm 读入 activation/weight block scale 计入 `T_qmm_gm_to_l1` 和本地存储压力。MX qmm 不默认计入 INT 路径的额外 antiquant/rescale 后处理。

## 5. 约束规则

- 不把这个模型当作 runtime 验证替代品
- 不从当前仓库的 recommendation 阈值反向约束理论结论
- 不擅自引入当前理论定义之外的复杂流水假设
- 不把固定开销写成平台配置字段
- 不在缺少 bit-width throughput mapping、scale/meta 或 `quant_granularity` 时输出伪精确结论
