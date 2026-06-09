# Dynamic Quant + QMM 计算示例

这份文档保存 `$dynamic-quant-qmm-analysis` 的手工计算示例。它用于解释 estimator 的计算过程，不是实测结果登记表。

## 1. 范围

- 当前 int 示例：`A2/A3 + int_w8a8`
- 当前 mx 示例：`A5 + mx_w8a8`
- 当前 baseline 默认 tile：`baseM=128`、`baseN=256`
- 当前 weight preparation：`offline_prepared`
- 当前 qmm candidate policy：`ops-style 默认候选集`
- 当前 fixed overhead：示例使用显式默认假设，不来自平台配置

## 2. A2-A3 大 Shape 示例

示例 shape：`M=4096,K=4096,N=12800`。

```text
X[M,K] @ W[K,N] -> Y[M,N]
```

估算口径：`A2/A3 + int_w8a8 + int8 activation + int8 weight`。权重默认离线量化，因此 target 不计 weight `DynamicQuant`。

### 2.1 平台理论参数

```text
cube cores = 24
frequency = 1.8 GHz
GM path rate = 32 B/cycle/core * 24 * 1.8 GHz = 1382.4 GB/s
L2 path rate = 110 B/cycle/core * 24 * 1.8 GHz = 4752 GB/s
bf16 cube throughput = 4096 MAC/cycle/core * 24 * 1.8 GHz = 176.95 TMAC/s
int8 cube throughput = 8192 MAC/cycle/core * 24 * 1.8 GHz = 353.89 TMAC/s
vector throughput, 32-bit = 64 ops/cycle/core * 24 * 1.8 GHz = 2764.8 GOPS/s
```

这些参数来自 `references/platform-assumptions.md` 的 `A2/A3` bucket。`GM path rate` 和 `L2 path rate` 是 theoretical path rate，不是真实有效带宽。

### 2.2 Baseline

主体计算：

```text
MACs = M * K * N
     = 4096 * 4096 * 12800
     = 214.75 GMAC

T_base_cube = 214.75 GMAC / 176.95 TMAC/s
            ~= 1213.63 us
```

MTE2 输入搬运使用默认 tile 冷/热数据模型。默认值固定在理论模型内部，不要求用户输入。

```text
baseM = 128
baseN = 256
dataType = 2 bytes for bf16

cold_hbm_bytes = 1 * M * K * dataType + 4 * K * N * dataType
               ~= 0.453 GB

hot_l2_bytes = [((N / baseN) - 1) * M * K + ((M / baseM) - 4) * K * N] * dataType
             ~= 4.580 GB

T_base_mte2 = hot_l2_bytes / L2_rate + cold_hbm_bytes / GM_rate
            ~= 4.580 GB / 4752 GB/s + 0.453 GB / 1382.4 GB/s
            ~= 1291.52 us
```

MTE3 结果写回：

```text
Y bytes = M * N * 2 ~= 0.105 GB
T_base_mte3 = Y bytes / GM path rate ~= 75.86 us
```

组合：

```text
T_baseline = max(T_base_cube, T_base_mte2, T_base_mte3) + T_base_fixed
           ~= 1291.52 us, when T_base_fixed = 0
```

### 2.3 Dynamic Quant

target 先把 activation `X` 从 bf16 动态量化成 int8，并生成 scale/meta。

```text
quant_granularity:
activation = PER_TOKEN / PER_ROW
weight = PER_CHANNEL

X_fp bytes = M * K * 2 ~= 0.034 GB
X_q bytes = M * K * 1 ~= 0.017 GB
activation_scale_count = M
activation_scale_bytes = M * 4 ~= 0.000016 GB
weight_scale_count = N
weight_scale_bytes = N * 4 ~= 0.000051 GB

T_dq_move_in ~= 24.27 us
T_dq_move_out ~= 12.15 us

ops_per_element = 8, rough estimate for abs/max/scale/round/clamp/cast
dq_vector_ops ~= M * K * ops_per_element
              = 4096 * 4096 * 8
              = 134.22 Mops

vector_throughput_32bit
= 64 ops/cycle/core * 24 cores * 1.8 GHz
= 2764.8 GOPS/s

T_dq_vector = dq_vector_ops / vector_throughput_32bit
             ~= 48.54 us

T_dynamic_quant = max(T_dq_move_in, T_dq_move_out, T_dq_vector) + T_dq_fixed
                ~= 58.54 us, when T_dq_fixed = 10 us
```

### 2.4 QMM 选中候选

本例使用 ops-style 默认候选集：4 种 transpose interpretation，每种最多包含 `basic_base` 和 `small_shape_base` 两个 base candidate。下面只展开最终选中候选的估算过程。

选中候选按执行顺序估算：

```text
T_qmm_fixed = 10 us

qmm_scale_meta_bytes = activation_scale_bytes + weight_scale_bytes
T_qmm_gm_to_l1 ~= bytes(X_q + W_q + qmm_scale_meta_bytes) / GM path rate
T_qmm_l1_to_l0 ~= bytes(X_q + W_q + qmm_scale_meta_bytes) / documented local path rate
T_qmm_input_movement ~= T_qmm_gm_to_l1 + T_qmm_l1_to_l0 + T_weight_transform

T_qmm_cube = 214.75 GMAC / 353.89 TMAC/s ~= 606.81 us

T_qmm_fixpipe = 0 us, optimistic default unless calibrated
T_qmm_antiquant_rescale ~= M * N * 4 / vector_throughput ~= 75.86 us
T_qmm_mte3 ~= 75.86 us
```

int qmm 路径口径：

```text
qmm fixed
-> input load(GM/L2 to L1, L1 to L0A/L0B/L0C)
-> qmm cube
-> output path(FixPipe result handling, INT antiquant/rescale, final MTE3 write)
```

组合：

```text
T_qmm_selected_candidate
= T_qmm_fixed
+ max(T_qmm_cube, T_qmm_input_movement)
+ T_qmm_fixpipe
+ T_qmm_antiquant_rescale
+ T_qmm_mte3
~= 768.52 us

T_target = T_dynamic_quant + T_qmm_selected_candidate
         ~= 827.06 us

estimated_speedup = 1291.52 / 827.06 ~= 1.56x
```

说明：`example-estimates.md` 使用同一公式批量重估。该大 shape 的 baseline 在默认 MTE2 冷/热数据模型下由 input movement 主导。

## 3. A5 MX 示例

示例 shape：`M=4096,K=4096,N=12800`，方案：`A5 + mx_w8a8`。

### 3.1 平台理论参数

```text
GM path rate = 80 B/cycle/core * 32 * 1.65 GHz = 4224 GB/s
L2 path rate = 100 B/cycle/core * 32 * 1.65 GHz = 5280 GB/s
bf16 cube throughput = 4096 MAC/cycle/core * 32 * 1.65 GHz = 216.27 TMAC/s
8-bit cube throughput assumption = 8192 MAC/cycle/core * 32 * 1.65 GHz = 432.54 TMAC/s
```

### 3.2 MX 默认格式

MX 按 per-block/block-scaled 格式估算：

```text
quant_granularity = PER_BLOCK
mx_block_size = 32 elements/block
mx_scale_dtype_size = 1 byte/block scale

activation value bytes = M * K * 1
activation_scale_count = ceil(M * K / 32)
activation scale bytes = activation_scale_count * 1
weight value bytes = K * N * 1
weight_scale_count = ceil(K * N / 32)
weight scale bytes = weight_scale_count * 1
```

MX target 路径：

```text
qmm fixed
-> input load(GM/L2 to L1, L1 to L0A/L0B/L0C)
-> qmm cube
-> output path(FixPipe result handling, final MTE3 write)
```

MX 路径默认不加入 INT 路径的额外 antiquant/rescale 后处理。

### 3.3 估算结果

使用默认 baseline tile 和 high-bandwidth A5：

```text
T_baseline ~= 992.97 us
T_target(mx_w8a8) ~= 581.03 us
estimated_speedup ~= 1.71x
```

关键假设：A5 mx 使用 8-bit cube-like throughput assumption；MX block size 为 `32`；scale/meta bytes 已计入 dynamic quant 写出和 qmm 输入搬运。

## 4. 报告中必须列出的假设

- platform bucket 和 A5 bandwidth variant
- baseline 默认 tile：`baseM=128`、`baseN=256`
- baseline cold GM bytes、hot L2 bytes
- qmm candidate policy 和选中候选，包括 `transpose_interpretation`、`baseM/baseN/baseK`、used core、tail 和冷/热数据比例
- dynamic quant vector ops 或经验系数
- INT 或 MX target 路径
- MX block size、scale dtype、scale/meta bytes
- fixed overhead 来源
