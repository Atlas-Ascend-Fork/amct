# Dynamic Quant + QMM Estimator 验证记录

这份文档用于验证 `$dynamic-quant-qmm-analysis` 理论 estimator 的方向准确性和误差范围。它不是某一个 qmm kernel 的专用样例。

## 1. 范围

- 目标路径：Linear `dynamic quant + qmm`
- 平台 bucket：`A2/A3`、`A5`
- `A2/A3` 理论 preset：`int_w8a8`、`int_w4a8`、`int_w4a4`
- `A5` 理论 preset：`int_w8a8`、`mx_w8a8`、`mx_w4a4`
- 已验证 preset：`int_w8a8`、`int_w4a4` on `A2/A3`
- deferred validation：`int_w4a8`、`mx_w8a8`、`mx_w4a4`
- 当前 weight preparation：`offline_prepared`
- 当前 qmm candidate policy：`ops-style 默认候选集`
- baseline 默认 tile：`baseM=128`、`baseN=256`
- MX 默认 block size：`32 elements/block`
- 权重默认离线量化，实测 target 不统计 weight `DynamicQuant`

理论时间是 bucket 级估算。实测表中的芯片时间是 runtime measurement，不是按芯片分别计算的理论时间。

## 2. 理论记录

### 2.1 A2-A3 理论记录

| 平台 | Preset | 激活格式 | 权重格式 | Candidate | Shape | T_baseline | T_target | 理论 speedup | 主瓶颈 | Confidence | Key assumptions | 理论方向 |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| `A2/A3` | `int_w8a8` | `int8` | `int8` | `ops-style default` | `M=4096,K=4096,N=12800` | `1291.52 us` | `827.06 us` | `1.56x` | `qmm cube / MTE2 baseline` | `High` | 24-core bucket, baseline baseM=128 baseN=256, offline weight | 正收益 |
| `A2/A3` | `int_w8a8` | `int8` | `int8` | `ops-style default` | `M=4096,K=4096,N=4096` | `424.99 us` | `311.27 us` | `1.37x` | `qmm cube / MTE2 baseline` | `High` | 24-core bucket, baseline baseM=128 baseN=256, offline weight | 正收益 |
| `A2/A3` | `int_w8a8` | `int8` | `int8` | `ops-style default` | `M=4096,K=1024,N=1024` | `29.79 us` | `56.41 us` | `0.53x` | `dynamic quant / fixed overhead` | `High` | 24-core bucket, fixed overhead assumption | 负收益 |
| `A2/A3` | `int_w8a8` | `int8` | `int8` | `ops-style default` | `M=256,K=1024,N=1024` | `6.78 us` | `22.61 us` | `0.30x` | `dynamic quant / fixed overhead` | `High` | 24-core bucket, fixed overhead assumption | 负收益 |
| `A2/A3` | `int_w8a8` | `int8` | `int8` | `ops-style default` | `M=64,K=1024,N=1024` | `6.25 us` | `21.30 us` | `0.29x` | `dynamic quant / fixed overhead` | `High` | 24-core bucket, fixed overhead assumption | 负收益 |
| `A2/A3` | `int_w4a8` | `int8` | `int4` | `ops-style default` | `M=4096,K=4096,N=12800` | `1291.52 us` | `523.66 us` | `2.47x` | `qmm cube` | `Medium` | A8W4 path uses explicit scale/packing defaults | 正收益 |
| `A2/A3` | `int_w4a8` | `int8` | `int4` | `ops-style default` | `M=4096,K=4096,N=4096` | `424.99 us` | `214.18 us` | `1.98x` | `qmm cube` | `Medium` | A8W4 path uses explicit scale/packing defaults | 正收益 |
| `A2/A3` | `int_w4a8` | `int8` | `int4` | `ops-style default` | `M=4096,K=1024,N=1024` | `29.79 us` | `50.34 us` | `0.59x` | `dynamic quant / fixed overhead` | `Medium` | A8W4 path uses explicit scale/packing defaults | 负收益 |
| `A2/A3` | `int_w4a8` | `int8` | `int4` | `ops-style default` | `M=256,K=1024,N=1024` | `6.78 us` | `22.17 us` | `0.31x` | `dynamic quant / fixed overhead` | `Medium` | A8W4 path uses explicit scale/packing defaults | 负收益 |
| `A2/A3` | `int_w4a8` | `int8` | `int4` | `ops-style default` | `M=64,K=1024,N=1024` | `6.25 us` | `20.87 us` | `0.30x` | `dynamic quant / fixed overhead` | `Medium` | A8W4 path uses explicit scale/packing defaults | 负收益 |
| `A2/A3` | `int_w4a4` | `int4` | `int4` | `ops-style default` | `M=4096,K=4096,N=12800` | `1291.52 us` | `568.89 us` | `2.27x` | `qmm cube / antiquant` | `Medium` | int4 packing and antiquant/rescale use explicit defaults | 正收益 |
| `A2/A3` | `int_w4a4` | `int4` | `int4` | `ops-style default` | `M=4096,K=4096,N=4096` | `424.99 us` | `246.52 us` | `1.72x` | `qmm cube / antiquant` | `Medium` | int4 packing and antiquant/rescale use explicit defaults | 正收益 |
| `A2/A3` | `int_w4a4` | `int4` | `int4` | `ops-style default` | `M=4096,K=1024,N=1024` | `29.79 us` | `59.93 us` | `0.50x` | `dynamic quant / fixed overhead` | `Medium` | int4 packing and antiquant/rescale use explicit defaults | 负收益 |
| `A2/A3` | `int_w4a4` | `int4` | `int4` | `ops-style default` | `M=256,K=1024,N=1024` | `6.78 us` | `24.54 us` | `0.28x` | `dynamic quant / fixed overhead` | `Medium` | int4 packing and antiquant/rescale use explicit defaults | 负收益 |
| `A2/A3` | `int_w4a4` | `int4` | `int4` | `ops-style default` | `M=64,K=1024,N=1024` | `6.25 us` | `22.96 us` | `0.27x` | `dynamic quant / fixed overhead` | `Medium` | int4 packing and antiquant/rescale use explicit defaults | 负收益 |

### 2.2 A5 理论记录

| 平台 | Preset | 激活格式 | 权重格式 | Candidate | Shape | T_baseline | T_target | 理论 speedup | 主瓶颈 | Confidence | Key assumptions | 理论方向 |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| `A5` | `int_w8a8` | `int8` | `int8` | `ops-style default` | `M=4096,K=4096,N=12800` | `992.97 us` | `643.09 us` | `1.54x` | `qmm cube / MTE2 baseline` | `Medium` | high-bandwidth A5, baseline baseM=128 baseN=256, offline weight | 正收益 |
| `A5` | `int_w8a8` | `int8` | `int8` | `ops-style default` | `M=4096,K=4096,N=4096` | `317.75 us` | `246.40 us` | `1.29x` | `qmm cube / MTE2 baseline` | `Medium` | high-bandwidth A5, baseline baseM=128 baseN=256, offline weight | 正收益 |
| `A5` | `int_w8a8` | `int8` | `int8` | `ops-style default` | `M=4096,K=1024,N=1024` | `19.86 us` | `46.81 us` | `0.42x` | `dynamic quant / fixed overhead` | `Medium` | high-bandwidth A5, fixed overhead assumption | 负收益 |
| `A5` | `int_w8a8` | `int8` | `int8` | `ops-style default` | `M=256,K=1024,N=1024` | `2.41 us` | `21.68 us` | `0.11x` | `dynamic quant / fixed overhead` | `Medium` | high-bandwidth A5, fixed overhead assumption | 负收益 |
| `A5` | `int_w8a8` | `int8` | `int8` | `ops-style default` | `M=64,K=1024,N=1024` | `2.09 us` | `20.63 us` | `0.10x` | `dynamic quant / fixed overhead` | `Medium` | high-bandwidth A5, fixed overhead assumption | 负收益 |
| `A5` | `mx_w8a8` | `mxfp8` | `mxfp8` | `ops-style default` | `M=4096,K=4096,N=12800` | `992.97 us` | `581.03 us` | `1.71x` | `qmm cube / scale meta` | `Medium` | MXFP8 maps to 8-bit throughput, block=32, scale/meta counted | 正收益 |
| `A5` | `mx_w8a8` | `mxfp8` | `mxfp8` | `ops-style default` | `M=4096,K=4096,N=4096` | `317.75 us` | `226.54 us` | `1.40x` | `qmm cube / scale meta` | `Medium` | MXFP8 maps to 8-bit throughput, block=32, scale/meta counted | 正收益 |
| `A5` | `mx_w8a8` | `mxfp8` | `mxfp8` | `ops-style default` | `M=4096,K=1024,N=1024` | `19.86 us` | `41.85 us` | `0.47x` | `dynamic quant / scale meta` | `Medium` | MXFP8 maps to 8-bit throughput, block=32, scale/meta counted | 负收益 |
| `A5` | `mx_w8a8` | `mxfp8` | `mxfp8` | `ops-style default` | `M=256,K=1024,N=1024` | `2.41 us` | `21.37 us` | `0.11x` | `dynamic quant / fixed overhead` | `Medium` | MXFP8 maps to 8-bit throughput, block=32, scale/meta counted | 负收益 |
| `A5` | `mx_w8a8` | `mxfp8` | `mxfp8` | `ops-style default` | `M=64,K=1024,N=1024` | `2.09 us` | `20.57 us` | `0.10x` | `dynamic quant / fixed overhead` | `Medium` | MXFP8 maps to 8-bit throughput, block=32, scale/meta counted | 负收益 |
| `A5` | `mx_w4a4` | `mxfp4` | `mxfp4` | `ops-style default` | `M=4096,K=4096,N=12800` | `992.97 us` | `602.89 us` | `1.65x` | `qmm cube / scale meta` | `Medium` | MXFP4 maps to 4-bit value bytes, block=32, scale/meta counted | 正收益 |
| `A5` | `mx_w4a4` | `mxfp4` | `mxfp4` | `ops-style default` | `M=4096,K=4096,N=4096` | `317.75 us` | `248.40 us` | `1.28x` | `qmm cube / scale meta` | `Medium` | MXFP4 maps to 4-bit value bytes, block=32, scale/meta counted | 正收益 |
| `A5` | `mx_w4a4` | `mxfp4` | `mxfp4` | `ops-style default` | `M=4096,K=1024,N=1024` | `19.86 us` | `48.81 us` | `0.41x` | `dynamic quant / scale meta` | `Medium` | MXFP4 maps to 4-bit value bytes, block=32, scale/meta counted | 负收益 |
| `A5` | `mx_w4a4` | `mxfp4` | `mxfp4` | `ops-style default` | `M=256,K=1024,N=1024` | `2.41 us` | `23.68 us` | `0.10x` | `dynamic quant / fixed overhead` | `Medium` | MXFP4 maps to 4-bit value bytes, block=32, scale/meta counted | 负收益 |
| `A5` | `mx_w4a4` | `mxfp4` | `mxfp4` | `ops-style default` | `M=64,K=1024,N=1024` | `2.09 us` | `22.46 us` | `0.09x` | `dynamic quant / fixed overhead` | `Medium` | MXFP4 maps to 4-bit value bytes, block=32, scale/meta counted | 负收益 |

## 3. 计算示例

完整手工计算示例已移到：

```text
skills/quant-tools/dynamic-quant-qmm-analysis/references/validation/example-calculations.md
```

## 4. 验证脚本

验证脚本位置：

```text
skills/quant-tools/dynamic-quant-qmm-analysis/validate_qmm.py
```

默认行为：

- shape 输入为 `M,K,N`
- profiler 输出到 `./profiling/<date>/quant`
- 默认 target 顺序为 `int_w8a8,int_w4a4`
- target 路径为 activation `DynamicQuant + QuantBatchMatmul*`
- weight 在 profiler 开始前预量化
- `int_w4a8` validation deferred，等待官方 A8W4 scale/packing 路径确认；validation 状态不改变理论 confidence

## 5. 实测规则

- baseline 时间：`bf16 MatMul`
- target 时间：activation `DynamicQuant + QuantBatchMatmul*`
- weight 默认离线量化，不统计 weight `DynamicQuant`
- 当前不把具体权重存储 layout 作为主表维度
- 后续只有发现某个平台或 shape 上 runtime packing/layout conversion 稳定明显影响结论时，再显式记录 `T_weight_transform`
- 每个平台内，`理论记录`、`实测汇总` 和 `Target Breakdown` 必须一一对应
- 对应 key 为 `平台 + Preset + 激活格式 + 权重格式 + Candidate + Shape`
- 新增 shape、platform 或 preset 时，先追加 theory row，再同步追加待实测 summary row 和 breakdown row
- 未实测 row 保留一行，时间列留空，只标记 pending validation

## 6. 实测记录

### 6.1 A2-A3 实测汇总

| 日期 | 平台 | 芯片 | CANN | Preset | 激活格式 | 权重格式 | Candidate | Shape | 理论 speedup | Baseline 时间 | Target 时间 | 实测 speedup | 方向匹配 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `2026-04-27` | `A2/A3` | `910B4` | `8.5.0` | `int_w8a8` | `int8` | `int8` | `ops-style default` | `M=4096,K=4096,N=12800` | `1.56x` | `1988.50 us` | `1471.28 us` | `1.35x` | yes |
| `2026-04-27` | `A2/A3` | `910B4` | `8.5.0` | `int_w8a8` | `int8` | `int8` | `ops-style default` | `M=4096,K=4096,N=4096` | `1.37x` | `679.29 us` | `556.75 us` | `1.22x` | yes |
| `2026-04-27` | `A2/A3` | `910B4` | `8.5.0` | `int_w8a8` | `int8` | `int8` | `ops-style default` | `M=4096,K=1024,N=1024` | `0.53x` | `87.84 us` | `109.91 us` | `0.80x` | yes |
| `2026-04-27` | `A2/A3` | `910B4` | `8.5.0` | `int_w8a8` | `int8` | `int8` | `ops-style default` | `M=256,K=1024,N=1024` | `0.30x` | `15.00 us` | `46.34 us` | `0.32x` | yes |
| `2026-04-27` | `A2/A3` | `910B4` | `8.5.0` | `int_w8a8` | `int8` | `int8` | `ops-style default` | `M=64,K=1024,N=1024` | `0.29x` | `9.08 us` | `31.62 us` | `0.29x` | yes |
|  | `A2/A3` |  |  | `int_w4a8` | `int8` | `int4` | `ops-style default` | `M=4096,K=4096,N=12800` | `2.47x` |  |  |  | pending |
|  | `A2/A3` |  |  | `int_w4a8` | `int8` | `int4` | `ops-style default` | `M=4096,K=4096,N=4096` | `1.98x` |  |  |  | pending |
|  | `A2/A3` |  |  | `int_w4a8` | `int8` | `int4` | `ops-style default` | `M=4096,K=1024,N=1024` | `0.59x` |  |  |  | pending |
|  | `A2/A3` |  |  | `int_w4a8` | `int8` | `int4` | `ops-style default` | `M=256,K=1024,N=1024` | `0.31x` |  |  |  | pending |
|  | `A2/A3` |  |  | `int_w4a8` | `int8` | `int4` | `ops-style default` | `M=64,K=1024,N=1024` | `0.30x` |  |  |  | pending |
| `2026-04-27` | `A2/A3` | `910B4` | `8.5.0` | `int_w4a4` | `int4` | `int4` | `ops-style default` | `M=4096,K=4096,N=12800` | `2.27x` | `1988.50 us` | `699.32 us` | `2.84x` | yes |
| `2026-04-27` | `A2/A3` | `910B4` | `8.5.0` | `int_w4a4` | `int4` | `int4` | `ops-style default` | `M=4096,K=4096,N=4096` | `1.72x` | `679.29 us` | `267.87 us` | `2.54x` | yes |
| `2026-04-27` | `A2/A3` | `910B4` | `8.5.0` | `int_w4a4` | `int4` | `int4` | `ops-style default` | `M=4096,K=1024,N=1024` | `0.50x` | `87.84 us` | `89.53 us` | `0.98x` | yes |
| `2026-04-27` | `A2/A3` | `910B4` | `8.5.0` | `int_w4a4` | `int4` | `int4` | `ops-style default` | `M=256,K=1024,N=1024` | `0.28x` | `15.00 us` | `31.16 us` | `0.48x` | yes |
| `2026-04-27` | `A2/A3` | `910B4` | `8.5.0` | `int_w4a4` | `int4` | `int4` | `ops-style default` | `M=64,K=1024,N=1024` | `0.27x` | `9.08 us` | `25.24 us` | `0.36x` | yes |

### 6.2 A2-A3 Target 分解

`QMM` 是 profiler 中 `QuantBatchMatmul*` operator duration，不等于理论模型里的纯 `T_qmm_cube`。

| 日期 | 平台 | 芯片 | Preset | 激活格式 | 权重格式 | Candidate | Shape | Target 时间 | 激活动态量化 | 权重转换 | QMM | Other |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `2026-04-27` | `A2/A3` | `910B4` | `int_w8a8` | `int8` | `int8` | `ops-style default` | `M=4096,K=4096,N=12800` | `1471.28 us` | `80.87 us` | `0 us` | `1390.41 us` | `0 us` |
| `2026-04-27` | `A2/A3` | `910B4` | `int_w8a8` | `int8` | `int8` | `ops-style default` | `M=4096,K=4096,N=4096` | `556.75 us` | `82.49 us` | `0 us` | `474.26 us` | `0 us` |
| `2026-04-27` | `A2/A3` | `910B4` | `int_w8a8` | `int8` | `int8` | `ops-style default` | `M=4096,K=1024,N=1024` | `109.91 us` | `49.76 us` | `0 us` | `60.15 us` | `0 us` |
| `2026-04-27` | `A2/A3` | `910B4` | `int_w8a8` | `int8` | `int8` | `ops-style default` | `M=256,K=1024,N=1024` | `46.34 us` | `20.78 us` | `0 us` | `25.56 us` | `0 us` |
| `2026-04-27` | `A2/A3` | `910B4` | `int_w8a8` | `int8` | `int8` | `ops-style default` | `M=64,K=1024,N=1024` | `31.62 us` | `13.51 us` | `0 us` | `18.11 us` | `0 us` |
|  | `A2/A3` |  | `int_w4a8` | `int8` | `int4` | `ops-style default` | `M=4096,K=4096,N=12800` |  |  |  |  | pending |
|  | `A2/A3` |  | `int_w4a8` | `int8` | `int4` | `ops-style default` | `M=4096,K=4096,N=4096` |  |  |  |  | pending |
|  | `A2/A3` |  | `int_w4a8` | `int8` | `int4` | `ops-style default` | `M=4096,K=1024,N=1024` |  |  |  |  | pending |
|  | `A2/A3` |  | `int_w4a8` | `int8` | `int4` | `ops-style default` | `M=256,K=1024,N=1024` |  |  |  |  | pending |
|  | `A2/A3` |  | `int_w4a8` | `int8` | `int4` | `ops-style default` | `M=64,K=1024,N=1024` |  |  |  |  | pending |
| `2026-04-27` | `A2/A3` | `910B4` | `int_w4a4` | `int4` | `int4` | `ops-style default` | `M=4096,K=4096,N=12800` | `699.32 us` | `112.92 us` | `0 us` | `586.40 us` | `0 us` |
| `2026-04-27` | `A2/A3` | `910B4` | `int_w4a4` | `int4` | `int4` | `ops-style default` | `M=4096,K=4096,N=4096` | `267.87 us` | `72.75 us` | `0 us` | `195.11 us` | `0 us` |
| `2026-04-27` | `A2/A3` | `910B4` | `int_w4a4` | `int4` | `int4` | `ops-style default` | `M=4096,K=1024,N=1024` | `89.53 us` | `40.98 us` | `0 us` | `48.55 us` | `0 us` |
| `2026-04-27` | `A2/A3` | `910B4` | `int_w4a4` | `int4` | `int4` | `ops-style default` | `M=256,K=1024,N=1024` | `31.16 us` | `13.74 us` | `0 us` | `17.42 us` | `0 us` |
| `2026-04-27` | `A2/A3` | `910B4` | `int_w4a4` | `int4` | `int4` | `ops-style default` | `M=64,K=1024,N=1024` | `25.24 us` | `11.92 us` | `0 us` | `13.32 us` | `0 us` |

### 6.3 A5 实测汇总

| 日期 | 平台 | 芯片 | CANN | Preset | 激活格式 | 权重格式 | Candidate | Shape | 理论 speedup | Baseline 时间 | Target 时间 | 实测 speedup | 方向匹配 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
|  | `A5` |  |  | `int_w8a8` | `int8` | `int8` | `ops-style default` | `M=4096,K=4096,N=12800` | `1.54x` |  |  |  | pending |
|  | `A5` |  |  | `int_w8a8` | `int8` | `int8` | `ops-style default` | `M=4096,K=4096,N=4096` | `1.29x` |  |  |  | pending |
|  | `A5` |  |  | `int_w8a8` | `int8` | `int8` | `ops-style default` | `M=4096,K=1024,N=1024` | `0.42x` |  |  |  | pending |
|  | `A5` |  |  | `int_w8a8` | `int8` | `int8` | `ops-style default` | `M=256,K=1024,N=1024` | `0.11x` |  |  |  | pending |
|  | `A5` |  |  | `int_w8a8` | `int8` | `int8` | `ops-style default` | `M=64,K=1024,N=1024` | `0.10x` |  |  |  | pending |
|  | `A5` |  |  | `mx_w8a8` | `mxfp8` | `mxfp8` | `ops-style default` | `M=4096,K=4096,N=12800` | `1.71x` |  |  |  | pending |
|  | `A5` |  |  | `mx_w8a8` | `mxfp8` | `mxfp8` | `ops-style default` | `M=4096,K=4096,N=4096` | `1.40x` |  |  |  | pending |
|  | `A5` |  |  | `mx_w8a8` | `mxfp8` | `mxfp8` | `ops-style default` | `M=4096,K=1024,N=1024` | `0.47x` |  |  |  | pending |
|  | `A5` |  |  | `mx_w8a8` | `mxfp8` | `mxfp8` | `ops-style default` | `M=256,K=1024,N=1024` | `0.11x` |  |  |  | pending |
|  | `A5` |  |  | `mx_w8a8` | `mxfp8` | `mxfp8` | `ops-style default` | `M=64,K=1024,N=1024` | `0.10x` |  |  |  | pending |
|  | `A5` |  |  | `mx_w4a4` | `mxfp4` | `mxfp4` | `ops-style default` | `M=4096,K=4096,N=12800` | `1.65x` |  |  |  | pending |
|  | `A5` |  |  | `mx_w4a4` | `mxfp4` | `mxfp4` | `ops-style default` | `M=4096,K=4096,N=4096` | `1.28x` |  |  |  | pending |
|  | `A5` |  |  | `mx_w4a4` | `mxfp4` | `mxfp4` | `ops-style default` | `M=4096,K=1024,N=1024` | `0.41x` |  |  |  | pending |
|  | `A5` |  |  | `mx_w4a4` | `mxfp4` | `mxfp4` | `ops-style default` | `M=256,K=1024,N=1024` | `0.10x` |  |  |  | pending |
|  | `A5` |  |  | `mx_w4a4` | `mxfp4` | `mxfp4` | `ops-style default` | `M=64,K=1024,N=1024` | `0.09x` |  |  |  | pending |

### 6.4 A5 Target 分解

| 日期 | 平台 | 芯片 | Preset | 激活格式 | 权重格式 | Candidate | Shape | Target 时间 | 激活动态量化 | 权重转换 | QMM | Other |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
|  | `A5` |  | `int_w8a8` | `int8` | `int8` | `ops-style default` | `M=4096,K=4096,N=12800` |  |  |  |  | pending |
|  | `A5` |  | `int_w8a8` | `int8` | `int8` | `ops-style default` | `M=4096,K=4096,N=4096` |  |  |  |  | pending |
|  | `A5` |  | `int_w8a8` | `int8` | `int8` | `ops-style default` | `M=4096,K=1024,N=1024` |  |  |  |  | pending |
|  | `A5` |  | `int_w8a8` | `int8` | `int8` | `ops-style default` | `M=256,K=1024,N=1024` |  |  |  |  | pending |
|  | `A5` |  | `int_w8a8` | `int8` | `int8` | `ops-style default` | `M=64,K=1024,N=1024` |  |  |  |  | pending |
|  | `A5` |  | `mx_w8a8` | `mxfp8` | `mxfp8` | `ops-style default` | `M=4096,K=4096,N=12800` |  |  |  |  | pending |
|  | `A5` |  | `mx_w8a8` | `mxfp8` | `mxfp8` | `ops-style default` | `M=4096,K=4096,N=4096` |  |  |  |  | pending |
|  | `A5` |  | `mx_w8a8` | `mxfp8` | `mxfp8` | `ops-style default` | `M=4096,K=1024,N=1024` |  |  |  |  | pending |
|  | `A5` |  | `mx_w8a8` | `mxfp8` | `mxfp8` | `ops-style default` | `M=256,K=1024,N=1024` |  |  |  |  | pending |
|  | `A5` |  | `mx_w8a8` | `mxfp8` | `mxfp8` | `ops-style default` | `M=64,K=1024,N=1024` |  |  |  |  | pending |
|  | `A5` |  | `mx_w4a4` | `mxfp4` | `mxfp4` | `ops-style default` | `M=4096,K=4096,N=12800` |  |  |  |  | pending |
|  | `A5` |  | `mx_w4a4` | `mxfp4` | `mxfp4` | `ops-style default` | `M=4096,K=4096,N=4096` |  |  |  |  | pending |
|  | `A5` |  | `mx_w4a4` | `mxfp4` | `mxfp4` | `ops-style default` | `M=4096,K=1024,N=1024` |  |  |  |  | pending |
|  | `A5` |  | `mx_w4a4` | `mxfp4` | `mxfp4` | `ops-style default` | `M=256,K=1024,N=1024` |  |  |  |  | pending |
|  | `A5` |  | `mx_w4a4` | `mxfp4` | `mxfp4` | `ops-style default` | `M=64,K=1024,N=1024` |  |  |  |  | pending |

## 7. 准确性

### 7.1 方向命中率

| 平台 | 芯片 | Preset | 覆盖 shape 数 | 方向命中率 | 结论 |
| --- | --- | --- | ---: | ---: | --- |
| `A2/A3` | `910B4` | `int_w8a8` | `5` | `5/5` | 正收益和负收益方向均命中。 |
| `A2/A3` | `910B4` | `int_w4a4` | `5` | `5/5` | 大 shape 正收益和小 shape 负收益方向均命中。 |

### 7.2 观察结论

- 当前文档验证的是 estimator，后续新增方案时追加 `Preset / 激活格式 / 权重格式 / 平台 / shape` 维度即可。
- `int_w8a8` 和 `int_w4a4` 的大 shape 理论收益方向为正；小 shape 或窄 N/K 的理论收益方向为负，当前 910B4 实测方向均命中。
- `int_w4a8` validation deferred，因为当前 A8W4 grouped path 的 official scale/packing 方式尚未确认；该状态不改变理论 confidence。
- A5 示例表当前只登记 `int_w8a8`、`mx_w8a8`、`mx_w4a4`；其他 A5 格式能力在理论模型中支持，待命名和验证口径明确后追加。
- A5 `mx_w8a8` 和 `mx_w4a4` 当前仅为 theoretical-only rows；runtime validation 状态只用于标记验证覆盖度，不改变理论模型本身。
- `M=4096,K=1024,N=1024` 的 `int_w4a4` measured speedup 接近 `1.0x`，该 shape 处于收益边界附近，后续应优先作为 calibration point。

## 8. 下游接入案例

`Qwen3-4B` 的 `$scheme-recommendation` 接入前后正收益案例见：

```text
skills/quant-tools/dynamic-quant-qmm-analysis/references/validation/qwen3-4b-scheme-recommendation-integration.md
```

该案例按 Qwen3 wrapper 的实际调用口径组织 4 个部署比较单元，展示了调用方如何把 shared activation quant 的 `q/k/v` group 纳入 qmm 理论评估，并用结果补充 `$scheme-recommendation` 的性能风险说明；该案例不触发模块级回退。

`Qwen3-Next-80B-A3B` 的 `$scheme-recommendation` 接入后模块级回退案例见：

```text
skills/quant-tools/dynamic-quant-qmm-analysis/references/validation/qwen3-next-80b-a3b-scheme-recommendation-rollback.md
```

该案例按 Qwen3-Next wrapper 的实际调用口径组织 attention projection linear 和 MoE expert 比较单元，展示了 qmm estimator 如何识别 Linear-dominant `moe` 模块级整体负收益，并把首推方案从 `attn + moe` 回退到 `attn-only`；该回退不是由 attention projection linear 单独触发完整 attention module 回退。
