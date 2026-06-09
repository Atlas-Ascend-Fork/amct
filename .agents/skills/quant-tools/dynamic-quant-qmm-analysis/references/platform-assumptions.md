# Platform Assumptions

这份文档说明 `dynamic quant + quant matmul` 理论评估使用的平台术语、支持范围和默认参数口径。

## 1. 概览

### 1.1 目的

平台参数只用于粗粒度理论估算，帮助判断性能收益的主要来源和主要瓶颈。

这些参数是理论估算参数，不表示每个算子都能达到的真实上限。实际可达性能还会受到 tiling、地址连续性、对齐、小块搬运、流水重叠和实现细节影响。

### 1.2 术语

- `throughput`
  - 吞吐能力，表示单位时间内硬件能完成多少计算工作。
- `bandwidth`
  - 数据搬运带宽，表示单位时间内能搬运多少字节。
- `GM`
  - Device global memory 的抽象名称。实际外部存储介质或路径可能表现为 HBM 或 DDR，取决于平台和文档口径。
- `HBM / DDR`
  - GM 背后的外部存储介质或路径名称。本文中的 `DDR/GM path rate` 沿用 CANN `ddr_rate` 字段命名，含义是 GM/HBM/DDR 外部存储路径的理论速率，不表示物理介质一定是 DDR。
- `L2`
  - 片上共享缓存。比全局存储更近，通常路径速率更高，但容量有限。
- `cube`
  - 主要承担矩阵乘等块状计算的硬件侧。
- `vector`
  - 主要承担量化、scale、elementwise、antiquant/rescale 等向量计算的硬件侧。
- `MTE`
  - 数据搬运流水。理论耗时通常按 `搬运字节数 / 路径速率` 估算。
- `MTE2`
  - 输入侧搬运流水，本文用于估算 GM/L2 到计算所需层级的输入搬运。
- `MTE3`
  - 输出侧搬运流水，本文用于估算结果写回 GM 的搬运。
- `FixPipe`
  - 矩阵计算结果搬出、格式转换、输出链路相关流水。
- `theoretical path rate`
  - 由平台参数推导出的理论路径速率。它是 `bytes / rate` 估算的下界参数，不等于实际有效带宽，也不等于某个 kernel 的实测带宽。

## 2. 范围

### 2.1 平台

当前 skill 只支持 Ascend NPU 相关平台：

- `A2/A3`
- `A5`

`A2` 和 `A3` 在当前 qmm 理论估算中使用同一组 bucket 级代表性单芯片参数。这里不覆盖所有 SKU 变体；如果用户关心特定 SKU，应把结论降级为需要单独确认的估算。

当前 `A2/A3` bucket 不分别为 `910B3` 和 `910B4` 建立理论参数。`910B3/910B4` 可以出现在实测登记中作为芯片标签，但同一 shape 在同一 `A2/A3` bucket 下应使用相同的理论 `T_baseline / T_target / speedup`。

### 2.2 理论依据

理论估算遵循公开性能分析中的通用原则：

- 搬运理论耗时：`bytes / bandwidth`
- 计算理论耗时：`work / throughput`
- 多条搬运共享同一路径时，不能把每条搬运都按独享峰值计算
- 小块、不连续或未对齐搬运通常无法达到理论路径速率
- 头开销对微秒级小算子可能显著

### 2.3 配置来源

默认平台参数参考 CANN `platform_config/*.ini` 中的代表性字段，但对外只保留 `A2/A3` 与 `A5` 两档。

当前用到的字段包括：

- `cube_core_cnt`
- `vector_core_cnt`
- `cube_vector_combine`
- `cube_freq`
- `ddr_rate`
- `l2_rate`
- `l2_size`
- `ub_size`

不要把这个文档扩展成完整 SKU 表。qmm skill 只需要能稳定解释当前 `A2/A3` 与 `A5` 的理论估算口径。

## 3. A2-A3

### 3.1 参数

- Applies to: `A2` and `A3` bucket for this single-chip qmm estimator
- Cube core count: `24`
- Vector core count: `48`
- CV mode: `split`
- Cube frequency: `1.8 GHz`
- DDR/GM path rate per core: `32 B/cycle`
- L2 path rate per core: `110 B/cycle`
- Total DDR/GM path rate estimate: `32 * 24 * 1.8 GHz = 1382.4 GB/s`
- Total L2 path rate estimate: `110 * 24 * 1.8 GHz = 4752 GB/s`
- Shared L2 size: `192 MB`
- UB per vector core: `192 KB`
- 16-bit cube throughput hint: `4096 MAC/cycle/core`
- 8-bit cube throughput hint: `8192 MAC/cycle/core`
- 4-bit cube throughput hint: `16384 MAC/cycle/core`
- Vector throughput hint: `128 x 16-bit ops/cycle/core`, `64 x 32-bit ops/cycle/core`

### 3.2 QMM 影响

对 qmm 的直接含义：

- split-CV 下 cube 与 vector 间的数据交接更容易显性化
- `dynamic quant` 写出 `X_q + scale/meta` 的成本计入 `T_dynamic_quant`，qmm 读入这些数据的成本计入 `T_qmm_input_movement`
- 大 shape 更容易受益于 int8/int4 cube throughput；小 shape 更容易被 quant、antiquant/rescale、MTE3 和固定开销吃掉收益

### 3.3 芯片标签

`A2/A3` 当前使用 `24-core representative bucket`。该 bucket 来自现有平台能力参考和 CANN `B1/B2/B2C` 类配置。`910B3/910B4` 可以作为实测芯片标签出现，但当前 qmm theory table 不展开 chip-level facts，也不按这些芯片分别重算。

## 4. A5

### 4.1 参数

- Applies to: `A5` for this qmm estimator
- Cube core count: `32`
- Vector core count: `64`
- CV mode: `merged`
- Cube frequency: `1.65 GHz`
- DDR/GM path rate per core: `80 B/cycle` by default for high-bandwidth A5
- Low-bandwidth A5 DDR/GM path rate per core: `31 B/cycle`, only when user specifies this variant
- L2 path rate per core: `100 B/cycle`
- Total DDR/GM path rate estimate: `80 * 32 * 1.65 GHz = 4224 GB/s`
- Low-bandwidth total DDR/GM path rate estimate: `31 * 32 * 1.65 GHz = 1636.8 GB/s`
- Total L2 path rate estimate: `100 * 32 * 1.65 GHz = 5280 GB/s`
- Shared L2 size: `128 MB`
- UB per vector core: `248 KB`
- 16-bit cube throughput hint: `4096 MAC/cycle/core`
- 8-bit cube throughput hint: `8192 MAC/cycle/core`, used for int8 and explicit fp8/mx throughput assumptions
- 16-bit total cube throughput estimate: `4096 * 32 * 1.65 GHz = 216.27 TMAC/s`
- 8-bit total cube throughput estimate: `8192 * 32 * 1.65 GHz = 432.54 TMAC/s`
- Vector throughput hint: use the same current modeling values as `A2/A3`, `128 x 16-bit ops/cycle/core` and `64 x 32-bit ops/cycle/core`, until A5-specific vector data is added

## 5. 速率计算

### 5.1 公式

估算时优先使用有效用核数，而不是无条件使用满核数：

```text
active_core_count = min(used_core_num, platform_cube_core_count)

theoretical_ddr_path_rate
= ddr_rate_per_core * active_core_count * cube_frequency

theoretical_l2_path_rate
= l2_rate_per_core * active_core_count * cube_frequency

cube_throughput(dtype)
= mac_per_cycle_per_core(dtype) * active_core_count * cube_frequency

vector_throughput(dtype)
= vector_ops_per_cycle_per_core(dtype) * active_core_count * cube_frequency
```

其中：

- `used_core_num` 来自 tile/core 估算
- `cube_frequency` 使用 GHz 时，结果单位自然落到 `Gbyte/s` 或 `GMAC/s`
- `mac_per_cycle_per_core(dtype)` 按 dtype 选择 16-bit、8-bit 或显式 mx/fp8 假设；A5 不使用普通 int4 hint
- `vector_ops_per_cycle_per_core(dtype)` 按 vector 数据位宽选择；缺少明确数据时降低 confidence

`theoretical_ddr_path_rate` 的名字来自 CANN `ddr_rate` 字段。报告中可以写成 `DDR/GM path rate` 或 `GM path rate`，但必须说明它只是外部存储路径理论速率，不是实测 bandwidth。

### 5.2 假设

- baseline 默认是 `bf16 float matmul`
- target 默认是 `dynamic quant + quant matmul`
- `PER_BLOCK / PER_TILE / PER_TOKEN` 这类量化粒度默认会引入额外 vector 或 antiquant/rescale 成本
- `dynamic quant` 输出的 `X_q + x_scale` 写出计入 `T_dynamic_quant`，qmm 读入计入 `T_qmm_input_movement`
- tile reuse 只是一种理论复用提示，不应解释为真实 cache counter

## 6. 澄清规则

出现下面情况时，应向用户确认，而不是强行套默认值：

- 平台不是 `A2/A3/A5`
- 用户关心特定 SKU、特定 kernel 实现或特定融合路径
- 量化路径不属于当前 skill 范围内的 `dynamic quant + qmm`
- 用户明确给出与默认平台参数冲突的芯片配置
