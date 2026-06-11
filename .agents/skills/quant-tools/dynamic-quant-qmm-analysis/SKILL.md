---
name: dynamic-quant-qmm-analysis
description: 对 Ascend NPU 上的 Linear `dynamic quant + quant matmul` 做粗粒度理论评估，输出 speedup、收益方向、主导瓶颈、置信度和关键假设。
user-invocable: true
---

# Dynamic Quant + QMM 理论评估

## 定位

这个 skill 只评估 Linear `dynamic quant + qmm` 相对 float/bf16 matmul baseline 的理论性能收益。

它不运行实验，不输出推荐决策，不把理论估算写成已验证 runtime 收益。调用方如果需要 `Go / Conditional Go / No-Go` 语义，应在调用方语境中消费 estimator 结论后自行映射。

## 支持范围

- 平台：`A2/A3`、`A5`
- 平台格式能力：`A2/A3` 支持 `INT8`、`INT4`；`A5` 支持 `INT8`、`FP8`、`MXFP8`、`MXFP6`、`MXFP4`
- 常用量化方案：`int_w8a8`、`int_w4a8`、`int_w4a4`、`fp_w8a8`、`mx_w8a8`、`mx_w6a6`、`mx_w4a4`
- `int_w4a8`：理论 row 可保留，但 validation deferred，等待官方 A8W4 scale/packing 路径确认
- `mx`：按 per-block/block-scaled 格式建模；缺少 bit-width throughput mapping、scale/meta 口径或量化粒度时降低 confidence

`A2` 和 `A3` 在当前 skill 中使用同一组 bucket 级平台假设。不要在 qmm skill 对外报告里扩展完整 SKU 表；如果用户关心特定芯片 SKU，应在关键假设中说明需要单独确认。

## 输入与来源

必需输入：

- `m`：Linear 的 activation 行数，来自用户或调用方确认的性能 workload 展开结果
- `k`：Linear 输入特征维度，来自用户输入、实际 Linear 权重 shape、module 配置或模型配置推导
- `n`：Linear 输出特征维度，来自用户输入、实际 Linear 权重 shape、module 配置或模型配置推导
- `quant_scheme`：候选量化方案，例如 `int_w8a8`、`int_w4a4`、`mx_w8a8`
- `platform`：当前硬件、用户指定平台或部署目标，例如 `A2/A3`、`A5`

常用可选输入：

- `scenario`
- `baseline_dtype`
- `output_dtype`
- `quant_granularity`
- `weight_preparation`
- `a5_bandwidth_variant`

缺少必需输入时，交互式调用应先向用户确认；无法补齐时返回 `blocked`。`blocked` 表示不输出收益方向或 speedup，只列出 `missing_inputs`。

输出时必须写明 shape 来源。用户显式提供 `M/K/N` 时优先使用，并标记为 `user_provided`；若 `K/N` 来自 Linear 权重，按 `weight.shape = [N, K]` 解释；若来自模型配置推导，应标记为 `model_config_derived`。若输出 single Linear、fused Linear 或模块内 group 的聚合结果，必须提供可从来源推导出各比较单元 `M/K/N` 的 shape summary 或来源表。

qmm skill 只判断 Linear `dynamic quant + qmm` 子路径的理论收益方向，不判断完整 attention module 性能。若调用方组织的是 attention projection group，输出命名应使用 `attention linear`、`attention projection` 或类似表述，不应写成完整 `attention module` 收益结论。

## 参考材料

执行前按顺序读取：

1. `references/platform-assumptions.md`
2. `references/theory-model.md`
3. `references/report-template.md`

需要核对历史 theory rows 或 validation rows 时，再读取 `references/validation/example-estimates.md`。需要手工复核公式展开时，再读取 `references/validation/example-calculations.md`。需要查看下游 `$scheme-recommendation` 接入效果时，再读取 `references/validation/qwen3-4b-scheme-recommendation-integration.md` 或 `references/validation/qwen3-next-80b-a3b-scheme-recommendation-rollback.md`。

## 工作流程

### 第一步：确认估算口径

确认 `m/k/n/platform/quant_scheme`，并明确 baseline/target 的默认假设。`m/k/n` 对应 Linear `X[M,K] @ W[K,N] -> Y[M,N]`。

- `baseline_dtype` 缺省按 `bf16`
- `weight_preparation` 缺省按 `default_api_prepared` 或 `offline_prepared`
- `a5_bandwidth_variant` 缺省按 high-bandwidth，即 `ddr_rate = 80 B/cycle/core`

如果 dtype、scale/meta、量化粒度、candidate policy 或 A5 bandwidth variant 缺失但仍可方向性估计，先确认用户是否能提供；无法补齐时必须写入关键假设并降低 confidence。缺少必需输入时不要做方向性估计。

若调用方已按实际部署图整理单 Linear、fused Linear 或模块内 group 的比较单元，应在输入和输出中说明 shape 来源与比较口径，并给出可推导各比较单元 `M/K/N` 的 shape summary。qmm skill 按调用方提供的 `M/K/N/platform/quant_scheme` 和假设执行理论评估，不自行推断部署图。

### 第二步：估算 Baseline

估算 float/bf16 matmul 的理论耗时 `T_baseline`。

baseline 使用主导项近似：cube compute、MTE2 input movement 和 MTE3 result write 可流水重叠，fixed overhead 单独计入。详细公式见 `references/theory-model.md`。

### 第三步：估算 Target

按执行顺序估算 target：

- `T_dynamic_quant`：activation 读入、`X_q + x_scale/meta` 写出、vector quant 和 fixed overhead
- `T_qmm_fixed`：qmm 头开销
- `T_qmm_gm_to_l1`：从 GM 读取 `X_q`、`W_q`、scale/meta；L2 reuse 只作为冷/热数据比例修正
- `T_qmm_l1_to_l0`：从 L1 搬入 L0A/L0B/L0C 所需数据
- `T_qmm_cube`：qmm cube 主体计算
- `T_qmm_fixpipe`：cube 结果经 FixPipe 搬出到 GM
- `T_qmm_antiquant_rescale`：int 路径从 GM 到 UB 做 antiquant/rescale
- `T_qmm_mte3`：结果写出相关 MTE3

INT 路径口径：`qmm fixed -> input load(GM/L2 to L1, L1 to L0A/L0B/L0C) -> qmm cube -> output path(FixPipe result handling, INT antiquant/rescale, final MTE3 write)`。

MX 路径口径：`qmm fixed -> input load(GM/L2 to L1, L1 to L0A/L0B/L0C) -> qmm cube -> output path(FixPipe result handling, final MTE3 write)`。

`dynamic quant` 和 qmm 必须分开建模；不同 quant format 会先改变 activation quant、scale/meta，再改变 qmm cube dtype、tile 约束和 qmm 搬运成本。

### 第四步：选择 Target 候选

默认使用 ops-style 默认候选集：4 种 transpose interpretation，每种最多包含 `basic_base` 和 `small_shape_base` 两个 base candidate。分别估算候选的 used core、tail、冷/热数据比例、input movement、cube、FixPipe、antiquant/rescale 和 MTE3，再取最小 `T_target`。

如果调用方指定固定 candidate 或跳过默认候选集，必须把 candidate policy 写入关键假设。小 shape、K/N 不规则、per-block 或 mx 格式不应无说明地沿用单一固定 candidate。

### 第五步：输出收益结论

输出：

- `estimated_speedup = T_baseline / T_target`
- 收益方向：正收益、负收益或不确定
- `dominant_bottleneck`
- `confidence`
- `platform_assumption_confidence`
- `key_assumptions`
- `missing_inputs`

当 `estimated_speedup` 接近 `1`、confidence 低、或关键假设会改变结论时，不要输出强结论。

## 输出要求

默认按 `references/report-template.md` 输出简洁 Markdown 报告。

如果由其他 skill 调用，可只消费报告中的收益方向、confidence 和关键假设。

## 约束规则

- 不声称这是已验证 runtime 结果
- 不把 theoretical path rate 写成真实有效带宽
- 不把固定开销写成平台配置字段
- 不把权重 storage layout 当作主维度；默认使用 `weight_preparation`
- 不输出无法追溯到 `references/theory-model.md` 的主导时间项
- 不在 skill 本体中写死仓库输出路径或推荐决策阈值

## 验证脚本

`validate_qmm.py` 是随 skill 提供的轻量 runtime 验证脚本，用于采集 profiler 数据并验证 estimator 方向是否可靠。

默认口径：

- shape 输入为 `M,K,N`
- baseline 路径是 `bf16 MatMul`
- target 路径是 activation `DynamicQuant + qmm`
- weight 在 profiler 开始前预量化；统计 profiler 时不把 weight `DynamicQuant` 计入 target
- 默认验证 target 顺序为 `int_w8a8,int_w4a4`
- `int_w4a8` validation deferred

示例：

```bash
python skills/quant-tools/dynamic-quant-qmm-analysis/validate_qmm.py \
  --shapes 4096,4096,12800 4096,4096,4096 4096,1024,1024
```
