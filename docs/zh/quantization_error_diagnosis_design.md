# 量化误差诊断与混合精度决策软件设计方案

## 1. 背景

`docs/zh/quantization_error_diagnosis.md` 定义了一套量化误差诊断闭环：从 Block 级筛选、Layer 级下钻，到 MoE 专项分析、W/A 拆因和混合精度决策。

本设计方案的目标是把这套流程产品化，形成一个可复用、可配置、可回归验证的软件模块，集成到 `amct_pytorch` 的量化侧能力中。

## 2. 设计目标

1. 支持对 FP 基线模型和量化后模型做误差诊断。
2. 支持 Block、Layer、Expert 三个粒度的敏感性分析。
3. 支持 QSNR、NMSE、MSE、Relative Error、Cosine Similarity、Max Error 等误差指标。
4. 支持实际 `ActivationQuantizer` 输入的 outlier 分析，用于定位 activation 量化风险入口。
5. 支持 MoE 模型的专家频次和路由稳定性分析。
6. 支持整网 NLL / PPL 消融验证，用真实语言建模损失确认敏感层排序。
7. 支持自动生成 W/A 分离的混合精度建议，并输出结构化报告。
8. 保持原模型不变，所有诊断过程默认 dry-run。

## 3. 设计范围

### 3.1 范围内

- 误差采集与统计
- 分层敏感性分析
- ActivationQuantizer 输入 outlier 分析
- MoE 专项诊断
- 混合精度策略生成
- 诊断报告输出

### 3.2 范围外

- 量化参数训练
- 模型导出与部署
- 任务级评测框架
- 自定义训练逻辑

## 4. 总体架构

建议在 `amct_pytorch/quantization/` 下新增 `diagnostics/` 子包，和现有 `dtypes/`、`modules/` 并列。

```text
输入模型 + 校准数据 + 诊断配置
          ↓
      输入适配层
          ↓
      激活采集层
          ↓
      指标计算层
          ↓
      分析引擎
     /    |     \
 Block  Layer   MoE
      \    |     /
      决策引擎 / 混合精度规划
          ↓
       报告生成
```

### 4.1 模块职责

| 模块 | 职责 |
|---|---|
| 输入适配层 | 统一处理 `tuple` / `dict` / `DataLoader` / 自定义 batch |
| 激活采集层 | 通过 forward hook 采集 Block / Layer / Expert 级别张量，并通过 forward-pre hook 采集 ActivationQuantizer 输入 |
| 指标计算层 | 计算 QSNR、NMSE、MSE、Relative Error、Cosine Similarity、Max Error |
| 输入 Outlier 分析层 | 按 per-token 量化粒度统计实际进入 ActivationQuantizer 的输入 `Token Ratio P99`、`Token Ratio Max` 等指标 |
| MoE 分析层 | 统计 expert frequency、expert error、routing mismatch |
| 整网消融验证层 | 计算 single-layer quant、keep-sensitive FP 的 mean NLL / PPL / gold-token logprob delta |
| 决策引擎 | 根据阈值输出层级精度建议 |
| 报告生成层 | 生成结构化报告和摘要文本 |

## 5. 建议目录结构

```text
amct_pytorch/quantization/diagnostics/
  ├── __init__.py
  ├── api.py
  ├── config.py
  ├── report.py
  ├── collector.py
  ├── metrics.py
  ├── moe.py
  ├── decision.py
  └── utils.py
```

### 5.1 文件职责

| 文件 | 职责 |
|---|---|
| `api.py` | 对外主入口 |
| `config.py` | 诊断配置与阈值定义 |
| `report.py` | 诊断结果数据结构 |
| `collector.py` | hook 采集与缓存管理 |
| `metrics.py` | 通用误差指标计算 |
| `moe.py` | MoE 统计与路由分析 |
| `decision.py` | 混合精度判定规则 |
| `utils.py` | 模块路径解析、命名规范、辅助函数 |

当前工程中的最小实现已经落在 `amct_pytorch/quantization/diagnostics/`，核心文件为 `api.py`、`config.py`、`metrics.py`、`decision.py`、`report.py`。后续可按上表把 collector、moe 等逻辑继续拆分。

## 6. 核心数据结构

### 6.1 诊断配置

```python
@dataclass
class QuantDiagnosisConfig:
    top_k_strategy: str
    top_k_value: float
    block_qsnr_percentile: float
    block_nmse_percentile: float
    layer_upgrade_min_hit_count: int
    moe_enable: bool
    diagnosis_modes: tuple[str, ...]  # "w_only", "a_only", "wa"
    precision_map: dict
    early_stop_enabled: bool
```

### 6.2 指标数据

```python
@dataclass
class BlockMetric:
    name: str
    qsnr: float
    nmse: float
    score: float


@dataclass
class LayerMetric:
    name: str
    qsnr: float
    mse: float
    relative_error: float
    cosine_similarity: float
    max_error: float


@dataclass
class InputOutlierMetric:
    name: str
    index: int
    kind: str
    max_abs: float
    p99_abs: float
    p999_abs: float
    token_ratio_p99: float
    token_ratio_max: float
    max_to_p999: float
    sample_count: int
    token_count: int


@dataclass
class MoEMetric:
    block_name: str
    expert_frequency: dict
    expert_error: dict
    routing_mismatch: float


@dataclass
class PrecisionAdvice:
    name: str
    w_bits: int
    a_bits: int
    source: str  # block / layer / moe / user
    reason: str
    confidence: str


@dataclass
class ModeMetric:
    name: str
    mode: str  # w_only / a_only / wa
    qsnr: float
    nmse: float
    relative_error: float
    cosine_similarity: float


@dataclass
class EndToEndAblationMetric:
    name: str
    mode: str  # single_layer_quant / keep_sensitive_fp
    mean_nll: float
    baseline_nll: float
    delta_nll: float
    gold_logprob_delta: float
    ppl: float
    baseline_ppl: float
    delta_ppl: float
    token_count: int


@dataclass
class QuantDiagnosisReport:
    block_metrics: list[BlockMetric]
    layer_metrics: list[LayerMetric]
    input_outlier_metrics: list[InputOutlierMetric]
    moe_metrics: list[MoEMetric]
    mode_metrics: list[ModeMetric]
    end_to_end_ablation_metrics: list[EndToEndAblationMetric]
    mixed_precision_plan: dict[str, PrecisionAdvice]
    notes: list[str]
```

说明：当前最小实现中的 `precision` 字段是兼容层，含义是“保护等级”。当未运行 W-only / A-only 对照时，它不应被解释为只提升 weight 位宽；默认可落地为 W/A 同步保护，例如 `precision=16` 对应候选 `W16A16`。

## 7. 对外接口设计

建议提供一个主入口，优先适配 AMCT LLM adapter：

```python
diagnose_quantization_error(
    adapter,
    data,
    config=None,
) -> QuantDiagnosisReport
```

其中 `adapter` 应提供 `do_embedding_forward()`、`block()`、`build_quant_block()`、`get_block_forward_kwargs()` 等能力。这样 dense、linear attention、MoE 模型可以共用同一诊断流程。

后续如需支持普通 `torch.nn.Module`，可再增加兼容入口：

```python
diagnose_quantization_error_for_modules(
    fp_model,
    quant_model,
    data,
    config=None,
    batch_adapter=None,
    evaluator=None,
) -> QuantDiagnosisReport
```

### 7.1 参数说明

| 参数 | 说明 |
|---|---|
| `adapter` | AMCT LLM 模型适配器，用于加载 FP block 和构造 fake-quant block |
| `data` | 校准数据或评估数据 |
| `config` | 诊断配置，支持 `dict` 或 `QuantDiagnosisConfig` |

### 7.2 返回值

返回 `QuantDiagnosisReport`，至少包含：

- Block 排名
- Layer 排名
- ActivationQuantizer 输入 outlier 排名
- MoE 分析结果
- W/A 联合或 W/A 分离的混合精度建议
- 告警和跳过原因
- 可读摘要

## 8. 处理流程

### 8.1 输入准备

1. 校验模型可前向推理。
2. 解析数据输入格式。
3. 建立模块命名映射，确定 Block、Layer、Expert 的归属关系。

### 8.2 Block 级筛选

1. 遍历模型 Block。
2. 计算 Block 级 QSNR / NMSE。
3. 按阈值和排序规则筛出 Top-K Block。
4. 输出 Block 保护等级；未做 W/A 拆因时只表示该 Block 存在 W/A 联合输出误差风险。

### 8.3 Layer 级分析

1. 对候选 Block 内部的 Attention / MLP / Linear 分别统计误差。
2. 计算 MSE、Relative Error、Cosine Similarity、Max Error。
3. 按规则判定是否提升 W/A 保护等级。

### 8.4 ActivationQuantizer 输入 outlier 分析

1. 在量化 block 中查找 `ActivationQuantizer`。
2. 通过 forward-pre hook 采集其输入张量，而不是采集 Block 输出。
3. 按 activation 量化粒度做 per-token 统计：把最后一维作为 channel 维，对每个 token 分别计算 `token_max`、`token_p99`、`token_p99.9`、`token_max / token_p99.9`。
4. 对所有 token 的 ratio 聚合输出 `token_ratio_p99`、`token_ratio_max` 和 `token_count`。默认按 `token_ratio_p99` 排序，`token_ratio_max` 只作为极端 token 参考。
5. 将结果作为 activation 量化风险线索输出，不参与 Block Error Score。

### 8.5 MoE 专项分析

1. 统计 expert frequency。
2. 统计 expert quantization error。
3. 计算 routing mismatch。
4. 输出高频专家和路由路径的保精度建议。

### 8.6 决策与验证

1. 汇总各层分析结果。
2. 对候选敏感层执行可选 W-only / A-only / W/A 对照。
3. 对候选敏感层执行可选整网 single-layer quant 和 keep-sensitive FP 消融。
4. 生成包含 `w_bits` 和 `a_bits` 的混合精度映射表。
5. 如提供 `evaluator`，执行最终 PPL / Task Accuracy 验证。
6. 输出通过/失败和回退建议。

### 8.7 W/A 拆因设计

W/A 拆因不是全量默认动作，而是对 Top-K Block 和 Top-N Layer 做二阶段验证，避免全模型三倍扫描成本。

| 模式 | Weight | Activation | 输出 |
|---|---|---|---|
| `w_only` | fake quant | FP / BF16 | weight 误差贡献 |
| `a_only` | FP / BF16 | fake quant | activation 误差贡献 |
| `wa` | fake quant | fake quant | 真实部署候选误差 |

拆因决策规则：

1. `w_only` 接近 `wa`：生成 `W16A8` 或 `W8A8` 建议，优先调 weight 位宽或 weight PTQ 算法。
2. `a_only` 接近 `wa`：生成 `W8A16` 或 `W16A16` 建议，优先调 activation clipping、smooth 或 activation 位宽。
3. `w_only` 和 `a_only` 单独较低但 `wa` 高：生成 `W16A16` 建议，标记为耦合误差。
4. MoE routing mismatch 高：router 输入、router 权重和 expert 输出分别拆因；router 保护优先级高于专家权重压缩。

### 8.8 整网 NLL / Logit 消融设计

整网消融验证层用于把局部误差指标落到真实语言建模目标上。它不替代 QSNR / NMSE / 输入 outlier 的归因能力，但应作为敏感层排序和混合精度保护优先级的最终证据。

| 模式 | 行为 | 输出 |
|---|---|---|
| `single_layer_quant` | 每次只量化一个 layer，其它 layer 保持 FP / BF16 | 该 layer 单独量化造成的 `delta_nll` / `delta_ppl` |
| `keep_sensitive_fp` | 其它 layer 量化，候选敏感 layer 保持 FP / BF16 | 保护候选层后能恢复多少全量量化损失 |
| `single_layer_wa_split` | 候选层分别跑 W-only / A-only / W/A | 判断整网损失来自 weight、activation 还是耦合误差 |

指标定义：

```text
mean_nll = sum(cross_entropy(logits, labels)) / token_count
ppl = exp(mean_nll)
delta_nll = mean_nll_quant - mean_nll_fp
gold_logprob_delta = -delta_nll
delta_ppl = ppl_quant - ppl_fp
```

判读规则：

1. `delta_nll` 越大，表示整网 gold token 平均 NLL 恶化越明显，该层越应优先保护。
2. `gold_logprob_delta` 越负，表示 gold token 平均 log probability 下降越明显。
3. 在同一数据集、同一 token 集合和同一 baseline 下，`ppl = exp(mean_nll)`，所以 `delta_nll` 与 `delta_ppl` 排序应单调一致。
4. 如果 QSNR / NMSE / 输入 outlier 排序与整网 NLL 消融冲突，最终敏感层排序以整网 NLL / PPL 为准，局部指标用于解释误差来源。

## 9. 判定规则设计

### 9.1 默认阈值

| 项目 | 默认值 |
|---|---|
| Top-K Block | `max(4, ceil(Block数 * 0.1))`，上限 8 |
| 输出误差敏感 Block 判定 | QSNR 底部 10% 或 NMSE 顶部 10% |
| 输入 outlier 关注点 | `token_ratio_p99` 排名前列，辅看 `token_ratio_max` |
| Layer 升级 | 指标集合中至少 2 项恶化 |
| 整网损失敏感层 | `delta_nll` 或 `delta_ppl` 排名前列 |
| MoE 路由异常 | 专家一致率下降或频次偏移超过 20% |

### 9.2 决策优先级

1. Router / 路由相关路径
2. 输入 outlier 明显且 A-only 误差高的 activation 量化入口
3. 高频专家
4. Attention
5. MLP / Linear

### 9.3 混合精度输出语义

| 诊断结论 | 推荐表达 | 含义 |
|---|---|---|
| 未拆因，只知道路径敏感 | `protect=16` 或 `W16A16(candidate)` | W/A 联合保护候选，不归因到 weight |
| weight 主导 | `W16A8` / `W8A8` | 主要提升 weight 位宽 |
| activation 主导 | `W8A16` / `W16A16` | 主要提升 activation 位宽或裁剪策略 |
| W/A 耦合 | `W16A16` | weight 和 activation 同时保护 |
| MoE routing 主导 | `router=BF16/W16, expert=W8/W16` | 优先保证路由稳定性 |

### 9.4 早停规则

- 前 1 到 2 个 Block 已能稳定定位误差源时，停止扩展 Top-K。
- Block 级和 Layer 级已足够定位时，跳过更深层扫描。
- 模型不含 MoE 时，跳过 MoE 分支。

## 10. 复用与扩展

### 10.1 可复用能力

- 参考 `amct_pytorch/pruning/diagnostics.py` 的 dry-run 和 report 风格。
- 参考 `accuracy_based_auto_calibration` 的敏感度评估模式。
- 参考 `SensitivityBase` 的可插拔指标设计。

### 10.2 扩展点

- 自定义 Block 划分规则
- 自定义指标实现
- 自定义阈值和早停策略
- 自定义 MoE 路由稳定性定义
- 自定义混合精度映射规则

## 11. 日志与错误处理

### 11.1 约束

- 默认不修改原模型。
- 任何可选分支失败都应写入 `notes`，不直接中断主流程。
- 采集失败时，应降级输出已有结果。

### 11.2 建议日志

- 未发现可诊断目标
- 某个 Block 无法完成 forward probe
- MoE 分支被跳过
- 量化误差已被单一敏感层解释
- 混合精度建议已生成

## 12. 验收标准

1. 给定同一模型和数据，多次运行结果稳定。
2. 原模型不被修改。
3. 能输出完整的 Block / Layer / MoE / 策略报告。
4. 能生成可执行的混合精度配置。
5. 报告中必须区分“W/A 联合敏感”与“weight-only 敏感”，未做 W/A 拆因时不得写成 weight-only 建议。
6. 默认报告不做 Block 输出 outlier 分析；输入 outlier 只统计实际进入 `ActivationQuantizer` 的张量，并按 per-token activation 量化粒度计算。
7. 报告中必须明确整网 NLL / logprob delta 的口径：它来自最终 logits 与 gold labels 的 token-level loss，不是局部 hidden-state 指标。
8. 能在最终验证阶段给出明确通过/失败结论。

## 13. 结论

这个软件设计方案的核心是把“误差分析”做成一个可复用的诊断引擎，而不是一组分散的统计脚本。
它应当作为 `amct_pytorch` 量化侧的诊断能力，与现有量化、自动校准和剪枝诊断形成一致的产品边界。
