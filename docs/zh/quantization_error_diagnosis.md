# 量化误差诊断与混合精度决策流程

本文给出一套面向大模型量化的误差诊断流程。目标不是只找出“哪个层最差”，而是把误差来源、敏感位置和混合精度策略串成一个可执行闭环。

## 优化原则

1. 先粗后细：先做 Block 级筛选，再下钻到 Layer 和子模块。
2. 指标分工明确：QSNR / NMSE 负责当前层量化输出误差排序，Input Outlier 只分析实际进入 `ActivationQuantizer` 的输入分布，MSE / Relative Error / Cosine Similarity / Max Error 负责确认。
3. MoE 作为分支处理：只有在模型包含 MoE 时，才进入专家频次和路由分析。
4. 结果要可落地：最终输出必须能直接转成混合精度配置，而不是只停留在分析图表。

## 输入与输出

### 输入

- FP 基线模型
- 校准数据集
- 量化后模型或 fake quant 模型
- 统计粒度定义，例如 Block、Layer、Expert
- 业务容差，例如允许的 PPL 掉点或任务精度损失

### 输出

- Block 输出误差敏感排名
- ActivationQuantizer 输入 outlier 排名
- 敏感 Layer 排名
- MoE 分析结果（如适用）
- 混合精度配置建议
- 最终验证结果

## 适用场景

- 大模型从 FP 基线向低比特量化迁移时，先做误差定位再定混合精度。
- 量化结果出现明显掉点，需要区分是 Block、Layer 还是 MoE 路由问题。
- MoE 模型在专家频次、路由稳定性或专家误差上出现异常。

## 不适用场景

- 只想做一次性黑盒压测，不打算回到模型结构层面定位问题。
- 没有校准集，或者校准集和真实业务分布差异过大。
- 量化目标和验证指标尚未明确，无法判断“通过”标准。

## 优化后的流程

```text
FP 基线模型
  ↓
1. 准备校准集与统计口径
   - 固定校准数据
   - 统一统计粒度与对齐方式
  ↓
2. Block 级输出误差敏感性筛选
   - 计算每个 Block 的 QSNR / NMSE
   - 按敏感度排序，得到 Top-K Block
  ↓
3. Block 内部 Layer 级分析
   - Attention / MLP / Linear 分别计算 QSNR
  ↓
4. ActivationQuantizer 输入 outlier 分析
   - 只采集实际进入激活量化器的输入张量
   - 按 per-token 口径统计 Token Max / Token P99 / Token P99.9 / Token Ratio P99 / Token Ratio Max
  ↓
5. 定位敏感 Layer
   - 选出误差最集中的层或子模块
  ↓
6. Quantization Error 深入分析
   - MSE
   - Relative Error
   - Cosine Similarity
   - Max Error
  ↓
7. MoE 专项分析（仅 MoE 模型）
   - Expert Frequency
   - Expert Quantization Error
   - Routing Mismatch
  ↓
8. W/A 拆因
   - W-only
   - A-only
   - W/A
  ↓
9. 整网损失消融验证
   - Single-layer quant delta NLL / delta PPL
   - Keep-sensitive FP delta NLL / delta PPL
   - Gold-token logprob delta
  ↓
10. 制定混合精度策略
   - 敏感层：W/A 联合保护，优先 W8A8 / W16A16
   - 一般层：可作为 W4A8 / W4A4 候选
   - MoE 路由与高频专家优先保留更高精度
  ↓
11. 最终验证
   - PPL
   - Task Accuracy
```

## 建议的判定规则

### 1. Block 级筛选

- 先用 QSNR / NMSE 做全局排序，不建议一开始就直接看最细层。
- Top-K 可以按固定数量、比例，或按“累计误差贡献”确定。
- 如果某个 Block 表现出低 QSNR 或高 NMSE，优先判定为当前 Block 输出误差敏感候选。
### 2. ActivationQuantizer 输入 outlier

- 输入 outlier 只统计实际进入 `ActivationQuantizer` 的输入张量，例如 attention 的 `inp_afq/o_proj_afq`、MLP 的 `input_quant/hidden_quant`、MoE expert 的 activation quantizer。
- 不统计 Block output 作为 outlier 结论，因为 Block output 不一定在当前边界被量化。
- activation fake quant 当前采用 per-token 动态 scale，因此输入 outlier 必须按 token 维度统计，不能把整个 tensor flatten 后直接统计。
- 推荐口径：对每个 token 的最后一维向量计算 `token_max / token_p99.9`，再对所有 token 的 ratio 统计 `token_ratio_p99` 和 `token_ratio_max`。
- `token_ratio_p99` 偏高说明该激活量化入口在较多 token 上存在长尾输入，可能导致 activation quantization 对这些 token 过度敏感；`token_ratio_max` 只表示最极端 token，不建议单独作为排序依据。
- 输入 outlier 只是归因线索，不能单独证明该层是最终敏感层；必须和 Layer 输出误差、W/A 拆因或 PPL 结果一起判断。

### 3. Layer 级与误差剖面

- Attention、MLP、Linear 不要平铺平均看，建议按模块单独排名。
- MSE 适合衡量总体偏移，Cosine Similarity 适合判断方向是否保持，Max Error 适合定位极端失真。
- 当多个指标同时恶化时，该层通常值得提升到更高保护等级。默认建议应表达为 `W8A8`、`W16A8`、`W8A16` 或 `W16A16`，避免只写 `W8/W16` 造成“只针对 weight”的误解。
- 若当前诊断只运行了 W/A 同时量化实验，则只能说明“该路径对量化敏感”，不能直接判定误差完全来自 weight 或 activation。

### 4. MoE 专项

- 先看 Expert Frequency，再看 Expert Quantization Error。
- 如果少数专家频次极高，优先保护这些专家。
- 如果路由结果在量化前后变化明显，通常需要优先处理 router 或相关归一化路径，而不是只调专家权重。

### 5. W/A 拆因对照

为了判断敏感来源是 weight、activation 还是二者耦合，建议在候选敏感层上增加三组对照：

| 对照组 | Weight | Activation | 用途 |
|---|---|---|---|
| W-only | 量化 | FP / BF16 | 判断权重量化是否主导误差 |
| A-only | FP / BF16 | 量化 | 判断激活量化是否主导误差 |
| W/A | 量化 | 量化 | 判断真实部署口径下的联合误差 |

判读方式：

1. W-only 误差接近 W/A，A-only 较小：优先提高 weight 位宽或改 weight PTQ 算法。
2. A-only 误差接近 W/A，W-only 较小：优先提高 activation 位宽、做 clipping 或 smooth。
3. W-only 与 A-only 都不高，但 W/A 明显恶化：说明存在 W/A 耦合误差，优先保护整条子路径。
4. MoE routing mismatch 高时，应把 router 输入、router 权重和 expert 输出分别纳入 W/A 对照。

### 6. 整网 NLL / Logit 消融验证

QSNR / NMSE / outlier 是局部诊断指标，只能说明某个输出或量化入口的误差形态；最终敏感层应通过整网语言建模损失或任务指标验证。推荐对候选层增加整网消融：

| 消融方式 | 口径 | 用途 |
|---|---|---|
| Single-layer quant | 每次只量化一个 layer，其它 layer 保持 FP / BF16 | 判断该层单独量化带来的真实整网损失 |
| Keep-sensitive FP | 其它层量化，候选敏感层保持 FP / BF16 | 验证保护这些层是否能恢复全量量化掉点 |
| W-only / A-only / W/A | 只在候选层上拆分 weight 和 activation | 判断误差来源是 weight、activation 还是二者耦合 |

推荐记录 `mean_nll`、`delta_nll`、`gold_logprob_delta`、`ppl` 和 `delta_ppl`：

```text
mean_nll = token-level cross entropy 的平均值
ppl = exp(mean_nll)
delta_nll = mean_nll_quant - mean_nll_fp
gold_logprob_delta = -delta_nll
delta_ppl = ppl_quant - ppl_fp
```

在同一数据集、同一 token 集合和同一 baseline 下，`ppl = exp(mean_nll)`，因此按 `delta_nll` 与按 `delta_ppl` 排序应当单调一致。`gold_logprob_delta` 为负表示量化后 gold token 的平均 log probability 下降。这个指标是整网结果，不是某个 layer 的局部 hidden-state 误差。

## 推荐阈值

下面阈值适合作为默认起点，实际项目可按模型规模、数据分布和业务容差再调。

| 项目 | 推荐阈值 | 含义 |
|---|---|---|
| Top-K Block | `max(4, ceil(Block 数 * 0.1))`，上限 8 | 先保留最敏感的一小部分 Block |
| 输出误差敏感 Block 判定 | QSNR 处于底部 10% 或 NMSE 处于顶部 10% | 认为该 Block 值得优先下钻 |
| 输入 outlier 关注点 | `token_ratio_p99` 排名前列，辅看 `token_ratio_max` | 只作为 activation 量化风险线索，不直接替代误差排序 |
| Layer 级升级 | MSE、Relative Error、Cosine Similarity、Max Error 中至少 2 项恶化 | 值得提升 W/A 保护等级 |
| 整网 NLL 消融敏感层 | `delta_nll` 或 `delta_ppl` 排名前列 | 作为混合精度保护层的优先依据 |
| MoE 路由异常 | Top-1 或 Top-K 专家一致率明显下降，或专家频次偏移超过 20% | 路由稳定性受到量化影响 |
| 结果通过 | PPL 掉点满足业务容差，或任务精度满足预设阈值 | 进入可交付状态 |

## 早停与跳过条件

- 如果前 1 到 2 个 Block 已经能稳定定位误差源，可以停止继续扩展 Top-K。
- 如果模型本身不含 MoE，直接跳过 MoE 专项分析。
- 如果某个 Block 的 Layer 级分析已能明确定位到单一敏感层，可以不再对同类层做全量扫描。
- 如果 QSNR / NMSE 无明显异常，可以优先保持 `W4A8` 或 `W4A4` 候选，不必盲目升精度。

## 注意事项

- 阈值是默认起点，不是绝对标准；模型越大、分布越偏，越需要结合业务重新校准。
- 不建议只看单一指标下结论，至少要同时看一个排序指标和一个误差解释指标。
- QSNR / NMSE / 输入 outlier 不能直接等价为 PPL 敏感层；当它们与整网 NLL/PPL 消融冲突时，应以整网消融作为敏感层结论，以局部指标作为归因线索。
- 如果一个层既是高频路径又是高误差层，应优先保守处理，而不是继续压位宽。

## 执行步骤

1. 固定校准集和统计口径，避免不同实验结果不可比。
2. 先对所有 Block 计算 QSNR / NMSE，筛出候选 Top-K。
3. 对实际 `ActivationQuantizer` 输入做 per-token outlier 统计，作为 activation 量化归因线索。
4. 对命中的 Block 继续做 Layer 级分析，定位 Attention、MLP、Linear 中的敏感层。
5. 对敏感层做误差剖面分析，确认误差是否已影响表示方向或局部峰值。
6. 若模型含 MoE，则补做专家频次和路由分析。
7. 对候选敏感层做 W-only / A-only / W/A 拆因。
8. 对候选层做整网 single-layer quant 和 keep-sensitive FP 消融，按 `delta_nll` / `delta_ppl` 确认真实敏感层。
9. 根据规则生成混合精度方案。
10. 回归验证 PPL / Task Accuracy，判断是否达到业务容差。

## 混合精度落地规则

建议用“命中任一条件即升级”的方式生成混合精度配置：

1. Block 命中低 QSNR / 高 NMSE 判定，优先从 `W4A8/W4A4` 升到 `W8A8`。
2. Layer 命中敏感判定，且误差确认通过，根据 W/A 拆因结果升到 `W16A8`、`W8A16` 或 `W16A16`。
3. MoE 的 router 或高频专家命中异常，优先单独保精度；routing mismatch 高时 router 路径不应只做 weight 升级。
4. 未命中的普通层保持 `W4A8/W4A4` 候选，避免过度保守。

当同一层同时命中多个高风险条件时，优先级通常为：

1. Router / 路由相关路径
2. 高频专家
3. Attention
4. MLP / Linear

## 输出物

建议最终产出以下结果，便于直接进入混合精度配置阶段：

- 输出误差敏感 Block 列表
- ActivationQuantizer 输入 outlier 列表
- 敏感 Layer 列表
- MoE 专项分析结果
- 混合精度映射表
- 最终 PPL / Task Accuracy 验证结果

## 记录模板

建议每次诊断至少记录以下字段，便于复现实验和横向比较：

| 字段 | 示例 |
|---|---|
| 模型名称 | Qwen / DeepSeek / 自研模型 |
| 量化目标 | W4A8 / W8A8 / 混合精度 |
| 校准集 | 校准集名称或版本号 |
| Top-K 规则 | 10% / 8 层上限 |
| 输出误差敏感 Block | `block.3`, `block.7` |
| 输入 outlier 入口 | `block.3.mlp.hidden_quant` |
| 敏感 Layer | `attn.q_proj`, `mlp.up_proj` |
| MoE 结论 | 高频专家受影响 / 无 MoE |
| 最终策略 | Attention W8A8，MLP W4A8 |
| 验证结果 | PPL +0.08，任务精度下降 0.2% |

## 推荐的混合精度策略

| 对象 | 推荐 W/A 策略 | 处理方式 |
|---|---|---|
| 明显敏感的 Attention 层 | W8A8 / W16A8 / W16A16 | 依据 W/A 拆因决定保护 weight、activation 还是整条路径 |
| 高误差 MLP / Linear 层 | W8A8 / W16A8 | 视误差程度和 W-only 对照决定是否升级 weight |
| 激活量化主导的层 | W8A16 / W16A16 | 依据 A-only 对照提高 activation 位宽或调整 activation 量化策略 |
| 普通稳定层 | W4A8 / W4A4 | 保持低比特 |
| MoE Router / 高频专家 | W16A16 / BF16 局部保留 | 优先保护路由稳定性 |

## 结论

这套流程的核心是把“找问题”改成“找原因并形成策略”。最终目标不是让所有层都过一遍指标，而是用最少的分析成本，稳定地得到可部署的混合精度方案。
