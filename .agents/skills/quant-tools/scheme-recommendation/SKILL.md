---
name: scheme-recommendation
description: 为已适配模型推荐第一轮量化方案。适用于用户只想先拿量化方案参考、想基于 casebook 与理论性能评估选择直转起点、或在正式量化实验前先确定候选方案、风险点、升级条件和回退路径。
---

# 量化方案推荐

## 用户调用格式

- 如果用户明确点名调用这个 skill，先返回这个 skill 对应的固定输入模板，再开始执行。
- 不替用户改成别的 skill；只有当用户没有指定 skill、只是描述目标时，才建议是否回到 `$model-workflow` 或 `$quantization`。
- 固定输入模板见：[../../reference/skill-input-template.md](../../reference/skill-input-template.md)

这个 skill 只用于**推荐第一轮量化方案**。

它回答的问题是：

- 第一枪先量哪些模块
- 用什么 `dtype / bits / algos / granularity`
- 为什么先从这个方案起步
- 候选方案中的关键模块是否已有可复用的理论性能评估，哪里需要补评估或回退
- 如果失败，先缩哪里、查哪里

它不负责：

- 真正跑量化实验
- 产出 `ppl_bf16 / ppl_quant / delta`
- 做误差定位
- 下“最终最优方案”结论

按下面顺序阅读：

1. `../../../docs/agent/casebook/...` 中对应系列的 `README.md`
2. 如已有个案，再读最相近的 `<series>/<case>.md`
3. 如需确认当前能力边界，再读 [../../../docs/agent/repo-map.md](../../../docs/agent/repo-map.md)
4. 如候选方案涉及 Linear `dynamic quant + qmm`，按 `QMM Estimator 使用规则` 调用 `$dynamic-quant-qmm-analysis` 判断收益方向
5. 如需确认当前量化口径，再读 [../../tasks/quantization/references/metrics-and-thresholds.md](../../tasks/quantization/references/metrics-and-thresholds.md)

## 固定流程

1. 先明确当前任务口径：
   - 模型名称 / 系列
   - 结构标签：`dense / moe / packed-expert / special-attn`
   - 评测集、指标、`seq_len`、`granularity`
   - 时间预算、算力预算、是否优先保守
2. 先查现有结果：
   - 系列 `casebook` 是否已有默认起步方案
   - `outputs/` 或日志里是否已有相同口径的量化结果
   - 当前模型和历史案例是否足够相似
3. 按相似度筛选参考经验，优先级是：
   - 同系列
   - 同结构
   - 同量化目标
   - 同评测口径
4. 对候选方案中涉及 Linear `dynamic quant + qmm` 的模块，按 `QMM Estimator 使用规则` 调用 `$dynamic-quant-qmm-analysis` 判断收益方向。
5. 至少给出 2 到 3 个候选方案：
   - 保守方案
   - 默认首推方案
   - 激进覆盖方案
6. 从候选中选一个“首推方案”，并明确理由：
   - 证据最强
   - 成本最低
   - 失败后最容易定位
7. 用经过证据筛选后的最终保留模块填写风险点、升级条件和回退路径；若 qmm estimator 结论改变了首推方案，必须同步更新这些字段。
8. 输出“量化方案推荐卡”，卡片内容以经过证据筛选后的最终保留方案为准。

## QMM Estimator 使用规则

- 候选方案涉及 Linear `dynamic quant + qmm` 时，调用 `$dynamic-quant-qmm-analysis` 判断 Linear/projection 子路径收益方向；qmm 不判断完整 attention module 性能。
- 调用前准备 `M/K/N/platform/quant_scheme`：`K/N` 来自 Linear 权重 shape 或模型配置推导；`M` 来自 workload 展开后的代表性 token/row 数；`platform` 和 `quant_scheme` 来自当前硬件、部署目标和候选量化方案。缺少关键字段时，不要伪造收益方向，应优先向用户确认补字段。
- 调用前按模型脚本、量化 wrapper 或部署图确认比较单元。比较单元可以是 single Linear、shared activation quant group、fused/packed projection；shared activation group 的 baseline 是多个 float/bf16 matmul，target 是一次 activation quant 加多个 qmm；fused/packed projection按实际情况比较。
- recommendation 只消费收益方向：`estimated_speedup > 1` 为正收益倾向，`estimated_speedup < 1` 为负收益倾向，接近 `1` 或低 confidence 写成不确定或待验证。qmm 结论只能作为第一轮方案选择和风险标注依据，不得写成已验证 runtime 收益。
- qmm 单 Linear 负收益不自动改变候选方案排序。只有 Linear-dominant 模块内相关比较单元整体明确负收益时，才可建议当前仓支持的模块级回退，例如 `mlp` 或 `moe expert`。
- attention projection Linear 负收益不能单独触发完整 `attn` 回退。若当前 target 明确是 `attn-linear`，可建议回退 `attn-linear`；若当前 target 是完整 attention，需要 full attention estimator、runtime validation 或用户显式确认后才建议完整 `attn` 回退。

## 边界规则

- 这是**起步方案推荐**，不是全仓最优方案搜索。
- 不要按全仓平均经验直接推荐，要优先参考同系列、同结构案例。
- 如果评估依据不足，要明确写“弱推荐”或“待验证”，不要装作确定。
- 如果用户已经明确给了量化方案，停止推荐，转到评测或验证流程。
- 如果模型还没适配完成，停止推荐，回到 `$model-adapter`。
- 如果当前仓库里已有同口径、同模型、同目标的稳定结论，优先复用，不重复发明方案。
- qmm 收益方向要映射到现有推荐卡字段，不新增卡片结构。
- 当前推荐卡主字段仍按模块级表达，不强制做子算子级拆分。
- `Conditional Go` 不直接触发回退，但必须进入风险点或待验证说明。
- 理论评估只能作为弱到中等依据；除非有稳定 casebook 佐证，不要把它写成“已验证结论”。
- `planned / unsupported` 只能表示后续待补，不得直接写成“当前建议启用”。
- 引用仓库内代码、脚本和文档路径时，一律使用仓库相对路径；不要写绝对路径。

## 输出要求

结束时必须给出一张“量化方案推荐卡”，至少包含下面字段：

- `当前模型`
- `任务口径`
- `相似案例`
- `候选方案`
- `首推方案`
- `推荐理由`
- `风险点`
- `升级条件`
- `回退方案`
- `证据强度`

在不新增卡片字段的前提下，qmm 收益方向要映射到现有字段：

- `首推方案`：按当前仓可执行粒度填写；qmm 单 Linear 风险可写入风险点，不要直接写成当前不可执行的 Linear 级启停
- `推荐理由`：优先写系列 casebook 和 qmm estimator 的正收益倾向
- `风险点`：写 qmm 负收益倾向、不确定收益、待评估项，以及当前缺失的量化粒度、中间结果物化方式、平台假设等不确定项
- `升级条件`：写哪些风险需要在下一轮通过补 qmm 理论评估、补更多代表性 shape 或补更具体的平台假设进一步确认
- `回退方案`：只写当前仓实际支持的回退粒度；qmm 相关回退必须遵守上方 `QMM Estimator 使用规则` 的回退边界
- `证据强度`：写明主结论依据来自系列 casebook、qmm estimator 理论评估，还是暂时只能回退到结构经验；若多个来源混用，以主结论来源为主，并补充当前未覆盖范围

## 量化方案推荐卡

```md
# 量化方案推荐卡

## 当前模型
- 模型名称：
- 模型系列：
- 结构标签：
- 关键结构特征：

## 任务口径
- 数据集：
- 指标：
- `seq_len`：
- `granularity`：
- 时间预算：
- 算力预算：
- 特殊约束：

## 相似案例
- 案例 1：
  - 来源：
  - 相似原因：
  - 采用方案：
  - 结果摘要：
- 案例 2：
  - 来源：
  - 相似原因：
  - 采用方案：
  - 结果摘要：
- 案例 3：
  - 来源：
  - 相似原因：
  - 采用方案：
  - 结果摘要：

## 候选方案
- 候选方案 A（保守）：
  - 量化模块：
  - `quant_dtype`：
  - `a_bits / w_bits`：
  - `algos`：
  - 预期成本：
  - 预期风险：
- 候选方案 B（首推）：
  - 量化模块：
  - `quant_dtype`：
  - `a_bits / w_bits`：
  - `algos`：
  - 预期成本：
  - 预期风险：
- 候选方案 C（激进）：
  - 量化模块：
  - `quant_dtype`：
  - `a_bits / w_bits`：
  - `algos`：
  - 预期成本：
  - 预期风险：

## 首推方案
- 量化模块：
- `quant_dtype`：
- `a_bits / w_bits`：
- `algos`：
- `granularity`：

## 推荐理由
- 同系列经验：
- 结构相似性：
- 成本与收益平衡：
- 失败后是否容易定位：

## 风险点
- 风险 1：
- 风险 2：
- 风险 3：

## 升级条件
- 保持当前方案的条件：
- 进入误差定位的条件：
- 进入 PTQ 的条件：

## 回退方案
- 回退路径 1：
- 回退路径 2：
- 回退路径 3：

## 证据强度
- 强 / 中 / 弱
- 判断理由：
```
