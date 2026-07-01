---
name: algorithm-recommendation
description: 为已适配模型推荐下一步值得尝试的量化算法（仅限已注册：lwc/lac/omniquant/autoround）。触发场景：已知直转掉点想按特征选后续算法 / 无直转结果时先据 casebook+结构+仓库算法能力给先验候选。不适用于：跑算法训练、判算法收益（执行→ $quant-run，判读→ $algorithm-validation）。
---

# 量化算法推荐

## 概述

为已适配模型推荐**下一步值得尝试的量化算法**（只推荐、不跑实验）：回答“先试哪一类 / 哪个具体算法、各适合什么掉点特征与模块、先试失败切哪个更合理”。支持两种输入：**已知直转结果**（有 `ppl/delta`，按掉点 + 目标模块推荐）或 **先验推荐**（无直转结果，据 casebook + 结构 + 仓库算法能力给候选）。**不负责**：运行算法实验、产出 `ppl/delta`、误差定位、下“最终最优算法”结论。

---

## 用户调用格式

- 用户明确点名调用，先返回本 skill 固定输入模板再执行。
- 不替用户改成别的 skill；仅当用户没指定 skill、只描述目标时，才建议是否回 `$quant-workflow`。
- 固定输入模板见 [../references/skill-input-template.md](../references/skill-input-template.md)。

---

## 先读哪些文件

1. `.agents/docs/casebook/...` 对应系列 `README.md`
2. 已有个案再读最相近 `<series>/<case>.md`
3. 确认量化口径读 [../references/metrics-and-thresholds.md](../references/metrics-and-thresholds.md)
4. 确认直转判读规则读 [../references/direct-quant.md](../references/direct-quant.md)
5. 可用算法**以 new-path `register_algorithms()` 实际 import 的为准** = `autoround / lac / lwc / omniquant`（gptq/awq/mxfp 视分支移植，先确认在 ALGO_REGISTRY）；**别用 `grep amct_pytorch/algorithms` 命中名册判定可用**——classic 图量化 path 列了 gptq/awq 等但 LLM PTQ 不走，据此选会运行期 `KeyError: '...' is not registered in 'algo'`。

## 当前算法分组

**可用算法 = new-path `ALGO_REGISTRY` 已注册的四个**（`register_algorithms()` 实际 import），按类型理解：

- `weight` 类：`lwc`、`autoround`
- `activation` 类：`lac`
- `structure` 类：`omniquant`

> **不要推荐以下未注册算法**：`awq` / `gptq` / `svdquant` / `smoothquant` / `flatquant` / `learnable_had`（源在 AMCT-Q，需分支移植后才进 `ALGO_REGISTRY`；当前 `--algos` 传入会运行期 `KeyError: '...' is not registered in 'algo'`）。upstream 示例脚本里出现这些名字也不代表可用。

推荐时必须考虑当前模块类型和 registry `targets` 是否匹配。

## 固定流程

1. 先判断当前属于哪种输入模式：
   - 已知直转量化结果
   - 先验经验推荐
2. 先明确当前上下文：
   - 模型名称 / 系列
   - 结构标签：`dense / moe / packed-expert / special-attn`
   - 目标模块
   - 评测口径：数据集、指标、`seq_len`、`granularity`
3. 先查现有结果：
   - 系列 `casebook` 是否已有算法经验
   - `outputs/` 或日志里是否已有同模型、同模块、同口径的算法结果
   - 当前结构和历史案例是否足够相似
4. 如果已知直转结果，先根据当前掉点判断更该试哪类算法：
   - 权重量化主导的问题，优先考虑 `weight` 类算法
   - 激活裁剪或激活范围明显异常，优先考虑 `activation` 类算法
   - 跨层缩放、结构变换、模块耦合更明显时，优先考虑 `structure` 类算法
5. 如果还没有直转结果，只给先验推荐，并明确证据强度较低。
6. 至少给出 2 到 3 个候选算法：
   - 默认首推
   - 保守替代
   - 激进替代
7. 对每个候选算法说明：
   - 适用模块
   - 适用掉点特征
   - 预期成本
   - 主要风险
8. 输出“量化算法推荐卡”。

## 边界规则

- 这是**算法推荐**，不是算法实验。
- 如果用户已经明确指定算法并要看效果，转到后续算法验证流程。
- 如果用户只想先拿第一轮直转方案，回到 `$scheme-recommendation`。
- 如果模型还没适配完成，回到 `$model-adapter`。
- 如果当前还没有直转结果，可以做先验推荐，但要明确写出证据强度下降。
- 不要按算法名热度或外部印象推荐，优先参考仓库里已注册算法、casebook 和现有结果。
- 推荐时不要跨越 registry `targets` 乱配算法。
- 引用仓库内代码、脚本和文档路径时，一律使用仓库相对路径；不要写绝对路径。

## 输出要求

结束时必须给出一张“量化算法推荐卡”，至少包含下面字段：

- `输入模式`
- `当前背景`
- `已有直转结果`
- `候选算法`
- `首推算法`
- `推荐理由`
- `适用条件`
- `主要风险`
- `后续验证建议`
- `证据强度`

## 量化算法推荐卡

```md
# 量化算法推荐卡

## 输入模式
- 已知直转结果 / 先验经验推荐

## 当前背景
- 模型名称：
- 模型系列：
- 结构标签：
- 目标模块：
- 评测口径：

## 已有直转结果
- `ppl_bf16`：
- `ppl_quant`：
- `delta`：
- 当前主要问题：

## 候选算法
- 候选算法 A（首推）：
  - 算法名：
  - `targets`：
  - 适用原因：
  - 预期成本：
  - 主要风险：
- 候选算法 B（保守）：
  - 算法名：
  - `targets`：
  - 适用原因：
  - 预期成本：
  - 主要风险：
- 候选算法 C（激进）：
  - 算法名：
  - `targets`：
  - 适用原因：
  - 预期成本：
  - 主要风险：

## 首推算法
- 算法名：
- 适用模块：
- 适用掉点特征：
- 为什么先试它：

## 推荐理由
- 同系列经验：
- 结构相似性：
- 与当前问题的匹配性：
- 如果失败，下一步是否容易切换：

## 适用条件
- 前置条件 1：
- 前置条件 2：
- 前置条件 3：

## 主要风险
- 风险 1：
- 风险 2：
- 风险 3：

## 后续验证建议
- 首轮验证应固定哪些变量：
- 如果验证成功：
- 如果验证失败：

## 证据强度
- 强 / 中 / 弱
- 判断理由：
```
