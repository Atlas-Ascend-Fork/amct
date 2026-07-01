---
name: algorithm-validation
description: 判读 PTQ 算法收益（reviewer 用，只读不跑）：读 quant-run 跑出的 ppl_algo 与直转参考 ppl_direct，算 delta_gain、判该算法相对直转是否真改善。触发场景：PTQ 算法结果已出、要判相对直转的收益并给停 / 换算法 / 缩范围建议。不适用：跑 ptq / 评测命令（→ $quant-run）。
---

# PTQ 算法收益判读

reviewer 的判读 skill：对 `quant-run` 已跑出的 PTQ 算法结果做收益判定。**只读结果、不跑命令**——ptq 训练与结果评测都归 `$quant-run`（implementer）。

它回答的问题是：

- 该算法相对直转量化有没有改善精度、改善多少、值不值得继续
- 没改善时该停、换算法还是缩范围

它不负责：

- 跑 ptq / 评测命令（→ `$quant-run`）
- 推荐先试哪个算法（→ `$algorithm-recommendation`）

按需读：[../references/metrics-and-thresholds.md](../references/metrics-and-thresholds.md)。

## 固定流程

1. 从 progress 读 `quant-run` 写入的：当前算法 + 目标模块 + `ppl_algo`，以及同位宽**直转参考** `ppl_direct`。缺数则回报：需先由 implementer 跑 `$quant-run`（含 ptq 训练 + 带 `--algos` 的 PTQ 结果评测）。
2. 基准必须是**同位宽直转**（不是 BF16）。计算：
   - `delta_direct = ppl_direct - ppl_bf16`
   - `delta_algo = ppl_algo - ppl_bf16`
   - `delta_gain = delta_direct - delta_algo`
3. 判读：
   - `delta_algo` 明显优于 `delta_direct`（`delta_gain` 为正且可观）→ 算法值得继续；
   - 改善很小 / 不稳定 → 建议停用或换算法（W8A8 直转常已近最优、PTQ 无收益属正常，不是失败）；
   - `ppl_algo` 明显优于 BF16 很多 / 异常 → 先让 implementer 查链路。
   - 注意 solver 打印的 loss 经归一，跨 epoch 不动不代表没学到，以 PPL 为准。
   - **`KeyError: Submodule '...weight_quantizer.algorithms.<algo>' is not found`**：先查 eval/deploy 是否**漏带与 ptq 一致的 `--algos`**（quant-run 内建），**不要误判为框架闭环/结构 bug**。
4. 输出“PTQ 算法收益判读卡”，写回 progress。

## 边界规则

- 只判读**已跑出的算法结果**，不跑 ptq / 评测、不替用户选算法。
- 无直转参考也可判，但须注明缺对照基线、结论强度下降。
- 发现问题输出诊断（问题｜位置｜诊断），交 implementer 修。
- 引用仓库内路径一律相对路径。

## PTQ 算法收益判读卡

```md
# PTQ 算法收益判读卡

## 当前算法
- 算法名 / `targets` / 目标模块 / `quant_dtype` / `a_bits / w_bits` / `granularity`：

## 评测口径
- 数据集 / 指标 / `seq_len` / 模型版本·代码口径：

## 直转参考（取自 quant-run）
- `ppl_bf16` / `ppl_direct` / `delta_direct`：

## 算法结果（取自 quant-run）
- `ppl_algo` / `delta_algo` / `delta_gain`：

## 收益判读
- 是否优于直转 / 改善是否明显 / 是否稳定 / 是否值得继续 / 是否异常：

## 异常检查（交 implementer）
- 优先检查 1 / 2 / 3：

## 下一步建议
- 保持当前算法 / 更换算法 / 缩小验证范围：

## 结果复用情况
- 是否复用旧结果 / 未复用原因：
```
