---
name: direct-quant-eval
description: 判读直转量化结果（reviewer 用，只读不跑）：读 quant-run 跑出的 ppl_bf16 / ppl_quant，算 delta、判是否达标（默认 delta ≤ 0.2）、给下一步建议（保持 / 缩范围 / 转 PTQ）。触发场景：直转 ppl 已出、要判 delta 是否达标并定下一步。不适用：跑评测命令（→ $quant-run）。
---

# 直转量化判读

reviewer 的判读 skill：对 `quant-run` 已跑出的直转结果做判定。**只读结果、不跑命令**——评测执行归 `$quant-run`（implementer）。

它回答的问题是：

- `ppl_bf16 / ppl_quant / delta` 是多少、是否达标
- 结果异常时优先该怀疑什么
- 下一步是保持方案、缩范围，还是转入 PTQ 升级判读

它不负责：

- 跑评测命令（→ `$quant-run`）
- 推荐 / 重新设计方案（→ `$scheme-recommendation`）
- PTQ 收益判读（→ `$algorithm-validation`）

按需读：[../references/metrics-and-thresholds.md](../references/metrics-and-thresholds.md)。

## 固定流程

1. 从 progress 读 `quant-run` 写入的当前方案 + `ppl_bf16 / ppl_quant`。缺数则回报：需先由 implementer 跑 `$quant-run`，不自行跑评测。
2. 确认口径一致（Wikitext PPL `seq_len=4096`、同模型版本/代码口径）；口径不同的旧结果不能直接横比。
3. 计算 `delta = ppl_quant - ppl_bf16`。
4. 判读：
   - `delta <= 0.2`：当前方案可接受；
   - `delta > 0.2`：不可接受，建议缩量化范围或转 PTQ 升级；
   - 量化后 PPL 明显优于 BF16 很多 / 数值异常：先让 implementer 查链路，不直接下结论。
   - **`delta ≈ 0` 或 quant PPL 与 bf16 逐位相同**：量化很可能**未真正生效**（如 `granularity=model` 没真量化、`quant_target` 未落到算子），**不可直接判达标**——让 implementer 复核 `--granularity block` + BitPolicy + quant_target 是否落到算子。
5. 输出“直转量化判读卡”，把判定结论写回 progress。

## 边界规则

- 只判读**已跑出的直转结果**，不跑评测、不发明方案、不做误差定位 / PTQ。
- 主判断指标 = Wikitext PPL 与 `delta <= 0.2`。
- 发现问题输出诊断（问题｜位置｜诊断），交 implementer 修，不自己改代码 / 改方案。
- 引用仓库内路径一律相对路径。

## 直转量化判读卡

```md
# 直转量化判读卡

## 当前方案
- 量化模块 / `quant_dtype` / `a_bits / w_bits` / `algos` / `granularity`：

## 评测口径
- 数据集 / 指标 / `seq_len` / 模型版本·代码口径：

## 结果（取自 quant-run）
- `ppl_bf16` / `ppl_quant` / `delta`：

## 结果判读
- 是否达标 / 判断依据 / 是否异常：

## 异常检查（交 implementer）
- 优先检查 1 / 2 / 3：

## 下一步建议
- 保持当前方案 / 缩小量化范围 / 转入 PTQ 升级判读：

## 结果复用情况
- 是否复用旧结果 / 未复用原因：
```
