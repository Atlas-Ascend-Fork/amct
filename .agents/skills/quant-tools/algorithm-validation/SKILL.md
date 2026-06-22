---
name: algorithm-validation
description: 验证指定量化算法在当前模型上的实际效果。适用于用户已经明确给出一个算法名，想比较该算法相对直转量化是否改善了精度，或想验证某个 weight / activation / structure 算法在当前目标模块上的收益、风险和下一步决策。
---

# 量化算法验证

## 用户调用格式

- 如果用户明确点名调用这个 skill，先返回这个 skill 对应的固定输入模板，再开始执行。
- 不替用户改成别的 skill；只有当用户没有指定 skill、只是描述目标时，才建议是否回到 `$quant-workflow`。
- 固定输入模板见：[../../reference/skill-input-template.md](../../reference/skill-input-template.md)

这个 skill 只用于**验证指定量化算法的实际效果**。

它回答的问题是：

- 当前指定算法能不能跑通
- 相比直转量化，它有没有改善精度
- 改善了多少，值不值得继续
- 如果没有改善，下一步该停、换算法还是缩范围

它不负责：

- 推荐应该先试哪个算法
- 推荐第一轮直转量化方案
- 做完整的误差定位路线设计

按下面顺序阅读：

1. [../../quant-workflow/references/metrics-and-thresholds.md](../../quant-workflow/references/metrics-and-thresholds.md)
2. [../../quant-workflow/references/direct-quant.md](../../quant-workflow/references/direct-quant.md)
3. 如需确认当前可用算法名，再读 `amct_pytorch/algorithms/quant/__init__.py` 和 `amct_pytorch/algorithms/quant/*.py`
4. 如需确认当前能力边界，再读 [../../../docs/repo-map.md](../../../docs/repo-map.md)

## 固定流程

1. 先确认用户已经明确给出：
   - 算法名
   - 目标模块
   - `quant_dtype`
   - `a_bits / w_bits`
   - `algos`
   - `granularity`
2. 如果用户还没选算法，停止验证，转到 `$algorithm-recommendation`。
3. 先查现有结果：
   - 是否已有同模型、同模块、同口径、同算法结果
   - 是否已有可对比的直转量化结果
4. 先检查前置条件：
   - 模型已经接进框架
   - BF16 baseline 已拿到
   - 直转量化参考结果最好已存在
   - 已验证关闭量化后保持浮点等价
5. 运行当前算法验证，拿到 `ppl_algo`。**评测 PTQ 结果的命令必须带 `--algos <与 ptq 训练一致>` + `--<target>_param_dir <ptq 参数目录>`**——否则评测侧 quant 模块不构建 `weight_quantizer.algorithms.<algo>` 子模块，`load_module` 报 `KeyError: Submodule '...algorithms.<algo>' is not found`：
   ```bash
   python -m amct_pytorch.eval --model <path> --model_name <name> --device npu:N \
     --eval_mode quant --quant_target <mlp|moe|attn-linear> --quant_dtype int --bit_config <yaml> \
     --algos <同 ptq> --<target>_param_dir <ptq_params 目录> --granularity block --seq_len 4096
   ```
6. 如果存在直转参考结果，计算：
   - `delta_direct = ppl_direct - ppl_bf16`
   - `delta_algo = ppl_algo - ppl_bf16`
   - `delta_gain = delta_direct - delta_algo`
7. 根据结果做判读：
   - 如果 `delta_algo` 明显优于 `delta_direct`，当前算法值得继续
   - 如果改善很小或不稳定，建议停用或换算法
   - 如果结果异常，先检查链路，不直接下结论
8. 输出“量化算法验证卡”。

## 边界规则

- 这里只验证**用户已指定的算法**，不负责替用户选算法。
- 如果用户只想知道“该先试哪个算法”，回到 `$algorithm-recommendation`。
- 如果用户只想评测指定直转方案，不涉及特定算法收益，回到 `$direct-quant-eval`。
- 如果模型还没适配完成，回到 `$model-adapter`。
- 如果没有直转参考结果，也可以验证算法，但必须明确缺少对照基线，结论强度下降。
- 当前主判断仍以 Wikitext PPL 为主，优先比较算法相对直转量化是否带来收益。
- 如果算法结果比 BF16 明显更好很多，先检查链路，不直接当成功。
- 引用仓库内代码、脚本和文档路径时，一律使用仓库相对路径；不要写绝对路径。

## 输出要求

结束时必须给出一张“量化算法验证卡”，至少包含下面字段：

- `当前算法`
- `评测口径`
- `直转参考结果`
- `算法结果`
- `收益比较`
- `结果判读`
- `异常检查`
- `下一步建议`
- `结果复用情况`

## 量化算法验证卡

```md
# 量化算法验证卡

## 当前算法
- 算法名：
- `targets`：
- 目标模块：
- `quant_dtype`：
- `a_bits / w_bits`：
- `granularity`：

## 评测口径
- 数据集：
- 指标：
- `seq_len`：
- 模型版本 / 代码口径：

## 直转参考结果
- `ppl_bf16`：
- `ppl_direct`：
- `delta_direct`：

## 算法结果
- `ppl_algo`：
- `delta_algo`：
- `delta_gain`：

## 收益比较
- 是否优于直转：
- 改善是否明显：
- 结果是否稳定：

## 结果判读
- 当前算法是否值得继续：
- 判断依据：
- 是否异常：

## 异常检查
- 优先检查 1：
- 优先检查 2：
- 优先检查 3：

## 下一步建议
- 保持当前算法：
- 更换算法：
- 缩小验证范围：

## 结果复用情况
- 是否复用旧结果：
- 如果没有复用，原因：
```
