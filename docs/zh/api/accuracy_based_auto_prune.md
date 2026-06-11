# accuracy_based_auto_prune

## 产品支持情况

| 产品        | 是否支持 |
| ----------- | -------- |
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √        |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √        |

## 功能说明

按可接受精度损失（tolerance）自动二分搜索并应用满足容差的最大剪枝率，返回 AutoTuneResult。精度被抽象为 `quality_fn(model) -> float`（越大越好）；未提供 `quality_fn` 时使用默认的免标签 top-1 保真度指标（以原模型在校准数据上的 top-1 预测为基准）。在剪枝率网格上假设精度损失随剪枝率单调非减，二分查找精度损失 ≤ tolerance 的最大剪枝率；若无满足者则不剪枝。

## 函数原型

```python
accuracy_based_auto_prune(
    model,
    config=None,
    *,
    data=None,
    tolerance=0.02,
    quality_fn=None,
    evaluator=None,
    eval_iterations=1,
    eval_data=None,
    ratio_grid=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
    batch_adapter=None,
    safe_skip_attention=True,
    finetune_fn=None,
    quant_fn=None,
    apply=True,
)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| ------ | --------- | ------------------------------------------------------------ |
| model | 输入/输出 | 含义：待剪枝模型；当 `apply=True` 时被原地（in-place）修改。<br>数据类型：torch.nn.Module。 |
| config | 输入 | 含义：剪枝配置；每个 method 的 `prune_ratio` 会被搜索到的值覆盖。为 `None` 时使用 CNN/Dense/MoE 三域默认配置。<br>默认值：None。<br>数据类型：dict / PruneConfig / None。 |
| data | 输入 | 含义：校准数据（方差类方法需要）；当未提供 `eval_data` 时也作为默认保真度指标的评估集。<br>默认值：None。<br>数据类型：可迭代对象（batch 序列）。 |
| tolerance | 输入 | 含义：可接受精度损失的上界，量纲与 `quality_fn` 一致（默认保真度指标下为 top-1 翻转比例）。<br>默认值：0.02。<br>数据类型：float。 |
| quality_fn | 输入 | 含义：精度评估函数（越大越好）。为 `None` 时使用默认的免标签 top-1 保真度指标。<br>默认值：None。<br>数据类型：Callable[[torch.nn.Module], float]。 |
| evaluator | 输入 | 含义：暴露 `evaluate(model, iterations) -> float` 的对象；提供时作为精度来源。优先级：quality_fn > evaluator > 默认保真度。<br>默认值：None。<br>数据类型：object。 |
| eval_iterations | 输入 | 含义：传给 `evaluator.evaluate` 的迭代次数。<br>默认值：1。<br>数据类型：int。 |
| eval_data | 输入 | 含义：默认保真度指标使用的评估批；未提供时回退到 `data`。<br>默认值：None。<br>数据类型：可迭代对象（batch 序列）。 |
| ratio_grid | 输入 | 含义：候选剪枝率（升序）。<br>默认值：(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)。<br>数据类型：float 序列。 |
| batch_adapter | 输入 | 含义：将一个 data batch 映射为模型前向所需的 `(args, kwargs)`；为 `None` 时使用合理的默认解析。<br>默认值：None。<br>数据类型：Callable / None。 |
| safe_skip_attention | 输入 | 含义：是否将注意力子模块（self_attn/attention/attn）并入 `skip_layers`，避免 Dense 域把并行的 q/k/v 投影误判为两层 FFN。<br>默认值：True。<br>数据类型：bool。 |
| finetune_fn | 输入 | 含义：可选的剪枝后恢复回调；提供时每个候选会在评估前对剪枝副本微调一次，使搜索衡量"剪枝+恢复"后的可达精度（每个候选微调一次，更慢）。<br>默认值：None。<br>数据类型：Callable[[torch.nn.Module], Any]。 |
| quant_fn | 输入 | 含义：可选的回调，在评估前（在 `finetune_fn` 之后）作用于剪枝副本，使搜索在该状态下决策；返回的模型仍为仅剪枝模型。<br>默认值：None。<br>数据类型：Callable[[torch.nn.Module], Any]。 |
| apply | 输入 | 含义：为 `True` 时将选定剪枝率原地应用到 model；为 `False` 时仅搜索、不改动原模型。<br>默认值：True。<br>数据类型：bool。 |

## 返回值说明

返回 `AutoTuneResult`（dataclass），其关键字段：

- `chosen_ratio`：选定的剪枝率；容差内无可行率时为 `None`。
- `tolerance`：本次搜索使用的容差。
- `baseline_quality` / `pruned_quality` / `quality_drop`：基准精度、选定率下的精度、以及精度损失。
- `params_before` / `params_after` / `weight_reduction`：剪枝前后的参数量及实际削减比例。
- `applied`：是否已将剪枝原地应用到模型（受 `apply` 控制）。
- `report`：剪枝报告 `PruneReport`（未应用时为 `None`）。
- `trials`：各候选剪枝率的试探结果列表。
- `summary()`：返回一行可读摘要。

## 调用示例

```python
import amct_pytorch as amct

# model 为用户提供的 torch.nn.Module，calib 为校准/评估数据（batch 序列）
result = amct.accuracy_based_auto_prune(
    model,
    data=calib,
    tolerance=0.02,   # 最多可接受 2% 精度损失
)
# 原地剪枝完成，查看结果摘要
print(result.summary())
```
