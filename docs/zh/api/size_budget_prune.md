# size_budget_prune

## 产品支持情况

| 产品        | 是否支持 |
| ----------- | -------- |
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √        |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √        |

## 功能说明

按目标尺寸/保留比例（size budget）反推并应用各域剪枝率。在候选剪枝率网格上二分搜索出参数量满足目标上界的最小剪枝率，并报告该点的实际尺寸与精度。

## 函数原型

```python
size_budget_prune(model, config=None, *, data=None, target_keep_ratio=None,
                  target_params=None, evaluator=None,
                  eval_iterations=1, eval_data=None, ratio_grid=DEFAULT_RATIO_GRID,
                  batch_adapter=None, safe_skip_attention=True, finetune_fn=None,
                  quant_fn=None, apply=True, budget_scope="total")
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| ------ | --------- | ------------------------------------------------------------ |
| model | 输入/输出 | 含义：待剪枝模型；当 apply=True 时原地修改。<br>数据类型：torch.nn.Module。 |
| config | 输入 | 含义：剪枝配置，可为 dict / PruneConfig / None；None 时使用三域默认配置。<br>默认值：None。<br>数据类型：dict、PruneConfig 或 None。 |
| data | 输入 | 含义：校准数据；当未提供 eval_data 时同时用作默认精度评估集。<br>默认值：None。<br>数据类型：Sequence 或 None。 |
| target_keep_ratio | 输入 | 含义：参数保留比例（如 0.7 表示保留 70%、剪掉 30%）；与 target_params 二者择一使用。<br>默认值：None。<br>数据类型：float 或 None。 |
| target_params | 输入 | 含义：目标参数量的直接上界；与 target_keep_ratio 二者择一使用。<br>默认值：None。<br>数据类型：int 或 None。 |
| evaluator | 输入 | 含义：精度来源：Callable[[model], float] 或 AutoCalibrationEvaluatorBase（与量化共用）；为 None 时用默认 top-1 保真度。<br>默认值：None。<br>数据类型：Callable / AutoCalibrationEvaluatorBase。 |
| eval_iterations | 输入 | 含义：传给 evaluator.evaluate 的迭代次数。<br>默认值：1。<br>数据类型：int。 |
| eval_data | 输入 | 含义：默认保真度指标使用的评估批次；缺省时回退使用 data。<br>默认值：None。<br>数据类型：Sequence 或 None。 |
| ratio_grid | 输入 | 含义：候选剪枝率网格（升序排列）。<br>默认值：DEFAULT_RATIO_GRID。<br>数据类型：Sequence[float]。 |
| batch_adapter | 输入 | 含义：将数据批次映射为模型前向所需的 (args, kwargs)。<br>默认值：None。<br>数据类型：BatchAdapter 或 None。 |
| safe_skip_attention | 输入 | 含义：True 时将注意力子模块并入 skip_layers，以保护并行的 q/k/v 投影。<br>默认值：True。<br>数据类型：bool。 |
| finetune_fn | 输入 | 含义：可选的剪枝后恢复回调，在评估前对每个剪枝副本执行。<br>默认值：None。<br>数据类型：Callable[[torch.nn.Module], Any] 或 None。 |
| quant_fn | 输入 | 含义：可选的量化感知回调，在评估前应用，使搜索在量化状态下决策。<br>默认值：None。<br>数据类型：Callable[[torch.nn.Module], Any] 或 None。 |
| apply | 输入 | 含义：True 时将选定的剪枝率原地应用于模型；False 时仅搜索不应用。<br>默认值：True。<br>数据类型：bool。 |
| budget_scope | 输入 | 含义：预算口径。'total'（默认，预算覆盖全部参数）或 'prunable'（预算仅覆盖可剪枝质量，排除 embedding 等）；当预算低于不可剪枝质量下界时告警并记录 budget_unreachable=True，不抛异常。<br>默认值："total"。<br>数据类型：str。 |

## 返回值说明
AutoTuneResult。其中 weight_reduction 为实际剪掉的比例，quality_drop 为该尺寸下实测的精度损失，tolerance 固定为 -1（非容差模式），chosen_ratio 在无任何剪枝率满足预算时为 None；budget_unreachable 标记预算是否低于不可剪枝质量下界，prunable_fraction 为可剪枝质量占总参数的比例。

## 调用示例
```python
import amct_pytorch as amct
from amct_pytorch.pruning import size_budget_prune

# 准备待剪枝模型与校准数据
# model = ...
# calib = ...

# 按保留 70% 参数的目标预算原地剪枝
result = size_budget_prune(model, data=calib, target_keep_ratio=0.7)
print(result.summary())
```
