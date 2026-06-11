# prune

## 产品支持情况

| 产品        | 是否支持 |
| ----------- | -------- |
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √        |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √        |

## 功能说明

结构化剪枝主入口：对已实例化的 torch.nn.Module 原地剪枝（dense FFN 中间维 / CNN 通道 / MoE 专家）；传入 tolerance 时等价为按精度损失上界自动搜索最大剪枝率。传入 size_budget（保留比例）时按尺寸预算剪枝（收容 size_budget_prune）。与 amct.quantize 对齐，原地修改，无进程级全局状态：统计通过可选的 `report=PruneReport()` sink 获取，搜索结果由 accuracy_based_auto_prune / size_budget_prune 的返回值获取。与 amct.quantize 不同的是，结构化剪枝会原地改写模型的 `config` 维度（intermediate_size / num_experts），使剪枝后的模型能正确保存/重载。

## 函数原型

```python
prune(model, config=None, *, data=None, batch_adapter=None, tolerance=None,
      evaluator=None, eval_iterations=1, eval_data=None, ratio_grid=None,
      size_budget=None, finetune_fn=None, quant_fn=None, report=None)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| ------ | --------- | ------------------------------------------------------------ |
| model | 输入/输出 | 含义：待剪枝模型，调用方负责实例化（如 from_pretrained）；原地剪枝。<br>数据类型：torch.nn.Module。 |
| config | 输入 | 含义：剪枝配置，覆盖各域（cnn/dense/moe）默认剪枝率与方法；None 时使用默认 PruneConfig。<br>默认值：None。<br>数据类型：dict / PruneConfig / None。 |
| data | 输入 | 含义：校准数据（DataLoader / tensor / dict），方差类方法必需，也作默认评估集。<br>默认值：None。<br>数据类型：Any。 |
| batch_adapter | 输入 | 含义：将自定义 batch 解包为前向 args、kwargs 的回调。<br>默认值：None。<br>数据类型：Callable / None。 |
| tolerance | 输入 | 含义：可接受的精度损失上界；给定时开启精度驱动自动剪枝，在 ratio_grid 上搜索容差内的最大剪枝率。<br>默认值：None（固定剪枝率模式）。<br>数据类型：float / None。 |
| evaluator | 输入 | 含义：精度来源，与量化共用 —— Callable[[model], float] 或 AutoCalibrationEvaluatorBase（其 evaluate(model) 返回标量）；None 时使用默认 top-1 保真度评估。<br>默认值：None。<br>数据类型：Callable / AutoCalibrationEvaluatorBase / None。 |
| eval_iterations | 输入 | 含义：容差模式下的评估迭代次数。<br>默认值：1。<br>数据类型：int。 |
| eval_data | 输入 | 含义：容差模式下传入自动剪枝的评估数据；默认回退到 data。<br>默认值：None。<br>数据类型：Any。 |
| ratio_grid | 输入 | 含义：容差模式下的候选剪枝率（升序）；None 时使用默认网格。<br>默认值：None。<br>数据类型：Any / None。 |
| size_budget | 输入 | 含义：给定时（保留比例，(0,1]）按尺寸预算剪枝，等价 size_budget_prune；与 tolerance 互斥。<br>默认值：None。<br>数据类型：float / None。 |
| finetune_fn | 输入 | 含义：容差模式下对每个候选率剪枝后做（轻量）微调恢复再评估的回调，使搜索能选出更激进的可用率。<br>默认值：None。<br>数据类型：Callable[[nn.Module], Any] / None。 |
| quant_fn | 输入 | 含义：容差搜索过程中对模型做量化的回调（量化感知搜索）。<br>默认值：None。<br>数据类型：Callable[[nn.Module], Any] / None。 |
| report | 输入/输出 | 含义：调用方提供的统计 sink；传入后剪枝统计写入其中，事后通过 `report.as_dict()` 读取（无进程级全局状态）。<br>默认值：None。<br>数据类型：amct.PruneReport / None。 |

## 返回值说明

None。剪枝为原地操作。剪枝统计通过调用方提供的 `report=amct.PruneReport()` sink 获取（事后读 `report.as_dict()`，含 backend、params_before、params_after、per_layer_sparsity、warnings、budget_unreachable 等字段）；容差/尺寸预算模式的搜索结果直接由 `accuracy_based_auto_prune(...)` / `size_budget_prune(...)` 的返回值（AutoTuneResult）获取（无进程级全局状态）。

## 调用示例

```python
import amct_pytorch as amct

# 固定剪枝率：dense 域中间维剪 50%（自动避开注意力投影）
cfg = {"methods": {"dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.5}}}}
rep = amct.PruneReport()
amct.prune(model, cfg, data=calib, report=rep)
print(rep.as_dict())

# 容差驱动：最多损失 2%，自动搜索并应用满足容差的最大剪枝率
res = amct.accuracy_based_auto_prune(model, data=calib, tolerance=0.02)
print(res.summary())
```
