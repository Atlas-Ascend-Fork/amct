# guarded_recovery_hybrid_prune

## 产品支持情况

| 产品        | 是否支持 |
| ----------- | -------- |
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √        |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √        |

## 功能说明

dense FFN / CNN 通道剪枝的守卫式 recovery-HYBRID：固定选神经元的判据，对 recovery 菜单（none/bias/ls）各剪一份并在验证集上择优，再把最佳补偿原地应用到模型。守卫仅当某 recovery 在验证集上严格优于 fallback(none) 时才切换，因此保证在验证集上绝不劣于无恢复（不补偿）基线。

## 函数原型

```python
guarded_recovery_hybrid_prune(model, *, data=None, prune_ratio=0.50, domain="dense", recovery_menu=("none", "bias", "ls"), ridge=1e-2, eval_data=None, quality_fn=None, evaluator=None, eval_iterations=1, batch_adapter=None, apply=True)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| ------ | --------- | ------------------------------------------------------------ |
| model | 输入/输出 | 含义：含 dense FFN（或 CNN sequential conv）的待剪枝模型；当 apply=True 时原地修改。<br>数据类型：torch.nn.Module。 |
| data | 输入 | 含义：校准数据（每种 recovery 各跑一次）；当未提供 eval_data 时也作为默认验证集。<br>默认值：None。<br>数据类型：Any（可迭代的批数据）。 |
| prune_ratio | 输入 | 含义：中间神经元 / 通道的剪枝比例（所有 recovery 共用同一比例与同一选神经元判据，仅补偿方式不同）。取值范围 [0.0, 1.0)。<br>默认值：0.50。<br>数据类型：float。 |
| domain | 输入 | 含义：剪枝域，"dense"（FFN）或 "cnn"（sequential conv 通道）。<br>默认值："dense"。<br>数据类型：str。 |
| recovery_menu | 输入 | 含义：候选补偿方式菜单；第 0 项为 fallback。none = 朴素切片（fallback），bias = 均值折叠到 consumer bias（无最小二乘），ls = 最小二乘重构 consumer 权重。<br>默认值：("none", "bias", "ls")。<br>数据类型：Sequence[str]。 |
| ridge | 输入 | 含义：最小二乘恢复（ls）的岭回归正则化系数。<br>默认值：1e-2。<br>数据类型：float。 |
| eval_data | 输入 | 含义：可选的显式验证批数据；为 None 时回退到 data。务必使用代表部署分布的验证数据。<br>默认值：None。<br>数据类型：Optional[Sequence[Any]]。 |
| quality_fn | 输入 | 含义：可选的验证质量来源（返回值越大越好）。<br>默认值：None。<br>数据类型：Optional[QualityFn]。 |
| evaluator | 输入 | 含义：当未提供 quality_fn 时使用的可选评估器。<br>默认值：None。<br>数据类型：Optional[Any]。 |
| eval_iterations | 输入 | 含义：评估迭代次数。<br>默认值：1。<br>数据类型：int。 |
| batch_adapter | 输入 | 含义：可选的适配器，把一个批数据映射为模型输入。<br>默认值：None。<br>数据类型：Optional[BatchAdapter]。 |
| apply | 输入 | 含义：为 True 时将择优出的 recovery 原地应用到 model。<br>默认值：True。<br>数据类型：bool。 |

## 返回值说明
返回 RecoveryHybridResult（dataclass），关键字段包括：chosen_recovery（选中的 recovery）、fallback_recovery（fallback 名）、recovery_quality（各 recovery 的验证质量字典）、chosen_quality / fallback_quality、gain_over_fallback（相对 fallback 的验证增益）、prune_ratio、domain、params_before / params_after、weight_reduction（权重削减比例）、applied（是否已原地应用）、report（剪枝报告）、trials（按质量排序的试验列表）。可调用 res.summary() 打印摘要。

## 调用示例
```python
import amct_pytorch as amct

# model 为用户提供的含 dense FFN 的模型，calib 为校准数据，val 为代表部署分布的验证数据
res = amct.guarded_recovery_hybrid_prune(
    model, data=calib, prune_ratio=0.5, domain="dense", eval_data=val)  # 原地剪枝
print(res.summary())  # 选中的 recovery + 相对 none 的验证增益 + 削减比例
```
