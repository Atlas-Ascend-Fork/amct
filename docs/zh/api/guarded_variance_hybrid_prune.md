# guarded_variance_hybrid_prune

## 产品支持情况

| 产品        | 是否支持 |
| ----------- | -------- |
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √        |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √        |

## 功能说明

MoE 专家剪枝的守卫式 variance-HYBRID：一次校准 → 按判据菜单（MENU）各剪一份副本 → 在验证集上实测择优 → 应用胜出者。仅当某方差判据在验证集上严格优于 fallback（默认 `mass`）时才切换，否则回退 fallback，因此 HYBRID 在验证集上绝不劣于 fallback。

## 函数原型

```python
guarded_variance_hybrid_prune(model, *, data=None, prune_ratio=0.50, top_k=8,
                              eval_data=None, quality_fn=None, evaluator=None,
                              eval_iterations=1, batch_adapter=None, menu=None,
                              apply=True)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| ------ | --------- | ------------------------------------------------------------ |
| model | 输入/输出 | 含义：含 MoE 层的待剪枝模型，`apply=True` 时就地修改。<br>数据类型：torch.nn.Module。 |
| data | 输入 | 含义：校准数据（每个判据各做一次校准）；未提供 `eval_data` 时同时用作默认验证集。<br>默认值：None。<br>数据类型：Any。 |
| prune_ratio | 输入 | 含义：每层移除的专家比例（所有判据使用同一比例），取值范围 [0.0, 1.0)。<br>默认值：0.50。<br>数据类型：float。 |
| top_k | 输入 | 含义：每个 token 选择的专家数。<br>默认值：8。<br>数据类型：int。 |
| eval_data | 输入 | 含义：守卫用的验证集（建议与 `data` 不同）；缺省时回退到 `data`。<br>默认值：None。<br>数据类型：Sequence 或 None。 |
| quality_fn | 输入 | 含义：验证质量函数（值越大越好），None 时使用默认 top-1 保真度；优先级 `quality_fn` > `evaluator` > 默认保真度。<br>默认值：None。<br>数据类型：Callable[[torch.nn.Module], float] 或 None。 |
| evaluator | 输入 | 含义：与量化侧共享的评估抽象，`evaluate(model[, iters]) -> float`。<br>默认值：None。<br>数据类型：object 或 None。 |
| eval_iterations | 输入 | 含义：传给 `evaluator` 的评估迭代次数。<br>默认值：1。<br>数据类型：int。 |
| batch_adapter | 输入 | 含义：将原始 batch 适配为模型输入。<br>默认值：None。<br>数据类型：BatchAdapter 或 None。 |
| menu | 输入 | 含义：判据菜单（name, MethodSpec），None 时使用 DEFAULT_MENU；第 0 项为 fallback 且必须存在。<br>默认值：None。<br>数据类型：Sequence[Tuple[str, Dict]] 或 None。 |
| apply | 输入 | 含义：True 时将胜出判据就地应用到 `model`；False 时仅搜索不应用。<br>默认值：True。<br>数据类型：bool。 |

## 返回值说明
返回 `HybridResult` 数据类，关键字段包括：`chosen_criterion`（选中的判据）、`fallback_criterion`（fallback 判据）、`criterion_quality`（各判据的验证质量，失败者为 -inf）、`gain_over_fallback`（相对 fallback 的验证增益，守卫保证 >= 0）、`prune_ratio`、`params_before` / `params_after`（剪枝前后参数量）、`weight_reduction`（参数削减比例）、`applied`（是否已就地应用）、`report`（剪枝报告）。可调用 `res.summary()` 获取一行摘要。

## 调用示例
```python
import amct_pytorch as amct

# model 为用户提供的含 MoE 层的 torch.nn.Module，calib / val 为校准与验证数据
res = amct.guarded_variance_hybrid_prune(
    model, data=calib, prune_ratio=0.5, top_k=8, eval_data=val)  # 就地剪枝
print(res.summary())  # 选中的判据 + 相对 fallback 的验证增益 + 削减比例
```
