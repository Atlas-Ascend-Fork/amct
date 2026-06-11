# 预定义剪枝配置（Predefined Pruning Configs）<a name="ZH-CN_TOPIC_0000002600000601"></a>

## 产品支持情况<a name="zh-cn_topic_0000002600000601_section106000010001"></a>

<a name="zh-cn_topic_0000002600000601_table106000010002"></a>

| 产品                                        | 是否支持 |
| ------------------------------------------- | -------- |
| Ascend 950PR/Ascend 950DT                   | √        |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √        |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √        |



## 功能说明<a name="zh-cn_topic_0000002600000601_section106000010003"></a>

这些是开箱即用的 `PruneConfig` 预设配置，作为 `amct.prune(model, CFG, data=calib)` 的第二个参数传入。每个预设针对不同的网络结构（CNN、Dense、MoE）封装了对应的剪枝方法与超参，无需手动构造配置即可直接使用。

## 配置列表<a name="zh-cn_topic_0000002600000601_section106000010004"></a>

### CNN_RECONSTRUCT_PRUNE_CFG<a name="zh-cn_topic_0000002600000601_section106000010005"></a>

针对 CNN 卷积通道做“输出保持（重建）”剪枝：损伤较低，剪枝后对下游消费权重做最小二乘重建以恢复精度。

```python
{
    "methods": {
        "cnn": {
            "name": "reconstruct",
            "kwargs": {"prune_ratio": 0.30},
        },
    },
    "missing_data_policy": "warn_skip",
}
```

### CNN_VARIANCE_PRUNE_CFG<a name="zh-cn_topic_0000002600000601_section106000010006"></a>

针对 CNN 卷积通道做方差通道剪枝，按激活方差移除信息量最低的通道。

```python
{
    "methods": {
        "cnn": {"name": "variance_channel", "kwargs": {"prune_ratio": 0.30}},
    },
    "missing_data_policy": "warn_skip",
}
```

### DENSE_LOWVAR_PRUNE_CFG<a name="zh-cn_topic_0000002600000601_section106000010007"></a>

针对 Dense（全连接）层做低方差剪枝，移除方差最低的神经元。

```python
{
    "methods": {
        "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.50}},
    },
    "missing_data_policy": "warn_skip",
}
```

### FULL_STRUCTURED_PRUNE_CFG<a name="zh-cn_topic_0000002600000601_section106000010008"></a>

一次性对 CNN、Dense、MoE 三种结构同时做结构化剪枝，是覆盖全网的组合预设。

```python
{
    "methods": {
        "cnn": {"name": "variance_channel", "kwargs": {"prune_ratio": 0.30}},
        "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.50}},
        "moe": {
            "name": "mass_variance",
            "kwargs": {"prune_ratio": 0.50, "boundary": 10},
        },
    },
    "missing_data_policy": "warn_skip",
}
```

### MOE_MASSVAR_PRUNE_CFG<a name="zh-cn_topic_0000002600000601_section106000010009"></a>

针对 MoE 结构做质量-方差（mass_variance）剪枝，结合权重质量与方差并以 `boundary` 控制边界。

```python
{
    "methods": {
        "moe": {
            "name": "mass_variance",
            "kwargs": {"prune_ratio": 0.50, "boundary": 10},
        },
    },
    "missing_data_policy": "warn_skip",
}
```

### MOE_OUTPUT_MERGE_PRUNE_CFG<a name="zh-cn_topic_0000002600000601_section106000010010"></a>

针对 MoE 做输出空间专家合并：以最小二乘残差贪心配对 + drop 回退，并用 calib-NLL 选择器保证不劣于 activation_count 基线（选择结果写入报告，不会静默进行）。

```python
{
    "methods": {
        "moe": {
            "name": "output_merge",
            "kwargs": {"keep_ratio": 0.50, "selector": "calib_nll"},
        },
    },
    "missing_data_policy": "warn_skip",
}
```

### SENSITIVITY_ALLOC_PRUNE_CFG<a name="zh-cn_topic_0000002600000601_section106000010011"></a>

按逐层敏感度分配剪枝率：在相同总削减量下，对稳健层多剪、对敏感层少剪；calib_nll 守卫在结果变差时自动回退到均匀分配并告警。需要校准数据（`data=`）。

```python
{
    "methods": {
        "dense": {"name": "low_variance", "kwargs": {"prune_ratio": 0.50}},
    },
    "allocation": {
        "strategy": "sensitivity",
        "ref_ratio": 0.50,
        "min_ratio": 0.05,
        "max_ratio": 0.90,
        "guard": "calib_nll",
    },
    "missing_data_policy": "warn_skip",
}
```

## 调用示例<a name="zh-cn_topic_0000002600000601_section106000010012"></a>

```python
import amct_pytorch as amct

# 直接传入预定义剪枝配置作为第二个参数
amct.prune(model, amct.MOE_OUTPUT_MERGE_PRUNE_CFG, data=calib)
```
