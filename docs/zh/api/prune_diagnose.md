# prune_diagnose

## 产品支持情况

| 产品        | 是否支持 |
| ----------- | -------- |
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √        |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √        |

## 功能说明

剪枝可行性诊断：识别各域（cnn/dense/moe）可剪目标、做固定率剪枝 dry-run、做 acc 二分搜索 dry-run，返回诊断报告。诊断全程在模型副本上进行（dry-run），不修改传入的原模型。

## 函数原型

```python
prune_diagnose(model, *, data=None, config=None, prune_ratio=0.5, tolerance=0.05, batch_adapter=None)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| ------ | --------- | ------------------------------------------------------------ |
| model  | 输入      | 含义：待诊断的模型（在副本上 dry-run，原模型不变）。<br>数据类型：torch.nn.Module。 |
| data   | 输入      | 含义：校准/评估数据；方差类方法与 acc 二分搜索需要，为 None 时跳过对应检查。<br>默认值：None。<br>数据类型：Tensor、list、DataLoader 等可迭代对象，或 None。 |
| config | 输入      | 含义：剪枝配置；None 时使用 cnn/dense/moe 三域默认方法。<br>默认值：None。<br>数据类型：dict、PruneConfig 或 None。 |
| prune_ratio | 输入  | 含义：固定率剪枝 dry-run 所用的剪枝率。<br>默认值：0.5。<br>数据类型：float。 |
| tolerance | 输入    | 含义：acc 二分搜索 dry-run 所用的可接受精度损失上界。<br>默认值：0.05。<br>数据类型：float。 |
| batch_adapter | 输入 | 含义：可选回调，把一个 batch 映射为前向所需的 (args, kwargs)。<br>默认值：None。<br>数据类型：Callable，或 None。 |

## 返回值说明
返回 DiagnosisReport（dataclass），主要字段：
- targets：各域可剪目标数量（dict，键为 cnn/dense/moe）。
- prune_works：固定率剪枝是否实际生效（参数量是否下降，bool）。
- prune_reduction：实测权重削减比例（float）。
- prune_forward_ok：剪枝后前向是否正常（bool；无样本时为 None）。
- search_works：acc 二分搜索是否跑通并选出剪枝率（bool）。
- search_chosen_ratio：搜索选出的剪枝率（float；未选出为 None）。
- notes：诊断过程中的提示/告警信息（list）。

可调用 `report.summary()` 打印人类可读摘要，或用 `report.any_domain_detected` 判断是否检测到任一可剪域。

## 调用示例
```python
import amct_pytorch as amct
from amct_pytorch.pruning import prune_diagnose

# 在用户已实例化的模型 + 校准数据上做剪枝可行性诊断（dry-run，不改原模型）
report = prune_diagnose(model, data=calib, prune_ratio=0.5, tolerance=0.05)
print(report.summary())
```
