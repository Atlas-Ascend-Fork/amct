# prune_finetune

## 产品支持情况

| 产品        | 是否支持 |
| ----------- | -------- |
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √        |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √        |

## 功能说明

剪枝后轻量微调恢复（供 accuracy 搜索的 `finetune_fn` 使用）。对（通常是已剪枝的）模型做就地（in-place）的少量梯度步微调，以恢复因结构化剪枝损失的精度。可直接调用，也可包装成 `finetune_fn=lambda m: prune_finetune(m, train_data, ...)` 传给 `amct.prune` 在搜索时对每个候选率先恢复再评估。

## 函数原型

```python
prune_finetune(model, data, *, loss_fn=None, steps=200, lr=1e-4, weight_decay=0.0, warmup=0, optimizer=None, batch_adapter=None, device=None, grad_clip=1.0, log_every=0)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| ------ | --------- | ------------------------------------------------------------ |
| model | 输入/输出 | 含义：待微调的模型，权重就地（in-place）更新。<br>数据类型：torch.nn.Module。 |
| data | 输入 | 含义：训练批序列，循环复用直至达到 `steps` 指定的步数。<br>数据类型：Sequence[Any]。 |
| loss_fn | 输入 | 含义：自定义损失函数 `loss_fn(model, batch) -> Tensor`；为 None 时使用内置默认损失（dict 含 `input_ids` 走因果 LM 自监督损失，`(x, y)` 二元组走交叉熵分类损失）。<br>默认值：None。<br>数据类型：Callable[[nn.Module, Any], torch.Tensor] 或 None。 |
| steps | 输入 | 含义：梯度更新步数；≤0 时直接返回空结果不做训练。<br>默认值：200。<br>数据类型：int。 |
| lr | 输入 | 含义：默认 AdamW 优化器的学习率（恢复预训练 LM 时建议 1e-5 量级并配合 warmup，1e-4 易发散）。<br>默认值：1e-4。<br>数据类型：float。 |
| weight_decay | 输入 | 含义：默认 AdamW 优化器的权重衰减。<br>默认值：0.0。<br>数据类型：float。 |
| warmup | 输入 | 含义：线性 warmup 步数，前 `warmup` 步内学习率从 0 线性升至 `lr`。<br>默认值：0。<br>数据类型：int。 |
| optimizer | 输入 | 含义：自定义 torch 优化器；为 None 时基于可训练参数自动构建 AdamW。<br>默认值：None。<br>数据类型：torch.optim.Optimizer 或 None。 |
| batch_adapter | 输入 | 含义：批适配器，将一个 batch 映射为 `(args, kwargs)` 供模型前向；默认损失会用其输出的 `.loss`。<br>默认值：None。<br>数据类型：BatchAdapter 或 None。 |
| device | 输入 | 含义：目标设备；为 None 时从模型自动推断。<br>默认值：None。<br>数据类型：torch.device / str 或 None。 |
| grad_clip | 输入 | 含义：梯度裁剪的最大范数；为假值（None/0）时关闭裁剪。<br>默认值：1.0。<br>数据类型：float 或 None。 |
| log_every | 输入 | 含义：>0 时对返回的 `loss_history` 做下采样（每步 loss 始终记录）。<br>默认值：0。<br>数据类型：int。 |

## 返回值说明
返回一个 dict，包含以下键：

- `steps`：实际执行的梯度步数（int）。
- `initial_loss`：首步损失，无步数时为 None。
- `final_loss`：末步损失，无步数时为 None。
- `loss_history`：逐步损失列表（`log_every > 0` 时为下采样后的列表）。

模型权重为就地更新，无需接收返回的模型对象。

## 调用示例
```python
import amct_pytorch as amct

# model 为用户已剪枝的 torch.nn.Module；calib 为校准/训练批序列
result = amct.prune_finetune(model, calib, steps=300, lr=1e-5, warmup=20)
print(result["initial_loss"], "->", result["final_loss"])
```
