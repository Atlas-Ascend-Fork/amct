# convert

## 产品支持情况

| 产品        | 是否支持 |
| ----------- | -------- |
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √        |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √        |

## 功能说明

将量化校准模型转换为量化部署模型。可选从 quant-result artifact 恢复校准状态后再转换。

## 函数原型

```python
convert(model, quant_result_dir=None)
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                                         |
| ------ | --------- | ------------------------------------------------------------ |
| model  | 输入/输出 | 输入含义：量化校准模型。<br>输出含义：量化部署模型。<br>数据类型：torch.nn.Module。 |
| quant_result_dir | 输入 | 可选的 quant-result artifact 目录。传入时，`model` 应为与 artifact fingerprint 匹配的原始浮点模型；接口先恢复量化校准状态，再原地转换。默认值为 `None`。 |


## 返回值说明
无返回值。接口原地修改 `model`。

## 调用示例
```python
# 调用前序接口生成量化校准模型
quantize(model, cfg)
# 调用接口将量化校准模型转换为量化部署模型
convert(model)
```

从 artifact 恢复并转换：

```python
# fresh_model 是重新构建、尚未量化的同一浮点模型
convert(fresh_model, quant_result_dir="./quant_result")
```

`quant_result_dir=None` 时保持原有内存流程不变。artifact 格式和失败边界参见[量化结果 artifact](../quant_result_artifact.md)。
