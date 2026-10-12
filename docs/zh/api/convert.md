# convert

## 产品支持情况

| 产品        | 是否支持 |
| ----------- | -------- |
| Ascend 950PR&950DT系列产品 | √ |
| Atlas A3系列产品 | √        |
| Atlas A2系列产品 | √        |

## 功能说明

将量化校准模型转换为量化部署模型。

## 函数原型

```python
convert(model, quant_result_dir=None)
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                                         |
| ------ | --------- | ------------------------------------------------------------ |
| model  | 输入/输出 | 输入含义：量化校准模型。<br>输出含义：量化部署模型。<br>数据类型：torch.nn.Module。 |
| quant_result_dir | 输入 | 输入含义：quantize(..., quant_result_dir=...) 生成的已发布量化结果目录；传入时 model 必须是生成该 artifact 时使用的原始浮点模型，接口先恢复校准状态，再执行部署转换。<br>数据类型：string 或 None。默认 None 表示 model 已是量化校准模型，保持原有转换流程。 |


## 返回值说明

量化部署模型。

## 注意事项

convert(..., quant_result_dir=...) 只接受已发布的量化结果目录，即目录中必须存在 quant_manifest.json。未完成校准的目录不是有效 artifact；接口会报错并提示未发布单元，请重新执行校准直到最后一个根模型 forward 生成 quant_manifest.json。

## 调用示例
```python
# 调用前序接口生成量化校准模型
quantize(model, cfg)
# 调用接口将量化校准模型转换为量化部署模型
convert(model)
```
