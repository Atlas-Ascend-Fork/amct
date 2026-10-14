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
convert(model)
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                                         |
| ------ | --------- | ------------------------------------------------------------ |
| model  | 输入/输出 | 输入含义：量化校准模型。<br>输出含义：量化部署模型。<br>数据类型：torch.nn.Module。 |


## 返回值说明
量化部署模型。

## 调用示例
```python
# 调用前序接口生成量化校准模型
quantize(model, cfg)
# 调用接口将量化校准模型转换为量化部署模型
convert(model)
```

## 共享输入融合与部署权重兼容性

对已登记的 Transformers Qwen3DecoderLayer 和 ViTLayer 结构，MinMax、SmoothQuant 的共享输入组在校准时统一计算激活参数。SmoothQuant 还计算公共平滑参数，并在 convert 时融合到前置 Norm 和后置 Linear。未登记模型保持原量化流程。

静态量化的多成员组会将前置算子包装为 `FusedPreOp`，增加一次共享激活量化。部署模型仍为 `torch.nn.Module`，但包装位置的模块路径和 `state_dict` 键名会改变。例如：

| 原路径 | 融合部署模型中的路径 |
| --- | --- |
| `model.layers.0.input_layernorm.weight` | `model.layers.0.input_layernorm.pre_op.weight` |
| `model.layers.0.input_layernorm.bias`（如存在） | `model.layers.0.input_layernorm.pre_op.bias` |
| 无 | `model.layers.0.input_layernorm.shared.act_scale` |
| 无 | `model.layers.0.input_layernorm.shared.act_offset` |

ViT 的 `layernorm_before` 以及其他已融合前置算子采用相同映射。单成员静态组不插入共享量化包装。动态 per-token 通过替换所在层并显式传递量化输入和 token scale，不使用上述 `FusedPreOp` 包装。

保存、加载和下游导出应使用同一部署结构，并保留共享量化 buffer。`FusedPreOp.load_state_dict` 的递归加载兼容原前置算子路径，但仅用于已经融合好的部署参数的键名迁移；它不会自动将未融合浮点 checkpoint 转换为部署参数。共享量化 buffer 缺失时，严格加载仍会报错。按模块路径访问 Norm 参数的脚本应使用新的 `.pre_op` 路径。

动态融合层沿用当前依赖版本 Transformers 的 attention 后端、掩码、KV cache 和 dropout 逻辑。自定义层及未登记类型不会仅凭属性同名而被替换；模型根节点可直接作为已登记层参与结构识别和融合。
