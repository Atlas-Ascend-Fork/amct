# algorithm\_register

## 产品支持情况

| 产品        | 是否支持 |
| ----------- | -------- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √        |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √        |

## 功能说明

将用户提供的自定义算法注册到AMCT工具。

## 函数原型

```python
algorithm_register(name, src_op, quant_op, deploy_op)
```

## 参数说明

| 参数名    | 输入/输出 | 说明                                             |
| --------- | --------- | ------------------------------------------------ |
| name      | 输入      | 含义：算法名称。<br/>数据类型：string。          |
| src_op    | 输入      | 含义：替换的算子。<br/>数据类型：string。        |
| quant_op  | 输入      | 含义：量化算子。<br/>数据类型：torch.nn.Module。 |
| deploy_op | 输入      | 含义：部署算子。<br/>数据类型：torch.nn.Module。 |

## 返回值说明

无

## 调用示例

```python
# 自定义算法名称
name = 'customize_algo'
# 需要量化的算子类型
src_op = 'Linear'
# 用户自己实现的量化算子
class CustomizedQuantOp(BaseQuantizeModule):
    def __init__(self,
                 ori_module,
                 layer_name,
                 quant_config):
        super().__init__(ori_module, layer_name, quant_config)
        
    @torch.no_grad()
    def forward(self, inputs):
        return
quant_op = CustomizedQuantOp
# 用户自己实现的部署算子
class CustomizedDeployOp(torch.nn.Module):
    def __init__(self, quant_module):
        super().__init__()
    
    def forward(self, x):
        return
deploy_op = CustomizedDeployOp
# 注册自定义算法
algorithm_register(name, src_op, quant_op, deploy_op)
```

