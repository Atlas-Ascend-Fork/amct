# 接口介绍

## 支持的接口列表

-   **[quantize](./api/quantize.md)** ：高精度模型转换为校准模型，得到量化校准模型，推理后计算得到量化参数。
-   **[convert](./api/convert.md)** ： 将量化校准模型转换为量化部署模型。
-   **[algorithm\_register](./api/algorithm_register.md)** ：将用户提供的自定义算法注册到AMCT工具。



## 量化支持的层以及约束

| 支持的层类型    | 原始数据类型                                       | 支持的量化类型组合              | 约束                                                         |
| --------------- | -------------------------------------------------- | ------------------------------- | ------------------------------------------------------------ |
| torch.nn.Linear | float32（fp32）、float16（fp16）、bfloat16（bf16） | wts_type:int4                   | 权重量化粒度支持per-tensor、per-channel、per-group。<br>权重量化算法支持minmax、awq、gptq。 |
| -               | float16（fp16）、bfloat16（bf16）                  | wts_type:int8                   | 权重量化粒度支持per-tensor、per-channel、per-group。<br>权重量化算法支持minmax、awq、gptq。 |
| -               | float16（fp16）、bfloat16（bf16）                  | wts_type:int8<br/>act_type:int8 | 权重量化粒度支持per-tensor、per-channel。<br>权重量化算法支持minmax、smoothquant。<br>数据量化粒度支持per-tensor、per-token。<br>数据量化算法支持minmax、smoothquant。 |

注：act_type和wts_type参数分别指[quantize](./api/quantize.md)接口配置文件中激活（inputs）和权重（weight）的量化类型。

## 接口调用流程

![](figures/接口调用流程.png)

蓝色部分为用户实现，灰色部分为用户调用AMCT提供的API实现：

1. 用户首先构造PyTorch的原始模型（请确保原始模型可以正常推理）和量化配置，然后调用[quantize](./api/quantize.md)接口生成量化校准模型。
2. 然后调用[convert](./api/convert.md)接口，将量化校准算子转换为NPU对应的量化部署算子，并输出量化部署模型。

## 调用示例

1. 导入AMCT包。

   ```python
   import amct_pytorch as amct
   ```

2. 调用AMCT，量化模型。

   1. 生成量化校准模型。

      ```python
      # 建立待进行量化的网络图结构
      ori_model = build_model()
      model = copy.deepcopy(ori_model)
      # 量化配置
      cfg = {
              'batch_num': 1,
              'quant_cfg': {
                  'weights': {
                      'type': 'int8',
                      'symmetric': True,
                      'strategy': 'tensor',
                  },
              },
              'algorithm': {'minmax'},
              }
      # 调用量化接口生成量化校准模型
      amct.quantize(model, cfg)
      ```

   2. 生成量化部署模型。

      ```python
      # 调用接口将量化校准模型转换为量化部署模型
      amct.convert(model)
      ```