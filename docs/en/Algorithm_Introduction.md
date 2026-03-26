# Algorithm Introduction

## Weight-Only Quantization Algorithms

### ARQ

ARQ (Adaptive Range Quantization) algorithm is an algorithm that directly quantizes weights. This algorithm provides two methods: channel_wise (whether to use different quantization factors for each channel, for parameter explanation please refer to [Post-Training Quantization Configuration Parameters](../context/训练后量化配置参数.md)) quantization and non-channel_wise quantization. The quantization principle diagrams are as follows:

**Figure 1**  Non-channel_wise quantization<a name="fig7600113961719"></a.  
![](../figures/非channel_wise量化.png "Non-channel_wise quantization")

**Figure 2**  channel_wise quantization<a name="fig6351104815177"></a.  
![](../figures/channel_wise量化.png "channel_wise quantization")

In the quantization configuration, the parameter channel_wise provided by ARQuantize (taking PyTorch framework as an example, for parameter explanation please refer to [Table 1](../训练后量化简易配置文件.md#zh-cn_topic_0240188735_table1225503375617)) is used to select the quantization method:

- When channel_wise is False, select [Figure 1](#fig7600113961719) quantization method. All filters are analyzed together for data distribution and quantized. All different channels in a certain layer share one quantization factor.
- When channel_wise is True, select [Figure 2](#fig6351104815177) quantization method. Each filter performs data analysis independently and then is quantized. Each channel in the same layer has an independent quantization factor.

Generally, each filter is quantized independently, that is, performing [Figure 2](#fig6351104815177) quantization, which has higher quantization accuracy. If the data volume of each filter is relatively small, the quantization effect will be relatively poor. In this case, it is recommended to use [Figure 1](#fig7600113961719) quantization.

Note: Fully connected layers and average downsampling layers Pooling (downsampling method is AVE, and not global pooling) have no channels. When channel_wise is set to True, an error message will be prompted.
