# quantize

## 产品支持情况

| 产品        | 是否支持 |
| ----------- | -------- |
| Ascend 910C | √        |
| Ascend 910B | √        |

## 功能说明

高精度模型转换为校准模型，得到量化校准模型，推理后计算得到量化参数。

## 函数原型

```python
quantize(model, config)
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                                         |
| ------ | --------- | ------------------------------------------------------------ |
| model  | 输入/输出 | 输入含义：待量化的高精度模型。<br/>输出含义：量化校准模型。<br/>数据类型：torch.nn.Module |
| config | 输入      | 含义：量化配置。<br/>数据类型：自定义dict，其中包含weight/input/algorithm/skip_layers的配置，详细配置参数请参见[config详细配置](#config详细配置)。 |

## 返回值说明

无

## 调用示例

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
quantize(model, cfg)
```

## config详细配置

| key         | -       | -          | value                                                        |
| ----------- | ------- | ---------- | ------------------------------------------------------------ |
| batch_num   | -       | -          | uint32类型，量化使用的batch数量。                            |
| quant_cfg   | -       | -          | 量化配置。                                                   |
| -           | weights | -          | 仅权重量化配置。                                             |
| -           | -       | type       | string类型，权重（weight）量化粒度。当前支持int4和int8。     |
| -           | -       | symmetric  | bool类型，权重是否为对称量化。<br>True：对称量化。<br>False：非对称量化。 |
| -           | -       | strategy   | string类型，权重量化粒度。<br>tensor，对应per-tensor。<br>channel，对应per-channel。<br>group，对应per-group。<br>量化粒度介绍请参见[量化基础介绍](../量化概念.md)。 |
| -           | -       | group_size | 仅权重量化场景配置，per-group量化粒度下group的大小，该参数只有配置了per-group后，才能配置。<br>要求传入值的范围为[32, K-1]且必须是32的倍数。 |
| -           | inputs  | -          | 数据量化配置。                                               |
| -           | -       | type       | string类型，数据（activation）量化粒度。目前仅支持int8。     |
| -           | -       | symmetric  | bool类型，数据是否为对称量化。<br>True：对称量化。<br>False：非对称量化。 |
| -           | -       | strategy   | string类型，数据量化粒度。<br>tensor，对应per-tensor。<br>token，对应per-token。 |
| algorithm   | -       | -          | string类型，量化算法，支持如下配置：<br/>awq：grids_num，uint32类型，搜索格点数量。AWQ算法求解量化参数的过程中，对候选值做网格划分，grids_num越大，搜索粒度越大，量化误差越小，但计算耗时增加。默认为20。<br/>gptq。<br/>minmax。<br/>smoothquant：smooth_strength，float类型，迁移强度，代表将activation数据上的量化难度迁移至weight权重的程度。默认值0.5，数据分布的离群值越大迁移强度应设置较小。<br/>具体请参见[量化算法介绍](../算法介绍.md)。 |
| skip_layers | -       | -          | string类型，按层名跳过哪些层不做量化，全局配置参数。指定层名后，只要层名包括用户设置的字符串，就跳过该层不做量化。 |
