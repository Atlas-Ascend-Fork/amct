# create\_distill\_config<a name="ZH-CN_TOPIC_0000002517028788"></a>

## 产品支持情况<a name="section185612964420"></a>

<a name="zh-cn_topic_0000002517188794_table38301303189"></a>

| 产品                                        | 是否支持 |
| ------------------------------------------- | -------- |
| Ascend 950PR/Ascend 950DT                   | √        |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √        |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √        |



## 功能说明<a name="section14397134012517"></a>

蒸馏接口，根据图的结构找到所有可蒸馏量化的层和可蒸馏量化的结构，自动生成蒸馏量化配置文件，并将可蒸馏量化层的量化配置和蒸馏结构写入配置文件。

## 函数原型<a name="section5661182817510"></a>

```python
create_distill_config(config_file, model, input_data, config_defination=None)
```

## 参数说明<a name="section73811524135618"></a>

<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="12.93%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="7.57%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240187365_p1769255516412"><a name="zh-cn_topic_0240187365_p1769255516412"></a><a name="zh-cn_topic_0240187365_p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="79.5%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0240187365_p15231205416325"><a name="zh-cn_topic_0240187365_p15231205416325"></a><a name="zh-cn_topic_0240187365_p15231205416325"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="12.93%" headers="mcps1.1.4.1.1 "><p id="p3829146134310"><a name="p3829146134310"></a><a name="p3829146134310"></a>config_file</p>
</td>
<td class="cellrowborder" valign="top" width="7.57%" headers="mcps1.1.4.1.2 "><p id="p11274056174319"><a name="p11274056174319"></a><a name="p11274056174319"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.5%" headers="mcps1.1.4.1.3 "><p id="p1835396124311"><a name="p1835396124311"></a><a name="p1835396124311"></a>含义：待生成的蒸馏量化配置文件存放路径及名称。如果存放路径下已经存在该文件，则调用该接口时会覆盖已有文件。</p>
<p id="p1727418563433"><a name="p1727418563433"></a><a name="p1727418563433"></a>数据类型：string</p>
</td>
</tr>
<tr id="row10475143684319"><td class="cellrowborder" valign="top" width="12.93%" headers="mcps1.1.4.1.1 "><p id="p128298463436"><a name="p128298463436"></a><a name="p128298463436"></a>model</p>
</td>
<td class="cellrowborder" valign="top" width="7.57%" headers="mcps1.1.4.1.2 "><p id="p8274556154316"><a name="p8274556154316"></a><a name="p8274556154316"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.5%" headers="mcps1.1.4.1.3 "><p id="p13317181214313"><a name="p13317181214313"></a><a name="p13317181214313"></a>含义：待进行蒸馏量化的原始浮点模型，已加载权重。</p>
<p id="p17274105617434"><a name="p17274105617434"></a><a name="p17274105617434"></a>数据类型：torch.nn.Module</p>
</td>
</tr>
<tr id="row11737244204314"><td class="cellrowborder" valign="top" width="12.93%" headers="mcps1.1.4.1.1 "><p id="p18829134610436"><a name="p18829134610436"></a><a name="p18829134610436"></a>input_data</p>
</td>
<td class="cellrowborder" valign="top" width="7.57%" headers="mcps1.1.4.1.2 "><p id="p18274156124317"><a name="p18274156124317"></a><a name="p18274156124317"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.5%" headers="mcps1.1.4.1.3 "><p id="p20817101314320"><a name="p20817101314320"></a><a name="p20817101314320"></a>含义：模型的输入数据。一个torch.tensor会被等价为tuple(torch.tensor)。</p>
<p id="p927405684319"><a name="p927405684319"></a><a name="p927405684319"></a>数据类型：tuple</p>
</td>
</tr>
<tr id="row41919427437"><td class="cellrowborder" valign="top" width="12.93%" headers="mcps1.1.4.1.1 "><p id="p158301146144319"><a name="p158301146144319"></a><a name="p158301146144319"></a>config_defination</p>
</td>
<td class="cellrowborder" valign="top" width="7.57%" headers="mcps1.1.4.1.2 "><p id="p1027410568437"><a name="p1027410568437"></a><a name="p1027410568437"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.5%" headers="mcps1.1.4.1.3 "><p id="p14527715174319"><a name="p14527715174319"></a><a name="p14527715174319"></a>含义：简易配置文件。基于distill_config_pytorch.proto文件生成的简易配置文件distill.cfg，*.proto文件所在路径为：<em id="i12274856104314"><a name="i12274856104314"></a><a name="i12274856104314"></a><span id="ph244511481409"><a name="ph244511481409"></a><a name="ph244511481409"></a>AMCT</span>安装目录</em>/amct_pytorch/proto/。*.proto文件参数解释以及生成的<em id="i3473122313913"><a name="i3473122313913"></a><a name="i3473122313913"></a>distill</em>.cfg简易量化配置文件样例请参见<a href="../context/蒸馏简易配置文件.md">蒸馏简易配置文件</a>。</p>
<p id="p1275756114320"><a name="p1275756114320"></a><a name="p1275756114320"></a>默认值：None。</p>
<p id="p3275175644316"><a name="p3275175644316"></a><a name="p3275175644316"></a>数据类型：string</p>
</td>
</tr>
</tbody>
</table>


## 返回值说明<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section293415513458"></a>

无

## 调用示例<a name="zh-cn_topic_0240188739_section64231658994"></a>

```python
import amct_pytorch as amct
# 建立待进行蒸馏量化的网络图结构
model = build_model()
model.load_state_dict(torch.load(state_dict_path))
input_data = tuple([torch.randn(input_shape)])
 
# 生成蒸馏配置文件
amct.create_distill_config(config_file="./configs/config.json",
                           model,
                           input_data,
                           config_defination="./configs/distill.cfg")
```

落盘文件说明：

生成JSON格式的蒸馏量化配置文件，样例如下（重新执行蒸馏时，该接口输出的配置文件将会被覆盖，如下为INT8量化场景下的配置文件）：

```json
{
    "version":1,
    "batch_num":1,
    "group_size":1,
    "data_dump":false,
    "distill_group":[
        [
            "conv1",
            "bn",
            "relu"
        ],
        [
            "conv2",
            "bn2",
            "relu2"
        ]
    ],
    "conv1":{
        "quant_enable":true,
        "distill_data_config":{
            "algo":"ulq_quantize",
            "dst_type":"INT8"
        },
        "distill_weight_config":{
            "algo":"arq_distill",
            "channel_wise":true,
            "dst_type":"INT8"
        }
    },
    "conv2":{
        "quant_enable":true,
        "distill_data_config":{
            "algo":"ulq_quantize",
            "dst_type":"INT8"
        },
        "distill_weight_config":{
            "algo":"arq_distill",
            "channel_wise":true,
            "dst_type":"INT8"
        }
    }
}
```

