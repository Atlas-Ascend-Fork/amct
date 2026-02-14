# create\_quant\_retrain\_config<a name="ZH-CN_TOPIC_0000002517028880"></a>

## 产品支持情况<a name="section197451857688"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.95%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42.05%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row574891710101"><td class="cellrowborder" valign="top" width="57.95%" headers="mcps1.1.3.1.1 "><p id="p177491117171015"><a name="p177491117171015"></a><a name="p177491117171015"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42.05%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002517188794_p14226338117"><a name="zh-cn_topic_0000002517188794_p14226338117"></a><a name="zh-cn_topic_0000002517188794_p14226338117"></a>√</p>
</td>
</tr>
<tr id="row220181016240"><td class="cellrowborder" valign="top" width="57.95%" headers="mcps1.1.3.1.1 "><p id="p48327011813"><a name="p48327011813"></a><a name="p48327011813"></a><span id="ph583230201815"><a name="ph583230201815"></a><a name="ph583230201815"></a><term id="zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term131434243115"><a name="zh-cn_topic_0000001312391781_term131434243115"></a><a name="zh-cn_topic_0000001312391781_term131434243115"></a>Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42.05%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002517188794_p108715341013"><a name="zh-cn_topic_0000002517188794_p108715341013"></a><a name="zh-cn_topic_0000002517188794_p108715341013"></a>√</p>
</td>
</tr>
<tr id="row173226882415"><td class="cellrowborder" valign="top" width="57.95%" headers="mcps1.1.3.1.1 "><p id="p14832120181815"><a name="p14832120181815"></a><a name="p14832120181815"></a><span id="ph1483216010188"><a name="ph1483216010188"></a><a name="ph1483216010188"></a><term id="zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term184716139811"><a name="zh-cn_topic_0000001312391781_term184716139811"></a><a name="zh-cn_topic_0000001312391781_term184716139811"></a>Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42.05%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002517188794_p19948143911820"><a name="zh-cn_topic_0000002517188794_p19948143911820"></a><a name="zh-cn_topic_0000002517188794_p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="zh-cn_topic_0240187365_section15406195619561"></a>

量化感知训练接口，根据图的结构找到所有可量化的层，自动生成量化配置文件，并将可量化层的量化配置信息写入配置文件。

## 函数原型<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_section428121323411"></a>

```python
create_quant_retrain_config(config_file, model, input_data, config_defination=None)
```

## 参数说明<a name="section73811524135618"></a>

<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="12.370000000000001%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="7.290000000000001%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240187365_p1769255516412"><a name="zh-cn_topic_0240187365_p1769255516412"></a><a name="zh-cn_topic_0240187365_p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="80.34%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0240187365_p15231205416325"><a name="zh-cn_topic_0240187365_p15231205416325"></a><a name="zh-cn_topic_0240187365_p15231205416325"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="12.370000000000001%" headers="mcps1.1.4.1.1 "><p id="p3829146134310"><a name="p3829146134310"></a><a name="p3829146134310"></a>config_file</p>
</td>
<td class="cellrowborder" valign="top" width="7.290000000000001%" headers="mcps1.1.4.1.2 "><p id="p11274056174319"><a name="p11274056174319"></a><a name="p11274056174319"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.34%" headers="mcps1.1.4.1.3 "><p id="p20591103915201"><a name="p20591103915201"></a><a name="p20591103915201"></a>含义：待生成的量化感知训练配置文件存放路径及名称。如果存放路径下已经存在该文件，则调用该接口时会覆盖已有文件。</p>
<p id="p1727418563433"><a name="p1727418563433"></a><a name="p1727418563433"></a>数据类型：string</p>
</td>
</tr>
<tr id="row10475143684319"><td class="cellrowborder" valign="top" width="12.370000000000001%" headers="mcps1.1.4.1.1 "><p id="p128298463436"><a name="p128298463436"></a><a name="p128298463436"></a>model</p>
</td>
<td class="cellrowborder" valign="top" width="7.290000000000001%" headers="mcps1.1.4.1.2 "><p id="p8274556154316"><a name="p8274556154316"></a><a name="p8274556154316"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.34%" headers="mcps1.1.4.1.3 "><p id="p121542444203"><a name="p121542444203"></a><a name="p121542444203"></a>含义：待进行量化感知训练的模型，已加载权重。</p>
<p id="p17274105617434"><a name="p17274105617434"></a><a name="p17274105617434"></a>数据类型：torch.nn.Module</p>
</td>
</tr>
<tr id="row11737244204314"><td class="cellrowborder" valign="top" width="12.370000000000001%" headers="mcps1.1.4.1.1 "><p id="p18829134610436"><a name="p18829134610436"></a><a name="p18829134610436"></a>input_data</p>
</td>
<td class="cellrowborder" valign="top" width="7.290000000000001%" headers="mcps1.1.4.1.2 "><p id="p18274156124317"><a name="p18274156124317"></a><a name="p18274156124317"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.34%" headers="mcps1.1.4.1.3 "><p id="p1970764572015"><a name="p1970764572015"></a><a name="p1970764572015"></a>含义：模型的输入数据。一个torch.tensor会被等价为tuple(torch.tensor)。</p>
<p id="p927405684319"><a name="p927405684319"></a><a name="p927405684319"></a>数据类型：tuple</p>
</td>
</tr>
<tr id="row41919427437"><td class="cellrowborder" valign="top" width="12.370000000000001%" headers="mcps1.1.4.1.1 "><p id="p158301146144319"><a name="p158301146144319"></a><a name="p158301146144319"></a>config_defination</p>
</td>
<td class="cellrowborder" valign="top" width="7.290000000000001%" headers="mcps1.1.4.1.2 "><p id="p1027410568437"><a name="p1027410568437"></a><a name="p1027410568437"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.34%" headers="mcps1.1.4.1.3 "><p id="p17143647122018"><a name="p17143647122018"></a><a name="p17143647122018"></a>含义：简易配置文件。</p>
<p id="p827435611439"><a name="p827435611439"></a><a name="p827435611439"></a>基于retrain_config_pytorch.proto文件生成的简易配置文件quant.cfg，*.proto文件所在路径为：<em id="i12274856104314"><a name="i12274856104314"></a><a name="i12274856104314"></a><span id="ph244511481409"><a name="ph244511481409"></a><a name="ph244511481409"></a>AMCT</span>安装目录</em>/amct_pytorch/proto/。*.proto文件参数解释以及生成的<em id="i1127455694315"><a name="i1127455694315"></a><a name="i1127455694315"></a>quant</em>.cfg简易量化配置文件样例请参见<a href="../context/量化感知训练简易配置文件.md">量化感知训练简易配置文件</a>。</p>
<p id="p1275756114320"><a name="p1275756114320"></a><a name="p1275756114320"></a>默认值：None。</p>
<p id="p3275175644316"><a name="p3275175644316"></a><a name="p3275175644316"></a>数据类型：string</p>
</td>
</tr>
</tbody>
</table>



## 量化支持的层及约束

<a name="table206799319355"></a>

<table><thead align="left"><tr id="row26791539351"><th class="cellrowborder" valign="top" width="19.521952195219523%" id="mcps1.2.4.1.1"><p id="p1921472241319"><a name="p1921472241319"></a><a name="p1921472241319"></a>支持的层类型</p>
</th>
<th class="cellrowborder" valign="top" width="51.93519351935193%" id="mcps1.2.4.1.2"><p id="p16791313517"><a name="p16791313517"></a><a name="p16791313517"></a>约束</p>
</th>
<th class="cellrowborder" valign="top" width="28.542854285428543%" id="mcps1.2.4.1.3"><p id="p86801434351"><a name="p86801434351"></a><a name="p86801434351"></a>备注</p>
</th>
</tr>
</thead>
<tbody><tr id="row168043183517"><td class="cellrowborder" valign="top" width="19.521952195219523%" headers="mcps1.2.4.1.1 "><p id="p46801332355"><a name="p46801332355"></a><a name="p46801332355"></a>torch.nn.Linear</p>
</td>
<td class="cellrowborder" valign="top" width="51.93519351935193%" headers="mcps1.2.4.1.2 "><p id="p136801833354"><a name="p136801833354"></a><a name="p136801833354"></a>-</p>
</td>
<td class="cellrowborder" rowspan="3" valign="top" width="28.542854285428543%" headers="mcps1.2.4.1.3 "><p id="p15127478257"><a name="p15127478257"></a><a name="p15127478257"></a>复用层（共用weight和bias参数）不支持量化。</p>
</td>
</tr>
<tr id="row6680153123510"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1868011363517"><a name="p1868011363517"></a><a name="p1868011363517"></a>torch.nn.Conv2d</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><a name="ul16547311918"></a><a name="ul16547311918"></a><ul id="ul16547311918"><li>padding_mode为zeros</li><li>由于硬件约束，原始模型中输入通道数Cin&lt;=16时不建议进行量化感知训练，否则可能会导致量化后的部署模型推理时精度下降</li><li>只支持input data的shape为(N, Cin, Hin, Win)</li></ul>
</td>
</tr>
<tr id="row24121633571"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p04138331776"><a name="p04138331776"></a><a name="p04138331776"></a>torch.nn.ConvTranspose2d</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><a name="ul4861040154317"></a><a name="ul4861040154317"></a><ul id="ul4861040154317"><li>padding_mode为zeros</li><li>由于硬件约束，原始模型中输入通道数Cin&lt;=16时不建议进行量化感知训练，否则可能会导致量化后的部署模型推理时精度下降</li><li>只支持input data的shape为(N, Cin, Hin, Win)</li></ul>
</td>
</tr>
</tbody>
</table>



## 返回值说明<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section293415513458"></a>

无

## 调用示例<a name="zh-cn_topic_0240188739_section64231658994"></a>

```python
import amct_pytorch as amct
# 建立待量化的网络图结构
model = build_model()
model.load_state_dict(torch.load(state_dict_path))
input_data = tuple([torch.randn(input_shape)])
 
# 生成量化配置文件
amct.create_quant_retrain_config(config_file="./configs/config.json",
                            model=model,
                            input_data=input_data)
```

落盘文件说明：生成JSON格式的量化感知训练配置文件，样例如下（重新执行量化感知训练时，该接口输出的配置文件将会被覆盖），参数解释请参见[量化感知训练配置参数](../context/量化感知训练配置参数.md)。

```json
{
    "version":1,
    "batch_num":1,
    "conv1":{
        "retrain_enable":true,
        "retrain_data_config":{
            "algo":"ulq_quantize",
            "dst_type":"INT8"
        },
        "retrain_weight_config":{
            "algo":"arq_retrain",
            "channel_wise":true,
            "dst_type":"INT8"
        }
    },
    "layer1.0.conv1":{
        "retrain_enable":true,
        "retrain_data_config":{
            "algo":"ulq_quantize",
            "dst_type":"INT8"
        },
        "retrain_weight_config":{
            "algo":"arq_retrain",
            "channel_wise":true,
            "dst_type":"INT8"
        }
    },
    "fc":{
        "retrain_enable":true,
        "retrain_data_config":{
            "algo":"ulq_quantize",
            "dst_type":"INT8"
        },
        "retrain_weight_config":{
            "algo":"arq_retrain",
            "channel_wise":false,
            "dst_type":"INT8"
        }
    }
}
```

