# create\_quant\_cali\_config<a name="ZH-CN_TOPIC_0000002517188700"></a>

## 产品支持情况<a name="section197451857688"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row574891710101"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p177491117171015"><a name="p177491117171015"></a><a name="p177491117171015"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p14226338117"><a name="p14226338117"></a><a name="p14226338117"></a>√</p>
</td>
</tr>
<tr id="row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p48327011813"><a name="p48327011813"></a><a name="p48327011813"></a><span id="ph583230201815"><a name="ph583230201815"></a><a name="ph583230201815"></a><term id="zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term131434243115"><a name="zh-cn_topic_0000001312391781_term131434243115"></a><a name="zh-cn_topic_0000001312391781_term131434243115"></a>Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p108715341013"><a name="p108715341013"></a><a name="p108715341013"></a>√</p>
</td>
</tr>
<tr id="row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p14832120181815"><a name="p14832120181815"></a><a name="p14832120181815"></a><span id="ph1483216010188"><a name="ph1483216010188"></a><a name="ph1483216010188"></a><term id="zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品</term>/<term id="zh-cn_topic_0000001312391781_term184716139811"><a name="zh-cn_topic_0000001312391781_term184716139811"></a><a name="zh-cn_topic_0000001312391781_term184716139811"></a>Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p19948143911820"><a name="p19948143911820"></a><a name="p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

**注：标记“x”的产品，调用接口不会报错，但是获取不到性能收益。**

## 功能说明<a name="section14397134012517"></a>

KV-cache量化接口，根据用户传入模型、量化层信息与量化配置信息，生成每个层的详细量化配置。

## 函数原型<a name="section5661182817510"></a>

```python
create_quant_cali_config(config_file,model,quant_layers=None,config_defination=None)
```

## 参数说明<a name="section73811524135618"></a>

<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="12.898710128987101%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="8.18918108189181%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240187365_p1769255516412"><a name="zh-cn_topic_0240187365_p1769255516412"></a><a name="zh-cn_topic_0240187365_p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="78.91210878912109%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0240187365_p15231205416325"><a name="zh-cn_topic_0240187365_p15231205416325"></a><a name="zh-cn_topic_0240187365_p15231205416325"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="12.898710128987101%" headers="mcps1.1.4.1.1 "><p id="p3829146134310"><a name="p3829146134310"></a><a name="p3829146134310"></a>config_file</p>
</td>
<td class="cellrowborder" valign="top" width="8.18918108189181%" headers="mcps1.1.4.1.2 "><p id="p11274056174319"><a name="p11274056174319"></a><a name="p11274056174319"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.91210878912109%" headers="mcps1.1.4.1.3 "><p id="p1227053374911"><a name="p1227053374911"></a><a name="p1227053374911"></a>含义：待生成的量化配置文件存放路径及名称，文件为JSON格式，包含每个KV Cache量化层的量化配置信息。如果存放路径下已经存在该文件，则调用该接口时会覆盖已有文件。</p>
<p id="p1727418563433"><a name="p1727418563433"></a><a name="p1727418563433"></a>数据类型：string</p>
</td>
</tr>
<tr id="row10475143684319"><td class="cellrowborder" valign="top" width="12.898710128987101%" headers="mcps1.1.4.1.1 "><p id="p128298463436"><a name="p128298463436"></a><a name="p128298463436"></a>model</p>
</td>
<td class="cellrowborder" valign="top" width="8.18918108189181%" headers="mcps1.1.4.1.2 "><p id="p8274556154316"><a name="p8274556154316"></a><a name="p8274556154316"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.91210878912109%" headers="mcps1.1.4.1.3 "><p id="p13700164115496"><a name="p13700164115496"></a><a name="p13700164115496"></a>含义：用户提供的待量化模型。</p>
<p id="p17274105617434"><a name="p17274105617434"></a><a name="p17274105617434"></a>数据类型：torch.nn.Module</p>
</td>
</tr>
<tr id="row11737244204314"><td class="cellrowborder" valign="top" width="12.898710128987101%" headers="mcps1.1.4.1.1 "><p id="p18829134610436"><a name="p18829134610436"></a><a name="p18829134610436"></a>quant_layers</p>
</td>
<td class="cellrowborder" valign="top" width="8.18918108189181%" headers="mcps1.1.4.1.2 "><p id="p18274156124317"><a name="p18274156124317"></a><a name="p18274156124317"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.91210878912109%" headers="mcps1.1.4.1.3 "><p id="p661214394919"><a name="p661214394919"></a><a name="p661214394919"></a>含义：量化层信息，通过字典表示；如果传入了量化简易配置文件，则以配置文件为准。</p>
<p id="p1477212427156"><a name="p1477212427156"></a><a name="p1477212427156"></a>KV-cache量化示例如下：</p>
<pre class="screen" id="screen660818309166"><a name="screen660818309166"></a><a name="screen660818309166"></a>{'kv_cache_quant_layers': ['MatMul_1']}</pre>
<p id="p026696131716"><a name="p026696131716"></a><a name="p026696131716"></a>默认值：None</p>
<p id="p927405684319"><a name="p927405684319"></a><a name="p927405684319"></a>数据类型：dict</p>
<p id="p6591173620174"><a name="p6591173620174"></a><a name="p6591173620174"></a>使用约束：quant_layers既可以在参数中指定，也可以在简易配置文件添加：当取值为None时，以参数传递为准；否则以简易配置文件为准。</p>
</td>
</tr>
<tr id="row41919427437"><td class="cellrowborder" valign="top" width="12.898710128987101%" headers="mcps1.1.4.1.1 "><p id="p158301146144319"><a name="p158301146144319"></a><a name="p158301146144319"></a>config_defination</p>
</td>
<td class="cellrowborder" valign="top" width="8.18918108189181%" headers="mcps1.1.4.1.2 "><p id="p1027410568437"><a name="p1027410568437"></a><a name="p1027410568437"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="78.91210878912109%" headers="mcps1.1.4.1.3 "><p id="p55493456496"><a name="p55493456496"></a><a name="p55493456496"></a>含义：量化简易配置文件。</p>
<p id="p827435611439"><a name="p827435611439"></a><a name="p827435611439"></a>基于quant_calibration_config_pytorch.proto生成的简易配置文件quant.cfg，*.proto文件所在路径为：<em id="i12274856104314"><a name="i12274856104314"></a><a name="i12274856104314"></a><span id="ph244511481409"><a name="ph244511481409"></a><a name="ph244511481409"></a>AMCT</span>安装目录</em>/amct_pytorch/proto/。*.proto文件参数解释以及生成的quant.cfg简易量化配置文件样例请参见<a href="../context/KV-Cache量化简易配置文件.md">KV Cache量化简易配置文件</a>。</p>
<p id="p1275756114320"><a name="p1275756114320"></a><a name="p1275756114320"></a>默认值：None</p>
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
# 建立待量化的网络图结构
model = build_model()
model.load_state_dict(torch.load(state_dict_path))
input_data = tuple([torch.randn(input_shape)])

# 生成量化配置文件
amct.create_quant_cali_config(config_file="./configs/config.json",
                             model=model,
                             quant_layers=None,
                             config_defination="./configs/quant.cfg")
```

落盘文件说明：

生成JSON格式的量化配置文件（重新执行量化时，该接口输出的配置文件将会被覆盖），样例如下：

```json
{
    "batch_num":1,
    "activation_offset":true,
    "matmul1":{
        "kv_data_quant_config":{
            "act_algo":"hfmg",
            "num_of_bins":4096,
            "quant_granularity":0
        }
    },
    "matmul2":{
        "kv_data_quant_config":{
            "act_algo":"hfmg",
            "num_of_bins":4096,
            "quant_granularity":0
        }
    },
    "matmul3":{
        "kv_data_quant_config":{
            "act_algo":"ifmr",
            "max_percentile":0.999999,
            "min_percentile":0.999999,
            "search_range":[
                0.7,
                1.3
            ],
            "search_step":0.01,
            "quant_granularity":0
        }
    }
}
```

