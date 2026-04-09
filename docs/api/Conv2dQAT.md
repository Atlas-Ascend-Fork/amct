# Conv2dQAT<a name="ZH-CN_TOPIC_0000002548788615"></a>

## 产品支持情况<a name="section185612964420"></a>

<a name="zh-cn_topic_0000002517188794_table38301303189"></a>

| 产品                                        | 是否支持 |
| ------------------------------------------- | -------- |
| Ascend 950PR/Ascend 950DT                   | √        |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √        |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √        |


## 功能说明<a name="zh-cn_topic_0240187365_section15406195619561"></a>

构造Conv2d的QAT算子。

## 函数原型<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_section428121323411"></a>

-   直接构造接口：

    ```python
    qat = amct_pytorch.nn.module.quantization.conv2d.Conv2dQAT(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype, config)
    ```

-   基于原生算子构造接口：

    ```python
    qat = amct_pytorch.nn.module.quantization.conv2d.Conv2dQAT.from_float(mod, config)
    ```

## 参数说明<a name="section73811524135618"></a>

**表 1**  直接构造接口参数说明

<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="11.600000000000001%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="7.82%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0240187365_p1769255516412"><a name="zh-cn_topic_0240187365_p1769255516412"></a><a name="zh-cn_topic_0240187365_p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="80.58%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0240187365_p15231205416325"><a name="zh-cn_topic_0240187365_p15231205416325"></a><a name="zh-cn_topic_0240187365_p15231205416325"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="11.600000000000001%" headers="mcps1.2.4.1.1 "><p id="p558952613111"><a name="p558952613111"></a><a name="p558952613111"></a>in_channels</p>
</td>
<td class="cellrowborder" valign="top" width="7.82%" headers="mcps1.2.4.1.2 "><p id="p19832329363"><a name="p19832329363"></a><a name="p19832329363"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.58%" headers="mcps1.2.4.1.3 "><p id="p66038591458"><a name="p66038591458"></a><a name="p66038591458"></a>含义：输入channel个数。</p>
<p id="p193951754413"><a name="p193951754413"></a><a name="p193951754413"></a>数据类型：int</p>
</td>
</tr>
<tr id="row10475143684319"><td class="cellrowborder" valign="top" width="11.600000000000001%" headers="mcps1.2.4.1.1 "><p id="p162911910113611"><a name="p162911910113611"></a><a name="p162911910113611"></a>out_channels</p>
</td>
<td class="cellrowborder" valign="top" width="7.82%" headers="mcps1.2.4.1.2 "><p id="p19831321362"><a name="p19831321362"></a><a name="p19831321362"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.58%" headers="mcps1.2.4.1.3 "><p id="p155218704612"><a name="p155218704612"></a><a name="p155218704612"></a>含义：输出channel个数。</p>
<p id="p4403105711149"><a name="p4403105711149"></a><a name="p4403105711149"></a>数据类型：int</p>
</td>
</tr>
<tr id="row11737244204314"><td class="cellrowborder" valign="top" width="11.600000000000001%" headers="mcps1.2.4.1.1 "><p id="p258918267312"><a name="p258918267312"></a><a name="p258918267312"></a>kernel_size</p>
</td>
<td class="cellrowborder" valign="top" width="7.82%" headers="mcps1.2.4.1.2 "><p id="p1598233212367"><a name="p1598233212367"></a><a name="p1598233212367"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.58%" headers="mcps1.2.4.1.3 "><p id="p1515559144620"><a name="p1515559144620"></a><a name="p1515559144620"></a>含义：卷积核大小。</p>
<p id="p1239175124112"><a name="p1239175124112"></a><a name="p1239175124112"></a>数据类型：int/tuple</p>
</td>
</tr>
<tr id="row41919427437"><td class="cellrowborder" valign="top" width="11.600000000000001%" headers="mcps1.2.4.1.1 "><p id="p18589162616312"><a name="p18589162616312"></a><a name="p18589162616312"></a>stride</p>
</td>
<td class="cellrowborder" valign="top" width="7.82%" headers="mcps1.2.4.1.2 "><p id="p098023253618"><a name="p098023253618"></a><a name="p098023253618"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.58%" headers="mcps1.2.4.1.3 "><p id="p1251221124619"><a name="p1251221124619"></a><a name="p1251221124619"></a>含义：卷积步长。</p>
<p id="p227516563436"><a name="p227516563436"></a><a name="p227516563436"></a>数据类型：int/tuple</p>
<p id="p20418832171519"><a name="p20418832171519"></a><a name="p20418832171519"></a>默认值：1</p>
</td>
</tr>
<tr id="row1746261023713"><td class="cellrowborder" valign="top" width="11.600000000000001%" headers="mcps1.2.4.1.1 "><p id="p17589926163112"><a name="p17589926163112"></a><a name="p17589926163112"></a>padding</p>
</td>
<td class="cellrowborder" valign="top" width="7.82%" headers="mcps1.2.4.1.2 "><p id="p1146341043712"><a name="p1146341043712"></a><a name="p1146341043712"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.58%" headers="mcps1.2.4.1.3 "><p id="p12137171316463"><a name="p12137171316463"></a><a name="p12137171316463"></a>含义：填充大小。</p>
<p id="p15356205881619"><a name="p15356205881619"></a><a name="p15356205881619"></a>数据类型：int/tuple</p>
<p id="p6356105818169"><a name="p6356105818169"></a><a name="p6356105818169"></a>默认值：0</p>
</td>
</tr>
<tr id="row13859101213717"><td class="cellrowborder" valign="top" width="11.600000000000001%" headers="mcps1.2.4.1.1 "><p id="p19589152613111"><a name="p19589152613111"></a><a name="p19589152613111"></a>dilation</p>
</td>
<td class="cellrowborder" valign="top" width="7.82%" headers="mcps1.2.4.1.2 "><p id="p4859151215377"><a name="p4859151215377"></a><a name="p4859151215377"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.58%" headers="mcps1.2.4.1.3 "><p id="p1655091434616"><a name="p1655091434616"></a><a name="p1655091434616"></a>含义：kernel元素之间的间距。</p>
<p id="p114891831716"><a name="p114891831716"></a><a name="p114891831716"></a>数据类型：int/tuple</p>
<p id="p8859171273710"><a name="p8859171273710"></a><a name="p8859171273710"></a>默认值：1</p>
</td>
</tr>
<tr id="row1444111152375"><td class="cellrowborder" valign="top" width="11.600000000000001%" headers="mcps1.2.4.1.1 "><p id="p658962613112"><a name="p658962613112"></a><a name="p658962613112"></a>groups</p>
</td>
<td class="cellrowborder" valign="top" width="7.82%" headers="mcps1.2.4.1.2 "><p id="p5441151512370"><a name="p5441151512370"></a><a name="p5441151512370"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.58%" headers="mcps1.2.4.1.3 "><p id="p99806173469"><a name="p99806173469"></a><a name="p99806173469"></a>含义：输入和输出的连接关系。</p>
<p id="p444121512372"><a name="p444121512372"></a><a name="p444121512372"></a>数据类型：int</p>
<p id="p205195115174"><a name="p205195115174"></a><a name="p205195115174"></a>默认值：1</p>
</td>
</tr>
<tr id="row7355184753719"><td class="cellrowborder" valign="top" width="11.600000000000001%" headers="mcps1.2.4.1.1 "><p id="p758902643119"><a name="p758902643119"></a><a name="p758902643119"></a>bias</p>
</td>
<td class="cellrowborder" valign="top" width="7.82%" headers="mcps1.2.4.1.2 "><p id="p9355247133712"><a name="p9355247133712"></a><a name="p9355247133712"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.58%" headers="mcps1.2.4.1.3 "><p id="p178722119462"><a name="p178722119462"></a><a name="p178722119462"></a>含义：是否开启偏置项参与学习。</p>
<p id="p173551747113712"><a name="p173551747113712"></a><a name="p173551747113712"></a>数据类型：bool，其他数据类型（比如整数，字符串，列表等）按照Python真值判断规则转换。</p>
<p id="p1635281111813"><a name="p1635281111813"></a><a name="p1635281111813"></a>默认值：True</p>
</td>
</tr>
<tr id="row1381165643710"><td class="cellrowborder" valign="top" width="11.600000000000001%" headers="mcps1.2.4.1.1 "><p id="p1158910260319"><a name="p1158910260319"></a><a name="p1158910260319"></a>padding_mode</p>
</td>
<td class="cellrowborder" valign="top" width="7.82%" headers="mcps1.2.4.1.2 "><p id="p18382115633714"><a name="p18382115633714"></a><a name="p18382115633714"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.58%" headers="mcps1.2.4.1.3 "><p id="p84961522124618"><a name="p84961522124618"></a><a name="p84961522124618"></a>含义：填充方式。</p>
<p id="p10382135617375"><a name="p10382135617375"></a><a name="p10382135617375"></a>使用约束：仅支持zeros</p>
</td>
</tr>
<tr id="row13521134203811"><td class="cellrowborder" valign="top" width="11.600000000000001%" headers="mcps1.2.4.1.1 "><p id="p1058982614312"><a name="p1058982614312"></a><a name="p1058982614312"></a>device</p>
</td>
<td class="cellrowborder" valign="top" width="7.82%" headers="mcps1.2.4.1.2 "><p id="p65217483815"><a name="p65217483815"></a><a name="p65217483815"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.58%" headers="mcps1.2.4.1.3 "><p id="p99524154613"><a name="p99524154613"></a><a name="p99524154613"></a>含义：运行设备。</p>
<p id="p752118411383"><a name="p752118411383"></a><a name="p752118411383"></a>默认值：None</p>
</td>
</tr>
<tr id="row19812101220389"><td class="cellrowborder" valign="top" width="11.600000000000001%" headers="mcps1.2.4.1.1 "><p id="p195894262310"><a name="p195894262310"></a><a name="p195894262310"></a>dtype</p>
</td>
<td class="cellrowborder" valign="top" width="7.82%" headers="mcps1.2.4.1.2 "><p id="p14812412133816"><a name="p14812412133816"></a><a name="p14812412133816"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.58%" headers="mcps1.2.4.1.3 "><p id="p16911825144614"><a name="p16911825144614"></a><a name="p16911825144614"></a>含义：torch数值类型。</p>
<p id="p281221212389"><a name="p281221212389"></a><a name="p281221212389"></a>torch数据类型，仅支持torch.float32</p>
</td>
</tr>
<tr id="row1292921916384"><td class="cellrowborder" valign="top" width="11.600000000000001%" headers="mcps1.2.4.1.1 "><p id="p19291619133812"><a name="p19291619133812"></a><a name="p19291619133812"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="7.82%" headers="mcps1.2.4.1.2 "><p id="p1892951920388"><a name="p1892951920388"></a><a name="p1892951920388"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.58%" headers="mcps1.2.4.1.3 "><p id="p9236112719464"><a name="p9236112719464"></a><a name="p9236112719464"></a>含义：量化配置，配置参考样例如下，量化配置参数的具体说明请参见<a href="../context/单算子模式量化配置参数.md">量化配置参数说明</a></p>
<pre class="screen" id="screen168684451264"><a name="screen168684451264"></a><a name="screen168684451264"></a>config = {
    "retrain_enable":true,
    "retrain_data_config": {
        "dst_type": "INT8",
        "batch_num": 10,
        "fixed_min": False,
        "clip_min": -1.0,
        "clip_max": 1.0
    },
    "retrain_weight_config": {
        "dst_type": "INT8",
        "weights_retrain_algo": "arq_retrain",
        "channel_wise": False
    }
}</pre>
<p id="p3275175644316"><a name="p3275175644316"></a><a name="p3275175644316"></a>数据类型：dict</p>
<p id="p260612831619"><a name="p260612831619"></a><a name="p260612831619"></a>默认值：None</p>
</td>
</tr>
</tbody>
</table>


**表 2**  基于原生算子构造接口

<a name="table18947259560"></a>
<table><thead align="left"><tr id="row1589410251565"><th class="cellrowborder" valign="top" width="12.15%" id="mcps1.2.4.1.1"><p id="p1089415253564"><a name="p1089415253564"></a><a name="p1089415253564"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="7.79%" id="mcps1.2.4.1.2"><p id="p18894425155619"><a name="p18894425155619"></a><a name="p18894425155619"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="80.06%" id="mcps1.2.4.1.3"><p id="p188944258569"><a name="p188944258569"></a><a name="p188944258569"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row19894112555615"><td class="cellrowborder" valign="top" width="12.15%" headers="mcps1.2.4.1.1 "><p id="p2203115517562"><a name="p2203115517562"></a><a name="p2203115517562"></a>mod</p>
</td>
<td class="cellrowborder" valign="top" width="7.79%" headers="mcps1.2.4.1.2 "><p id="p4894142514561"><a name="p4894142514561"></a><a name="p4894142514561"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.06%" headers="mcps1.2.4.1.3 "><p id="p1672202911461"><a name="p1672202911461"></a><a name="p1672202911461"></a>含义：待量化的原生Conv2d算子</p>
<p id="p118941025175610"><a name="p118941025175610"></a><a name="p118941025175610"></a>数据类型：torch.nn.Module</p>
</td>
</tr>
<tr id="row178961825155618"><td class="cellrowborder" valign="top" width="12.15%" headers="mcps1.2.4.1.1 "><p id="p128526535334"><a name="p128526535334"></a><a name="p128526535334"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="7.79%" headers="mcps1.2.4.1.2 "><p id="p385255363313"><a name="p385255363313"></a><a name="p385255363313"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="80.06%" headers="mcps1.2.4.1.3 "><p id="p493593134619"><a name="p493593134619"></a><a name="p493593134619"></a>含义：量化配置，配置参考样例如下，量化配置参数的具体说明请参见<a href="../context/单算子模式量化配置参数.md">量化配置参数说明</a></p>
<pre class="screen" id="screen178520531333"><a name="screen178520531333"></a><a name="screen178520531333"></a>config = {
    "retrain_enable":true,
    "retrain_data_config": {
        "dst_type": "INT8",
        "batch_num": 10,
        "fixed_min": False,
        "clip_min": -1.0,
        "clip_max": 1.0
    },
    "retrain_weight_config": {
        "dst_type": "INT8",
        "weights_retrain_algo": "arq_retrain",
        "channel_wise": False
    }
}</pre>
<p id="p8852653133318"><a name="p8852653133318"></a><a name="p8852653133318"></a>数据类型：dict</p>
<p id="p6983716191613"><a name="p6983716191613"></a><a name="p6983716191613"></a>默认值：None</p>
</td>
</tr>
</tbody>
</table>


## 返回值说明<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section293415513458"></a>

-   直接构造：返回构造的QAT单算子实例。
-   基于原生算子构造：torch.nn.Module转化后的QAT单算子。

## 调用示例<a name="zh-cn_topic_0240188739_section64231658994"></a>

-   直接构造：

    ```python
    from amct_pytorch.nn.module.quantization.conv2d import Conv2dQAT
    
    Conv2dQAT(in_channels=1, out_channels=1, kernel_size=1, stride=1,
              padding=0, dilation=1, groups=1, bias=True,
              padding_mode='zeros', device=None, dtype=None, config=None)
    ```

-   基于原生算子构造：

    ```python
    import torch
    
    from amct_pytorch.nn.module.quantization.conv2d import Conv2dQAT
    
    conv2d_op = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1,
                                padding=0, dilation=1, groups=1, bias=True,
                                padding_mode='zeros', device=None, dtype=None)
    Conv2dQAT.from_float(mod=conv2d_op, config=None)
    ```

