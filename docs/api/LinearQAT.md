# LinearQAT<a name="ZH-CN_TOPIC_0000002517028804"></a>

## 产品支持情况<a name="section185612964420"></a>

<a name="zh-cn_topic_0000002517188794_table38301303189"></a>

| 产品                                        | 是否支持 |
| ------------------------------------------- | -------- |
| Ascend 950PR/Ascend 950DT                   | √        |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √        |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √        |


## 功能说明<a name="zh-cn_topic_0240187365_section15406195619561"></a>

构造Linear的QAT算子。

## 函数原型<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_section428121323411"></a>

-   直接构造接口：

    ```python
    qat = amct_pytorch.nn.module.quantization.linear.LinearQAT(in_features, out_features, bias, device, dtype, config)
    ```

-   基于原生算子构造接口：

    ```python
    qat = amct_pytorch.nn.module.quantization.linear.LinearQAT.from_float(mod, config)
    ```

## 参数说明<a name="section73811524135618"></a>

**表 1**  直接构造接口参数说明

<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001511339104_zh-cn_topic_0240187365_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="11.91%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001511339104_zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0000001511339104_zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0000001511339104_zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="8.25%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001511339104_zh-cn_topic_0240187365_p1769255516412"><a name="zh-cn_topic_0000001511339104_zh-cn_topic_0240187365_p1769255516412"></a><a name="zh-cn_topic_0000001511339104_zh-cn_topic_0240187365_p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="79.84%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001511339104_zh-cn_topic_0240187365_p15231205416325"><a name="zh-cn_topic_0000001511339104_zh-cn_topic_0240187365_p15231205416325"></a><a name="zh-cn_topic_0000001511339104_zh-cn_topic_0240187365_p15231205416325"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001511339104_zh-cn_topic_0240187365_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="11.91%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001511339104_p558952613111"><a name="zh-cn_topic_0000001511339104_p558952613111"></a><a name="zh-cn_topic_0000001511339104_p558952613111"></a>in_features</p>
</td>
<td class="cellrowborder" valign="top" width="8.25%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001511339104_p19832329363"><a name="zh-cn_topic_0000001511339104_p19832329363"></a><a name="zh-cn_topic_0000001511339104_p19832329363"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.84%" headers="mcps1.2.4.1.3 "><p id="p17841574317"><a name="p17841574317"></a><a name="p17841574317"></a>含义：输入特征数。</p>
<p id="zh-cn_topic_0000001511339104_p193951754413"><a name="zh-cn_topic_0000001511339104_p193951754413"></a><a name="zh-cn_topic_0000001511339104_p193951754413"></a>数据类型：int</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001511339104_row10475143684319"><td class="cellrowborder" valign="top" width="11.91%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001511339104_p162911910113611"><a name="zh-cn_topic_0000001511339104_p162911910113611"></a><a name="zh-cn_topic_0000001511339104_p162911910113611"></a>out_features</p>
</td>
<td class="cellrowborder" valign="top" width="8.25%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001511339104_p19831321362"><a name="zh-cn_topic_0000001511339104_p19831321362"></a><a name="zh-cn_topic_0000001511339104_p19831321362"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.84%" headers="mcps1.2.4.1.3 "><p id="p38618138313"><a name="p38618138313"></a><a name="p38618138313"></a>含义：输出特征数。</p>
<p id="zh-cn_topic_0000001511339104_p4403105711149"><a name="zh-cn_topic_0000001511339104_p4403105711149"></a><a name="zh-cn_topic_0000001511339104_p4403105711149"></a>数据类型：int</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001511339104_row7355184753719"><td class="cellrowborder" valign="top" width="11.91%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001511339104_p758902643119"><a name="zh-cn_topic_0000001511339104_p758902643119"></a><a name="zh-cn_topic_0000001511339104_p758902643119"></a>bias</p>
</td>
<td class="cellrowborder" valign="top" width="8.25%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001511339104_p9355247133712"><a name="zh-cn_topic_0000001511339104_p9355247133712"></a><a name="zh-cn_topic_0000001511339104_p9355247133712"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.84%" headers="mcps1.2.4.1.3 "><p id="p374431414316"><a name="p374431414316"></a><a name="p374431414316"></a>含义：是否开启偏置项参与学习。</p>
<p id="zh-cn_topic_0000001511339104_p173551747113712"><a name="zh-cn_topic_0000001511339104_p173551747113712"></a><a name="zh-cn_topic_0000001511339104_p173551747113712"></a>数据类型：bool，其他数据类型（比如整数，字符串，列表等）按照Python真值判断规则转换。</p>
<p id="zh-cn_topic_0000001511339104_p1635281111813"><a name="zh-cn_topic_0000001511339104_p1635281111813"></a><a name="zh-cn_topic_0000001511339104_p1635281111813"></a>默认值为True</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001511339104_row13521134203811"><td class="cellrowborder" valign="top" width="11.91%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001511339104_p1058982614312"><a name="zh-cn_topic_0000001511339104_p1058982614312"></a><a name="zh-cn_topic_0000001511339104_p1058982614312"></a>device</p>
</td>
<td class="cellrowborder" valign="top" width="8.25%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001511339104_p65217483815"><a name="zh-cn_topic_0000001511339104_p65217483815"></a><a name="zh-cn_topic_0000001511339104_p65217483815"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.84%" headers="mcps1.2.4.1.3 "><p id="p103939161312"><a name="p103939161312"></a><a name="p103939161312"></a>含义：运行设备。</p>
<p id="zh-cn_topic_0000001511339104_p752118411383"><a name="zh-cn_topic_0000001511339104_p752118411383"></a><a name="zh-cn_topic_0000001511339104_p752118411383"></a>默认值：None</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001511339104_row19812101220389"><td class="cellrowborder" valign="top" width="11.91%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001511339104_p195894262310"><a name="zh-cn_topic_0000001511339104_p195894262310"></a><a name="zh-cn_topic_0000001511339104_p195894262310"></a>dtype</p>
</td>
<td class="cellrowborder" valign="top" width="8.25%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001511339104_p14812412133816"><a name="zh-cn_topic_0000001511339104_p14812412133816"></a><a name="zh-cn_topic_0000001511339104_p14812412133816"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.84%" headers="mcps1.2.4.1.3 "><p id="p088915177312"><a name="p088915177312"></a><a name="p088915177312"></a>含义：torch数值类型。</p>
<p id="zh-cn_topic_0000001511339104_p281221212389"><a name="zh-cn_topic_0000001511339104_p281221212389"></a><a name="zh-cn_topic_0000001511339104_p281221212389"></a>数据类型：torch数据类型，仅支持torch.float32</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001511339104_row1292921916384"><td class="cellrowborder" valign="top" width="11.91%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001511339104_p19291619133812"><a name="zh-cn_topic_0000001511339104_p19291619133812"></a><a name="zh-cn_topic_0000001511339104_p19291619133812"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="8.25%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001511339104_p1892951920388"><a name="zh-cn_topic_0000001511339104_p1892951920388"></a><a name="zh-cn_topic_0000001511339104_p1892951920388"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.84%" headers="mcps1.2.4.1.3 "><p id="p8612203016312"><a name="p8612203016312"></a><a name="p8612203016312"></a>含义：量化配置，配置参考样例如下，量化配置参数的具体说明请参见<a href="../context/单算子模式量化配置参数.md">量化配置参数说明</a>。</p>
<pre class="screen" id="zh-cn_topic_0000001511339104_screen168684451264"><a name="zh-cn_topic_0000001511339104_screen168684451264"></a><a name="zh-cn_topic_0000001511339104_screen168684451264"></a>config = {
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
<p id="zh-cn_topic_0000001511339104_p3275175644316"><a name="zh-cn_topic_0000001511339104_p3275175644316"></a><a name="zh-cn_topic_0000001511339104_p3275175644316"></a>数据类型：dict</p>
<p id="zh-cn_topic_0000001511339104_p260612831619"><a name="zh-cn_topic_0000001511339104_p260612831619"></a><a name="zh-cn_topic_0000001511339104_p260612831619"></a>默认值：None</p>
</td>
</tr>
</tbody>
</table>

**表 2**  基于原生算子构造接口

<a name="table18947259560"></a>
<table><thead align="left"><tr id="row1589410251565"><th class="cellrowborder" valign="top" width="11.88%" id="mcps1.2.4.1.1"><p id="p1089415253564"><a name="p1089415253564"></a><a name="p1089415253564"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="8.21%" id="mcps1.2.4.1.2"><p id="p18894425155619"><a name="p18894425155619"></a><a name="p18894425155619"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="79.91%" id="mcps1.2.4.1.3"><p id="p188944258569"><a name="p188944258569"></a><a name="p188944258569"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row19894112555615"><td class="cellrowborder" valign="top" width="11.88%" headers="mcps1.2.4.1.1 "><p id="p2203115517562"><a name="p2203115517562"></a><a name="p2203115517562"></a>mod</p>
</td>
<td class="cellrowborder" valign="top" width="8.21%" headers="mcps1.2.4.1.2 "><p id="p4894142514561"><a name="p4894142514561"></a><a name="p4894142514561"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.91%" headers="mcps1.2.4.1.3 "><p id="p1159863515338"><a name="p1159863515338"></a><a name="p1159863515338"></a>含义：待量化的原生Linear算子。</p>
<p id="p118941025175610"><a name="p118941025175610"></a><a name="p118941025175610"></a>数据类型：torch.nn.Module</p>
</td>
</tr>
<tr id="row178961825155618"><td class="cellrowborder" valign="top" width="11.88%" headers="mcps1.2.4.1.1 "><p id="p128526535334"><a name="p128526535334"></a><a name="p128526535334"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="8.21%" headers="mcps1.2.4.1.2 "><p id="p385255363313"><a name="p385255363313"></a><a name="p385255363313"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="79.91%" headers="mcps1.2.4.1.3 "><p id="p7788541163313"><a name="p7788541163313"></a><a name="p7788541163313"></a>含义：量化配置。配置参考样例如下，量化配置参数的具体说明请参见<a href="../context/单算子模式量化配置参数.md">量化配置参数说明</a>。</p>
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
<p id="p1687534110168"><a name="p1687534110168"></a><a name="p1687534110168"></a>默认值：None</p>
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
    from amct_pytorch.nn.module.quantization.linear import LinearQAT
    
    LinearQAT(in_features=1, out_features=1, bias=True,
              device=None, dtype=None, config=None)
    ```

-   基于原生算子构造：

    ```python
    import torch
    
    from amct_pytorch.nn.module.quantization.linear import LinearQAT
    
    linear_op = torch.nn.Linear(in_features=1, out_features=1, bias=True, device=None, dtype=None)
    LinearQAT.from_float(mod=linear_op, config=None)
    ```

