# LinearQAT<a name="ZH-CN_TOPIC_0000002517028804"></a>

<!-- md-trans-meta sourceCommit=4a47814eed18048e444f9a2acc42d9864ddc477f translatedAt=2026-10-16T08:27:07.731Z pushedAt=2026-11-05T01:12:03.416Z nl=2 -->

## Supported Products<a name="section185612964420"></a>

<a name="zh-cn_topic_0000002517188794_table38301303189"></a>

| Product | Supported |
| ------------------------------------------- | -------- |
| Ascend 950PR&950DT products | √ |
| Atlas A3 products | √ |
| Atlas A2 products | √ |

## Description<a name="zh-cn_topic_0240187365_section15406195619561"></a>

Constructs a QAT operator for Linear.

## Prototype<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_section428121323411"></a>

- Direct construction API:

    ```python
    qat = amct_pytorch.nn.module.quantization.linear.LinearQAT(in_features, out_features, bias, device, dtype, config)
    ```

- Native-based operator construction API:

    ```python
    qat = amct_pytorch.nn.module.quantization.linear.LinearQAT.from_float(mod, config)
    ```

## Parameters<a name="section73811524135618"></a>

**Table 1**  Direct construction API parameters

<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001511339104_zh-cn_topic_0240187365_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="11.91%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001511339104_zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0000001511339104_zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0000001511339104_zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a>Parameter</p></th>
<th class="cellrowborder" valign="top" width="8.25%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001511339104_zh-cn_topic_0240187365_p1769255516412"><a name="zh-cn_topic_0000001511339104_zh-cn_topic_0240187365_p1769255516412"></a><a name="zh-cn_topic_0000001511339104_zh-cn_topic_0240187365_p1769255516412"></a>Input/Output</p></th>
<th class="cellrowborder" valign="top" width="79.84%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001511339104_zh-cn_topic_0240187365_p15231205416325"><a name="zh-cn_topic_0000001511339104_zh-cn_topic_0240187365_p15231205416325"></a><a name="zh-cn_topic_0000001511339104_zh-cn_topic_0240187365_p15231205416325"></a>Description</p></th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001511339104_zh-cn_topic_0240187365_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="11.91%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001511339104_p558952613111"><a name="zh-cn_topic_0000001511339104_p558952613111"></a><a name="zh-cn_topic_0000001511339104_p558952613111"></a>in_features</p></td>
<td class="cellrowborder" valign="top" width="8.25%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001511339104_p19832329363"><a name="zh-cn_topic_0000001511339104_p19832329363"></a><a name="zh-cn_topic_0000001511339104_p19832329363"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="79.84%" headers="mcps1.2.4.1.3 "><p id="p17841574317"><a name="p17841574317"></a><a name="p17841574317"></a>Meaning: Number of input features.</p>
<p id="zh-cn_topic_0000001511339104_p193951754413"><a name="zh-cn_topic_0000001511339104_p193951754413"></a><a name="zh-cn_topic_0000001511339104_p193951754413"></a>Data type: int.</p></td>
</tr>
<tr id="zh-cn_topic_0000001511339104_row10475143684319"><td class="cellrowborder" valign="top" width="11.91%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001511339104_p162911910113611"><a name="zh-cn_topic_0000001511339104_p162911910113611"></a><a name="zh-cn_topic_0000001511339104_p162911910113611"></a>out_features</p></td>
<td class="cellrowborder" valign="top" width="8.25%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001511339104_p19831321362"><a name="zh-cn_topic_0000001511339104_p19831321362"></a><a name="zh-cn_topic_0000001511339104_p19831321362"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="79.84%" headers="mcps1.2.4.1.3 "><p id="p38618138313"><a name="p38618138313"></a><a name="p38618138313"></a>Meaning: Number of output features.</p>
<p id="zh-cn_topic_0000001511339104_p4403105711149"><a name="zh-cn_topic_0000001511339104_p4403105711149"></a><a name="zh-cn_topic_0000001511339104_p4403105711149"></a>Data Type: int</p></td>
</tr>
<tr id="zh-cn_topic_0000001511339104_row7355184753719"><td class="cellrowborder" valign="top" width="11.91%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001511339104_p758902643119"><a name="zh-cn_topic_0000001511339104_p758902643119"></a><a name="zh-cn_topic_0000001511339104_p758902643119"></a>bias</p></td>
<td class="cellrowborder" valign="top" width="8.25%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001511339104_p9355247133712"><a name="zh-cn_topic_0000001511339104_p9355247133712"></a><a name="zh-cn_topic_0000001511339104_p9355247133712"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="79.84%" headers="mcps1.2.4.1.3 "><p id="p374431414316"><a name="p374431414316"></a><a name="p374431414316"></a>Meaning: Whether to enable the bias term to participate in learning.</p>
<p id="zh-cn_topic_0000001511339104_p173551747113712"><a name="zh-cn_topic_0000001511339104_p173551747113712"></a><a name="zh-cn_topic_0000001511339104_p173551747113712"></a>Data Type: bool. Other data types (such as integers, strings, and lists) are converted according to Python truth value rules.</p>
<p id="zh-cn_topic_0000001511339104_p1635281111813"><a name="zh-cn_topic_0000001511339104_p1635281111813"></a><a name="zh-cn_topic_0000001511339104_p1635281111813"></a>Default Value: True</p></td>
</tr>
<tr id="zh-cn_topic_0000001511339104_row13521134203811"><td class="cellrowborder" valign="top" width="11.91%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001511339104_p1058982614312"><a name="zh-cn_topic_0000001511339104_p1058982614312"></a><a name="zh-cn_topic_0000001511339104_p1058982614312"></a>device</p></td>
<td class="cellrowborder" valign="top" width="8.25%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001511339104_p65217483815"><a name="zh-cn_topic_0000001511339104_p65217483815"></a><a name="zh-cn_topic_0000001511339104_p65217483815"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="79.84%" headers="mcps1.2.4.1.3 "><p id="p103939161312"><a name="p103939161312"></a><a name="p103939161312"></a>Meaning: Device on which the operation runs.</p>
<p id="zh-cn_topic_0000001511339104_p752118411383"><a name="zh-cn_topic_0000001511339104_p752118411383"></a><a name="zh-cn_topic_0000001511339104_p752118411383"></a>Default Value: None</p></td>
</tr>
<tr id="zh-cn_topic_0000001511339104_row19812101220389"><td class="cellrowborder" valign="top" width="11.91%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001511339104_p195894262310"><a name="zh-cn_topic_0000001511339104_p195894262310"></a><a name="zh-cn_topic_0000001511339104_p195894262310"></a>dtype</p></td>
<td class="cellrowborder" valign="top" width="8.25%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001511339104_p14812412133816"><a name="zh-cn_topic_0000001511339104_p14812412133816"></a><a name="zh-cn_topic_0000001511339104_p14812412133816"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="79.84%" headers="mcps1.2.4.1.3 "><p id="p088915177312"><a name="p088915177312"></a><a name="p088915177312"></a>Meaning: torch numeric type.</p>
<p id="zh-cn_topic_0000001511339104_p281221212389"><a name="zh-cn_topic_0000001511339104_p281221212389"></a><a name="zh-cn_topic_0000001511339104_p281221212389"></a>Data Type: torch data type. Only torch.float32 is supported.</p></td>
</tr>
<tr id="zh-cn_topic_0000001511339104_row1292921916384"><td class="cellrowborder" valign="top" width="11.91%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001511339104_p19291619133812"><a name="zh-cn_topic_0000001511339104_p19291619133812"></a><a name="zh-cn_topic_0000001511339104_p19291619133812"></a>config</p></td>
<td class="cellrowborder" valign="top" width="8.25%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001511339104_p1892951920388"><a name="zh-cn_topic_0000001511339104_p1892951920388"></a><a name="zh-cn_topic_0000001511339104_p1892951920388"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="79.84%" headers="mcps1.2.4.1.3 "><p id="p8612203016312"><a name="p8612203016312"></a><a name="p8612203016312"></a>Meaning: Quantization configuration. A sample configuration is as follows. For details about the quantization configuration parameters, see <a href="../context/singleop_config.md">Quantization Configuration Parameter Description</a>.</p>
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
<p id="zh-cn_topic_0000001511339104_p3275175644316"><a name="zh-cn_topic_0000001511339104_p3275175644316"></a><a name="zh-cn_topic_0000001511339104_p3275175644316"></a>Data Type: dict</p>
<p id="zh-cn_topic_0000001511339104_p260612831619"><a name="zh-cn_topic_0000001511339104_p260612831619"></a><a name="zh-cn_topic_0000001511339104_p260612831619"></a>Default Value: None</p></td>
</tr>
</tbody>
</table>

**Table 2** Native-based operator construction API

<a name="table18947259560"></a>
<table><thead align="left"><tr id="row1589410251565"><th class="cellrowborder" valign="top" width="11.88%" id="mcps1.2.4.1.1"><p id="p1089415253564"><a name="p1089415253564"></a><a name="p1089415253564"></a>Parameter</p></th>
<th class="cellrowborder" valign="top" width="8.21%" id="mcps1.2.4.1.2"><p id="p18894425155619"><a name="p18894425155619"></a><a name="p18894425155619"></a>Input/Output</p></th>
<th class="cellrowborder" valign="top" width="79.91%" id="mcps1.2.4.1.3"><p id="p188944258569"><a name="p188944258569"></a><a name="p188944258569"></a>Description</p></th>
</tr>
</thead>
<tbody><tr id="row19894112555615"><td class="cellrowborder" valign="top" width="11.88%" headers="mcps1.2.4.1.1 "><p id="p2203115517562"><a name="p2203115517562"></a><a name="p2203115517562"></a>mod</p></td>
<td class="cellrowborder" valign="top" width="8.21%" headers="mcps1.2.4.1.2 "><p id="p4894142514561"><a name="p4894142514561"></a><a name="p4894142514561"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="79.91%" headers="mcps1.2.4.1.3 "><p id="p1159863515338"><a name="p1159863515338"></a><a name="p1159863515338"></a>Meaning: Native Linear operator to be quantized.</p>
<p id="p118941025175610"><a name="p118941025175610"></a><a name="p118941025175610"></a>Data Type: torch.nn.Module</p></td>
</tr>
<tr id="row178961825155618"><td class="cellrowborder" valign="top" width="11.88%" headers="mcps1.2.4.1.1 "><p id="p128526535334"><a name="p128526535334"></a><a name="p128526535334"></a>config</p></td>
<td class="cellrowborder" valign="top" width="8.21%" headers="mcps1.2.4.1.2 "><p id="p385255363313"><a name="p385255363313"></a><a name="p385255363313"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="79.91%" headers="mcps1.2.4.1.3 "><p id="p7788541163313"><a name="p7788541163313"></a><a name="p7788541163313"></a>Meaning: Quantization configuration. A sample configuration is as follows. For details about the quantization configuration parameters, see <a href="../context/singleop_config.md">Quantization Configuration Parameter Description</a>.</p>
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
<p id="p8852653133318"><a name="p8852653133318"></a><a name="p8852653133318"></a>Data Type: dict</p>
<p id="p1687534110168"><a name="p1687534110168"></a><a name="p1687534110168"></a>Default Value: None</p></td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section293415513458"></a>

- Direct construction: Returns the constructed QAT single-operator instance.
- Native-based operator construction: QAT single operator converted from **torch.nn.Module**.

## Example<a name="zh-cn_topic_0240188739_section64231658994"></a>

- Direct construction:

    ```python
    from amct_pytorch.nn.module.quantization.linear import LinearQAT
    
    LinearQAT(in_features=1, out_features=1, bias=True,
              device=None, dtype=None, config=None)
    ```

- Native-based operator construction:

    ```python
    import torch
    
    from amct_pytorch.nn.module.quantization.linear import LinearQAT
    
    linear_op = torch.nn.Linear(in_features=1, out_features=1, bias=True, device=None, dtype=None)
    LinearQAT.from_float(mod=linear_op, config=None)
    ```
