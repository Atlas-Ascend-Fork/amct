# ConvTranspose2dQAT<a name="ZH-CN_TOPIC_0000002548788613"></a>

<!-- md-trans-meta sourceCommit=4a47814eed18048e444f9a2acc42d9864ddc477f translatedAt=2026-10-16T08:13:59.815Z pushedAt=2026-11-05T01:12:03.407Z nl=2 -->

## Supported Products<a name="section185612964420"></a>

<a name="zh-cn_topic_0000002517188794_table38301303189"></a>

| Product | Supported |
| ------------------------------------------- | -------- |
| Ascend 950PR&950DT products | √ |
| Atlas A3 products | √ |
| Atlas A2 products | √ |

## Description<a name="zh-cn_topic_0240187365_section15406195619561"></a>

Constructs the QAT operator for ConvTranspose2d.

## Prototype<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_section428121323411"></a>

- Direct construction API:

    ```python
    qat = amct_pytorch.nn.module.quantization.conv_transpose_2d.ConvTranspose2dQAT(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype, config)
    ```

- Native-based operator construction API:

    ```python
    qat = amct_pytorch.nn.module.quantization.conv_transpose_2d.ConvTranspose2dQAT.from_float(mod, config)
    ```

## Parameters<a name="section73811524135618"></a>

**Table 1** Parameters of the direct construction API

<a name="table1754714963812"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002548788615_zh-cn_topic_0240187365_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="11.600000000000001%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000002548788615_zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0000002548788615_zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0000002548788615_zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a>Parameter</p></th>
<th class="cellrowborder" valign="top" width="7.82%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000002548788615_zh-cn_topic_0240187365_p1769255516412"><a name="zh-cn_topic_0000002548788615_zh-cn_topic_0240187365_p1769255516412"></a><a name="zh-cn_topic_0000002548788615_zh-cn_topic_0240187365_p1769255516412"></a>Input/Output</p></th>
<th class="cellrowborder" valign="top" width="80.58%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000002548788615_zh-cn_topic_0240187365_p15231205416325"><a name="zh-cn_topic_0000002548788615_zh-cn_topic_0240187365_p15231205416325"></a><a name="zh-cn_topic_0000002548788615_zh-cn_topic_0240187365_p15231205416325"></a>Description</p></th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002548788615_zh-cn_topic_0240187365_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="11.600000000000001%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002548788615_p558952613111"><a name="zh-cn_topic_0000002548788615_p558952613111"></a><a name="zh-cn_topic_0000002548788615_p558952613111"></a>in_channels</p></td>
<td class="cellrowborder" valign="top" width="7.82%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002548788615_p19832329363"><a name="zh-cn_topic_0000002548788615_p19832329363"></a><a name="zh-cn_topic_0000002548788615_p19832329363"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.58%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002548788615_p66038591458"><a name="zh-cn_topic_0000002548788615_p66038591458"></a><a name="zh-cn_topic_0000002548788615_p66038591458"></a>Meaning: Number of input channels.</p>
<p id="zh-cn_topic_0000002548788615_p193951754413"><a name="zh-cn_topic_0000002548788615_p193951754413"></a><a name="zh-cn_topic_0000002548788615_p193951754413"></a>Data Type: int.</p></td>
</tr>
<tr id="zh-cn_topic_0000002548788615_row10475143684319"><td class="cellrowborder" valign="top" width="11.600000000000001%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002548788615_p162911910113611"><a name="zh-cn_topic_0000002548788615_p162911910113611"></a><a name="zh-cn_topic_0000002548788615_p162911910113611"></a>out_channels</p></td>
<td class="cellrowborder" valign="top" width="7.82%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002548788615_p19831321362"><a name="zh-cn_topic_0000002548788615_p19831321362"></a><a name="zh-cn_topic_0000002548788615_p19831321362"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.58%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002548788615_p155218704612"><a name="zh-cn_topic_0000002548788615_p155218704612"></a><a name="zh-cn_topic_0000002548788615_p155218704612"></a>Meaning: Number of output channels.</p>
<p id="zh-cn_topic_0000002548788615_p4403105711149"><a name="zh-cn_topic_0000002548788615_p4403105711149"></a><a name="zh-cn_topic_0000002548788615_p4403105711149"></a>Data Type: int</p></td>
</tr>
<tr id="zh-cn_topic_0000002548788615_row11737244204314"><td class="cellrowborder" valign="top" width="11.600000000000001%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002548788615_p258918267312"><a name="zh-cn_topic_0000002548788615_p258918267312"></a><a name="zh-cn_topic_0000002548788615_p258918267312"></a>kernel_size</p></td>
<td class="cellrowborder" valign="top" width="7.82%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002548788615_p1598233212367"><a name="zh-cn_topic_0000002548788615_p1598233212367"></a><a name="zh-cn_topic_0000002548788615_p1598233212367"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.58%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002548788615_p1515559144620"><a name="zh-cn_topic_0000002548788615_p1515559144620"></a><a name="zh-cn_topic_0000002548788615_p1515559144620"></a>Meaning: Convolution kernel size.</p>
<p id="zh-cn_topic_0000002548788615_p1239175124112"><a name="zh-cn_topic_0000002548788615_p1239175124112"></a><a name="zh-cn_topic_0000002548788615_p1239175124112"></a>Data Type: int/tuple</p></td>
</tr>
<tr id="zh-cn_topic_0000002548788615_row41919427437"><td class="cellrowborder" valign="top" width="11.600000000000001%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002548788615_p18589162616312"><a name="zh-cn_topic_0000002548788615_p18589162616312"></a><a name="zh-cn_topic_0000002548788615_p18589162616312"></a>stride</p></td>
<td class="cellrowborder" valign="top" width="7.82%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002548788615_p098023253618"><a name="zh-cn_topic_0000002548788615_p098023253618"></a><a name="zh-cn_topic_0000002548788615_p098023253618"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.58%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002548788615_p1251221124619"><a name="zh-cn_topic_0000002548788615_p1251221124619"></a><a name="zh-cn_topic_0000002548788615_p1251221124619"></a>Meaning: Convolution stride.</p>
<p id="zh-cn_topic_0000002548788615_p227516563436"><a name="zh-cn_topic_0000002548788615_p227516563436"></a><a name="zh-cn_topic_0000002548788615_p227516563436"></a>Data Type: int/tuple</p>
<p id="zh-cn_topic_0000002548788615_p20418832171519"><a name="zh-cn_topic_0000002548788615_p20418832171519"></a><a name="zh-cn_topic_0000002548788615_p20418832171519"></a>Default Value: 1</p></td>
</tr>
<tr id="zh-cn_topic_0000002548788615_row1746261023713"><td class="cellrowborder" valign="top" width="11.600000000000001%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002548788615_p17589926163112"><a name="zh-cn_topic_0000002548788615_p17589926163112"></a><a name="zh-cn_topic_0000002548788615_p17589926163112"></a>padding</p></td>
<td class="cellrowborder" valign="top" width="7.82%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002548788615_p1146341043712"><a name="zh-cn_topic_0000002548788615_p1146341043712"></a><a name="zh-cn_topic_0000002548788615_p1146341043712"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.58%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002548788615_p12137171316463"><a name="zh-cn_topic_0000002548788615_p12137171316463"></a><a name="zh-cn_topic_0000002548788615_p12137171316463"></a>Meaning: Padding size.</p>
<p id="zh-cn_topic_0000002548788615_p15356205881619"><a name="zh-cn_topic_0000002548788615_p15356205881619"></a><a name="zh-cn_topic_0000002548788615_p15356205881619"></a>Data Type: int/tuple</p>
<p id="zh-cn_topic_0000002548788615_p6356105818169"><a name="zh-cn_topic_0000002548788615_p6356105818169"></a><a name="zh-cn_topic_0000002548788615_p6356105818169"></a>Default Value: 0</p></td>
</tr>
<tr id="zh-cn_topic_0000002548788615_row13859101213717"><td class="cellrowborder" valign="top" width="11.600000000000001%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002548788615_p19589152613111"><a name="zh-cn_topic_0000002548788615_p19589152613111"></a><a name="zh-cn_topic_0000002548788615_p19589152613111"></a>dilation</p></td>
<td class="cellrowborder" valign="top" width="7.82%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002548788615_p4859151215377"><a name="zh-cn_topic_0000002548788615_p4859151215377"></a><a name="zh-cn_topic_0000002548788615_p4859151215377"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.58%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002548788615_p1655091434616"><a name="zh-cn_topic_0000002548788615_p1655091434616"></a><a name="zh-cn_topic_0000002548788615_p1655091434616"></a>Meaning: Spacing between kernel elements.</p>
<p id="zh-cn_topic_0000002548788615_p114891831716"><a name="zh-cn_topic_0000002548788615_p114891831716"></a><a name="zh-cn_topic_0000002548788615_p114891831716"></a>Data Type: int/tuple</p>
<p id="zh-cn_topic_0000002548788615_p8859171273710"><a name="zh-cn_topic_0000002548788615_p8859171273710"></a><a name="zh-cn_topic_0000002548788615_p8859171273710"></a>Default Value: 1</p></td>
</tr>
<tr id="zh-cn_topic_0000002548788615_row1444111152375"><td class="cellrowborder" valign="top" width="11.600000000000001%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002548788615_p658962613112"><a name="zh-cn_topic_0000002548788615_p658962613112"></a><a name="zh-cn_topic_0000002548788615_p658962613112"></a>groups</p></td>
<td class="cellrowborder" valign="top" width="7.82%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002548788615_p5441151512370"><a name="zh-cn_topic_0000002548788615_p5441151512370"></a><a name="zh-cn_topic_0000002548788615_p5441151512370"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.58%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002548788615_p99806173469"><a name="zh-cn_topic_0000002548788615_p99806173469"></a><a name="zh-cn_topic_0000002548788615_p99806173469"></a>Meaning: Connection relationship between input and output.</p>
<p id="zh-cn_topic_0000002548788615_p444121512372"><a name="zh-cn_topic_0000002548788615_p444121512372"></a><a name="zh-cn_topic_0000002548788615_p444121512372"></a>Data Type: int</p>
<p id="zh-cn_topic_0000002548788615_p205195115174"><a name="zh-cn_topic_0000002548788615_p205195115174"></a><a name="zh-cn_topic_0000002548788615_p205195115174"></a>Default Value: 1</p></td>
</tr>
<tr id="zh-cn_topic_0000002548788615_row7355184753719"><td class="cellrowborder" valign="top" width="11.600000000000001%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002548788615_p758902643119"><a name="zh-cn_topic_0000002548788615_p758902643119"></a><a name="zh-cn_topic_0000002548788615_p758902643119"></a>bias</p></td>
<td class="cellrowborder" valign="top" width="7.82%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002548788615_p9355247133712"><a name="zh-cn_topic_0000002548788615_p9355247133712"></a><a name="zh-cn_topic_0000002548788615_p9355247133712"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.58%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002548788615_p178722119462"><a name="zh-cn_topic_0000002548788615_p178722119462"></a><a name="zh-cn_topic_0000002548788615_p178722119462"></a>Meaning: Whether to enable the bias term to participate in learning.</p>
<p id="zh-cn_topic_0000002548788615_p173551747113712"><a name="zh-cn_topic_0000002548788615_p173551747113712"></a><a name="zh-cn_topic_0000002548788615_p173551747113712"></a>Data Type: bool. Other data types (such as integers, strings, and lists) are converted according to Python truth value rules.</p>
<p id="zh-cn_topic_0000002548788615_p1635281111813"><a name="zh-cn_topic_0000002548788615_p1635281111813"></a><a name="zh-cn_topic_0000002548788615_p1635281111813"></a>Default Value: True</p></td>
</tr>
<tr id="zh-cn_topic_0000002548788615_row1381165643710"><td class="cellrowborder" valign="top" width="11.600000000000001%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002548788615_p1158910260319"><a name="zh-cn_topic_0000002548788615_p1158910260319"></a><a name="zh-cn_topic_0000002548788615_p1158910260319"></a>padding_mode</p></td>
<td class="cellrowborder" valign="top" width="7.82%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002548788615_p18382115633714"><a name="zh-cn_topic_0000002548788615_p18382115633714"></a><a name="zh-cn_topic_0000002548788615_p18382115633714"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.58%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002548788615_p84961522124618"><a name="zh-cn_topic_0000002548788615_p84961522124618"></a><a name="zh-cn_topic_0000002548788615_p84961522124618"></a>Meaning: Padding mode.</p>
<p id="zh-cn_topic_0000002548788615_p10382135617375"><a name="zh-cn_topic_0000002548788615_p10382135617375"></a><a name="zh-cn_topic_0000002548788615_p10382135617375"></a>Constraint: Only zeros is supported.</p></td>
</tr>
<tr id="zh-cn_topic_0000002548788615_row13521134203811"><td class="cellrowborder" valign="top" width="11.600000000000001%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002548788615_p1058982614312"><a name="zh-cn_topic_0000002548788615_p1058982614312"></a><a name="zh-cn_topic_0000002548788615_p1058982614312"></a>device</p></td>
<td class="cellrowborder" valign="top" width="7.82%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002548788615_p65217483815"><a name="zh-cn_topic_0000002548788615_p65217483815"></a><a name="zh-cn_topic_0000002548788615_p65217483815"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.58%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002548788615_p99524154613"><a name="zh-cn_topic_0000002548788615_p99524154613"></a><a name="zh-cn_topic_0000002548788615_p99524154613"></a>Meaning: Running device.</p>
<p id="zh-cn_topic_0000002548788615_p752118411383"><a name="zh-cn_topic_0000002548788615_p752118411383"></a><a name="zh-cn_topic_0000002548788615_p752118411383"></a>Default value: None</p></td>
</tr>
<tr id="zh-cn_topic_0000002548788615_row19812101220389"><td class="cellrowborder" valign="top" width="11.600000000000001%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002548788615_p195894262310"><a name="zh-cn_topic_0000002548788615_p195894262310"></a><a name="zh-cn_topic_0000002548788615_p195894262310"></a>dtype</p></td>
<td class="cellrowborder" valign="top" width="7.82%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002548788615_p14812412133816"><a name="zh-cn_topic_0000002548788615_p14812412133816"></a><a name="zh-cn_topic_0000002548788615_p14812412133816"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.58%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002548788615_p16911825144614"><a name="zh-cn_topic_0000002548788615_p16911825144614"></a><a name="zh-cn_topic_0000002548788615_p16911825144614"></a>Meaning: torch numeric type.</p>
<p id="zh-cn_topic_0000002548788615_p281221212389"><a name="zh-cn_topic_0000002548788615_p281221212389"></a><a name="zh-cn_topic_0000002548788615_p281221212389"></a>torch data type. Only torch.float32 is supported.</p></td>
</tr>
<tr id="zh-cn_topic_0000002548788615_row1292921916384"><td class="cellrowborder" valign="top" width="11.600000000000001%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000002548788615_p19291619133812"><a name="zh-cn_topic_0000002548788615_p19291619133812"></a><a name="zh-cn_topic_0000002548788615_p19291619133812"></a>config</p></td>
<td class="cellrowborder" valign="top" width="7.82%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000002548788615_p1892951920388"><a name="zh-cn_topic_0000002548788615_p1892951920388"></a><a name="zh-cn_topic_0000002548788615_p1892951920388"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.58%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002548788615_p9236112719464"><a name="zh-cn_topic_0000002548788615_p9236112719464"></a><a name="zh-cn_topic_0000002548788615_p9236112719464"></a>Meaning: Quantization configuration. A sample configuration is as follows. For details about the quantization configuration parameters, see <a href="../context/singleop_config.md">Quantization Configuration Parameter Description</a>.</p>
<pre class="screen" id="zh-cn_topic_0000002548788615_screen168684451264"><a name="zh-cn_topic_0000002548788615_screen168684451264"></a><a name="zh-cn_topic_0000002548788615_screen168684451264"></a>config = {
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
<p id="zh-cn_topic_0000002548788615_p3275175644316"><a name="zh-cn_topic_0000002548788615_p3275175644316"></a><a name="zh-cn_topic_0000002548788615_p3275175644316"></a>Data Type: dict</p>
<p id="zh-cn_topic_0000002548788615_p260612831619"><a name="zh-cn_topic_0000002548788615_p260612831619"></a><a name="zh-cn_topic_0000002548788615_p260612831619"></a>Default Value: None</p></td>
</tr>
</tbody>
</table>

**Table 2**  Native-based operator construction API

<a name="table18947259560"></a>
<table><thead align="left"><tr id="row1589410251565"><th class="cellrowborder" valign="top" width="10.56%" id="mcps1.2.4.1.1"><p id="p1089415253564"><a name="p1089415253564"></a><a name="p1089415253564"></a>Parameter</p></th>
<th class="cellrowborder" valign="top" width="9.67%" id="mcps1.2.4.1.2"><p id="p18894425155619"><a name="p18894425155619"></a><a name="p18894425155619"></a>Input/Output</p></th>
<th class="cellrowborder" valign="top" width="79.77%" id="mcps1.2.4.1.3"><p id="p188944258569"><a name="p188944258569"></a><a name="p188944258569"></a>Description</p></th>
</tr>
</thead>
<tbody><tr id="row19894112555615"><td class="cellrowborder" valign="top" width="10.56%" headers="mcps1.2.4.1.1 "><p id="p2203115517562"><a name="p2203115517562"></a><a name="p2203115517562"></a>mod</p></td>
<td class="cellrowborder" valign="top" width="9.67%" headers="mcps1.2.4.1.2 "><p id="p4894142514561"><a name="p4894142514561"></a><a name="p4894142514561"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="79.77%" headers="mcps1.2.4.1.3 "><p id="p175801624194913"><a name="p175801624194913"></a><a name="p175801624194913"></a>Meaning: The native ConvTranspose2d operator to be quantized.</p>
<p id="p118941025175610"><a name="p118941025175610"></a><a name="p118941025175610"></a>Data Type: torch.nn.Module</p></td>
</tr>
<tr id="row178961825155618"><td class="cellrowborder" valign="top" width="10.56%" headers="mcps1.2.4.1.1 "><p id="p128526535334"><a name="p128526535334"></a><a name="p128526535334"></a>config</p></td>
<td class="cellrowborder" valign="top" width="9.67%" headers="mcps1.2.4.1.2 "><p id="p385255363313"><a name="p385255363313"></a><a name="p385255363313"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="79.77%" headers="mcps1.2.4.1.3 "><p id="p1864122554918"><a name="p1864122554918"></a><a name="p1864122554918"></a>Meaning: Quantization configuration. A sample configuration is as follows. For details about the quantization configuration parameters, see <a href="../context/singleop_config.md">Quantization Configuration Parameter Description</a>.</p>
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
<p id="p456112119162"><a name="p456112119162"></a><a name="p456112119162"></a>Default Value: None</p></td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section293415513458"></a>

- Direct construction: Returns the constructed QAT single-operator instance.
- Native-based operator construction: QAT single operator converted from **torch.nn.Module**.

## Example<a name="zh-cn_topic_0240188739_section64231658994"></a>

- Direct construction:

    ```python
    from amct_pytorch.nn.module.quantization.conv_transpose_2d import ConvTranspose2dQAT

    ConvTranspose2dQAT(in_channels=1, out_channels=1, kernel_size=1, stride=1,
                       padding=0, dilation=1, groups=1, bias=True,
                       padding_mode='zeros', device=None, dtype=None, config=None)
    ```

- Native-based operator construction:

    ```python
    import torch

    from amct_pytorch.nn.module.quantization.conv_transpose_2d import ConvTranspose2dQAT

    conv_transpose2d_op = torch.nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
    ConvTranspose2dQAT.from_float(mod=conv_transpose2d_op, config=None)
    ```
