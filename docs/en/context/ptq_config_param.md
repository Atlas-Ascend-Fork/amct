# Post-Training Quantization Configuration Parameter Description

<!-- md-trans-meta sourceCommit=50092c1ff927c3d7fb95cc796b97264520f8a34e translatedAt=2026-10-16T08:46:07.688Z pushedAt=2026-11-05T01:12:03.435Z nl=2 -->

If the inference accuracy of the **config.json** quantization configuration file generated through the [create\_quant\_config](../api/create_quant_config.md) API does not meet the requirements, adjust the content of the **config.json** file by referring to this section (when modifying the JSON file, ensure that layer names are unique) until the accuracy meets the requirements. For a sample JSON quantization configuration file, see the [Example](../api/create_quant_config.md#zh-cn_topic_0240187365_section64231658994) section of the API.

The parameters in the configuration file are described as follows:

**Table 2**  version parameter description

<a name="zh-cn_topic_0240188722_table2626104713369"></a>
<table><tbody><tr id="zh-cn_topic_0240188722_row116261847163611"><th class="firstcol" valign="top" width="20.96%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0240188722_p462654715365"><a name="zh-cn_topic_0240188722_p462654715365"></a><a name="zh-cn_topic_0240188722_p462654715365"></a>Function</p></th>
<td class="cellrowborder" valign="top" width="79.03999999999999%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0240188722_p16269472361"><a name="zh-cn_topic_0240188722_p16269472361"></a><a name="zh-cn_topic_0240188722_p16269472361"></a>Controls the version number of the quantization configuration file.</p></td>
</tr>
<tr id="zh-cn_topic_0240188722_row362684710366"><th class="firstcol" valign="top" width="20.96%" id="mcps1.2.3.2.1"><p id="zh-cn_topic_0240188722_p762684713611"><a name="zh-cn_topic_0240188722_p762684713611"></a><a name="zh-cn_topic_0240188722_p762684713611"></a>Type</p></th>
<td class="cellrowborder" valign="top" width="79.03999999999999%" headers="mcps1.2.3.2.1 "><p id="zh-cn_topic_0240188722_p19626124713367"><a name="zh-cn_topic_0240188722_p19626124713367"></a><a name="zh-cn_topic_0240188722_p19626124713367"></a>int</p></td>
</tr>
<tr id="zh-cn_topic_0240188722_row146267479367"><th class="firstcol" valign="top" width="20.96%" id="mcps1.2.3.3.1"><p id="zh-cn_topic_0240188722_p18626194711368"><a name="zh-cn_topic_0240188722_p18626194711368"></a><a name="zh-cn_topic_0240188722_p18626194711368"></a>Value Range</p></th>
<td class="cellrowborder" valign="top" width="79.03999999999999%" headers="mcps1.2.3.3.1 "><p id="zh-cn_topic_0240188722_p186261147193612"><a name="zh-cn_topic_0240188722_p186261147193612"></a><a name="zh-cn_topic_0240188722_p186261147193612"></a>1</p></td>
</tr>
<tr id="zh-cn_topic_0240188722_row7626194719361"><th class="firstcol" valign="top" width="20.96%" id="mcps1.2.3.4.1"><p id="zh-cn_topic_0240188722_p1462644712368"><a name="zh-cn_topic_0240188722_p1462644712368"></a><a name="zh-cn_topic_0240188722_p1462644712368"></a>Parameter Description</p></th>
<td class="cellrowborder" valign="top" width="79.03999999999999%" headers="mcps1.2.3.4.1 "><p id="zh-cn_topic_0240188722_p156261247113612"><a name="zh-cn_topic_0240188722_p156261247113612"></a><a name="zh-cn_topic_0240188722_p156261247113612"></a>Currently there is only one version number, 1.</p></td>
</tr>
<tr id="zh-cn_topic_0240188722_row17626144713363"><th class="firstcol" valign="top" width="20.96%" id="mcps1.2.3.5.1"><p id="zh-cn_topic_0240188722_p1462634743614"><a name="zh-cn_topic_0240188722_p1462634743614"></a><a name="zh-cn_topic_0240188722_p1462634743614"></a>Recommended Configuration</p></th>
<td class="cellrowborder" valign="top" width="79.03999999999999%" headers="mcps1.2.3.5.1 "><p id="zh-cn_topic_0240188722_p1162664743620"><a name="zh-cn_topic_0240188722_p1162664743620"></a><a name="zh-cn_topic_0240188722_p1162664743620"></a>1</p></td>
</tr>
<tr id="zh-cn_topic_0240188722_row311394219392"><th class="firstcol" valign="top" width="20.96%" id="mcps1.2.3.6.1"><p id="zh-cn_topic_0240188722_p1811484213910"><a name="zh-cn_topic_0240188722_p1811484213910"></a><a name="zh-cn_topic_0240188722_p1811484213910"></a>Required</p></th>
<td class="cellrowborder" valign="top" width="79.03999999999999%" headers="mcps1.2.3.6.1 "><p id="zh-cn_topic_0240188722_p17114742143918"><a name="zh-cn_topic_0240188722_p17114742143918"></a><a name="zh-cn_topic_0240188722_p17114742143918"></a>No</p></td>
</tr>
</tbody>
</table>

**Table 3**  batch\_num parameter description

<a name="zh-cn_topic_0240188002_table1993174915423"></a>
<table><tbody><tr id="zh-cn_topic_0240188002_row0944492423"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0240188002_p294134994210"><a name="zh-cn_topic_0240188002_p294134994210"></a><a name="zh-cn_topic_0240188002_p294134994210"></a>Function</p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0240188002_p694164934219"><a name="zh-cn_topic_0240188002_p694164934219"></a><a name="zh-cn_topic_0240188002_p694164934219"></a>Controls how many batches of data are used for quantization.</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row294249104214"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.2.1"><p id="zh-cn_topic_0240188002_p1794164974216"><a name="zh-cn_topic_0240188002_p1794164974216"></a><a name="zh-cn_topic_0240188002_p1794164974216"></a><span>Type</span></p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.2.1 "><p id="zh-cn_topic_0240188002_p1894449164210"><a name="zh-cn_topic_0240188002_p1894449164210"></a><a name="zh-cn_topic_0240188002_p1894449164210"></a>int</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row199420495426"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.3.1"><p id="zh-cn_topic_0240188002_p39419491427"><a name="zh-cn_topic_0240188002_p39419491427"></a><a name="zh-cn_topic_0240188002_p39419491427"></a><span>Value Range</span></p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.3.1 "><p id="zh-cn_topic_0240188002_p694134954217"><a name="zh-cn_topic_0240188002_p694134954217"></a><a name="zh-cn_topic_0240188002_p694134954217"></a>Greater than 0.</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row394164944213"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.4.1"><p id="zh-cn_topic_0240188002_p1694104916421"><a name="zh-cn_topic_0240188002_p1694104916421"></a><a name="zh-cn_topic_0240188002_p1694104916421"></a>Parameter Description</p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.4.1 "><p id="zh-cn_topic_0240188002_p159444919429"><a name="zh-cn_topic_0240188002_p159444919429"></a><a name="zh-cn_topic_0240188002_p159444919429"></a>If this parameter is not configured, the default value 1 is used. It is recommended that the number of calibration set images not exceed 50. Calculate the corresponding batch_num value based on the batch size batch_size.</p>
<p id="zh-cn_topic_0240188002_p1223319101118"><a name="zh-cn_topic_0240188002_p1223319101118"></a><a name="zh-cn_topic_0240188002_p1223319101118"></a>batch_num * batch_size is the number of calibration set images used for quantization.</p>
<p id="zh-cn_topic_0240188002_p201320014422"><a name="zh-cn_topic_0240188002_p201320014422"></a><a name="zh-cn_topic_0240188002_p201320014422"></a>batch_size is the number of images used in each batch.</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row8941849114212"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.5.1"><p id="zh-cn_topic_0240188002_p19945496428"><a name="zh-cn_topic_0240188002_p19945496428"></a><a name="zh-cn_topic_0240188002_p19945496428"></a>Recommended Configuration</p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.5.1 "><p id="zh-cn_topic_0240188002_p10941249164219"><a name="zh-cn_topic_0240188002_p10941249164219"></a><a name="zh-cn_topic_0240188002_p10941249164219"></a>1</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row194104974210"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.6.1"><p id="zh-cn_topic_0240188002_p694174913424"><a name="zh-cn_topic_0240188002_p694174913424"></a><a name="zh-cn_topic_0240188002_p694174913424"></a>Required</p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.6.1 "><p id="zh-cn_topic_0240188002_p89524910428"><a name="zh-cn_topic_0240188002_p89524910428"></a><a name="zh-cn_topic_0240188002_p89524910428"></a>No</p></td>
</tr>
</tbody>
</table>

**Table 4**  activation\_offset parameter description

<a name="zh-cn_topic_0240188002_table9951049114210"></a>
<table><tbody><tr id="zh-cn_topic_0240188002_row129514490428"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0240188002_p3951149184213"><a name="zh-cn_topic_0240188002_p3951149184213"></a><a name="zh-cn_topic_0240188002_p3951149184213"></a>Function</p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.1.1 "><p id="p168261724123812"><a name="p168261724123812"></a><a name="p168261724123812"></a>Controls whether data quantization is symmetric or asymmetric. Global configuration parameter.</p>
<p id="p12429175873714"><a name="p12429175873714"></a><a name="p12429175873714"></a>If both activation_offset and asymmetric are present in the configuration file, the asymmetric parameter takes precedence over the activation_offset parameter.</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row18951449164219"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.2.1"><p id="zh-cn_topic_0240188002_p0951493420"><a name="zh-cn_topic_0240188002_p0951493420"></a><a name="zh-cn_topic_0240188002_p0951493420"></a>Type</p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.2.1 "><p id="zh-cn_topic_0240188002_p99534916427"><a name="zh-cn_topic_0240188002_p99534916427"></a><a name="zh-cn_topic_0240188002_p99534916427"></a>bool</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row189516493421"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.3.1"><p id="zh-cn_topic_0240188002_p1895649144219"><a name="zh-cn_topic_0240188002_p1895649144219"></a><a name="zh-cn_topic_0240188002_p1895649144219"></a>Value Range</p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.3.1 "><p id="zh-cn_topic_0240188002_p59574994218"><a name="zh-cn_topic_0240188002_p59574994218"></a><a name="zh-cn_topic_0240188002_p59574994218"></a>true or false</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row169584954213"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.4.1"><p id="zh-cn_topic_0240188002_p12959499422"><a name="zh-cn_topic_0240188002_p12959499422"></a><a name="zh-cn_topic_0240188002_p12959499422"></a>Parameter Description</p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.4.1 "><a name="ul11492132182611"></a><a name="ul11492132182611"></a><ul id="ul11492132182611"><li>true: Asymmetric quantization is used for data quantization.</li><li>false: Symmetric quantization is used for data quantization.</li></ul></td>
</tr>
<tr id="zh-cn_topic_0240188002_row139564944214"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.5.1"><p id="zh-cn_topic_0240188002_p59524916426"><a name="zh-cn_topic_0240188002_p59524916426"></a><a name="zh-cn_topic_0240188002_p59524916426"></a>Recommended Configuration</p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.5.1 "><p id="zh-cn_topic_0240188002_p169674919424"><a name="zh-cn_topic_0240188002_p169674919424"></a><a name="zh-cn_topic_0240188002_p169674919424"></a>true</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row496164924211"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.6.1"><p id="zh-cn_topic_0240188002_p139613497421"><a name="zh-cn_topic_0240188002_p139613497421"></a><a name="zh-cn_topic_0240188002_p139613497421"></a>Required</p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.6.1 "><p id="zh-cn_topic_0240188002_p1096154916421"><a name="zh-cn_topic_0240188002_p1096154916421"></a><a name="zh-cn_topic_0240188002_p1096154916421"></a>No</p></td>
</tr>
</tbody>
</table>

**Table 5**  do\_fusion parameter description

<a name="table9868105620512"></a>
<table><tbody><tr id="row108683561257"><th class="firstcol" valign="top" width="21.25%" id="mcps1.2.3.1.1"><p id="p208681156158"><a name="p208681156158"></a><a name="p208681156158"></a>Function</p></th>
<td class="cellrowborder" valign="top" width="78.75%" headers="mcps1.2.3.1.1 "><p id="p20412381073"><a name="p20412381073"></a><a name="p20412381073"></a>Whether to enable the fusion function.</p></td>
</tr>
<tr id="row48685563515"><th class="firstcol" valign="top" width="21.25%" id="mcps1.2.3.2.1"><p id="p10868256059"><a name="p10868256059"></a><a name="p10868256059"></a>Type</p></th>
<td class="cellrowborder" valign="top" width="78.75%" headers="mcps1.2.3.2.1 "><p id="p58681756659"><a name="p58681756659"></a><a name="p58681756659"></a>bool</p></td>
</tr>
<tr id="row7869356054"><th class="firstcol" valign="top" width="21.25%" id="mcps1.2.3.3.1"><p id="p1586985616513"><a name="p1586985616513"></a><a name="p1586985616513"></a>Value Range</p></th>
<td class="cellrowborder" valign="top" width="78.75%" headers="mcps1.2.3.3.1 "><p id="p138165258720"><a name="p138165258720"></a><a name="p138165258720"></a>true or false</p></td>
</tr>
<tr id="row14869556159"><th class="firstcol" valign="top" width="21.25%" id="mcps1.2.3.4.1"><p id="p586913561757"><a name="p586913561757"></a><a name="p586913561757"></a>Parameter Description</p></th>
<td class="cellrowborder" valign="top" width="78.75%" headers="mcps1.2.3.4.1 "><a name="ul14901289110"></a><a name="ul14901289110"></a><ul id="ul14901289110"><li>true: Enables the fusion function.</li><li>false: Disables the fusion function.</li></ul>
<p id="p1731815714168"><a name="p1731815714168"></a><a name="p1731815714168"></a>Currently, only Conv+BN fusion is supported.</p></td>
</tr>
<tr id="row68692566513"><th class="firstcol" valign="top" width="21.25%" id="mcps1.2.3.5.1"><p id="p16869956256"><a name="p16869956256"></a><a name="p16869956256"></a>Recommended Configuration</p></th>
<td class="cellrowborder" valign="top" width="78.75%" headers="mcps1.2.3.5.1 "><p id="p1486912563512"><a name="p1486912563512"></a><a name="p1486912563512"></a>true</p></td>
</tr>
<tr id="row118691556053"><th class="firstcol" valign="top" width="21.25%" id="mcps1.2.3.6.1"><p id="p086911565512"><a name="p086911565512"></a><a name="p086911565512"></a>Required</p></th>
<td class="cellrowborder" valign="top" width="78.75%" headers="mcps1.2.3.6.1 "><p id="p1286914567516"><a name="p1286914567516"></a><a name="p1286914567516"></a>No</p></td>
</tr>
</tbody>
</table>

**Table 6**  skip\_fusion\_layers parameter description

<a name="table1098019224612"></a>
<table><tbody><tr id="row109801522763"><th class="firstcol" valign="top" width="21.25%" id="mcps1.2.3.1.1"><p id="p198092215618"><a name="p198092215618"></a><a name="p198092215618"></a>Function</p></th>
<td class="cellrowborder" valign="top" width="78.75%" headers="mcps1.2.3.1.1 "><p id="p107159541274"><a name="p107159541274"></a><a name="p107159541274"></a>Skips the layers that can be fused.</p></td>
</tr>
<tr id="row1598115221363"><th class="firstcol" valign="top" width="21.25%" id="mcps1.2.3.2.1"><p id="p16981182210619"><a name="p16981182210619"></a><a name="p16981182210619"></a>Type</p></th>
<td class="cellrowborder" valign="top" width="78.75%" headers="mcps1.2.3.2.1 "><p id="p1930820365514"><a name="p1930820365514"></a><a name="p1930820365514"></a>string</p></td>
</tr>
<tr id="row99818221967"><th class="firstcol" valign="top" width="21.25%" id="mcps1.2.3.3.1"><p id="p169811226611"><a name="p169811226611"></a><a name="p169811226611"></a>Value Range</p></th>
<td class="cellrowborder" valign="top" width="78.75%" headers="mcps1.2.3.3.1 "><p id="p030823625118"><a name="p030823625118"></a><a name="p030823625118"></a>Layer names of the layers that can be fused. Currently, only Conv+BN fusion is supported.</p></td>
</tr>
<tr id="row398112226614"><th class="firstcol" valign="top" width="21.25%" id="mcps1.2.3.4.1"><p id="p11981122210615"><a name="p11981122210615"></a><a name="p11981122210615"></a>Parameter Description</p></th>
<td class="cellrowborder" valign="top" width="78.75%" headers="mcps1.2.3.4.1 "><p id="p330813363513"><a name="p330813363513"></a><a name="p330813363513"></a>Layers that do not need to be fused.</p></td>
</tr>
<tr id="row69814229618"><th class="firstcol" valign="top" width="21.25%" id="mcps1.2.3.5.1"><p id="p39819221620"><a name="p39819221620"></a><a name="p39819221620"></a>Recommended Configuration</p></th>
<td class="cellrowborder" valign="top" width="78.75%" headers="mcps1.2.3.5.1 "><p id="p6308536115120"><a name="p6308536115120"></a><a name="p6308536115120"></a>-</p></td>
</tr>
<tr id="row1998115223611"><th class="firstcol" valign="top" width="21.25%" id="mcps1.2.3.6.1"><p id="p1098116221464"><a name="p1098116221464"></a><a name="p1098116221464"></a>Required</p></th>
<td class="cellrowborder" valign="top" width="78.75%" headers="mcps1.2.3.6.1 "><p id="p1030813365511"><a name="p1030813365511"></a><a name="p1030813365511"></a>No</p></td>
</tr>
</tbody>
</table>

**Table 7**  layer\_config parameter description

<a name="zh-cn_topic_0240188002_table597149164211"></a>
<table><tbody><tr id="zh-cn_topic_0240188002_row3982499420"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0240188002_p19981049174216"><a name="zh-cn_topic_0240188002_p19981049174216"></a><a name="zh-cn_topic_0240188002_p19981049174216"></a>Function</p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0240188002_p0992049114220"><a name="zh-cn_topic_0240188002_p0992049114220"></a><a name="zh-cn_topic_0240188002_p0992049114220"></a>Specifies the quantization configuration of a network layer.</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row139910496428"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.2.1"><p id="zh-cn_topic_0240188002_p16991149194212"><a name="zh-cn_topic_0240188002_p16991149194212"></a><a name="zh-cn_topic_0240188002_p16991149194212"></a>Type</p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.2.1 "><p id="zh-cn_topic_0240188002_p1699174944218"><a name="zh-cn_topic_0240188002_p1699174944218"></a><a name="zh-cn_topic_0240188002_p1699174944218"></a>object</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row11991494422"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.3.1"><p id="zh-cn_topic_0240188002_p199917499424"><a name="zh-cn_topic_0240188002_p199917499424"></a><a name="zh-cn_topic_0240188002_p199917499424"></a>Value Range</p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.3.1 "><p id="zh-cn_topic_0240188002_p59994964210"><a name="zh-cn_topic_0240188002_p59994964210"></a><a name="zh-cn_topic_0240188002_p59994964210"></a>-</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row9100154913424"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.4.1"><p id="zh-cn_topic_0240188002_p1610211491423"><a name="zh-cn_topic_0240188002_p1610211491423"></a><a name="zh-cn_topic_0240188002_p1610211491423"></a><span>Parameter Description</span></p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.4.1 "><p id="zh-cn_topic_0240188002_p1610224984216"><a name="zh-cn_topic_0240188002_p1610224984216"></a><a name="zh-cn_topic_0240188002_p1610224984216"></a><span>The parameter contains the following parameters:</span></p>
<a name="zh-cn_topic_0240188002_ul131041949174214"></a><a name="zh-cn_topic_0240188002_ul131041949174214"></a><ul id="zh-cn_topic_0240188002_ul131041949174214"><li>quant_enable</li><li>activation_quant_params</li><li>weight_quant_params</li></ul></td>
</tr>
<tr id="zh-cn_topic_0240188002_row310514492425"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.5.1"><p id="zh-cn_topic_0240188002_p101051149114212"><a name="zh-cn_topic_0240188002_p101051149114212"></a><a name="zh-cn_topic_0240188002_p101051149114212"></a><span>Recommended Configuration</span></p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.5.1 "><p id="zh-cn_topic_0240188002_p14105149164216"><a name="zh-cn_topic_0240188002_p14105149164216"></a><a name="zh-cn_topic_0240188002_p14105149164216"></a>-</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row1010584913423"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.6.1"><p id="zh-cn_topic_0240188002_p10105049144220"><a name="zh-cn_topic_0240188002_p10105049144220"></a><a name="zh-cn_topic_0240188002_p10105049144220"></a>Required</p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.6.1 "><p id="zh-cn_topic_0240188002_p181051490427"><a name="zh-cn_topic_0240188002_p181051490427"></a><a name="zh-cn_topic_0240188002_p181051490427"></a>No</p></td>
</tr>
</tbody>
</table>

**Table 8**  quant\_enable parameter description

<a name="zh-cn_topic_0240188002_table9105154944219"></a>
<table><tbody><tr id="zh-cn_topic_0240188002_row2105449124216"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0240188002_p61051849124214"><a name="zh-cn_topic_0240188002_p61051849124214"></a><a name="zh-cn_topic_0240188002_p61051849124214"></a>Function</p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0240188002_p1910520496429"><a name="zh-cn_topic_0240188002_p1910520496429"></a><a name="zh-cn_topic_0240188002_p1910520496429"></a>Whether to quantize this layer.</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row71051491425"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.2.1"><p id="zh-cn_topic_0240188002_p1610554910428"><a name="zh-cn_topic_0240188002_p1610554910428"></a><a name="zh-cn_topic_0240188002_p1610554910428"></a><span>Type</span></p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.2.1 "><p id="zh-cn_topic_0240188002_p1106114914425"><a name="zh-cn_topic_0240188002_p1106114914425"></a><a name="zh-cn_topic_0240188002_p1106114914425"></a>bool</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row141061749144215"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.3.1"><p id="zh-cn_topic_0240188002_p61066491428"><a name="zh-cn_topic_0240188002_p61066491428"></a><a name="zh-cn_topic_0240188002_p61066491428"></a><span>Value Range</span></p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.3.1 "><p id="zh-cn_topic_0240188002_p161061549144216"><a name="zh-cn_topic_0240188002_p161061549144216"></a><a name="zh-cn_topic_0240188002_p161061549144216"></a>true or false</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row310615491425"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.4.1"><p id="zh-cn_topic_0240188002_p101061049114215"><a name="zh-cn_topic_0240188002_p101061049114215"></a><a name="zh-cn_topic_0240188002_p101061049114215"></a>Parameter Description</p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.4.1 "><a name="ul124349583119"></a><a name="ul124349583119"></a><ul id="ul124349583119"><li>true: Quantize this layer.</li><li>false: Do not quantize this layer.</li></ul></td>
</tr>
<tr id="zh-cn_topic_0240188002_row1110624918422"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.5.1"><p id="zh-cn_topic_0240188002_p1010614964211"><a name="zh-cn_topic_0240188002_p1010614964211"></a><a name="zh-cn_topic_0240188002_p1010614964211"></a><span>Recommended Configuration</span></p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.5.1 "><p id="zh-cn_topic_0240188002_p11106194920426"><a name="zh-cn_topic_0240188002_p11106194920426"></a><a name="zh-cn_topic_0240188002_p11106194920426"></a>true</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row8106104964219"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.6.1"><p id="zh-cn_topic_0240188002_p1710614974215"><a name="zh-cn_topic_0240188002_p1710614974215"></a><a name="zh-cn_topic_0240188002_p1710614974215"></a>Required</p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.6.1 "><p id="zh-cn_topic_0240188002_p10106349204212"><a name="zh-cn_topic_0240188002_p10106349204212"></a><a name="zh-cn_topic_0240188002_p10106349204212"></a>No</p></td>
</tr>
</tbody>
</table>

**Table 9**  dmq\_balancer\_param parameter description

<a name="table71745135336"></a>
<table><tbody><tr id="row16174813193319"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.1.1"><p id="p1175101313331"><a name="p1175101313331"></a><a name="p1175101313331"></a>Function</p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.1.1 "><p id="p12175201323319"><a name="p12175201323319"></a><a name="p12175201323319"></a>Transfer strength in the DMQ balancing algorithm.</p></td>
</tr>
<tr id="row817551312338"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.2.1"><p id="p1917513137339"><a name="p1917513137339"></a><a name="p1917513137339"></a><span>Type</span></p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.2.1 "><p id="p37291410133717"><a name="p37291410133717"></a><a name="p37291410133717"></a>float</p></td>
</tr>
<tr id="row1017511320330"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.3.1"><p id="p51751135334"><a name="p51751135334"></a><a name="p51751135334"></a><span>Value Range</span></p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.3.1 "><p id="p582111673812"><a name="p582111673812"></a><a name="p582111673812"></a>[0.2, 0.8]</p></td>
</tr>
<tr id="row17175713173311"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.4.1"><p id="p118111393314"><a name="p118111393314"></a><a name="p118111393314"></a>Parameter Description</p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.4.1 "><p id="p513117514346"><a name="p513117514346"></a><a name="p513117514346"></a>Represents the degree to which the quantization difficulty of the activation data is transferred to the weight. The larger the outliers in the data distribution, the smaller the transfer strength should be set.</p></td>
</tr>
<tr id="row1318111310337"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.5.1"><p id="p2181131315338"><a name="p2181131315338"></a><a name="p2181131315338"></a><span>Recommended Configuration</span></p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.5.1 "><p id="p1618281303310"><a name="p1618281303310"></a><a name="p1618281303310"></a>0.5</p></td>
</tr>
<tr id="row21821513133310"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.6.1"><p id="p1718231373317"><a name="p1718231373317"></a><a name="p1718231373317"></a>Required</p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.6.1 "><p id="p3182111393314"><a name="p3182111393314"></a><a name="p3182111393314"></a>No</p></td>
</tr>
</tbody>
</table>

**Table 10**  activation\_quant\_params parameter description

<a name="zh-cn_topic_0240188002_table1106164914218"></a>
<table><tbody><tr id="zh-cn_topic_0240188002_row1310710496428"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0240188002_p181071749154218"><a name="zh-cn_topic_0240188002_p181071749154218"></a><a name="zh-cn_topic_0240188002_p181071749154218"></a>Function</p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0240188002_p15107194934212"><a name="zh-cn_topic_0240188002_p15107194934212"></a><a name="zh-cn_topic_0240188002_p15107194934212"></a>Parameters for data quantization of this layer.</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row171071549124219"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.2.1"><p id="zh-cn_topic_0240188002_p1210784934213"><a name="zh-cn_topic_0240188002_p1210784934213"></a><a name="zh-cn_topic_0240188002_p1210784934213"></a><span>Type</span></p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.2.1 "><p id="zh-cn_topic_0240188002_p710844904219"><a name="zh-cn_topic_0240188002_p710844904219"></a><a name="zh-cn_topic_0240188002_p710844904219"></a>object</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row61082049194213"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.3.1"><p id="zh-cn_topic_0240188002_p12108114915423"><a name="zh-cn_topic_0240188002_p12108114915423"></a><a name="zh-cn_topic_0240188002_p12108114915423"></a><span>Value Range</span></p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.3.1 "><p id="zh-cn_topic_0240188002_p910934911424"><a name="zh-cn_topic_0240188002_p910934911424"></a><a name="zh-cn_topic_0240188002_p910934911424"></a>-</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row151093498424"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.4.1"><p id="zh-cn_topic_0240188002_p310964964219"><a name="zh-cn_topic_0240188002_p310964964219"></a><a name="zh-cn_topic_0240188002_p310964964219"></a><span>Parameter Description</span></p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.4.1 "><p id="zh-cn_topic_0240188002_p3109194917426"><a name="zh-cn_topic_0240188002_p3109194917426"></a><a name="zh-cn_topic_0240188002_p3109194917426"></a><span>activation_quant_params contains the following parameters. Parameters related to the </span>IFMR algorithm and parameters related to the HFMG algorithm cannot appear in the same layer simultaneously:</p>
<a name="ul2041339173710"></a><a name="ul2041339173710"></a><ul id="ul2041339173710"><li>Parameters involved in the IFMR data quantization algorithm:<a name="zh-cn_topic_0240188002_ul5109194913426"></a><a name="zh-cn_topic_0240188002_ul5109194913426"></a><ul id="zh-cn_topic_0240188002_ul5109194913426"><li>max_percentile</li><li><span>min_percentile</span></li><li>search_range</li><li>search_step</li><li>act_algo</li><li>num_bits</li><li>asymmetric</li></ul>
</li><li>Parameters involved in the HFMG data quantization algorithm:<a name="ul987033263710"></a><a name="ul987033263710"></a><ul id="ul987033263710"><li>act_algo</li><li>num_of_bins</li><li>num_bits</li><li>asymmetric</li></ul>
</li></ul></td>
</tr>
<tr id="zh-cn_topic_0240188002_row910984944210"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.5.1"><p id="zh-cn_topic_0240188002_p9109549164220"><a name="zh-cn_topic_0240188002_p9109549164220"></a><a name="zh-cn_topic_0240188002_p9109549164220"></a><span>Recommended Configuration</span></p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.5.1 "><p id="zh-cn_topic_0240188002_p10109749114219"><a name="zh-cn_topic_0240188002_p10109749114219"></a><a name="zh-cn_topic_0240188002_p10109749114219"></a>-</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row11109649154214"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.6.1"><p id="zh-cn_topic_0240188002_p810994910422"><a name="zh-cn_topic_0240188002_p810994910422"></a><a name="zh-cn_topic_0240188002_p810994910422"></a>Required</p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.6.1 "><p id="zh-cn_topic_0240188002_p81091549174210"><a name="zh-cn_topic_0240188002_p81091549174210"></a><a name="zh-cn_topic_0240188002_p81091549174210"></a>No</p></td>
</tr>
</tbody>
</table>

**Table 11**  weight\_quant\_params parameter description

<a name="zh-cn_topic_0240188002_table41099497422"></a>
<table><tbody><tr id="zh-cn_topic_0240188002_row7110149114213"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0240188002_p8110154914218"><a name="zh-cn_topic_0240188002_p8110154914218"></a><a name="zh-cn_topic_0240188002_p8110154914218"></a>Function</p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0240188002_p4110144974216"><a name="zh-cn_topic_0240188002_p4110144974216"></a><a name="zh-cn_topic_0240188002_p4110144974216"></a>Parameters for weight quantization of this layer.</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row511084915425"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.2.1"><p id="zh-cn_topic_0240188002_p1711034920429"><a name="zh-cn_topic_0240188002_p1711034920429"></a><a name="zh-cn_topic_0240188002_p1711034920429"></a>Type</p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.2.1 "><p id="zh-cn_topic_0240188002_p2110184934216"><a name="zh-cn_topic_0240188002_p2110184934216"></a><a name="zh-cn_topic_0240188002_p2110184934216"></a>object</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row14110949114219"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.3.1"><p id="zh-cn_topic_0240188002_p511017497427"><a name="zh-cn_topic_0240188002_p511017497427"></a><a name="zh-cn_topic_0240188002_p511017497427"></a><span>Value Range</span></p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.3.1 "><p id="zh-cn_topic_0240188002_p1611044919422"><a name="zh-cn_topic_0240188002_p1611044919422"></a><a name="zh-cn_topic_0240188002_p1611044919422"></a>-</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row10110849174215"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.4.1"><p id="zh-cn_topic_0240188002_p311010490425"><a name="zh-cn_topic_0240188002_p311010490425"></a><a name="zh-cn_topic_0240188002_p311010490425"></a><span>Parameter Description</span></p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.4.1 "><div class="p" id="p188547561511"><a name="p188547561511"></a><a name="p188547561511"></a>Includes the following parameters:<a name="ul76391953135412"></a><a name="ul76391953135412"></a><ul id="ul76391953135412"><li>num_bits</li><li>wts_algo</li><li>channel_wise</li></ul>
</div></td>
</tr>
<tr id="zh-cn_topic_0240188002_row7110134911422"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.5.1"><p id="zh-cn_topic_0240188002_p611012493427"><a name="zh-cn_topic_0240188002_p611012493427"></a><a name="zh-cn_topic_0240188002_p611012493427"></a><span>Recommended Configuration</span></p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.5.1 "><p id="zh-cn_topic_0240188002_p51111049114211"><a name="zh-cn_topic_0240188002_p51111049114211"></a><a name="zh-cn_topic_0240188002_p51111049114211"></a>-</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row191112499424"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.6.1"><p id="zh-cn_topic_0240188002_p8111184954215"><a name="zh-cn_topic_0240188002_p8111184954215"></a><a name="zh-cn_topic_0240188002_p8111184954215"></a>Required</p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.6.1 "><p id="zh-cn_topic_0240188002_p911116497423"><a name="zh-cn_topic_0240188002_p911116497423"></a><a name="zh-cn_topic_0240188002_p911116497423"></a>No</p></td>
</tr>
</tbody>
</table>

**Table 12**  num\_bits parameter description

<a name="table843315412593"></a>
<table><tbody><tr id="row743317549597"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.1.1"><p id="p19433205412598"><a name="p19433205412598"></a><a name="p19433205412598"></a>Function</p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.1.1 "><p id="p6433195419598"><a name="p6433195419598"></a><a name="p6433195419598"></a>Quantization bit width.</p></td>
</tr>
<tr id="row943355417593"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.2.1"><p id="p1443465418595"><a name="p1443465418595"></a><a name="p1443465418595"></a><span>Type</span></p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.2.1 "><p id="p5434854155918"><a name="p5434854155918"></a><a name="p5434854155918"></a>int</p></td>
</tr>
<tr id="row2434105415594"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.3.1"><p id="p13434115405916"><a name="p13434115405916"></a><a name="p13434115405916"></a><span>Value Range</span></p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.3.1 "><p id="p8434854105914"><a name="p8434854105914"></a><a name="p8434854105914"></a>[4,6,7,8,16]</p></td>
</tr>
<tr id="row10434135414594"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.4.1"><p id="p743495495913"><a name="p743495495913"></a><a name="p743495495913"></a><span>Parameter Description</span></p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.4.1 "><p id="p135891417116"><a name="p135891417116"></a><a name="p135891417116"></a>Activation quantization: the parameter value is [8,16].<br>Weight quantization: the parameter value is [4,6,7,8].</p></td>
</tr>
<tr id="row18435125416596"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.5.1"><p id="p11435954165919"><a name="p11435954165919"></a><a name="p11435954165919"></a><span>Recommended Configuration</span></p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.5.1 "><p id="p943565405919"><a name="p943565405919"></a><a name="p943565405919"></a>Currently only supports configuring it to 8, which indicates the use of INT8 quantization bit width.</p></td>
</tr>
<tr id="row1743510544597"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.6.1"><p id="p15435175485919"><a name="p15435175485919"></a><a name="p15435175485919"></a>Required</p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.6.1 "><p id="p44357542598"><a name="p44357542598"></a><a name="p44357542598"></a>Yes</p></td>
</tr>
</tbody>
</table>

**Table 13**  act\_algo parameter description

<a name="table14530541105619"></a>
<table><tbody><tr id="row4530194145620"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.1.1"><p id="p17530164135613"><a name="p17530164135613"></a><a name="p17530164135613"></a>Function</p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.1.1 "><p id="p12530141145615"><a name="p12530141145615"></a><a name="p12530141145615"></a>Data quantization algorithm.</p></td>
</tr>
<tr id="row17530194117562"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.2.1"><p id="p2053114411560"><a name="p2053114411560"></a><a name="p2053114411560"></a><span>Type</span></p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.2.1 "><p id="p1531941195618"><a name="p1531941195618"></a><a name="p1531941195618"></a>string</p></td>
</tr>
<tr id="row1453118416563"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.3.1"><p id="p1253104119568"><a name="p1253104119568"></a><a name="p1253104119568"></a><span>Value Range</span></p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.3.1 "><p id="p15190145571"><a name="p15190145571"></a><a name="p15190145571"></a>ifmr or hfmg</p></td>
</tr>
<tr id="row853104118568"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.4.1"><p id="p2531104114568"><a name="p2531104114568"></a><a name="p2531104114568"></a><span>Parameter Description</span></p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.4.1 "><p id="p5231143755711"><a name="p5231143755711"></a><a name="p5231143755711"></a>IFMR data quantization algorithm: ifmr</p>
<p id="p5271145810572"><a name="p5271145810572"></a><a name="p5271145810572"></a>HFMG data quantization algorithm: hfmg</p></td>
</tr>
<tr id="row453110419569"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.5.1"><p id="p3531194125614"><a name="p3531194125614"></a><a name="p3531194125614"></a><span>Recommended Configuration</span></p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.5.1 "><p id="p05321941125618"><a name="p05321941125618"></a><a name="p05321941125618"></a>-</p></td>
</tr>
<tr id="row175321941135610"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.6.1"><p id="p95321441105612"><a name="p95321441105612"></a><a name="p95321441105612"></a>Required</p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.6.1 "><p id="p165325412567"><a name="p165325412567"></a><a name="p165325412567"></a>No</p></td>
</tr>
</tbody>
</table>

**Table 14** asymmetric parameter description

<a name="table9528173141517"></a>
<table><tbody><tr id="row8528233157"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.1.1"><p id="p852813311159"><a name="p852813311159"></a><a name="p852813311159"></a>Function</p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.1.1 "><p id="p20528153111517"><a name="p20528153111517"></a><a name="p20528153111517"></a>Controls whether data quantization is symmetric or asymmetric. Used to control the selection of the per-layer quantization algorithm.</p>
<p id="p13978363386"><a name="p13978363386"></a><a name="p13978363386"></a>If both activation_offset and asymmetric exist in the configuration file, the asymmetric parameter takes precedence over the activation_offset parameter.</p></td>
</tr>
<tr id="row652816320159"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.2.1"><p id="p16529831152"><a name="p16529831152"></a><a name="p16529831152"></a>Type</p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.2.1 "><p id="p1529103181515"><a name="p1529103181515"></a><a name="p1529103181515"></a>bool</p></td>
</tr>
<tr id="row1152983101517"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.3.1"><p id="p1752963191510"><a name="p1752963191510"></a><a name="p1752963191510"></a>Value Range</p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.3.1 "><p id="p652917311151"><a name="p652917311151"></a><a name="p652917311151"></a>true or false</p></td>
</tr>
<tr id="row6529439151"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.4.1"><p id="p75299361513"><a name="p75299361513"></a><a name="p75299361513"></a>Parameter Description</p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.4.1 "><a name="ul152963121514"></a><a name="ul152963121514"></a><ul id="ul152963121514"><li>true: Asymmetric quantization is used for data quantization.</li><li>false: Symmetric quantization is used for data quantization.</li></ul></td>
</tr>
<tr id="row145298351511"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.5.1"><p id="p17529236158"><a name="p17529236158"></a><a name="p17529236158"></a>Recommended Configuration</p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.5.1 "><p id="p1352917391519"><a name="p1352917391519"></a><a name="p1352917391519"></a>true</p></td>
</tr>
<tr id="row135297311154"><th class="firstcol" valign="top" width="21.349999999999998%" id="mcps1.2.3.6.1"><p id="p1152919312158"><a name="p1152919312158"></a><a name="p1152919312158"></a>Required</p></th>
<td class="cellrowborder" valign="top" width="78.64999999999999%" headers="mcps1.2.3.6.1 "><p id="p1052915391510"><a name="p1052915391510"></a><a name="p1052915391510"></a>No</p></td>
</tr>
</tbody>
</table>

**Table 15**  max\_percentile parameter description

<a name="zh-cn_topic_0240188002_table21111149194212"></a>
<table><tbody><tr id="zh-cn_topic_0240188002_row1711118493423"><th class="firstcol" valign="top" width="20.549999999999997%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0240188002_p1811174974215"><a name="zh-cn_topic_0240188002_p1811174974215"></a><a name="zh-cn_topic_0240188002_p1811174974215"></a>Function</p></th>
<td class="cellrowborder" valign="top" width="79.45%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0240188002_zh-cn_topic_0171620822_p15573501614"><a name="zh-cn_topic_0240188002_zh-cn_topic_0171620822_p15573501614"></a><a name="zh-cn_topic_0240188002_zh-cn_topic_0171620822_p15573501614"></a><span>Maximum value search position parameter in the IFMR data quantization algorithm.</span></p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row1711194917429"><th class="firstcol" valign="top" width="20.549999999999997%" id="mcps1.2.3.2.1"><p id="zh-cn_topic_0240188002_p811111491421"><a name="zh-cn_topic_0240188002_p811111491421"></a><a name="zh-cn_topic_0240188002_p811111491421"></a><span>Type</span></p></th>
<td class="cellrowborder" valign="top" width="79.45%" headers="mcps1.2.3.2.1 "><p id="zh-cn_topic_0240188002_p14111144994215"><a name="zh-cn_topic_0240188002_p14111144994215"></a><a name="zh-cn_topic_0240188002_p14111144994215"></a><span>float</span></p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row11121149164219"><th class="firstcol" valign="top" width="20.549999999999997%" id="mcps1.2.3.3.1"><p id="zh-cn_topic_0240188002_p161121349164212"><a name="zh-cn_topic_0240188002_p161121349164212"></a><a name="zh-cn_topic_0240188002_p161121349164212"></a><span>Value Range</span></p></th>
<td class="cellrowborder" valign="top" width="79.45%" headers="mcps1.2.3.3.1 "><p id="zh-cn_topic_0240188002_p9112449194210"><a name="zh-cn_topic_0240188002_p9112449194210"></a><a name="zh-cn_topic_0240188002_p9112449194210"></a>(0.5,1]</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row2112349134219"><th class="firstcol" valign="top" width="20.549999999999997%" id="mcps1.2.3.4.1"><p id="zh-cn_topic_0240188002_p131121249134215"><a name="zh-cn_topic_0240188002_p131121249134215"></a><a name="zh-cn_topic_0240188002_p131121249134215"></a>Parameter Description</p></th>
<td class="cellrowborder" valign="top" width="79.45%" headers="mcps1.2.3.4.1 "><p id="zh-cn_topic_0240188002_p2518182681514"><a name="zh-cn_topic_0240188002_p2518182681514"></a><a name="zh-cn_topic_0240188002_p2518182681514"></a><span>In a set of numbers sorted in descending order, this parameter determines which largest number to take. For example, if there are</span><span> 100 </span><span>numbers,</span><span>1.0 </span><span>means taking the</span><span> 100-100*1.0=0</span><span>th number, which corresponds to the first largest number.</span></p>
<p id="zh-cn_topic_0240188002_p811274914216"><a name="zh-cn_topic_0240188002_p811274914216"></a><a name="zh-cn_topic_0240188002_p811274914216"></a>When truncating the data to be quantized, a larger value means that the upper bound of truncation is closer to the maximum value of the data to be quantized.</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row21125493429"><th class="firstcol" valign="top" width="20.549999999999997%" id="mcps1.2.3.5.1"><p id="zh-cn_topic_0240188002_p9112174974214"><a name="zh-cn_topic_0240188002_p9112174974214"></a><a name="zh-cn_topic_0240188002_p9112174974214"></a><span>Recommended Configuration</span></p></th>
<td class="cellrowborder" valign="top" width="79.45%" headers="mcps1.2.3.5.1 "><p id="zh-cn_topic_0240188002_p181122049204214"><a name="zh-cn_topic_0240188002_p181122049204214"></a><a name="zh-cn_topic_0240188002_p181122049204214"></a>0.999999</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row17112164913423"><th class="firstcol" valign="top" width="20.549999999999997%" id="mcps1.2.3.6.1"><p id="zh-cn_topic_0240188002_p15112174920423"><a name="zh-cn_topic_0240188002_p15112174920423"></a><a name="zh-cn_topic_0240188002_p15112174920423"></a>Required</p></th>
<td class="cellrowborder" valign="top" width="79.45%" headers="mcps1.2.3.6.1 "><p id="zh-cn_topic_0240188002_p11121449194220"><a name="zh-cn_topic_0240188002_p11121449194220"></a><a name="zh-cn_topic_0240188002_p11121449194220"></a>No</p></td>
</tr>
</tbody>
</table>

**Table 16**  min\_percentile parameter description

<a name="zh-cn_topic_0240188002_table81121349164210"></a>
<table><tbody><tr id="zh-cn_topic_0240188002_row13112104924219"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0240188002_p2011274934216"><a name="zh-cn_topic_0240188002_p2011274934216"></a><a name="zh-cn_topic_0240188002_p2011274934216"></a>Function</p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0240188002_p1047333518552"><a name="zh-cn_topic_0240188002_p1047333518552"></a><a name="zh-cn_topic_0240188002_p1047333518552"></a><span>Minimum value search position</span> parameter in the IFMR data quantization algorithm.</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row13112164974211"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.2.1"><p id="zh-cn_topic_0240188002_p311264916426"><a name="zh-cn_topic_0240188002_p311264916426"></a><a name="zh-cn_topic_0240188002_p311264916426"></a><span>Type</span></p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.2.1 "><p id="zh-cn_topic_0240188002_p1911344915420"><a name="zh-cn_topic_0240188002_p1911344915420"></a><a name="zh-cn_topic_0240188002_p1911344915420"></a>float</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row3113449204220"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.3.1"><p id="zh-cn_topic_0240188002_p2113204917421"><a name="zh-cn_topic_0240188002_p2113204917421"></a><a name="zh-cn_topic_0240188002_p2113204917421"></a><span>Value Range</span></p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.3.1 "><p id="zh-cn_topic_0240188002_p011316492420"><a name="zh-cn_topic_0240188002_p011316492420"></a><a name="zh-cn_topic_0240188002_p011316492420"></a>(0.5,1]</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row171131049154212"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.4.1"><p id="zh-cn_topic_0240188002_p11113174944211"><a name="zh-cn_topic_0240188002_p11113174944211"></a><a name="zh-cn_topic_0240188002_p11113174944211"></a>Parameter Description</p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.4.1 "><p id="zh-cn_topic_0240188002_p2082735115146"><a name="zh-cn_topic_0240188002_p2082735115146"></a><a name="zh-cn_topic_0240188002_p2082735115146"></a><span>In a set of numbers sorted in ascending order, determines which smallest number to take. For example, with</span><span> 100 </span><span>numbers,</span><span>1.0 </span><span>means taking the</span><span> 100-100*1.0=0</span><span>, which corresponds to the first smallest number.</span></p>
<p id="zh-cn_topic_0240188002_p15400171119157"><a name="zh-cn_topic_0240188002_p15400171119157"></a><a name="zh-cn_topic_0240188002_p15400171119157"></a>When truncating the data to be quantized, a larger value means that the lower bound of truncation is closer to the minimum value of the data to be quantized.</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row2113849184218"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.5.1"><p id="zh-cn_topic_0240188002_p1811364994213"><a name="zh-cn_topic_0240188002_p1811364994213"></a><a name="zh-cn_topic_0240188002_p1811364994213"></a><span>Recommended Configuration</span></p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.5.1 "><p id="zh-cn_topic_0240188002_p911324912427"><a name="zh-cn_topic_0240188002_p911324912427"></a><a name="zh-cn_topic_0240188002_p911324912427"></a>0.999999</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row151131849114219"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.6.1"><p id="zh-cn_topic_0240188002_p711304916429"><a name="zh-cn_topic_0240188002_p711304916429"></a><a name="zh-cn_topic_0240188002_p711304916429"></a>Required</p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.6.1 "><p id="zh-cn_topic_0240188002_p8113134914215"><a name="zh-cn_topic_0240188002_p8113134914215"></a><a name="zh-cn_topic_0240188002_p8113134914215"></a>No</p></td>
</tr>
</tbody>
</table>

**Table 17**  search\_range parameter description

<a name="zh-cn_topic_0240188002_table16114114994220"></a>
<table><tbody><tr id="zh-cn_topic_0240188002_row41141749104219"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0240188002_p6114949184219"><a name="zh-cn_topic_0240188002_p6114949184219"></a><a name="zh-cn_topic_0240188002_p6114949184219"></a>Function</p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.1.1 "><p id="p151191040112320"><a name="p151191040112320"></a><a name="p151191040112320"></a>In the IFMR data quantization algorithm, controls the search range [search_range_start, search_range_end] of the quantization factor.</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row511474912426"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.2.1"><p id="zh-cn_topic_0240188002_p19114154994210"><a name="zh-cn_topic_0240188002_p19114154994210"></a><a name="zh-cn_topic_0240188002_p19114154994210"></a><span>Type</span></p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.2.1 "><p id="p101951353443"><a name="p101951353443"></a><a name="p101951353443"></a>list, with two elements of Type float.</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row1511454924217"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.3.1"><p id="zh-cn_topic_0240188002_p141149498424"><a name="zh-cn_topic_0240188002_p141149498424"></a><a name="zh-cn_topic_0240188002_p141149498424"></a><span>Value Range</span></p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.3.1 "><p id="p151959352442"><a name="p151959352442"></a><a name="p151959352442"></a>0&lt;search_range_start&lt;search_range_end</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row7114449104220"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.4.1"><p id="zh-cn_topic_0240188002_p181141949124212"><a name="zh-cn_topic_0240188002_p181141949124212"></a><a name="zh-cn_topic_0240188002_p181141949124212"></a><span>Parameter Description</span></p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.4.1 "><p id="p2019543584419"><a name="p2019543584419"></a><a name="p2019543584419"></a>Controls the floating range of the upper bound of truncation.</p>
<a name="ul1019563544418"></a><a name="ul1019563544418"></a><ul id="ul1019563544418"><li>search_range_start: <span>Determines the position where the search starts</span>.</li><li>search_range_end: <span>Determines the position where the search ends</span>.</li></ul></td>
</tr>
<tr id="zh-cn_topic_0240188002_row17115144916422"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.5.1"><p id="zh-cn_topic_0240188002_p3115749174210"><a name="zh-cn_topic_0240188002_p3115749174210"></a><a name="zh-cn_topic_0240188002_p3115749174210"></a><span>Recommended Configuration</span></p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.5.1 "><p id="p81961835144412"><a name="p81961835144412"></a><a name="p81961835144412"></a>[0.7,1.3]</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row13115114912422"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.6.1"><p id="zh-cn_topic_0240188002_p8115104913424"><a name="zh-cn_topic_0240188002_p8115104913424"></a><a name="zh-cn_topic_0240188002_p8115104913424"></a>Required</p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.6.1 "><p id="p7196183520448"><a name="p7196183520448"></a><a name="p7196183520448"></a>No</p></td>
</tr>
</tbody>
</table>

**Table 18**  search\_step parameter description

<a name="zh-cn_topic_0240188002_table4116144911429"></a>
<table><tbody><tr id="zh-cn_topic_0240188002_row191164492426"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0240188002_p12117184915425"><a name="zh-cn_topic_0240188002_p12117184915425"></a><a name="zh-cn_topic_0240188002_p12117184915425"></a>Function</p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0240188002_p111814492426"><a name="zh-cn_topic_0240188002_p111814492426"></a><a name="zh-cn_topic_0240188002_p111814492426"></a>In the IFMR data quantization algorithm, controls the search step of the quantization factor.</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row5118449204214"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.2.1"><p id="zh-cn_topic_0240188002_p811834964210"><a name="zh-cn_topic_0240188002_p811834964210"></a><a name="zh-cn_topic_0240188002_p811834964210"></a>Type</p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.2.1 "><p id="zh-cn_topic_0240188002_p11118114919423"><a name="zh-cn_topic_0240188002_p11118114919423"></a><a name="zh-cn_topic_0240188002_p11118114919423"></a>float</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row61181649184211"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.3.1"><p id="zh-cn_topic_0240188002_p12118124964211"><a name="zh-cn_topic_0240188002_p12118124964211"></a><a name="zh-cn_topic_0240188002_p12118124964211"></a>Value Range</p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.3.1 "><p id="zh-cn_topic_0240188002_p181184492427"><a name="zh-cn_topic_0240188002_p181184492427"></a><a name="zh-cn_topic_0240188002_p181184492427"></a>(0,  (search_range_end-search_range_start)]</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row14118164915427"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.4.1"><p id="zh-cn_topic_0240188002_p1911874954219"><a name="zh-cn_topic_0240188002_p1911874954219"></a><a name="zh-cn_topic_0240188002_p1911874954219"></a>Parameter Description</p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.4.1 "><p id="zh-cn_topic_0240188002_p17119164954213"><a name="zh-cn_topic_0240188002_p17119164954213"></a><a name="zh-cn_topic_0240188002_p17119164954213"></a>Controls the floating range step of the truncation upper bound. A smaller value results in a smaller floating step.</p>
<p id="p1863301145212"><a name="p1863301145212"></a><a name="p1863301145212"></a>The number of searches search_iteration=(search_range_end-search_range_start)/search_step. If the number of searches is too large, the search time will be very long, which may cause a problem similar to process hang in this scenario.</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row13119124920425"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.5.1"><p id="zh-cn_topic_0240188002_p911917495423"><a name="zh-cn_topic_0240188002_p911917495423"></a><a name="zh-cn_topic_0240188002_p911917495423"></a><span>Recommended Configuration</span></p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.5.1 "><p id="zh-cn_topic_0240188002_p1911914912428"><a name="zh-cn_topic_0240188002_p1911914912428"></a><a name="zh-cn_topic_0240188002_p1911914912428"></a>0.01</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row011994914218"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.6.1"><p id="zh-cn_topic_0240188002_p1611920497425"><a name="zh-cn_topic_0240188002_p1611920497425"></a><a name="zh-cn_topic_0240188002_p1611920497425"></a>Required</p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.6.1 "><p id="zh-cn_topic_0240188002_p511994910423"><a name="zh-cn_topic_0240188002_p511994910423"></a><a name="zh-cn_topic_0240188002_p511994910423"></a>No</p></td>
</tr>
</tbody>
</table>

**Table 19**  num\_of\_bins parameter description

<a name="table1875524217587"></a>
<table><tbody><tr id="row37551942105813"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.1.1"><p id="p5756134235815"><a name="p5756134235815"></a><a name="p5756134235815"></a>Function</p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.1.1 "><p id="p1796548105920"><a name="p1796548105920"></a><a name="p1796548105920"></a>The HFMG data quantization algorithm is used to adjust the number of bins (the smallest unit of a histogram) in the histogram.</p></td>
</tr>
<tr id="row675614216580"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.2.1"><p id="p1675684217584"><a name="p1675684217584"></a><a name="p1675684217584"></a><span>Type</span></p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.2.1 "><p id="p177968482596"><a name="p177968482596"></a><a name="p177968482596"></a>unsigned int</p></td>
</tr>
<tr id="row2756164265814"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.3.1"><p id="p57561442105816"><a name="p57561442105816"></a><a name="p57561442105816"></a><span>Value Range</span></p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.3.1 "><p id="p187967484594"><a name="p187967484594"></a><a name="p187967484594"></a>{1024, 2048, 4096, 8192}</p></td>
</tr>
<tr id="row5756124225815"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.4.1"><p id="p17757194217581"><a name="p17757194217581"></a><a name="p17757194217581"></a><span>Parameter Description</span></p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.4.1 "><p id="p6796948145917"><a name="p6796948145917"></a><a name="p6796948145917"></a>A larger num_of_bins value enables the histogram to better fit the original data distribution, which may yield better quantization results, but it also increases the time consumed by the post-training quantization process.</p></td>
</tr>
<tr id="row16757742145818"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.5.1"><p id="p1575713426583"><a name="p1575713426583"></a><a name="p1575713426583"></a><span>Recommended Configuration</span></p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.5.1 "><p id="p1179616484591"><a name="p1179616484591"></a><a name="p1179616484591"></a>4096</p></td>
</tr>
<tr id="row7757642155817"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.6.1"><p id="p4758134217582"><a name="p4758134217582"></a><a name="p4758134217582"></a>Required</p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.6.1 "><p id="p335617395715"><a name="p335617395715"></a><a name="p335617395715"></a>In the HFMG algorithm quantization scenario, this parameter is optional.</p></td>
</tr>
</tbody>
</table>

**Table 20**  wts\_algo parameter description

<a name="table1845148114717"></a>
<table><tbody><tr id="row54527864713"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.1.1"><p id="p1345216813478"><a name="p1345216813478"></a><a name="p1345216813478"></a>Function</p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.1.1 "><p id="p1645217810478"><a name="p1645217810478"></a><a name="p1645217810478"></a>Weight quantization algorithm</p></td>
</tr>
<tr id="row194521812479"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.2.1"><p id="p1245318884717"><a name="p1245318884717"></a><a name="p1245318884717"></a><span>Type</span></p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.2.1 "><p id="p64531785473"><a name="p64531785473"></a><a name="p64531785473"></a>string</p></td>
</tr>
<tr id="row1145338154715"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.3.1"><p id="p164532816477"><a name="p164532816477"></a><a name="p164532816477"></a><span>Value Range</span></p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.3.1 "><p id="p188318148481"><a name="p188318148481"></a><a name="p188318148481"></a>arq_quantize</p></td>
</tr>
<tr id="row345398124719"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.4.1"><p id="p445314874710"><a name="p445314874710"></a><a name="p445314874710"></a><span>Parameter Description</span></p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.4.1 "><p id="p87021917114917"><a name="p87021917114917"></a><a name="p87021917114917"></a>ARQ weight quantization algorithm: arq_quantize</p></td>
</tr>
<tr id="row2045416894711"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.5.1"><p id="p1645418114713"><a name="p1645418114713"></a><a name="p1645418114713"></a><span>Recommended Configuration</span></p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.5.1 "><p id="p745416814717"><a name="p745416814717"></a><a name="p745416814717"></a>-</p></td>
</tr>
<tr id="row6454385476"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.6.1"><p id="p1445418874710"><a name="p1445418874710"></a><a name="p1445418874710"></a>Required</p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.6.1 "><p id="p174543811479"><a name="p174543811479"></a><a name="p174543811479"></a>No</p></td>
</tr>
</tbody>
</table>

**Table 21**  channel\_wise search-related parameter description

<a name="zh-cn_topic_0240188002_table20119449114211"></a>
<table><tbody><tr id="zh-cn_topic_0240188002_row1312004914427"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0240188002_p212034912427"><a name="zh-cn_topic_0240188002_p212034912427"></a><a name="zh-cn_topic_0240188002_p212034912427"></a>Function</p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0240188002_p412015494425"><a name="zh-cn_topic_0240188002_p412015494425"></a><a name="zh-cn_topic_0240188002_p412015494425"></a>Whether to use a different quantization factor for each channel in the ARQ weight quantization algorithm.</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row1112014498421"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.2.1"><p id="zh-cn_topic_0240188002_p1212054934216"><a name="zh-cn_topic_0240188002_p1212054934216"></a><a name="zh-cn_topic_0240188002_p1212054934216"></a><span>Type</span></p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.2.1 "><p id="zh-cn_topic_0240188002_p1120124919429"><a name="zh-cn_topic_0240188002_p1120124919429"></a><a name="zh-cn_topic_0240188002_p1120124919429"></a>bool</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row13120104916422"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.3.1"><p id="zh-cn_topic_0240188002_p141201049104216"><a name="zh-cn_topic_0240188002_p141201049104216"></a><a name="zh-cn_topic_0240188002_p141201049104216"></a><span>Value Range</span></p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.3.1 "><p id="zh-cn_topic_0240188002_p212094914428"><a name="zh-cn_topic_0240188002_p212094914428"></a><a name="zh-cn_topic_0240188002_p212094914428"></a>true or false</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row17120249164211"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.4.1"><p id="zh-cn_topic_0240188002_p1912018499427"><a name="zh-cn_topic_0240188002_p1912018499427"></a><a name="zh-cn_topic_0240188002_p1912018499427"></a><span>Parameter Description</span></p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.4.1 "><a name="zh-cn_topic_0240188002_ul1212114910429"></a><a name="zh-cn_topic_0240188002_ul1212114910429"></a><ul id="zh-cn_topic_0240188002_ul1212114910429"><li>true: Each channel is quantized independently with a different quantization factor.</li><li>false: All channels are quantized simultaneously and share the same quantization factor.</li></ul></td>
</tr>
<tr id="zh-cn_topic_0240188002_row312154964217"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.5.1"><p id="zh-cn_topic_0240188002_p61211449144215"><a name="zh-cn_topic_0240188002_p61211449144215"></a><a name="zh-cn_topic_0240188002_p61211449144215"></a><span>Recommended Configuration</span></p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.5.1 "><p id="zh-cn_topic_0240188002_p1112154954214"><a name="zh-cn_topic_0240188002_p1112154954214"></a><a name="zh-cn_topic_0240188002_p1112154954214"></a>true</p></td>
</tr>
<tr id="zh-cn_topic_0240188002_row41211549174220"><th class="firstcol" valign="top" width="20.52%" id="mcps1.2.3.6.1"><p id="zh-cn_topic_0240188002_p131211249104219"><a name="zh-cn_topic_0240188002_p131211249104219"></a><a name="zh-cn_topic_0240188002_p131211249104219"></a>Required</p></th>
<td class="cellrowborder" valign="top" width="79.47999999999999%" headers="mcps1.2.3.6.1 "><p id="zh-cn_topic_0240188002_p151212493424"><a name="zh-cn_topic_0240188002_p151212493424"></a><a name="zh-cn_topic_0240188002_p151212493424"></a>No</p></td>
</tr>
</tbody>
</table>
