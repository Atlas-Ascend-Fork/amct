# 量化感知训练配置参数说明<a name="ZH-CN_TOPIC_0000002517028796"></a>

如果通过[create\_quant\_retrain\_config](../api/create_quant_retrain_config.md)接口生成的config.json量化感知训练配置文件，推理精度不满足要求，则需要参见该章节不断调整config.json文件中的内容，直至精度满足要求，该文件部分内容样例请参见接口中的调用示例部分（用户修改json文件时，请确保层名唯一）。

配置文件中参数说明如下，其中[表9](#table1459214218113)\~[表11](#table1988120453112)的参数说明在手动调整量化配置文件时才会使用。

**表 2**  version参数说明

<a name="table1933120499584"></a>
<table><tbody><tr id="row2410154917588"><th class="firstcol" valign="top" width="29.459999999999997%" id="mcps1.2.3.1.1"><p id="p5410154919582"><a name="p5410154919582"></a><a name="p5410154919582"></a>作用</p>
</th>
<td class="cellrowborder" valign="top" width="70.54%" headers="mcps1.2.3.1.1 "><p id="p341012497587"><a name="p341012497587"></a><a name="p341012497587"></a>控制量化配置文件版本号。</p>
</td>
</tr>
<tr id="row1410184918581"><th class="firstcol" valign="top" width="29.459999999999997%" id="mcps1.2.3.2.1"><p id="p14410144912584"><a name="p14410144912584"></a><a name="p14410144912584"></a>类型</p>
</th>
<td class="cellrowborder" valign="top" width="70.54%" headers="mcps1.2.3.2.1 "><p id="p12410104910587"><a name="p12410104910587"></a><a name="p12410104910587"></a>int</p>
</td>
</tr>
<tr id="row1441024919580"><th class="firstcol" valign="top" width="29.459999999999997%" id="mcps1.2.3.3.1"><p id="p134106498581"><a name="p134106498581"></a><a name="p134106498581"></a>取值范围</p>
</th>
<td class="cellrowborder" valign="top" width="70.54%" headers="mcps1.2.3.3.1 "><p id="p18410174905818"><a name="p18410174905818"></a><a name="p18410174905818"></a>1</p>
</td>
</tr>
<tr id="row15410194985816"><th class="firstcol" valign="top" width="29.459999999999997%" id="mcps1.2.3.4.1"><p id="p20410184910588"><a name="p20410184910588"></a><a name="p20410184910588"></a>参数说明</p>
</th>
<td class="cellrowborder" valign="top" width="70.54%" headers="mcps1.2.3.4.1 "><p id="p34105493582"><a name="p34105493582"></a><a name="p34105493582"></a>目前仅有一个版本号1。</p>
</td>
</tr>
<tr id="row17410174919583"><th class="firstcol" valign="top" width="29.459999999999997%" id="mcps1.2.3.5.1"><p id="p2041010499589"><a name="p2041010499589"></a><a name="p2041010499589"></a>推荐配置</p>
</th>
<td class="cellrowborder" valign="top" width="70.54%" headers="mcps1.2.3.5.1 "><p id="p5410194915589"><a name="p5410194915589"></a><a name="p5410194915589"></a>1</p>
</td>
</tr>
<tr id="row841010493589"><th class="firstcol" valign="top" width="29.459999999999997%" id="mcps1.2.3.6.1"><p id="p94101149125820"><a name="p94101149125820"></a><a name="p94101149125820"></a>必选或可选</p>
</th>
<td class="cellrowborder" valign="top" width="70.54%" headers="mcps1.2.3.6.1 "><p id="p1041094913584"><a name="p1041094913584"></a><a name="p1041094913584"></a>可选</p>
</td>
</tr>
</tbody>
</table>

**表 3**  batch\_num参数说明

<a name="zh-cn_topic_0240188002_table1993174915423"></a>
<table><tbody><tr id="zh-cn_topic_0240188002_row0944492423"><th class="firstcol" valign="top" width="28.970000000000002%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0240188002_p294134994210"><a name="zh-cn_topic_0240188002_p294134994210"></a><a name="zh-cn_topic_0240188002_p294134994210"></a>作用</p>
</th>
<td class="cellrowborder" valign="top" width="71.03%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0240188002_p694164934219"><a name="zh-cn_topic_0240188002_p694164934219"></a><a name="zh-cn_topic_0240188002_p694164934219"></a>控制量化感知训练推理阶段使用多少个batch的数据。</p>
</td>
</tr>
<tr id="zh-cn_topic_0240188002_row294249104214"><th class="firstcol" valign="top" width="28.970000000000002%" id="mcps1.2.3.2.1"><p id="zh-cn_topic_0240188002_p1794164974216"><a name="zh-cn_topic_0240188002_p1794164974216"></a><a name="zh-cn_topic_0240188002_p1794164974216"></a><span>类型</span></p>
</th>
<td class="cellrowborder" valign="top" width="71.03%" headers="mcps1.2.3.2.1 "><p id="zh-cn_topic_0240188002_p1894449164210"><a name="zh-cn_topic_0240188002_p1894449164210"></a><a name="zh-cn_topic_0240188002_p1894449164210"></a>int</p>
</td>
</tr>
<tr id="zh-cn_topic_0240188002_row199420495426"><th class="firstcol" valign="top" width="28.970000000000002%" id="mcps1.2.3.3.1"><p id="zh-cn_topic_0240188002_p39419491427"><a name="zh-cn_topic_0240188002_p39419491427"></a><a name="zh-cn_topic_0240188002_p39419491427"></a><span>取值范围</span></p>
</th>
<td class="cellrowborder" valign="top" width="71.03%" headers="mcps1.2.3.3.1 "><p id="zh-cn_topic_0240188002_p694134954217"><a name="zh-cn_topic_0240188002_p694134954217"></a><a name="zh-cn_topic_0240188002_p694134954217"></a>大于0</p>
</td>
</tr>
<tr id="zh-cn_topic_0240188002_row394164944213"><th class="firstcol" valign="top" width="28.970000000000002%" id="mcps1.2.3.4.1"><p id="zh-cn_topic_0240188002_p1694104916421"><a name="zh-cn_topic_0240188002_p1694104916421"></a><a name="zh-cn_topic_0240188002_p1694104916421"></a>参数说明</p>
</th>
<td class="cellrowborder" valign="top" width="71.03%" headers="mcps1.2.3.4.1 "><p id="zh-cn_topic_0240188002_p159444919429"><a name="zh-cn_topic_0240188002_p159444919429"></a><a name="zh-cn_topic_0240188002_p159444919429"></a>如果不配置，则使用默认值1，建议校准集图片数量不超过50张，根据batch的大小batch_size计算相应的batch_num数值。</p>
<p id="zh-cn_topic_0240188002_p1223319101118"><a name="zh-cn_topic_0240188002_p1223319101118"></a><a name="zh-cn_topic_0240188002_p1223319101118"></a>batch_num*batch_size为量化使用的校准集图片数量。</p>
<p id="zh-cn_topic_0240188002_p201320014422"><a name="zh-cn_topic_0240188002_p201320014422"></a><a name="zh-cn_topic_0240188002_p201320014422"></a>其中batch_size为每个batch所用的图片数量。</p>
</td>
</tr>
<tr id="zh-cn_topic_0240188002_row8941849114212"><th class="firstcol" valign="top" width="28.970000000000002%" id="mcps1.2.3.5.1"><p id="zh-cn_topic_0240188002_p19945496428"><a name="zh-cn_topic_0240188002_p19945496428"></a><a name="zh-cn_topic_0240188002_p19945496428"></a>推荐配置</p>
</th>
<td class="cellrowborder" valign="top" width="71.03%" headers="mcps1.2.3.5.1 "><p id="zh-cn_topic_0240188002_p10941249164219"><a name="zh-cn_topic_0240188002_p10941249164219"></a><a name="zh-cn_topic_0240188002_p10941249164219"></a>1</p>
</td>
</tr>
<tr id="zh-cn_topic_0240188002_row194104974210"><th class="firstcol" valign="top" width="28.970000000000002%" id="mcps1.2.3.6.1"><p id="zh-cn_topic_0240188002_p694174913424"><a name="zh-cn_topic_0240188002_p694174913424"></a><a name="zh-cn_topic_0240188002_p694174913424"></a>必选或可选</p>
</th>
<td class="cellrowborder" valign="top" width="71.03%" headers="mcps1.2.3.6.1 "><p id="zh-cn_topic_0240188002_p89524910428"><a name="zh-cn_topic_0240188002_p89524910428"></a><a name="zh-cn_topic_0240188002_p89524910428"></a>可选</p>
</td>
</tr>
</tbody>
</table>

**表 4**  retrain\_enable参数说明

<a name="table4499101514598"></a>
<table><tbody><tr id="row75841715185920"><th class="firstcol" valign="top" width="29.509999999999998%" id="mcps1.2.3.1.1"><p id="p15584101575915"><a name="p15584101575915"></a><a name="p15584101575915"></a>作用</p>
</th>
<td class="cellrowborder" valign="top" width="70.49%" headers="mcps1.2.3.1.1 "><p id="p1658414155596"><a name="p1658414155596"></a><a name="p1658414155596"></a>该层是否进行量化感知训练。</p>
</td>
</tr>
<tr id="row058416153590"><th class="firstcol" valign="top" width="29.509999999999998%" id="mcps1.2.3.2.1"><p id="p1358413155596"><a name="p1358413155596"></a><a name="p1358413155596"></a>类型</p>
</th>
<td class="cellrowborder" valign="top" width="70.49%" headers="mcps1.2.3.2.1 "><p id="p2584015105915"><a name="p2584015105915"></a><a name="p2584015105915"></a>bool</p>
</td>
</tr>
<tr id="row958417153595"><th class="firstcol" valign="top" width="29.509999999999998%" id="mcps1.2.3.3.1"><p id="p175840158598"><a name="p175840158598"></a><a name="p175840158598"></a>取值范围</p>
</th>
<td class="cellrowborder" valign="top" width="70.49%" headers="mcps1.2.3.3.1 "><p id="p16584111517591"><a name="p16584111517591"></a><a name="p16584111517591"></a>true或false</p>
</td>
</tr>
<tr id="row155847155595"><th class="firstcol" valign="top" width="29.509999999999998%" id="mcps1.2.3.4.1"><p id="p85846153595"><a name="p85846153595"></a><a name="p85846153595"></a>参数说明</p>
</th>
<td class="cellrowborder" valign="top" width="70.49%" headers="mcps1.2.3.4.1 "><a name="ul119319439315"></a><a name="ul119319439315"></a><ul id="ul119319439315"><li>true：该层需要进行量化感知训练。</li><li>false：该层不进行量化感知训练。</li></ul>
</td>
</tr>
<tr id="row19584171525916"><th class="firstcol" valign="top" width="29.509999999999998%" id="mcps1.2.3.5.1"><p id="p1758417157597"><a name="p1758417157597"></a><a name="p1758417157597"></a>推荐配置</p>
</th>
<td class="cellrowborder" valign="top" width="70.49%" headers="mcps1.2.3.5.1 "><p id="p1658481519592"><a name="p1658481519592"></a><a name="p1658481519592"></a>true</p>
</td>
</tr>
<tr id="row1584161525915"><th class="firstcol" valign="top" width="29.509999999999998%" id="mcps1.2.3.6.1"><p id="p18584161525919"><a name="p18584161525919"></a><a name="p18584161525919"></a>必选或可选</p>
</th>
<td class="cellrowborder" valign="top" width="70.49%" headers="mcps1.2.3.6.1 "><p id="p158519158597"><a name="p158519158597"></a><a name="p158519158597"></a>可选</p>
</td>
</tr>
</tbody>
</table>

**表 5**  retrain\_data\_config参数说明

<a name="table295313665910"></a>
<table><tbody><tr id="row35993775914"><th class="firstcol" valign="top" width="29.759999999999998%" id="mcps1.2.3.1.1"><p id="p205933755911"><a name="p205933755911"></a><a name="p205933755911"></a>作用</p>
</th>
<td class="cellrowborder" valign="top" width="70.24000000000001%" headers="mcps1.2.3.1.1 "><p id="p1759837175919"><a name="p1759837175919"></a><a name="p1759837175919"></a>该层数据量化配置。</p>
</td>
</tr>
<tr id="row459183710598"><th class="firstcol" valign="top" width="29.759999999999998%" id="mcps1.2.3.2.1"><p id="p15943713598"><a name="p15943713598"></a><a name="p15943713598"></a>类型</p>
</th>
<td class="cellrowborder" valign="top" width="70.24000000000001%" headers="mcps1.2.3.2.1 "><p id="p359193715916"><a name="p359193715916"></a><a name="p359193715916"></a>object</p>
</td>
</tr>
<tr id="row85953716596"><th class="firstcol" valign="top" width="29.759999999999998%" id="mcps1.2.3.3.1"><p id="p205983717598"><a name="p205983717598"></a><a name="p205983717598"></a>取值范围</p>
</th>
<td class="cellrowborder" valign="top" width="70.24000000000001%" headers="mcps1.2.3.3.1 "><p id="p1959113720598"><a name="p1959113720598"></a><a name="p1959113720598"></a>-</p>
</td>
</tr>
<tr id="row45993718596"><th class="firstcol" valign="top" width="29.759999999999998%" id="mcps1.2.3.4.1"><p id="p559123715917"><a name="p559123715917"></a><a name="p559123715917"></a>参数说明</p>
</th>
<td class="cellrowborder" valign="top" width="70.24000000000001%" headers="mcps1.2.3.4.1 "><p id="p727754111012"><a name="p727754111012"></a><a name="p727754111012"></a>包含如下参数：</p>
<a name="ul68351742201012"></a><a name="ul68351742201012"></a><ul id="ul68351742201012"><li>algo：量化算法选择，默认是ulq_quantize。</li><li>clip_max：截断量化算法上限，默认不选。</li><li>clip_min：截断量化算法下限，默认不选。</li><li>fixed_min：截断量化算法最小值固定为0，默认不选。</li><li>dst_type：用以选择INT8或INT4量化位宽，默认为INT8。</li></ul>
</td>
</tr>
<tr id="row559113713595"><th class="firstcol" valign="top" width="29.759999999999998%" id="mcps1.2.3.5.1"><p id="p1859183705914"><a name="p1859183705914"></a><a name="p1859183705914"></a>推荐配置</p>
</th>
<td class="cellrowborder" valign="top" width="70.24000000000001%" headers="mcps1.2.3.5.1 "><p id="p125903705913"><a name="p125903705913"></a><a name="p125903705913"></a>-</p>
</td>
</tr>
<tr id="row859103775914"><th class="firstcol" valign="top" width="29.759999999999998%" id="mcps1.2.3.6.1"><p id="p259103711595"><a name="p259103711595"></a><a name="p259103711595"></a>必选或可选</p>
</th>
<td class="cellrowborder" valign="top" width="70.24000000000001%" headers="mcps1.2.3.6.1 "><p id="p16590373599"><a name="p16590373599"></a><a name="p16590373599"></a>可选</p>
</td>
</tr>
</tbody>
</table>

**表 6**  retrain\_weight\_config参数说明

<a name="table8436159205910"></a>
<table><tbody><tr id="row1452275913591"><th class="firstcol" valign="top" width="30.14%" id="mcps1.2.3.1.1"><p id="p45221959115918"><a name="p45221959115918"></a><a name="p45221959115918"></a>作用</p>
</th>
<td class="cellrowborder" valign="top" width="69.86%" headers="mcps1.2.3.1.1 "><p id="p1252215935915"><a name="p1252215935915"></a><a name="p1252215935915"></a>该层权重量化配置。</p>
</td>
</tr>
<tr id="row19522145965915"><th class="firstcol" valign="top" width="30.14%" id="mcps1.2.3.2.1"><p id="p1852215917591"><a name="p1852215917591"></a><a name="p1852215917591"></a>类型</p>
</th>
<td class="cellrowborder" valign="top" width="69.86%" headers="mcps1.2.3.2.1 "><p id="p1852219596595"><a name="p1852219596595"></a><a name="p1852219596595"></a>object</p>
</td>
</tr>
<tr id="row11522185911599"><th class="firstcol" valign="top" width="30.14%" id="mcps1.2.3.3.1"><p id="p145225597594"><a name="p145225597594"></a><a name="p145225597594"></a>取值范围</p>
</th>
<td class="cellrowborder" valign="top" width="69.86%" headers="mcps1.2.3.3.1 "><p id="p2522159135920"><a name="p2522159135920"></a><a name="p2522159135920"></a>-</p>
</td>
</tr>
<tr id="row205221559105919"><th class="firstcol" valign="top" width="30.14%" id="mcps1.2.3.4.1"><p id="p12522165935911"><a name="p12522165935911"></a><a name="p12522165935911"></a>参数说明</p>
</th>
<td class="cellrowborder" valign="top" width="69.86%" headers="mcps1.2.3.4.1 "><p id="p17933820151112"><a name="p17933820151112"></a><a name="p17933820151112"></a>包含如下参数：</p>
<a name="ul13748202351116"></a><a name="ul13748202351116"></a><ul id="ul13748202351116"><li>algo：量化算法选择，默认是arq_retrain</li><li>channel_wise</li></ul>
</td>
</tr>
<tr id="row20522759115911"><th class="firstcol" valign="top" width="30.14%" id="mcps1.2.3.5.1"><p id="p7522959115911"><a name="p7522959115911"></a><a name="p7522959115911"></a>推荐配置</p>
</th>
<td class="cellrowborder" valign="top" width="69.86%" headers="mcps1.2.3.5.1 "><p id="p1352265919591"><a name="p1352265919591"></a><a name="p1352265919591"></a>-</p>
</td>
</tr>
<tr id="row2522195912593"><th class="firstcol" valign="top" width="30.14%" id="mcps1.2.3.6.1"><p id="p12522155955910"><a name="p12522155955910"></a><a name="p12522155955910"></a>必选或可选</p>
</th>
<td class="cellrowborder" valign="top" width="69.86%" headers="mcps1.2.3.6.1 "><p id="p7522759105918"><a name="p7522759105918"></a><a name="p7522759105918"></a>可选</p>
</td>
</tr>
</tbody>
</table>

**表 7**  algo参数说明

<a name="table16511191903"></a>
<table><tbody><tr id="row11351719107"><th class="firstcol" valign="top" width="30.020000000000003%" id="mcps1.2.3.1.1"><p id="p81352194014"><a name="p81352194014"></a><a name="p81352194014"></a>作用</p>
</th>
<td class="cellrowborder" valign="top" width="69.98%" headers="mcps1.2.3.1.1 "><p id="p11359192003"><a name="p11359192003"></a><a name="p11359192003"></a>该层选择使用的量化算法。</p>
</td>
</tr>
<tr id="row41353191201"><th class="firstcol" valign="top" width="30.020000000000003%" id="mcps1.2.3.2.1"><p id="p51351719900"><a name="p51351719900"></a><a name="p51351719900"></a>类型</p>
</th>
<td class="cellrowborder" valign="top" width="69.98%" headers="mcps1.2.3.2.1 "><p id="p813515190015"><a name="p813515190015"></a><a name="p813515190015"></a>object</p>
</td>
</tr>
<tr id="row51357191903"><th class="firstcol" valign="top" width="30.020000000000003%" id="mcps1.2.3.3.1"><p id="p111350191400"><a name="p111350191400"></a><a name="p111350191400"></a>取值范围</p>
</th>
<td class="cellrowborder" valign="top" width="69.98%" headers="mcps1.2.3.3.1 "><p id="p11355191907"><a name="p11355191907"></a><a name="p11355191907"></a>-</p>
</td>
</tr>
<tr id="row1913611192016"><th class="firstcol" valign="top" width="30.020000000000003%" id="mcps1.2.3.4.1"><p id="p14136101917017"><a name="p14136101917017"></a><a name="p14136101917017"></a>参数说明</p>
</th>
<td class="cellrowborder" valign="top" width="69.98%" headers="mcps1.2.3.4.1 "><a name="ul12852131315124"></a><a name="ul12852131315124"></a><ul id="ul12852131315124"><li>ulq_quantize：ulq截断上下限量化算法。</li><li>arq_retrain：arq量化算法。</li></ul>
</td>
</tr>
<tr id="row813613194017"><th class="firstcol" valign="top" width="30.020000000000003%" id="mcps1.2.3.5.1"><p id="p1136219806"><a name="p1136219806"></a><a name="p1136219806"></a>推荐配置</p>
</th>
<td class="cellrowborder" valign="top" width="69.98%" headers="mcps1.2.3.5.1 "><p id="p891753512124"><a name="p891753512124"></a><a name="p891753512124"></a>数据量化使用ulq_quantize，权重量化使用arq_retrain。</p>
</td>
</tr>
<tr id="row9136719007"><th class="firstcol" valign="top" width="30.020000000000003%" id="mcps1.2.3.6.1"><p id="p2136171915018"><a name="p2136171915018"></a><a name="p2136171915018"></a>必选或可选</p>
</th>
<td class="cellrowborder" valign="top" width="69.98%" headers="mcps1.2.3.6.1 "><p id="p2136519602"><a name="p2136519602"></a><a name="p2136519602"></a>可选</p>
</td>
</tr>
</tbody>
</table>

**表 8**  channel\_wise参数说明

<a name="table19960439106"></a>
<table><tbody><tr id="row19630401702"><th class="firstcol" valign="top" width="30.330000000000002%" id="mcps1.2.3.1.1"><p id="p16630401405"><a name="p16630401405"></a><a name="p16630401405"></a>作用</p>
</th>
<td class="cellrowborder" valign="top" width="69.67%" headers="mcps1.2.3.1.1 "><p id="p18631740902"><a name="p18631740902"></a><a name="p18631740902"></a>是否对每个channel采用不同的量化因子。</p>
</td>
</tr>
<tr id="row6631403012"><th class="firstcol" valign="top" width="30.330000000000002%" id="mcps1.2.3.2.1"><p id="p2639401605"><a name="p2639401605"></a><a name="p2639401605"></a>类型</p>
</th>
<td class="cellrowborder" valign="top" width="69.67%" headers="mcps1.2.3.2.1 "><p id="p7641340708"><a name="p7641340708"></a><a name="p7641340708"></a>bool</p>
</td>
</tr>
<tr id="row176416405012"><th class="firstcol" valign="top" width="30.330000000000002%" id="mcps1.2.3.3.1"><p id="p1364154019019"><a name="p1364154019019"></a><a name="p1364154019019"></a>取值范围</p>
</th>
<td class="cellrowborder" valign="top" width="69.67%" headers="mcps1.2.3.3.1 "><p id="p06474012018"><a name="p06474012018"></a><a name="p06474012018"></a>true或false</p>
</td>
</tr>
<tr id="row11644401600"><th class="firstcol" valign="top" width="30.330000000000002%" id="mcps1.2.3.4.1"><p id="p2647406019"><a name="p2647406019"></a><a name="p2647406019"></a>参数说明</p>
</th>
<td class="cellrowborder" valign="top" width="69.67%" headers="mcps1.2.3.4.1 "><a name="ul145214136711"></a><a name="ul145214136711"></a><ul id="ul145214136711"><li>true：每个channel独立量化，量化因子不同。</li><li>false：每个channel同时量化，共享量化因子。</li></ul>
</td>
</tr>
<tr id="row5644406018"><th class="firstcol" valign="top" width="30.330000000000002%" id="mcps1.2.3.5.1"><p id="p12644401701"><a name="p12644401701"></a><a name="p12644401701"></a>推荐配置</p>
</th>
<td class="cellrowborder" valign="top" width="69.67%" headers="mcps1.2.3.5.1 "><p id="p12641040401"><a name="p12641040401"></a><a name="p12641040401"></a>true</p>
</td>
</tr>
<tr id="row9649401506"><th class="firstcol" valign="top" width="30.330000000000002%" id="mcps1.2.3.6.1"><p id="p16444012018"><a name="p16444012018"></a><a name="p16444012018"></a>必选或可选</p>
</th>
<td class="cellrowborder" valign="top" width="69.67%" headers="mcps1.2.3.6.1 "><p id="p106444019018"><a name="p106444019018"></a><a name="p106444019018"></a>可选</p>
</td>
</tr>
</tbody>
</table>

**表 9**  fixed\_min参数说明

<a name="table1459214218113"></a>
<table><tbody><tr id="row273316210110"><th class="firstcol" valign="top" width="30.4%" id="mcps1.2.3.1.1"><p id="p2734529111"><a name="p2734529111"></a><a name="p2734529111"></a>作用</p>
</th>
<td class="cellrowborder" valign="top" width="69.6%" headers="mcps1.2.3.1.1 "><p id="p9796912758"><a name="p9796912758"></a><a name="p9796912758"></a>设置数据量化算法下限的开关。</p>
</td>
</tr>
<tr id="row1573418219111"><th class="firstcol" valign="top" width="30.4%" id="mcps1.2.3.2.1"><p id="p7734122712"><a name="p7734122712"></a><a name="p7734122712"></a>类型</p>
</th>
<td class="cellrowborder" valign="top" width="69.6%" headers="mcps1.2.3.2.1 "><p id="p1573412218119"><a name="p1573412218119"></a><a name="p1573412218119"></a>bool</p>
</td>
</tr>
<tr id="row573414213113"><th class="firstcol" valign="top" width="30.4%" id="mcps1.2.3.3.1"><p id="p473414214110"><a name="p473414214110"></a><a name="p473414214110"></a>取值范围</p>
</th>
<td class="cellrowborder" valign="top" width="69.6%" headers="mcps1.2.3.3.1 "><p id="p37341321010"><a name="p37341321010"></a><a name="p37341321010"></a>true或false</p>
</td>
</tr>
<tr id="row97341522015"><th class="firstcol" valign="top" width="30.4%" id="mcps1.2.3.4.1"><p id="p177341821611"><a name="p177341821611"></a><a name="p177341821611"></a>参数说明</p>
</th>
<td class="cellrowborder" valign="top" width="69.6%" headers="mcps1.2.3.4.1 "><a name="ul2060616411812"></a><a name="ul2060616411812"></a><ul id="ul2060616411812"><li>true：数据量化算法固定下限，并且下限为0。</li><li>false：数据量化算法不固定下限。</li></ul>
<p id="p126671643181416"><a name="p126671643181416"></a><a name="p126671643181416"></a>如果不选此项，<span id="zh-cn_topic_0278900994_ph121541194518"><a name="zh-cn_topic_0278900994_ph121541194518"></a><a name="zh-cn_topic_0278900994_ph121541194518"></a>AMCT</span>根据图的结构自动设置。</p>
<p id="p1446794731520"><a name="p1446794731520"></a><a name="p1446794731520"></a>如果选择此项，并且网络模型量化层的前一层是relu层，则该参数需要手动设置为true，如果为非relu层，则要手动设置为false。</p>
</td>
</tr>
<tr id="row9734521119"><th class="firstcol" valign="top" width="30.4%" id="mcps1.2.3.5.1"><p id="p5734623111"><a name="p5734623111"></a><a name="p5734623111"></a>推荐配置</p>
</th>
<td class="cellrowborder" valign="top" width="69.6%" headers="mcps1.2.3.5.1 "><p id="p47340218119"><a name="p47340218119"></a><a name="p47340218119"></a>不选此项</p>
</td>
</tr>
<tr id="row07341224118"><th class="firstcol" valign="top" width="30.4%" id="mcps1.2.3.6.1"><p id="p173417211112"><a name="p173417211112"></a><a name="p173417211112"></a>必选或可选</p>
</th>
<td class="cellrowborder" valign="top" width="69.6%" headers="mcps1.2.3.6.1 "><p id="p15734128116"><a name="p15734128116"></a><a name="p15734128116"></a>可选</p>
</td>
</tr>
</tbody>
</table>

**表 10**  clip\_max参数说明

<a name="table15662152613113"></a>
<table><tbody><tr id="row1884192620114"><th class="firstcol" valign="top" width="30.78%" id="mcps1.2.3.1.1"><p id="p15841926411"><a name="p15841926411"></a><a name="p15841926411"></a>作用</p>
</th>
<td class="cellrowborder" valign="top" width="69.22%" headers="mcps1.2.3.1.1 "><p id="p158412268115"><a name="p158412268115"></a><a name="p158412268115"></a>数据量化算法上限。</p>
</td>
</tr>
<tr id="row20841192612115"><th class="firstcol" valign="top" width="30.78%" id="mcps1.2.3.2.1"><p id="p8841132617118"><a name="p8841132617118"></a><a name="p8841132617118"></a>类型</p>
</th>
<td class="cellrowborder" valign="top" width="69.22%" headers="mcps1.2.3.2.1 "><p id="p1841926412"><a name="p1841926412"></a><a name="p1841926412"></a>float</p>
</td>
</tr>
<tr id="row7841132616113"><th class="firstcol" valign="top" width="30.78%" id="mcps1.2.3.3.1"><p id="p98411626218"><a name="p98411626218"></a><a name="p98411626218"></a>取值范围</p>
</th>
<td class="cellrowborder" valign="top" width="69.22%" headers="mcps1.2.3.3.1 "><p id="p1628854314316"><a name="p1628854314316"></a><a name="p1628854314316"></a>clip_max&gt;0</p>
<p id="p5540113223113"><a name="p5540113223113"></a><a name="p5540113223113"></a>根据不同层activation的数据分布找到最大值max，推荐取值范围为：0.3*max~1.7*max</p>
</td>
</tr>
<tr id="row28412026112"><th class="firstcol" valign="top" width="30.78%" id="mcps1.2.3.4.1"><p id="p184116261613"><a name="p184116261613"></a><a name="p184116261613"></a>参数说明</p>
</th>
<td class="cellrowborder" valign="top" width="69.22%" headers="mcps1.2.3.4.1 "><p id="p784192610116"><a name="p784192610116"></a><a name="p784192610116"></a>截断上下限数据量化算法，如果选择此项则固定算法截断上限。如果不选此项，通过ifmr算法学习获取上限。</p>
</td>
</tr>
<tr id="row1384115264120"><th class="firstcol" valign="top" width="30.78%" id="mcps1.2.3.5.1"><p id="p88411265115"><a name="p88411265115"></a><a name="p88411265115"></a>推荐配置</p>
</th>
<td class="cellrowborder" valign="top" width="69.22%" headers="mcps1.2.3.5.1 "><p id="p188411726716"><a name="p188411726716"></a><a name="p188411726716"></a>不选此项</p>
</td>
</tr>
<tr id="row5841182617115"><th class="firstcol" valign="top" width="30.78%" id="mcps1.2.3.6.1"><p id="p10841102613112"><a name="p10841102613112"></a><a name="p10841102613112"></a>必选或可选</p>
</th>
<td class="cellrowborder" valign="top" width="69.22%" headers="mcps1.2.3.6.1 "><p id="p15842172615111"><a name="p15842172615111"></a><a name="p15842172615111"></a>可选</p>
</td>
</tr>
</tbody>
</table>

**表 11**  clip\_min参数说明

<a name="table1988120453112"></a>
<table><tbody><tr id="row159518451718"><th class="firstcol" valign="top" width="30.84%" id="mcps1.2.3.1.1"><p id="p169511945018"><a name="p169511945018"></a><a name="p169511945018"></a>作用</p>
</th>
<td class="cellrowborder" valign="top" width="69.16%" headers="mcps1.2.3.1.1 "><p id="p595117451316"><a name="p595117451316"></a><a name="p595117451316"></a>数据量化算法下限。</p>
</td>
</tr>
<tr id="row159519452115"><th class="firstcol" valign="top" width="30.84%" id="mcps1.2.3.2.1"><p id="p159527451915"><a name="p159527451915"></a><a name="p159527451915"></a>类型</p>
</th>
<td class="cellrowborder" valign="top" width="69.16%" headers="mcps1.2.3.2.1 "><p id="p25425199451"><a name="p25425199451"></a><a name="p25425199451"></a>float</p>
</td>
</tr>
<tr id="row195216451916"><th class="firstcol" valign="top" width="30.84%" id="mcps1.2.3.3.1"><p id="p295217450114"><a name="p295217450114"></a><a name="p295217450114"></a>取值范围</p>
</th>
<td class="cellrowborder" valign="top" width="69.16%" headers="mcps1.2.3.3.1 "><p id="p189461558113113"><a name="p189461558113113"></a><a name="p189461558113113"></a>clip_min&lt;0</p>
<p id="p663113717614"><a name="p663113717614"></a><a name="p663113717614"></a>根据不同层activation的数据分布找到最小值min，推荐取值范围为：0.3*min~1.7*min</p>
</td>
</tr>
<tr id="row1695244517115"><th class="firstcol" valign="top" width="30.84%" id="mcps1.2.3.4.1"><p id="p89522451612"><a name="p89522451612"></a><a name="p89522451612"></a>参数说明</p>
</th>
<td class="cellrowborder" valign="top" width="69.16%" headers="mcps1.2.3.4.1 "><p id="p1895217453114"><a name="p1895217453114"></a><a name="p1895217453114"></a>截断上下限数据量化算法，如果选择此项则固定算法截断下限。如果不选此项，通过ifmr算法学习获取下限。</p>
</td>
</tr>
<tr id="row18952164511118"><th class="firstcol" valign="top" width="30.84%" id="mcps1.2.3.5.1"><p id="p159521045517"><a name="p159521045517"></a><a name="p159521045517"></a>推荐配置</p>
</th>
<td class="cellrowborder" valign="top" width="69.16%" headers="mcps1.2.3.5.1 "><p id="p59521545716"><a name="p59521545716"></a><a name="p59521545716"></a>不选此项</p>
</td>
</tr>
<tr id="row9952245118"><th class="firstcol" valign="top" width="30.84%" id="mcps1.2.3.6.1"><p id="p49522451211"><a name="p49522451211"></a><a name="p49522451211"></a>必选或可选</p>
</th>
<td class="cellrowborder" valign="top" width="69.16%" headers="mcps1.2.3.6.1 "><p id="p395214459119"><a name="p395214459119"></a><a name="p395214459119"></a>可选</p>
</td>
</tr>
</tbody>
</table>

**表 12**  dst\_type参数说明

<a name="table139781115205"></a>
<table><tbody><tr id="row119791515102"><th class="firstcol" valign="top" width="30.84%" id="mcps1.2.3.1.1"><p id="p197912151904"><a name="p197912151904"></a><a name="p197912151904"></a>作用</p>
</th>
<td class="cellrowborder" valign="top" width="69.16%" headers="mcps1.2.3.1.1 "><p id="p1797911155015"><a name="p1797911155015"></a><a name="p1797911155015"></a>量化类型。</p>
</td>
</tr>
<tr id="row197913151001"><th class="firstcol" valign="top" width="30.84%" id="mcps1.2.3.2.1"><p id="p15980161511011"><a name="p15980161511011"></a><a name="p15980161511011"></a>类型</p>
</th>
<td class="cellrowborder" valign="top" width="69.16%" headers="mcps1.2.3.2.1 "><p id="p18980315006"><a name="p18980315006"></a><a name="p18980315006"></a>string</p>
</td>
</tr>
<tr id="row159801151604"><th class="firstcol" valign="top" width="30.84%" id="mcps1.2.3.3.1"><p id="p129801715706"><a name="p129801715706"></a><a name="p129801715706"></a>取值范围</p>
</th>
<td class="cellrowborder" valign="top" width="69.16%" headers="mcps1.2.3.3.1 "><p id="p1684619615"><a name="p1684619615"></a><a name="p1684619615"></a>INT8或INT4，默认为INT8。当前版本仅支持INT8。</p>
</td>
</tr>
<tr id="row179807151805"><th class="firstcol" valign="top" width="30.84%" id="mcps1.2.3.4.1"><p id="p169806156011"><a name="p169806156011"></a><a name="p169806156011"></a>参数说明</p>
</th>
<td class="cellrowborder" valign="top" width="69.16%" headers="mcps1.2.3.4.1 "><p id="p16847145315215"><a name="p16847145315215"></a><a name="p16847145315215"></a>量化时用于选择是INT8量化还是INT4量化。</p>
</td>
</tr>
<tr id="row1898112151405"><th class="firstcol" valign="top" width="30.84%" id="mcps1.2.3.5.1"><p id="p49812015801"><a name="p49812015801"></a><a name="p49812015801"></a>推荐配置</p>
</th>
<td class="cellrowborder" valign="top" width="69.16%" headers="mcps1.2.3.5.1 "><p id="p398118151606"><a name="p398118151606"></a><a name="p398118151606"></a>-</p>
</td>
</tr>
<tr id="row09810151202"><th class="firstcol" valign="top" width="30.84%" id="mcps1.2.3.6.1"><p id="p69811115109"><a name="p69811115109"></a><a name="p69811115109"></a>必选或可选</p>
</th>
<td class="cellrowborder" valign="top" width="69.16%" headers="mcps1.2.3.6.1 "><p id="p89821515602"><a name="p89821515602"></a><a name="p89821515602"></a>可选</p>
</td>
</tr>
</tbody>
</table>

