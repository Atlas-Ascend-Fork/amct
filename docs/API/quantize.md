# quantize

## 产品支持情况

| 产品                                        | 是否支持 |
| ------------------------------------------- | -------- |
| Ascend 950PR/Ascend 950DT                   | √        |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √        |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √        |


## 功能说明

基于torch module的训练后量化接口，高精度模型转换为校准模型，得到量化校准模型，推理后计算得到量化参数。

## 函数原型

```python
quantize(model, config)
```

## 参数说明

<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="8.76%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.05%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240187365_p1769255516412"><a name="zh-cn_topic_0240187365_p1769255516412"></a><a name="zh-cn_topic_0240187365_p1769255516412"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="81.19%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0240187365_p15231205416325"><a name="zh-cn_topic_0240187365_p15231205416325"></a><a name="zh-cn_topic_0240187365_p15231205416325"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row550615345319"><td class="cellrowborder" valign="top" width="8.76%" headers="mcps1.1.4.1.1 "><p id="p6506175375310"><a name="p6506175375310"></a><a name="p6506175375310"></a>model</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.1.4.1.2 "><p id="p150610539536"><a name="p150610539536"></a><a name="p150610539536"></a>输入/输出</p>
</td>
<td class="cellrowborder" valign="top" width="81.19%" headers="mcps1.1.4.1.3 "><p id="p79841942143911"><a name="p79841942143911"></a><a name="p79841942143911"></a>输入含义：待量化的高精度模型</p>
<p id="p1537184463920"><a name="p1537184463920"></a><a name="p1537184463920"></a>输出含义：量化校准模型</p>
<p id="p1967934093915"><a name="p1967934093915"></a><a name="p1967934093915"></a>数据类型：torch.nn.Module</p>
</td>
</tr>
<tr id="row8672528631"><td class="cellrowborder" valign="top" width="8.76%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0240188739_p12612155914458"><a name="zh-cn_topic_0240188739_p12612155914458"></a><a name="zh-cn_topic_0240188739_p12612155914458"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0240188739_p46121659184520"><a name="zh-cn_topic_0240188739_p46121659184520"></a><a name="zh-cn_topic_0240188739_p46121659184520"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="81.19%" headers="mcps1.1.4.1.3 "><p id="p129452810402"><a name="p129452810402"></a><a name="p129452810402"></a><span>含义：量化配置。</span></p>
<p id="p334514515403"><a name="p334514515403"></a><a name="p334514515403"></a><span>数据类型：自定义dict，其中包含weight/input/algorithm/skip_layers的配置，详细配置参数请参见</span><a href="https://gitcode.com/cann/amct/blob/master/docs/API/quantize.md#config详细配置" target="_blank" rel="noopener noreferrer">config详细配置</a><span>。</span></p>
</td>
</tr>
</tbody>
</table>


## 全量化支持的层<a name="section1326262414717"></a>

原始模型中数据类型为float32（fp32）、float16（fp16）、bfloat16（bf16）时，可以通过本节介绍的内容，量化后转换为HiFloat8（HiF8）、float8（fp8）、MXFP4数据格式，通过对数据格式的压缩，实现模型轻量化。

该特性支持的层如下。

**表 1**  支持量化的层以及约束

<a name="table32131022111318"></a>

<table><thead align="left"><tr id="row9214202215138"><th class="cellrowborder" valign="top" width="10.63%" id="mcps1.2.5.1.1"><p id="p1921472241319"><a name="p1921472241319"></a><a name="p1921472241319"></a>支持的层类型</p>
</th>
<th class="cellrowborder" valign="top" width="20.09%" id="mcps1.2.5.1.2"><p id="p27472162329"><a name="p27472162329"></a><a name="p27472162329"></a>原始数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="32.01%" id="mcps1.2.5.1.3"><p id="p10685452175710"><a name="p10685452175710"></a><a name="p10685452175710"></a>支持的量化类型组合</p>
</th>
<th class="cellrowborder" valign="top" width="37.269999999999996%" id="mcps1.2.5.1.4"><p id="p122141822101318"><a name="p122141822101318"></a><a name="p122141822101318"></a>约束</p>
</th>
</tr>
</thead>
<tbody><tr id="row7214182271318"><td class="cellrowborder" rowspan="5" valign="top" width="10.63%" headers="mcps1.2.5.1.1 "><p id="p32146222139"><a name="p32146222139"></a><a name="p32146222139"></a>torch.nn.Linear</p>
<p id="p20540182234618"><a name="p20540182234618"></a><a name="p20540182234618"></a></p>
<p id="p85010423713"><a name="p85010423713"></a><a name="p85010423713"></a></p>
</td>
<td class="cellrowborder" valign="top" width="20.09%" headers="mcps1.2.5.1.2 "><p id="p77471416113220"><a name="p77471416113220"></a><a name="p77471416113220"></a>float32（fp32）、float16（fp16）、bfloat16（bf16）</p>
</td>
<td class="cellrowborder" valign="top" width="32.01%" headers="mcps1.2.5.1.3 "><p id="p641017541626"><a name="p641017541626"></a><a name="p641017541626"></a>act_type: HIFLOAT8  wts_type: HIFLOAT8</p>
<p id="p6410054727"><a name="p6410054727"></a><a name="p6410054727"></a>act_type: FLOAT8_E4M3FN  wts_type: FLOAT8_E4M3FN</p>
</td>
<td class="cellrowborder" valign="top" width="37.269999999999996%" headers="mcps1.2.5.1.4 "><p id="p187277572518"><a name="p187277572518"></a><a name="p187277572518"></a>支持PER_TENSOR/PER_CHANNEL量化</p>
<p id="p1894815554584"><a name="p1894815554584"></a><a name="p1894815554584"></a>量化算法为OFMR</p>
</td>
</tr>
<tr id="row3281122665615"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p1711435320187"><a name="p1711435320187"></a><a name="p1711435320187"></a>bfloat16（bf16）</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p172641614579"><a name="p172641614579"></a><a name="p172641614579"></a>act_type: MXFP8_E4M3FN  wts_type: MXFP8_E4M3FN</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p142815266561"><a name="p142815266561"></a><a name="p142815266561"></a>支持2~6维数据输入、PER_GROUP量化、舍入模式为RINT、支持对称量化、cin长度除以32向上取整后是2的整数倍</p>
<p id="p6287164525916"><a name="p6287164525916"></a><a name="p6287164525916"></a>不使用量化算法</p>
</td>
</tr>
<tr id="row12826122110328"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p1827112123214"><a name="p1827112123214"></a><a name="p1827112123214"></a>bfloat16（bf16）</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p11827132111327"><a name="p11827132111327"></a><a name="p11827132111327"></a>act_type: MXFP8_E4M3FN  wts_type: MXFP4_E2M1</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p485761119358"><a name="p485761119358"></a><a name="p485761119358"></a>支持2~6维数据输入、输入数据cin是64整数倍、</p>
<p id="p16857141117358"><a name="p16857141117358"></a><a name="p16857141117358"></a>PER_GROUP量化、舍入模式为RINT、支持对称量化</p>
<p id="p432465775920"><a name="p432465775920"></a><a name="p432465775920"></a>不使用量化算法</p>
</td>
</tr>
<tr id="row65391422154618"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p145409227465"><a name="p145409227465"></a><a name="p145409227465"></a>bfloat16（bf16）</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p16851142184618"><a name="p16851142184618"></a><a name="p16851142184618"></a>act_type: FLOAT8_E4M3FN  wts_type: FLOAT4_E2M1</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p1183219292620"><a name="p1183219292620"></a><a name="p1183219292620"></a>支持2~6维数据输入、舍入模式为RINT、bias为false</p>
<p id="p103822021171315"><a name="p103822021171315"></a><a name="p103822021171315"></a>激活（数据）支持shape为(m,k)，权重支持shape为(n,k)，其中k是64整数倍</p>
<p id="p12540182214619"><a name="p12540182214619"></a><a name="p12540182214619"></a>激活（数据）支持PER_TENSOR量化，权重支持PER_GROUP量化</p>
<p id="p844695365417"><a name="p844695365417"></a><a name="p844695365417"></a>激活和权重都仅支持对称量化</p>
<p id="p1877817173019"><a name="p1877817173019"></a><a name="p1877817173019"></a>支持MIN-MAX量化算法，SmoothQuant算法（<a href="quantize.md#section1536112219183">config详细配置</a>中必须配置<span>smoothquant</span>选项）</p>
</td>
</tr>
<tr id="row9500144213717"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p175010422712"><a name="p175010422712"></a><a name="p175010422712"></a>float16（fp16）、bfloat16（bf16）</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p650118423714"><a name="p650118423714"></a><a name="p650118423714"></a>act_type: INT8  wts_type: INT8</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><a name="ul178665278918"></a><a name="ul178665278918"></a><ul id="ul178665278918"><li>支持2~6维数据输入，舍入模式为RINT</li><li>激活（数据）支持PER_TENSOR量化，支持对称非对称量化，bias量化为INT32；权重支持PER_TENSOR/PER_CHANNEL量化，支持对称量化</li><li>激活（数据）支持PER-TOKEN对称量化，bias不量化，k是16的倍数，n是8的倍数；权重支持PER_TENSOR/PER_CHANNEL对称量化</li><li>支持MIN-MAX量化算法、SmoothQuant算法（<a href="quantize.md#section1536112219183">config详细配置</a>中必须配置<span>smoothquant</span>选项）</li></ul>
</td>
</tr>
<tr id="row1521462251317"><td class="cellrowborder" valign="top" width="10.63%" headers="mcps1.2.5.1.1 "><p id="p1021414226138"><a name="p1021414226138"></a><a name="p1021414226138"></a>torch.nn.Conv2d</p>
</td>
<td class="cellrowborder" valign="top" width="20.09%" headers="mcps1.2.5.1.2 "><p id="p12999111203315"><a name="p12999111203315"></a><a name="p12999111203315"></a>float32（fp32）、float16（fp16）、bfloat16（bf16）</p>
</td>
<td class="cellrowborder" valign="top" width="32.01%" headers="mcps1.2.5.1.3 "><p id="p174971639"><a name="p174971639"></a><a name="p174971639"></a>act_type: HIFLOAT8  wts_type: HIFLOAT8</p>
<p id="p14749418315"><a name="p14749418315"></a><a name="p14749418315"></a>act_type: FLOAT8_E4M3FN  wts_type: FLOAT8_E4M3FN</p>
</td>
<td class="cellrowborder" valign="top" width="37.269999999999996%" headers="mcps1.2.5.1.4 "><p id="p1885141034514"><a name="p1885141034514"></a><a name="p1885141034514"></a>padding_mode为zeros、支持PER_TENSOR/PER_CHANNEL量化</p>
<p id="p1274062255916"><a name="p1274062255916"></a><a name="p1274062255916"></a>量化算法为OFMR</p>
</td>
</tr>
</tbody>
</table>

注：act\_type和wts\_type参数分别指[config详细配置](quantize.md#section1536112219183)中的数据（activation）和权重（weight）量化类型。

## 仅权重量化支持的层<a name="section831519712409"></a>

本章节介绍仅权重量化特性，以及对应的优化算法：AWQ、GPTQ等权重量化算法。

原始模型中数据类型为float16（fp16）、bfloat16（bf16）时，可以通过本节介绍的内容，仅权重量化后转换为HIFloat8（HiF8）、float8（fp8）、MXFP4数据格式，通过对权重的压缩，实现模型轻量化。

该特性支持的层如下：

**表 2**  支持量化的层以及约束

<a name="table1116172344016"></a>

<table><thead align="left"><tr id="row1016142314404"><th class="cellrowborder" valign="top" width="11.99%" id="mcps1.2.5.1.1"><p id="p516023164015"><a name="p516023164015"></a><a name="p516023164015"></a>支持的层类型</p>
</th>
<th class="cellrowborder" valign="top" width="18.279999999999998%" id="mcps1.2.5.1.2"><p id="p2161723194015"><a name="p2161723194015"></a><a name="p2161723194015"></a>原始数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="22%" id="mcps1.2.5.1.3"><p id="p16168230408"><a name="p16168230408"></a><a name="p16168230408"></a>支持的量化类型组合</p>
</th>
<th class="cellrowborder" valign="top" width="47.73%" id="mcps1.2.5.1.4"><p id="p816152364010"><a name="p816152364010"></a><a name="p816152364010"></a>约束</p>
</th>
</tr>
</thead>
<tbody><tr id="row177511228786"><td class="cellrowborder" rowspan="5" valign="top" width="11.99%" headers="mcps1.2.5.1.1 "><p id="p5390433491"><a name="p5390433491"></a><a name="p5390433491"></a>torch.nn.Linear</p>
<p id="p321913119411"><a name="p321913119411"></a><a name="p321913119411"></a></p>
<p id="p3841056175716"><a name="p3841056175716"></a><a name="p3841056175716"></a></p>
</td>
<td class="cellrowborder" rowspan="4" valign="top" width="18.279999999999998%" headers="mcps1.2.5.1.2 "><p id="p475117281482"><a name="p475117281482"></a><a name="p475117281482"></a>float16（fp16）、bfloat16（bf16）</p>
<p id="p1321951541"><a name="p1321951541"></a><a name="p1321951541"></a></p>
</td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.2.5.1.3 "><p id="p1032617312478"><a name="p1032617312478"></a><a name="p1032617312478"></a>wts_type: HIFLOAT8</p>
<p id="p1978123313505"><a name="p1978123313505"></a><a name="p1978123313505"></a>wts_type: FLOAT8_E4M3FN</p>
</td>
<td class="cellrowborder" valign="top" width="47.73%" headers="mcps1.2.5.1.4 "><a name="ul1681123611612"></a><a name="ul1681123611612"></a><ul id="ul1681123611612"><li>支持PER_TENSOR/PER_CHANNEL量化，支持对称量化</li><li>支持2~6维数据输入</li><li>支持MIN-MAX量化算法、GPTQ量化算法（<a href="quantize.md#section1536112219183">config详细配置</a>中必须配置<span>gptq</span>选项）</li></ul>
</td>
</tr>
<tr id="row1326719415381"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p278010227380"><a name="p278010227380"></a><a name="p278010227380"></a>wts_type: MXFP4_E2M1</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><a name="ul9303114315395"></a><a name="ul9303114315395"></a><ul id="ul9303114315395"><li>支持PER_GROUP量化，舍入模式为RINT，支持对称量化</li><li>支持2~6维数据输入</li><li>支持MIN-MAX量化算法、GPTQ量化算法（<a href="quantize.md#section1536112219183">config详细配置</a>中必须配置<span>gptq</span>选项）、AWQ量化算法（<a href="quantize.md#section1536112219183">config详细配置</a>中必须配置<span>awq</span>选项）</li></ul>
</td>
</tr>
<tr id="row131281133204815"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p1812833324819"><a name="p1812833324819"></a><a name="p1812833324819"></a>wts_type: FLOAT4_E2M1或FLOAT4_E1M2</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><a name="ul185841815118"></a><a name="ul185841815118"></a><ul id="ul185841815118"><li>支持PER_GROUP量化，舍入模式为RINT，支持对称量化</li><li>支持2~6维数据输入</li><li>支持MIN-MAX量化算法、GPTQ量化算法（<a href="quantize.md#section1536112219183">config详细配置</a>中必须配置<span>gptq</span>选项）、AWQ量化算法（<a href="quantize.md#section1536112219183">config详细配置</a>中必须配置<span>awq</span>选项）</li></ul>
</td>
</tr>
<tr id="row1219111249"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p20219811048"><a name="p20219811048"></a><a name="p20219811048"></a>wts_type: INT8</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><a name="ul06850346713"></a><a name="ul06850346713"></a><ul id="ul06850346713"><li>支持PER_TENSOR/PER_CHANNEL/PER_GROUP量化，舍入模式为RINT，支持对称量化/非对称量化</li><li>支持MIN-MAX量化算法、GPTQ量化算法（<a href="quantize.md#section1536112219183">config详细配置</a>中必须配置<span>gptq</span>选项）、AWQ量化算法（<a href="quantize.md#section1536112219183">config详细配置</a>中必须配置<span>awq</span>选项）</li><li>wts_type为INT8时，原始模型weight需要K,N轴32元素对齐；wts_type为INT4时，需要K,N轴64元素对齐</li></ul>
</td>
</tr>
<tr id="row1184114561578"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p284185625716"><a name="p284185625716"></a><a name="p284185625716"></a>float32（fp32）、float16（fp16）、bfloat16（bf16）</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p58415569572"><a name="p58415569572"></a><a name="p58415569572"></a>wts_type: INT4</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><a name="ul15592626145816"></a><a name="ul15592626145816"></a><ul id="ul15592626145816"><li>支持PER_TENSOR/PER_CHANNEL/PER_GROUP量化，舍入模式为RINT，支持对称量化/非对称量化</li><li>支持MIN-MAX量化算法、GPTQ量化算法（<a href="quantize.md#section1536112219183">config详细配置</a>中必须配置<span>gptq</span>选项）、AWQ量化算法（<a href="quantize.md#section1536112219183">config详细配置</a>中必须配置<span>awq</span>选项）</li><li>wts_type为INT8时，原始模型weight需要K,N轴32元素对齐；wts_type为INT4时，需要K,N轴64元素对齐</li></ul>
</td>
</tr>
</tbody>
</table>



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

<a name="table9577255181816"></a>
<table><thead align="left"><tr id="row176031155111812"><th class="cellrowborder" valign="top" width="11.04%" id="mcps1.1.5.1.1"><p id="p86039551189"><a name="p86039551189"></a><a name="p86039551189"></a>key</p>
</th>
<th class="cellrowborder" valign="top" width="8.04%" id="mcps1.1.5.1.2"><p id="p17603135519185"><a name="p17603135519185"></a><a name="p17603135519185"></a>-</p>
</th>
<th class="cellrowborder" valign="top" width="11.07%" id="mcps1.1.5.1.3"><p id="p460385519182"><a name="p460385519182"></a><a name="p460385519182"></a>-</p>
</th>
<th class="cellrowborder" valign="top" width="69.85%" id="mcps1.1.5.1.4"><p id="p1660318559188"><a name="p1660318559188"></a><a name="p1660318559188"></a>value</p>
</th>
</tr>
</thead>
<tbody><tr id="row19603115516188"><td class="cellrowborder" valign="top" width="11.04%" headers="mcps1.1.5.1.1 "><p id="p1960345551813"><a name="p1960345551813"></a><a name="p1960345551813"></a>batch_num</p>
</td>
<td class="cellrowborder" valign="top" width="8.04%" headers="mcps1.1.5.1.2 "><p id="p1660319551189"><a name="p1660319551189"></a><a name="p1660319551189"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="11.07%" headers="mcps1.1.5.1.3 "><p id="p2603195513180"><a name="p2603195513180"></a><a name="p2603195513180"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="69.85%" headers="mcps1.1.5.1.4 "><p id="p36030559182"><a name="p36030559182"></a><a name="p36030559182"></a>uint32类型，量化使用的batch数量。</p>
</td>
</tr>
<tr id="row160365541815"><td class="cellrowborder" valign="top" width="11.04%" headers="mcps1.1.5.1.1 "><p id="p36031055101817"><a name="p36031055101817"></a><a name="p36031055101817"></a>quant_cfg</p>
</td>
<td class="cellrowborder" valign="top" width="8.04%" headers="mcps1.1.5.1.2 "><p id="p06031555181817"><a name="p06031555181817"></a><a name="p06031555181817"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="11.07%" headers="mcps1.1.5.1.3 "><p id="p1260345520188"><a name="p1260345520188"></a><a name="p1260345520188"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="69.85%" headers="mcps1.1.5.1.4 "><p id="p17603125516185"><a name="p17603125516185"></a><a name="p17603125516185"></a>量化配置。</p>
</td>
</tr>
<tr id="row760395512182"><td class="cellrowborder" valign="top" width="11.04%" headers="mcps1.1.5.1.1 "><p id="p860312558188"><a name="p860312558188"></a><a name="p860312558188"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="8.04%" headers="mcps1.1.5.1.2 "><p id="p176038551185"><a name="p176038551185"></a><a name="p176038551185"></a>weights</p>
</td>
<td class="cellrowborder" valign="top" width="11.07%" headers="mcps1.1.5.1.3 "><p id="p560365517181"><a name="p560365517181"></a><a name="p560365517181"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="69.85%" headers="mcps1.1.5.1.4 "><p id="p14603755171815"><a name="p14603755171815"></a><a name="p14603755171815"></a>仅权重量化配置。</p>
</td>
</tr>
<tr id="row16031555181820"><td class="cellrowborder" valign="top" width="11.04%" headers="mcps1.1.5.1.1 "><p id="p156031955201812"><a name="p156031955201812"></a><a name="p156031955201812"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="8.04%" headers="mcps1.1.5.1.2 "><p id="p1260315520181"><a name="p1260315520181"></a><a name="p1260315520181"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="11.07%" headers="mcps1.1.5.1.3 "><p id="p1860325561813"><a name="p1860325561813"></a><a name="p1860325561813"></a>type</p>
</td>
<td class="cellrowborder" valign="top" width="69.85%" headers="mcps1.1.5.1.4 "><p id="p15462475258"><a name="p15462475258"></a><a name="p15462475258"></a>string类型，权重（weight）量化粒度。当前支持如下类型：</p>
<a name="ul1911188102519"></a><a name="ul1911188102519"></a><ul id="ul1911188102519"><li>hifloat8</li><li>float8_e4m3fn</li><li>mxfp4_e2m1</li><li>float4_e2m1</li><li>int4</li><li>int8</li></ul>
</td>
</tr>
<tr id="row196031155121813"><td class="cellrowborder" valign="top" width="11.04%" headers="mcps1.1.5.1.1 "><p id="p060355517182"><a name="p060355517182"></a><a name="p060355517182"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="8.04%" headers="mcps1.1.5.1.2 "><p id="p116031855111811"><a name="p116031855111811"></a><a name="p116031855111811"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="11.07%" headers="mcps1.1.5.1.3 "><p id="p56031855131819"><a name="p56031855131819"></a><a name="p56031855131819"></a>symmetric</p>
</td>
<td class="cellrowborder" valign="top" width="69.85%" headers="mcps1.1.5.1.4 "><p id="p860385513186"><a name="p860385513186"></a><a name="p860385513186"></a>bool类型，权重是否为对称量化。</p>
<a name="ul16491192891913"></a><a name="ul16491192891913"></a><ul id="ul16491192891913"><li>True：对称量化。</li><li>False：非对称量化。</li></ul>
</td>
</tr>
<tr id="row060315581820"><td class="cellrowborder" valign="top" width="11.04%" headers="mcps1.1.5.1.1 "><p id="p860385571813"><a name="p860385571813"></a><a name="p860385571813"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="8.04%" headers="mcps1.1.5.1.2 "><p id="p060345519188"><a name="p060345519188"></a><a name="p060345519188"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="11.07%" headers="mcps1.1.5.1.3 "><p id="p460405511816"><a name="p460405511816"></a><a name="p460405511816"></a>strategy</p>
</td>
<td class="cellrowborder" valign="top" width="69.85%" headers="mcps1.1.5.1.4 "><p id="p13604125511814"><a name="p13604125511814"></a><a name="p13604125511814"></a>string类型，权重量化粒度。</p>
<a name="ul8259163310191"></a><a name="ul8259163310191"></a><ul id="ul8259163310191"><li>tensor，对应per-tensor。</li><li>channel，对应per-channel。</li><li>group，对应per-group。</li></ul>
<p id="p10604115511810"><a name="p10604115511810"></a><a name="p10604115511810"></a>量化粒度介绍请参见<a href="https://gitcode.com/cann/amct/blob/master/docs/量化概念.md" target="_blank" rel="noopener noreferrer">量化基础介绍</a>。</p>
</td>
</tr>
<tr id="row1460485515186"><td class="cellrowborder" valign="top" width="11.04%" headers="mcps1.1.5.1.1 "><p id="p1260495514181"><a name="p1260495514181"></a><a name="p1260495514181"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="8.04%" headers="mcps1.1.5.1.2 "><p id="p86047556183"><a name="p86047556183"></a><a name="p86047556183"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="11.07%" headers="mcps1.1.5.1.3 "><p id="p17604125519187"><a name="p17604125519187"></a><a name="p17604125519187"></a>group_size</p>
</td>
<td class="cellrowborder" valign="top" width="69.85%" headers="mcps1.1.5.1.4 "><p id="p1060465519185"><a name="p1060465519185"></a><a name="p1060465519185"></a>仅权重量化场景配置，per-group量化粒度下group的大小，该参数只有配置了per-group后，才能配置。</p>
<p id="p360416554186"><a name="p360416554186"></a><a name="p360416554186"></a>要求传入值的范围为[32, K-1]且必须是32的倍数。</p>
</td>
</tr>
<tr id="row206041655121811"><td class="cellrowborder" valign="top" width="11.04%" headers="mcps1.1.5.1.1 "><p id="p196041557187"><a name="p196041557187"></a><a name="p196041557187"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="8.04%" headers="mcps1.1.5.1.2 "><p id="p860445513180"><a name="p860445513180"></a><a name="p860445513180"></a>inputs</p>
</td>
<td class="cellrowborder" valign="top" width="11.07%" headers="mcps1.1.5.1.3 "><p id="p1560435510182"><a name="p1560435510182"></a><a name="p1560435510182"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="69.85%" headers="mcps1.1.5.1.4 "><p id="p196043556186"><a name="p196043556186"></a><a name="p196043556186"></a>数据量化配置。</p>
</td>
</tr>
<tr id="row13604855101811"><td class="cellrowborder" valign="top" width="11.04%" headers="mcps1.1.5.1.1 "><p id="p76046553183"><a name="p76046553183"></a><a name="p76046553183"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="8.04%" headers="mcps1.1.5.1.2 "><p id="p7604165581818"><a name="p7604165581818"></a><a name="p7604165581818"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="11.07%" headers="mcps1.1.5.1.3 "><p id="p1260495516188"><a name="p1260495516188"></a><a name="p1260495516188"></a>type</p>
</td>
<td class="cellrowborder" valign="top" width="69.85%" headers="mcps1.1.5.1.4 "><p id="p188704582212"><a name="p188704582212"></a><a name="p188704582212"></a>string类型，数据（activation）量化粒度。目前支持如下类型：</p>
<a name="ul1560524672215"></a><a name="ul1560524672215"></a><ul id="ul1560524672215"><li>hifloat8</li><li>float8_e4m3fn</li><li>mxfp8_e4m3fn</li><li><span>int8</span></li></ul>
</td>
</tr>
<tr id="row760415581817"><td class="cellrowborder" valign="top" width="11.04%" headers="mcps1.1.5.1.1 "><p id="p1260435521812"><a name="p1260435521812"></a><a name="p1260435521812"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="8.04%" headers="mcps1.1.5.1.2 "><p id="p760485516184"><a name="p760485516184"></a><a name="p760485516184"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="11.07%" headers="mcps1.1.5.1.3 "><p id="p11604165516183"><a name="p11604165516183"></a><a name="p11604165516183"></a>symmetric</p>
</td>
<td class="cellrowborder" valign="top" width="69.85%" headers="mcps1.1.5.1.4 "><p id="p46041355111814"><a name="p46041355111814"></a><a name="p46041355111814"></a>bool类型，数据是否为对称量化。</p>
<a name="ul2313916200"></a><a name="ul2313916200"></a><ul id="ul2313916200"><li>True：对称量化。</li><li>False：非对称量化。</li></ul>
</td>
</tr>
<tr id="row136046556186"><td class="cellrowborder" valign="top" width="11.04%" headers="mcps1.1.5.1.1 "><p id="p36041255101819"><a name="p36041255101819"></a><a name="p36041255101819"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="8.04%" headers="mcps1.1.5.1.2 "><p id="p860445516186"><a name="p860445516186"></a><a name="p860445516186"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="11.07%" headers="mcps1.1.5.1.3 "><p id="p1960411556189"><a name="p1960411556189"></a><a name="p1960411556189"></a>strategy</p>
</td>
<td class="cellrowborder" valign="top" width="69.85%" headers="mcps1.1.5.1.4 "><p id="p5604355171813"><a name="p5604355171813"></a><a name="p5604355171813"></a>string类型，数据量化粒度。</p>
<a name="ul7111191082014"></a><a name="ul7111191082014"></a><ul id="ul7111191082014"><li>tensor，对应per-tensor。</li><li>token，对应per-token。</li></ul>
</td>
</tr>
<tr id="row1160485581816"><td class="cellrowborder" valign="top" width="11.04%" headers="mcps1.1.5.1.1 "><p id="p6604455151820"><a name="p6604455151820"></a><a name="p6604455151820"></a>algorithm</p>
</td>
<td class="cellrowborder" valign="top" width="8.04%" headers="mcps1.1.5.1.2 "><p id="p2060425511813"><a name="p2060425511813"></a><a name="p2060425511813"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="11.07%" headers="mcps1.1.5.1.3 "><p id="p1060425541813"><a name="p1060425541813"></a><a name="p1060425541813"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="69.85%" headers="mcps1.1.5.1.4 "><p id="p66040553184"><a name="p66040553184"></a><a name="p66040553184"></a>string类型，量化算法，支持如下配置：</p>
<a name="ul2808111772013"></a><a name="ul2808111772013"></a><ul id="ul2808111772013"><li>awq：grids_num，uint32类型，搜索格点数量。AWQ算法求解量化参数的过程中，对候选值做网格划分，grids_num越大，搜索粒度越大，量化误差越小，但计算耗时增加。默认为20。</li><li>gptq。</li><li>minmax。</li><li>smoothquant：smooth_strength，float类型，迁移强度，代表将activation数据上的量化难度迁移至weight权重的程度。默认值0.5，数据分布的离群值越大迁移强度应设置较小。</li></ul>
<p id="p4604195517184"><a name="p4604195517184"></a><a name="p4604195517184"></a>具体请参见<a href="../算法介绍.md" target="_blank" rel="noopener noreferrer">量化算法介绍</a>。</p>
</td>
</tr>
<tr id="row1060418558184"><td class="cellrowborder" valign="top" width="11.04%" headers="mcps1.1.5.1.1 "><p id="p9604135541813"><a name="p9604135541813"></a><a name="p9604135541813"></a>skip_layers</p>
</td>
<td class="cellrowborder" valign="top" width="8.04%" headers="mcps1.1.5.1.2 "><p id="p10604145516182"><a name="p10604145516182"></a><a name="p10604145516182"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="11.07%" headers="mcps1.1.5.1.3 "><p id="p1260435581811"><a name="p1260435581811"></a><a name="p1260435581811"></a>-</p>
</td>
<td class="cellrowborder" valign="top" width="69.85%" headers="mcps1.1.5.1.4 "><p id="p10604355191815"><a name="p10604355191815"></a><a name="p10604355191815"></a>string类型，按层名跳过哪些层不做量化，全局配置参数。指定层名后，只要层名包括用户设置的字符串，就跳过该层不做量化。</p>
</td>
</tr>
</tbody>
</table>


