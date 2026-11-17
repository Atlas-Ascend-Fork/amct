# quantize

<!-- md-trans-meta sourceCommit=2fb2b85b3f6f8ef42c259dcaec4ffe19306dc78d translatedAt=2026-10-16T08:33:44.533Z pushedAt=2026-11-05T01:12:03.424Z nl=2 -->

## Supported Products

| Product                                        | Supported |
| ------------------------------------------- | -------- |
| Ascend 950PR&950DT products                   | √        |
| Atlas A3 products | √        |
| Atlas A2 products | √        |

## Description

A post-training quantization API based on torch module. It converts a high-precision model into a calibration model to obtain a quantized calibration model, and computes quantization parameters after inference.

## Prototype

```python
quantize(model, config)
```

## Parameters

<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="8.76%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a>Parameter</p></th>
<th class="cellrowborder" valign="top" width="10.05%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240187365_p1769255516412"><a name="zh-cn_topic_0240187365_p1769255516412"></a><a name="zh-cn_topic_0240187365_p1769255516412"></a>Input/Output</p></th>
<th class="cellrowborder" valign="top" width="81.19%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0240187365_p15231205416325"><a name="zh-cn_topic_0240187365_p15231205416325"></a><a name="zh-cn_topic_0240187365_p15231205416325"></a>Description</p></th>
</tr>
</thead>
<tbody><tr id="row550615345319"><td class="cellrowborder" valign="top" width="8.76%" headers="mcps1.1.4.1.1 "><p id="p6506175375310"><a name="p6506175375310"></a><a name="p6506175375310"></a>model</p></td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.1.4.1.2 "><p id="p150610539536"><a name="p150610539536"></a><a name="p150610539536"></a>Input/Output</p></td>
<td class="cellrowborder" valign="top" width="81.19%" headers="mcps1.1.4.1.3 "><p id="p79841942143911"><a name="p79841942143911"></a><a name="p79841942143911"></a>Input meaning: high-precision model to be quantized.</p>
<p id="p1537184463920"><a name="p1537184463920"></a><a name="p1537184463920"></a>Output meaning: quantized calibration model.</p>
<p id="p1967934093915"><a name="p1967934093915"></a><a name="p1967934093915"></a>Data type: torch.nn.Module.</p></td>
</tr>
<tr id="row8672528631"><td class="cellrowborder" valign="top" width="8.76%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0240188739_p12612155914458"><a name="zh-cn_topic_0240188739_p12612155914458"></a><a name="zh-cn_topic_0240188739_p12612155914458"></a>config</p></td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0240188739_p46121659184520"><a name="zh-cn_topic_0240188739_p46121659184520"></a><a name="zh-cn_topic_0240188739_p46121659184520"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="81.19%" headers="mcps1.1.4.1.3 "><p id="p129452810402"><a name="p129452810402"></a><a name="p129452810402"></a><span>Meaning: Quantization configuration.</span></p>
<p id="p334514515403"><a name="p334514515403"></a><a name="p334514515403"></a><span>Data type: Custom dict, containing the configuration of weight/input/algorithm/skip_layers. For config details parameters, see </span><a href="#section1536112219183" target="_blank" rel="noopener noreferrer">config details</a><span>.</span></p></td>
</tr>
</tbody>
</table>

## Layers Supported by Full Quantization<a name="section1326262414717"></a>

When the data type in the original model is float32 (fp32), float16 (fp16), or bfloat16 (bf16), you can use the content described in this section to convert it to the HiFloat8 (HiF8), float8 (fp8), MXFP8, or float4 data format after quantization. By compressing the data format, model lightweighting is achieved.

The layers supported by this feature are as follows.

**Table 1** Layers that support quantization and their constraints

<a name="table32131022111318"></a>

<table><thead align="left"><tr id="row9214202215138"><th class="cellrowborder" valign="top" width="10.63%" id="mcps1.2.5.1.1"><p id="p1921472241319"><a name="p1921472241319"></a><a name="p1921472241319"></a>Supported Layer Type</p></th>
<th class="cellrowborder" valign="top" width="20.09%" id="mcps1.2.5.1.2"><p id="p27472162329"><a name="p27472162329"></a><a name="p27472162329"></a>Original Data Type</p></th>
<th class="cellrowborder" valign="top" width="32.01%" id="mcps1.2.5.1.3"><p id="p10685452175710"><a name="p10685452175710"></a><a name="p10685452175710"></a>Supported Quantization Type Combinations</p></th>
<th class="cellrowborder" valign="top" width="37.269999999999996%" id="mcps1.2.5.1.4"><p id="p122141822101318"><a name="p122141822101318"></a><a name="p122141822101318"></a>Constraints</p></th>
</tr>
</thead>
<tbody><tr id="row7214182271318"><td class="cellrowborder" rowspan="4" valign="top" width="10.63%" headers="mcps1.2.5.1.1 "><p id="p32146222139"><a name="p32146222139"></a><a name="p32146222139"></a>torch.nn.Linear</p>
<p id="p20540182234618"><a name="p20540182234618"></a><a name="p20540182234618"></a></p>
<p id="p85010423713"><a name="p85010423713"></a><a name="p85010423713"></a></p></td>
<td class="cellrowborder" valign="top" width="20.09%" headers="mcps1.2.5.1.2 "><p id="p77471416113220"><a name="p77471416113220"></a><a name="p77471416113220"></a>float32 (fp32), float16 (fp16), bfloat16 (bf16)</p></td>
<td class="cellrowborder" valign="top" width="32.01%" headers="mcps1.2.5.1.3 "><p id="p641017541626"><a name="p641017541626"></a><a name="p641017541626"></a>act_type: HIFLOAT8  wts_type: HIFLOAT8</p>
<p id="p6410054727"><a name="p6410054727"></a><a name="p6410054727"></a>act_type: FLOAT8_E4M3FN  wts_type: FLOAT8_E4M3FN</p></td>
<td class="cellrowborder" valign="top" width="37.269999999999996%" headers="mcps1.2.5.1.4 "><p id="p134301428434"><a name="p134301428434"></a><a name="p134301428434"></a>Activation (data) supports PER_TENSOR/PER_TOKEN quantization, and weight supports PER_TENSOR/PER_CHANNEL quantization.</p>
<p id="p1894815554584"><a name="p1894815554584"></a><a name="p1894815554584"></a>When act_type: HIFLOAT8 and wts_type: HIFLOAT8, the OFMR quantization algorithm, Cast direct data conversion algorithm, Quantile quantization algorithm, and SmoothQuant algorithm are supported. The ofmr, cast, quantile, and smoothquant options must be configured separately in <a href="#section1536112219183">config details</a>.</p>
<p id="p1144561153616"><a name="p1144561153616"></a><a name="p1144561153616"></a>When act_type: FLOAT8_E4M3FN and wts_type: FLOAT8_E4M3FN, the OFMR quantization algorithm is supported. The ofmr option must be configured in <a href="#section1536112219183">config details</a>.</p></td>
</tr>
<tr id="row3281122665615"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p1711435320187"><a name="p1711435320187"></a><a name="p1711435320187"></a>bfloat16 (bf16)</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p172641614579"><a name="p172641614579"></a><a name="p172641614579"></a>act_type: MXFP8_E4M3FN  wts_type: MXFP8_E4M3FN</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p142815266561"><a name="p142815266561"></a><a name="p142815266561"></a>Supports 2- to 6-dimensional data input, PER_GROUP quantization, and symmetric quantization. The cin length divided by 32 and rounded up must be an integer multiple of 2.</p>
<p id="p6287164525916"><a name="p6287164525916"></a><a name="p6287164525916"></a>No quantization algorithm is used, that is, only mx data type conversion is performed. The mxquant option must be configured in <a href="#section1536112219183">config details</a>.</p></td>
</tr>
<tr id="row65391422154618"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p145409227465"><a name="p145409227465"></a><a name="p145409227465"></a>bfloat16 (bf16)</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p16851142184618"><a name="p16851142184618"></a><a name="p16851142184618"></a>act_type: FLOAT8_E4M3FN  wts_type: FLOAT4_E2M1</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p1183219292620"><a name="p1183219292620"></a><a name="p1183219292620"></a>Supports 2- to 6-dimensional data input, and bias is false.</p>
<p id="p103822021171315"><a name="p103822021171315"></a><a name="p103822021171315"></a>Activation (data) supports shape (m,k), and weight supports shape (n,k), where k is an integer multiple of 64.</p>
<p id="p12540182214619"><a name="p12540182214619"></a><a name="p12540182214619"></a>Activation (data) supports PER_TENSOR quantization, and weight supports PER_GROUP quantization.</p>
<p id="p844695365417"><a name="p844695365417"></a><a name="p844695365417"></a>Both activation and weight support only symmetric quantization.</p>
<p id="p1877817173019"><a name="p1877817173019"></a><a name="p1877817173019"></a>The MIN-MAX quantization algorithm and SmoothQuant algorithm are supported. The minmax and smoothquant options must be configured separately in <a href="#section1536112219183">config details</a>.</p></td>
</tr>
<tr id="row9500144213717"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p175010422712"><a name="p175010422712"></a><a name="p175010422712"></a>float16 (fp16), bfloat16 (bf16)</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p650118423714"><a name="p650118423714"></a><a name="p650118423714"></a>act_type: INT8  wts_type: INT8</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><a name="ul178665278918"></a><a name="ul178665278918"></a><ul id="ul178665278918"><li>Supports 2- to 6-dimensional data input.</li><li>Activation (data) supports PER_TENSOR quantization, symmetric and asymmetric quantization, and bias quantized to INT32; weight supports PER_TENSOR/PER_CHANNEL quantization and symmetric quantization.</li><li>Activation (data) supports PER-TOKEN symmetric quantization, bias is not quantized, k is a multiple of 16, and n is a multiple of 8; weight supports PER_TENSOR/PER_CHANNEL symmetric quantization.</li><li>Supports the MIN-MAX quantization algorithm and the SmoothQuant algorithm. The minmax and smoothquant options must be configured separately in <a href="#section1536112219183">config details</a>.</li></ul></td>
</tr>
<tr id="row1521462251317"><td class="cellrowborder" valign="top" width="10.63%" headers="mcps1.2.5.1.1 "><p id="p1021414226138"><a name="p1021414226138"></a><a name="p1021414226138"></a>torch.nn.Conv2d</p></td>
<td class="cellrowborder" valign="top" width="20.09%" headers="mcps1.2.5.1.2 "><p id="p12999111203315"><a name="p12999111203315"></a><a name="p12999111203315"></a>float32 (fp32), float16 (fp16), bfloat16 (bf16)</p></td>
<td class="cellrowborder" valign="top" width="32.01%" headers="mcps1.2.5.1.3 "><p id="p174971639"><a name="p174971639"></a><a name="p174971639"></a>act_type: HIFLOAT8  wts_type: HIFLOAT8</p>
<p id="p14749418315"><a name="p14749418315"></a><a name="p14749418315"></a>act_type: FLOAT8_E4M3FN  wts_type: FLOAT8_E4M3FN</p></td>
<td class="cellrowborder" valign="top" width="37.269999999999996%" headers="mcps1.2.5.1.4 "><p id="p1885141034514"><a name="p1885141034514"></a><a name="p1885141034514"></a>padding_mode is zeros, supports PER_TENSOR/PER_CHANNEL quantization.</p>
<p id="p1274062255916"><a name="p1274062255916"></a><a name="p1274062255916"></a>The quantization algorithm is OFMR. The ofmr option must be configured in <a href="#section1536112219183">config details</a>.</p></td>
</tr>
</tbody>
</table>

Note: The **act\_type** and **wts\_type** parameters refer to the data (activation) and weight quantization types in [config details](#section1536112219183), respectively.

## Layers Supported by Weight-Only Quantization<a name="section831519712409"></a>

This section describes the weight-only quantization feature and the corresponding optimization algorithms, such as the AWQ and GPTQ weight quantization algorithms.

When the data types in the original model are float16 (fp16) and bfloat16 (bf16), you can use the content described in this section to perform weight-only quantization and convert them to the HIFloat8 (HiF8), float8 (fp8), MXFP4, and float4 data formats, thereby compressing the weights to achieve model lightweighting.

**Note:** Because TorchNPU does not yet support the Ascend 950PR&950DT products, the following features currently support only **wts_type: INT8** or **wts_type: INT4**.

The layers supported by this feature are as follows:

**Table 2** Layers that support quantization and their constraints

<a name="table1116172344016"></a>

<table><thead align="left"><tr id="row1016142314404"><th class="cellrowborder" valign="top" width="11.99%" id="mcps1.2.5.1.1"><p id="p516023164015"><a name="p516023164015"></a><a name="p516023164015"></a>Supported Layer Type</p></th>
<th class="cellrowborder" valign="top" width="18.279999999999998%" id="mcps1.2.5.1.2"><p id="p2161723194015"><a name="p2161723194015"></a><a name="p2161723194015"></a>Original Data Type</p></th>
<th class="cellrowborder" valign="top" width="22%" id="mcps1.2.5.1.3"><p id="p16168230408"><a name="p16168230408"></a><a name="p16168230408"></a>Supported Quantization Type Combinations</p></th>
<th class="cellrowborder" valign="top" width="47.73%" id="mcps1.2.5.1.4"><p id="p816152364010"><a name="p816152364010"></a><a name="p816152364010"></a>Constraints</p></th>
</tr>
</thead>
<tbody><tr id="row177511228786"><td class="cellrowborder" rowspan="5" valign="top" width="11.99%" headers="mcps1.2.5.1.1 "><p id="p5390433491"><a name="p5390433491"></a><a name="p5390433491"></a>torch.nn.Linear</p>
<p id="p321913119411"><a name="p321913119411"></a><a name="p321913119411"></a></p>
<p id="p3841056175716"><a name="p3841056175716"></a><a name="p3841056175716"></a></p></td>
<td class="cellrowborder" rowspan="4" valign="top" width="18.279999999999998%" headers="mcps1.2.5.1.2 "><p id="p475117281482"><a name="p475117281482"></a><a name="p475117281482"></a>float16 (fp16), bfloat16 (bf16)</p>
<p id="p1321951541"><a name="p1321951541"></a><a name="p1321951541"></a></p></td>
<td class="cellrowborder" valign="top" width="22%" headers="mcps1.2.5.1.3 "><p id="p1032617312478"><a name="p1032617312478"></a><a name="p1032617312478"></a>wts_type: HIFLOAT8</p>
<p id="p1978123313505"><a name="p1978123313505"></a><a name="p1978123313505"></a>wts_type: FLOAT8_E4M3FN</p></td>
<td class="cellrowborder" valign="top" width="47.73%" headers="mcps1.2.5.1.4 "><a name="ul1681123611612"></a><a name="ul1681123611612"></a><ul id="ul1681123611612"><li>Supports PER_TENSOR/PER_CHANNEL quantization and symmetric quantization.</li><li>Supports 2- to 6-dimensional data input.</li><li>When wts_type is HIFLOAT8, supports the OFMR quantization algorithm, GPTQ quantization algorithm, Cast direct data conversion algorithm, and Quantile quantization algorithm (the <span>ofmr</span>, <span>gptq</span>, <span>cast</span>, and <span>quantile</span> options must be configured in <a href="#section1536112219183">config details</a>).</li><li>When wts_type is FLOAT8_E4M3FN, supports the OFMR quantization algorithm and GPTQ quantization algorithm (the <span>ofmr</span> and <span>gptq</span> options must be configured in <a href="#section1536112219183">config details</a>).</li></ul></td>
</tr>
<tr id="row1326719415381"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p278010227380"><a name="p278010227380"></a><a name="p278010227380"></a>wts_type: MXFP4_E2M1</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><a name="ul9303114315395"></a><a name="ul9303114315395"></a><ul id="ul9303114315395"><li>Supports PER_GROUP quantization and symmetric quantization.</li><li>Supports 2- to 6-dimensional data input.</li><li>Supports the option of not configuring a quantization algorithm (that is, configuring the <span>mxquant</span> option in <a href="#section1536112219183">config details</a>), the GPTQ quantization algorithm (the <span>gptq</span> option must be configured in <a href="#section1536112219183">config details</a>), and the AWQ quantization algorithm (the <span>awq</span> option must be configured in <a href="#section1536112219183">config details</a>).</li></ul></td>
</tr>
<tr id="row131281133204815"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p1812833324819"><a name="p1812833324819"></a><a name="p1812833324819"></a>wts_type: FLOAT4_E2M1</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><a name="ul185841815118"></a><a name="ul185841815118"></a><ul id="ul185841815118"><li>Supports PER_GROUP quantization and symmetric quantization.</li><li>Supports 2- to 6-dimensional data input.</li><li>Supports the MIN-MAX quantization algorithm, GPTQ quantization algorithm, and AWQ quantization algorithm (the <span>minmax</span>, <span>gptq</span>, and <span>awq</span> options must be configured in <a href="#section1536112219183">config details</a>).</li></ul></td>
</tr>
<tr id="row1219111249"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p20219811048"><a name="p20219811048"></a><a name="p20219811048"></a>wts_type: INT8</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><a name="ul06850346713"></a><a name="ul06850346713"></a><ul id="ul06850346713"><li>Supports PER_TENSOR/PER_CHANNEL/PER_GROUP quantization, and supports symmetric/asymmetric quantization.</li><li>Supports the MIN-MAX quantization algorithm, GPTQ quantization algorithm, and AWQ quantization algorithm (the <span>minmax</span>, <span>gptq</span>, and <span>awq</span> options must be configured in <a href="#section1536112219183">config details</a>).</li><li>When wts_type is INT8, the weight of the original model must be aligned to 32 elements on the K and N axes; when wts_type is INT4, it must be aligned to 64 elements on the K and N axes.</li></ul></td>
</tr>
<tr id="row1184114561578"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p284185625716"><a name="p284185625716"></a><a name="p284185625716"></a>float32 (fp32), float16 (fp16), bfloat16 (bf16)</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p58415569572"><a name="p58415569572"></a><a name="p58415569572"></a>wts_type: INT4</p></td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><a name="ul15592626145816"></a><a name="ul15592626145816"></a><ul id="ul15592626145816"><li>Supports PER_TENSOR/PER_CHANNEL/PER_GROUP quantization, and supports symmetric/asymmetric quantization.</li><li>Supports the MIN-MAX quantization algorithm, GPTQ quantization algorithm, and AWQ quantization algorithm (the <span>minmax</span>, <span>gptq</span>, and <span>awq</span> options must be configured in <a href="#section1536112219183">config details</a>).</li><li>When wts_type is INT4, the weight of the original model must be aligned to 64 elements on the K and N axes.</li></ul></td>
</tr>
</tbody>
</table>

## Return Value

None

## Example

```python
# Build the network graph structure to be quantized.
ori_model = build_model()
model = copy.deepcopy(ori_model)
# Quantization configuration.
cfg = {
        'batch_num': 1,
        'quant_cfg': {
            'weights': {
                'type': 'int8',
                'symmetric': True,
                'strategy': 'tensor',
            },
        },
        'algorithm': 'minmax',
        }
# Call the quantize API to generate the quantized calibration model.
quantize(model, cfg)
```

## config Details<a name="section1536112219183"></a>

<a name="table9577255181816"></a>
<table><thead align="left"><tr id="row176031155111812"><th class="cellrowborder" valign="top" width="11.04%" id="mcps1.1.5.1.1"><p id="p86039551189"><a name="p86039551189"></a><a name="p86039551189"></a>Key</p></th>
<th class="cellrowborder" valign="top" width="8.04%" id="mcps1.1.5.1.2"><p id="p17603135519185"><a name="p17603135519185"></a><a name="p17603135519185"></a>-</p></th>
<th class="cellrowborder" valign="top" width="11.07%" id="mcps1.1.5.1.3"><p id="p460385519182"><a name="p460385519182"></a><a name="p460385519182"></a>-</p></th>
<th class="cellrowborder" valign="top" width="69.85%" id="mcps1.1.5.1.4"><p id="p1660318559188"><a name="p1660318559188"></a><a name="p1660318559188"></a>Value</p></th>
</tr>
</thead>
<tbody><tr id="row19603115516188"><td class="cellrowborder" valign="top" width="11.04%" headers="mcps1.1.5.1.1 "><p id="p1960345551813"><a name="p1960345551813"></a><a name="p1960345551813"></a>batch_num</p></td>
<td class="cellrowborder" valign="top" width="8.04%" headers="mcps1.1.5.1.2 "><p id="p1660319551189"><a name="p1660319551189"></a><a name="p1660319551189"></a>-</p></td>
<td class="cellrowborder" valign="top" width="11.07%" headers="mcps1.1.5.1.3 "><p id="p2603195513180"><a name="p2603195513180"></a><a name="p2603195513180"></a>-</p></td>
<td class="cellrowborder" valign="top" width="69.85%" headers="mcps1.1.5.1.4 "><p id="p36030559182"><a name="p36030559182"></a><a name="p36030559182"></a>uint32 type, number of batches used for quantization.</p></td>
</tr>
<tr id="row160365541815"><td class="cellrowborder" valign="top" width="11.04%" headers="mcps1.1.5.1.1 "><p id="p36031055101817"><a name="p36031055101817"></a><a name="p36031055101817"></a>quant_cfg</p></td>
<td class="cellrowborder" valign="top" width="8.04%" headers="mcps1.1.5.1.2 "><p id="p06031555181817"><a name="p06031555181817"></a><a name="p06031555181817"></a>-</p></td>
<td class="cellrowborder" valign="top" width="11.07%" headers="mcps1.1.5.1.3 "><p id="p1260345520188"><a name="p1260345520188"></a><a name="p1260345520188"></a>-</p></td>
<td class="cellrowborder" valign="top" width="69.85%" headers="mcps1.1.5.1.4 "><p id="p17603125516185"><a name="p17603125516185"></a><a name="p17603125516185"></a>Quantization configuration.</p></td>
</tr>
<tr id="row760395512182"><td class="cellrowborder" valign="top" width="11.04%" headers="mcps1.1.5.1.1 "><p id="p860312558188"><a name="p860312558188"></a><a name="p860312558188"></a>-</p></td>
<td class="cellrowborder" valign="top" width="8.04%" headers="mcps1.1.5.1.2 "><p id="p176038551185"><a name="p176038551185"></a><a name="p176038551185"></a>weights</p></td>
<td class="cellrowborder" valign="top" width="11.07%" headers="mcps1.1.5.1.3 "><p id="p560365517181"><a name="p560365517181"></a><a name="p560365517181"></a>-</p></td>
<td class="cellrowborder" valign="top" width="69.85%" headers="mcps1.1.5.1.4 "><p id="p14603755171815"><a name="p14603755171815"></a><a name="p14603755171815"></a>Weight-only quantization configuration.</p></td>
</tr>
<tr id="row16031555181820"><td class="cellrowborder" valign="top" width="11.04%" headers="mcps1.1.5.1.1 "><p id="p156031955201812"><a name="p156031955201812"></a><a name="p156031955201812"></a>-</p></td>
<td class="cellrowborder" valign="top" width="8.04%" headers="mcps1.1.5.1.2 "><p id="p1260315520181"><a name="p1260315520181"></a><a name="p1260315520181"></a>-</p></td>
<td class="cellrowborder" valign="top" width="11.07%" headers="mcps1.1.5.1.3 "><p id="p1860325561813"><a name="p1860325561813"></a><a name="p1860325561813"></a>type</p></td>
<td class="cellrowborder" valign="top" width="69.85%" headers="mcps1.1.5.1.4 "><p id="p15462475258"><a name="p15462475258"></a><a name="p15462475258"></a>String type, weight quantization granularity. Currently, the following types are supported:</p>
<a name="ul1911188102519"></a><a name="ul1911188102519"></a><ul id="ul1911188102519"><li>hifloat8</li><li>float8_e4m3fn</li><li>mxfp4_e2m1</li><li>float4_e2m1</li><li>int4</li><li>int8</li></ul></td>
</tr>
<tr id="row196031155121813"><td class="cellrowborder" valign="top" width="11.04%" headers="mcps1.1.5.1.1 "><p id="p060355517182"><a name="p060355517182"></a><a name="p060355517182"></a>-</p></td>
<td class="cellrowborder" valign="top" width="8.04%" headers="mcps1.1.5.1.2 "><p id="p116031855111811"><a name="p116031855111811"></a><a name="p116031855111811"></a>-</p></td>
<td class="cellrowborder" valign="top" width="11.07%" headers="mcps1.1.5.1.3 "><p id="p56031855131819"><a name="p56031855131819"></a><a name="p56031855131819"></a>symmetric</p></td>
<td class="cellrowborder" valign="top" width="69.85%" headers="mcps1.1.5.1.4 "><p id="p860385513186"><a name="p860385513186"></a><a name="p860385513186"></a>Bool type, whether the weight uses symmetric quantization.</p>
<a name="ul16491192891913"></a><a name="ul16491192891913"></a><ul id="ul16491192891913"><li>True: Symmetric quantization.</li><li>False: Asymmetric quantization.</li></ul></td>
</tr>
<tr id="row060315581820"><td class="cellrowborder" valign="top" width="11.04%" headers="mcps1.1.5.1.1 "><p id="p860385571813"><a name="p860385571813"></a><a name="p860385571813"></a>-</p></td>
<td class="cellrowborder" valign="top" width="8.04%" headers="mcps1.1.5.1.2 "><p id="p060345519188"><a name="p060345519188"></a><a name="p060345519188"></a>-</p></td>
<td class="cellrowborder" valign="top" width="11.07%" headers="mcps1.1.5.1.3 "><p id="p460405511816"><a name="p460405511816"></a><a name="p460405511816"></a>strategy</p></td>
<td class="cellrowborder" valign="top" width="69.85%" headers="mcps1.1.5.1.4 "><p id="p13604125511814"><a name="p13604125511814"></a><a name="p13604125511814"></a>String type, weight quantization granularity.</p>
<a name="ul8259163310191"></a><a name="ul8259163310191"></a><ul id="ul8259163310191"><li>tensor, corresponding to per-tensor.</li><li>channel, corresponding to per-channel.</li><li>group, corresponding to per-group.</li></ul>
<p id="p10604115511810"><a name="p10604115511810"></a><a name="p10604115511810"></a>For details about quantization granularity, see <a href="../compression_concepts.md" target="_blank" rel="noopener noreferrer">Compression Concepts</a>.</p></td>
</tr>
<tr id="row1460485515186"><td class="cellrowborder" valign="top" width="11.04%" headers="mcps1.1.5.1.1 "><p id="p1260495514181"><a name="p1260495514181"></a><a name="p1260495514181"></a>-</p></td>
<td class="cellrowborder" valign="top" width="8.04%" headers="mcps1.1.5.1.2 "><p id="p86047556183"><a name="p86047556183"></a><a name="p86047556183"></a>-</p></td>
<td class="cellrowborder" valign="top" width="11.07%" headers="mcps1.1.5.1.3 "><p id="p17604125519187"><a name="p17604125519187"></a><a name="p17604125519187"></a>group_size</p></td>
<td class="cellrowborder" valign="top" width="69.85%" headers="mcps1.1.5.1.4 "><p id="p1060465519185"><a name="p1060465519185"></a><a name="p1060465519185"></a>Configured in weight-only quantization scenarios. It specifies the group size under per-group quantization granularity. This parameter can be configured only after per-group is configured.</p>
<p id="p360416554186"><a name="p360416554186"></a><a name="p360416554186"></a>The value must be in the range [32, K-1] and must be a multiple of 32.</p></td>
</tr>
<tr id="row206041655121811"><td class="cellrowborder" valign="top" width="11.04%" headers="mcps1.1.5.1.1 "><p id="p196041557187"><a name="p196041557187"></a><a name="p196041557187"></a>-</p></td>
<td class="cellrowborder" valign="top" width="8.04%" headers="mcps1.1.5.1.2 "><p id="p860445513180"><a name="p860445513180"></a><a name="p860445513180"></a>inputs</p></td>
<td class="cellrowborder" valign="top" width="11.07%" headers="mcps1.1.5.1.3 "><p id="p1560435510182"><a name="p1560435510182"></a><a name="p1560435510182"></a>-</p></td>
<td class="cellrowborder" valign="top" width="69.85%" headers="mcps1.1.5.1.4 "><p id="p196043556186"><a name="p196043556186"></a><a name="p196043556186"></a>Data quantization configuration.</p></td>
</tr>
<tr id="row13604855101811"><td class="cellrowborder" valign="top" width="11.04%" headers="mcps1.1.5.1.1 "><p id="p76046553183"><a name="p76046553183"></a><a name="p76046553183"></a>-</p></td>
<td class="cellrowborder" valign="top" width="8.04%" headers="mcps1.1.5.1.2 "><p id="p7604165581818"><a name="p7604165581818"></a><a name="p7604165581818"></a>-</p></td>
<td class="cellrowborder" valign="top" width="11.07%" headers="mcps1.1.5.1.3 "><p id="p1260495516188"><a name="p1260495516188"></a><a name="p1260495516188"></a>type</p></td>
<td class="cellrowborder" valign="top" width="69.85%" headers="mcps1.1.5.1.4 "><p id="p188704582212"><a name="p188704582212"></a><a name="p188704582212"></a>string type, data (activation) quantization granularity. Currently supports the following types:</p>
<a name="ul1560524672215"></a><a name="ul1560524672215"></a><ul id="ul1560524672215"><li>hifloat8</li><li>float8_e4m3fn</li><li>mxfp8_e4m3fn</li><li><span>int8</span></li></ul></td>
</tr>
<tr id="row760415581817"><td class="cellrowborder" valign="top" width="11.04%" headers="mcps1.1.5.1.1 "><p id="p1260435521812"><a name="p1260435521812"></a><a name="p1260435521812"></a>-</p></td>
<td class="cellrowborder" valign="top" width="8.04%" headers="mcps1.1.5.1.2 "><p id="p760485516184"><a name="p760485516184"></a><a name="p760485516184"></a>-</p></td>
<td class="cellrowborder" valign="top" width="11.07%" headers="mcps1.1.5.1.3 "><p id="p11604165516183"><a name="p11604165516183"></a><a name="p11604165516183"></a>symmetric</p></td>
<td class="cellrowborder" valign="top" width="69.85%" headers="mcps1.1.5.1.4 "><p id="p46041355111814"><a name="p46041355111814"></a><a name="p46041355111814"></a>bool type, whether the data uses symmetric quantization.</p>
<a name="ul2313916200"></a><a name="ul2313916200"></a><ul id="ul2313916200"><li>True: Symmetric quantization.</li><li>False: Asymmetric quantization.</li></ul></td>
</tr>
<tr id="row136046556186"><td class="cellrowborder" valign="top" width="11.04%" headers="mcps1.1.5.1.1 "><p id="p36041255101819"><a name="p36041255101819"></a><a name="p36041255101819"></a>-</p></td>
<td class="cellrowborder" valign="top" width="8.04%" headers="mcps1.1.5.1.2 "><p id="p860445516186"><a name="p860445516186"></a><a name="p860445516186"></a>-</p></td>
<td class="cellrowborder" valign="top" width="11.07%" headers="mcps1.1.5.1.3 "><p id="p1960411556189"><a name="p1960411556189"></a><a name="p1960411556189"></a>strategy</p></td>
<td class="cellrowborder" valign="top" width="69.85%" headers="mcps1.1.5.1.4 "><p id="p5604355171813"><a name="p5604355171813"></a><a name="p5604355171813"></a>string type, data quantization granularity.</p>
<a name="ul7111191082014"></a><a name="ul7111191082014"></a><ul id="ul7111191082014"><li>tensor: Corresponds to per-tensor.</li><li>token: Corresponds to per-token.</li><li>group: Corresponds to per-group. This parameter can be configured only when the quantization data type is mxfp8_e4m3fn.</li></ul></td>
</tr>
<tr id="row1160485581816"><td class="cellrowborder" valign="top" width="11.04%" headers="mcps1.1.5.1.1 "><p id="p6604455151820"><a name="p6604455151820"></a><a name="p6604455151820"></a>algorithm</p></td>
<td class="cellrowborder" valign="top" width="8.04%" headers="mcps1.1.5.1.2 "><p id="p2060425511813"><a name="p2060425511813"></a><a name="p2060425511813"></a>-</p></td>
<td class="cellrowborder" valign="top" width="11.07%" headers="mcps1.1.5.1.3 "><p id="p1060425541813"><a name="p1060425541813"></a><a name="p1060425541813"></a>-</p></td>
<td class="cellrowborder" valign="top" width="69.85%" headers="mcps1.1.5.1.4 "><p id="p66040553184"><a name="p66040553184"></a><a name="p66040553184"></a>string type, quantization algorithm. When only one parameter-free algorithm is configured, pass the algorithm name string directly. When multiple algorithms are configured, use the set form (a set of parameter-free algorithm names, for example {'minmax', 'gptq'}) or the dict form (a mapping from algorithm names to parameters, for example {'awq': {'grids_num': 20}}). The supported configurations are as follows:</p>
<a name="ul2808111772013"></a><a name="ul2808111772013"></a><ul id="ul2808111772013"><li>awq: grids_num, uint32 type, number of search grid points, defaulting to 20.</li><li>gptq.</li><li>minmax.</li><li>smoothquant: smooth_strength, float type, migration strength, defaulting to 0.5.</li><li>ofmr.</li><li>mxquant: only performs mx data type conversion.</li><li>cast: HiFloat8 direct data conversion algorithm, used to directly convert weights and activations to the HiFloat8 type.</li><li>quantile: HiFloat8 quantile quantization algorithm, used to compute HiFloat8 quantization parameters based on quantile statistics.</li></ul>
<p id="p66040553185"><a name="p66040553185"></a><a name="p66040553185"></a>Multiple algorithms can be configured simultaneously. The algorithms divide work according to the source operators (src_op) they support, meaning that different types of layers can be processed by different algorithms. The same source operator can be handled by only one algorithm. If multiple configured algorithms declare support for the same source operator, an error is reported during the configuration parsing phase.</p>
<p id="p4604195517184"><a name="p4604195517184"></a><a name="p4604195517184"></a>For details, see <a href="../algorithm_brief.md" target="_blank" rel="noopener noreferrer">Quantization Algorithm Introduction</a>.</p></td>
</tr>
<tr id="row1060418558184"><td class="cellrowborder" valign="top" width="11.04%" headers="mcps1.1.5.1.1 "><p id="p9604135541813"><a name="p9604135541813"></a><a name="p9604135541813"></a>skip_layers</p></td>
<td class="cellrowborder" valign="top" width="8.04%" headers="mcps1.1.5.1.2 "><p id="p10604145516182"><a name="p10604145516182"></a><a name="p10604145516182"></a>-</p></td>
<td class="cellrowborder" valign="top" width="11.07%" headers="mcps1.1.5.1.3 "><p id="p1260435581811"><a name="p1260435581811"></a><a name="p1260435581811"></a>-</p></td>
<td class="cellrowborder" valign="top" width="69.85%" headers="mcps1.1.5.1.4 "><p id="p10604355191815"><a name="p10604355191815"></a><a name="p10604355191815"></a>string type, skips the specified layers by layer name so that they are not quantized. This is a global configuration parameter. After a layer name is specified, any layer whose name contains the user-specified string is skipped and not quantized.</p></td>
</tr>
</tbody>
</table>
