# create\_quant\_cali\_model<a name="ZH-CN_TOPIC_0000002548668597"></a>

<!-- md-trans-meta sourceCommit=9d54d17e0942052193c827180dc43312c580acc7 translatedAt=2026-10-16T08:21:47.499Z pushedAt=2026-11-05T01:12:03.411Z nl=2 -->

## Supported Products<a name="section1610172518544"></a>

<a name="zh-cn_topic_0000002517188700_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002517188700_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002517188700_p1883113061818"><a name="zh-cn_topic_0000002517188700_p1883113061818"></a><a name="zh-cn_topic_0000002517188700_p1883113061818"></a><span id="zh-cn_topic_0000002517188700_ph20833205312295"><a name="zh-cn_topic_0000002517188700_ph20833205312295"></a><a name="zh-cn_topic_0000002517188700_ph20833205312295"></a>Product</span></p></th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002517188700_p783113012187"><a name="zh-cn_topic_0000002517188700_p783113012187"></a><a name="zh-cn_topic_0000002517188700_p783113012187"></a>Supported</p></th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002517188700_row574891710101"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002517188700_p177491117171015"><a name="zh-cn_topic_0000002517188700_p177491117171015"></a><a name="zh-cn_topic_0000002517188700_p177491117171015"></a><span id="zh-cn_topic_0000002517188700_ph2272194216543"><a name="zh-cn_topic_0000002517188700_ph2272194216543"></a><a name="zh-cn_topic_0000002517188700_ph2272194216543"></a>Ascend 950PR&950DT products</span></p></td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002517188700_p14226338117"><a name="zh-cn_topic_0000002517188700_p14226338117"></a><a name="zh-cn_topic_0000002517188700_p14226338117"></a>√</p></td>
</tr>
<tr id="zh-cn_topic_0000002517188700_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002517188700_p48327011813"><a name="zh-cn_topic_0000002517188700_p48327011813"></a><a name="zh-cn_topic_0000002517188700_p48327011813"></a><span id="zh-cn_topic_0000002517188700_ph583230201815"><a name="zh-cn_topic_0000002517188700_ph583230201815"></a><a name="zh-cn_topic_0000002517188700_ph583230201815"></a><term id="zh-cn_topic_0000002517188700_zh-cn_topic_0000001312391781_term1253731311225">Atlas A3 products</term></span></p></td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002517188700_p108715341013"><a name="zh-cn_topic_0000002517188700_p108715341013"></a><a name="zh-cn_topic_0000002517188700_p108715341013"></a>√</p></td>
</tr>
<tr id="zh-cn_topic_0000002517188700_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002517188700_p14832120181815"><a name="zh-cn_topic_0000002517188700_p14832120181815"></a><a name="zh-cn_topic_0000002517188700_p14832120181815"></a><span id="zh-cn_topic_0000002517188700_ph1483216010188"><a name="zh-cn_topic_0000002517188700_ph1483216010188"></a><a name="zh-cn_topic_0000002517188700_ph1483216010188"></a><term id="zh-cn_topic_0000002517188700_zh-cn_topic_0000001312391781_term11962195213215">Atlas A2 products</term></span></p></td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000002517188700_p19948143911820"><a name="zh-cn_topic_0000002517188700_p19948143911820"></a><a name="zh-cn_topic_0000002517188700_p19948143911820"></a>√</p></td>
</tr>
</tbody>
</table>

**Note: For products marked with "x", calling the API does not report an error, but no performance gain is obtained.**

## Description<a name="zh-cn_topic_0240187365_section15406195619561"></a>

The KV-cache quantization API modifies the user model based on the model and detailed quantization configuration, replacing the Linear operator to be quantized with a quantization operator that performs IFMR/HFMG quantization on its output. The user then performs online calibration on the model, and the generated quantization factors are saved in **record\_file**.

## Prototype<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_section428121323411"></a>

```python
calibration_model = create_quant_cali_model(config_file, record_file, model)
```

## Parameters<a name="section73811524135618"></a>

<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="9.09%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a>Parameter</p></th>
<th class="cellrowborder" valign="top" width="7.35%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240187365_p1769255516412"><a name="zh-cn_topic_0240187365_p1769255516412"></a><a name="zh-cn_topic_0240187365_p1769255516412"></a>Input/Output</p></th>
<th class="cellrowborder" valign="top" width="83.56%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0240187365_p15231205416325"><a name="zh-cn_topic_0240187365_p15231205416325"></a><a name="zh-cn_topic_0240187365_p15231205416325"></a>Description</p></th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="9.09%" headers="mcps1.1.4.1.1 "><p id="p31941230154616"><a name="p31941230154616"></a><a name="p31941230154616"></a>config_file</p></td>
<td class="cellrowborder" valign="top" width="7.35%" headers="mcps1.1.4.1.2 "><p id="p1219493024610"><a name="p1219493024610"></a><a name="p1219493024610"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.1.4.1.3 "><p id="p859151319514"><a name="p859151319514"></a><a name="p859151319514"></a>Meaning: Path of the generated quantization configuration file. The configuration file is in JSON format.</p>
<p id="p7194123094612"><a name="p7194123094612"></a><a name="p7194123094612"></a>Data type: string</p>
<p id="p9725102314288"><a name="p9725102314288"></a><a name="p9725102314288"></a>Constraint: The config.json input to this API must be the same as the config.json input to the <a href="create_quant_cali_config.md">create_quant_cali_config</a> API.</p></td>
</tr>
<tr id="row36995816444"><td class="cellrowborder" valign="top" width="9.09%" headers="mcps1.1.4.1.1 "><p id="p269918812448"><a name="p269918812448"></a><a name="p269918812448"></a>record_file</p></td>
<td class="cellrowborder" valign="top" width="7.35%" headers="mcps1.1.4.1.2 "><p id="p14699985449"><a name="p14699985449"></a><a name="p14699985449"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.1.4.1.3 "><p id="p179501119125112"><a name="p179501119125112"></a><a name="p179501119125112"></a>Meaning: Path and file name for saving the online calibration quantization factors.</p>
<p id="p66998816443"><a name="p66998816443"></a><a name="p66998816443"></a>Data type: string</p></td>
</tr>
<tr id="row4280131024613"><td class="cellrowborder" valign="top" width="9.09%" headers="mcps1.1.4.1.1 "><p id="p4194143084610"><a name="p4194143084610"></a><a name="p4194143084610"></a>model</p></td>
<td class="cellrowborder" valign="top" width="7.35%" headers="mcps1.1.4.1.2 "><p id="p14194430114612"><a name="p14194430114612"></a><a name="p14194430114612"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="83.56%" headers="mcps1.1.4.1.3 "><p id="p03091421185117"><a name="p03091421185117"></a><a name="p03091421185117"></a>Meaning: Model to be quantized provided by the user.</p>
<p id="p19194123013463"><a name="p19194123013463"></a><a name="p19194123013463"></a>Data type: torch.nn.Module</p></td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section293415513458"></a>

Quantization calibration model with the operators replaced by calibration operators.

## Example<a name="zh-cn_topic_0240188739_section64231658994"></a>

```python
import amct_pytorch as amct
# Build the network graph structure to be quantized.
model = build_model()
model.load_state_dict(torch.load(state_dict_path))

record_file = os.path.join(TMP, 'kv_cache.txt')
# Insert the quantization API to generate the quantization calibration model.
calibration_model = amct.create_quant_cali_model(
                    config_file="./configs/config.json",  # Generated quantization factor recording file.
                    record_file=record_file,
                    model=model)
```
