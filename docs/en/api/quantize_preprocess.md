# quantize\_preprocess<a name="ZH-CN_TOPIC_0000002517028800"></a>

<!-- md-trans-meta sourceCommit=4a47814eed18048e444f9a2acc42d9864ddc477f translatedAt=2026-10-16T08:32:12.740Z pushedAt=2026-11-05T01:12:03.422Z nl=2 -->

## Supported Products<a name="section185612964420"></a>

<a name="zh-cn_topic_0000002517188794_table38301303189"></a>

| Product | Supported |
| ------------------------------------------- | -------- |
| Ascend 950PR&950DT products | √ |
| Atlas A3 products | √ |
| Atlas A2 products | √ |

## Description<a name="zh-cn_topic_0240187365_section15406195619561"></a>

The quantization data equalization preprocessing API quantizes the input graph structure to be quantized according to the given quantization configuration file, inserts balanced quantization-related operators into the input graph structure, generates a balanced quantization factor record file **record\_file**, and returns the modified torch.nn.Module calibration model.

## Prototype<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_section428121323411"></a>

```python
calibration_model = quantize_preprocess(config_file, record_file, model, input_data)
```

## Parameters<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_section795991810344"></a>

<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="10.13%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a>Parameter</p></th>
<th class="cellrowborder" valign="top" width="9.6%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240187365_p1769255516412"><a name="zh-cn_topic_0240187365_p1769255516412"></a><a name="zh-cn_topic_0240187365_p1769255516412"></a>Input/Output</p></th>
<th class="cellrowborder" valign="top" width="80.27%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0240187365_p15231205416325"><a name="zh-cn_topic_0240187365_p15231205416325"></a><a name="zh-cn_topic_0240187365_p15231205416325"></a>Description</p></th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="10.13%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0240187994_p11214202165418"><a name="zh-cn_topic_0240187994_p11214202165418"></a><a name="zh-cn_topic_0240187994_p11214202165418"></a>config_file</p></td>
<td class="cellrowborder" valign="top" width="9.6%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0240187994_p8693185517417"><a name="zh-cn_topic_0240187994_p8693185517417"></a><a name="zh-cn_topic_0240187994_p8693185517417"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.27%" headers="mcps1.1.4.1.3 "><p id="p195971352101120"><a name="p195971352101120"></a><a name="p195971352101120"></a>Meaning: User-generated quantization configuration file, used to specify the configuration of quantization layers in the model network.</p>
<p id="zh-cn_topic_0240187994_p131301850191420"><a name="zh-cn_topic_0240187994_p131301850191420"></a><a name="zh-cn_topic_0240187994_p131301850191420"></a>Data type: string</p></td>
</tr>
<tr id="zh-cn_topic_0240187365_row2997141123911"><td class="cellrowborder" valign="top" width="10.13%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0240187994_p1353832472811"><a name="zh-cn_topic_0240187994_p1353832472811"></a><a name="zh-cn_topic_0240187994_p1353832472811"></a>record_file</p></td>
<td class="cellrowborder" valign="top" width="9.6%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0240187994_p1153815249285"><a name="zh-cn_topic_0240187994_p1153815249285"></a><a name="zh-cn_topic_0240187994_p1153815249285"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.27%" headers="mcps1.1.4.1.3 "><p id="p20262125871111"><a name="p20262125871111"></a><a name="p20262125871111"></a>Meaning: Path and name of the balanced quantization factor record file.</p>
<p id="zh-cn_topic_0240187994_p12538152410286"><a name="zh-cn_topic_0240187994_p12538152410286"></a><a name="zh-cn_topic_0240187994_p12538152410286"></a>Data type: string</p></td>
</tr>
<tr id="zh-cn_topic_0240187365_row13558046183914"><td class="cellrowborder" valign="top" width="10.13%" headers="mcps1.1.4.1.1 "><p id="p10871153295919"><a name="p10871153295919"></a><a name="p10871153295919"></a>model</p></td>
<td class="cellrowborder" valign="top" width="9.6%" headers="mcps1.1.4.1.2 "><p id="p832794819212"><a name="p832794819212"></a><a name="p832794819212"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.27%" headers="mcps1.1.4.1.3 "><p id="p37210595113"><a name="p37210595113"></a><a name="p37210595113"></a>Meaning: Model to be quantized, with weights already loaded.</p>
<p id="p88711432155917"><a name="p88711432155917"></a><a name="p88711432155917"></a>Data type: torch.nn.Module</p></td>
</tr>
<tr id="zh-cn_topic_0240187365_row473005916274"><td class="cellrowborder" valign="top" width="10.13%" headers="mcps1.1.4.1.1 "><p id="p349123714599"><a name="p349123714599"></a><a name="p349123714599"></a>input_data</p></td>
<td class="cellrowborder" valign="top" width="9.6%" headers="mcps1.1.4.1.2 "><p id="p18331648329"><a name="p18331648329"></a><a name="p18331648329"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.27%" headers="mcps1.1.4.1.3 "><p id="p1265301111213"><a name="p1265301111213"></a><a name="p1265301111213"></a>Meaning: Input data of the model. A <span>torch.tensor</span><span> is treated as a </span><span>tuple</span> <span>(</span><span>torch.tensor</span><span>)</span>.</p>
<p id="p7501837155915"><a name="p7501837155915"></a><a name="p7501837155915"></a>Data type: tuple</p></td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_section293415513458"></a>

Returns the modified torch.nn.Module calibration model.

## Example<a name="zh-cn_topic_0240187365_section64231658994"></a>

```python
import amct_pytorch as amct
# Build the network graph structure to be quantized.
model = build_model()
model.load_state_dict(torch.load(state_dict_path))
input_data = tuple([torch.randn(input_shape)])

tensor_balance_factor_record_file = os.path.join(TMP, 'tensor_balance_factor_record.txt')
modified_model = os.path.join(TMP, 'modified_model.onnx')
# Insert the quantization API.
calibration_model = amct.quantize_preprocess(config_json_file,
                                             tensor_balance_factor_record_file,
                                             model,
                                             input_data)

```
