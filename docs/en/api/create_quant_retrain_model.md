# create\_quant\_retrain\_model<a name="ZH-CN_TOPIC_0000002548788619"></a>

<!-- md-trans-meta sourceCommit=4a47814eed18048e444f9a2acc42d9864ddc477f translatedAt=2026-10-16T08:25:17.437Z pushedAt=2026-11-05T01:12:03.415Z nl=2 -->

## Supported Products<a name="section185612964420"></a>

| Product | Supported |
| ------------------------------------------- | -------- |
| Ascend 950PR&950DT products | √ |
| Atlas A3 products | √ |
| Atlas A2 products | √ |

## Description<a name="zh-cn_topic_0240187365_section15406195619561"></a>

The quantization-aware training interface quantizes the input graph structure to be quantized according to the given quantization configuration file, inserts quantization-related operators (quantization-aware training layers for data and weights, and the layer for finding N) into the input graph structure, generates a quantization factor record file **record\_file**, and returns the modified torch.nn.Module model that can be used for quantization-aware training.

## Prototype<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_section428121323411"></a>

```python
quant_retrain_model = create_quant_retrain_model (config_file, model, record_file, input_data)
```

## Parameters<a name="section73811524135618"></a>

<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="11.81%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a>Parameter</p></th>
<th class="cellrowborder" valign="top" width="9.719999999999999%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240187365_p1769255516412"><a name="zh-cn_topic_0240187365_p1769255516412"></a><a name="zh-cn_topic_0240187365_p1769255516412"></a>Input/Output</p></th>
<th class="cellrowborder" valign="top" width="78.47%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0240187365_p15231205416325"><a name="zh-cn_topic_0240187365_p15231205416325"></a><a name="zh-cn_topic_0240187365_p15231205416325"></a>Description</p></th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="11.81%" headers="mcps1.1.4.1.1 "><p id="p31941230154616"><a name="p31941230154616"></a><a name="p31941230154616"></a>config_file</p></td>
<td class="cellrowborder" valign="top" width="9.719999999999999%" headers="mcps1.1.4.1.2 "><p id="p1219493024610"><a name="p1219493024610"></a><a name="p1219493024610"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.47%" headers="mcps1.1.4.1.3 "><p id="p121814042412"><a name="p121814042412"></a><a name="p121814042412"></a>Meaning: User-generated quantization-aware training configuration file, used to specify the configuration of quantization layers in the model network.</p>
<p id="p7194123094612"><a name="p7194123094612"></a><a name="p7194123094612"></a>Data type: string</p></td>
</tr>
<tr id="row4280131024613"><td class="cellrowborder" valign="top" width="11.81%" headers="mcps1.1.4.1.1 "><p id="p4194143084610"><a name="p4194143084610"></a><a name="p4194143084610"></a>model</p></td>
<td class="cellrowborder" valign="top" width="9.719999999999999%" headers="mcps1.1.4.1.2 "><p id="p14194430114612"><a name="p14194430114612"></a><a name="p14194430114612"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.47%" headers="mcps1.1.4.1.3 "><p id="p101129412418"><a name="p101129412418"></a><a name="p101129412418"></a>Meaning: Model to undergo quantization-aware training, with weights loaded.</p>
<p id="p19194123013463"><a name="p19194123013463"></a><a name="p19194123013463"></a>Data type: torch.nn.Module</p></td>
</tr>
<tr id="row16306625164620"><td class="cellrowborder" valign="top" width="11.81%" headers="mcps1.1.4.1.1 "><p id="p19194113044616"><a name="p19194113044616"></a><a name="p19194113044616"></a>record_file</p></td>
<td class="cellrowborder" valign="top" width="9.719999999999999%" headers="mcps1.1.4.1.2 "><p id="p1519433016468"><a name="p1519433016468"></a><a name="p1519433016468"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.47%" headers="mcps1.1.4.1.3 "><p id="p126443717247"><a name="p126443717247"></a><a name="p126443717247"></a>Meaning: Path and name of the quantization factor record file.</p>
<p id="p31945306466"><a name="p31945306466"></a><a name="p31945306466"></a>Data type: string</p></td>
</tr>
<tr id="row156638166467"><td class="cellrowborder" valign="top" width="11.81%" headers="mcps1.1.4.1.1 "><p id="p1619443010466"><a name="p1619443010466"></a><a name="p1619443010466"></a>input_data</p></td>
<td class="cellrowborder" valign="top" width="9.719999999999999%" headers="mcps1.1.4.1.2 "><p id="p919503013466"><a name="p919503013466"></a><a name="p919503013466"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="78.47%" headers="mcps1.1.4.1.3 "><p id="p182273982410"><a name="p182273982410"></a><a name="p182273982410"></a>Meaning: Input data of the model. A <span>torch.tensor</span><span> is equivalent to tuple(torch.tensor)</span>.</p>
<p id="p1195430114611"><a name="p1195430114611"></a><a name="p1195430114611"></a>Data type: tuple</p></td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section293415513458"></a>

Returns the modified torch.nn.Module model that can be used for quantization-aware training.

## Example<a name="zh-cn_topic_0240188739_section64231658994"></a>

```python
import amct_pytorch as amct
# Build the network graph structure for quantization-aware training.
model = build_model()
model.load_state_dict(torch.load(state_dict_path))
input_data = tuple([torch.randn(input_shape)])
 
scale_offset_record_file = os.path.join(TMP, 'scale_offset_record.txt')
# Insert the quantization API.
quant_retrain_model = amct.create_quant_retrain_model(
               config_json_file,
               model,
               scale_offset_record_file,
               input_data)
```
