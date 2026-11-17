# create\_distill\_model<a name="ZH-CN_TOPIC_0000002517028774"></a>

<!-- md-trans-meta sourceCommit=4a47814eed18048e444f9a2acc42d9864ddc477f translatedAt=2026-10-16T08:18:03.681Z pushedAt=2026-11-05T01:12:03.409Z nl=2 -->

## Supported Products<a name="section185612964420"></a>

<a name="zh-cn_topic_0000002517188794_table38301303189"></a>

| Product | Supported |
| ------------------------------------------- | -------- |
| Ascend 950PR&950DT products | √ |
| Atlas A3 products | √ |
| Atlas A2 products | √ |

## Description<a name="zh-cn_topic_0240187365_section15406195619561"></a>

Quantizes the input graph structure to be quantized and compressed according to the given distillation quantization configuration file, inserts quantization-related operators (distillation quantization layers for data and weights, and the layer for finding N) into the input graph structure, and returns a modified torch.nn.Module model that can be used for distillation.

## Prototype<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_section428121323411"></a>

```python
compress_model = create_distill_model(config_file, model, input_data)
```

## Parameters<a name="section73811524135618"></a>

<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="10.901090109010902%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a>Parameter</p></th>
<th class="cellrowborder" valign="top" width="7.4007400740074%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240187365_p1769255516412"><a name="zh-cn_topic_0240187365_p1769255516412"></a><a name="zh-cn_topic_0240187365_p1769255516412"></a>Input/Output</p></th>
<th class="cellrowborder" valign="top" width="81.69816981698169%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0240187365_p15231205416325"><a name="zh-cn_topic_0240187365_p15231205416325"></a><a name="zh-cn_topic_0240187365_p15231205416325"></a>Description</p></th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="10.901090109010902%" headers="mcps1.1.4.1.1 "><p id="p31941230154616"><a name="p31941230154616"></a><a name="p31941230154616"></a>config_file</p></td>
<td class="cellrowborder" valign="top" width="7.4007400740074%" headers="mcps1.1.4.1.2 "><p id="p1219493024610"><a name="p1219493024610"></a><a name="p1219493024610"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="81.69816981698169%" headers="mcps1.1.4.1.3 "><p id="p2828559124411"><a name="p2828559124411"></a><a name="p2828559124411"></a>Meaning: User-generated distillation quantization configuration file, which specifies the configuration of quantization layers and the distillation structure in the model network.</p>
<p id="p7194123094612"><a name="p7194123094612"></a><a name="p7194123094612"></a>Data type: string</p>
<p id="p9725102314288"><a name="p9725102314288"></a><a name="p9725102314288"></a>Constraint: The config.json input to this API must be consistent with the config.json input to the <a href="create_distill_config.md">create_distill_config</a> API.</p></td>
</tr>
<tr id="row4280131024613"><td class="cellrowborder" valign="top" width="10.901090109010902%" headers="mcps1.1.4.1.1 "><p id="p4194143084610"><a name="p4194143084610"></a><a name="p4194143084610"></a>model</p></td>
<td class="cellrowborder" valign="top" width="7.4007400740074%" headers="mcps1.1.4.1.2 "><p id="p14194430114612"><a name="p14194430114612"></a><a name="p14194430114612"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="81.69816981698169%" headers="mcps1.1.4.1.3 "><p id="p1980219674518"><a name="p1980219674518"></a><a name="p1980219674518"></a>Meaning: Original floating-point model to be distilled and quantized, with weights loaded.</p>
<p id="p19194123013463"><a name="p19194123013463"></a><a name="p19194123013463"></a>Data type: torch.nn.Module</p></td>
</tr>
<tr id="row156638166467"><td class="cellrowborder" valign="top" width="10.901090109010902%" headers="mcps1.1.4.1.1 "><p id="p1619443010466"><a name="p1619443010466"></a><a name="p1619443010466"></a>input_data</p></td>
<td class="cellrowborder" valign="top" width="7.4007400740074%" headers="mcps1.1.4.1.2 "><p id="p919503013466"><a name="p919503013466"></a><a name="p919503013466"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="81.69816981698169%" headers="mcps1.1.4.1.3 "><p id="p54624844511"><a name="p54624844511"></a><a name="p54624844511"></a>Meaning: Input data of the model. A <span>torch.tensor</span><span> is equivalent to tuple(torch.tensor)</span>.</p>
<p id="p1195430114611"><a name="p1195430114611"></a><a name="p1195430114611"></a>Data type: tuple</p></td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section293415513458"></a>

Returns the modified torch.nn.Module model that can be used for distillation.

## Example<a name="zh-cn_topic_0240188739_section64231658994"></a>

```python
import amct_pytorch as amct
# Build the network graph structure to be distilled and quantized.
model = build_model()
model.load_state_dict(torch.load(state_dict_path))
input_data = tuple([torch.randn(input_shape)])

# Generate the compressed model.
compress_model = amct.create_distill_model(
                 config_json_file,
                 model,
                 input_data)
```
