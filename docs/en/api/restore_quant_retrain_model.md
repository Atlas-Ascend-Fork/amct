# restore\_quant\_retrain\_model<a name="ZH-CN_TOPIC_0000002548668607"></a>

<!-- md-trans-meta sourceCommit=4a47814eed18048e444f9a2acc42d9864ddc477f translatedAt=2026-10-16T08:36:34.842Z pushedAt=2026-11-05T01:12:03.425Z nl=2 -->

## Supported Products<a name="section185612964420"></a>

<a name="zh-cn_topic_0000002517028880_table38301303189"></a>

|Product|Supported|
|--------------------------|--------|
|Ascend 950PR&950DT products|√|
|Atlas A3 products|√|
|Atlas A2 products|√|

## Description<a name="zh-cn_topic_0240187365_section15406195619561"></a>

The quantization-aware training API quantizes the input graph structure to be quantized according to the given quantization-aware training configuration file, inserts quantization-aware training-related operators (quantization-aware training layers for data and weights, and find-N layers) into the input graph structure, generates a quantization factor record file **record\_file**, loads the checkpoint weight parameters saved during training, and returns the modified torch.nn.Module quantization-aware training model.

## Prototype<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_section428121323411"></a>

```python
quant_retrain_model = restore_quant_retrain_model (config_file, model, record_file, input_data, pth_file, state_dict_name=None)
```

## Parameters<a name="section73811524135618"></a>

<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="10.70892910708929%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a>Parameter</p></th>
<th class="cellrowborder" valign="top" width="9.189081091890811%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240187365_p1769255516412"><a name="zh-cn_topic_0240187365_p1769255516412"></a><a name="zh-cn_topic_0240187365_p1769255516412"></a>Input/Output</p></th>
<th class="cellrowborder" valign="top" width="80.1019898010199%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0240187365_p15231205416325"><a name="zh-cn_topic_0240187365_p15231205416325"></a><a name="zh-cn_topic_0240187365_p15231205416325"></a>Description</p></th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="10.70892910708929%" headers="mcps1.1.4.1.1 "><p id="p9475183920481"><a name="p9475183920481"></a><a name="p9475183920481"></a>config_file</p></td>
<td class="cellrowborder" valign="top" width="9.189081091890811%" headers="mcps1.1.4.1.2 "><p id="p14475173934819"><a name="p14475173934819"></a><a name="p14475173934819"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.1019898010199%" headers="mcps1.1.4.1.3 "><p id="p1024512308252"><a name="p1024512308252"></a><a name="p1024512308252"></a>Meaning: User-generated quantization-aware training configuration file, which specifies the configuration of quantization layers in the model network.</p>
<p id="p174753395482"><a name="p174753395482"></a><a name="p174753395482"></a>Data type: string</p>
<p id="p18263256125215"><a name="p18263256125215"></a><a name="p18263256125215"></a>Constraint: The config.json input to this API must be the same as the config.json input to the <a href="create_quant_retrain_model.md">create_quant_retrain_model</a> API.</p></td>
</tr>
<tr id="row1280816105481"><td class="cellrowborder" valign="top" width="10.70892910708929%" headers="mcps1.1.4.1.1 "><p id="p347663984815"><a name="p347663984815"></a><a name="p347663984815"></a>model</p></td>
<td class="cellrowborder" valign="top" width="9.189081091890811%" headers="mcps1.1.4.1.2 "><p id="p124761039114818"><a name="p124761039114818"></a><a name="p124761039114818"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.1019898010199%" headers="mcps1.1.4.1.3 "><p id="p3779183722515"><a name="p3779183722515"></a><a name="p3779183722515"></a>Meaning: The original model to undergo quantization-aware training, without loaded weights.</p>
<p id="p4476103911488"><a name="p4476103911488"></a><a name="p4476103911488"></a>Data type: torch.nn.Module</p></td>
</tr>
<tr id="row085142094819"><td class="cellrowborder" valign="top" width="10.70892910708929%" headers="mcps1.1.4.1.1 "><p id="p847693974814"><a name="p847693974814"></a><a name="p847693974814"></a>record_file</p></td>
<td class="cellrowborder" valign="top" width="9.189081091890811%" headers="mcps1.1.4.1.2 "><p id="p2476183994811"><a name="p2476183994811"></a><a name="p2476183994811"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.1019898010199%" headers="mcps1.1.4.1.3 "><p id="p5139940172519"><a name="p5139940172519"></a><a name="p5139940172519"></a>Meaning: Path and name of the quantization factor record file.</p>
<p id="p5476133918487"><a name="p5476133918487"></a><a name="p5476133918487"></a>Data type: string</p></td>
</tr>
<tr id="row33518174483"><td class="cellrowborder" valign="top" width="10.70892910708929%" headers="mcps1.1.4.1.1 "><p id="p14476239194819"><a name="p14476239194819"></a><a name="p14476239194819"></a>input_data</p></td>
<td class="cellrowborder" valign="top" width="9.189081091890811%" headers="mcps1.1.4.1.2 "><p id="p047603920487"><a name="p047603920487"></a><a name="p047603920487"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.1019898010199%" headers="mcps1.1.4.1.3 "><p id="p167732416259"><a name="p167732416259"></a><a name="p167732416259"></a>Meaning: Input data of the model. A <span>torch.tensor</span><span>is treated as equivalent to a tuple(torch.tensor)</span>.</p>
<p id="p15476239164810"><a name="p15476239164810"></a><a name="p15476239164810"></a>Data type: tuple</p></td>
</tr>
<tr id="row63661714484"><td class="cellrowborder" valign="top" width="10.70892910708929%" headers="mcps1.1.4.1.1 "><p id="p19477173954813"><a name="p19477173954813"></a><a name="p19477173954813"></a>pth_file</p></td>
<td class="cellrowborder" valign="top" width="9.189081091890811%" headers="mcps1.1.4.1.2 "><p id="p184774396482"><a name="p184774396482"></a><a name="p184774396482"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.1019898010199%" headers="mcps1.1.4.1.3 "><p id="p66164342514"><a name="p66164342514"></a><a name="p66164342514"></a>Meaning: Weight file saved during training.</p>
<p id="p1247716394489"><a name="p1247716394489"></a><a name="p1247716394489"></a>Data type: string</p></td>
</tr>
<tr id="row1315215262487"><td class="cellrowborder" valign="top" width="10.70892910708929%" headers="mcps1.1.4.1.1 "><p id="p154772395483"><a name="p154772395483"></a><a name="p154772395483"></a>state_dict_name</p></td>
<td class="cellrowborder" valign="top" width="9.189081091890811%" headers="mcps1.1.4.1.2 "><p id="p20477939124816"><a name="p20477939124816"></a><a name="p20477939124816"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.1019898010199%" headers="mcps1.1.4.1.3 "><p id="p4735114462517"><a name="p4735114462517"></a><a name="p4735114462517"></a>Meaning: Key value corresponding to the weight in the weight file.</p>
<p id="p225724119501"><a name="p225724119501"></a><a name="p225724119501"></a>Default value: None</p>
<p id="p94771439144818"><a name="p94771439144818"></a><a name="p94771439144818"></a>Data type: string</p></td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section293415513458"></a>

Returns the modified torch.nn.Module quantization-aware training model.

## Example<a name="zh-cn_topic_0240188739_section64231658994"></a>

```python
import amct_pytorch as amct
# Build the network graph structure to be quantized.
model = build_model()
input_data = tuple([torch.randn(input_shape)])
 
scale_offset_record_file = os.path.join(TMP, 'scale_offset_record.txt')
# Insert the quantization API.
quant_retrain_model = amct.restore_quant_retrain_model(
               config_json_file,
               model,
               scale_offset_record_file,
               input_data,
               pth_file)
```
