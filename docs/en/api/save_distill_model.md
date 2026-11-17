# save\_distill\_model<a name="ZH-CN_TOPIC_0000002517188690"></a>

<!-- md-trans-meta sourceCommit=4a47814eed18048e444f9a2acc42d9864ddc477f translatedAt=2026-10-16T08:37:48.967Z pushedAt=2026-11-05T01:12:03.426Z nl=2 -->

## Supported Products<a name="section185612964420"></a>

<a name="zh-cn_topic_0000002517188794_table38301303189"></a>

| Product | Supported |
| ------------------------------------------- | -------- |
| Ascend 950PR&950DT products | √ |
| Atlas A3 products | √ |
| Atlas A2 products | √ |

## Description<a name="zh-cn_topic_0240187365_section15406195619561"></a>

The distillation API generates the final quantization accuracy simulation model and the quantization deployment model based on the user's final distilled model.

## Prototype<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_section428121323411"></a>

```python
save_distill_model(model, save_path, input_data, record_file=None, input_names=None, output_names=None, dynamic_axes=None)
```

## Parameters<a name="section73811524135618"></a>

<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="9.84%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a>Parameter</p></th>
<th class="cellrowborder" valign="top" width="7.66%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240187365_p1769255516412"><a name="zh-cn_topic_0240187365_p1769255516412"></a><a name="zh-cn_topic_0240187365_p1769255516412"></a>Input/Output</p></th>
<th class="cellrowborder" valign="top" width="82.5%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0240187365_p15231205416325"><a name="zh-cn_topic_0240187365_p15231205416325"></a><a name="zh-cn_topic_0240187365_p15231205416325"></a>Description</p></th>
</tr>
</thead>
<tbody><tr id="row148501718185117"><td class="cellrowborder" valign="top" width="9.84%" headers="mcps1.1.4.1.1 "><p id="p15703125017513"><a name="p15703125017513"></a><a name="p15703125017513"></a>model</p></td>
<td class="cellrowborder" valign="top" width="7.66%" headers="mcps1.1.4.1.2 "><p id="p970311508516"><a name="p970311508516"></a><a name="p970311508516"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="82.5%" headers="mcps1.1.4.1.3 "><p id="p1246265654716"><a name="p1246265654716"></a><a name="p1246265654716"></a>Meaning: Quantized model after distillation.</p>
<p id="p157031950195111"><a name="p157031950195111"></a><a name="p157031950195111"></a>Data type: torch.nn.Module</p></td>
</tr>
<tr id="row588872319519"><td class="cellrowborder" valign="top" width="9.84%" headers="mcps1.1.4.1.1 "><p id="p8703550155111"><a name="p8703550155111"></a><a name="p8703550155111"></a>save_path</p></td>
<td class="cellrowborder" valign="top" width="7.66%" headers="mcps1.1.4.1.2 "><p id="p177031550185118"><a name="p177031550185118"></a><a name="p177031550185118"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="82.5%" headers="mcps1.1.4.1.3 "><p id="p660915344817"><a name="p660915344817"></a><a name="p660915344817"></a>Meaning: Storage path of the quantized model after distillation. The path must include the model name prefix, for example, ./quantized_model/*model.</p>
<p id="p87041950175120"><a name="p87041950175120"></a><a name="p87041950175120"></a>Data type: string</p></td>
</tr>
<tr id="row1739023214510"><td class="cellrowborder" valign="top" width="9.84%" headers="mcps1.1.4.1.1 "><p id="p8704105085113"><a name="p8704105085113"></a><a name="p8704105085113"></a>input_data</p></td>
<td class="cellrowborder" valign="top" width="7.66%" headers="mcps1.1.4.1.2 "><p id="p3704135016518"><a name="p3704135016518"></a><a name="p3704135016518"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="82.5%" headers="mcps1.1.4.1.3 "><p id="p12260524816"><a name="p12260524816"></a><a name="p12260524816"></a>Meaning: Input data of the model. A <span>torch.tensor</span><span> is equivalent to a tuple(torch.tensor)</span>.</p>
<p id="p37045500510"><a name="p37045500510"></a><a name="p37045500510"></a>Data type: tuple</p></td>
</tr>
<tr id="row14467165613613"><td class="cellrowborder" valign="top" width="9.84%" headers="mcps1.1.4.1.1 "><p id="p372671579"><a name="p372671579"></a><a name="p372671579"></a>record_file</p></td>
<td class="cellrowborder" valign="top" width="7.66%" headers="mcps1.1.4.1.2 "><p id="p157267118720"><a name="p157267118720"></a><a name="p157267118720"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="82.5%" headers="mcps1.1.4.1.3 "><p id="p111303704818"><a name="p111303704818"></a><a name="p111303704818"></a>Meaning: Path and name of the quantization factor record file.</p>
<p id="p167321191674"><a name="p167321191674"></a><a name="p167321191674"></a>Default value: None</p>
<p id="p12726411675"><a name="p12726411675"></a><a name="p12726411675"></a>Data type: string</p>
<p id="p17762111782"><a name="p17762111782"></a><a name="p17762111782"></a>Constraint: When the input value is None, the quantization factor record file is stored in the amct_log folder.</p></td>
</tr>
<tr id="row539163214513"><td class="cellrowborder" valign="top" width="9.84%" headers="mcps1.1.4.1.1 "><p id="p37041850185117"><a name="p37041850185117"></a><a name="p37041850185117"></a>input_names</p></td>
<td class="cellrowborder" valign="top" width="7.66%" headers="mcps1.1.4.1.2 "><p id="p117041506519"><a name="p117041506519"></a><a name="p117041506519"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="82.5%" headers="mcps1.1.4.1.3 "><p id="p1715710924811"><a name="p1715710924811"></a><a name="p1715710924811"></a>Meaning: Names of the model input nodes, used for display in the saved quantized onnx model.</p>
<p id="p4704195095118"><a name="p4704195095118"></a><a name="p4704195095118"></a>Default value: None</p>
<p id="p14704850115116"><a name="p14704850115116"></a><a name="p14704850115116"></a>Data type: list(string)</p></td>
</tr>
<tr id="row1391532115114"><td class="cellrowborder" valign="top" width="9.84%" headers="mcps1.1.4.1.1 "><p id="p107040501510"><a name="p107040501510"></a><a name="p107040501510"></a>output_names</p></td>
<td class="cellrowborder" valign="top" width="7.66%" headers="mcps1.1.4.1.2 "><p id="p16705195017512"><a name="p16705195017512"></a><a name="p16705195017512"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="82.5%" headers="mcps1.1.4.1.3 "><p id="p9720410144818"><a name="p9720410144818"></a><a name="p9720410144818"></a>Meaning: Name of the model output node, used for display in the saved quantized onnx model.</p>
<p id="p147058501513"><a name="p147058501513"></a><a name="p147058501513"></a>Default value: None</p>
<p id="p170525018516"><a name="p170525018516"></a><a name="p170525018516"></a>Data type: list(string)</p></td>
</tr>
<tr id="row1021713282517"><td class="cellrowborder" valign="top" width="9.84%" headers="mcps1.1.4.1.1 "><p id="p1705195075116"><a name="p1705195075116"></a><a name="p1705195075116"></a>dynamic_axes</p></td>
<td class="cellrowborder" valign="top" width="7.66%" headers="mcps1.1.4.1.2 "><p id="p117054506516"><a name="p117054506516"></a><a name="p117054506516"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="82.5%" headers="mcps1.1.4.1.3 "><p id="p172491012114819"><a name="p172491012114819"></a><a name="p172491012114819"></a>Meaning: Specifies the dynamic axes of the model input and output. For example, for inputs (NCHW) where N, H, and W are of uncertain size, and outputs (NL) where N is of uncertain size, the specification is {"inputs": [0,2,3], "outputs": [0]}, where 0, 2, and 3 indicate the indices of N, H, and W respectively.</p>
<p id="p17705165016517"><a name="p17705165016517"></a><a name="p17705165016517"></a>Default value: None</p>
<p id="p270525085117"><a name="p270525085117"></a><a name="p270525085117"></a>Data type: dict&lt;string, dict&lt;python:int, string&gt;&gt; or dict&lt;string, list(int)<em id="i17705135016515"><a name="i17705135016515"></a><a name="i17705135016515"></a>&gt;</em></p></td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section293415513458"></a>

None

## Example<a name="zh-cn_topic_0240188739_section64231658994"></a>

```python
import amct_pytorch as amct
# Build the network graph structure to be distilled and quantized.
model = build_model()
model.load_state_dict(torch.load(state_dict_path))
input_data = tuple([torch.randn(input_shape)])
 
# Insert the distillation API and save the distilled model as an ONNX file.
amct.save_distill_model(
               model, 
               "./model/distilled",
               input_data,
               record_file="./results/records.txt",
               input_names=['input'],
               output_names=['output'],
               dynamic_axes={'input':{0: 'batch_size'},
                             'output':{0: 'batch_size'}})
```

Files written to disk:

- Accuracy simulation model file: an ONNX-format model file whose name contains **fake\_quant**. It can be used for accuracy simulation in the ONNX Runtime environment.

- Deployment model file: an ONNX-format model file whose name contains **deploy**. After conversion by the ATC conversion tool, it can be deployed to the AI processor.

When distillation is re-executed, the files output by this API will be overwritten.
