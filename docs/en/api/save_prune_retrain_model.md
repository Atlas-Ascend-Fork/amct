# save\_prune\_retrain\_model<a name="ZH-CN_TOPIC_0000002548788567"></a>

<!-- md-trans-meta sourceCommit=4a47814eed18048e444f9a2acc42d9864ddc477f translatedAt=2026-10-16T08:39:14.246Z pushedAt=2026-11-05T01:12:03.428Z nl=2 -->

## Supported Products<a name="section18210163319189"></a>

<a name="zh-cn_topic_0000002548668559_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002548668559_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002548668559_p1883113061818"><a name="zh-cn_topic_0000002548668559_p1883113061818"></a><a name="zh-cn_topic_0000002548668559_p1883113061818"></a><span id="zh-cn_topic_0000002548668559_ph20833205312295"><a name="zh-cn_topic_0000002548668559_ph20833205312295"></a><a name="zh-cn_topic_0000002548668559_ph20833205312295"></a>Product</span></p></th>
<th class="cellrowborder" align="left" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002548668559_p783113012187"><a name="zh-cn_topic_0000002548668559_p783113012187"></a><a name="zh-cn_topic_0000002548668559_p783113012187"></a>Supported</p></th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002548668559_row574891710101"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002548668559_p177491117171015"><a name="zh-cn_topic_0000002548668559_p177491117171015"></a><a name="zh-cn_topic_0000002548668559_p177491117171015"></a><span id="zh-cn_topic_0000002548668559_ph2272194216543"><a name="zh-cn_topic_0000002548668559_ph2272194216543"></a><a name="zh-cn_topic_0000002548668559_ph2272194216543"></a>Ascend 950PR&950DT products</span></p></td>
<td class="cellrowborder" align="left" valign="top" width="42%" headers="mcps1.1.3.1.2 "><a name="zh-cn_topic_0000002548668559_ul1367712433612"></a><a name="zh-cn_topic_0000002548668559_ul1367712433612"></a><ul id="zh-cn_topic_0000002548668559_ul1367712433612"><li>Channel sparsity: Supported</li><li>4:2 structured sparsity: Not supported</li></ul></td>
</tr>
<tr id="zh-cn_topic_0000002548668559_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002548668559_p48327011813"><a name="zh-cn_topic_0000002548668559_p48327011813"></a><a name="zh-cn_topic_0000002548668559_p48327011813"></a><span id="zh-cn_topic_0000002548668559_ph583230201815"><a name="zh-cn_topic_0000002548668559_ph583230201815"></a><a name="zh-cn_topic_0000002548668559_ph583230201815"></a><term id="zh-cn_topic_0000002548668559_zh-cn_topic_0000001312391781_term1253731311225">Atlas A3 products</term></span></p></td>
<td class="cellrowborder" align="left" valign="top" width="42%" headers="mcps1.1.3.1.2 "><a name="zh-cn_topic_0000002548668559_ul66191179379"></a><a name="zh-cn_topic_0000002548668559_ul66191179379"></a><ul id="zh-cn_topic_0000002548668559_ul66191179379"><li>Channel sparsity: Supported</li><li>4:2 structured sparsity: Supported</li></ul></td>
</tr>
<tr id="zh-cn_topic_0000002548668559_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002548668559_p14832120181815"><a name="zh-cn_topic_0000002548668559_p14832120181815"></a><a name="zh-cn_topic_0000002548668559_p14832120181815"></a><span id="zh-cn_topic_0000002548668559_ph1483216010188"><a name="zh-cn_topic_0000002548668559_ph1483216010188"></a><a name="zh-cn_topic_0000002548668559_ph1483216010188"></a><term id="zh-cn_topic_0000002548668559_zh-cn_topic_0000001312391781_term11962195213215">Atlas A2 products</term></span></p></td>
<td class="cellrowborder" align="left" valign="top" width="42%" headers="mcps1.1.3.1.2 "><a name="zh-cn_topic_0000002548668559_ul19623147153713"></a><a name="zh-cn_topic_0000002548668559_ul19623147153713"></a><ul id="zh-cn_topic_0000002548668559_ul19623147153713"><li>Channel sparsity: Supported</li><li>4:2 structured sparsity: Supported</li></ul></td>
</tr>
</tbody>
</table>

**Note: For the 4:2 structured sparsity feature, calling the API on products marked with "x" does not report an error, but no performance gain is obtained.**

## Description<a name="section139511944317"></a>

The sparse API generates the final ONNX simulation model and deployment model based on the user's final retrained sparse model.

## Constraints<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_section1443392021419"></a>

For a sparse model, the two files generated through this API are ONNX files directly exported using PyTorch, with identical content, and their file names contain the **deploy** and **fake\_quant** keywords, respectively.

## Prototype<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_section428121323411"></a>

```python
save_prune_retrain_model(model, save_path, input_data, input_names=None, output_names=None, dynamic_axes=None)
```

## Parameters<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_section795991810344"></a>

<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="12.35%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_p1438834363520"></a>Parameter</p></th>
<th class="cellrowborder" valign="top" width="7.670000000000001%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p1769255516412"><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p1769255516412"></a><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p1769255516412"></a>Input/Output</p></th>
<th class="cellrowborder" valign="top" width="79.97999999999999%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p15231205416325"><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p15231205416325"></a><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p15231205416325"></a>Description</p></th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="12.35%" headers="mcps1.1.4.1.1 "><p id="p12108101818814"><a name="p12108101818814"></a><a name="p12108101818814"></a>model</p></td>
<td class="cellrowborder" valign="top" width="7.670000000000001%" headers="mcps1.1.4.1.2 "><p id="p13482131324113"><a name="p13482131324113"></a><a name="p13482131324113"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="79.97999999999999%" headers="mcps1.1.4.1.3 "><p id="p17691129844"><a name="p17691129844"></a><a name="p17691129844"></a>Meaning: PyTorch model that has been pruned.</p>
<p id="p9607425115012"><a name="p9607425115012"></a><a name="p9607425115012"></a>Data type: torch.nn.Module</p></td>
</tr>
<tr id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_row2997141123911"><td class="cellrowborder" valign="top" width="12.35%" headers="mcps1.1.4.1.1 "><p id="p1110815181988"><a name="p1110815181988"></a><a name="p1110815181988"></a>save_path</p></td>
<td class="cellrowborder" valign="top" width="7.670000000000001%" headers="mcps1.1.4.1.2 "><p id="p141811497475"><a name="p141811497475"></a><a name="p141811497475"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="79.97999999999999%" headers="mcps1.1.4.1.3 "><p id="p154071639641"><a name="p154071639641"></a><a name="p154071639641"></a>Meaning: Path for saving the compressed model. The path must include the model name prefix, for example, ./prune_model/*model.</p>
<p id="p3624142513506"><a name="p3624142513506"></a><a name="p3624142513506"></a>Data type: string</p></td>
</tr>
<tr id="row51780230810"><td class="cellrowborder" valign="top" width="12.35%" headers="mcps1.1.4.1.1 "><p id="p017819231680"><a name="p017819231680"></a><a name="p017819231680"></a>input_data</p></td>
<td class="cellrowborder" valign="top" width="7.670000000000001%" headers="mcps1.1.4.1.2 "><p id="p61604531884"><a name="p61604531884"></a><a name="p61604531884"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="79.97999999999999%" headers="mcps1.1.4.1.3 "><p id="p1666617401442"><a name="p1666617401442"></a><a name="p1666617401442"></a>Meaning: Model input data. A <span>torch.tensor</span><span> is equivalent to tuple(torch.tensor)</span>.</p>
<p id="p195118591985"><a name="p195118591985"></a><a name="p195118591985"></a>Data type: tuple</p></td>
</tr>
<tr id="row161899281186"><td class="cellrowborder" valign="top" width="12.35%" headers="mcps1.1.4.1.1 "><p id="p1810815183816"><a name="p1810815183816"></a><a name="p1810815183816"></a>input_names</p></td>
<td class="cellrowborder" valign="top" width="7.670000000000001%" headers="mcps1.1.4.1.2 "><p id="p1817012537814"><a name="p1817012537814"></a><a name="p1817012537814"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="79.97999999999999%" headers="mcps1.1.4.1.3 "><p id="p39306411349"><a name="p39306411349"></a><a name="p39306411349"></a>Meaning: Model input names, displayed in the saved sparse ONNX model.</p>
<p id="p78101226135"><a name="p78101226135"></a><a name="p78101226135"></a>Default value: None</p>
<p id="p1451615598811"><a name="p1451615598811"></a><a name="p1451615598811"></a>Data type: list(string)</p></td>
</tr>
<tr id="row9693030189"><td class="cellrowborder" valign="top" width="12.35%" headers="mcps1.1.4.1.1 "><p id="p18108718889"><a name="p18108718889"></a><a name="p18108718889"></a>output_names</p></td>
<td class="cellrowborder" valign="top" width="7.670000000000001%" headers="mcps1.1.4.1.2 "><p id="p91791053686"><a name="p91791053686"></a><a name="p91791053686"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="79.97999999999999%" headers="mcps1.1.4.1.3 "><p id="p512224319414"><a name="p512224319414"></a><a name="p512224319414"></a>Meaning: Model output names, displayed in the saved sparse ONNX model.</p>
<p id="p1631083319133"><a name="p1631083319133"></a><a name="p1631083319133"></a>Default value: None</p>
<p id="p115239594818"><a name="p115239594818"></a><a name="p115239594818"></a>Data type: list(string)</p></td>
</tr>
<tr id="row127516261687"><td class="cellrowborder" valign="top" width="12.35%" headers="mcps1.1.4.1.1 "><p id="p5275326787"><a name="p5275326787"></a><a name="p5275326787"></a>dynamic_axes</p></td>
<td class="cellrowborder" valign="top" width="7.670000000000001%" headers="mcps1.1.4.1.2 "><p id="p10185135316819"><a name="p10185135316819"></a><a name="p10185135316819"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="79.97999999999999%" headers="mcps1.1.4.1.3 "><p id="p164697449419"><a name="p164697449419"></a><a name="p164697449419"></a>Meaning: Specifies the dynamic axes of the model input and output. For example, for inputs (NCHW), N, H, and W are of uncertain size, and for outputs (NL), N is of uncertain size. The specification is as follows:</p>
<p id="p1217918111059"><a name="p1217918111059"></a><a name="p1217918111059"></a>{ "inputs": [0,2,3], "outputs": [0]}, where 0, 2, and 3 indicate the indices of N, H, and W, respectively.</p>
<p id="p1460565011316"><a name="p1460565011316"></a><a name="p1460565011316"></a>Default value: None</p>
<p id="p460555021319"><a name="p460555021319"></a><a name="p460555021319"></a>Data type: dict&lt;string, dict&lt;python:int, string&gt;&gt; or dict&lt;string, list(int)&gt;</p></td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_section293415513458"></a>

None

## Example<a name="zh-cn_topic_0254208276_section213682915617"></a>

```python
import amct_pytorch as amct
# Build the network graph structure to be compressed.
model = build_model()

# create selective prune model

# Train the retrain model and train the quantization factors.
train(pruned_retrain_model)
infer(pruned_retrain_model)

input_data = tuple([torch.randn(input_shape)])
save_path = os.path.join(OUTPUTS_DIR, 'custom_name')

# Insert the API for saving the combined compression model and convert it to an ONNX file.
amct.save_prune_retrain_model(
     pruned_retrain_model,
     save_path,
     input_data,
     input_names=['input'],
     output_names=['output'],
     dynamic_axes={'input':{0:'batch_size'}, 'output':{0:'batch_size'}})
```

Description of files written to disk:

- Accuracy simulation model file: an ONNX-format model file whose name contains **fake\_quant**. It can be used for accuracy simulation in the ONNX Runtime environment.
- Deployment model file: an ONNX-format model file whose name contains deploy. After conversion by the ATC tool, it can be deployed to an AI processor.
- (Optional) external data file. The sparsity feature does not distinguish between \*deploy.external and \*fakequant.external files:

    This type of file is generated only when the size of the saved accuracy simulation model and deployment model files \>= 2 GB. It is generated in the same directory as the compressed \*.onnx file and is used to store the data in tensors. Each tensor's data is saved in a separate file, and the file name is the same as the tensor name, for example, **conv\_1.weight**.

    When the compressed \*.onnx deployment model file is subsequently loaded by the ATC tool for model conversion, the tensor data in the external data file in the same directory is automatically read.

When the sparsity feature is executed again, the files output by this API will be overwritten.
