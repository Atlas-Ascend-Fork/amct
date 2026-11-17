# save\_compressed\_retrain\_model<a name="ZH-CN_TOPIC_0000002517028750"></a>

<!-- md-trans-meta sourceCommit=4a47814eed18048e444f9a2acc42d9864ddc477f translatedAt=2026-10-16T08:37:33.987Z pushedAt=2026-11-05T01:12:03.427Z nl=2 -->

## Supported Products<a name="section329012164010"></a>

<a name="zh-cn_topic_0000002517188752_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002517188752_row20831180131817"><th class="cellrowborder" valign="top" width="47.92%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000002517188752_p1883113061818"><a name="zh-cn_topic_0000002517188752_p1883113061818"></a><a name="zh-cn_topic_0000002517188752_p1883113061818"></a><span id="zh-cn_topic_0000002517188752_ph20833205312295"><a name="zh-cn_topic_0000002517188752_ph20833205312295"></a><a name="zh-cn_topic_0000002517188752_ph20833205312295"></a>Product</span></p></th>
<th class="cellrowborder" align="left" valign="top" width="52.080000000000005%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000002517188752_p783113012187"><a name="zh-cn_topic_0000002517188752_p783113012187"></a><a name="zh-cn_topic_0000002517188752_p783113012187"></a>Supported</p></th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002517188752_row574891710101"><td class="cellrowborder" valign="top" width="47.92%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002517188752_p177491117171015"><a name="zh-cn_topic_0000002517188752_p177491117171015"></a><a name="zh-cn_topic_0000002517188752_p177491117171015"></a><span id="zh-cn_topic_0000002517188752_ph2272194216543"><a name="zh-cn_topic_0000002517188752_ph2272194216543"></a><a name="zh-cn_topic_0000002517188752_ph2272194216543"></a>Ascend 950PR&950DT products</span></p></td>
<td class="cellrowborder" align="left" valign="top" width="52.080000000000005%" headers="mcps1.1.3.1.2 "><a name="zh-cn_topic_0000002517188752_ul1367712433612"></a><a name="zh-cn_topic_0000002517188752_ul1367712433612"></a><ul id="zh-cn_topic_0000002517188752_ul1367712433612"><li>Quantization-aware training: <a name="zh-cn_topic_0000002517188752_ul5366163611332"></a><a name="zh-cn_topic_0000002517188752_ul5366163611332"></a><ul id="zh-cn_topic_0000002517188752_ul5366163611332"><li>INT8 quantization: √</li><li>INT4 quantization: x</li></ul>
</li><li>Channel sparsity: √</li><li>4:2 structured sparsity: x</li></ul></td>
</tr>
<tr id="zh-cn_topic_0000002517188752_row220181016240"><td class="cellrowborder" valign="top" width="47.92%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002517188752_p48327011813"><a name="zh-cn_topic_0000002517188752_p48327011813"></a><a name="zh-cn_topic_0000002517188752_p48327011813"></a><span id="zh-cn_topic_0000002517188752_ph583230201815"><a name="zh-cn_topic_0000002517188752_ph583230201815"></a><a name="zh-cn_topic_0000002517188752_ph583230201815"></a><term id="zh-cn_topic_0000002517188752_zh-cn_topic_0000001312391781_term1253731311225">Atlas A3 products</term></span></p></td>
<td class="cellrowborder" align="left" valign="top" width="52.080000000000005%" headers="mcps1.1.3.1.2 "><a name="zh-cn_topic_0000002517188752_ul66191179379"></a><a name="zh-cn_topic_0000002517188752_ul66191179379"></a><ul id="zh-cn_topic_0000002517188752_ul66191179379"><li>Quantization-aware training: <a name="zh-cn_topic_0000002517188752_ul7274174217362"></a><a name="zh-cn_topic_0000002517188752_ul7274174217362"></a><ul id="zh-cn_topic_0000002517188752_ul7274174217362"><li>INT8 quantization: √</li><li>INT4 quantization: x</li></ul>
</li><li>Channel sparsity: √</li><li>4:2 structured sparsity: √</li></ul></td>
</tr>
<tr id="zh-cn_topic_0000002517188752_row173226882415"><td class="cellrowborder" valign="top" width="47.92%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000002517188752_p14832120181815"><a name="zh-cn_topic_0000002517188752_p14832120181815"></a><a name="zh-cn_topic_0000002517188752_p14832120181815"></a><span id="zh-cn_topic_0000002517188752_ph1483216010188"><a name="zh-cn_topic_0000002517188752_ph1483216010188"></a><a name="zh-cn_topic_0000002517188752_ph1483216010188"></a><term id="zh-cn_topic_0000002517188752_zh-cn_topic_0000001312391781_term11962195213215">Atlas A2 products</term></span></p></td>
<td class="cellrowborder" align="left" valign="top" width="52.080000000000005%" headers="mcps1.1.3.1.2 "><a name="zh-cn_topic_0000002517188752_ul19623147153713"></a><a name="zh-cn_topic_0000002517188752_ul19623147153713"></a><ul id="zh-cn_topic_0000002517188752_ul19623147153713"><li>Quantization-aware training: <a name="zh-cn_topic_0000002517188752_ul0992647173610"></a><a name="zh-cn_topic_0000002517188752_ul0992647173610"></a><ul id="zh-cn_topic_0000002517188752_ul0992647173610"><li>INT8 quantization: √</li><li>INT4 quantization: x</li></ul>
</li><li>Channel sparsity: √</li><li>4:2 structured sparsity: √</li></ul></td>
</tr>
</tbody>
</table>

**Note: For products marked with "x" in the feature, calling the API does not report an error, but no performance gain can be obtained.**

## Description<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_section15406195619561"></a>

Static combination compression API. Based on the user's final retrained model, it generates the final static combination compression accuracy simulation model and deployment model.

## Prototype<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_section428121323411"></a>

```python
save_compressed_retrain_model(model, record_file, save_path, input_data, input_names=None, output_names=None, dynamic_axes=None)
```

## Parameters<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_section795991810344"></a>

<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="11.16%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_p1438834363520"></a>Parameter</p></th>
<th class="cellrowborder" valign="top" width="7.95%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p1769255516412"><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p1769255516412"></a><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p1769255516412"></a>Input/Output</p></th>
<th class="cellrowborder" valign="top" width="80.89%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p15231205416325"><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p15231205416325"></a><a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_p15231205416325"></a>Description</p></th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="11.16%" headers="mcps1.1.4.1.1 "><p id="p12108101818814"><a name="p12108101818814"></a><a name="p12108101818814"></a>model</p></td>
<td class="cellrowborder" valign="top" width="7.95%" headers="mcps1.1.4.1.2 "><p id="p13482131324113"><a name="p13482131324113"></a><a name="p13482131324113"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.89%" headers="mcps1.1.4.1.3 "><p id="p20897173423615"><a name="p20897173423615"></a><a name="p20897173423615"></a>Meaning: PyTorch model that has undergone static combination compression.</p>
<p id="p9607425115012"><a name="p9607425115012"></a><a name="p9607425115012"></a>Data type: torch.nn.Module</p></td>
</tr>
<tr id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_row16254133823918"><td class="cellrowborder" valign="top" width="11.16%" headers="mcps1.1.4.1.1 "><p id="p810815183816"><a name="p810815183816"></a><a name="p810815183816"></a>record_file</p></td>
<td class="cellrowborder" valign="top" width="7.95%" headers="mcps1.1.4.1.2 "><p id="p175011134411"><a name="p175011134411"></a><a name="p175011134411"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.89%" headers="mcps1.1.4.1.3 "><p id="p145771141123612"><a name="p145771141123612"></a><a name="p145771141123612"></a>Meaning: Path and name of the file that records sparsity and quantization factors.</p>
<p id="p17619162585010"><a name="p17619162585010"></a><a name="p17619162585010"></a>Data type: string</p></td>
</tr>
<tr id="zh-cn_topic_0254208276_zh-cn_topic_0240188739_row2997141123911"><td class="cellrowborder" valign="top" width="11.16%" headers="mcps1.1.4.1.1 "><p id="p1110815181988"><a name="p1110815181988"></a><a name="p1110815181988"></a>save_path</p></td>
<td class="cellrowborder" valign="top" width="7.95%" headers="mcps1.1.4.1.2 "><p id="p141811497475"><a name="p141811497475"></a><a name="p141811497475"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.89%" headers="mcps1.1.4.1.3 "><p id="p6973842133613"><a name="p6973842133613"></a><a name="p6973842133613"></a>Meaning: Path used to save the compressed model.</p>
<p id="p3624142513506"><a name="p3624142513506"></a><a name="p3624142513506"></a>Data type: string</p></td>
</tr>
<tr id="row51780230810"><td class="cellrowborder" valign="top" width="11.16%" headers="mcps1.1.4.1.1 "><p id="p017819231680"><a name="p017819231680"></a><a name="p017819231680"></a>input_data</p></td>
<td class="cellrowborder" valign="top" width="7.95%" headers="mcps1.1.4.1.2 "><p id="p61604531884"><a name="p61604531884"></a><a name="p61604531884"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.89%" headers="mcps1.1.4.1.3 "><p id="p10320204414363"><a name="p10320204414363"></a><a name="p10320204414363"></a>Meaning: Input data of the model. A <span>torch.tensor</span><span> is equivalent to a tuple(torch.tensor)</span>.</p>
<p id="p195118591985"><a name="p195118591985"></a><a name="p195118591985"></a>Data type: tuple</p></td>
</tr>
<tr id="row161899281186"><td class="cellrowborder" valign="top" width="11.16%" headers="mcps1.1.4.1.1 "><p id="p1810815183816"><a name="p1810815183816"></a><a name="p1810815183816"></a>input_names</p></td>
<td class="cellrowborder" valign="top" width="7.95%" headers="mcps1.1.4.1.2 "><p id="p1817012537814"><a name="p1817012537814"></a><a name="p1817012537814"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.89%" headers="mcps1.1.4.1.3 "><p id="p4611124553611"><a name="p4611124553611"></a><a name="p4611124553611"></a>Meaning: Names of the model inputs, displayed in the saved quantized ONNX model.</p>
<p id="p78101226135"><a name="p78101226135"></a><a name="p78101226135"></a>Default value: None</p>
<p id="p1451615598811"><a name="p1451615598811"></a><a name="p1451615598811"></a>Data type: list(string)</p></td>
</tr>
<tr id="row9693030189"><td class="cellrowborder" valign="top" width="11.16%" headers="mcps1.1.4.1.1 "><p id="p18108718889"><a name="p18108718889"></a><a name="p18108718889"></a>output_names</p></td>
<td class="cellrowborder" valign="top" width="7.95%" headers="mcps1.1.4.1.2 "><p id="p91791053686"><a name="p91791053686"></a><a name="p91791053686"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.89%" headers="mcps1.1.4.1.3 "><p id="p915214753618"><a name="p915214753618"></a><a name="p915214753618"></a>Meaning: Names of the model outputs, displayed in the saved quantized onnx model.</p>
<p id="p1631083319133"><a name="p1631083319133"></a><a name="p1631083319133"></a>Default value: None</p>
<p id="p115239594818"><a name="p115239594818"></a><a name="p115239594818"></a>Data type: list(string)</p></td>
</tr>
<tr id="row127516261687"><td class="cellrowborder" valign="top" width="11.16%" headers="mcps1.1.4.1.1 "><p id="p5275326787"><a name="p5275326787"></a><a name="p5275326787"></a>dynamic_axes</p></td>
<td class="cellrowborder" valign="top" width="7.95%" headers="mcps1.1.4.1.2 "><p id="p10185135316819"><a name="p10185135316819"></a><a name="p10185135316819"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="80.89%" headers="mcps1.1.4.1.3 "><p id="p0969164883618"><a name="p0969164883618"></a><a name="p0969164883618"></a>Meaning: Specifies the dynamic axes of the model input and output. For example, for inputs (NCHW), where N, H, and W are of uncertain size, and outputs (NL), where N is of uncertain size, the specification is:</p>
<p id="p13121617183715"><a name="p13121617183715"></a><a name="p13121617183715"></a>{ "inputs": [0,2,3], "outputs": [0]}, where 0, 2, and 3 represent the indices of N, H, and W, respectively.</p>
<p id="p1460565011316"><a name="p1460565011316"></a><a name="p1460565011316"></a>Default value: None</p>
<p id="p460555021319"><a name="p460555021319"></a><a name="p460555021319"></a>Data type: dict&lt;string, dict&lt;python:int, string&gt;&gt; or dict&lt;string, list(int)&gt;</p></td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_zh-cn_topic_0122830089_section293415513458"></a>

None

## Constraints<a name="zh-cn_topic_0254208276_zh-cn_topic_0240188739_section1443392021419"></a>

If the model only undergoes sparsity (without quantization), the two files generated by this API are ONNX files directly exported using PyTorch, with identical content, and their file names include the **deploy** and **fake\_quant** keywords, respectively.

## Example<a name="zh-cn_topic_0254208276_section213682915617"></a>

```python
import amct_pytorch as amct
# Build the network graph to be compressed.
model = build_model()

# create compressed model

# Train the retrain model and train the quantization factors.
train(compressed_retrain_model)
infer(compressed_retrain_model)

input_data = tuple([torch.randn(input_shape)])
save_path = os.path.join(OUTPUTS_DIR, 'custom_name')
record_file = os.path.join(TMP, 'compressed_record.txt')

# Insert the API for saving the combined compression model and convert it to an ONNX file.
amct.save_compressed_retrain_model(
     compressed_retrain_model,
     record_file,
     save_path,
     input_data,
     input_names=['input'],
     output_names=['output'],
     dynamic_axes={'input':{0:'batch_size'}, 'output':{0:'batch_size'}})
```

Description of the generated files:

- Accuracy simulation model file: an ONNX-format model file whose name contains **fake\_quant**. It can be used for accuracy simulation in the ONNX Runtime environment.
- Deployment model file: an ONNX-format model file whose name contains **deploy**. After conversion by the ATC tool, it can be deployed to AI processors.
- (Optional) \*.external files, including \*deploy.external and \*fakequant.external:

    These files are generated only when the saved accuracy simulation model and deployment model file sizes \>=2 GB, and they are generated in the same directory as the compressed \*.onnx model file. They are used to save the data in tensors, with each tensor's data saved in a separate \*.external file. The file name is the same as the tensor name, for example, **_conv1.weight_\_deploy.external** and **_conv1.weight_\_fakequant.external**.

    When the compressed \*.onnx deployment model file is subsequently loaded by the ATC tool for model conversion, the tensor data in the \*.external files in the same directory is automatically read.

When static combination compression is re-executed, the files output by this API will be overwritten.
