# save\_quant\_retrain\_model<a name="ZH-CN_TOPIC_0000002548788575"></a>

<!-- md-trans-meta sourceCommit=9d54d17e0942052193c827180dc43312c580acc7 translatedAt=2026-10-16T08:40:17.490Z pushedAt=2026-11-05T01:12:03.430Z nl=2 -->

## Supported Products<a name="section185612964420"></a>

| Product | Supported |
| ------------------------------------------- | -------- |
| Ascend 950PR&950DT products | √ |
| Atlas A3 products | √ |
| Atlas A2 products | √ |

## Description<a name="zh-cn_topic_0240187365_section15406195619561"></a>

A quantization-aware training API. Based on the user's final retrained model, it inserts operators such as AscendQuant and AscendDequant to generate the final quantized accuracy simulation model and the quantized deployment model.

## Prototype<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_section428121323411"></a>

```python
save_quant_retrain_model (config_file, model, record_file, save_path,  input_data, input_names=None, output_names=None, dynamic_axes=None)
```

## Parameters<a name="section73811524135618"></a>

<a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="11.35%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240187365_zh-cn_topic_0122830089_p1438834363520"></a>Parameter</p></th>
<th class="cellrowborder" valign="top" width="9.33%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240187365_p1769255516412"><a name="zh-cn_topic_0240187365_p1769255516412"></a><a name="zh-cn_topic_0240187365_p1769255516412"></a>Input/Output</p></th>
<th class="cellrowborder" valign="top" width="79.32000000000001%" id="mcps1.1.4.1.3"><p id="p14891104892614"><a name="p14891104892614"></a><a name="p14891104892614"></a>Description</p></th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0240187365_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="11.35%" headers="mcps1.1.4.1.1 "><p id="p17702250135118"><a name="p17702250135118"></a><a name="p17702250135118"></a>config_file</p></td>
<td class="cellrowborder" valign="top" width="9.33%" headers="mcps1.1.4.1.2 "><p id="p170365015514"><a name="p170365015514"></a><a name="p170365015514"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="79.32000000000001%" headers="mcps1.1.4.1.3 "><p id="p11668105412611"><a name="p11668105412611"></a><a name="p11668105412611"></a>Meaning: User-generated quantization-aware training configuration file, used to specify the configuration of quantization layers in the model network.</p>
<p id="p27038501514"><a name="p27038501514"></a><a name="p27038501514"></a>Data type: string</p></td>
</tr>
<tr id="row148501718185117"><td class="cellrowborder" valign="top" width="11.35%" headers="mcps1.1.4.1.1 "><p id="p15703125017513"><a name="p15703125017513"></a><a name="p15703125017513"></a>model</p></td>
<td class="cellrowborder" valign="top" width="9.33%" headers="mcps1.1.4.1.2 "><p id="p970311508516"><a name="p970311508516"></a><a name="p970311508516"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="79.32000000000001%" headers="mcps1.1.4.1.3 "><p id="p72892446271"><a name="p72892446271"></a><a name="p72892446271"></a>Meaning: Quantized model after quantization-aware training.</p>
<p id="p157031950195111"><a name="p157031950195111"></a><a name="p157031950195111"></a>Data type: torch.nn.Module</p></td>
</tr>
<tr id="row3887102315512"><td class="cellrowborder" valign="top" width="11.35%" headers="mcps1.1.4.1.1 "><p id="p127036506519"><a name="p127036506519"></a><a name="p127036506519"></a>record_file</p></td>
<td class="cellrowborder" valign="top" width="9.33%" headers="mcps1.1.4.1.2 "><p id="p17035502517"><a name="p17035502517"></a><a name="p17035502517"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="79.32000000000001%" headers="mcps1.1.4.1.3 "><p id="p37411454275"><a name="p37411454275"></a><a name="p37411454275"></a>Meaning: Path and name of the quantization factor record file.</p>
<p id="p9703195055110"><a name="p9703195055110"></a><a name="p9703195055110"></a>Data type: string</p></td>
</tr>
<tr id="row588872319519"><td class="cellrowborder" valign="top" width="11.35%" headers="mcps1.1.4.1.1 "><p id="p8703550155111"><a name="p8703550155111"></a><a name="p8703550155111"></a>save_path</p></td>
<td class="cellrowborder" valign="top" width="9.33%" headers="mcps1.1.4.1.2 "><p id="p177031550185118"><a name="p177031550185118"></a><a name="p177031550185118"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="79.32000000000001%" headers="mcps1.1.4.1.3 "><p id="p189971465277"><a name="p189971465277"></a><a name="p189971465277"></a>Meaning: Path for storing the quantized model. This path must include the model name prefix, for example, ./quantized_model/*model.</p>
<p id="p87041950175120"><a name="p87041950175120"></a><a name="p87041950175120"></a>Data type: string</p></td>
</tr>
<tr id="row1739023214510"><td class="cellrowborder" valign="top" width="11.35%" headers="mcps1.1.4.1.1 "><p id="p8704105085113"><a name="p8704105085113"></a><a name="p8704105085113"></a>input_data</p></td>
<td class="cellrowborder" valign="top" width="9.33%" headers="mcps1.1.4.1.2 "><p id="p3704135016518"><a name="p3704135016518"></a><a name="p3704135016518"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="79.32000000000001%" headers="mcps1.1.4.1.3 "><p id="p426418480279"><a name="p426418480279"></a><a name="p426418480279"></a>Meaning: Input data of the model. A <span>torch.tensor</span><span> is equivalent to a tuple(torch.tensor)</span>.</p>
<p id="p37045500510"><a name="p37045500510"></a><a name="p37045500510"></a>Data type: tuple</p></td>
</tr>
<tr id="row539163214513"><td class="cellrowborder" valign="top" width="11.35%" headers="mcps1.1.4.1.1 "><p id="p37041850185117"><a name="p37041850185117"></a><a name="p37041850185117"></a>input_names</p></td>
<td class="cellrowborder" valign="top" width="9.33%" headers="mcps1.1.4.1.2 "><p id="p117041506519"><a name="p117041506519"></a><a name="p117041506519"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="79.32000000000001%" headers="mcps1.1.4.1.3 "><p id="p18609049122716"><a name="p18609049122716"></a><a name="p18609049122716"></a>Meaning: Names of the model inputs, displayed in the saved quantized ONNX model.</p>
<p id="p4704195095118"><a name="p4704195095118"></a><a name="p4704195095118"></a>Default value: None</p>
<p id="p14704850115116"><a name="p14704850115116"></a><a name="p14704850115116"></a>Data type: list(string)</p></td>
</tr>
<tr id="row1391532115114"><td class="cellrowborder" valign="top" width="11.35%" headers="mcps1.1.4.1.1 "><p id="p107040501510"><a name="p107040501510"></a><a name="p107040501510"></a>output_names</p></td>
<td class="cellrowborder" valign="top" width="9.33%" headers="mcps1.1.4.1.2 "><p id="p16705195017512"><a name="p16705195017512"></a><a name="p16705195017512"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="79.32000000000001%" headers="mcps1.1.4.1.3 "><p id="p12926523273"><a name="p12926523273"></a><a name="p12926523273"></a>Meaning: Names of the model outputs, displayed in the saved quantized ONNX model.</p>
<p id="p147058501513"><a name="p147058501513"></a><a name="p147058501513"></a>Default value: None</p>
<p id="p170525018516"><a name="p170525018516"></a><a name="p170525018516"></a>Data type: list(string)</p></td>
</tr>
<tr id="row1021713282517"><td class="cellrowborder" valign="top" width="11.35%" headers="mcps1.1.4.1.1 "><p id="p1705195075116"><a name="p1705195075116"></a><a name="p1705195075116"></a>dynamic_axes</p></td>
<td class="cellrowborder" valign="top" width="9.33%" headers="mcps1.1.4.1.2 "><p id="p117054506516"><a name="p117054506516"></a><a name="p117054506516"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="79.32000000000001%" headers="mcps1.1.4.1.3 "><p id="p13724125717271"><a name="p13724125717271"></a><a name="p13724125717271"></a>Meaning: Specifies the dynamic axes of the model inputs and outputs. For example, for inputs (NCHW), where N, H, and W are of uncertain size, and outputs (NL), where N is of uncertain size, the specification is as follows:</p>
<p id="p064281193019"><a name="p064281193019"></a><a name="p064281193019"></a>{"inputs": [0,2,3], "outputs": [0]}, where 0, 2, and 3 indicate the indices of the positions of N, H, and W, respectively.</p>
<p id="p17705165016517"><a name="p17705165016517"></a><a name="p17705165016517"></a>Default value: None</p>
<p id="p270525085117"><a name="p270525085117"></a><a name="p270525085117"></a>Data type: dict&lt;string, dict&lt;python:int, string&gt;&gt; or dict&lt;string, list(int)<em id="i17705135016515"><a name="i17705135016515"></a><a name="i17705135016515"></a>&gt;</em></p>
<p id="p4451162412318"><a name="p4451162412318"></a><a name="p4451162412318"></a>Constraint: The int value must be a non-negative number.</p></td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0240188739_zh-cn_topic_0122830089_section293415513458"></a>

None

## Example<a name="zh-cn_topic_0240188739_section64231658994"></a>

```python
import amct_pytorch as amct
# Build the network graph structure to be quantized.
model = build_model()
model.load_state_dict(torch.load(state_dict_path))
input_data = tuple([torch.randn(input_shape)])
# Train the quantized retrain model and train the quantization factors.
train_model(quant_retrain_model, input_batch)
# Infer the quantized retrain model and export the quantization factors.
infer_model(quant_retrain_model, input_batch)

# Insert the quantization API and save the quantization-aware trained model as an ONNX file.
amct.save_quant_retrain_model(
               config_json_file,
               quant_retrain_model,
               record_file,
               save_path="./results/model",
               input_data=input_data,
               input_names=['input'],
               output_names=['output'],
               dynamic_axes={'input':{0: 'batch_size'},
                             'output':{0: 'batch_size'}})
```

Description of the generated files:

- Accuracy simulation model file: An ONNX-format model file whose name contains **fake\_quant**. It can be used for accuracy simulation in the ONNX Runtime environment.
- Deployment model file: An ONNX-format model file whose name contains **deploy**. After conversion by the ATC tool, it can be deployed to the AI processor.
- (Optional) \*.external file, including \*deploy.external and \*fakequant.external:

    These files are generated only when the size of the saved accuracy simulation model and deployment model files is \>= 2 GB, and they are generated in the same directory as the compressed \*.onnx model file. They are used to store the data in tensors, with each tensor's data saved in a separate \*.external file whose name is the same as the tensor name, for example, **_conv1.weight_\_deploy.external** and **_conv1.weight_\_fakequant.external**.

    Later, when the compressed \*.onnx deployment model file is loaded through the ATC tool for model conversion, the tensor data in the \*.external file in the same directory is automatically read.

When quantization-aware training is re-executed, the files output by this API will be overwritten.
