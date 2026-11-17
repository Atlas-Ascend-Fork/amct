# save\_model<a name="ZH-CN_TOPIC_0000002548668677"></a>

<!-- md-trans-meta sourceCommit=4a47814eed18048e444f9a2acc42d9864ddc477f translatedAt=2026-10-16T08:38:28.865Z pushedAt=2026-11-05T01:12:03.428Z nl=2 -->

## Supported Products<a name="section185612964420"></a>

<a name="zh-cn_topic_0000002517188794_table38301303189"></a>

| Product | Supported |
| ------------------------------------------- | -------- |
| Ascend 950PR&950DT products | √ |
| Atlas A3 products | √ |
| Atlas A2 products | √ |

## Description<a name="zh-cn_topic_0240188006_section15406195619561"></a>

Post-training quantization API. Based on the quantization factor record file **record\_file** and the modified model, it inserts operators such as AscendQuant and AscendDequant, and then saves a fake\_quant model that can be used for precision simulation in the ONNX Runtime environment, as well as a deployment model that can be used for inference on AI processors.

## Prototype<a name="zh-cn_topic_0240188006_zh-cn_topic_0122830089_section428121323411"></a>

```python
save_model(modfied_onnx_file, record_file, save_path)
```

## Parameters<a name="zh-cn_topic_0240188006_zh-cn_topic_0122830089_section795991810344"></a>

<a name="zh-cn_topic_0240188006_zh-cn_topic_0122830089_table438764393513"></a>
<table><thead align="left"><tr id="zh-cn_topic_0240188006_zh-cn_topic_0122830089_row53871743113510"><th class="cellrowborder" valign="top" width="10.69%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0240188006_zh-cn_topic_0122830089_p1438834363520"><a name="zh-cn_topic_0240188006_zh-cn_topic_0122830089_p1438834363520"></a><a name="zh-cn_topic_0240188006_zh-cn_topic_0122830089_p1438834363520"></a>Parameter</p></th>
<th class="cellrowborder" valign="top" width="7.290000000000001%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0240188006_p1769255516412"><a name="zh-cn_topic_0240188006_p1769255516412"></a><a name="zh-cn_topic_0240188006_p1769255516412"></a>Input/Output</p></th>
<th class="cellrowborder" valign="top" width="82.02000000000001%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0240188006_p15231205416325"><a name="zh-cn_topic_0240188006_p15231205416325"></a><a name="zh-cn_topic_0240188006_p15231205416325"></a>Description</p></th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0240188006_zh-cn_topic_0122830089_row2038874343514"><td class="cellrowborder" valign="top" width="10.69%" headers="mcps1.1.4.1.1 "><p id="p61329565575"><a name="p61329565575"></a><a name="p61329565575"></a>modfied_onnx_file</p></td>
<td class="cellrowborder" valign="top" width="7.290000000000001%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0240188006_p1255438123917"><a name="zh-cn_topic_0240188006_p1255438123917"></a><a name="zh-cn_topic_0240188006_p1255438123917"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="82.02000000000001%" headers="mcps1.1.4.1.3 "><p id="p18172549175815"><a name="p18172549175815"></a><a name="p18172549175815"></a>Meaning: File name, storing the fused model in ONNX format.</p>
<p id="zh-cn_topic_0240188006_p182551638143919"><a name="zh-cn_topic_0240188006_p182551638143919"></a><a name="zh-cn_topic_0240188006_p182551638143919"></a>Data type: string</p></td>
</tr>
<tr id="zh-cn_topic_0240188006_row12264154113552"><td class="cellrowborder" valign="top" width="10.69%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0240188006_p172650416558"><a name="zh-cn_topic_0240188006_p172650416558"></a><a name="zh-cn_topic_0240188006_p172650416558"></a>record_file</p></td>
<td class="cellrowborder" valign="top" width="7.290000000000001%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0240188006_p82657418551"><a name="zh-cn_topic_0240188006_p82657418551"></a><a name="zh-cn_topic_0240188006_p82657418551"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="82.02000000000001%" headers="mcps1.1.4.1.3 "><p id="p1071345865812"><a name="p1071345865812"></a><a name="p1071345865812"></a>Meaning: Path and name of the quantization factor record file.</p>
<p id="zh-cn_topic_0240188006_p1760663182215"><a name="zh-cn_topic_0240188006_p1760663182215"></a><a name="zh-cn_topic_0240188006_p1760663182215"></a>Data type: string</p></td>
</tr>
<tr id="zh-cn_topic_0240188006_row73395458557"><td class="cellrowborder" valign="top" width="10.69%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0240188006_p9489111813337"><a name="zh-cn_topic_0240188006_p9489111813337"></a><a name="zh-cn_topic_0240188006_p9489111813337"></a>save_path</p></td>
<td class="cellrowborder" valign="top" width="7.290000000000001%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0240188006_p934014518550"><a name="zh-cn_topic_0240188006_p934014518550"></a><a name="zh-cn_topic_0240188006_p934014518550"></a>Input</p></td>
<td class="cellrowborder" valign="top" width="82.02000000000001%" headers="mcps1.1.4.1.3 "><p id="p1749919013596"><a name="p1749919013596"></a><a name="p1749919013596"></a>Meaning: Path for storing the model. The path must include the model name prefix, for example, ./quantized_model/*<em id="zh-cn_topic_0240188713_i162671794111"><a name="zh-cn_topic_0240188713_i162671794111"></a><a name="zh-cn_topic_0240188713_i162671794111"></a>model</em>.</p>
<p id="zh-cn_topic_0240188006_p021914293368"><a name="zh-cn_topic_0240188006_p021914293368"></a><a name="zh-cn_topic_0240188006_p021914293368"></a>Data type: string</p></td>
</tr>
</tbody>
</table>

## Return Value<a name="zh-cn_topic_0240188006_zh-cn_topic_0122830089_section293415513458"></a>

None

## Constraints<a name="zh-cn_topic_0240188006_section1443392021419"></a>

- Call this API only after the number of batches in network inference reaches **batch\_num**; otherwise, the quantization factors are incorrect and the quantization result is incorrect.
- This API accepts only ONNX model files generated by the [quantize\_model](./quantize_model.md) API.
- This API requires the quantization factor record file as input. The quantization factor record file is generated in the [quantize\_model](./quantize_model.md) phase and filled with valid values in the model inference phase.

## Example<a name="zh-cn_topic_0240188006_section149503402234"></a>

```python
import amct_pytorch as amct
# Perform network inference and complete quantization during the process.
for i in range(batch_num):
    output = calibration_model(input_batch)

# Insert the API to save the quantized model as an ONNX file.
amct.save_model(modfied_onnx_file="./tmp/modfied_model.onnx",
                record_file="./tmp/scale_offset_record.txt",
                save_path="./results/model")
```

Description of the files written to disk:

- Precision simulation model file: an ONNX-format model file whose name contains **fake\_quant**. It can be used for precision simulation in the ONNX Runtime environment.
- Deployment model file: an ONNX-format model file whose name contains **deploy**. After conversion by the ATC tool, it can be deployed to AI processors.
- (Optional) \*.external files, including \*deploy.external and \*fakequant.external:

    These files are generated only when the sizes of the saved precision simulation model and deployment model files \>= 2 GB. They are generated in the same directory as the compressed \*.onnx model file and are used to store the data in tensors. Each tensor's data is saved as a separate \*.external file, with the file name identical to the tensor name, for example, **_conv1.weight_\_deploy.external** and **_conv1.weight_\_fakequant.external**.

    When the compressed \*.onnx deployment model file is subsequently loaded by the ATC tool for model conversion, the tensor data in the \*.external files in the same directory is automatically read.

When quantization is performed again, the files output by this API will be overwritten.
